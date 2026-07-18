"""Opt-in numerical alignment for official RT-DETRv3 checkpoints."""

from copy import deepcopy
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from ppdet_pytorch import modeling as _torch_modeling  # noqa: F401
from ppdet_pytorch.conversion.converter import WeightConverter
from ppdet_pytorch.conversion.models import ConversionConfig
from ppdet_pytorch.core.workspace import create as torch_create
from ppdet_pytorch.core.workspace import load_config as torch_load_config


pytestmark = [pytest.mark.numerical, pytest.mark.paddle, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "configs/checkpoints/rtdetrv3_coco.yml"
EXPECTED_MISSING_KEYS = {
    "aux_o2m_head.anchor_points",
    "aux_o2m_head.stride_tensor",
}
CHECKPOINT_CASES = [
    pytest.param(
        "rtdetrv3_r18vd_6x_coco",
        "RTDETRV3_R18_PADDLE_CHECKPOINT",
        571,
        75,
        384,
        1e-4,
        0,
        id="r18",
    ),
    pytest.param(
        "rtdetrv3_r34vd_6x_coco",
        "RTDETRV3_R34_PADDLE_CHECKPOINT",
        681,
        91,
        462,
        1e-4,
        0,
        id="r34",
    ),
    pytest.param(
        "rtdetrv3_r50vd_6x_coco",
        "RTDETRV3_R50_PADDLE_CHECKPOINT",
        789,
        103,
        445,
        3e-4,
        2,
        id="r50",
    ),
]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value.numpy()


def _clone_workspace_entry(value):
    if not isinstance(value, dict) or not hasattr(value, "schema"):
        return deepcopy(value)
    cloned = value.copy()
    cloned.schema = value.schema.copy()
    for key, item in value.items():
        cloned[key] = deepcopy(item)
    return cloned


def _assert_allclose(paddle_value, torch_value, rtol=1e-4, err_msg=""):
    np.testing.assert_allclose(
        _as_numpy(paddle_value),
        _as_numpy(torch_value),
        rtol=rtol,
        atol=1e-5,
        err_msg=err_msg,
    )


def _build_torch_model(config_path):
    cfg = torch_load_config(str(config_path))
    model = torch_create(cfg.architecture)
    transpose_target_keys = {
        f"{name}.weight" if name else "weight"
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    return model, transpose_target_keys


def _gradient_alignment_metrics(paddle_model, torch_model, transpose_target_keys):
    paddle_parameters = dict(paddle_model.named_parameters())
    torch_parameters = dict(torch_model.named_parameters())
    difference_sq = 0.0
    paddle_sq = 0.0
    torch_sq = 0.0
    dot_product = 0.0
    sign_mismatches = 0
    sign_elements = 0
    compared_count = 0

    for name in sorted(set(paddle_parameters) & set(torch_parameters)):
        paddle_gradient = paddle_parameters[name].grad
        torch_gradient = torch_parameters[name].grad
        if paddle_gradient is None or torch_gradient is None:
            continue
        paddle_array = _as_numpy(paddle_gradient)
        torch_array = _as_numpy(torch_gradient)
        if name in transpose_target_keys:
            paddle_array = paddle_array.T
        paddle_flat = paddle_array.astype(np.float64, copy=False).reshape(-1)
        torch_flat = torch_array.astype(np.float64, copy=False).reshape(-1)
        difference = paddle_flat - torch_flat
        difference_sq += float(np.dot(difference, difference))
        paddle_sq += float(np.dot(paddle_flat, paddle_flat))
        torch_sq += float(np.dot(torch_flat, torch_flat))
        dot_product += float(np.dot(paddle_flat, torch_flat))
        active = (np.abs(paddle_flat) > 1e-7) | (np.abs(torch_flat) > 1e-7)
        sign_mismatches += int(
            np.count_nonzero(
                np.signbit(paddle_flat[active]) != np.signbit(torch_flat[active])
            )
        )
        sign_elements += int(np.count_nonzero(active))
        compared_count += 1

    return {
        "compared_count": compared_count,
        "relative_l2": (difference_sq / paddle_sq) ** 0.5,
        "cosine": dot_product / (paddle_sq * torch_sq) ** 0.5,
        "sign_mismatch_fraction": sign_mismatches / sign_elements,
    }


@pytest.mark.parametrize(
    (
        "model_name",
        "checkpoint_env",
        "expected_converted_count",
        "expected_bn_counters",
        "expected_gradient_count",
        "alignment_rtol",
        "max_label_mismatches",
    ),
    CHECKPOINT_CASES,
)
def test_official_checkpoint_alignment(
    tmp_path,
    monkeypatch,
    isolated_workspace,
    request,
    model_name,
    checkpoint_env,
    expected_converted_count,
    expected_bn_counters,
    expected_gradient_count,
    alignment_rtol,
    max_label_mismatches,
):
    checkpoint_value = os.environ.get(checkpoint_env)
    if not checkpoint_value:
        pytest.skip(f"set {checkpoint_env} to the official .pdparams file")

    paddle = pytest.importorskip(
        "paddle", reason="requires the PaddlePaddle development extra"
    )
    torch_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    request.addfinalizer(lambda: torch.set_num_threads(torch_threads))
    checkpoint_path = Path(checkpoint_value).expanduser().resolve()
    if not checkpoint_path.is_file():
        pytest.skip(f"official checkpoint does not exist: {checkpoint_path}")

    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    model_manifest = manifest["models"][model_name]
    assert checkpoint_path.stat().st_size == model_manifest["source_size_bytes"]
    assert _sha256(checkpoint_path) == model_manifest["source_sha256"]

    torch_config_path = ROOT / model_manifest["config"]
    torch_model, transpose_target_keys = _build_torch_model(torch_config_path)
    converted_path = tmp_path / f"{model_name}.pth"
    request.addfinalizer(lambda: converted_path.unlink(missing_ok=True))
    converter = WeightConverter(ConversionConfig(strict_mode=True))
    session = converter.convert(
        input_path=str(checkpoint_path),
        output_path=str(converted_path),
        target_model_state_dict=torch_model.state_dict(),
        transpose_target_keys=transpose_target_keys,
    )
    assert session.statistics.converted_count == expected_converted_count
    assert session.statistics.skipped_count == 0
    assert not session.statistics.unmapped_source_keys
    unmapped_target_keys = set(session.statistics.unmapped_target_keys)
    assert (
        sum(key.endswith("num_batches_tracked") for key in unmapped_target_keys)
        == expected_bn_counters
    )
    assert {
        key for key in unmapped_target_keys if not key.endswith("num_batches_tracked")
    } == EXPECTED_MISSING_KEYS

    converted = torch.load(converted_path, map_location="cpu", weights_only=False)
    incompatible = torch_model.load_state_dict(converted["model"], strict=False)
    assert set(incompatible.missing_keys) == EXPECTED_MISSING_KEYS
    assert not incompatible.unexpected_keys
    torch_model.eval()

    paddle.set_device("cpu")
    paddle_source = ROOT / "third-party/RT-DETRv3-paddle"
    monkeypatch.syspath_prepend(str(paddle_source))
    from ppdet import modeling as _paddle_modeling  # noqa: F401
    from ppdet.core.workspace import create as paddle_create
    from ppdet.core.workspace import global_config as paddle_global_config
    from ppdet.core.workspace import load_config as paddle_load_config

    paddle_workspace_snapshot = {
        key: _clone_workspace_entry(value)
        for key, value in paddle_global_config.items()
    }

    def restore_paddle_workspace():
        paddle_global_config.clear()
        paddle_global_config.update(paddle_workspace_snapshot)

    request.addfinalizer(restore_paddle_workspace)

    paddle_config_path = paddle_source / model_manifest["config"]
    paddle_cfg = paddle_load_config(str(paddle_config_path))
    paddle_model = paddle_create(paddle_cfg.architecture)
    paddle_state_dict = paddle.load(str(checkpoint_path))
    mappings = converter.name_mapper.apply_naming_rules(
        list(paddle_state_dict), set(converted["model"])
    )
    for mapping in mappings:
        source_value = paddle_state_dict[mapping.source_name].numpy()
        if mapping.target_name in transpose_target_keys:
            source_value = source_value.T
        np.testing.assert_array_equal(
            source_value,
            converted["model"][mapping.target_name].numpy(),
        )

    paddle_model.set_state_dict(paddle_state_dict)
    paddle_model.eval()

    rng = np.random.default_rng(2026)
    image = rng.standard_normal((1, 3, 640, 640), dtype=np.float32)
    im_shape = np.array([[640.0, 640.0]], dtype=np.float32)
    scale_factor = np.array([[1.0, 1.0]], dtype=np.float32)
    paddle_inputs = {
        "image": paddle.to_tensor(image.copy()),
        "im_shape": paddle.to_tensor(im_shape.copy()),
        "scale_factor": paddle.to_tensor(scale_factor.copy()),
    }
    torch_inputs = {
        "image": torch.from_numpy(image.copy()),
        "im_shape": torch.from_numpy(im_shape.copy()),
        "scale_factor": torch.from_numpy(scale_factor.copy()),
    }

    with paddle.no_grad(), torch.no_grad():
        paddle_backbone = paddle_model.backbone(paddle_inputs)
        torch_backbone = torch_model.backbone(torch_inputs)
        for paddle_value, torch_value in zip(paddle_backbone, torch_backbone):
            _assert_allclose(paddle_value, torch_value, rtol=alignment_rtol)

        paddle_neck = paddle_model.neck(paddle_backbone)
        torch_neck = torch_model.neck(torch_backbone)
        for paddle_value, torch_value in zip(paddle_neck, torch_neck):
            _assert_allclose(paddle_value, torch_value, rtol=alignment_rtol)

        paddle_transformer = paddle_model.transformer(paddle_neck, None, paddle_inputs)
        torch_transformer = torch_model.transformer(torch_neck, None, torch_inputs)
        for paddle_value, torch_value in zip(
            paddle_transformer[:4], torch_transformer[:4]
        ):
            _assert_allclose(paddle_value, torch_value, rtol=alignment_rtol)

        paddle_head = paddle_model.detr_head(paddle_transformer, paddle_neck)
        torch_head = torch_model.detr_head(torch_transformer, torch_neck)
        for paddle_value, torch_value in zip(paddle_head[:2], torch_head[:2]):
            _assert_allclose(paddle_value, torch_value, rtol=alignment_rtol)

        paddle_bbox, paddle_bbox_num, _ = paddle_model.post_process(
            paddle_head,
            paddle_inputs["im_shape"],
            paddle_inputs["scale_factor"],
            paddle_inputs["image"][2:].shape,
        )
        torch_bbox, torch_bbox_num, _ = torch_model.post_process(
            torch_head,
            torch_inputs["im_shape"],
            torch_inputs["scale_factor"],
            torch_inputs["image"][2:].shape,
        )

    paddle_bbox = _as_numpy(paddle_bbox)
    torch_bbox = _as_numpy(torch_bbox)
    stable_candidates = paddle_bbox[:, 0] == torch_bbox[:, 0]
    label_mismatches = np.count_nonzero(~stable_candidates)
    assert label_mismatches <= max_label_mismatches
    np.testing.assert_allclose(
        paddle_bbox[:, 1], torch_bbox[:, 1], rtol=alignment_rtol, atol=1e-5
    )
    np.testing.assert_allclose(
        paddle_bbox[stable_candidates, 2:] / 640.0,
        torch_bbox[stable_candidates, 2:] / 640.0,
        rtol=alignment_rtol,
        atol=1e-5,
    )
    np.testing.assert_array_equal(_as_numpy(paddle_bbox_num), _as_numpy(torch_bbox_num))

    gt_bbox = np.array(
        [[0.30, 0.40, 0.20, 0.15], [0.72, 0.65, 0.12, 0.25]],
        dtype=np.float32,
    )
    gt_class = np.array([[2], [17]], dtype=np.int64)
    paddle_loss_inputs = {
        "gt_bbox": [paddle.to_tensor(gt_bbox.copy())],
        "gt_class": [paddle.to_tensor(gt_class.copy())],
    }
    torch_loss_inputs = {
        "gt_bbox": [torch.from_numpy(gt_bbox.copy())],
        "gt_class": [torch.from_numpy(gt_class.copy())],
    }

    paddle_loss_outputs = []
    for value in paddle_transformer[:4]:
        leaf = value.detach().clone()
        leaf.stop_gradient = False
        paddle_loss_outputs.append(leaf)
    torch_loss_outputs = [
        value.detach().clone().requires_grad_(True) for value in torch_transformer[:4]
    ]

    # Isolate head/loss semantics from framework-specific denoising RNG.
    paddle_model.detr_head.train()
    torch_model.detr_head.train()
    paddle_losses = paddle_model.detr_head(
        tuple(paddle_loss_outputs) + (None,),
        paddle_neck,
        paddle_loss_inputs,
    )
    torch_losses = torch_model.detr_head(
        tuple(torch_loss_outputs) + (None,),
        torch_neck,
        torch_loss_inputs,
    )

    assert set(paddle_losses) == set(torch_losses)
    for name in paddle_losses:
        _assert_allclose(paddle_losses[name], torch_losses[name], rtol=alignment_rtol)

    paddle_total = paddle.add_n(list(paddle_losses.values()))
    torch_total = sum(torch_losses.values())
    _assert_allclose(paddle_total, torch_total, rtol=alignment_rtol)
    paddle_total.backward()
    torch_total.backward()
    for paddle_value, torch_value in zip(paddle_loss_outputs, torch_loss_outputs):
        _assert_allclose(paddle_value.grad, torch_value.grad, rtol=alignment_rtol)

    for transformer in (paddle_model.transformer, torch_model.transformer):
        transformer.num_noises = 0
        transformer.num_groups = 2
        transformer.num_queries = [20, 20]
        transformer.num_denoising = 4
        transformer.num_noise_denoising = 4
        transformer.label_noise_ratio = 0.0
        transformer.box_noise_scale = 0.0
        transformer.num_queries_o2m = 20
    paddle_model.detr_head.num_queries_o2m = 20
    torch_model.detr_head.num_queries_o2m = 20
    paddle_model.clear_gradients()
    torch_model.zero_grad(set_to_none=True)
    paddle_model.train()
    torch_model.train()

    train_size = 96
    train_image = np.random.default_rng(2026).standard_normal(
        (1, 3, train_size, train_size), dtype=np.float32
    )
    origin_gt_bbox = np.empty_like(gt_bbox)
    origin_gt_bbox[:, :2] = (gt_bbox[:, :2] - gt_bbox[:, 2:] / 2) * train_size
    origin_gt_bbox[:, 2:] = (gt_bbox[:, :2] + gt_bbox[:, 2:] / 2) * train_size
    paddle_train_inputs = {
        "image": paddle.to_tensor(train_image.copy()),
        "im_shape": paddle.to_tensor([[train_size, train_size]], dtype="float32"),
        "scale_factor": paddle.to_tensor([[1.0, 1.0]], dtype="float32"),
        "gt_bbox": [paddle.to_tensor(gt_bbox.copy())],
        "gt_class": [paddle.to_tensor(gt_class.copy())],
        "origin_gt_bbox": paddle.to_tensor(origin_gt_bbox[None].copy()),
        "origin_gt_class": paddle.to_tensor(gt_class[None].copy()),
        "pad_origin_gt_mask": paddle.ones([1, 2, 1]),
        "epoch_id": 0,
    }
    torch_train_inputs = {
        "image": torch.from_numpy(train_image.copy()),
        "im_shape": torch.tensor([[train_size, train_size]], dtype=torch.float32),
        "scale_factor": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        "gt_bbox": [torch.from_numpy(gt_bbox.copy())],
        "gt_class": [torch.from_numpy(gt_class.copy())],
        "origin_gt_bbox": torch.from_numpy(origin_gt_bbox[None].copy()),
        "origin_gt_class": torch.from_numpy(gt_class[None].copy()),
        "pad_origin_gt_mask": torch.ones(1, 2, 1),
        "epoch_id": 0,
    }

    # Reduced queries and zero noise isolate deterministic training semantics.
    paddle_train_losses = paddle_model(paddle_train_inputs)
    torch_train_losses = torch_model(torch_train_inputs)
    assert set(paddle_train_losses) == set(torch_train_losses)
    for name in paddle_train_losses:
        _assert_allclose(
            paddle_train_losses[name],
            torch_train_losses[name],
            rtol=alignment_rtol,
            err_msg=name,
        )

    paddle_train_total = paddle.add_n(list(paddle_train_losses.values()))
    torch_train_total = sum(torch_train_losses.values())
    _assert_allclose(paddle_train_total, torch_train_total, rtol=alignment_rtol)
    paddle_train_total.backward()
    torch_train_total.backward()
    gradient_metrics = _gradient_alignment_metrics(
        paddle_model, torch_model, transpose_target_keys
    )
    assert gradient_metrics["compared_count"] == expected_gradient_count
    assert gradient_metrics["relative_l2"] < 0.01
    assert gradient_metrics["cosine"] > 0.9999
    assert gradient_metrics["sign_mismatch_fraction"] < 0.005
