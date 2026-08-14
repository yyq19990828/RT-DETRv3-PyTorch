import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import ppdet_pytorch.modeling.teachers.dinov3 as dinov3_module
from ppdet_pytorch.engine.trainer import Trainer
from ppdet_pytorch.modeling.teachers.dinov3 import DINOv3TeacherModel

HUBCONF = """
import torch
from torch import nn
import torch.nn.functional as F


class FakeDINOv3(nn.Module):
    embed_dim = 768
    patch_size = 16

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.last_input = None

    def forward(self, images, is_training=False, masks="unexpected"):
        assert is_training is True
        assert masks is None
        self.last_input = images.detach().clone()
        pooled = F.avg_pool2d(images.mean(dim=1, keepdim=True), 16, 16)
        tokens = pooled.flatten(2).transpose(1, 2).repeat(1, 1, 768)
        return {"x_norm_patchtokens": tokens * self.scale}


def dinov3_vitb16(weights):
    model = FakeDINOv3()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model
"""


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fake_assets(tmp_path, monkeypatch, *, hubconf=HUBCONF, state=None):
    repo = tmp_path / "dinov3"
    repo.mkdir()
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    (repo / "hubconf.py").write_text(hubconf, encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", ".gitignore", "hubconf.py")
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fake dinov3",
        ],
        cwd=repo,
        check=True,
    )
    revision = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(dinov3_module, "PINNED_DINOV3_REVISION", revision)

    weights = tmp_path / "dinov3_vitb16_pretrain_lvd1689m-test.pth"
    torch.save({"scale": torch.tensor(1.0)} if state is None else state, weights)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    return (
        repo,
        weights,
        {
            "dinov3_repo_path": str(repo),
            "dinov3_weights_path": str(weights),
            "weights_filename": weights.name,
            "weights_size_bytes": weights.stat().st_size,
            "weights_sha256": digest,
        },
    )


def test_fake_teacher_normalizes_downsamples_and_freezes(tmp_path, monkeypatch):
    _, _, config = _fake_assets(tmp_path, monkeypatch)
    teacher = DINOv3TeacherModel(**config)
    images = torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(0))

    features = teacher(images)

    expected_input = F.avg_pool2d(
        (images - teacher.mean) / teacher.std, kernel_size=2, stride=2
    )
    assert torch.equal(teacher.model.last_input, expected_input)
    assert features.shape == (2, 768, 2, 2)
    assert not features.requires_grad
    assert not teacher.training
    assert not teacher.model.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    teacher.train()
    assert not teacher.training
    assert not teacher.model.training


def test_pinned_hub_works_on_project_torch_with_strict_state(tmp_path, monkeypatch):
    _, _, config = _fake_assets(tmp_path, monkeypatch)
    teacher = DINOv3TeacherModel(**config)
    assert teacher.weights_sha256 == config["weights_sha256"]

    bad_root = tmp_path / "bad-state"
    bad_root.mkdir()
    _, _, bad_config = _fake_assets(
        bad_root, monkeypatch, state={"unexpected": torch.tensor(1.0)}
    )
    with pytest.raises(RuntimeError, match="state_dict"):
        DINOv3TeacherModel(**bad_config)


def test_rejects_python310_before_hub_load(tmp_path, monkeypatch):
    _, _, config = _fake_assets(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "version_info", (3, 10, 14))
    with pytest.raises(RuntimeError, match="Python 3.11"):
        DINOv3TeacherModel(**config)


def test_rejects_wrong_revision(tmp_path, monkeypatch):
    _, _, config = _fake_assets(tmp_path, monkeypatch)
    monkeypatch.setattr(dinov3_module, "PINNED_DINOV3_REVISION", "0" * 40)
    with pytest.raises(ValueError, match="revision mismatch"):
        DINOv3TeacherModel(**config)


def test_rejects_missing_entry(tmp_path, monkeypatch):
    _, _, config = _fake_assets(
        tmp_path, monkeypatch, hubconf="dependencies = ['torch']\n"
    )
    with pytest.raises(RuntimeError, match="dinov3_vitb16"):
        DINOv3TeacherModel(**config)


def test_rejects_bad_hash(tmp_path, monkeypatch):
    _, _, config = _fake_assets(tmp_path, monkeypatch)
    config["weights_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        DINOv3TeacherModel(**config)


def test_rejects_safetensors(tmp_path, monkeypatch):
    repo, weights, config = _fake_assets(tmp_path, monkeypatch)
    safetensors = weights.with_suffix(".safetensors")
    weights.rename(safetensors)
    config.update(
        {
            "dinov3_repo_path": str(repo),
            "dinov3_weights_path": str(safetensors),
            "weights_filename": safetensors.name,
        }
    )
    with pytest.raises(ValueError, match="safetensors"):
        DINOv3TeacherModel(**config)


def test_rejects_nonsquare_patch_features(tmp_path, monkeypatch):
    _, _, config = _fake_assets(tmp_path, monkeypatch)
    teacher = DINOv3TeacherModel(**config)
    teacher.model.forward = lambda *args, **kwargs: {
        "x_norm_patchtokens": torch.ones(1, 6, 768)
    }
    with pytest.raises(ValueError, match="square feature map"):
        teacher(torch.rand(1, 3, 64, 64))


@pytest.mark.parametrize(
    ("hubconf", "message"),
    [
        (HUBCONF.replace("embed_dim = 768", "embed_dim = 384"), "embed_dim"),
        (HUBCONF.replace("patch_size = 16", "patch_size = (16, 8)"), "16x16"),
    ],
)
def test_rejects_wrong_model_geometry(tmp_path, monkeypatch, hubconf, message):
    _, _, config = _fake_assets(tmp_path, monkeypatch, hubconf=hubconf)
    with pytest.raises(ValueError, match=message):
        DINOv3TeacherModel(**config)


def _patch_minimal_trainer(monkeypatch):
    monkeypatch.setattr(
        Trainer, "_build_data", lambda self, cfg: setattr(self, "loader", [])
    )
    monkeypatch.setattr(
        Trainer,
        "_build_model",
        lambda self, cfg: setattr(self, "model", nn.Linear(1, 1)),
    )
    monkeypatch.setattr(Trainer, "_init_callbacks", lambda self: None)
    monkeypatch.setattr(Trainer, "_init_metrics", lambda self, validate=False: None)
    monkeypatch.setattr(Trainer, "_reset_metrics", lambda self: None)


def test_student_without_teacher_assets_builds_in_eval(tmp_path, monkeypatch):
    _patch_minimal_trainer(monkeypatch)
    cfg = {
        "architecture": "RTDETRV4",
        "save_dir": str(tmp_path / "eval"),
        "teacher_model": {
            "type": "DINOv3TeacherModel",
            "dinov3_repo_path": str(tmp_path / "deleted-repo"),
            "dinov3_weights_path": str(tmp_path / "deleted.pth"),
        },
    }

    trainer = Trainer(cfg, mode="eval")

    assert trainer.teacher_model is None
    assert trainer.optimizer is None


def test_teacher_preflight_fails_before_optimizer(tmp_path, monkeypatch):
    _patch_minimal_trainer(monkeypatch)
    optimizer_built = []
    monkeypatch.setattr(
        Trainer, "_build_optimizer", lambda self, cfg: optimizer_built.append(True)
    )
    cfg = {
        "architecture": "RTDETRV4",
        "save_dir": str(tmp_path / "train"),
        "teacher_model": {
            "type": "DINOv3TeacherModel",
            "dinov3_repo_path": str(tmp_path / "missing-repo"),
            "dinov3_weights_path": str(tmp_path / "missing.pth"),
            "weights_filename": "missing.pth",
            "weights_size_bytes": 1,
            "weights_sha256": "0" * 64,
        },
    }

    with pytest.raises(FileNotFoundError, match="repository checkout"):
        Trainer(cfg, mode="train")
    assert not optimizer_built


def test_trainer_attaches_detached_teacher_features():
    class Teacher(nn.Module):
        def forward(self, images):
            return images.mean(dim=1, keepdim=True).detach()

    trainer = Trainer.__new__(Trainer)
    trainer.teacher_model = Teacher()
    batch = {"image": torch.rand(2, 3, 4, 4, requires_grad=True)}

    result = trainer._attach_teacher_features(batch)

    assert "teacher_encoder_output" not in batch
    assert result["teacher_encoder_output"].shape == (2, 1, 4, 4)
    assert not result["teacher_encoder_output"].requires_grad
