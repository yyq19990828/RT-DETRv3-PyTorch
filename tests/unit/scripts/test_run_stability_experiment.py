import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "scripts/run_stability_experiment.py"


def load_script():
    spec = importlib.util.spec_from_file_location(
        "run_stability_experiment", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_fake_inputs(tmp_path):
    coco_root = tmp_path / "coco"
    (coco_root / "train2017").mkdir(parents=True)
    (coco_root / "val2017").mkdir()
    (coco_root / "annotations").mkdir()
    (coco_root / "annotations/instances_train2017.json").write_text("{}")
    (coco_root / "annotations/instances_val2017.json").write_text("{}")
    pretrain = tmp_path / "pretrain.pth"
    pretrain.write_bytes(b"checkpoint")
    return coco_root, pretrain


def test_dry_run_builds_frozen_protocol_without_outputs(tmp_path, capsys):
    script = load_script()
    coco_root, pretrain = make_fake_inputs(tmp_path)
    output_root = tmp_path / "outputs"

    exit_code = script.main(
        [
            "--model",
            "r18",
            "--seed",
            "1",
            "--coco-root",
            str(coco_root),
            "--pretrain",
            str(pretrain),
            "--output-root",
            str(output_root),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "--nproc_per_node=2" in output
    assert "--seed 1" in output
    assert "TrainReader.batch_size=8" in output
    assert "epoch=72" in output
    assert "use_ema=true" in output
    assert "--use-ema" in output
    assert not output_root.exists()


def test_resume_command_does_not_reload_pretraining(tmp_path):
    script = load_script()
    coco_root, _ = make_fake_inputs(tmp_path)
    resume = tmp_path / "epoch_3.pth"
    resume.write_bytes(b"checkpoint")
    args = script.parse_args(
        [
            "--model",
            "r18",
            "--seed",
            "0",
            "--coco-root",
            str(coco_root),
            "--resume",
            str(resume),
        ]
    )
    args.coco_root = coco_root.resolve()
    args.resume = resume.resolve()

    command = script.build_train_command(
        args,
        ROOT / script.MODEL_SPECS["r18"]["config"],
        None,
        tmp_path / "run",
    )

    assert "--resume" in command
    assert str(resume.resolve()) in command
    assert not any("pretrain_weights=" in item for item in command)


@pytest.mark.parametrize("gpu_ids", ["", "0,00", "0,cuda:1"])
def test_invalid_gpu_list_is_rejected(gpu_ids):
    script = load_script()
    with pytest.raises(ValueError):
        script.parse_gpu_ids(gpu_ids)


def test_eval_metric_parser_reads_summary_only(tmp_path):
    script = load_script()
    log_path = tmp_path / "eval.log"
    log_path.write_text(
        "Average Precision (AP) @[ IoU=0.50:0.95 ] = 0.481\n"
        "rtdetrv3.eval INFO:   AP   : 0.480477\n"
        "rtdetrv3.eval INFO:   AP50 : 0.662000\n"
        "rtdetrv3.eval INFO:   AP75 : 0.514000\n"
    )

    assert script.parse_eval_metrics(log_path) == {
        "AP": 0.480477,
        "AP50": 0.662,
        "AP75": 0.514,
    }


def test_checkpoint_tensor_hash_ignores_metadata(tmp_path):
    import torch

    script = load_script()
    first = tmp_path / "first.pth"
    second = tmp_path / "second.pth"
    state_dict = {"backbone.weight": torch.arange(6).reshape(2, 3)}
    torch.save({"model": state_dict, "metadata": {"session": "first"}}, first)
    torch.save({"model": state_dict, "metadata": {"session": "second"}}, second)

    assert script.sha256(first) != script.sha256(second)
    assert script.checkpoint_tensor_sha256(first) == script.checkpoint_tensor_sha256(
        second
    )


def test_resume_preserves_initial_run_evidence(tmp_path):
    script = load_script()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    original = {
        "created_at": "2026-07-19T00:00:00+00:00",
        "git": {"commit": "abc"},
        "environment": {"torch": "2.5.1"},
        "inputs": {"pretrain_tensor_sha256": "tensor-hash"},
        "commands": {"train": ["torchrun"]},
    }
    (run_dir / "metadata.json").write_text(json.dumps(original))

    assert script.load_initial_run(run_dir) == original
