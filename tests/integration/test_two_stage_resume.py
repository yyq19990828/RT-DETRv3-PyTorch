from copy import deepcopy

import pytest
import torch
from torch import nn

from ppdet_pytorch.engine.trainer import Trainer
from ppdet_pytorch.engine.training_protocol import TwoStageDetectionProtocol
from ppdet_pytorch.optimizer.ema import ModelEMA


def _trainer(tmp_path, family="dfine"):
    trainer = Trainer.__new__(Trainer)
    trainer.model = nn.Linear(1, 1)
    trainer.optimizer = torch.optim.SGD(
        trainer.model.parameters(), lr=0.1, momentum=0.9
    )
    trainer.lr = torch.optim.lr_scheduler.StepLR(trainer.optimizer, 1, gamma=0.8)
    trainer.scaler = None
    trainer.use_ema = True
    trainer.ema = ModelEMA(
        trainer.model,
        decay=0.9999,
        ema_decay_type="exponential",
        warmups=10,
        device="cpu",
    )
    trainer.training_protocol = TwoStageDetectionProtocol(family, 2, 0.9998)
    trainer.cfg = {"architecture": "Tiny"}
    trainer.save_dir = str(tmp_path)
    trainer.status = {"epoch_id": 0, "global_step": 1}
    trainer.global_step = 1
    return trainer


def _fingerprint(trainer):
    return {
        "model": deepcopy(trainer.model.state_dict()),
        "optimizer": deepcopy(trainer.optimizer.state_dict()),
        "scheduler": deepcopy(trainer.lr.state_dict()),
        "ema": deepcopy(trainer.ema.state_dict_for_save()),
        "protocol": deepcopy(trainer.training_protocol.state_dict()),
    }


def _assert_fingerprint(actual, expected):
    assert actual["optimizer"] == expected["optimizer"]
    assert actual["scheduler"] == expected["scheduler"]
    assert actual["protocol"] == expected["protocol"]
    for section in ("model",):
        for key in actual[section]:
            assert torch.equal(actual[section][key], expected[section][key])
    for key in actual["ema"]["ema_state_dict"]:
        assert torch.equal(
            actual["ema"]["ema_state_dict"][key],
            expected["ema"]["ema_state_dict"][key],
        )


@pytest.mark.parametrize("family", ["dfine", "deim", "rtdetrv4"])
def test_transition_restores_stage1_snapshot_and_updates_real_ema_decay(
    tmp_path, family
):
    trainer = _trainer(tmp_path, family)
    trainer.notify_validation({"bbox": 0.4})
    stage1 = _fingerprint(trainer)

    with torch.no_grad():
        trainer.model.weight.add_(10)
    trainer.optimizer.param_groups[0]["lr"] = 0.01
    trainer.lr.last_epoch = 9
    trainer.global_step = 8
    trainer.status.update(epoch_id=2, global_step=8)
    trainer.training_protocol.before_epoch(2, {})
    trainer._execute_protocol_actions()

    assert trainer.training_protocol.stage == 2
    assert trainer.ema.decay == pytest.approx(0.9998)
    assert trainer.status["epoch_id"] == 2
    assert trainer.global_step == 1
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, stage1["model"][key])
    assert trainer.optimizer.state_dict() == stage1["optimizer"]
    assert trainer.lr.state_dict() == stage1["scheduler"]


def test_rejects_tampered_companion_before_live_state_mutation(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.notify_validation({"bbox": 0.4})
    companion = tmp_path / "best_stg1.pth"
    companion.write_bytes(companion.read_bytes() + b"tampered")
    trainer.status["epoch_id"] = 2
    trainer.training_protocol.before_epoch(2, {})
    before = _fingerprint(trainer)

    with pytest.raises(ValueError, match="companion SHA-256 mismatch"):
        trainer._execute_protocol_actions()

    _assert_fingerprint(_fingerprint(trainer), before)
    assert trainer.training_protocol.pop_actions()[0]["name"] == "transition"


def test_stage2_best_and_no_improvement_restart_are_resumable(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.notify_validation({"bbox": 0.4})
    trainer.status["epoch_id"] = 2
    trainer.training_protocol.before_epoch(2, {})
    trainer._execute_protocol_actions()
    trainer.notify_validation({"bbox": 0.3})
    trainer.notify_validation({"bbox": 0.41})

    assert (tmp_path / "best_stg2.pth").is_file()
    assert trainer.training_protocol.top_metric == 0.41

    trainer.notify_validation({"bbox": 0.4})
    trainer.notify_validation({"bbox": 0.39})
    assert trainer.training_protocol.restart_count == 1
    assert trainer.training_protocol.top_metric == 0.41
    assert trainer.ema.decay == pytest.approx(0.9997)


def test_ema_evaluation_state_does_not_advance_lifecycle(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.ema.update(trainer.model)
    before = deepcopy(trainer.ema.state_dict_for_save())

    observed = trainer.ema.evaluation_state_dict()

    assert trainer.ema.state_dict_for_save()["epoch"] == before["epoch"]
    assert trainer.ema.state_dict_for_save()["step"] == before["step"]
    for key, value in observed.items():
        assert torch.equal(value, before["ema_state_dict"][key])


def test_epoch_boundary_resume_matches_uninterrupted_protocol_trace(tmp_path):
    uninterrupted = _trainer(tmp_path / "uninterrupted")
    uninterrupted.notify_validation({"bbox": 0.4})
    uninterrupted.status["epoch_id"] = 2
    uninterrupted.training_protocol.before_epoch(2, {})
    uninterrupted._execute_protocol_actions()
    uninterrupted.notify_validation({"bbox": 0.3})
    resume_path = tmp_path / "uninterrupted" / "epoch_3.pth"
    uninterrupted._save_protocol_checkpoint(str(resume_path))

    resumed = _trainer(tmp_path / "uninterrupted")
    resumed.resume_weights(str(resume_path))
    for metric in (0.35, 0.34, 0.41):
        uninterrupted.notify_validation({"bbox": metric})
        resumed.notify_validation({"bbox": metric})

    _assert_fingerprint(_fingerprint(resumed), _fingerprint(uninterrupted))
    assert resumed.global_step == uninterrupted.global_step
    assert resumed.training_protocol.state_dict() == (
        uninterrupted.training_protocol.state_dict()
    )
