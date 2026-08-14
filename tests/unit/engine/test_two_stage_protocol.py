from copy import deepcopy

import pytest

from ppdet_pytorch.engine.trainer import Trainer
from ppdet_pytorch.engine.training_protocol import TwoStageDetectionProtocol


@pytest.mark.parametrize("family", ["dfine", "deim", "rtdetrv4"])
def test_two_stage_metric_transition_and_restart_trace(family):
    protocol = TwoStageDetectionProtocol(
        family=family, stop_epoch=3, ema_restart_decay=0.9998
    )

    protocol.after_validation({"bbox": 0.4})
    save = protocol.pop_actions()
    assert save == [
        {
            "metric": 0.4,
            "name": "save_best",
            "path": "best_stg1.pth",
            "stage": 1,
        }
    ]
    protocol.complete_action(save[0], basename="best_stg1.pth", sha256="1" * 64)
    protocol.after_validation({"bbox": 0.3})
    assert protocol.pop_actions() == []

    protocol.before_epoch(3, {})
    transition = protocol.pop_actions()[0]
    assert transition["name"] == "transition"
    protocol.complete_action(transition)
    assert protocol.stage == 2
    assert protocol.top_metric == 0.4

    protocol.after_validation({"bbox": 0.35})
    assert protocol.pop_actions() == []
    protocol.after_validation({"bbox": 0.34})
    restart = protocol.pop_actions()[0]
    assert restart["name"] == "restart"
    assert restart["decay"] == pytest.approx(0.9997)
    assert protocol.current_decay == pytest.approx(0.9998)
    protocol.complete_action(restart)
    assert protocol.restart_count == 1

    protocol.after_validation({"bbox": 0.41})
    stage2_save = protocol.pop_actions()[0]
    assert stage2_save["path"] == "best_stg2.pth"


def test_two_stage_state_round_trip_preserves_all_decisions():
    protocol = TwoStageDetectionProtocol("rtdetrv4", 2, 0.9998, current_gam_weight=1.5)
    protocol.after_validation({"bbox": 0.4})
    action = protocol.pop_actions()[0]
    protocol.complete_action(action, basename="best_stg1.pth", sha256="a" * 64)
    protocol.before_epoch(2, {})
    protocol.complete_action(protocol.pop_actions()[0])
    protocol.after_validation({"bbox": 0.3})

    restored = TwoStageDetectionProtocol("rtdetrv4", 2, 0.9998)
    restored.load_state_dict(deepcopy(protocol.state_dict()))

    assert restored.state_dict() == protocol.state_dict()
    assert restored.current_gam_weight == 1.5


def test_rejects_missing_bbox_metric_without_state_change():
    protocol = TwoStageDetectionProtocol("dfine", 2, 0.9999)
    before = protocol.state_dict()

    with pytest.raises(ValueError, match="missing required bbox"):
        protocol.after_validation({"loss": 1.0})

    assert protocol.state_dict() == before


def test_rejects_missing_companion_at_transition():
    protocol = TwoStageDetectionProtocol("deim", 2, 0.9999)

    with pytest.raises(FileNotFoundError, match="companion is missing"):
        protocol.before_epoch(2, {})


def test_rejects_family_mismatch_and_rejects_stage_mismatch():
    protocol = TwoStageDetectionProtocol("dfine", 2, 0.9999)
    state = protocol.state_dict()
    wrong_family = dict(state, family="deim")
    wrong_stage = dict(state, stage=2)

    with pytest.raises(ValueError, match="family mismatch"):
        protocol.load_state_dict(wrong_family)
    with pytest.raises(ValueError, match="stage 2 requires"):
        protocol.load_state_dict(wrong_stage)


def test_trainer_rejects_two_stage_protocol_without_validation(tmp_path):
    config = {
        "TrainingProtocol": {
            "name": "TwoStageDetectionProtocol",
            "family": "dfine",
            "stop_epoch": 2,
            "ema_restart_decay": 0.9999,
        },
        "log_ranks": "0",
        "save_dir": str(tmp_path),
    }

    with pytest.raises(ValueError, match="requires validation"):
        Trainer(config, mode="train")
