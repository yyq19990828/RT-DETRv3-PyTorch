import math

import pytest
import torch

from detrs.core.workspace import create
from detrs.optimizer.optimizer import (
    FlatCosineLRScheduler,
    LearningRate,
)
from detrs.utils.checkpoint import load_checkpoint, save_checkpoint


def _make_scheduler(**overrides):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD(
        [
            {"params": [parameter], "lr": 0.2},
            {"params": [torch.nn.Parameter(torch.tensor(2.0))], "lr": 0.02},
        ]
    )
    config = {
        "total_epochs": 10,
        "warmup_iter": 2,
        "flat_epochs": 4,
        "no_aug_epochs": 2,
        "lr_gamma": 0.1,
    }
    config.update(overrides)
    scheduler = LearningRate(
        base_lr=0.2,
        schedulers=[FlatCosineLRScheduler(**config)],
    )(step_per_epoch=2, optimizer=optimizer)
    return optimizer, scheduler


def _upstream_lr(current_iter, base_lr=0.2):
    total_iter = 20
    warmup_iter = 2
    flat_iter = 8
    no_aug_iter = 4
    min_lr = base_lr * 0.1
    if current_iter <= warmup_iter:
        return base_lr * (current_iter / warmup_iter) ** 2
    if current_iter <= flat_iter:
        return base_lr
    if current_iter >= total_iter - no_aug_iter:
        return min_lr
    cosine = 0.5 * (
        1
        + math.cos(
            math.pi
            * (current_iter - flat_iter)
            / (total_iter - flat_iter - no_aug_iter)
        )
    )
    return min_lr + (base_lr - min_lr) * cosine


def test_flat_cosine_trace_matches_upstream_at_every_phase_boundary():
    optimizer, scheduler = _make_scheduler()

    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [0.2, 0.02]
    )
    trace = []
    for current_iter in range(21):
        optimizer.step()
        scheduler.step()
        trace.append(optimizer.param_groups[0]["lr"])
        assert scheduler.current_iter == current_iter
        assert optimizer.param_groups[1]["lr"] == pytest.approx(
            optimizer.param_groups[0]["lr"] * 0.1
        )

    assert trace == pytest.approx([_upstream_lr(i) for i in range(21)])
    for boundary in (0, 2, 3, 8, 9, 15, 16, 20):
        assert trace[boundary] == pytest.approx(_upstream_lr(boundary))


def test_flat_cosine_resume_matches_uninterrupted_trace():
    optimizer, scheduler = _make_scheduler()
    for _ in range(11):
        optimizer.step()
        scheduler.step()

    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    expected = []
    for _ in range(7):
        optimizer.step()
        scheduler.step()
        expected.append([group["lr"] for group in optimizer.param_groups])

    resumed_optimizer, resumed_scheduler = _make_scheduler()
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)
    actual = []
    for _ in range(7):
        resumed_optimizer.step()
        resumed_scheduler.step()
        actual.append([group["lr"] for group in resumed_optimizer.param_groups])

    assert torch.tensor(actual) == pytest.approx(torch.tensor(expected))
    assert resumed_scheduler.state_dict() == scheduler.state_dict()


def test_flat_cosine_checkpoint_resume_after_cosine_start(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9)
    scheduler = LearningRate(
        base_lr=0.2,
        schedulers=[
            FlatCosineLRScheduler(
                total_epochs=10,
                warmup_iter=2,
                flat_epochs=4,
                no_aug_epochs=2,
                lr_gamma=0.1,
            )
        ],
    )(step_per_epoch=2, optimizer=optimizer)
    for _ in range(11):
        optimizer.zero_grad()
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        scheduler.step()
    path = tmp_path / "cosine.pth"
    save_checkpoint(model, optimizer, 1, 11, str(path), scheduler=scheduler)

    resumed_model = torch.nn.Linear(1, 1)
    resumed_optimizer = torch.optim.SGD(
        resumed_model.parameters(), lr=0.2, momentum=0.9
    )
    resumed_scheduler = LearningRate(
        base_lr=0.2,
        schedulers=[
            FlatCosineLRScheduler(
                total_epochs=10,
                warmup_iter=2,
                flat_epochs=4,
                no_aug_epochs=2,
                lr_gamma=0.1,
            )
        ],
    )(step_per_epoch=2, optimizer=resumed_optimizer)
    load_checkpoint(
        str(path),
        resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
    )

    assert resumed_scheduler.state_dict() == scheduler.state_dict()
    assert resumed_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


def test_flat_cosine_n_schedule_is_constant_after_quadratic_warmup():
    optimizer, scheduler = _make_scheduler(
        total_epochs=160,
        flat_epochs=7800,
        no_aug_epochs=12,
        lr_gamma=1.0,
    )

    trace = []
    for _ in range(330):
        optimizer.step()
        scheduler.step()
        trace.append(optimizer.param_groups[0]["lr"])

    assert trace[:3] == pytest.approx([0.0, 0.05, 0.2])
    assert trace[3:] == pytest.approx([0.2] * 327)


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"total_epochs": -1}, "total_epochs"),
        ({"warmup_iter": -1}, "warmup_iter"),
        ({"flat_epochs": -1}, "flat_epochs"),
        ({"no_aug_epochs": -1}, "no_aug_epochs"),
    ],
)
def test_flat_cosine_rejects_negative_epoch(overrides, field):
    with pytest.raises(ValueError, match=field):
        _make_scheduler(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"flat_epochs": 0},
        {"flat_epochs": 9, "no_aug_epochs": 2},
        {"flat_epochs": 11},
    ],
)
def test_flat_cosine_rejects_inconsistent_phase_lengths(overrides):
    with pytest.raises(ValueError, match="flat_epochs|no_aug_epochs|warmup_iter"):
        _make_scheduler(**overrides)


def test_flat_cosine_detects_step_drift_during_resume():
    optimizer, scheduler = _make_scheduler()
    optimizer.step()
    scheduler.step()
    state = scheduler.state_dict()
    optimizer.param_groups[0]["lr"] = 123.0

    with pytest.raises(ValueError, match="step drift"):
        scheduler.load_state_dict(state)


def test_flat_cosine_builds_from_learning_rate_dict_config():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.2)
    scheduler = LearningRate(
        schedulers=[
            {
                "name": "FlatCosineLRScheduler",
                "total_epochs": 10,
                "warmup_iter": 2,
                "flat_epochs": 4,
                "no_aug_epochs": 2,
                "lr_gamma": 0.1,
            }
        ]
    )(step_per_epoch=2, optimizer=optimizer)

    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.0


def test_flat_cosine_is_registered():
    scheduler = create(
        {
            "name": "FlatCosineLRScheduler",
            "total_epochs": 10,
            "warmup_iter": 2,
            "flat_epochs": 4,
            "no_aug_epochs": 2,
        }
    )

    assert isinstance(scheduler, FlatCosineLRScheduler)
