import json
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from detrs.engine.trainer import Trainer
from detrs.engine.training_protocol import TwoStageDetectionProtocol


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return server.getsockname()[1]


class _Criterion:
    def __init__(self):
        self.weight = None

    def set_distillation_weight(self, weight):
        self.weight = float(weight)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.criterion = _Criterion()


def _worker(rank, world_size, port, output_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        trainer = Trainer.__new__(Trainer)
        trainer.model = _Model()
        trainer.save_dir = output_dir
        trainer.training_protocol = TwoStageDetectionProtocol(
            "rtdetrv4",
            stop_epoch=50,
            ema_restart_decay=0.9998,
            current_gam_weight=5,
            gam_rho=50,
            gam_delta=1,
            gam_default_weight=20,
        )
        local = {
            "encoder_l1": torch.tensor(1.0 if rank == 0 else 3.0),
            "total_l1": torch.tensor(2.0 if rank == 0 else 6.0),
        }
        reduced = trainer._reduce_protocol_observation(local)
        trainer.training_protocol.after_successful_optimizer_step(reduced, {})
        trainer.training_protocol.after_epoch(0, {})
        trainer._execute_protocol_actions()

        assert trainer.training_protocol.current_gam_weight == 5
        assert trainer.model.criterion.weight == 5

        trainer.model.weight.grad = torch.tensor(float("inf") if rank == 1 else 1.0)
        assert not trainer._amp_gradients_are_finite()

        trainer.training_protocol.current_gam_weight = 5.0 + rank
        divergence_rejected = False
        try:
            trainer._synchronize_gam_weight(require_equal=True)
        except ValueError as error:
            divergence_rejected = "diverged across ranks" in str(error)
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(divergence_rejected, gathered, dst=0)
        if rank == 0:
            assert gathered == [True, True]
            Path(output_dir, "gam-ddp.json").write_text(
                json.dumps({"status": "APPROVE"}), encoding="utf-8"
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
def test_gam_allreduce_broadcast_amp_skip_and_divergence_rejection(tmp_path):
    mp.spawn(
        _worker,
        args=(2, _free_port(), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    assert json.loads((tmp_path / "gam-ddp.json").read_text()) == {"status": "APPROVE"}
