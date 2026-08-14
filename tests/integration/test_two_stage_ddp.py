import json
import os
import socket
from pathlib import Path

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.integration.test_two_stage_resume import _trainer


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return server.getsockname()[1]


def _worker(rank, world_size, port, output_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        trainer = _trainer(Path(output_dir))
        trainer.notify_validation({"bbox": 0.4})
        dist.barrier()
        trainer.status["epoch_id"] = 2
        trainer.training_protocol.before_epoch(2, {})
        trainer._execute_protocol_actions()
        state = trainer.training_protocol.state_dict()
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(state, gathered, dst=0)
        if rank == 0:
            assert gathered[0] == gathered[1]
            assert state["stage"] == 2
            assert (Path(output_dir) / "best_stg1.pth").is_file()
            (Path(output_dir) / "ddp-result.json").write_text(
                json.dumps({"status": "APPROVE"}), encoding="utf-8"
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
def test_rank0_publish_and_transition_state_are_synchronized(tmp_path):
    mp.spawn(
        _worker,
        args=(2, _free_port(), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    assert json.loads((tmp_path / "ddp-result.json").read_text()) == {
        "status": "APPROVE"
    }
