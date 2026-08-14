import torch
from torch import nn

from ppdet_pytorch.engine.training_protocol import TrainingProtocol
from ppdet_pytorch.utils.checkpoint import load_checkpoint, save_checkpoint


class _CounterProtocol(TrainingProtocol):
    def __init__(self):
        self.updates = 0
        self.saved = 0
        self.loaded = 0

    def state_dict(self):
        return {"updates": self.updates}

    def validate_state_dict(self, state_dict, checkpoint_path):
        super().validate_state_dict(state_dict, checkpoint_path)
        if set(state_dict) != {"updates"}:
            raise ValueError("corrupt protocol state")

    def load_state_dict(self, state_dict):
        self.validate_state_dict(state_dict, "")
        self.updates = state_dict["updates"]

    def after_save(self, training_state):
        assert training_state["protocol_stage"] == "default"
        self.saved += 1

    def after_load(self, training_state, metadata):
        assert metadata["training_state"] is training_state
        self.loaded += 1


def _step(model, optimizer, scheduler, protocol):
    optimizer.zero_grad()
    loss = model(torch.tensor([[2.0]])).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    protocol.updates += 1
    return loss.detach()


def _components(seed):
    torch.manual_seed(seed)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    return model, optimizer, scheduler, _CounterProtocol()


def test_protocol_state_resume_matches_uninterrupted(tmp_path):
    model, optimizer, scheduler, protocol = _components(7)
    _step(model, optimizer, scheduler, protocol)
    path = tmp_path / "epoch.pth"
    save_checkpoint(
        model,
        optimizer,
        epoch=1,
        iteration=1,
        save_path=str(path),
        scheduler=scheduler,
        training_state=protocol.checkpoint_state("Tiny"),
    )
    assert protocol.saved == 1
    expected_loss = _step(model, optimizer, scheduler, protocol)
    expected_state = model.state_dict()

    resumed_model, resumed_optimizer, resumed_scheduler, resumed_protocol = _components(
        99
    )
    load_checkpoint(
        str(path),
        resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        protocol=resumed_protocol,
        expected_model_identity="Tiny",
        restore_rng=True,
    )
    actual_loss = _step(
        resumed_model, resumed_optimizer, resumed_scheduler, resumed_protocol
    )

    assert torch.equal(actual_loss, expected_loss)
    assert resumed_protocol.updates == protocol.updates
    assert resumed_protocol.loaded == 1
    for name, tensor in resumed_model.state_dict().items():
        assert torch.equal(tensor, expected_state[name])
