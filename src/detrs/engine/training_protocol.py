"""Optional training lifecycle protocol hooks."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from ..core.workspace import register

__all__ = ["TrainingProtocol", "TwoStageDetectionProtocol"]


@register
class TrainingProtocol:
    """No-op, checkpointable training protocol interface.

    Hooks intentionally receive observations and status only. They do not
    receive the trainer, model, or optimizer, so protocol implementations
    cannot mutate training components through the hook API.
    """

    def preflight(self) -> None:
        pass

    def before_epoch(self, epoch: int, status: Mapping[str, Any]) -> None:
        pass

    def after_epoch(self, epoch: int, status: Mapping[str, Any]) -> None:
        pass

    def after_validation(self, metrics: Mapping[str, Any]) -> None:
        pass

    def after_save(self, training_state: Mapping[str, Any]) -> None:
        pass

    def after_load(
        self, training_state: Mapping[str, Any], metadata: Mapping[str, Any]
    ) -> None:
        pass

    def after_backward(
        self,
        gradients: Mapping[str, torch.Tensor],
        status: Mapping[str, Any],
    ) -> Optional[Any]:
        return None

    def after_successful_optimizer_step(
        self, observation: Any, status: Mapping[str, Any]
    ) -> None:
        pass

    @property
    def identity(self) -> str:
        return type(self).__name__

    @property
    def checkpoint_stage(self) -> Any:
        return self.state_dict().get("stage", "default")

    def validate_checkpoint_stage(self, stage: Any) -> None:
        if not isinstance(stage, (str, int)):
            raise ValueError("training protocol stage must be a string or integer")

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def validate_state_dict(
        self, state_dict: Mapping[str, Any], checkpoint_path: str
    ) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("training protocol state must be a mapping")

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.validate_state_dict(state_dict, "")

    def checkpoint_state(self, model_identity: str) -> Dict[str, Any]:
        state = deepcopy(self.state_dict())
        if not isinstance(state, Mapping):
            raise TypeError("training protocol state_dict() must return a mapping")
        training_state = {
            "model_identity": model_identity,
            "protocol_identity": self.identity,
            "protocol_stage": self.checkpoint_stage,
            "protocol_state": state,
        }
        self.after_save(training_state)
        return training_state

    def pop_actions(self) -> list[Dict[str, Any]]:
        return []

    def restore_actions(self, actions: list[Mapping[str, Any]]) -> None:
        pass

    def complete_action(self, action: Mapping[str, Any], **result: Any) -> None:
        pass


@register
class TwoStageDetectionProtocol(TrainingProtocol):
    """Deterministic best-checkpoint and EMA restart state machine.

    Tracks the best evaluation metric in stage one; from `stop_epoch` the
    EMA is restarted from the best student weights and its decay is stepped
    down by `decay_decrement` on every subsequent improvement. For
    `rtdetrv4` it can additionally run GAM, adapting a loss weight from the
    observed encoder/total gradient-L1 percentage toward the
    [`gam_rho` - `gam_delta`, `gam_rho` + `gam_delta`] band.

    Args:
        family (str): One of `dfine`, `deim`, `deimv2`, `rtdetrv4`.
        stop_epoch (int): Epoch (1-based) at which stage two starts.
        ema_restart_decay (float): Initial EMA decay used after restart,
            between 0 and 1.
        metric_name (str): Evaluation metric tracked for best checkpoints,
            e.g. `bbox`.
        decay_decrement (float): EMA decay decrement applied per
            improvement; must stay below `ema_restart_decay`.
        current_gam_weight (float|None): Current GAM loss weight; required
            when GAM is enabled.
        gam_rho (float|None): Target band center (percent) of the
            encoder/total gradient-L1 ratio.
        gam_delta (float|None): Half-width of the GAM target band.
        gam_default_weight (float|None): GAM weight applied once training
            reaches `stop_epoch` or the ratio degenerates.
    """

    FAMILIES = {"dfine", "deim", "deimv2", "rtdetrv4"}

    def __init__(
        self,
        family: str,
        stop_epoch: int,
        ema_restart_decay: float,
        metric_name: str = "bbox",
        decay_decrement: float = 0.0001,
        current_gam_weight: Optional[float] = None,
        gam_rho: Optional[float] = None,
        gam_delta: Optional[float] = None,
        gam_default_weight: Optional[float] = None,
    ):
        if family not in self.FAMILIES:
            raise ValueError("unsupported two-stage family: {}".format(family))
        if not isinstance(stop_epoch, int) or stop_epoch < 1:
            raise ValueError("two-stage stop_epoch must be a positive integer")
        if not 0 < ema_restart_decay < 1:
            raise ValueError("EMA restart decay must be between zero and one")
        if not 0 < decay_decrement < ema_restart_decay:
            raise ValueError("EMA decay decrement must be positive and bounded")
        if not metric_name:
            raise ValueError("two-stage metric_name must be non-empty")
        gam_values = (gam_rho, gam_delta, gam_default_weight)
        if any(value is not None for value in gam_values) and not all(
            value is not None for value in gam_values
        ):
            raise ValueError("GAM requires rho, delta, and default weight")
        gam_enabled = all(value is not None for value in gam_values)
        if gam_enabled and family != "rtdetrv4":
            raise ValueError("GAM is only supported for rtdetrv4")
        if gam_enabled:
            assert gam_rho is not None
            assert gam_delta is not None
            assert gam_default_weight is not None
            rho = float(gam_rho)
            delta = float(gam_delta)
            default_weight = float(gam_default_weight)
            if not 0 < rho < 100 or not 0 < delta < min(rho, 100 - rho):
                raise ValueError("GAM rho/delta must define a bounded percentage")
            if not torch.isfinite(torch.tensor(default_weight)) or default_weight < 0:
                raise ValueError("GAM default weight must be finite and non-negative")
            if current_gam_weight is None:
                raise ValueError("GAM requires current_gam_weight")
        if current_gam_weight is not None and (
            not torch.isfinite(torch.tensor(float(current_gam_weight)))
            or float(current_gam_weight) < 0
        ):
            raise ValueError("current_gam_weight must be finite and non-negative")
        self.family = family
        self.stop_epoch = stop_epoch
        self.ema_restart_decay = float(ema_restart_decay)
        self.metric_name = metric_name
        self.decay_decrement = float(decay_decrement)
        self.stage = 1
        self.top_metric: Optional[float] = None
        self.stage_metric: Optional[float] = None
        self.restart_count = 0
        self.current_decay = float(ema_restart_decay)
        self.companion_basename: Optional[str] = None
        self.companion_sha256: Optional[str] = None
        self.current_gam_weight = (
            None if current_gam_weight is None else float(current_gam_weight)
        )
        self.gam_rho = None if gam_rho is None else float(gam_rho)
        self.gam_delta = None if gam_delta is None else float(gam_delta)
        self.gam_default_weight = (
            None if gam_default_weight is None else float(gam_default_weight)
        )
        self._gam_percentage_sum = 0.0
        self._gam_observation_count = 0
        self._actions: list[Dict[str, Any]] = []

    @property
    def gam_enabled(self) -> bool:
        return self.gam_rho is not None

    @property
    def identity(self) -> str:
        return "TwoStageDetectionProtocol:{}".format(self.family)

    @property
    def checkpoint_stage(self) -> int:
        return self.stage

    def preflight(self) -> None:
        self._actions.clear()
        self._gam_percentage_sum = 0.0
        self._gam_observation_count = 0

    def before_epoch(self, epoch: int, status: Mapping[str, Any]) -> None:
        del status
        if epoch < self.stop_epoch or self.stage == 2:
            return
        if epoch != self.stop_epoch:
            raise ValueError(
                "two-stage transition was missed: epoch={}, stop_epoch={}".format(
                    epoch, self.stop_epoch
                )
            )
        if not self.companion_basename or not self.companion_sha256:
            raise FileNotFoundError("stage-1 best checkpoint companion is missing")
        self._actions.append(
            {
                "decay": self.ema_restart_decay,
                "current_gam_weight": self.current_gam_weight,
                "name": "transition",
                "path": self.companion_basename,
                "sha256": self.companion_sha256,
            }
        )

    def after_backward(
        self,
        gradients: Mapping[str, torch.Tensor],
        status: Mapping[str, Any],
    ) -> Optional[Any]:
        del status
        if not self.gam_enabled:
            return None
        total = None
        encoder = None
        for name, gradient in gradients.items():
            value = gradient.detach().abs().sum()
            total = value if total is None else total + value
            normalized_name = name.removeprefix("module.")
            if normalized_name.startswith("encoder.encoder."):
                encoder = value if encoder is None else encoder + value
        if total is None:
            return None
        if encoder is None:
            encoder = torch.zeros_like(total)
        return {"encoder_l1": encoder, "total_l1": total}

    def after_successful_optimizer_step(
        self, observation: Any, status: Mapping[str, Any]
    ) -> None:
        del status
        if not self.gam_enabled:
            return
        if not isinstance(observation, Mapping) or set(observation) != {
            "encoder_l1",
            "total_l1",
        }:
            raise ValueError("GAM requires encoder_l1 and total_l1 observation")
        encoder = float(torch.as_tensor(observation["encoder_l1"]).item())
        total = float(torch.as_tensor(observation["total_l1"]).item())
        if not torch.isfinite(torch.tensor([encoder, total])).all():
            raise ValueError("GAM gradient observation must be finite")
        percentage = 100.0 * encoder / total if total > 0 else 0.0
        self._gam_percentage_sum += percentage
        self._gam_observation_count += 1

    def _next_gam_weight(self, epoch: int, average_percentage: float) -> float:
        if not self.gam_enabled or self.current_gam_weight is None:
            raise RuntimeError("GAM weight update requires an enabled GAM protocol")
        assert self.gam_rho is not None
        assert self.gam_delta is not None
        assert self.gam_default_weight is not None
        current = self.current_gam_weight
        if average_percentage < 1e-6 or epoch >= self.stop_epoch:
            return self.gam_default_weight
        lower = self.gam_rho - self.gam_delta
        upper = self.gam_rho + self.gam_delta
        if lower <= average_percentage <= upper or current <= 1e-6:
            return current
        target = upper if average_percentage < lower else lower
        current_fraction = average_percentage / 100.0
        target_fraction = target / 100.0
        denominator = current_fraction * (1 - target_fraction)
        if abs(denominator) < 1e-9:
            return current
        ratio = max(
            target_fraction * (1 - current_fraction) / denominator,
            0.1,
        )
        return min(max(current * ratio, current / 10), current * 10)

    def after_epoch(self, epoch: int, status: Mapping[str, Any]) -> None:
        del status
        if not self.gam_enabled:
            return
        average = (
            self._gam_percentage_sum / self._gam_observation_count
            if self._gam_observation_count
            else 0.0
        )
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self._actions.append(
            {
                "name": "set_gam_weight",
                "weight": self._next_gam_weight(epoch, average) if rank == 0 else None,
            }
        )
        self._gam_percentage_sum = 0.0
        self._gam_observation_count = 0

    def after_validation(self, metrics: Mapping[str, Any]) -> None:
        if self.metric_name not in metrics:
            raise ValueError(
                "validation metrics are missing required {} metric".format(
                    self.metric_name
                )
            )
        metric = float(metrics[self.metric_name])
        if not torch.isfinite(torch.tensor(metric)):
            raise ValueError("validation metric must be finite")
        improved_global = self.top_metric is None or metric > self.top_metric
        improved_stage = self.stage_metric is None or metric > self.stage_metric
        if improved_stage:
            self.stage_metric = metric
        if improved_global:
            self.top_metric = metric
            self._actions.append(
                {
                    "metric": metric,
                    "name": "save_best",
                    "path": "best_stg{}.pth".format(self.stage),
                    "stage": self.stage,
                }
            )
        elif self.stage == 2 and not improved_stage:
            if not self.companion_basename or not self.companion_sha256:
                raise FileNotFoundError("stage-1 best checkpoint companion is missing")
            next_decay = self.current_decay - self.decay_decrement
            if next_decay <= 0:
                raise ValueError("EMA decay adjustment became non-positive")
            self._actions.append(
                {
                    "decay": next_decay,
                    "current_gam_weight": self.current_gam_weight,
                    "name": "restart",
                    "path": self.companion_basename,
                    "restart_count": self.restart_count + 1,
                    "sha256": self.companion_sha256,
                    "top_metric": self.top_metric,
                }
            )

    def pop_actions(self) -> list[Dict[str, Any]]:
        actions, self._actions = self._actions, []
        return deepcopy(actions)

    def restore_actions(self, actions: list[Mapping[str, Any]]) -> None:
        self._actions = [deepcopy(dict(action)) for action in actions] + self._actions

    def complete_action(self, action: Mapping[str, Any], **result: Any) -> None:
        if action["name"] == "save_best" and action["stage"] == 1:
            basename = result.get("basename")
            sha256 = result.get("sha256")
            if not basename or Path(basename).name != basename:
                raise ValueError("stage-1 companion basename is invalid")
            if not isinstance(sha256, str) or len(sha256) != 64:
                raise ValueError("stage-1 companion SHA-256 is invalid")
            int(sha256, 16)
            self.companion_basename = basename
            self.companion_sha256 = sha256
        elif action["name"] == "transition":
            self.stage = 2
            self.stage_metric = None
            self.current_decay = float(action["decay"])
            self.current_gam_weight = action["current_gam_weight"]
        elif action["name"] == "restart":
            self.stage = 2
            self.stage_metric = None
            self.restart_count = int(action["restart_count"])
            self.top_metric = action["top_metric"]
            self.current_gam_weight = action["current_gam_weight"]
            self.current_decay = float(action["decay"])
        elif action["name"] == "set_gam_weight":
            candidate = result.get("weight", action.get("weight"))
            if candidate is None:
                raise ValueError("GAM weight action is missing a value")
            weight = float(candidate)
            if not torch.isfinite(torch.tensor(weight)) or weight < 0:
                raise ValueError("GAM weight must be finite and non-negative")
            self.current_gam_weight = weight

    def validate_checkpoint_stage(self, stage: Any) -> None:
        if stage not in (1, 2):
            raise ValueError("two-stage protocol stage must be 1 or 2")

    def state_dict(self) -> Dict[str, Any]:
        return {
            "companion_basename": self.companion_basename,
            "companion_sha256": self.companion_sha256,
            "current_decay": self.current_decay,
            "current_gam_weight": self.current_gam_weight,
            "decay_decrement": self.decay_decrement,
            "ema_restart_decay": self.ema_restart_decay,
            "family": self.family,
            "gam_default_weight": self.gam_default_weight,
            "gam_delta": self.gam_delta,
            "gam_rho": self.gam_rho,
            "metric_name": self.metric_name,
            "restart_count": self.restart_count,
            "stage": self.stage,
            "stage_metric": self.stage_metric,
            "stop_epoch": self.stop_epoch,
            "top_metric": self.top_metric,
        }

    def validate_state_dict(
        self, state_dict: Mapping[str, Any], checkpoint_path: str
    ) -> None:
        super().validate_state_dict(state_dict, checkpoint_path)
        expected = {
            "companion_basename",
            "companion_sha256",
            "current_decay",
            "current_gam_weight",
            "decay_decrement",
            "ema_restart_decay",
            "family",
            "gam_default_weight",
            "gam_delta",
            "gam_rho",
            "metric_name",
            "restart_count",
            "stage",
            "stage_metric",
            "stop_epoch",
            "top_metric",
        }
        if set(state_dict) != expected:
            raise ValueError("invalid two-stage protocol state fields")
        if state_dict["family"] != self.family:
            raise ValueError("two-stage family mismatch")
        if state_dict["stop_epoch"] != self.stop_epoch:
            raise ValueError("two-stage stop epoch mismatch")
        if state_dict["metric_name"] != self.metric_name:
            raise ValueError("two-stage metric identity mismatch")
        if state_dict["ema_restart_decay"] != self.ema_restart_decay:
            raise ValueError("two-stage EMA restart decay mismatch")
        if state_dict["decay_decrement"] != self.decay_decrement:
            raise ValueError("two-stage EMA decay decrement mismatch")
        for field in ("gam_rho", "gam_delta", "gam_default_weight"):
            if state_dict[field] != getattr(self, field):
                raise ValueError("two-stage GAM configuration mismatch")
        self.validate_checkpoint_stage(state_dict["stage"])
        companion = state_dict["companion_basename"]
        companion_sha = state_dict["companion_sha256"]
        if (companion is None) != (companion_sha is None):
            raise ValueError("companion basename and SHA-256 must both be present")
        if companion is not None:
            if Path(companion).name != companion:
                raise ValueError("companion_basename must not contain a path")
            if not isinstance(companion_sha, str) or len(companion_sha) != 64:
                raise ValueError("invalid companion SHA-256")
            int(companion_sha, 16)
        if state_dict["stage"] == 2 and companion is None:
            raise ValueError("stage 2 requires a stage-1 companion")
        decay = state_dict["current_decay"]
        if not isinstance(decay, (float, int)) or not 0 < decay < 1:
            raise ValueError("invalid current EMA decay")
        metric = state_dict["top_metric"]
        if metric is not None and not torch.isfinite(torch.tensor(float(metric))):
            raise ValueError("invalid top metric")
        stage_metric = state_dict["stage_metric"]
        if stage_metric is not None and not torch.isfinite(
            torch.tensor(float(stage_metric))
        ):
            raise ValueError("invalid stage metric")
        if (
            not isinstance(state_dict["restart_count"], int)
            or state_dict["restart_count"] < 0
        ):
            raise ValueError("invalid restart count")
        gam_weight = state_dict["current_gam_weight"]
        if self.gam_enabled and gam_weight is None:
            raise ValueError("RT-DETRv4 GAM checkpoint weight is missing")
        if gam_weight is not None and (
            not isinstance(gam_weight, (float, int))
            or not torch.isfinite(torch.tensor(float(gam_weight)))
            or float(gam_weight) < 0
        ):
            raise ValueError("invalid current GAM weight")

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.validate_state_dict(state_dict, "")
        self.stage = int(state_dict["stage"])
        self.top_metric = state_dict["top_metric"]
        self.stage_metric = state_dict["stage_metric"]
        self.restart_count = state_dict["restart_count"]
        self.current_decay = float(state_dict["current_decay"])
        self.companion_basename = state_dict["companion_basename"]
        self.companion_sha256 = state_dict["companion_sha256"]
        self.current_gam_weight = state_dict["current_gam_weight"]
        self._gam_percentage_sum = 0.0
        self._gam_observation_count = 0
        self._actions.clear()
