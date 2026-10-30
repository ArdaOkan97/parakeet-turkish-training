"""Inactive row-only acquisition boundary; this file does not launch training.

Adam stores three full tables, but masked gradients permit updates only in
8199:8711. Source207 Adam is an opaque, separately pinned artifact, never loaded
here. The caller owns CLOSED-source admission, ASR/data gates and runtime guards.
Torch is imported only when constructing the boundary without torch_module.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

TABLE_NAMES = ("decoder.embedding.weight", "joint.head.weight", "joint.head.bias")
BPE_ROWS = (8199, 8711)
DURATION_ROWS = (8711, 8716)
SOURCE_WEIGHT_COUNT = 207
PHASE_KIND = "HF_TSC_BPE512_new_rows_only_acquisition_state_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def artifact_record(path):
    """Byte binding only: this does not deserialize or certify a CLOSED source."""
    path = Path(path).resolve()
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "Parked source artifact changed while hashing")
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": after.st_size}


def checked_parked_record(value):
    require(isinstance(value, dict) and set(value) == {"path", "sha256", "bytes"},
            "Explicit separately pinned source artifact record required")
    require(type(value["bytes"]) is int and value["bytes"] > 0
            and Path(value["path"]).is_absolute(), "Positive absolute parked artifact required")
    require(artifact_record(value["path"]) == value, "Parked source artifact changed")
    return dict(value)


class BPEOnlyAcquisition:
    """Fresh three-table AdamW; never pass a source optimizer to this API.

    Call step() after the caller's complete accumulated native ASR backward.
    It masks BEFORE clipping. Keep the frozen decoder/joint forward under
    autograd; scope no_grad to the encoder/projector only. No forward is patched.
    """

    def __init__(self, model, source_weight_names, parked_source_record, *,
                 source_lora_metadata, lr=1e-4, torch_module=None):
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.model = model
        require(math.isfinite(lr) and lr > 0, "Positive declared acquisition LR required")
        self.base_lr = float(lr)
        self.names = tuple(source_weight_names)
        require(len(self.names) == SOURCE_WEIGHT_COUNT and len(set(self.names)) == SOURCE_WEIGHT_COUNT
                and all(isinstance(name, str) for name in self.names)
                and set(TABLE_NAMES).issubset(self.names), "Exact explicit207 learned-weight names required")
        self.parked_source = checked_parked_record(parked_source_record)
        self.lora_metadata = copy.deepcopy(source_lora_metadata)
        self.parameters = dict(model.named_parameters())
        require(set(self.names).issubset(self.parameters), "A frozen learned source tensor is missing")
        for name in self.names:
            require(self.parameters[name].dtype == self.torch.float32, "All207 learned weights must be FP32")
        embedding, head, bias = (self.parameters[name] for name in TABLE_NAMES)
        require(embedding.ndim == head.ndim == 2 and bias.ndim == 1
                and embedding.shape[0] == 8711 and head.shape[0] == bias.shape[0] == 8716,
                "Expected embedding8711 and joint8716 expanded geometry")
        config = getattr(model, "config", None)
        require(config is not None and config.vocab_size == 8711 and config.blank_token_id == 8192
                and tuple(config.durations) == (0, 1, 2, 3, 4), "Student vocabulary/blank/durations differ")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.tables = (embedding, head, bias)
        for parameter in self.tables:
            parameter.requires_grad_(True)
        self.optimizer = self.torch.optim.AdamW(list(self.tables), lr=self.base_lr, weight_decay=0.0)
        self.declared_optimizer_group = copy.deepcopy({key: value for key, value in
            self.optimizer.param_groups[0].items() if key != "params"})
        self.optimizer_updates_completed = 0
        self._check_contract()

    def _check_contract(self):
        require({name for name, value in self.model.named_parameters() if value.requires_grad}
                == set(TABLE_NAMES), "Only the three acquisition tables may require gradients")
        groups = self.optimizer.param_groups
        require(len(groups) == 1 and len(groups[0]["params"]) == 3
                and all(left is right for left, right in zip(groups[0]["params"], self.tables))
                and groups[0]["weight_decay"] == 0.0
                and tuple(groups[0]["betas"]) == (.9, .999) and groups[0]["eps"] == 1e-8
                and not groups[0]["amsgrad"] and not groups[0]["maximize"]
                and math.isfinite(groups[0]["lr"]) and groups[0]["lr"] > 0,
                "Fresh three-table zero-decay AdamW contract changed")
        require(set(self.optimizer.state).issubset(set(self.tables)), "Foreign/source optimizer state loaded")
        require({key: value for key, value in groups[0].items() if key != "params"}
                == self.declared_optimizer_group,
                "Constant phase optimizer recipe changed; scheduler/unfreeze policy is not declared")

    def zero_grad(self):
        self._check_contract()
        self.optimizer.zero_grad(set_to_none=True)

    def mask_then_clip(self, max_norm=1.0):
        """Mask lexicalold/blank/six/duration gradients before the norm is formed."""
        self._check_contract()
        require(math.isfinite(max_norm) and max_norm > 0, "Positive clipping norm required")
        require(all(parameter.grad is not None for parameter in self.tables),
                "Complete accumulated objective gradients for all three tables required")
        lower, upper = BPE_ROWS
        for parameter in self.tables:
            if parameter.grad is not None:
                require(not parameter.grad.is_sparse and parameter.grad.shape == parameter.shape,
                        "Dense matching table gradient required")
                parameter.grad[:lower].zero_()
                parameter.grad[upper:].zero_()
        return self.torch.nn.utils.clip_grad_norm_(self.tables, max_norm, error_if_nonfinite=True)

    def step(self, max_norm=1.0):
        norm = self.mask_then_clip(max_norm)
        self.optimizer.step()
        self.optimizer_updates_completed += 1
        # No six-character boost, BPE multiplier, or deferred unfreeze is applied.
        return norm

    def phase_identity(self):
        source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return {"kind": "HF_TSC_BPE512_new_rows_only_acquisition_identity_v1",
                "utility_sha256": source_hash, "source_weight_names": list(self.names),
                "parked_source_artifact": dict(self.parked_source), "base_lr": self.base_lr,
                "active_table_names": list(TABLE_NAMES), "effective_update_rows": list(BPE_ROWS),
                "blank_id": 8192, "protected_six_rows": [8193, 8199],
                "protected_duration_rows": list(DURATION_ROWS), "optimizer_weight_decay": 0.0,
                "Adam_betas": [.9, .999], "Adam_eps": 1e-8,
                "declared_optimizer_group": copy.deepcopy(self.declared_optimizer_group),
                "optimizer_storage": "three full expanded tables; not compact row Parameters",
                "mask_before_clipping": True, "old_source_optimizer_active": False,
                "six_character_delta_multiplier_applied": False, "BPE_delta_multiplier": 1.0,
                "unfreeze_allowed": False, "closed_source_admission_performed_by_utility": False}

    def _cpu_clone(self, value):
        if isinstance(value, self.torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: self._cpu_clone(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._cpu_clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._cpu_clone(item) for item in value)
        return copy.deepcopy(value)

    def serialize_weights(self):
        """Explicit207-name export independent of requires_grad; includes frozen LoRA."""
        self._check_contract()
        return {name: self.parameters[name].detach().cpu().clone() for name in self.names}

    def state_dict(self, *, runtime_state=None):
        """Return a NEW phase state, not a legacy schema2/source-exact checkpoint.

        The caller must separately bind runtime sampler/RNG/history/gates. This
        utility neither claims nor constructs those actual ASR worker proofs.
        """
        self._check_contract()
        checked_parked_record(self.parked_source)
        return {"schema": 1, "kind": PHASE_KIND, "phase_identity": self.phase_identity(),
                "weights": self.serialize_weights(), "lora_metadata": copy.deepcopy(self.lora_metadata),
                "acquisition_optimizer": self._cpu_clone(self.optimizer.state_dict()),
                "acquisition_optimizer_updates": self.optimizer_updates_completed,
                "parked_source_optimizer": {"artifact": dict(self.parked_source),
                    "expected_source_optimizer_states": 207, "deserialized_by_utility": False,
                    "loaded_into_acquisition_optimizer": False,
                    "actual_CLOSED_full_state_admission": "external caller responsibility"},
                "runtime_state": self._cpu_clone(runtime_state)}

    def load_phase_state(self, state):
        """Restore this phase only. A source207 optimizer/exact state is rejected."""
        require(state.get("schema") == 1 and state.get("kind") == PHASE_KIND
                and state.get("phase_identity") == self.phase_identity()
                and state.get("lora_metadata") == self.lora_metadata,
                "Only SAME acquisition phase state is accepted; legacy source-exact is forbidden")
        require(state.get("parked_source_optimizer") == {
                    "artifact": dict(self.parked_source), "expected_source_optimizer_states": 207,
                    "deserialized_by_utility": False, "loaded_into_acquisition_optimizer": False,
                    "actual_CLOSED_full_state_admission": "external caller responsibility"},
                "Parked source optimizer reference/scope changed")
        checked_parked_record(self.parked_source)
        require(set(state["weights"]) == set(self.names), "All207 frozen+active weight names required")
        for name, tensor in state["weights"].items():
            require(isinstance(tensor, self.torch.Tensor) and tensor.dtype == self.torch.float32
                    and tensor.shape == self.parameters[name].shape and bool(self.torch.isfinite(tensor).all()),
                    "Phase learned-weight shape/finiteness differs: " + name)
        optimizer = state["acquisition_optimizer"]
        groups = optimizer.get("param_groups", [])
        require(len(groups) == 1 and len(groups[0]["params"]) == 3 and groups[0]["weight_decay"] == 0.0,
                "This phase has one fresh three-table optimizer, never source207 Adam")
        require(tuple(groups[0]["betas"]) == (.9, .999) and groups[0]["eps"] == 1e-8
                and not groups[0]["amsgrad"] and not groups[0]["maximize"]
                and math.isfinite(groups[0]["lr"]) and groups[0]["lr"] > 0,
                "Phase optimizer recipe changed")
        require({key: value for key, value in groups[0].items() if key != "params"}
                == self.declared_optimizer_group,
                "SAME phase must retain every declared optimizer hyperparameter")
        ids = groups[0]["params"]
        require(len(set(ids)) == 3 and set(optimizer.get("state", {})).issubset(ids), "Foreign optimizer IDs")
        updates = state.get("acquisition_optimizer_updates")
        require(type(updates) is int and updates >= 0, "Nonnegative phase update count required")
        states = optimizer.get("state", {})
        require((not states) if updates == 0 else set(states) == set(ids),
                "Phase Adam state must be empty at zero updates and complete thereafter")
        for identifier, parameter in zip(ids, self.tables):
            moments = states.get(identifier)
            if moments is None:
                continue
            require(set(moments) == {"step", "exp_avg", "exp_avg_sq"}, "Standard phase Adam moments required")
            for key in ("exp_avg", "exp_avg_sq"):
                tensor = moments[key]
                require(tensor.dtype == self.torch.float32 and tensor.shape == parameter.shape
                        and bool(self.torch.isfinite(tensor).all())
                        and not bool(self.torch.count_nonzero(tensor[:BPE_ROWS[0]]))
                        and not bool(self.torch.count_nonzero(tensor[BPE_ROWS[1]:])),
                        "Protected rows have foreign/old Adam moments")
            require(bool((moments["exp_avg_sq"] >= 0).all()), "Negative phase Adam variance")
            clock = moments["step"]
            require(isinstance(clock, self.torch.Tensor) and clock.numel() == 1
                    and bool(self.torch.isfinite(clock)) and float(clock) == updates,
                    "Every phase Adam clock must equal the completed integer update count")
        with self.torch.no_grad():
            for name, tensor in state["weights"].items():
                self.parameters[name].copy_(tensor)
        # Deep-copy CPU step tensors: load_state_dict may otherwise alias its input.
        self.optimizer.load_state_dict(copy.deepcopy(optimizer))
        self.optimizer_updates_completed = updates
        self._check_contract()
        return self._cpu_clone(state.get("runtime_state"))

    def unfreeze(self):
        raise ValueError("Deferred unfreeze requires a separately declared optimizer/phase policy")
