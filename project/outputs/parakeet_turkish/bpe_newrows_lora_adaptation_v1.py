"""INACTIVE declared BPE512 + encoder-LoRA phase; no worker admission or launch.

The caller must verify actual CLOSED qualifying phase1 and original-source
artifacts, tokenizer/data/ASR proofs and deserialization provenance. This utility
validates their supplied tensors; an artifact hash alone is not that admission.
The original207 Adam remains parked. Only its unchanged192 LoRA states are
copied into a separate active optimizer. No old15 head state is reused.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

TABLES = ("decoder.embedding.weight", "joint.head.weight", "joint.head.bias")
ROWS = (8199, 8711)
KIND = "HF_TSC_BPE512_new_rows_plus_encoder_LoRA_state_v1"
IDENTITY_KIND = "HF_TSC_BPE512_new_rows_plus_encoder_LoRA_identity_v1"
ACQUISITION_SHA = "bc26638f4874aaa5e92029e86c30bef0624a477514957d8f8d054d7e92088724"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def source_lora_mapping(names, groups):
    """Original heads-first serialized IDs -> stable explicit LoRA names."""
    names = tuple(names)
    heads = tuple(name for name in names if not name.endswith(("lora_A", "lora_B")))
    adapters = tuple(name for name in names if name.endswith(("lora_A", "lora_B")))
    require(len(names) == len(set(names)) == 207 and len(heads) == 15 and len(adapters) == 192
        and set(TABLES).issubset(heads) and all(name.startswith("encoder.") for name in adapters),
        "Explicit original head15/encoder-LoRA192 names required")
    require(len(groups) == 2 and groups[0]["params"] == list(range(15))
        and groups[1]["params"] == list(range(15, 207)), "Original207 Adam ID order changed")
    return heads, adapters, tuple(zip(range(15, 207), adapters))


class BPENewRowsLoRAAdaptation:
    """Two disjoint Adam histories; one masked195-tensor global clip.

    Encoder/projector forwards MUST be differentiable in the future worker;
    freezing their base weights or keeping encoder.eval() does not mean no_grad.
    This class deliberately does not patch any forward, sampler or RNG state.
    """

    def __init__(self, model, phase1_state, original207_state, *,
                 original_source_record, phase1_record, torch_module=None):
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.model = model
        from bpe_newrows_acquisition_v1 import BPEOnlyAcquisition, checked_parked_record
        acquisition_path = Path(__file__).with_name("bpe_newrows_acquisition_v1.py")
        require(hashlib.sha256(acquisition_path.read_bytes()).hexdigest() == ACQUISITION_SHA,
                "Reviewed strict phase1 loader changed")
        self.original_record = checked_parked_record(original_source_record)
        self.phase1_record = checked_parked_record(phase1_record)
        require(self.original_record["path"] != self.phase1_record["path"], "Two separate immutable artifacts required")
        source = original207_state
        self.names = tuple(source["weights"])
        self.head_names, self.adapter_names, mapping = source_lora_mapping(self.names, source["optimizer"]["param_groups"])
        self.source_step = source.get("step")
        require(source.get("schema") == 2 and type(self.source_step) is int and self.source_step > 0
            and source.get("selected_step") == self.source_step, "Original full207 selected BEST required")
        self.acquisition_updates = phase1_state.get("acquisition_optimizer_updates")
        require(type(self.acquisition_updates) is int and self.acquisition_updates > 0,
                "Select an actual acquired phase1 state, not untouched initialization")
        runtime = phase1_state.get("runtime_state", {})
        require(runtime.get("source_selected_step") == self.source_step
            and runtime.get("step") == self.source_step + self.acquisition_updates,
            "Phase1 cumulative and active-Adam clocks differ")
        require(tuple(phase1_state["phase_identity"]["source_weight_names"]) == self.names
            and phase1_state["phase_identity"]["parked_source_artifact"] == self.original_record
            and source["lora_metadata"] == phase1_state["lora_metadata"], "Phase1 original names/artifact/layout differs")
        self.lora_metadata = copy.deepcopy(source["lora_metadata"])
        self.parameters = dict(model.named_parameters())
        require(set(self.names).issubset(self.parameters), "A learned source tensor is missing")
        self.shapes = {name: tuple(self.parameters[name].shape) for name in self.names}
        self._compare_initial_weights(phase1_state["weights"], source["weights"])
        self._audit_original_lora(source, mapping)
        # Complete strict phase1 validation/restoration happens while only3 tables
        # are active. The old class is never asked to accept adapter unfreezing.
        phase1 = BPEOnlyAcquisition(model, self.names, self.original_record,
            source_lora_metadata=self.lora_metadata, lr=1e-4, torch_module=self.torch)
        self.inherited_runtime = phase1.load_phase_state(phase1_state)
        self.tables = phase1.tables
        self.row_optimizer = phase1.optimizer
        self.row_group = copy.deepcopy(phase1.declared_optimizer_group)
        self.adapters = tuple(self.parameters[name] for name in self.adapter_names)
        for parameter in self.adapters:
            parameter.requires_grad_(True)
            parameter.grad = None
        original_group = source["optimizer"]["param_groups"][1]
        group_values = copy.deepcopy({key: value for key, value in original_group.items() if key != "params"})
        self.lora_optimizer = self.torch.optim.AdamW([{"params": list(self.adapters), **group_values}])
        target = self.lora_optimizer.state_dict()
        target["state"] = {target_id: copy.deepcopy(source["optimizer"]["state"][source_id])
            for target_id, (source_id, _) in zip(target["param_groups"][0]["params"], mapping)}
        self.lora_optimizer.load_state_dict(target)
        self.lora_group = copy.deepcopy({key: value for key, value in self.lora_optimizer.param_groups[0].items() if key != "params"})
        self.mapping_sha256 = self._canonical(list(mapping))
        self.fixed_weight_sha256 = self._fixed_digest(phase1_state["weights"])
        self.phase2_updates = 0
        self._incomplete_update = False
        self._check_contract()
        # No full source207 CPU dictionary is retained by this instance.

    @staticmethod
    def _canonical(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def _bytes(self, tensor):
        return tensor.detach().cpu().contiguous().reshape(-1).view(self.torch.uint8).numpy().tobytes()

    def _bits_equal(self, left, right):
        return left.shape == right.shape and left.dtype == right.dtype and self._bytes(left) == self._bytes(right)

    def _compare_initial_weights(self, acquired, original):
        require(set(acquired) == set(original) == set(self.names), "Both explicit207 weight sets required")
        for name in self.names:
            before, after = original[name], acquired[name]
            require(self.torch.is_tensor(before) and self.torch.is_tensor(after)
                and before.dtype == after.dtype == self.torch.float32
                and bool(self.torch.isfinite(before).all()) and bool(self.torch.isfinite(after).all()),
                "Original/acquired learned tensor FP32/finiteness differs: " + name)
            if name not in TABLES:
                require(self._bits_equal(before, after), "Original204 frozen tensor bytes changed: " + name)
            else:
                old_rows = 8199 if name == TABLES[0] else 8204
                new_rows = 8711 if name == TABLES[0] else 8716
                require(before.shape[0] == old_rows and after.shape[0] == new_rows
                    and before.shape[1:] == after.shape[1:]
                    and self._bits_equal(before[:8199], after[:8199])
                    and self._bits_equal(before[8199:], after[8711:]),
                    "Protected old/six/blank/duration bytes changed: " + name)

    def _audit_original_lora(self, source, mapping):
        optimizer = source["optimizer"]
        require(set(optimizer["state"]) == set(range(207)), "Original full207 Adam states required")
        for index, expected_lr, decay in ((0, 1e-5, .01), (1, 3e-4, 0.)):
            group = optimizer["param_groups"][index]
            require(group["lr"] == expected_lr and group["weight_decay"] == decay
                and tuple(group["betas"]) == (.9, .999) and group["eps"] == 1e-8
                and group["amsgrad"] is False and group["maximize"] is False
                and group["capturable"] is False and group["differentiable"] is False
                and group["foreach"] is None and group["fused"] is None,
                "Declared original head/LoRA Adam group recipe differs")
        for source_id, name in mapping:
            self._check_moments(optimizer["state"][source_id], source["weights"][name], self.source_step)

    def _check_moments(self, state, parameter, clock, *, protected=False):
        require(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "Standard three-field Adam state required")
        step = state["step"]
        require(self.torch.is_tensor(step) and step.dtype == self.torch.float32 and step.ndim == 0
            and bool(self.torch.isfinite(step)) and float(step) == clock, "Adam scalar clock differs")
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            require(self.torch.is_tensor(value) and value.dtype == self.torch.float32 and value.shape == parameter.shape
                and bool(self.torch.isfinite(value).all()), "Adam moment shape/FP32/finiteness differs")
            if protected:
                require(not bool(self.torch.count_nonzero(value[:8199]))
                    and not bool(self.torch.count_nonzero(value[8711:])), "Protected rows contain old Adam moments")
        require(bool((state["exp_avg_sq"] >= 0).all()), "Negative Adam variance")

    def _fixed_digest(self, weights):
        digest = hashlib.sha256()
        for name in self.head_names:
            tensor = weights[name]
            digest.update(name.encode()); digest.update(str(tensor.dtype).encode()); digest.update(str(tuple(tensor.shape)).encode())
            if name in TABLES:
                digest.update(self._bytes(tensor[:8199])); digest.update(self._bytes(tensor[8711:]))
            else:
                digest.update(self._bytes(tensor))
        return digest.hexdigest()

    def _check_contract(self):
        require(self.model.config.vocab_size == 8711 and self.model.config.blank_token_id == 8192
            and tuple(self.model.config.durations) == (0, 1, 2, 3, 4)
            and all(self.parameters[name].dtype == self.torch.float32
                    and tuple(self.parameters[name].shape) == self.shapes[name] for name in self.names),
            "Declared model geometry/FP32 learned shapes differ")
        require({name for name, p in self.model.named_parameters() if p.requires_grad} == set(TABLES) | set(self.adapter_names),
                "Exactly3 tables plus192 encoder LoRA parameters must be active")
        for optimizer, parameters, group_values in ((self.row_optimizer, self.tables, self.row_group),
                                                    (self.lora_optimizer, self.adapters, self.lora_group)):
            groups = optimizer.param_groups
            require(len(groups) == 1 and len(groups[0]["params"]) == len(parameters)
                and all(a is b for a, b in zip(groups[0]["params"], parameters))
                and {k: v for k, v in groups[0].items() if k != "params"} == group_values
                and set(optimizer.state) == set(parameters), "Named active optimizer/group contract differs")
        require(self.row_group["lr"] == 1e-4 and self.lora_group["lr"] == 3e-4
            and self.row_group["weight_decay"] == self.lora_group["weight_decay"] == 0., "Phase2 constant LR/decay differs")

    @property
    def cumulative_step(self):
        return self.source_step + self.acquisition_updates + self.phase2_updates

    def _require_complete_update(self):
        require(not self._incomplete_update,
                "Incomplete two-optimizer update; only a fully validated phase-state load may recover it")

    def zero_grad(self):
        self._require_complete_update()
        self._check_contract()
        self.row_optimizer.zero_grad(set_to_none=True)
        self.lora_optimizer.zero_grad(set_to_none=True)

    def step(self, max_norm=1.):
        self._require_complete_update()
        self._check_contract()
        require(max_norm == 1., "Declared single global195 clipping norm is1")
        active = self.tables + self.adapters
        require(all(p.grad is not None and not p.grad.is_sparse and p.grad.shape == p.shape for p in active),
                "Complete dense accumulated gradients for all195 tensors required")
        for parameter in self.tables:
            parameter.grad[:8199].zero_(); parameter.grad[8711:].zero_()
        norm = self.torch.nn.utils.clip_grad_norm_(active, max_norm, error_if_nonfinite=True)
        # These calls are not atomic. Any exception leaves this instance poisoned.
        # No forward, RNG use or save may occur between them. Tensor clocks and
        # moments are audited at save/load, not transferred every training step.
        next_count = self.phase2_updates + 1
        self._incomplete_update = True
        self.row_optimizer.step()
        self.lora_optimizer.step()
        self.phase2_updates = next_count
        require(self.cumulative_step == self.source_step + self.acquisition_updates + next_count,
                "Completed-update metadata differs")
        self._incomplete_update = False
        return norm

    def phase_identity(self):
        return {"kind": IDENTITY_KIND, "utility_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_weight_names": list(self.names), "original_source_artifact": dict(self.original_record),
            "phase1_artifact": dict(self.phase1_record), "source_selected_step": self.source_step,
            "selected_acquisition_updates": self.acquisition_updates, "row_group": copy.deepcopy(self.row_group),
            "lora_group": copy.deepcopy(self.lora_group), "LoRA_names": list(self.adapter_names),
            "source_LoRA_ID_mapping_sha256": self.mapping_sha256, "fixed_head_rows_bytes_sha256": self.fixed_weight_sha256,
            "learned_parameter_shapes": {name: list(shape) for name, shape in self.shapes.items()},
            "active_parameter_tensors": 195, "fixed_shared_head_tensors": 12, "effective_new_rows": [8199, 8711],
            "protected_blank": 8192, "protected_six_rows": [8193, 8199], "protected_duration_rows": [8711, 8716],
            "mask_before_single_global_clip": True, "clip_max_norm": 1., "old15_head_moments_active": False,
            "original207_optimizer_artifact_unchanged": True, "original207_exact_resume_claimed": False,
            "two_optimizer_update_atomic": False, "failed_update_requires_validated_state_reload": True,
            "external_CLOSED_data_ASR_admission": "caller responsibility", "further_unfreeze_allowed": False,
            "same_phase_utility_state_restore_supported": True, "external_worker_resume_admission_supported": False}

    def _cpu_clone(self, value):
        if self.torch.is_tensor(value): return value.detach().cpu().clone()
        if isinstance(value, dict): return {key: self._cpu_clone(item) for key, item in value.items()}
        if isinstance(value, list): return [self._cpu_clone(item) for item in value]
        if isinstance(value, tuple): return tuple(self._cpu_clone(item) for item in value)
        return copy.deepcopy(value)

    def state_dict(self, *, runtime_state=None):
        self._require_complete_update()
        self._check_contract()
        from bpe_newrows_acquisition_v1 import checked_parked_record
        checked_parked_record(self.original_record); checked_parked_record(self.phase1_record)
        weights = {name: self.parameters[name].detach().cpu().clone() for name in self.names}
        require(all(bool(self.torch.isfinite(value).all()) for value in weights.values()),
                "Nonfinite learned weight cannot be saved")
        require(self._fixed_digest(weights) == self.fixed_weight_sha256, "Protected12 shared heads/table rows changed")
        row_state = self._cpu_clone(self.row_optimizer.state_dict())
        lora_state = self._cpu_clone(self.lora_optimizer.state_dict())
        self._validate_optimizer_state(row_state, self.tables, self.row_group,
                                       self.acquisition_updates + self.phase2_updates, protected=True)
        self._validate_optimizer_state(lora_state, self.adapters, self.lora_group,
                                       self.source_step + self.phase2_updates, protected=False)
        return {"schema": 1, "kind": KIND, "phase_identity": self.phase_identity(), "weights": weights,
            "complete_two_optimizer_update_boundary": True, "lora_metadata": copy.deepcopy(self.lora_metadata),
            "row_optimizer": row_state, "lora_optimizer": lora_state, "phase2_updates": self.phase2_updates,
            "cumulative_step": self.cumulative_step, "original207_optimizer_parked": dict(self.original_record),
            "phase1_optimizer_source": dict(self.phase1_record), "runtime_state": self._cpu_clone(runtime_state)}

    def load_phase_state(self, state):
        """SAME new-phase utility restore only; no external worker resume claim."""
        require(state.get("schema") == 1 and state.get("kind") == KIND
            and state.get("complete_two_optimizer_update_boundary") is True
            and state.get("phase_identity") == self.phase_identity() and state.get("lora_metadata") == self.lora_metadata
            and state.get("original207_optimizer_parked") == self.original_record
            and state.get("phase1_optimizer_source") == self.phase1_record, "Only SAME declared phase2 state admitted")
        count = state.get("phase2_updates")
        require(type(count) is int and count >= 0
            and state.get("cumulative_step") == self.source_step + self.acquisition_updates + count, "Phase2 clocks differ")
        weights = state.get("weights", {})
        require(set(weights) == set(self.names), "All207 learned weights must be serialized")
        for name, value in weights.items():
            require(self.torch.is_tensor(value) and value.shape == self.parameters[name].shape
                and value.dtype == self.torch.float32 and bool(self.torch.isfinite(value).all()), "Phase2 weight shape/finiteness differs")
        require(self._fixed_digest(weights) == self.fixed_weight_sha256, "Protected shared/old/six/blank/duration bytes differ")
        for key, params, group, clock, protected in (("row_optimizer", self.tables, self.row_group, self.acquisition_updates + count, True),
            ("lora_optimizer", self.adapters, self.lora_group, self.source_step + count, False)):
            self._validate_optimizer_state(state[key], params, group, clock, protected=protected)
        from bpe_newrows_acquisition_v1 import checked_parked_record
        checked_parked_record(self.original_record); checked_parked_record(self.phase1_record)
        # Validate every field before copying weights or active optimizer state.
        # A failed load also leaves the instance poisoned; successful full load
        # can recover a previous partial update without claiming atomic steps.
        self._incomplete_update = True
        with self.torch.no_grad():
            for name, value in weights.items(): self.parameters[name].copy_(value)
        self.row_optimizer.load_state_dict(copy.deepcopy(state["row_optimizer"]))
        self.lora_optimizer.load_state_dict(copy.deepcopy(state["lora_optimizer"]))
        self.phase2_updates = count; self._check_contract()
        self._incomplete_update = False
        return self._cpu_clone(state.get("runtime_state"))

    def _validate_optimizer_state(self, optimizer, params, group, clock, *, protected):
        groups = optimizer.get("param_groups", [])
        require(len(groups) == 1 and groups[0]["params"] == list(range(len(params)))
            and {k: v for k, v in groups[0].items() if k != "params"} == group
            and set(optimizer.get("state", {})) == set(groups[0]["params"]),
            "Phase2 optimizer IDs/groups/state differs")
        for identifier, parameter in zip(groups[0]["params"], params):
            self._check_moments(optimizer["state"][identifier], parameter, clock, protected=protected)

    def load_state_dict(self, state):
        """Restore this utility's own phase2 schema; caller restores opaque runtime."""
        return self.load_phase_state(state)

    def unfreeze(self):
        raise ValueError("Any further shared-head/old-row unfreeze requires a separately declared phase")
