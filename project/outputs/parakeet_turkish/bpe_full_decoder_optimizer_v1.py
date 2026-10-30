"""Inactive decoder-only BPE optimizer; importing this file imports no ML runtime.

The actual 15 model tensors are clipped once. Disjoint storage-sharing views
give mature lexical/duration slices their retained Adam clock and appended BPE
slices a fresh clock. Encoder LoRA weights and their source moments are parked
under one immutable full-state source record. This is also the 8199 control path.
"""
from __future__ import annotations

import copy
import hashlib
import json

OLD_VOCAB = 8199
BPE_VOCAB = 8711
BLANK = 8192
DURATIONS = 5
TABLES = ("decoder.embedding.weight", "joint.head.weight", "joint.head.bias")
SHARED = ("encoder_projector.weight", "encoder_projector.bias",
          "decoder.lstm.weight_ih_l0", "decoder.lstm.weight_hh_l0",
          "decoder.lstm.bias_ih_l0", "decoder.lstm.bias_hh_l0",
          "decoder.lstm.weight_ih_l1", "decoder.lstm.weight_hh_l1",
          "decoder.lstm.bias_ih_l1", "decoder.lstm.bias_hh_l1",
          "decoder.decoder_projector.weight", "decoder.decoder_projector.bias")
KIND = "TSC_BPE_full_decoder_optimizer_state_v1"
IDENTITY_KIND = "TSC_BPE_full_decoder_optimizer_identity_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def partition_plan(names, vocabulary):
    """Pure metadata plan; source IDs preserve the actual original 15-head order."""
    require(vocabulary in (OLD_VOCAB, BPE_VOCAB), "Only original8199 or reviewed8711 geometry")
    names = tuple(names)
    heads = tuple(n for n in names if not n.endswith(("lora_A", "lora_B")))
    lora = tuple(n for n in names if n.endswith(("lora_A", "lora_B")))
    require(len(names) == len(set(names)) == 207 and len(heads) == 15 and len(lora) == 192
            and set(heads) == set(TABLES) | set(SHARED)
            and all(n.startswith("encoder.") for n in lora), "Exact original207 names required")
    mature, fresh = [], []
    for source_id, name in enumerate(heads):
        if name not in TABLES:
            mature.append(dict(name=name, part="whole", source_id=source_id,
                               source_rows=None, target_rows=None))
            continue
        mature.append(dict(name=name, part="lexical", source_id=source_id,
                           source_rows=[0, OLD_VOCAB], target_rows=[0, OLD_VOCAB]))
        if name != TABLES[0]:
            mature.append(dict(name=name, part="duration", source_id=source_id,
                               source_rows=[OLD_VOCAB, OLD_VOCAB + DURATIONS],
                               target_rows=[vocabulary, vocabulary + DURATIONS]))
        if vocabulary == BPE_VOCAB:
            fresh.append(dict(name=name, part="new512", source_id=None,
                              source_rows=None, target_rows=[OLD_VOCAB, BPE_VOCAB]))
    require(len(mature) == 17 and len(fresh) == (3 if vocabulary == BPE_VOCAB else 0),
            "Disjoint mature17/fresh3 partition changed")
    return heads, lora, tuple(mature), tuple(fresh)


class BPEFullDecoderOptimizer:
    """Two Adam histories over disjoint views, one global actual15-tensor clip."""
    def __init__(self, model, source_state, *, source_record, vocabulary,
                 shared_lr=1e-5, new_rows_lr=1e-4, six_token_gain=6., torch_module=None):
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.model = model
        self.vocabulary = vocabulary
        self.source_record = copy.deepcopy(source_record)
        require(set(source_record) >= {"path", "sha256", "bytes"}
                and len(source_record["sha256"]) == 64, "Immutable full-state source byte record required")
        require(shared_lr == 1e-5 and new_rows_lr == 1e-4 and six_token_gain == 6.,
                "Only declared mature1e-5/fresh1e-4/six-glyph gain6 policy")
        self.shared_lr, self.new_rows_lr = float(shared_lr), float(new_rows_lr)
        self.six_token_gain = float(six_token_gain)
        self.source_step = source_state.get("step")
        require(type(self.source_step) is int and self.source_step > 0
                and source_state.get("selected_step") == self.source_step,
                "Coherent strict-qualified selected source full state required")
        self.names = tuple(source_state["weights"])
        self.head_names, self.lora_names, self.mature_plan, self.fresh_plan = partition_plan(self.names, vocabulary)
        self.parameters = {n: p for n, p in model.named_parameters() if n in self.names}
        require(set(self.parameters) == set(self.names), "Every original source tensor must exist")
        self.heads = tuple(self.parameters[n] for n in self.head_names)
        self._validate_geometry(source_state)
        source_optimizer = source_state["optimizer"]
        groups = source_optimizer["param_groups"]
        require(len(groups) == 2 and groups[0]["params"] == list(range(15))
                and groups[1]["params"] == list(range(15, 207))
                and set(source_optimizer["state"]) == set(range(207)), "Original207 Adam ID order required")
        source_group = copy.deepcopy({k: v for k, v in groups[0].items() if k != "params"})
        require(source_group["lr"] == shared_lr and source_group["weight_decay"] == .01
                and tuple(source_group["betas"]) == (.9, .999) and source_group["eps"] == 1e-8
                and source_group["amsgrad"] is False, "Audit actual source head Adam recipe before transfer")
        self.parked_lora_group = copy.deepcopy({k: v for k, v in groups[1].items() if k != "params"})
        self._audit_source_moments(source_state)
        for n, p in self.parameters.items():
            p.requires_grad_(n in self.head_names)
            p.grad = None
        self.frozen_lora_sha256 = self._frozen_digest()
        self.mature = tuple(self._view(item) for item in self.mature_plan)
        self.fresh = tuple(self._view(item) for item in self.fresh_plan)
        self.mature_optimizer = self.torch.optim.AdamW([{"params": list(self.mature), **source_group}])
        target = self.mature_optimizer.state_dict()
        for target_id, item in zip(target["param_groups"][0]["params"], self.mature_plan):
            old = source_optimizer["state"][item["source_id"]]
            target["state"][target_id] = {
                "step": old["step"].detach().cpu().clone(),
                "exp_avg": self._slice(old["exp_avg"], item["source_rows"]).detach().clone(),
                "exp_avg_sq": self._slice(old["exp_avg_sq"], item["source_rows"]).detach().clone()}
        self.mature_optimizer.load_state_dict(target)
        for parameter, item in zip(self.mature, self.mature_plan):
            restored = self.mature_optimizer.state[parameter]
            original = source_optimizer["state"][item["source_id"]]
            require(restored["step"].item() == self.source_step
                    and all(self._bits_equal(restored[key], self._slice(original[key], item["source_rows"]))
                            for key in ("exp_avg", "exp_avg_sq")),
                    "Actual mature named Adam slice transfer is not byte exact")
        self.mature_optimizer.param_groups[0]["lr"] = self.shared_lr
        if "initial_lr" in self.mature_optimizer.param_groups[0]:
            self.mature_optimizer.param_groups[0]["initial_lr"] = self.shared_lr
        fresh_group = copy.deepcopy(source_group)
        fresh_group["lr"] = self.new_rows_lr
        if "initial_lr" in fresh_group:
            fresh_group["initial_lr"] = self.new_rows_lr
        self.fresh_optimizer = (self.torch.optim.AdamW([{"params": list(self.fresh), **fresh_group}])
                                if self.fresh else None)
        self.mature_group = copy.deepcopy({k: v for k, v in self.mature_optimizer.param_groups[0].items() if k != "params"})
        self.fresh_group = (None if self.fresh_optimizer is None else
                            copy.deepcopy({k: v for k, v in self.fresh_optimizer.param_groups[0].items() if k != "params"}))
        self.updates = 0
        self._incomplete_update = False
        self._validate_live_clocks()
        # No source weights or mature/frozen CPU moment dictionary is retained.

    @staticmethod
    def _slice(value, rows):
        return value if rows is None else value[rows[0]:rows[1]]

    def _bits_equal(self, left, right):
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        return self.torch.equal(left.detach().cpu().contiguous().reshape(-1).view(self.torch.uint8),
                                right.detach().cpu().contiguous().reshape(-1).view(self.torch.uint8))

    def _view(self, item):
        parent = self.parameters[item["name"]]
        rows = item["target_rows"]
        if rows is None:
            return parent
        value = self._slice(parent.detach(), rows)
        result = self.torch.nn.Parameter(value, requires_grad=True)
        require(result.data_ptr() == value.data_ptr() and result.shape == value.shape
                and result.stride() == value.stride(), "Optimizer view must share exact disjoint model storage")
        return result

    def _validate_geometry(self, source):
        t = self.torch
        for n, old in source["weights"].items():
            actual = self.parameters[n]
            require(actual.dtype == old.dtype == t.float32, "FP32 source/model parameters required: " + n)
            if n not in TABLES:
                require(actual.shape == old.shape and self._bits_equal(actual, old),
                        "Unchanged source parameter differs: " + n)
                continue
            require(old.shape[0] == OLD_VOCAB + (0 if n == TABLES[0] else DURATIONS)
                    and actual.shape[0] == self.vocabulary + (0 if n == TABLES[0] else DURATIONS)
                    and actual.shape[1:] == old.shape[1:], "Token/duration table geometry differs: " + n)
            require(self._bits_equal(actual[:OLD_VOCAB], old[:OLD_VOCAB]),
                    "Original lexical/blank/six rows changed at entry: " + n)
            if n != TABLES[0]:
                require(self._bits_equal(actual[self.vocabulary:], old[OLD_VOCAB:]),
                        "Duration values were not copied to explicit new row indices: " + n)

    def _audit_source_moments(self, source):
        t = self.torch
        for identifier, name in enumerate((*self.head_names, *self.lora_names)):
            state = source["optimizer"]["state"][identifier]
            require(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "Standard Adam source state required")
            require(state["step"].ndim == 0 and state["step"].dtype == t.float32
                    and state["step"].item() == self.source_step, "Actual source Adam clock differs: " + name)
            for key in ("exp_avg", "exp_avg_sq"):
                value = state[key]
                require(value.shape == source["weights"][name].shape and value.dtype == t.float32
                        and bool(t.isfinite(value).all()), "Source moment shape/finiteness differs: " + name)
            require(bool((state["exp_avg_sq"] >= 0).all()), "Negative source second moment")

    def _frozen_digest(self):
        digest = hashlib.sha256()
        for name in self.lora_names:
            value = self.parameters[name].detach().cpu().contiguous()
            digest.update(name.encode()); digest.update(str(tuple(value.shape)).encode())
            digest.update(value.reshape(-1).view(self.torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _require_complete(self):
        require(not self._incomplete_update, "Partial optimizer update is poisoned; no save/retry")

    def zero_grad(self, set_to_none=True):
        self._require_complete()
        require(set_to_none is True, "Only explicit none-gradient reset")
        for value in (*self.heads, *self.mature, *self.fresh):
            value.grad = None

    def step(self, max_norm=1.):
        self._require_complete()
        require(max_norm == 1. and all(p.grad is not None for p in self.heads),
                "One complete actual15-tensor gradient and global clip1 required")
        norm = self.torch.nn.utils.clip_grad_norm_(self.heads, max_norm, error_if_nonfinite=True)
        for parameter, item in zip((*self.mature, *self.fresh), (*self.mature_plan, *self.fresh_plan)):
            gradient = self.parameters[item["name"]].grad
            parameter.grad = self._slice(gradient, item["target_rows"]).detach()
        before_six = tuple(self.parameters[n][8193:8199].detach().clone() for n in TABLES)
        self._incomplete_update = True
        self.mature_optimizer.step()
        if self.fresh_optimizer is not None:
            self.fresh_optimizer.step()
        with self.torch.no_grad():
            for name, before in zip(TABLES, before_six):
                rows = self.parameters[name][8193:8199]
                rows.copy_(before + self.six_token_gain * (rows - before))
        self.updates += 1
        self._validate_live_clocks()
        self._incomplete_update = False
        return norm

    def _validate_live_clocks(self, *, full=False):
        for optimizer, parameters, clock in ((self.mature_optimizer, self.mature, self.source_step + self.updates),
                                             (self.fresh_optimizer, self.fresh, self.updates)):
            if optimizer is None:
                continue
            for parameter in parameters:
                state = optimizer.state.get(parameter, {})
                if clock == 0:
                    require(not state, "Fresh clock0 must have no fabricated mature moments")
                    continue
                require(set(state) == {"step", "exp_avg", "exp_avg_sq"}
                        and state["step"].ndim == 0 and state["step"].dtype == self.torch.float32
                        and state["step"].item() == clock
                        and state["exp_avg"].shape == state["exp_avg_sq"].shape == parameter.shape,
                        "Active Adam moment partition/clock differs")
                if full:
                    require(state["exp_avg"].dtype == state["exp_avg_sq"].dtype == self.torch.float32
                            and bool(self.torch.isfinite(state["exp_avg"]).all())
                            and bool(self.torch.isfinite(state["exp_avg_sq"]).all())
                            and bool((state["exp_avg_sq"] >= 0).all()),
                            "Active Adam moments must remain finite FP32/nonnegative")

    def phase_identity(self):
        return {"kind": IDENTITY_KIND, "source_record": self.source_record,
                "source_step": self.source_step, "source_weight_names": list(self.names),
                "vocabulary": self.vocabulary, "blank_id": BLANK,
                "old_lexical_rows": [0, OLD_VOCAB], "new_rows": ([OLD_VOCAB, BPE_VOCAB] if self.fresh else []),
                "duration_rows": [self.vocabulary, self.vocabulary + DURATIONS],
                "active_model_tensor_names": list(self.head_names), "active_model_tensors": 15,
                "mature_optimizer_partitions": list(self.mature_plan), "fresh_optimizer_partitions": list(self.fresh_plan),
                "mature_learning_rate": self.shared_lr,
                "fresh_learning_rate": self.new_rows_lr if self.fresh else None,
                "mature_optimizer_group": self.mature_group, "fresh_optimizer_group": self.fresh_group,
                "weight_decay": .01, "global_clip_actual_model_tensors_once": 1.,
                "six_glyph_post_Adam_gain": self.six_token_gain, "gain_applies_only_rows": [8193, 8199],
                "frozen_encoder_LoRA_names": list(self.lora_names), "frozen_encoder_LoRA_sha256": self.frozen_lora_sha256,
                "parked_encoder_Adam_group": self.parked_lora_group, "parked_encoder_Adam_clock": self.source_step,
                "source_moments_or_clock_reset": False,
                "fresh_row_clock_starts_at": 0 if self.fresh else None}

    def state_dict(self):
        self._require_complete(); self._validate_live_clocks(full=True)
        require(self._frozen_digest() == self.frozen_lora_sha256, "Frozen192 source weights changed")
        return {"kind": KIND, "identity": self.phase_identity(), "updates": self.updates,
                "mature_Adam_clock": self.source_step + self.updates,
                "new_row_Adam_clock": self.updates if self.fresh else None,
                "parked_encoder_Adam_clock": self.source_step,
                "mature_optimizer": self.mature_optimizer.state_dict(),
                "fresh_optimizer": None if self.fresh_optimizer is None else self.fresh_optimizer.state_dict()}

    def load_state_dict(self, state):
        self._require_complete()
        require(state.get("kind") == KIND and state.get("identity") == self.phase_identity(),
                "Own exact optimizer identity differs from immutable phase admission")
        count = state.get("updates")
        require(type(count) is int and 0 <= count <= 2000
                and state.get("mature_Adam_clock") == self.source_step + count
                and state.get("new_row_Adam_clock") == (count if self.fresh else None)
                and state.get("parked_encoder_Adam_clock") == self.source_step,
                "Own exact phase/mature/new/frozen clocks differ")
        require((state.get("fresh_optimizer") is None) == (self.fresh_optimizer is None),
                "Own exact BPE/control optimizer partition differs")
        for saved, expected, parameters in ((state["mature_optimizer"], self.mature_group, self.mature),
                                            (state["fresh_optimizer"], self.fresh_group, self.fresh)):
            if saved is None:
                continue
            groups = saved.get("param_groups", [])
            require(len(groups) == 1 and {k: v for k, v in groups[0].items() if k != "params"} == expected,
                    "Own exact Adam group hyperparameters differ")
            require(groups[0]["params"] == list(range(len(parameters)))
                    and set(saved.get("state", {})) == (set(range(len(parameters))) if count else
                         (set(range(len(parameters))) if saved is state["mature_optimizer"] else set())),
                    "Own exact named Adam partition IDs/order differ")
        self._incomplete_update = True
        self.mature_optimizer.load_state_dict(state["mature_optimizer"])
        if self.fresh_optimizer is not None:
            self.fresh_optimizer.load_state_dict(state["fresh_optimizer"])
        self.updates = count
        require(self.mature_optimizer.param_groups[0]["lr"] == self.shared_lr
                and (self.fresh_optimizer is None or self.fresh_optimizer.param_groups[0]["lr"] == self.new_rows_lr),
                "Own exact live learning rates changed")
        self._validate_live_clocks(full=True)
        require(self._frozen_digest() == self.frozen_lora_sha256, "Own exact frozen192 source weights differ")
        self._incomplete_update = False
