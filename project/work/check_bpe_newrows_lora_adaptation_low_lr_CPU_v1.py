"""Tiny synthetic LOW-LR phase2 fixture; stdlib prepare mode executes no Torch.

ROOT must wait until the actual GPU parent is CLOSED before choosing
--cpu-synthetic --parent-closed-confirmed. No real checkpoint/corpus/ASR is read.
"""
from __future__ import annotations
import argparse
import ast
import copy
import ctypes
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "outputs/parakeet_turkish"
UTILITY = DIRECTORY / "bpe_newrows_lora_adaptation_low_lr_v1.py"
ORIGINAL_UTILITY = DIRECTORY / "bpe_newrows_lora_adaptation_v1.py"
EXPECTED_UTILITY_SHA = "f32130ae2d621a806bd7dd9757934f65449fe5dfdc6197f475452b389ce57611"
EXPECTED_ORIGINAL_SHA = "2f70811d67af5fc950fa547727dc5ae18fa895a1e5e2cb572c4ff40bbc42476b"


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--cpu-synthetic", action="store_true")
    parser.add_argument("--parent-closed-confirmed", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    if cli.output.exists(): raise ValueError("Fresh receipt required")
    if hashlib.sha256(UTILITY.read_bytes()).hexdigest() != EXPECTED_UTILITY_SHA or hashlib.sha256(ORIGINAL_UTILITY.read_bytes()).hexdigest() != EXPECTED_ORIGINAL_SHA:
        raise ValueError("Frozen utility pins changed")
    started = time.monotonic()
    spec = importlib.util.spec_from_file_location("phase2_rows_fixture", UTILITY)
    utility = importlib.util.module_from_spec(spec); spec.loader.exec_module(utility)
    tree = ast.parse(UTILITY.read_text(encoding="utf-8")); compile(tree, str(UTILITY), "exec")
    checks, rejected = [], []
    def check(ok, name):
        if not ok: raise AssertionError(name)
        checks.append(name)
    adapters = [f"encoder.layers.{i}.self_attn.{projection}.lora_{suffix}"
        for i in range(24) for projection in ("q_proj", "k_proj", "v_proj", "o_proj") for suffix in ("A", "B")]
    heads = [*utility.TABLES, *(f"fixed_shared.{i}" for i in range(12))]
    groups = [{"params": list(range(15))}, {"params": list(range(15, 207))}]
    original_names = adapters + heads
    mapping = utility.source_lora_mapping(original_names, groups)
    check(mapping[0] == tuple(heads) and mapping[1] == tuple(adapters), "actual_pure_mapping_preserves_named_group_order")
    check(mapping[2] == tuple(zip(range(15, 207), adapters)), "actual_pure_mapping_source15_to206_LoRA_only")
    for label, names, bad_groups in (("wrong_source_ID_order", original_names, [groups[0], {"params": list(reversed(groups[1]["params"]))}]),
        ("duplicate_name", original_names[:-1] + [original_names[0]], groups)):
        try: utility.source_lora_mapping(names, bad_groups)
        except ValueError: rejected.append(label)
        else: raise AssertionError(label)
    methods = {node.name: node for node in next(node for node in tree.body if isinstance(node, ast.ClassDef)).body if isinstance(node, ast.FunctionDef)}
    step = ast.unparse(methods["step"])
    check(step.index("grad[:8199].zero_") < step.index("clip_grad_norm_") < step.index("row_optimizer.step") < step.index("lora_optimizer.step"),
          "source_AST_mask_single_global_clip_then_two_steps")
    check("self.tables + self.adapters" in step, "source_AST_global195_clip_includes_both_groups")
    check("load_phase_state" in methods and "load_state_dict" in methods and "state_dict" in methods,
          "distinct_phase2_utility_save_and_restore_API_exists")
    class FakeGrad:
        is_sparse = False
        shape = (1,)
        def __getitem__(self, item): return self
        def zero_(self): pass
    def fake_stage(fail_second=False):
        stage = utility.BPENewRowsLoRAAdaptation.__new__(utility.BPENewRowsLoRAAdaptation)
        stage.source_step = 100; stage.acquisition_updates = 1; stage.phase2_updates = 0
        stage._incomplete_update = False; stage._check_contract = lambda: None
        stage.tables = tuple(SimpleNamespace(grad=FakeGrad(), shape=(1,)) for _ in range(3))
        stage.adapters = tuple(SimpleNamespace(grad=FakeGrad(), shape=(1,)) for _ in range(192))
        calls = []
        def row_step(): calls.append("rows")
        def lora_step():
            calls.append("LoRA")
            if fail_second: raise RuntimeError("Explicit mocked second-Adam failure")
        def clip(parameters, max_norm, **kwargs):
            calls.append(("clip", len(parameters), max_norm, kwargs["error_if_nonfinite"]))
            return "explicitly_mocked_norm"
        stage.row_optimizer = SimpleNamespace(step=row_step, zero_grad=lambda **kw: None)
        stage.lora_optimizer = SimpleNamespace(step=lora_step, zero_grad=lambda **kw: None)
        stage.torch = SimpleNamespace(nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=clip)))
        return stage, calls
    completed, calls = fake_stage()
    check(completed.step() == "explicitly_mocked_norm" and completed.phase2_updates == 1
        and completed.cumulative_step == 102 and not completed._incomplete_update
        and calls == [("clip", 195, 1., True), "rows", "LoRA"],
        "actual_step_method_with_explicit_stdlib_stubs_commits_after_both_calls")
    poisoned, calls = fake_stage(fail_second=True)
    try: poisoned.step()
    except RuntimeError: pass
    else: raise AssertionError("second_step_failure")
    check(poisoned._incomplete_update and poisoned.phase2_updates == 0 and calls[-2:] == ["rows", "LoRA"],
        "actual_step_method_mocked_second_failure_leaves_poison_and_count_unadvanced")
    for name in ("step", "zero_grad", "state_dict"):
        try: getattr(poisoned, name)()
        except ValueError: rejected.append("poisoned_" + name)
        else: raise AssertionError("poisoned_" + name)
    check(len([name for name in rejected if name.startswith("poisoned_")]) == 3,
          "partial_update_cannot_step_zero_or_serialize")
    load = ast.unparse(methods["load_phase_state"])
    check(load.index("_validate_optimizer_state") < load.index("self._incomplete_update = True")
        < load.index("copy_(value)") < load.index("self.lora_optimizer.load_state_dict")
        < load.index("self._incomplete_update = False"), "source_AST_full_validation_then_both_restore_before_poison_clear")
    save = ast.unparse(methods["state_dict"])
    check(save.index("_require_complete_update") < save.index("_cpu_clone")
        and save.count("_validate_optimizer_state") == 2 and "isfinite" in save,
        "source_AST_durable_save_checks_complete_finite_weights_and_both_Adam_histories")
    check("_check_moments" not in step and "_bytes" not in step and "cpu(" not in step,
          "source_AST_no_per_update_full_moment_or_rawbyte_CPU_scan")
    constructor = ast.unparse(methods["__init__"])
    check(constructor.index("self._audit_original_lora") < constructor.index("self.lora_optimizer.load_state_dict(target)")
        < constructor.index("self.lora_optimizer.param_groups[0]['lr'] = 3e-05"),
        "lowLR_source_audit_and_restored_moments_precede_activeLR_change")
    check("self.row_group['lr'] == 0.0001" in ast.unparse(methods["_check_contract"])
        and "self.lora_group['lr'] == 3e-05" in ast.unparse(methods["_check_contract"]),
        "declared_row1e4_lowLoRa3e5_contract")
    check("torch" not in sys.modules, "prepare_and_pure_mapping_import_no_ML")
    if cli.prepare_only:
        result = {"status": "stdlib_prepared_CPU_synthetic_deferred", "actual_CPU_tensor_checks_executed": False,
                  "reason": "ROOT requires actual parent CLOSED before Torch import; current training has priority"}
    else:
        if not cli.parent_closed_confirmed: raise ValueError("ROOT must confirm actual parent CLOSED before this tiny CPU fixture")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        priority = {"requested": "BELOW_NORMAL_PRIORITY_CLASS", "platform": os.name}
        if os.name == "nt":
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = ctypes.c_void_p
            kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]; kernel.SetPriorityClass.restype = ctypes.c_int
            priority["applied"] = bool(kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000))
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(137)
        sys.path.insert(0, str(DIRECTORY))
        from bpe_newrows_acquisition_v1 import BPEOnlyAcquisition, artifact_record
        class Projection(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = torch.nn.Parameter(torch.full((1, 2), .01))
                self.lora_B = torch.nn.Parameter(torch.full((2, 1), .02))
            def forward(self, x): return x + (x @ self.lora_A.T @ self.lora_B.T) * .001
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.layers = torch.nn.ModuleList()
                for _ in range(24):
                    layer = torch.nn.Module(); layer.self_attn = torch.nn.Module()
                    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"): setattr(layer.self_attn, projection, Projection())
                    self.layers.append(layer)
            def forward(self, value):
                for layer in self.layers:
                    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"): value = getattr(layer.self_attn, projection)(value)
                return value
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.encoder = Encoder(); self.decoder = torch.nn.Module()
                self.decoder.embedding = torch.nn.Embedding(8711, 2)
                self.decoder.shared = torch.nn.Linear(2, 2)
                self.decoder.fixed = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([float(i)])) for i in range(10)])
                self.joint = torch.nn.Module(); self.joint.head = torch.nn.Linear(2, 8716)
                self.config = type("Geometry", (), {"vocab_size": 8711, "blank_token_id": 8192, "durations": [0,1,2,3,4]})()
                with torch.no_grad(): self.joint.head.bias[3] = -0.; self.joint.head.bias[8712] = -0.
        def raw(value): return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        def tensor_digest(weights):
            digest = hashlib.sha256()
            for name, value in weights.items(): digest.update(name.encode()); digest.update(raw(value))
            return digest.hexdigest()
        def backward(model, stage):
            stage.zero_grad()
            encoded = model.encoder(torch.ones(4, 2))
            hidden = model.decoder.shared(encoded + model.decoder.embedding(torch.tensor([4, 8195, 8199, 8710])))
            loss = model.joint.head(hidden).square().sum() + hidden.square().sum()
            loss.backward()
        with tempfile.TemporaryDirectory(prefix="phase2_rows_cpu_", dir=ROOT / "work") as temp:
            source_file = Path(temp) / "opaque_source.bin"; source_file.write_bytes(b"Synthetic original207 artifact; not a real checkpoint")
            phase_file = Path(temp) / "opaque_phase1.bin"; phase_file.write_bytes(b"Synthetic phase1 artifact; not a real checkpoint")
            source_record, phase_record = artifact_record(source_file), artifact_record(phase_file)
            model = Model(); names = tuple(name for name, _ in model.named_parameters()); parameters = dict(model.named_parameters())
            check(len(names) == 207, "tiny_model207_names_192_tiny_LoRA_12_fixed_heads_3_tables")
            original = {name: value.detach().clone() for name, value in parameters.items()}
            for name in utility.TABLES:
                value = original[name]
                original[name] = value[:8199].clone() if name == utility.TABLES[0] else torch.cat((value[:8199], value[8711:])).clone()
            head_names = [name for name in names if not name.endswith(("lora_A", "lora_B"))]
            adapter_names = [name for name in names if name.endswith(("lora_A", "lora_B"))]
            parked = torch.optim.AdamW([{"params": [parameters[n] for n in head_names], "lr": 1e-5, "weight_decay": .01},
                                       {"params": [parameters[n] for n in adapter_names], "lr": 3e-4, "weight_decay": 0.}]).state_dict()
            source_step = 100
            for identifier, name in enumerate(head_names + adapter_names):
                parked["state"][identifier] = {"step": torch.tensor(float(source_step)),
                    "exp_avg": torch.full_like(original[name], .01), "exp_avg_sq": torch.full_like(original[name], .02)}
            metadata = {"scope": "synthetic tiny LoRA shapes; not actual rank8/ASR geometry"}
            source = {"schema": 2, "step": source_step, "selected_step": source_step, "weights": original,
                      "optimizer": parked, "lora_metadata": metadata}
            source_clock_before = [state["step"].item() for state in parked["state"].values()]
            phase1 = BPEOnlyAcquisition(model, names, source_record, source_lora_metadata=metadata, torch_module=torch)
            backward(model, phase1); phase1.step()
            phase1_state = phase1.state_dict(runtime_state={"source_selected_step": source_step, "step": source_step + 1, "opaque": "fixture"})
            stage = utility.BPENewRowsLoRAAdaptation(model, phase1_state, source,
                original_source_record=source_record, phase1_record=phase_record, torch_module=torch)
            check(stage.cumulative_step == 101 and stage.phase2_updates == 0, "S100_K1_N0_cumulative101")
            check(parked["param_groups"][1]["lr"] == 3e-4 and stage.lora_group["lr"] == 3e-5
                and stage.row_group["lr"] == 1e-4, "source3e4_active3e5_row1e4_distinct_exact_groups")
            for target_id, name in enumerate(adapter_names):
                copied = stage.lora_optimizer.state[parameters[name]]
                donor = parked["state"][15 + target_id]
                check(all(raw(copied[key]) == raw(donor[key]) for key in ("step", "exp_avg", "exp_avg_sq")),
                      "copied_LoRA_moments_clock_exact_" + str(target_id))
            parked_before = {identifier: {key: raw(value) for key,value in state.items()}
                for identifier,state in parked["state"].items()}
            # A matched ordinary3e-4 comparison has identical weights,3-table K
            # and LoRA S moments. Both use exactly the same195-gradient objective.
            ordinary_spec = importlib.util.spec_from_file_location("ordinary_phase2_lr_reference", ORIGINAL_UTILITY)
            ordinary_utility = importlib.util.module_from_spec(ordinary_spec); ordinary_spec.loader.exec_module(ordinary_utility)
            ordinary_model = copy.deepcopy(model)
            ordinary_stage = ordinary_utility.BPENewRowsLoRAAdaptation(ordinary_model, phase1_state, source,
                original_source_record=source_record, phase1_record=phase_record, torch_module=torch)
            ordinary_parameters = dict(ordinary_model.named_parameters())
            before_values = {name: value.detach().clone() for name,value in model.named_parameters()}
            check(tensor_digest(ordinary_parameters) == tensor_digest(parameters), "matched_LR_models_identical_before_step")
            before = {name: raw(value) for name, value in model.named_parameters()}
            fixed_before = stage.fixed_weight_sha256
            backward(model, stage)
            check(all(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in stage.adapters), "tiny_encoder_LoRA_gradients_exist")
            check(bool(torch.count_nonzero(stage.tables[1].grad[8711:])), "protected_duration_gradient_exists_before_mask")
            backward(ordinary_model, ordinary_stage)
            check(all(torch.equal(parameters[name].grad, ordinary_parameters[name].grad)
                for name in (*utility.TABLES, *adapter_names)), "same_all195_actual_gradients_before_mask_clip_LR_steps")
            expected = torch.linalg.vector_norm(torch.cat([p.grad[8199:8711].flatten() for p in stage.tables] + [p.grad.flatten() for p in stage.adapters]))
            norm = stage.step()
            ordinary_norm = ordinary_stage.step()
            check(torch.equal(norm, ordinary_norm), "same_single195_global_gradient_clip_norm")
            check(all(raw(parameters[name]) == raw(ordinary_parameters[name]) for name in utility.TABLES),
                  "row1e4_actual_deltas_exact_across_LoRA_LR_challengers")
            low_delta = torch.cat([(parameters[name].detach() - before_values[name]).flatten() for name in adapter_names])
            ordinary_delta = torch.cat([(ordinary_parameters[name].detach() - before_values[name]).flatten() for name in adapter_names])
            check(bool(torch.count_nonzero(low_delta)) and bool(torch.count_nonzero(ordinary_delta)), "both_LoRA_actual_deltas_nonzero")
            # FP32 parameter addition rounds independently; the requested tenfold
            # step comparison uses2 ULPs of starting magnitude, not byte equality.
            delta_tolerance = 2 * torch.finfo(torch.float32).eps * max(float(before_values[name].abs().max()) for name in adapter_names)
            check(torch.allclose(low_delta, ordinary_delta * .1, rtol=.005, atol=delta_tolerance),
                  "actual_LoRA_parameter_delta_one_tenth_FP32_rounding_bounded")
            delta_ratio = float(torch.linalg.vector_norm(low_delta) / torch.linalg.vector_norm(ordinary_delta))
            check(abs(delta_ratio - .1) < .001, "actual_LoRA_delta_vector_norm_ratio_about0point1")
            check(all(all(raw(stage.lora_optimizer.state[parameters[name]][key]) ==
                raw(ordinary_stage.lora_optimizer.state[ordinary_parameters[name]][key])
                for key in ("step", "exp_avg", "exp_avg_sq")) for name in adapter_names),
                "LoRA192_actual_updated_moments_clocks_exact_despite_onlyLR_change")
            check({identifier: {key: raw(value) for key,value in state.items()} for identifier,state in parked["state"].items()}
                == parked_before and parked["param_groups"][1]["lr"] == 3e-4,
                "all_source207_Adam_moment_bytes_clocks_groupLR_unchanged_after_both_steps")
            check(torch.allclose(norm, expected, rtol=1e-6, atol=1e-6), "single195_norm_is_formed_after_mask")
            check(all(float(stage.row_optimizer.state[p]["step"]) == 2 for p in stage.tables)
                and all(float(stage.lora_optimizer.state[p]["step"]) == 101 for p in stage.adapters)
                and stage.cumulative_step == 102, "independent_Adam_KplusN_SplusN_and_global_clock")
            check(all(raw(parameters[n]) != before[n] for n in adapter_names), "192_tiny_LoRA_actual_parameter_deltas")
            check(stage._fixed_digest(parameters) == fixed_before, "12_shared_heads_old_six_blank_duration_rawbytes_unchanged")
            check([state["step"].item() for state in parked["state"].values()] == source_clock_before,
                  "source207_scalar_clocks_not_aliased_or_incremented")
            saved = stage.state_dict(runtime_state={"opaque": "sampler_rng_history_caller_owned", "cumulative_step": 102})
            check(len(saved["weights"]) == 207 and len(saved["row_optimizer"]["state"]) == 3
                and len(saved["lora_optimizer"]["state"]) == 192 and "optimizer" not in saved, "all207_two_disjoint_Adam_serializer")
            stream = io.BytesIO(); torch.save(saved, stream); stream.seek(0); restored = torch.load(stream, weights_only=True)
            with torch.no_grad(): stage.adapters[0].add_(1.)
            runtime = stage.load_state_dict(restored)
            check(tensor_digest(dict(model.named_parameters())) == tensor_digest(saved["weights"])
                and runtime == saved["runtime_state"], "own_phase2_serialized207_weights_and_opaque_runtime_restore")
            backward(model, stage)
            original_lora_step = stage.lora_optimizer.step
            def fail_lora_step(*args, **kwargs): raise RuntimeError("Synthetic failure after real row-Adam update")
            stage.lora_optimizer.step = fail_lora_step
            try: stage.step()
            except RuntimeError: pass
            else: raise AssertionError("partial_actual_CPU_update")
            finally: stage.lora_optimizer.step = original_lora_step
            check(stage._incomplete_update, "actual_CPU_partial_two_Adam_update_poisoned")
            try: stage.state_dict()
            except ValueError: rejected.append("partial_actual_CPU_state_save")
            else: raise AssertionError("partial_actual_CPU_state_save")
            stage.load_state_dict(saved)
            check(not stage._incomplete_update and tensor_digest(dict(model.named_parameters())) == tensor_digest(saved["weights"])
                and all(float(stage.row_optimizer.state[p]["step"]) == 2 for p in stage.tables)
                and all(float(stage.lora_optimizer.state[p]["step"]) == 101 for p in stage.adapters),
                "fully_validated_own_state_recovers_actual_CPU_partial_weights_and_both_clocks")
            for label, mutate in (("wrong_LoRA_clock", lambda state: state["lora_optimizer"]["state"][0]["step"].add_(1)),
                ("wrong_row_clock", lambda state: state["row_optimizer"]["state"][0]["step"].add_(1)),
                ("negative_variance", lambda state: state["lora_optimizer"]["state"][0]["exp_avg_sq"].fill_(-1)),
                ("missing_LoRA_state", lambda state: state["lora_optimizer"]["state"].pop(191)),
                ("protected_old_weight", lambda state: state["weights"][utility.TABLES[2]][0].add_(1)),
                ("protected_row_moment", lambda state: state["row_optimizer"]["state"][0]["exp_avg"][0].add_(1)),
                ("changed_LoRA_LR", lambda state: state["lora_optimizer"]["param_groups"][0].update(lr=1e-4)),
                ("ordinary_LoRA_LR_in_low_phase", lambda state: state["lora_optimizer"]["param_groups"][0].update(lr=3e-4)),
                ("incomplete_serialized_boundary", lambda state: state.update(complete_two_optimizer_update_boundary=False))):
                damaged = copy.deepcopy(saved); mutate(damaged); before_reject = tensor_digest(dict(model.named_parameters()))
                try: stage.load_state_dict(damaged)
                except (ValueError, KeyError, TypeError):
                    check(tensor_digest(dict(model.named_parameters())) == before_reject, label + "_reject_before_weight_copy")
                    rejected.append(label)
                else: raise AssertionError(label)
            check(not torch.cuda.is_initialized(), "CUDA_never_initialized")
            check(artifact_record(source_file) == source_record and artifact_record(phase_file) == phase_record, "opaque_artifact_files_unchanged")
            result = {"status": "tiny_CPU_synthetic_complete", "actual_CPU_tensor_checks_executed": True,
                      "priority": priority, "torch_threads": torch.get_num_threads(), "CUDA_initialized": torch.cuda.is_initialized(),
                      "source_LoRA_lr": 3e-4, "active_LoRA_lr": 3e-5, "row_lr": 1e-4,
                      "actual_LoRA_delta_norm_low_over_ordinary": delta_ratio,
                      "delta_comparison_FP32_tolerance": delta_tolerance,
                      "synthetic_source_clock": source_step, "synthetic_acquisition_clock": 1}
    report = {"schema": 1, "kind": "BPE_newrows_LoRA_low_lr_actual_CPU_synthetic_verification_v1", **result,
        "passed": True, "utility_sha256": hashlib.sha256(UTILITY.read_bytes()).hexdigest(),
        "original_utility_sha256": EXPECTED_ORIGINAL_SHA,
        "derived_original_fixture_sha256": "5ea134c53a821370b924b7a6740a9e362aa2e3b002b017b152033ffe0e144a69",
        "fixture_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "checks": checks,
        "check_count": len(checks), "rejected": rejected, "elapsed_seconds": time.monotonic() - started,
        "actual_ASR_source_checkpoint_or_GPU_test": False, "actual_CLOSED_admission_or_activation": False,
        "external_worker_resume_supported": False}
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    with cli.output.open("x", encoding="utf-8") as stream: json.dump(report, stream, indent=2); stream.write("\n")
    print(json.dumps({"status": report["status"], "checks": len(checks), "rejections": len(rejected)}))


if __name__ == "__main__": main()
