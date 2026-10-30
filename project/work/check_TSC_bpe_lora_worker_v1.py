"""Focused inactive phase2 worker checks: stdlib/AST and explicit serializer stubs.

No whole worker import, ML, tensors, checkpoints, corpus, GPU or watched fake runs.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "outputs/parakeet_turkish"
WORKER = HERE / "train_lora_batched_tsc_bpe_lora_v1.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    if cli.output.exists(): raise ValueError("Fresh proof output required")
    text = WORKER.read_text(encoding="utf-8")
    tree = ast.parse(text); compile(tree, str(WORKER), "exec")
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    checks, rejected = [], []
    def check(condition, name):
        if not condition: raise AssertionError(name)
        checks.append(name)
    source = ast.parse((HERE / "train_lora_batched_tsc_headlr_v1.py").read_text(encoding="utf-8"))
    source_parser = copy.deepcopy(next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "parser"))
    source_parser.name = "source_parser"
    with tempfile.TemporaryDirectory(prefix="phase2_worker_stdlib_", dir=ROOT / "work") as directory:
        temp = Path(directory)
        namespace = {"argparse": argparse, "Path": Path, "ROOT": temp, "DATA": temp, "OUTPUT": temp,
                     "METADATA_PATH": temp / "metadata.jsonl", "REPORT_PATH": temp / "metadata_proof.json",
                     "math": math, "random": random, "__doc__": ast.get_docstring(tree)}
        selected = [source_parser, *(functions[name] for name in ("require", "parser", "validate_arguments", "save_training_state"))]
        exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(WORKER), "exec"), namespace)
        observed = []
        namespace["adaptation_preflight"] = lambda args: observed.append("explicit_mock_external_source_admission")
        namespace["native_loss_identity"] = lambda args: observed.append("explicit_mock_native_identity")
        argv = ["--tsc-input-proof", str(temp / "input.json"), "--tsc-training-approval", str(temp / "approval.json"),
                "--source-closure", str(temp / "original_closure.json"), "--run-id", "fixture_only_not_watched",
                "--replay-seed", "913", "--resume-adaptation", str(temp / "opaque_phase1.pt"),
                "--adaptation-source-receipt", str(temp / "future_closed_receipt.json"), "--eval-every-updates", "250"]
        args = namespace["parser"]().parse_args(argv)
        check(args.adaptation_max_updates == 1000 and args.acquisition_lr == 1e-4
              and args.native_max_joint_bytes == 1850000000, "actual_parser_default1000_constant_rowLR_and_widened_bound")
        check(namespace["validate_arguments"](args) is args and len(observed) == 2,
              "actual_validator_calls_explicit_mock_external_admission_and_native_identity")
        for label, change in (("external_exact_not_wired", {"resume_exact": temp / "not_read.pt"}),
                              ("shallow_old_optimizer", {"resume_optimization": temp / "not_read.pt"}),
                              ("encoder_checkpoint_disabled", {"no_encoder_checkpoint": True}),
                              ("changed_LoRA_LR", {"lora_lr": 1e-4}), ("changed_row_LR", {"acquisition_lr": 2e-4}),
                              ("physical_B4", {"batch_size": 4}), ("old_joint_width_bound", {"native_max_joint_bytes": 1700000000}),
                              ("zero_phase_limit", {"adaptation_max_updates": 0})):
            damaged = copy.deepcopy(args)
            for key, value in change.items(): setattr(damaged, key, value)
            try: namespace["validate_arguments"](damaged)
            except ValueError: rejected.append(label)
            else: raise AssertionError(label)
        check(len(rejected) == 8, "meaningful_recipe_and_unimplemented_resume_rejections")
        run = ast.unparse(functions["run"])
        check("batch_objective" not in functions and any(isinstance(node, ast.ImportFrom)
            and node.module == "batched_lora_tr_metadata_v1" and any(alias.name == "batch_objective" for alias in node.names)
            for node in tree.body), "original_differentiable_encoder_native_objective_imported_without_no_grad_wrapper")
        check(run.count("checkpoint_encoder_blocks(model)") == 1 and "model.encoder.eval()" in run,
              "original_all_block_checkpointing_and_eval_statistics_preserved")
        check("prepare_sampler_state" not in run and "sampler.load_state_dict(runtime['sampler'])" in run,
              "direct_chosen_candidate_sampler_frontier_no_second_target_migration")
        check(run.index("initial_source_equivalence(") < run.index("adaptation.step("),
              "actual300_source_equivalence_before_any_phase2_update")
        check("best, selected_step = (chosen_gate['turkish_wer'], start_step)" in run,
              "phase_initial_BEST_uses_actual_chosen_gate_not_prior_BEST_score")
        check("gate['retention_gate_passed'] and gate['turkish_wer'] < best" in run,
              "strict_five_retention_gates_still_filter_trained_BEST")
        phase1 = ast.parse((HERE / "train_lora_batched_tsc_bpe_acquire_v1.py").read_text(encoding="utf-8"))
        phase1_bound = next(node for node in phase1.body if isinstance(node, ast.FunctionDef) and node.name == "native_loss_function")
        check(ast.dump(functions["native_loss_function"], include_attributes=False) == ast.dump(phase1_bound, include_attributes=False),
              "exact_phase1_all_TR_and_original_replay_joint_bound_function_retained")
        captured = {}
        names = [f"opaque_learned_{index}" for index in range(207)]
        class FakeAdaptation:
            source_step = 100; acquisition_updates = 3; phase2_updates = 2; cumulative_step = 105
            def state_dict(self, *, runtime_state):
                captured["runtime"] = runtime_state
                return {"schema": 1, "kind": "explicit_mock_own_phase2_state", "weights": dict.fromkeys(names, "opaque"),
                        "row_optimizer": {"state": dict.fromkeys(range(3), "opaque")},
                        "lora_optimizer": {"state": dict.fromkeys(range(192), "opaque")}, "runtime_state": runtime_state}
        def fake_save(state, path):
            captured["state"] = state; Path(path).write_bytes(b"EXPLICIT MOCK SERIALIZATION; NOT A CHECKPOINT")
        namespace["torch"] = SimpleNamespace(save=fake_save, get_rng_state=lambda: "explicit_CPU_RNG_stub",
            cuda=SimpleNamespace(get_rng_state_all=lambda: ["explicit_CUDA_RNG_stub"]))
        save_args = {"run_id": "fixture_only_not_watched", "step": 105, "config_identity": {"opaque": True},
            "history": [{"step": index} for index in range(1,106)], "best_score": .4, "selected_step": 103,
            "patience": 1, "retention_failures": 0, "next_eval_epoch": .2, "train_wall_seconds": 3.,
            "audio_seconds": 4., "language_audio_seconds": {"tr": 2.}, "evaluation_wall_seconds": 1.,
            "initial_gate": {"inherited": True}, "source_selected_step": 103}
        target = temp / "opaque_save.bin"
        sampler = SimpleNamespace(updates=105, state_dict=lambda: {"opaque_frontier": 105})
        namespace["save_training_state"](target, None, FakeAdaptation(), sampler, **save_args)
        state = captured["state"]; runtime = captured["runtime"]
        check(target.exists() and not Path(str(target) + ".tmp").exists(), "actual_serializer_temporary_atomic_replace_with_explicit_stub_bytes")
        check(len(state["weights"]) == 207 and len(state["row_optimizer"]["state"]) == 3
              and len(state["lora_optimizer"]["state"]) == 192 and "optimizer" not in state,
              "actual_worker_save_adapter_preserves_207_and_distinct3_192_serializer_contract")
        check(runtime["source_phase1_step"] == 103 and runtime["original_source_selected_step"] == 100
              and runtime["selected_acquisition_updates"] == 3 and runtime["adaptation_updates_completed"] == 2
              and state["step"] == 105, "worker_serializer_separates_originalS_acquisitionK_localN_and_global")
        damaged = dict(save_args, step=104)
        try: namespace["save_training_state"](temp / "must_not_exist.bin", None, FakeAdaptation(), sampler, **damaged)
        except ValueError: rejected.append("inconsistent_durable_frontier")
        else: raise AssertionError("inconsistent_durable_frontier")
        check(not (temp / "must_not_exist.bin").exists(), "wrong_global_frontier_rejected_before_any_serializer_write")
    check("torch" not in sys.modules, "no_ML_import_or_CPU_tensor_execution")
    sync = [node.lineno for node in ast.walk(functions["run"]) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and ast.unparse(node.func) == "torch.cuda.synchronize"]
    check(len(sync) == 1, "sole_runtime_synchronize_caller_location")
    report = {"schema": 1, "kind": "TSC_BPE_LoRA_worker_stdlib_source_verification_v1", "passed": True,
              "worker_sha256": hashlib.sha256(WORKER.read_bytes()).hexdigest(),
              "fixture_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "run_synchronize_line": sync[0],
              "checks": checks, "check_count": len(checks), "rejected": rejected,
              "external_source_native_ML_serializer_and_RNG_are_explicit_mocks": True,
              "actual_ASR_or_tensor_state_restore_or_GPU_executed": False,
              "actual_closed_admission_or_activation_claimed": False, "external_exact_resume_supported": False}
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    with cli.output.open("x", encoding="utf-8") as stream: json.dump(report, stream, indent=2); stream.write("\n")
    print(json.dumps({"checks": len(checks), "rejections": len(rejected), "sync": sync[0]}))


if __name__ == "__main__": main()
