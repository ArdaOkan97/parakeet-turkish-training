"""Focused CPU-only actual BEST admission plus small serializer/state-boundary fixture."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
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
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0, str(HERE))
import tsc_optimization_transition_v1 as admission


def main():
    start = time.perf_counter(); checks = []; rejections = []
    def check(ok, text):
        if not ok: raise AssertionError(text)
        checks.append(text)
    def reject(action, text):
        try: action()
        except (ValueError, KeyError): rejections.append(text)
        else: raise AssertionError("Accepted corruption: " + text)
    receipt_path = HERE / "TSC_domain30_01_actual_CLOSED_branch_receipt_v1.json"
    receipt = admission.read(receipt_path)
    parent = admission.read(receipt["artifacts"]["run_config"]["path"])
    original = ast.parse((HERE / admission.PARENT_WORKER).read_text(encoding="utf-8"))
    new = ast.parse((HERE / admission.WORKER).read_text(encoding="utf-8"))
    old_functions = {x.name: x for x in original.body if isinstance(x, (ast.FunctionDef, ast.ClassDef))}
    functions = {x.name: x for x in new.body if isinstance(x, (ast.FunctionDef, ast.ClassDef))}
    for name in ("BucketCycler", "BalancedBucketSampler", "save_training_state", "lr_factor", "next_evaluation_epoch",
                 "development_gate", "validate_trainable_checkpoint", "source_isolation", "annotate_cache_lengths", "native_loss_function"):
        check(ast.dump(old_functions[name], include_attributes=False) == ast.dump(functions[name], include_attributes=False),
              "Unchanged core " + name)
    old_loop = next(x for x in ast.walk(old_functions["run"]) if isinstance(x, ast.For) and ast.unparse(x.target) == "step")
    new_loop = next(x for x in ast.walk(functions["run"]) if isinstance(x, ast.For) and ast.unparse(x.target) == "step")
    check(ast.dump(old_loop, include_attributes=False) == ast.dump(new_loop, include_attributes=False), "Entire optimizer-update loop AST unchanged")
    env = {"argparse": argparse, "math": math, "Path": Path, "DATA": ROOT / "work/data", "OUTPUT": HERE, "__doc__": "CPU fixture"}
    exec(compile(ast.Module(body=[functions["parser"], functions["lr_factor"]], type_ignores=[]), "fixture", "exec"), env)
    parser = env["parser"]()
    args = argparse.Namespace(**{a.dest: parent.get(a.dest, a.default) for a in parser._actions if a.dest != "help"})
    for action in parser._actions:
        if action.type is Path and getattr(args, action.dest, None) is not None:
            setattr(args, action.dest, Path(getattr(args, action.dest)))
    args.run_id = receipt["declared_next_change"]["new_run_id"]
    args.head_lr = 1e-5; args.resume_weights = args.resume_exact = args.resume_runtime_migration = None
    args.resume_optimization = Path(receipt["artifacts"]["best_checkpoint"]["path"])
    args.optimization_source_receipt = receipt_path
    context = admission.preflight(args)
    check(context["parent_identity"] == parent["config_identity"], "Actual CLOSED recipe/selection/source pins admitted")
    for key, value in (("head_lr", 2e-5), ("lora_lr", 1e-4), ("replay_kl", 5.), ("rank", 16),
                       ("replay_seed", 0), ("steps", 250), ("sampling_mode", "uniform")):
        altered = copy.copy(args); setattr(altered, key, value)
        reject(lambda: admission.preflight(altered), "Reject changed " + key)
    altered = copy.copy(args); altered.resume_optimization = Path(receipt["artifacts"]["latest_checkpoint"]["path"])
    reject(lambda: admission.preflight(altered), "Reject latest2000 state")
    import torch
    torch.set_num_threads(1)
    check(not torch.cuda.is_initialized(), "CUDA not initialized before actual CPU BEST read")
    state = torch.load(args.resume_optimization, map_location="cpu", weights_only=True)
    boundary = admission.validate_parent_state(state, context)
    check(boundary["inherited_steps"] == 1500 and boundary["first_changed_lr_step"] == 1501, "Actual global LR boundary1501")
    check(len(state["optimizer"]["state"]) == len(state["weights"]) == 207, "Actual207 weights/Adam states")
    check(all(float(value["step"]) == 1500 for value in state["optimizer"]["state"].values()), "All actual207 Adam steps1500")
    check(all(torch.isfinite(value).all().item() for value in state["weights"].values()), "All actual weights finite")
    check(all(torch.isfinite(value[key]).all().item() for value in state["optimizer"]["state"].values() for key in ("exp_avg", "exp_avg_sq")), "All actual Adam moments finite")
    check(state["next_eval_epoch"] == .5 and state["retention_failures"] == 0, "Actual BEST evalclock/counters retained")
    for key, value in (("step", 2000), ("selected_step", 2000), ("retention_failures", 2)):
        altered = dict(state); altered[key] = value
        reject(lambda: admission.validate_parent_state(altered, context), "Reject mixed full-state " + key)
    altered = dict(state); altered["history"] = state["history"][:-1]
    reject(lambda: admission.validate_parent_state(altered, context), "Reject history rewind mismatch")
    altered = dict(state); altered["optimizer"] = dict(state["optimizer"])
    altered["optimizer"]["state"] = dict(state["optimizer"]["state"]); altered["optimizer"]["state"].pop(next(iter(altered["optimizer"]["state"])))
    reject(lambda: admission.validate_parent_state(altered, context), "Reject missing Adam moment state")
    factor = env["lr_factor"](1501, args.steps, args.warmup_steps, args.schedule, args.min_lr_ratio)
    check(factor == 1., "No warmup reset at1501")
    # A small207-parameter CPU optimizer tests restore/change/serializer without
    # allocating a model or claiming the actual GPU-equivalence check has run.
    params = [torch.nn.Parameter(torch.tensor([float(i)])) for i in range(207)]
    optimizer = torch.optim.AdamW([{"params": params[:15], "lr": 5e-5, "weight_decay": .01},
                                  {"params": params[15:], "lr": 3e-4, "weight_decay": 0.}])
    synthetic = {"state": {i: {"step": torch.tensor(1500.), "exp_avg": torch.tensor([.1]), "exp_avg_sq": torch.tensor([.2])} for i in range(207)},
                 "param_groups": copy.deepcopy(state["optimizer"]["param_groups"])}
    optimizer.load_state_dict(synthetic)
    optimizer.param_groups[0]["lr"] = args.head_lr * factor
    optimizer.param_groups[1]["lr"] = args.lora_lr * factor
    check(optimizer.param_groups[0]["lr"] == 1e-5 and optimizer.param_groups[1]["lr"] == 3e-4, "Only head-group LR changes")
    check(all(torch.equal(optimizer.state_dict()["state"][i][key], synthetic["state"][i][key]) for i in range(207) for key in ("step", "exp_avg", "exp_avg_sq")), "All207 synthetic moments/steps unchanged after restore")
    small_weights = {"fixture_" + str(i): p.detach() for i, p in enumerate(params)}
    save_env = {"torch": torch, "Path": Path, "random": random, "checkpoint_state": lambda model: small_weights,
                "encoder_lora_metadata": lambda model: state["lora_metadata"]}
    exec(compile(ast.Module(body=[functions["save_training_state"]], type_ignores=[]), "serializer", "exec"), save_env)
    original_cuda_get = torch.cuda.get_rng_state_all
    torch.cuda.get_rng_state_all = lambda: state["cuda_rng"]
    try:
        with tempfile.TemporaryDirectory(dir=ROOT / "work/research", prefix="headlr_serializer_") as folder:
            path = Path(folder) / "new_branch_best.pt"
            new_identity = copy.deepcopy(state["config_identity"]); new_identity["training_arguments"]["head_lr"] = 1e-5
            save_env["save_training_state"](path, None, optimizer, type("Sampler", (), {"state_dict": lambda self: state["sampler"]})(),
                run_id=args.run_id, step=1500, config_identity=new_identity, history=state["history"], best_score=state["best_score"],
                selected_step=1500, patience=state["patience"], retention_failures=state["retention_failures"],
                next_eval_epoch=state["next_eval_epoch"], train_wall_seconds=state["train_wall_seconds"],
                audio_seconds=state["audio_seconds"], language_audio_seconds=state["language_audio_seconds"],
                evaluation_wall_seconds=state["evaluation_wall_seconds"], initial_gate=state["initial_gate"])
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            check(loaded["run_id"] == args.run_id and loaded["step"] == loaded["selected_step"] == 1500, "Own NEW full BEST serializer namespace/boundary")
            for key in ("history", "sampler", "patience", "retention_failures", "next_eval_epoch", "train_wall_seconds", "audio_seconds", "language_audio_seconds", "evaluation_wall_seconds", "initial_gate"):
                check(loaded[key] == state[key], "Preserved serializer field " + key)
            check(loaded["config_identity"]["training_arguments"]["head_lr"] == 1e-5, "New serialized optimization identity")
    finally:
        torch.cuda.get_rng_state_all = original_cuda_get
    check(not torch.cuda.is_initialized(), "CUDA remains uninitialized")
    check(admission.record(args.resume_optimization) == receipt["artifacts"]["best_checkpoint"], "Parent BEST bytes unchanged after CPU checks")
    report = {"schema": 1, "kind": "TSC_headlr_state_fork_CPU_verification_v1", "passed": True,
        "checks": len(checks), "rejected_unapproved_differences": len(rejections), "check_labels": checks, "rejections": rejections,
        "source_receipt": admission.record(receipt_path), "actual_state_boundary": boundary,
        "actual_parent_checkpoint": receipt["artifacts"]["best_checkpoint"],
        "worker_sha256": admission.sha(HERE / admission.WORKER), "transition_sha256": admission.sha(admission.__file__),
        "CUDA_initialized": False, "actual_GPU_source300_equivalence": "pending ROOT worker execution before update1501",
        "small_serializer_fixture_not_an_actual_model_fork": True, "elapsed_seconds": time.perf_counter() - start}
    output = HERE / "TSC_headlr_state_fork_CPU_verification_v1.json"
    with output.open("x", encoding="utf-8") as stream: json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"passed": True, "checks": len(checks), "rejections": len(rejections), "path": str(output), "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
