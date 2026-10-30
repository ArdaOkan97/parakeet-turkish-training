"""INACTIVE E:-only owned decoder-screen controller and native dispatcher.

One child, immutable source plan, coherent CUDA12.9 preload before Torch, the
existing allocator/host guards, and actual complete-DEV closure before chaining.
No source selection is possible until the paired TRAIN study actually closes.
"""
from __future__ import annotations
import argparse
import ast
from collections import deque
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from atomic_io_retry_v1 import write_json, read_json
from append_only_history_v1 import HistoryJournal
from cuda_cache_reclaim_v2 import CudaCacheReclaimer
import launch_with_memory_guard_TSC_resume_v3 as memory
from resource_watchdog_v2 import ResourceGuard, sample_memory
from tsc_bpe_full_decoder_transition_v1 import (
    SOURCE, ROOT, STORAGE, PHASE, UPDATES, require, record, canonical, preflight,
    source_context, checked, validate_scores,
)

WORKER = "train_lora_batched_tsc_bpe_full_decoder_v1.py"
KIND = "owned_TSC_BPE_full_decoder_screen_controller_v1"
PINNED_SUPPORT = {
    "native_tdt_checkpointed_v4.py": "24b9cf5b8f3e11d5b043c22d0940da3735a46f8f70b6f5810b97f17d545cd90e",
    "append_only_history_v1.py": "e3822b4b6f37c9e389b5f89200475bad4c928d4bef9c3b0336c2980ec9594980",
    "launch_with_memory_guard_TSC_resume_v3.py": "61fec6d3ef911665bdbe876dc333f3c6a640badc012596fe92ded1b80a10c7fa",
    "preload_nemo_linker_v1.py": "f8787b0f3346c9f526903ff0c06de4c266f940ca3b0f41c55907cc795e7cc2ba",
    "cuda_cache_reclaim_v2.py": "869b9ffaac70c9b7b6e7590ba1d330363100d6b2ddb9f8f7af9b2b9865c1c7f9",
    "resource_watchdog_v2.py": "64d1393e1e45cf65dac14f78ac955331c91b4a3f0eafbf7535196e2b61778641",
    "paired_windows_commit_telemetry_v1.py": "2ae9f2cdfc388a294914e000c60b62fe09cddac147700c5762851d626a95a683",
    "atomic_io_retry_v1.py": "2d3fe379d51021fd15679e562c71286f7f9501180d28dde77893e8f72c80a1b0",
}


def now(): return datetime.now(timezone.utc).isoformat()


def output_path(value):
    p = Path(value).resolve()
    require(p.is_relative_to(STORAGE.resolve()), "Every new output belongs to designated E: storage")
    return p


def pins(context):
    result = dict(context["parent_identity"]["source_sha256"])
    result.update(PINNED_SUPPORT)
    for name in (WORKER, Path(__file__).name, "tsc_bpe_full_decoder_transition_v1.py",
                 "tsc_bpe_full_decoder_sampler_v1.py", "bpe_full_decoder_optimizer_v1.py",
                 "bpe_training_label_catalog_v1.py", "run_TSC_bpe_full_decoder_journal_v1.py",
                 "BPE_decoder_serialization_repair_v1.py"):
        result[name] = record(SOURCE / name)["sha256"]
    for name, digest in result.items():
        require(record(SOURCE / name)["sha256"] == digest, "Frozen dependency changed: " + name)
    return result


def require_storage():
    require(os.environ.get("GPU_TRAINING_STORAGE_ROOT") == str(STORAGE)
            and Path(os.environ.get("ASR_STORAGE_CONFIG", "")).resolve() == (STORAGE / "storage_config.json").resolve(),
            "Dot-source E: use_storage.ps1 before new preparation or native dispatch")
    config = json.loads((STORAGE / "storage_config.json").read_text(encoding="utf-8"))
    require(config["storage_root"] == str(STORAGE) and config["allow_fallback_outside_root"] is False,
            "Explicit no-fallback storage policy required")
    for key in ("TEMP", "TMP", "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "HF_HOME", "CUDA_CACHE_PATH"):
        require(os.environ.get(key) and Path(os.environ[key]).resolve().is_relative_to(STORAGE.resolve()),
                "New native cache/temp must stay under E: " + key)


def sync_line():
    tree = ast.parse((SOURCE / WORKER).read_text(encoding="utf-8"))
    calls = [n for f in tree.body if isinstance(f, ast.FunctionDef) and f.name == "run"
             for n in ast.walk(f) if isinstance(n, ast.Call) and ast.unparse(n.func) == "torch.cuda.synchronize"]
    require(len(calls) == 1 and not calls[0].args and not calls[0].keywords,
            "One explicit complete optimizer-end CUDA synchronization required")
    return calls[0].lineno


class DecoderCaller(CudaCacheReclaimer):
    def __init__(self, cuda):
        self.cuda = cuda; self.worker_path = SOURCE / WORKER
        self.expected_sha = record(self.worker_path)["sha256"]
        self.line = sync_line(); self.threshold_bytes = 3000000000; self.max_events = 128
        self.events = deque(maxlen=128)
        self.eligible_calls = self.ignored_calls = self.reclamations = self.failed_reclamations = 0
        self.cleanup_seconds = 0.; self.original = self.wrapped = None


def worker_namespace(args, paths):
    return SimpleNamespace(run_id=args.run_id, run_directory=STORAGE/"runs"/args.run_id,
        decoder_source_receipt=Path(args.source_receipt), decoder_arm=args.decoder_arm,
        decoder_updates=UPDATES, shared_lr=1e-5, new_rows_lr=1e-4, batch_size=8, accumulation=2,
        native_max_joint_bytes=1700000000, native_backend_provenance=paths["native"],
        resume_exact=Path(args.closed_latest) if args.closed_latest else None, stop_file=paths["stop"],
        decoder_runtime_repair_receipt=args.runtime_repair_receipt)


def prepare(args):
    require_storage()
    require(.1 <= args.window_hours <= 3 and 5 <= args.poll_seconds <= 15,
            "Bounded native windows and fresh memory observations required")
    loop = output_path(args.output_dir)
    require(not loop.exists(), "Fresh owned execution directory required")
    paths = {key: loop/name for key, name in (("native", "native_backend.json"),
        ("identity", "worker_identity.json"), ("completion", "native_completion.json"),
        ("live", "owned_live_memory.json"), ("memory", "memory_evidence.json"),
        ("stop", "stop.txt"), ("log", "worker.log"))}
    values = worker_namespace(args, paths)
    context = preflight(values)
    p = pins(context)
    argv = []
    for key, value in vars(values).items():
        if value is not None: argv.extend(("--"+key.replace("_", "-"), str(value)))
    previous = None; exact_checkpoint = None
    if args.runtime_repair_receipt:
        from BPE_decoder_serialization_repair_v1 import validate
        require(args.previous_status and args.closed_latest, "Explicit failed-state/checkpoint repair bindings required")
        repair, original = validate(args.runtime_repair_receipt, run_id=args.run_id,
                                   checkpoint=args.closed_latest, current_pins=p)
        require(record(args.previous_status) == repair["failed_controller"], "Exact archived failed controller required")
        require(record(values.run_directory/"run_config.json")["sha256"] == repair["original_run_config"]["sha256"],
                "Unmodified original failed run admission required")
        previous = read_json(args.previous_status)
        exact_checkpoint = repair["saved_checkpoint"]
        resume_step = repair["saved_step"]
    elif args.previous_status:
        previous = read_json(args.previous_status)
        require(previous["run_id"] == args.run_id and previous["status"] == "train_window_ended"
                and previous["stop_category"] == "budget" and previous["child_exit_code"] == 0
                and previous["actual_completion"]["passed"] is True
                and previous["source_sha256"] == p and args.closed_latest,
                "Only an actual verified same-ID CLOSED0 budget boundary may continue automatically")
        chosen = Path(args.closed_latest).resolve()
        require(record(chosen) == previous["actual_completion"]["latest_checkpoint"],
                "Exact checkpoint must be the prior verified closed frontier")
        for key in ("run_config", "initial_source350"):
            checked(previous["actual_completion"][key])
        resume_step = previous["completed_step"]
        exact_checkpoint = previous["actual_completion"]["latest_checkpoint"]
    else:
        require(not args.closed_latest and not values.run_directory.exists(), "Fresh screen namespace required")
        resume_step = context["source_step"]
    require(resume_step < context["configured_end_step"], "Declared decoder screen is already complete")
    return {"kind": KIND, "phase": PHASE, "run_id": args.run_id, "decoder_arm": args.decoder_arm,
        "source_receipt": context["receipt_record"], "source_checkpoint": context["source_checkpoint"],
        "exact_checkpoint": exact_checkpoint,
        "runtime_repair_receipt": record(args.runtime_repair_receipt) if args.runtime_repair_receipt else None,
        "source_sha256": p, "worker_arguments": argv, "worker_arguments_sha256": canonical(argv),
        "source_step": context["source_step"], "resume_step": resume_step,
        "configured_end_step": context["configured_end_step"], "screen_updates": UPDATES,
        "paths": {k: str(v) for k,v in paths.items()}, "run_directory": str(values.run_directory),
        "previous_status": str(args.previous_status) if args.previous_status else None,
        "previous_observed_peak": previous.get("peak_observed_owned_private_bytes") if previous else None,
        "reference_source_policy_private_peak": context["reference_source_policy_private_peak"],
        "command": [str(ROOT/"work/nemo-native-env-v2/Scripts/python.exe"), "-X", "utf8", "-B",
                    str(Path(__file__).resolve()), "--worker-plan", str(loop/"plan.json"), "--gpu-window-authorized"],
        "window_hours": args.window_hours, "actual_runtime_mode": "append_only_journal_periodic10_full_DEV_checkpoint_end",
        "retention_stop_policy": "monitor_only", "full_epoch_claimed": False,
        "status": "prepared", "GPU_launched": False, "created_at_utc": now()}


def admission(previous_peak=None, *, reference_peak):
    value = sample_memory(os.getpid(), "before_BPE_decoder_screen_GPU_dispatch")["system_memory"]
    require(value.get("available") is True, "Fresh authoritative system commitment required")
    # New frozen-encoder/partitioned-moment allocation is not a measured fit.
    # Prior complete decoder/native run peak remains a conservative reference.
    reference = previous_peak or reference_peak
    require(type(reference) is int and reference > 0 and value["commit_fraction"] < .9
            and value["commit_used_bytes"] + reference < .9 * value["commit_limit_bytes"]
            and value["commit_limit_bytes"] - value["commit_used_bytes"] - reference > 2000000000,
            "Fresh baseline plus observed reference peak fails unchanged strict admission")
    return {"sample": value, "reference_peak_private_bytes": reference,
        "same_policy_peak": previous_peak is not None, "new_partition_native_fit_not_yet_measured": previous_peak is None,
        "strict_90percent_and_2GB_reserve": True, "no_pagefile_or_guard_change": True}


def run_worker(plan):
    require_storage()
    require("torch" not in sys.modules, "Native linker preload must precede Torch import")
    args_record = plan["worker_arguments"]
    require(canonical(args_record) == plan["worker_arguments_sha256"], "Actual argument plan changed")
    for name,digest in plan["source_sha256"].items():
        require(record(SOURCE/name)["sha256"] == digest, "Admitted source changed: "+name)
    paths = {k: output_path(v) for k,v in plan["paths"].items()}
    require(not any(paths[k].exists() for k in ("native", "identity", "completion", "live")),
            "Fresh PID/runtime evidence must not replace a previous execution")
    from preload_nemo_linker_v1 import preload
    before = memory.safe_sample(os.getpid(), "before_native_preload_and_Torch")
    loaded = preload()
    require(loaded["kind"] == "coherent_native_cuda12_9_preload_v1"
            and loaded["actual_binding_linker"]["version"] == [12,9] and loaded["cuda_initialized"] is False,
            "Actual coherent CUDA12.9 pre-Torch native preload required")
    identity = {"kind": "BPE_decoder_owned_worker_identity_v1", "pid": os.getpid(), "ppid": os.getppid(),
        "run_id": plan["run_id"], "started_at_utc": now(), "process_creation_identity": memory.creation_identity(os.getpid()),
        "worker_arguments_sha256": plan["worker_arguments_sha256"], "source_sha256": plan["source_sha256"],
        "actual_controller_source": record(__file__), "source_receipt": plan["source_receipt"]}
    write_json(paths["identity"], identity)
    identity_record = record(paths["identity"])
    lock = threading.Lock(); latest = [-float("inf")]
    def publish(value):
        observed = value.get("observed_monotonic", time.monotonic())
        with lock:
            if observed < latest[0]: return
            write_json(paths["live"], {"kind": "BPE_decoder_owned_live_memory_v1", "pid": identity["pid"],
                "run_id": plan["run_id"], "worker_identity": identity_record, "sample": value}, retry_seconds=1.)
            latest[0] = observed
    publish(before)
    import torch
    cap = memory.validate_capacity(torch.cuda.get_device_properties(0).total_memory, 8.)
    torch.cuda.set_per_process_memory_fraction(cap["per_process_allocator_fraction"], device=0)
    ready = {"kind": "BPE_full_decoder_native_ready_v1", "passed": True, "run_id": plan["run_id"],
        "pid": os.getpid(), "ppid": os.getppid(), "preload": loaded, "CUDA_version": "12.9",
        "torch_CUDA_version": torch.version.cuda, "worker_sha256": plan["source_sha256"][WORKER],
        "native_adapter_sha256": PINNED_SUPPORT["native_tdt_checkpointed_v4.py"],
        "native_batch_cap_bytes": 8000000000, "native_end_cache_bytes": 3000000000,
        "maximum_full_joint_bytes": 1700000000, "cap_applied": cap,
        "worker_arguments_sha256": plan["worker_arguments_sha256"], "identity": identity_record,
        "source_sha256": plan["source_sha256"], "ready_at_utc": now()}
    ready["worker_arguments"] = list(args_record)
    ready["source_receipt"] = plan["source_receipt"]
    ready["source_checkpoint"] = plan["source_checkpoint"]
    ready["exact_checkpoint"] = plan["exact_checkpoint"]
    write_json(paths["native"], ready)
    telemetry = memory.UpdateTelemetry(os.getpid(), sink=publish)
    sampler = memory.LiveTelemetrySampler(os.getpid(), publish)
    caller = memory.telemetry_reclaimer_class(DecoderCaller, telemetry)(torch.cuda)
    outcome, error = "failed", None
    try:
        sampler.start(); caller.install()
        worker = importlib.import_module(Path(WORKER).stem)
        args = worker.parser().parse_args(args_record)
        worker.validate_arguments(args); worker.run(args)
        outcome = "complete"
    except BaseException as exc:
        error = repr(exc); raise
    finally:
        sampler.stop_and_join(); caller.restore()
        unchanged = all(record(SOURCE/name)["sha256"] == digest for name,digest in plan["source_sha256"].items())
        write_json(paths["memory"], {"status": outcome, "error": error,
            "all_phase_telemetry": sampler.evidence(), "update_telemetry": telemetry.evidence(),
            "cache_reclamation": caller.evidence(), "sources_unchanged": unchanged})
        write_json(paths["completion"], {"kind": "BPE_full_decoder_native_completion_v1",
            "status": outcome, "error": error, "pid": os.getpid(), "run_id": plan["run_id"],
            "sources_unchanged": unchanged, "native_ready": record(paths["native"]),
            "worker_identity": identity_record, "finished_at_utc": now()})
        require(unchanged, "An admitted scientific/native source changed while running")


def request_stop(path, reason):
    if not path.exists():
        with path.open("x", encoding="utf-8") as stream: stream.write("Cooperative full-update save and DEV: "+reason+"\n")


def verify_completion(plan):
    paths = {k:Path(v) for k,v in plan["paths"].items()}
    native = read_json(paths["completion"])
    folder = Path(plan["run_directory"])
    completed = read_json(folder/"worker_completion.json")
    require(native["status"] == "complete" and native["sources_unchanged"] is True
            and completed["sources_unchanged"] is True and completed["final350_complete"] is True
            and completed["frozen192_unchanged"] is True and completed["phase"] == PHASE,
            "Actual complete native close/full DEV/frozen192 proof required")
    step = completed["completed_step"]
    require(plan["resume_step"] <= step <= plan["configured_end_step"]
            and completed["decoder_updates"] == step-plan["source_step"]
            and completed["mature_Adam_clock"] == step
            and completed["fresh_row_Adam_clock"] == (step-plan["source_step"] if plan["decoder_arm"] == "bpe512" else None),
            "Actual source/local/mature/fresh clocks differ")
    scores = read_json(STORAGE/"results"/(plan["run_id"]+"_step_"+str(step))/"summary.json")
    validate_scores(scores)
    journal = HistoryJournal(folder/"history_journal")
    frontier = journal.frontier()
    row = journal.get_step(step)
    require(frontier["max_step"] == step and "validation" in row
            and row["final_development_evaluation"] is True
            and row["validation"]["turkish_wer"] == scores["validation_tr"]["asr"]["wer"],
            "Saved frontier must have actual complete final seven-language DEV")
    return {"passed": True, "completed_step": step, "native_completion": record(paths["completion"]),
            "worker_completion": record(folder/"worker_completion.json"),
            "latest_checkpoint": record(folder/"latest_checkpoint.pt"),
            "best_checkpoint": record(folder/"best_checkpoint.pt"),
            "selection": record(folder/"selection.json"), "run_config": record(folder/"run_config.json"),
            "initial_source350": record(folder/"BPE_full_decoder_initial_source_equivalence.json"),
            "final_DEV": record(STORAGE/"results"/(plan["run_id"]+"_step_"+str(step))/"summary.json")}


def run_window(plan, args):
    lease_path = ROOT/"work/gpu_training_owner.json"
    lease = read_json(lease_path)
    require(lease.get("owner") == "/root" and lease.get("reserved") is True
            and lease.get("actual_GPU_running") is not True,
            "Sole-owner lease and actual previous-child closure required before dispatch")
    loop = output_path(args.output_dir); loop.mkdir(parents=True)
    plan["admission"] = admission(plan["previous_observed_peak"], reference_peak=plan["reference_source_policy_private_peak"])
    write_json(loop/"plan.json", plan)
    paths = {k:Path(v) for k,v in plan["paths"].items()}
    state = {**plan, "status": "starting", "GPU_launched": True, "published_step": plan["resume_step"]}
    child, stop_at, stop_reason = None, None, None
    observer = None; resource = ResourceGuard(); self_resource = ResourceGuard(); seen_dev = plan["resume_step"]
    started = time.monotonic(); last_progress = started; last_step = plan["resume_step"]
    try:
        with paths["log"].open("x", encoding="utf-8") as log:
            child = subprocess.Popen(plan["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state.update(status="running", owned_launcher_pid=child.pid, started_at_utc=now())
            lease.update(status="actual_BPE_decoder_screen_RUNNING_startup", target_run_id=plan["run_id"],
                running=True, actual_GPU_running=True, actual_training_optimizer_running=False,
                actual_owned_native_launcher_pid=child.pid, actual_owned_worker_pid=None,
                actual_controller_loop=str(loop), actual_source_step=plan["source_step"],
                actual_completed_step=plan["resume_step"], actual_current_window_positive_updates=0,
                updated_at_utc=now())
            for key in ("actual_exit_code", "actual_closed_tool_chunk"):
                if key in lease: lease.setdefault("prior_closed_dispatch_fields", {})[key] = lease.pop(key)
            write_json(lease_path, lease)
            print(f"ACTUAL owned decoder {plan['run_id']} PID{child.pid} from{plan['resume_step']}", flush=True)
            while child.poll() is None:
                identity = read_json(paths["identity"]) if paths["identity"].exists() else None
                if identity:
                    require(identity["run_id"] == plan["run_id"] and identity["worker_arguments_sha256"] == plan["worker_arguments_sha256"]
                            and (identity["pid"] == child.pid or identity["ppid"] == child.pid)
                            and identity["source_sha256"] == plan["source_sha256"], "Actual owned PID/source/argv differs")
                    require(identity["process_creation_identity"].get("available") is True
                            and identity["process_creation_identity"]["pid"] == identity["pid"],
                            "Actual owned native self creation identity required")
                sample = sample_memory(identity["pid"] if identity else child.pid, "decoder_screen")
                reason, observation = resource.observe(sample["system_memory"], sample["worker_private_memory"],
                                                       observed_monotonic=sample["observed_monotonic"])
                live = read_json(paths["live"]) if paths["live"].exists() else None
                if identity and live:
                    require(live["pid"] == identity["pid"] and live["run_id"] == plan["run_id"]
                            and live["worker_identity"] == record(paths["identity"]), "Owned live telemetry binding changed")
                    self_sample = live["sample"]
                    if time.monotonic()-self_sample.get("observed_monotonic", 0) > 15:
                        reason = reason or "actual_worker_self_memory_stale"
                    self_reason, self_observation = self_resource.observe(self_sample["system_memory"],
                        self_sample["worker_private_memory"], observed_monotonic=self_sample["observed_monotonic"])
                    reason = reason or self_reason
                elif time.monotonic()-started > 120: reason = reason or "actual_worker_self_memory_unavailable"
                journal_dir = Path(plan["run_directory"])/"history_journal"
                restored_path = loop/"exact_restore.json"
                repair_ready = not plan["runtime_repair_receipt"]
                if not repair_ready and restored_path.exists():
                    restored = read_json(restored_path)
                    require(restored["passed"] is True and restored["saved_step"] == plan["resume_step"]
                            and restored["exact_checkpoint"] == plan["exact_checkpoint"]
                            and restored["repair_receipt"] == plan["runtime_repair_receipt"]
                            and restored["journal"]["max_step"] == plan["resume_step"],
                            "Failed publication suffix must be preserved before fresh progress is observed")
                    repair_ready = True
                if repair_ready and (journal_dir/"HEAD.json").exists():
                    if observer is None: observer = HistoryJournal(journal_dir)
                    frontier = observer.frontier(); step = frontier["max_step"]
                    if step > last_step:
                        last_step = step; last_progress = time.monotonic()
                        lease.update(actual_completed_step=step,
                            actual_current_window_positive_updates=step-plan["resume_step"],
                            actual_training_optimizer_running=True, updated_at_utc=now())
                        if identity: lease["actual_owned_worker_pid"] = identity["pid"]
                        write_json(lease_path, lease)
                    state["published_step"] = step
                    for row in observer.iter_latest(max(plan["resume_step"]+1, seen_dev), refresh=False):
                        if "validation" in row and row["step"] > seen_dev:
                            seen_dev = row["step"]
                            print(f"{plan['run_id']} DEV{seen_dev}: TR{row['validation']['turkish_wer']:.6f} all6={row['validation']['retention_gate_passed']}", flush=True)
                if time.monotonic()-last_progress > 1800: reason = reason or "bounded_training_progress_timeout"
                if time.monotonic()-started >= args.window_hours*3600-300: reason = reason or "window_budget_checkpoint_boundary"
                if paths["stop"].exists() and stop_at is None: reason = reason or "ROOT_manual_cooperative_stop"
                private = sample["worker_private_memory"]
                if private.get("available"): state["peak_observed_owned_private_bytes"] = max(state.get("peak_observed_owned_private_bytes", 0), private["private_bytes"])
                state.update(updated_at_utc=now(), memory_observation=observation, actual_worker_self_memory=live)
                if reason and stop_at is None:
                    stop_at = time.monotonic(); stop_reason = reason; request_stop(paths["stop"], reason)
                    state.update(cooperative_stop_reason=reason, cooperative_stop_at_utc=now())
                write_json(loop/"loop_status.json", state)
                write_json(STORAGE/"experiment_queue_state.json", {"kind": KIND, "status": state["status"],
                    "run_id": plan["run_id"], "published_step": state["published_step"],
                    "source_step": plan["source_step"], "configured_end_step": plan["configured_end_step"],
                    "decoder_arm": plan["decoder_arm"], "owned_launcher_pid": child.pid,
                    "updated_at_utc": state["updated_at_utc"], "actual_runtime_mode": plan["actual_runtime_mode"]})
                require(stop_at is None or time.monotonic()-stop_at <= 600, "Owned child exceeded cooperative save/DEV grace")
                time.sleep(args.poll_seconds)
        state["child_exit_code"] = child.wait()
        require(child.returncode == 0, "Actual child failed; no automatic GPU retry")
        state["actual_completion"] = verify_completion(plan)
        state["completed_step"] = state["actual_completion"]["completed_step"]
        state["stop_category"] = ("configured_steps_complete" if state["completed_step"] == plan["configured_end_step"] else
                                  "budget" if stop_reason == "window_budget_checkpoint_boundary" else "review_required")
        state["status"] = "train_window_ended"
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), stop_category="review_required")
        raise
    finally:
        if child is not None and child.poll() is None:
            request_stop(paths["stop"], "controller_exception_owned_cleanup")
            try: state["child_exit_code"] = child.wait(timeout=600)
            except subprocess.TimeoutExpired: state["child_closure_unknown"] = True
        elif child is not None:
            state["child_exit_code"] = child.wait()
        state.update(finished_at_utc=now(), GPU_running=bool(child is not None and child.poll() is None))
        write_json(loop/"loop_status.json", state)
        lease.update(status="actual_BPE_decoder_screen_"+state["status"], running=state["GPU_running"],
            actual_GPU_running=state["GPU_running"], actual_training_optimizer_running=state["GPU_running"],
            actual_completed_step=state.get("completed_step", state["published_step"]),
            actual_child_exit_code=state.get("child_exit_code"), updated_at_utc=now())
        write_json(lease_path, lease)
        write_json(STORAGE/"experiment_queue_state.json", {
            "kind": KIND, "status": state["status"], "run_id": plan["run_id"],
            "published_step": state["published_step"], "completed_step": state.get("completed_step"),
            "child_exit_code": state.get("child_exit_code"), "GPU_running": state["GPU_running"],
            "source_step": plan["source_step"], "configured_end_step": plan["configured_end_step"],
            "decoder_arm": plan["decoder_arm"], "controller_status": str(loop/"loop_status.json"),
            "updated_at_utc": state["finished_at_utc"]})
    return state


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--worker-plan", type=Path)
    p.add_argument("--source-receipt", type=Path)
    p.add_argument("--decoder-arm", choices=("bpe512", "original8199"))
    p.add_argument("--run-id")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--previous-status", type=Path)
    p.add_argument("--closed-latest", type=Path)
    p.add_argument("--runtime-repair-receipt", type=Path)
    p.add_argument("--window-hours", type=float, default=3.)
    p.add_argument("--poll-seconds", type=float, default=5.)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--preflight-output", type=Path)
    p.add_argument("--gpu-window-authorized", action="store_true")
    return p


def main(args):
    if args.worker_plan:
        require(args.gpu_window_authorized, "Explicit sole GPU authorization required")
        run_worker(read_json(args.worker_plan)); return
    require(args.source_receipt and args.run_id and args.output_dir and args.decoder_arm, "Complete purpose-bound entry required")
    plan = prepare(args)
    if args.validate_only:
        if args.preflight_output:
            target = output_path(args.preflight_output); require(not target.exists(), "Fresh preflight receipt required")
            write_json(target, {**plan, "passed": True, "GPU_launched": False, "Torch_imported": "torch" in sys.modules})
        print(json.dumps({"passed": True, "GPU_launched": False, "source_step": plan["source_step"],
                          "configured_end_step": plan["configured_end_step"]})); return
    require(args.gpu_window_authorized, "Explicit sole GPU authorization required")
    original = args.output_dir; number = 1
    while True:
        state = run_window(plan, args)
        if state["stop_category"] != "budget": return
        args.previous_status = args.output_dir/"loop_status.json"
        args.closed_latest = Path(state["actual_completion"]["latest_checkpoint"]["path"])
        args.runtime_repair_receipt = None
        number += 1; args.output_dir = original.with_name(original.name+"_window_"+str(number))
        plan = prepare(args)


if __name__ == "__main__":
    main(parser().parse_args())
