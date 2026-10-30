"""Root-only bounded exact358 continuation with the verified split native loss.

Preserves scientific arguments, cadence250, Adam/sampler/RNG/history/best0.
This controller never evaluates TEST, publishes a model, or completes the goal.
--validate-only is stdlib-only and performs no GPU/process launch.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from atomic_io_retry_v1 import read_json as read, write_json as write
import run_experiment_loop_continuation_v8 as previous
from resource_watchdog_v2 import ResourceGuard, sample_memory

HERE, ROOT = previous.OUTPUT, previous.ROOT
LOOP = HERE / "runs/adaptive_20261011_native_split_memory_01"
TARGET = "adaptive_20261011_native_split_resume_01"
PARENT = "adaptive_20261011_native_memory_fix_01_public_winner_extension"
RECEIPT = HERE / "native_memory_fix_01_extension_exit_receipt_v1.json"
RECEIPT_SHA = "1a49692e738d07621435dd617ea6bb67afc696a49cca9759f695e4ce4d2495c0"
FORK = LOOP / "fork_snapshot/initial_state_358/fork_manifest.json"
FORK_SHA = "1963c1e96f49d499b3004959635074789b28b2984a3cab33f94df9130c9831f6"
SOURCE_PINS = {
    "train_lora_batched_data_v10.py": "4581e9510d7fa31d4b40ab795cd3a1c8af2f2df34ef729703607d97a37c0eb3c",
    "native_tdt_checkpointed_v3.py": "839194dddf83c3b08bf608d01cb540b2f160feee426a91373d14cfc426ab7494",
    "launch_native_training_v4.py": "5998ed49ea7ce8949c06444ada38fc1b87ae9edb4cc13ae9c928ad1be9a7f9d5",
    "launch_with_io_retry_v8.py": "2b99a23868e8e793df3ccf782536cbb91b33e212306422dc6bb6b3edba99cce9",
    "launch_with_cache_reclaim_v11.py": "03e02fbeb706cab907fd95a9f8d89c5fc01b31b4b9e668460973f8f0bb589a4e",
    "launch_with_memory_guard_v14.py": "73c3ecc659f8be4f9f1b0199570f82dc9e7c4b553467b3f55f37817f44c449d3",
    "cuda_cache_reclaim_v2.py": "869b9ffaac70c9b7b6e7590ba1d330363100d6b2ddb9f8f7af9b2b9865c1c7f9",
    "resource_watchdog_v2.py": "64d1393e1e45cf65dac14f78ac955331c91b4a3f0eafbf7535196e2b61778641",
    "paired_windows_commit_telemetry_v1.py": "2ae9f2cdfc388a294914e000c60b62fe09cddac147700c5762851d626a95a683",
    "public_turkish_data_policy_v3.py": "08eee3896bf8571c9e97dae1be688ecef7766816c019741dd7631d70f640ce41",
    "preload_nemo_linker_v1.py": "f8787b0f3346c9f526903ff0c06de4c266f940ca3b0f41c55907cc795e7cc2ba",
    "run_experiment_loop_continuation_v8.py": "c65e9f37eb0db4641552d35ccda52b5604223223deb88d7d9ebfcfe61f0f051d",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path):
    path = previous.scoped(path)
    return {"path": str(path), "sha256": sha(path)}


def replace_option(argv, flag, value):
    require(argv.count(flag) <= 1, "Duplicate "+flag)
    if flag in argv:
        index = argv.index(flag)
        require(index+1 < len(argv), "Missing value for "+flag)
        del argv[index:index+2]
    if value is not None:
        argv.extend([flag, str(value)])


def option(argv, flag):
    require(argv.count(flag) == 1 and argv.index(flag)+1 < len(argv), "One value required for "+flag)
    return argv[argv.index(flag)+1]


def prepare(args):
    require(0 < args.window_hours <= 3 and 5 <= args.poll_seconds <= 15, "Bounded3h window / fresh5–15s polls required")
    require(sha(RECEIPT) == RECEIPT_SHA and sha(FORK) == FORK_SHA, "Closed receipt or exact358 fork changed")
    fork, receipt = read(FORK, {}), read(RECEIPT, {})
    require(fork["passed"] and fork["source_run_id"] == PARENT and fork["target_run_id"] == TARGET
        and fork["source_step"] == 358 and fork["inherited_selected_step"] == 0 and fork["CUDA_initialized"] is False,
        "Exact358/best0 fork required")
    for name, digest in SOURCE_PINS.items():
        require(sha(HERE/name) == digest, "Frozen deployment source changed: "+name)
    require(not (LOOP/"loop_status.json").exists(), "Do not restart or overwrite an existing controller state")
    require(not (HERE/"runs"/TARGET/"run_config.json").exists(), "New run config namespace required")
    require(not (ROOT/"work"/(TARGET+"_latest.pt")).exists(), "Resume from archive; mutable latest must initially be absent")
    for role, item in fork["fork_checkpoint"].items():
        require(sha(item["path"]) == item["sha256"], "Immutable fork changed: "+role)
    best = fork["fork_checkpoint"]["best"]
    require(sha(best["mutable_work_path"]) == best["sha256"], "Preserved qualifying BEST0 changed")
    for step, item in fork["development_reuse"].items():
        for relative, digest in item["files_sha256"].items():
            require(sha(Path(item["target_path"])/relative) == digest, "Reused DEV changed: "+step+"/"+relative)
    resource = previous.scoped(args.resource_verification)
    kernel = previous.scoped(args.kernel_verification)
    resource_report, kernel_report = read(resource, {}), read(kernel, {})
    require(resource_report.get("passed") is True and resource_report.get("cap_gb") == 8
        and resource_report.get("max_joint_bytes") == 1_700_000_000, "Actual matching cap8 fulljoint resource proof required")
    require(kernel_report.get("passed") is True and SOURCE_PINS["native_tdt_checkpointed_v3.py"] in json.dumps(kernel_report),
        "Actual split native3 loss/gradient GPU proof/source pin required")
    for filename, digest in resource_report["source_sha256"].items():
        require(sha(filename) == digest, "Actual resource proof source changed")
    config = read(fork["source_run_config"]["path"], {})
    require(config["config_identity"] == fork["source_config_identity"] and config["run_id"] == PARENT
        and config["batch_size"] == 8 and config["accumulation"] == 2 and config["eval_every_updates"] == 250
        and config["replay_fraction"] == .5, "Preserved scientific B8x2/replay50/cadence250 configuration required")
    paths = {key: LOOP/(TARGET+suffix) for key, suffix in {
        "native": "_native_backend.json", "io": "_io.json", "cache": "_cache.json", "memory": "_memory.json",
        "identity": "_identity.json", "stop": "_stop.txt", "log": ".log"}.items()}
    for path in paths.values():
        require(not path.exists(), "New runtime/output path required: "+str(path))
    latest = fork["fork_checkpoint"]["latest"]
    transition = {"kind": "native_runtime_state_transition_v2", "status": "CPU_prepared", "passed": True,
        "source_run_id": PARENT, "target_run_id": TARGET, "source_step": 358, "inherited_selected_step": 0,
        "parent_latest": fork["source_checkpoint"]["latest"], "parent_best": fork["source_checkpoint"]["best"],
        "fork_latest": latest, "fork_best": best, "source_run_config": fork["source_run_config"],
        "source_config_identity_sha256": fork["source_config_identity_sha256"], "approved_source_changes": {},
        "native_backend_transition": {"old_path": config["native_backend_provenance"],
            "old_sha256": sha(config["native_backend_provenance"]), "new_path": str(paths["native"]),
            "expected_new_kind": "native_tdt_training_backend_v3"},
        "declared_allocator_policy": {"cap_gb": 8, "end_cache_gb": 4, "native_inactive_threshold_bytes": 750_000_000},
        "actual_GPU_kernel_proof": record(kernel), "actual_GPU_resource_proof": record(resource)}
    for old, new in (("train_lora_batched_data_v9.py", "train_lora_batched_data_v10.py"),
                     ("native_tdt_checkpointed_v2.py", "native_tdt_checkpointed_v3.py"),
                     ("launch_native_training_v3.py", "launch_native_training_v4.py")):
        transition["approved_source_changes"][old] = {"old_sha256": sha(HERE/old), "new_filename": new, "new_sha256": SOURCE_PINS[new]}
    migration = LOOP/"runtime_migration_cap8_v2.json"
    if migration.exists():
        require(read(migration, {}) == transition, "Prepared migration receipt differs")
    else:
        with migration.open("x", encoding="utf-8") as stream:
            json.dump(transition, stream, ensure_ascii=False, indent=2)
    parent_state = read(Path(receipt["predecessor_status_path"]), {})
    old_trial = next(t for t in parent_state["trials"] if t["run_id"] == PARENT)
    argv = list(old_trial["arguments"])
    original = list(argv)
    for flag, value in (("--run-id", TARGET), ("--resume-weights", None), ("--resume-exact", latest["path"]),
        ("--native-backend-provenance", paths["native"]), ("--resume-runtime-migration", migration),
        ("--stop-file", paths["stop"]), ("--max-wall-seconds", max(1, args.window_hours*3600-120))):
        replace_option(argv, flag, value)
    scientific_flags = ["--epochs", "--steps", "--batch-size", "--accumulation", "--eval-every-updates",
        "--head-lr", "--lora-lr", "--replay-fraction", "--replay-kl", "--duration-kl", "--feature-distillation",
        "--warmup-steps", "--schedule", "--rank", "--alpha", "--seed", "--max-tokens", "--chunk-size"]
    for flag in scientific_flags:
        require(option(original, flag) == option(argv, flag), "Scientific argument changed: "+flag)
    memory = ["--provenance", str(paths["io"]), "--cache-provenance", str(paths["cache"]),
        "--memory-provenance", str(paths["memory"]), "--worker-identity", str(paths["identity"]),
        "--cap-gb", "8", "--threshold-gb", "4", "--", *argv]
    command = [str(previous.scoped(args.python)), str(HERE/"launch_native_training_v4.py"),
        "--native-provenance", str(paths["native"]), "--memory-launcher", str(HERE/"launch_with_memory_guard_v14.py"),
        "--memory-launcher-sha256", SOURCE_PINS["launch_with_memory_guard_v14.py"],
        "--trainer-sha256", SOURCE_PINS["train_lora_batched_data_v10.py"], "--", *memory]
    return {"passed": True, "run_id": TARGET, "loop_id": LOOP.name, "initial_step": 358,
        "inherited_selected_step": 0, "fork": record(FORK), "predecessor_exit_receipt": record(RECEIPT),
        "runtime_migration": record(migration), "paths": {k:str(p) for k,p in paths.items()}, "command": command,
        "worker_arguments": argv, "code_sha256": SOURCE_PINS, "controller_sha256": sha(__file__),
        "kernel_verification": record(kernel), "resource_verification": record(resource),
        "scientific_arguments_unchanged": True, "exact_optimizer_sampler_RNG_continuation": True,
        "heldout_tests_evaluated": False, "CUDA_initialized_by_preflight": False}


def validate_identity(value, command, child_pid, started):
    require(value["kind"] == "owned_training_worker_identity_v1" and value["expected_run_id"] == TARGET
        and value["worker_arguments_sha256"] == previous.argument_sha(command["worker_arguments"]), "Actual worker argv/run differs")
    require(type(value["pid"]) is int and value["pid"] > 0
        and (value["pid"] == child_pid or value["ppid"] == child_pid), "Actual worker is not owned child/self")
    for key, name in (("memory_launcher_sha256", "launch_with_memory_guard_v14.py"),
        ("cache_launcher_sha256", "launch_with_cache_reclaim_v11.py"), ("cache_helper_sha256", "cuda_cache_reclaim_v2.py"),
        ("resource_helper_sha256", "resource_watchdog_v2.py"), ("resource_telemetry_sha256", "paired_windows_commit_telemetry_v1.py"),
        ("frozen_worker_sha256", "train_lora_batched_data_v10.py"), ("data_policy_sha256", "public_turkish_data_policy_v3.py")):
        require(value[key] == SOURCE_PINS[name], "Actual worker source changed: "+key)
    creation = value["process_creation_identity"]
    require(creation["available"] and creation["pid"] == value["pid"]
        and previous.stamp(started).timestamp()-5 <= creation["creation_unix_seconds"] <= previous.stamp(value["started_at_utc"]).timestamp()+1,
        "Actual owned worker creation time invalid")
    return value


def main(args):
    plan = prepare(args)
    if args.validate_only:
        if args.preflight_output:
            write(args.preflight_output, plan)
        print(json.dumps({"passed": True, "run_id": TARGET, "initial_step": 358, "CUDA_initialized": False}))
        return 0
    require(args.gpu_window_authorized, "Only root may authorize the exclusive GPU window")
    paths = {k:Path(v) for k,v in plan["paths"].items()}
    state = {**plan, "kind": "native_exact358_split_runtime_window_v1", "status": "starting", "started_at_utc": now(),
        "objective_complete": False, "objective_status": "incomplete", "next_action": "Continue DEV-selected adaptive work until Turkish target and retention are verified"}
    write(LOOP/"loop_status.json", state)
    child, stop_at, stop_reason = None, None, None
    started, deadline = time.monotonic(), time.monotonic()+args.window_hours*3600
    resource, progress = ResourceGuard(), previous.ProgressGuard(started)
    awake = previous.prior.keep_awake(True)
    state["awake_request"] = awake
    try:
        with paths["log"].open("x", encoding="utf-8") as log:
            child = subprocess.Popen(plan["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state["owned_launcher_pid"] = child.pid
            while child.poll() is None:
                history = read(HERE/"runs"/TARGET/"training_history.json", [])
                identity = validate_identity(read(paths["identity"], {}), plan, child.pid, state["started_at_utc"]) if paths["identity"].exists() else None
                phase = previous.log_phase(paths["log"], history)
                sample = sample_memory(identity["pid"] if identity else child.pid, phase["phase"])
                reason, observation = resource.observe(sample["system_memory"], sample["worker_private_memory"], observed_monotonic=sample["observed_monotonic"])
                self_reason, self_observation = previous.self_memory(Path(str(paths["identity"])+".live_memory.json"), identity, paths["identity"]) if identity else (None, {"available":False})
                progress_reason, progress_info = progress.observe(history, time.monotonic())
                reason = reason or self_reason or progress_reason
                if not identity and time.monotonic()-started > 120:
                    reason = reason or "owned_worker_identity_unavailable"
                if identity and not self_observation["available"] and time.monotonic()-started > 120:
                    reason = reason or "actual_worker_self_memory_unavailable_or_stale"
                if not history and time.monotonic()-started > 1800:
                    reason = reason or "bounded_preparation_timeout"
                if time.monotonic() >= deadline-120:
                    reason = reason or "window_budget_checkpoint_boundary"
                state.update(status="running", phase=phase, published_step=history[-1]["step"] if history else 358,
                    system_and_owned_private_memory=observation, actual_worker_self_memory=self_observation,
                    progress=progress_info, updated_at_utc=now())
                with (LOOP/(TARGET+"_memory_observations.jsonl")).open("a", encoding="utf-8") as out:
                    out.write(json.dumps({"at_utc":now(), "phase":phase, "memory":observation, "worker_self":self_observation, "progress":progress_info})+"\n")
                if reason and stop_at is None:
                    stop_at, stop_reason = time.monotonic(), reason
                    paths["stop"].write_text("Cooperative save/DEV requested: "+reason+"\n", encoding="utf-8")
                    state.update(cooperative_stop_reason=reason, cooperative_stop_requested_at_utc=now())
                write(LOOP/"loop_status.json", state)
                if stop_at and time.monotonic()-stop_at > 300:
                    raise RuntimeError("Owned child exceeded cooperative grace; no next GPU child")
                time.sleep(args.poll_seconds)
        state["child_exit_code"] = child.returncode
        require(child.returncode == 0, "Training child failed; review before another GPU trial")
        selection = read(HERE/"runs"/TARGET/"selection.json", {})
        require(selection["run_id"] == TARGET and selection["completed_steps"] >= 358, "Durable continuation selection missing")
        for key in ("io", "cache", "memory"):
            require(read(paths[key], {}).get("status") == "complete", "Actual runtime completion missing: "+key)
        completion = read(paths["native"].with_suffix(".completion.json"), {})
        require(completion.get("status") == "complete" and completion.get("sources_unchanged") is True, "Native runtime completion/source proof missing")
        for name, digest in SOURCE_PINS.items():
            require(sha(HERE/name) == digest, "Runtime source changed: "+name)
        state.update(status="resource_review_required" if stop_reason and stop_reason != "window_budget_checkpoint_boundary" else "train_window_ended",
            completed_steps=selection["completed_steps"], selected_step=selection["selected_step"], selection=record(HERE/"runs"/TARGET/"selection.json"),
            resource_review_required=bool(stop_reason and stop_reason != "window_budget_checkpoint_boundary"), finished_at_utc=now())
        return 1 if state["resource_review_required"] else 0
    except BaseException as error:
        state.update(status="failed", error=repr(error), objective_complete=False, finished_at_utc=now())
        raise
    finally:
        if child is not None and child.poll() is None:
            if not paths["stop"].exists():
                paths["stop"].write_text("Controller exit: cooperative save/DEV requested\n", encoding="utf-8")
            try:
                child.wait(timeout=300)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=30)
        state["awake_clear"] = previous.prior.keep_awake(False)
        write(LOOP/"loop_status.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=ROOT/"work/nemo-native-env-v2/Scripts/python.exe")
    parser.add_argument("--kernel-verification", type=Path, required=True)
    parser.add_argument("--resource-verification", type=Path, default=HERE/"native_long_memory_cap8_v4.json")
    parser.add_argument("--window-hours", type=float, default=3.)
    parser.add_argument("--poll-seconds", type=float, default=5.)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--gpu-window-authorized", action="store_true")
    sys.exit(main(parser.parse_args()))
