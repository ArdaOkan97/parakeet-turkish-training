"""INACTIVE ROOT-only byte archive of a CLOSED own-phase2 latest boundary.

No checkpoint/model/ML reads occur at import. Future invocation requires ROOT's
actual owned exit0 observation and terminal budget/phase-bound JSON before the
first .pt byte. Latest/BEST remain distinct full two-Adam states. The original
native-ready and initial300 scientific tokens KEEP their original immutable
paths; archival backups do not become new worker inputs. No tensor validation,
process-closure inference, training launch, or new phase/scorer is performed.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PHASE = "HF_TSC_BPE512_new_rows_plus_encoder_LoRA_v1"
FIRST_CONTROLLER = "run_TSC_bpe_lora_native4_v2.py"
FIRST_CONTROLLER_SHA = "7fa384c0919f1780a0a415089d3cfd9287e0be3d55c51894060e2747c7670a3e"
EXACT_CONTROLLER = "resume_TSC_bpe_lora_same_phase_v1.py"
RECEIPT_KIND = "ROOT_owned_closed_TSC_BPE_phase2_exact_v1"
LANGUAGES = ("tr", "en", "es", "it", "fr", "de")
RETENTION_MAX = {"en": 69, "es": 52, "it": 30, "fr": 74, "de": 70}
IMMUTABLE_TOKENS = ("original_native_ready", "initial_source_equivalence")
CHECKPOINT_BYTE_READS = 0


def require(value, message):
    if not value:
        raise ValueError(message)


def scoped(path):
    value = Path(path).resolve()
    require(value.is_relative_to(ROOT), "Artifact must remain within this workspace")
    return value


def read(path):
    return json.loads(scoped(path).read_text(encoding="utf-8-sig"))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()


def record(path):
    global CHECKPOINT_BYTE_READS
    value = scoped(path)
    before = value.stat()
    if value.suffix == ".pt":
        CHECKPOINT_BYTE_READS += 1
    digest = hashlib.sha256()
    with value.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    after = value.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "Artifact changed while hashing: " + str(value))
    return {"path": str(value), "sha256": digest.hexdigest(), "bytes": after.st_size}


def write_new(path, value):
    value_path = scoped(path)
    require(not value_path.exists(), "Fresh output path required")
    value_path.parent.mkdir(parents=True, exist_ok=True)
    with value_path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def controller_kind(path):
    """Read a source literal, without importing an executor or its dependencies."""
    tree = ast.parse(scoped(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in
                ("CONTROLLER_KIND", "KIND") for target in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, str) and "controller" in value:
                return value
    raise ValueError("Reviewed executor's controller kind literal is missing")


def validate_closure_header(controller):
    """Pure metadata gate: never read checkpoint bytes or assume an OS exit."""
    require(controller.get("status") == "train_window_ended" and controller.get("child_exit_code") == 0
        and controller.get("resource_review_required") is False and controller.get("phase") == PHASE,
        "Source is not a successful CLOSED own-phase2 boundary; no checkpoint bytes read")
    require(controller.get("stop_category") in ("budget", "phase_bound")
        and controller.get("finished_at_utc") and controller.get("actual_runtime_completion"),
        "Only actual budget/phase-bound runtime completion may continue; no quality/resource/manual retry")
    require(controller.get("smoke_updates") in (None, 0), "A separate smoke is not an exact continuation source")
    run_id = controller.get("run_id")
    require(isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,110}", run_id),
            "Safe same scientific run ID required")
    return run_id


def inspect_closed(source_loop):
    """Future ROOT small-JSON inspection, before any .pt hashing/copying."""
    status_path = scoped(source_loop) / "loop_status.json"
    controller = read(status_path)
    run_id = validate_closure_header(controller)
    executor_name = FIRST_CONTROLLER if controller.get("controller_sha256") == FIRST_CONTROLLER_SHA else EXACT_CONTROLLER
    executor = HERE / executor_name
    require(executor.is_file() and record(executor)["sha256"] == controller.get("controller_sha256")
        and controller.get("kind") == controller_kind(executor), "Closed executor source/hash/kind differs")
    run = HERE / "runs" / run_id
    config, selection, history = (read(run / name) for name in
        ("run_config.json", "selection.json", "training_history.json"))
    identity = config["config_identity"]
    phase = identity["adaptation_phase"]
    s, k, n = (phase.get("original_source_selected_step"), phase.get("selected_acquisition_updates"),
        selection.get("adaptation_updates_completed"))
    require(all(type(value) is int and value > 0 for value in (s, k, n)), "Actual positive S/K/N required")
    step = s+k+n
    selected = selection.get("selected_step")
    require(config.get("run_id") == selection.get("run_id") == run_id
        and config.get("phase") == identity.get("phase") == selection.get("phase") == PHASE
        and config.get("smoke_updates") is None and config.get("steps") == 44958
        and type(selected) is int and s+k <= selected <= step
        and selection.get("completed_steps") == controller.get("completed_steps") == step
        and controller.get("selected_step") == selected and controller.get("adaptation_updates_completed") == n
        and selection.get("row_Adam_clock") == k+n and selection.get("LoRA_Adam_clock") == s+n
        and selection.get("source_phase1_step") == phase.get("source_phase1_step") == s+k
        and selection.get("original_source_selected_step") == s and selection.get("selected_acquisition_updates") == k,
        "Own latest N/clocks must not be replaced by local BEST or original207/phase1 state")
    require(phase.get("active_parameter_tensors") == 195 and phase.get("row_lr") == 1e-4
        and phase.get("lora_lr") == 3e-4 and phase.get("old15_head_moments_active") is False
        and phase.get("further_unfreeze_allowed") is False
        and len(history) == step and all(row.get("step") == index for index, row in enumerate(history, 1))
        and all(row.get("adaptation_updates_completed") == row["step"]-s-k for row in history[s+k:]),
        "Original phase2 optimizer policy and complete ordered inherited history required")
    paths = controller["paths"]
    original_ready = scoped(paths["native"])
    runtime_ready = scoped(paths.get("fresh_runtime", paths["native"]))
    completion_path = runtime_ready.with_suffix(".completion.json")
    ready, actual_ready, done, owned = (read(path) for path in
        (original_ready, runtime_ready, completion_path, scoped(paths["identity"])))
    require(ready.get("run_id") == actual_ready.get("run_id") == done.get("run_id") == run_id
        and ready.get("adaptation_phase", ready.get("phase")) == PHASE
        and done.get("status") == "complete" and done.get("sources_unchanged") is True
        and done.get("phase") == PHASE and not done.get("error")
        and done.get("pid") == actual_ready.get("pid") == owned.get("pid")
        and done.get("actual_dispatch_source_sha256") == controller["controller_sha256"]
        and done.get("backend_ready_sha256") == record(runtime_ready)["sha256"],
        "Actual CLOSED source native process/ready/completion must agree")
    runtime = {}
    for name, kind in (("memory", "TSC_BPE_adaptation_memory_runtime_v1"),
                       ("cache", "TSC_BPE_adaptation_cache_runtime_v1"), ("io", "TSC_BPE_adaptation_atomic_IO_v1")):
        value = read(paths[name])
        require(value.get("kind") == kind and value.get("status") == "complete"
            and value.get("sources_unchanged") is True and not value.get("error"), "Runtime completion failed: "+name)
        runtime[name] = value
    start_n = controller.get("phase2_updates_at_window_start", 0)
    require(type(start_n) is int and 0 <= start_n < n
        and runtime["cache"]["cache_reclamation"]["eligible_update_calls"] == n-start_n,
        "Actual fresh cache hook suffix must equal final N minus inherited window-start N")
    memory = runtime["memory"]
    require(memory.get("cap_bytes") == 8_000_000_000 and memory.get("cleanup_threshold_bytes") == 4_000_000_000
        and memory.get("allocator_cap_applied") is True
        and memory["all_phase_owned_memory_telemetry"]["thread_join_completed"] is True,
        "Original cap8/end4/owned-memory safeguards required")
    proof_path = run / "TSC_BPE_LoRA_initial_source_equivalence.json"
    proof = read(proof_path)
    require(proof.get("kind") == "TSC_BPE512_LoRA_before_update_source_equivalence_v1"
        and proof.get("run_id") == run_id and proof.get("source_selected_step") == s+k
        and proof.get("passed") is True and proof.get("actual_GPU_decode") is True
        and proof.get("before_phase2_updates") is True and proof.get("compared_fixed_DEV_utterances") == 300
        and proof.get("all_raw_transcripts_and_emitted_paths_exact") is True
        and proof.get("all_five_retention_gates_passed") is True
        and identity.get("initial_source_equivalence_sha256") == record(proof_path)["sha256"],
        "Original actual initial300 scientific proof required, without regeneration")
    selected_folder = HERE / "results" / f"{run_id}_step_{selected}"
    selected_summary = read(selected_folder / "summary.json")
    scores = {language: selected_summary[f"validation_{language}"]["asr"] for language in LANGUAGES}
    require(all(value.get("utterances") == 50 and type(value.get("word_errors")) is int
        and value.get("reference_words", 0) > 0 for value in scores.values())
        and all(scores[language]["word_errors"] <= bound for language, bound in RETENTION_MAX.items()),
        "Saved separate BEST must retain its actual fixed300/all-five evidence")
    sources = {"latest_checkpoint": scoped(selection["latest_weights"]), "best_checkpoint": scoped(selection["weights"]),
        "run_config": run / "run_config.json", "training_history": run / "training_history.json",
        "selection": run / "selection.json", "controller_status": status_path,
        "original_native_ready": original_ready, "runtime_native_ready": runtime_ready,
        "initial_source_equivalence": proof_path, "runtime_completion": completion_path,
        "fresh_runtime": scoped(paths["identity"]), "owned_worker_identity": scoped(paths["identity"]),
        **{name: scoped(paths[name]) for name in ("memory", "cache", "io")},
        "selected_DEV_summary": selected_folder / "summary.json", "selected_DEV_metadata": selected_folder / "metadata.json",
        **{f"selected_DEV_{language}": selected_folder / f"validation_{language}.jsonl" for language in LANGUAGES}}
    latest_folder = HERE / "results" / f"{run_id}_step_{step}"
    sources.update({"latest_DEV_summary": latest_folder / "summary.json", "latest_DEV_metadata": latest_folder / "metadata.json",
        **{f"latest_DEV_{language}": latest_folder / f"validation_{language}.jsonl" for language in LANGUAGES}})
    require(scoped(selection["latest_weights"]) != scoped(selection["weights"]), "Latest and BEST must be separately preserved")
    return {"controller": controller, "config": config, "selection": selection, "history": history,
        "executor": executor, "sources": sources, "original_step": s, "acquisition_updates": k,
        "phase2_updates": n, "completed_steps": step, "selected_step": selected, "selected_DEV_asr": scores}


def make_receipt(args):
    if not args.validate_only:
        require(args.archive_dir and args.receipt_output, "Fresh archive and receipt paths required")
        archive = scoped(args.archive_dir)
        require(archive.is_relative_to(ROOT / "work/closed-sources") and not archive.exists(),
                "Fresh closed-source archive required")
        require(not scoped(args.receipt_output).exists(), "Fresh receipt output required")
    context = inspect_closed(args.source_loop)
    controller, config = context["controller"], context["config"]
    observed = datetime.fromisoformat(args.owned_exit_observed_at.replace("Z", "+00:00"))
    finished = datetime.fromisoformat(controller["finished_at_utc"].replace("Z", "+00:00"))
    require(type(args.owned_session_id) is int and args.owned_session_id > 0
        and re.fullmatch(r"[0-9a-f]+", args.owned_exit_chunk or "") and observed.tzinfo is not None
        and finished.tzinfo is not None and observed >= finished, "Actual ROOT owned exit0 tool observation must follow closure")
    n, s, k = context["phase2_updates"], context["original_step"], context["acquisition_updates"]
    total = args.approved_total_phase2_updates
    require(type(total) is int and n < total <= 44958-s-k, "Explicit later total-N bound within unchanged full horizon required")
    initial_bound = config["config_identity"]["adaptation_phase"]["effective_max_updates"]
    require(type(initial_bound) is int and initial_bound > 0
        and (total <= initial_bound or args.allow_stopping_bound_extension),
        "Beyond immutable original screen requires explicit ROOT stopping-bound extension")
    # Only after all actual terminal/runtime/science/owned-exit gates above do
    # checkpoint byte hashing and archival copying become permissible.
    original = {name: record(path) for name, path in context["sources"].items()}
    from tsc_bpe_lora_exact_transition_v1 import validate_receipt
    receipt = {"schema": 1, "kind": RECEIPT_KIND, "status": "actual_CLOSED_complete_phase2_boundary",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_id": controller["run_id"],
        "target_run_id": controller["run_id"], "allow_same_phase2_exact_resume": True,
        "guards_retained": True, "further_unfreeze_allowed": False,
        "original_step": s, "acquisition_updates": k, "phase2_updates": n,
        "completed_steps": context["completed_steps"], "selected_step": context["selected_step"],
        "approved_total_phase2_updates": total, "allow_stopping_bound_extension": args.allow_stopping_bound_extension,
        "immutable_initial_phase2_bound": initial_bound,
        "scientific_config_identity_sha256": canonical(config["config_identity"]),
        "training_history_sha256": canonical(context["history"]),
        "source_controller_path": str(context["executor"]), "source_controller_sha256": controller["controller_sha256"],
        "source_stop_category": controller["stop_category"],
        "source_stop_reason": context["selection"].get("stop_early_reason"),
        "source_window_start_phase2_updates": controller.get("phase2_updates_at_window_start", 0),
        "selected_DEV_asr": context["selected_DEV_asr"], "existing_retention_max_word_errors": RETENTION_MAX,
        "artifacts": original, "original_artifact_records": original,
        "original_artifact_paths": {name: value["path"] for name, value in original.items()},
        "owned_actual_exits": [{"role": "trainer_controller", "session_id": args.owned_session_id,
            "exit_code": 0, "tool_chunk": args.owned_exit_chunk, "observed_at_utc": observed.isoformat()}],
        "receipt_creator": record(__file__), "checkpoint_tensors_audited_here": False,
        "actual_exact_utility_restore_and_tensor_frontier_validation_pending": True,
        "original_native_ready_and_initial300_paths_preserved": True,
        "latest_own_phase2_frontier_and_selected_BEST_not_mixed": True,
        "runtime_executor_is_separate_from_original_scientific_identity": True,
        "torch_imported": "torch" in sys.modules, "actual_training_activation": False}
    validate_receipt(receipt, controller["run_id"])
    if args.validate_only:
        return {"kind": "ROOT_closed_own_phase2_exact_source_preflight_v1", "passed": True,
            "archive_created": False, "run_id": controller["run_id"], "original_step": s, "acquisition_updates": k,
            "phase2_updates": n, "completed_steps": context["completed_steps"],
            "approved_total_phase2_updates": total, "original_artifacts": original,
            "checkpoint_tensors_audited": False}
    archive.mkdir(parents=True, exist_ok=False)
    archived, backups = {}, {}
    copies = {}
    for name, source in context["sources"].items():
        if source in copies:
            bound = copies[source]
        else:
            destination = archive / (name+source.suffix)
            shutil.copy2(source, destination)
            bound = record(destination)
            require((bound["sha256"], bound["bytes"]) == (original[name]["sha256"], original[name]["bytes"])
                and record(source) == original[name], "Source or archive changed: "+name)
            copies[source] = bound
        if name in IMMUTABLE_TOKENS:
            archived[name] = original[name]
            backups[name] = bound
        else:
            archived[name] = bound
    require(record(context["sources"]["controller_status"]) == original["controller_status"],
            "Actual closure changed during archive")
    for name in IMMUTABLE_TOKENS:
        require(record(context["sources"][name]) == original[name], "Immutable science token changed: "+name)
    receipt.update(artifacts=archived, archived_immutable_scientific_tokens=backups,
        archive_copies_byte_identical=True, archive_directory=str(archive))
    validate_receipt(receipt, controller["run_id"])
    write_new(args.receipt_output, receipt)
    return {"receipt": record(args.receipt_output), "run_id": controller["run_id"],
        "phase2_updates": n, "approved_total_phase2_updates": total, "archive_created": True,
        "original_scientific_tokens_paths_preserved": True, "actual_exact_resume_launched": False}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-loop", type=Path, required=True)
    result.add_argument("--approved-total-phase2-updates", type=int, required=True)
    result.add_argument("--allow-stopping-bound-extension", action="store_true")
    result.add_argument("--owned-session-id", type=int, required=True)
    result.add_argument("--owned-exit-chunk", required=True)
    result.add_argument("--owned-exit-observed-at", required=True)
    result.add_argument("--archive-dir", type=Path)
    result.add_argument("--receipt-output", type=Path)
    result.add_argument("--validate-only", action="store_true")
    result.add_argument("--validation-report", type=Path)
    return result


def main(args):
    try:
        result = make_receipt(args)
        result.update(passed=True, checkpoint_byte_reads=CHECKPOINT_BYTE_READS,
            torch_imported="torch" in sys.modules)
        code = 0
    except (ValueError, KeyError, TypeError, OSError, AttributeError) as error:
        result = {"kind": "ROOT_closed_own_phase2_exact_archive_rejection_v1", "passed": False,
            "error": str(error), "checkpoint_byte_reads": CHECKPOINT_BYTE_READS,
            "torch_imported": "torch" in sys.modules, "archive_or_receipt_admitted": False}
        code = 2
    if args.validation_report:
        write_new(args.validation_report, result)
    print(json.dumps(result, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
