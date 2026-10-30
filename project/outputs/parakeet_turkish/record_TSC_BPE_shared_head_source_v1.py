"""ROOT-only archive of a closed phase1 screen for declared BPE512-row plus12 shared-head adaptation.

This stdlib writer binds an explicit chosen checkpoint to its own DEV/runtime
frontier. It does not deserialize weights, admit a separate one-update smoke,
infer process closure, or activate training. ROOT supplies the owned exit0 tool
observation. The original207 parked artifact keeps its exact immutable path.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LANGUAGES = ("tr", "en", "es", "it", "fr", "de")
RETENTION_MAX = {"en": 69, "es": 52, "it": 30, "fr": 74, "de": 70}
CHECKPOINT_BYTE_READS = 0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scoped(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), "Artifact must remain in this workspace")
    return path


def read(path):
    return json.loads(scoped(path).read_text(encoding="utf-8-sig"))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


def record(path):
    global CHECKPOINT_BYTE_READS
    path = scoped(path)
    before = path.stat()
    if path.suffix == ".pt":
        CHECKPOINT_BYTE_READS += 1
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "Artifact changed while hashing: " + str(path))
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": after.st_size}


def write_new(path, value):
    path = scoped(path)
    require(not path.exists(), "Output must have a fresh path")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def inspect_closed(loop, role):
    """The actual closure metadata gate precedes every .pt byte read."""
    status_path = scoped(loop) / "loop_status.json"
    controller = read(status_path)
    require(controller.get("kind") == "guarded_TSC_BPE512_new_rows_only_acquisition_controller_v1"
            and controller.get("status") == "train_window_ended"
            and controller.get("child_exit_code") == 0
            and controller.get("resource_review_required") is False,
            "Source is not a successfully CLOSED acquisition screen; no checkpoint bytes read")
    require(controller.get("smoke_updates") in (None, 0),
            "A separate one-update smoke is not a phase2 scientific source")
    require(controller.get("stop_category") in ("phase_bound", "budget", "retention")
            and controller.get("finished_at_utc") and controller.get("actual_runtime_completion"),
            "Natural main-screen closure and actual native completion required")
    run_id = controller.get("run_id")
    require(isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,110}", run_id),
            "Safe actual acquisition run ID required")
    run = HERE / "runs" / run_id
    config, selection, history = (read(run / name) for name in
                                 ("run_config.json", "selection.json", "training_history.json"))
    original, completed = config.get("source_selected_step"), selection.get("completed_steps")
    require(type(original) is int and type(completed) is int and 0 < original < completed,
            "Positive completed acquisition updates required")
    chosen = completed if role == "latest" else selection.get("selected_step")
    require(type(chosen) is int and original < chosen <= completed,
            "Chosen own state must include learned acquisition updates")
    require(controller.get("completed_steps") == completed
            and controller.get("acquisition_updates_completed") == completed - original
            and selection.get("acquisition_updates_completed") == completed - original
            and config.get("smoke_updates") in (None, 0),
            "Controller, selection, and local acquisition clocks differ")
    dev = HERE / "results" / f"{run_id}_step_{chosen}"
    summary = read(dev / "summary.json")
    scores = {lang: summary[f"validation_{lang}"]["asr"] for lang in LANGUAGES}
    require(all(score.get("utterances") == 50 and type(score.get("word_errors")) is int
                and score.get("reference_words", 0) > 0 for score in scores.values()),
            "Actual complete fixed300 development result required")
    require(all(scores[lang]["word_errors"] <= limit for lang, limit in RETENTION_MAX.items()),
            "Chosen state fails an existing-language retention gate")
    paths = controller["paths"]
    native = scoped(paths.get("fresh_runtime", paths["native"]))
    completion = native.with_suffix(".completion.json")
    ready, done, owned = read(native), read(completion), read(paths["identity"])
    require(done.get("status") == "complete" and done.get("sources_unchanged") is True
            and not done.get("error") and done.get("run_id") == run_id
            and done.get("pid") == ready.get("pid") == owned.get("pid"),
            "Actual successful unchanged paired native process completion required")
    for name in ("io", "cache", "memory"):
        if name in paths:
            guard = read(paths[name])
            require(guard.get("status") == "complete" and guard.get("sources_unchanged") is True
                    and not guard.get("error"), "Runtime guard failed: " + name)
    sources = {
        "best_checkpoint": scoped(selection["weights"]),
        "latest_checkpoint": scoped(selection["latest_weights"]),
        "run_config": run / "run_config.json", "training_history": run / "training_history.json",
        "selection": run / "selection.json", "controller_status": status_path,
        "fresh_runtime": native, "runtime_completion": completion,
        "owned_worker_identity": scoped(paths["identity"]),
        "source_initial300_proof": run / "TSC_BPE_acquisition_initial_source_equivalence.json",
        "DEV_summary": dev / "summary.json", "DEV_metadata": dev / "metadata.json",
        **{f"DEV_{lang}": dev / f"validation_{lang}.jsonl" for lang in LANGUAGES},
        "original_source_receipt": scoped(config["source_receipt"]["path"]),
    }
    sources["chosen_checkpoint"] = sources[role + "_checkpoint"]
    parked = config["config_identity"]["acquisition_optimizer"]["parked_source_artifact"]
    require(scoped(parked["path"]).is_relative_to(ROOT / "work" / "closed-sources"),
            "Original207 source must be an immutable archival artifact")
    return controller, config, selection, history, summary, scores, chosen, sources, parked


def make_receipt(args):
    controller, config, selection, history, summary, scores, chosen, sources, parked = inspect_closed(
        args.source_loop, args.chosen_checkpoint_role)
    require(type(args.owned_session_id) is int and args.owned_session_id > 0
            and re.fullmatch(r"[0-9a-f]+", args.owned_exit_chunk or "") and args.owned_exit_observed_at,
            "ROOT must supply actual owned CLOSED0 tool evidence")
    observed = datetime.fromisoformat(args.owned_exit_observed_at.replace("Z", "+00:00"))
    finished = datetime.fromisoformat(controller["finished_at_utc"].replace("Z", "+00:00"))
    require(observed.tzinfo is not None and observed >= finished, "Exit observation must follow closure")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,110}", args.target_run_id or "")
            and args.target_run_id != controller["run_id"], "Distinct safe new target run required")
    from tsc_bpe_shared_head_transition_v1 import shared_head_declaration, validate_receipt, validate_chosen_selection
    original = {name: record(path) for name, path in sources.items()}
    require(record(parked["path"]) == parked, "Original207 immutable parked bytes changed")
    require(original["original_source_receipt"] == config["source_receipt"],
            "Original acquisition receipt bytes changed")
    require(read(sources["runtime_completion"])["backend_ready_sha256"] == original["fresh_runtime"]["sha256"],
            "Completed native startup bytes differ")
    original_step = config["source_selected_step"]
    receipt = {
        "schema": 1, "kind": "ROOT_owned_closed_TSC_BPE_phase1_for_shared_heads_v1",
        "status": "actual_CLOSED_qualified_phase1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_id": controller["run_id"], "target_run_id": args.target_run_id,
        "scientific_config_identity_sha256": canonical(config["config_identity"]),
        "original_step": original_step, "acquisition_updates": chosen - original_step,
        "chosen_step": chosen, "chosen_checkpoint_role": args.chosen_checkpoint_role,
        "completed_steps": selection["completed_steps"], "phase1_localBEST_step": selection["selected_step"],
        "source_stop_category": controller["stop_category"], "source_stop_reason": selection.get("stop_early_reason"),
        "retention_gate_passed_at_chosen_step": True, "retention_max_word_errors": RETENTION_MAX,
        "chosen_DEV_asr": scores, "allow_shared_head_adaptation": True,
        "chosen_weights_optimizer_runtime_sampler_and_clocks_must_not_be_mixed": True,
        "original_champion_separately_parked": True, "original207_artifact_path_preserved": True,
        "artifacts": {**original, "original_source_checkpoint": parked},
        "original_artifact_paths": {name: value["path"] for name, value in original.items()},
        "original_artifact_records": original,
        "owned_actual_exits": [{"role": "trainer_controller", "session_id": args.owned_session_id,
                                "exit_code": 0, "tool_chunk": args.owned_exit_chunk,
                                "observed_at_utc": observed.isoformat()}],
        "source_runtime_pid": read(sources["fresh_runtime"])["pid"],
        "source_runtime_ppid": read(sources["fresh_runtime"])["ppid"],
        "declared_shared_heads": shared_head_declaration(args.row_lr, args.shared_lr), "receipt_creator": record(__file__),
        "source_phase2_latest_used": False, "encoder_LoRA_active": False,
        "source_row_lr_audited": 1e-4, "active_row_lr": args.row_lr, "active_shared_lr": args.shared_lr,
        "phase1_three_table_moments_and_K_clock_retained": True, "active_parameter_tensors": 15, "active_shared_head_tensors": 12, "frozen_encoder_LoRA_tensors": 192,
        "source_shared_head_lr_audited": 1e-5, "shared12_original_moments_and_S_clock_retained": True,
        "checkpoint_tensors_audited_here": False, "acquired_learning_tensor_evidence_pending": True,
        "torch_imported": "torch" in sys.modules, "actual_GPU_or_training_activation": False,
        "limits": "ROOT-supplied owned exit and closed JSON/byte evidence only; future worker independently checks actual207/phase1 tensors, acquired row changes/moments, both Adam clocks and source300 before updates.",
    }
    validate_receipt(receipt, args.target_run_id, args.row_lr, args.shared_lr)
    validate_chosen_selection(receipt, config, selection, history, summary)
    if args.validate_only:
        return {"kind": "ROOT_closed_phase1_to_shared_heads_source_preflight_v1", "passed": True,
                "chosen_step": chosen, "archive_created": False, "original_artifacts": original,
                "original207_artifact_preserved": parked, "checkpoint_tensors_audited": False}
    archive = scoped(args.archive_dir)
    require(archive.is_relative_to(ROOT / "work" / "closed-sources") and not archive.exists(),
            "Fresh work/closed-sources archive required")
    require(args.receipt_output and not scoped(args.receipt_output).exists(), "Fresh receipt output required")
    archive.mkdir(parents=True, exist_ok=False)
    archived = {}
    for name, source in sources.items():
        if name == "chosen_checkpoint":
            continue
        destination = archive / (name + source.suffix)
        shutil.copy2(source, destination)
        bound = record(destination)
        require((bound["sha256"], bound["bytes"]) == (original[name]["sha256"], original[name]["bytes"])
                and record(source) == original[name], "Source or copy changed: " + name)
        archived[name] = bound
    archived["chosen_checkpoint"] = archived[args.chosen_checkpoint_role + "_checkpoint"]
    archived["original_source_checkpoint"] = parked
    require(record(sources["controller_status"]) == original["controller_status"], "Closure changed during archive")
    receipt["artifacts"] = archived
    receipt["archive_copies_byte_identical"] = True
    validate_chosen_selection(receipt, config, selection, history, summary)
    write_new(args.receipt_output, receipt)
    return {"receipt": record(args.receipt_output), "chosen_step": chosen,
            "acquisition_updates": chosen - original_step, "archive_created": True,
            "original207_artifact_path_preserved": True}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-loop", type=Path, required=True)
    result.add_argument("--chosen-checkpoint-role", choices=("latest", "best"), required=True)
    result.add_argument("--row-lr", type=float, choices=(1e-3,3e-4), default=1e-3)
    result.add_argument("--shared-lr", type=float, choices=(1e-5,3e-6), default=1e-5)
    result.add_argument("--archive-dir", type=Path)
    result.add_argument("--receipt-output", type=Path)
    result.add_argument("--target-run-id")
    result.add_argument("--owned-session-id", type=int)
    result.add_argument("--owned-exit-chunk")
    result.add_argument("--owned-exit-observed-at")
    result.add_argument("--validate-only", action="store_true")
    result.add_argument("--validation-report", type=Path)
    return result


def main(args):
    try:
        result = make_receipt(args)
        result.update(passed=True, checkpoint_byte_reads=CHECKPOINT_BYTE_READS,
                      torch_imported="torch" in sys.modules)
        if args.validation_report:
            write_new(args.validation_report, result)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        result = {"kind": "ROOT_closed_phase1_to_shared_heads_archive_rejection_v1", "passed": False,
                  "error": str(error), "checkpoint_byte_reads": CHECKPOINT_BYTE_READS,
                  "torch_imported": "torch" in sys.modules, "archive_or_receipt_admitted": False}
        if args.validation_report:
            write_new(args.validation_report, result)
        print(json.dumps(result, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
