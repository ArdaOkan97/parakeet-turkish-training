"""Archive an actually closed source for a separately declared BPE trial.

This is a stdlib receipt writer, not a checkpoint tensor auditor or launcher.
ROOT must supply an exit observed from its owned tool handle. Running or failed
controllers are rejected before any checkpoint bytes are read. Archival copies
keep later source continuation from invalidating the parked source optimizer.
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
PARENT_RUN = "adaptive_20261011_hf_TSC_headlr02_01"
LANGUAGES = ("tr", "en", "es", "it", "fr", "de")
RETENTION_MAX = {"en": 69, "es": 52, "it": 30, "fr": 74, "de": 70}
CHECKPOINT_BYTE_READS = 0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


def scoped(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), "Artifact must stay in the task workspace")
    return path


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
            "Artifact changed during hashing: " + str(path))
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": after.st_size}


def write_new(path, value):
    path = scoped(path)
    require(not path.exists(), "Receipt must have a fresh path")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def inspect_closed(loop, resource_stop_review=None):
    """The liveness gate intentionally precedes all model/checkpoint reads."""
    status_path = scoped(loop) / "loop_status.json"
    state = read(status_path)
    require(state.get("run_id") == PARENT_RUN, "Only the declared head-LR source is admitted")
    ordinary = (state.get("status") == "train_window_ended"
                and state.get("child_exit_code") == 0
                and state.get("resource_review_required") is False
                and state.get("stop_category") in ("budget", "retention"))
    if not ordinary:
        require(resource_stop_review is not None,
                "Source controller is not successfully CLOSED; checkpoint bytes were not read")
        review = read(resource_stop_review)
        require(state.get("status") == "resource_review_required"
                and state.get("child_exit_code") == 0 and state.get("resource_review_required") is True
                and state.get("stop_category") == "resource"
                and state.get("cooperative_stop_reason") == "system_commit_at_or_above95percent"
                and review.get("kind") == "ROOT_headlr_memory_resource_stop_review_v1"
                and review.get("passed") is True and review.get("source_run_id") == PARENT_RUN
                and review.get("source_controller_sha256") == state.get("controller_sha256")
                and review.get("allow_BPE512_row_only_acquisition") is True
                and review.get("guards_retained") is True
                and review.get("reviewed_status_artifact") == record(status_path),
                "Exact ROOT-reviewed critical-commit CLOSED child0 source required; no checkpoint bytes read")
    else:
        require(resource_stop_review is None, "Ordinary closure must not borrow a resource review")
    require(state.get("finished_at_utc") and state.get("actual_runtime_completion"),
            "Actual native completion and finished controller required")
    run = HERE / "runs" / PARENT_RUN
    config = read(run / "run_config.json")
    selection = read(run / "selection.json")
    history = read(run / "training_history.json")
    selected, completed = selection.get("selected_step"), selection.get("completed_steps")
    require(type(selected) is int and type(completed) is int and 1500 <= selected <= completed,
            "Dynamic selected and completed update boundaries required")
    require(state.get("selected_step") == selected and state.get("completed_steps") == completed,
            "Controller and durable selection boundaries differ")
    require(config.get("run_id") == PARENT_RUN and selection.get("run_id") == PARENT_RUN,
            "Config and selection source identities differ")
    require(isinstance(history, list) and history and history[-1].get("step") == completed,
            "Closed history must end at the completed boundary")
    rows = [row for row in history if row.get("step") == selected and row.get("validation")]
    require(len(rows) == 1 and rows[0]["validation"].get("retention_gate_passed") is True,
            "Selected source must be an actual all-five qualifying development point")
    recipe = config["config_identity"]["training_arguments"]
    require(recipe.get("batch_size") == 8 and recipe.get("accumulation") == 2
            and recipe.get("sampling_mode") == "domain30" and recipe.get("head_lr") == 1e-5
            and recipe.get("lora_lr") == 3e-4 and recipe.get("replay_kl") == 10.0,
            "Reviewed B8/domain30/head-LR source recipe required")
    dev = HERE / "results" / f"{PARENT_RUN}_step_{selected}"
    summary = read(dev / "summary.json")
    scores = {language: summary[f"validation_{language}"]["asr"] for language in LANGUAGES}
    require(all(score.get("utterances") == 50 for score in scores.values()),
            "Complete fixed six-language development point required")
    require(all(scores[language]["word_errors"] <= limit for language, limit in RETENTION_MAX.items()),
            "Selected source violates an existing-language retention gate")
    require(scores["tr"]["word_errors"] / scores["tr"]["reference_words"]
            == rows[0]["validation"]["turkish_wer"], "Selected history and saved DEV differ")
    paths = state["paths"]
    runtime = scoped(paths.get("fresh_runtime", paths["native"]))
    completion = runtime.with_suffix(".completion.json")
    native_done, native_ready, identity = read(completion), read(runtime), read(paths["identity"])
    require(native_done.get("status") == "complete" and native_done.get("run_id") == PARENT_RUN
            and native_done.get("sources_unchanged") is True and not native_done.get("error"),
            "Successful unchanged native-runtime completion required")
    require(native_done.get("pid") == native_ready.get("pid") == identity.get("pid")
            and identity.get("expected_run_id") == PARENT_RUN,
            "Completed actual worker identities differ")
    sources = {
        "best_checkpoint": ROOT / "work" / f"{PARENT_RUN}_best.pt",
        "latest_checkpoint": ROOT / "work" / f"{PARENT_RUN}_latest.pt",
        "run_config": run / "run_config.json", "training_history": run / "training_history.json",
        "selection": run / "selection.json", "controller_status": status_path,
        "runtime_completion": completion, "fresh_runtime": runtime,
        "owned_worker_identity": scoped(paths["identity"]),
        "DEV_summary": dev / "summary.json", "DEV_metadata": dev / "metadata.json",
        **{f"DEV_{language}": dev / f"validation_{language}.jsonl" for language in LANGUAGES},
    }
    if resource_stop_review is not None:
        sources["resource_stop_review"] = scoped(resource_stop_review)
    require(scoped(selection["weights"]) == sources["best_checkpoint"],
            "Selection must point to this source BEST, never LATEST")
    for name, digest in state["source_sha256"].items():
        require(record(HERE / name)["sha256"] == digest, "Closed runtime source changed: " + name)
    return state, config, selection, history, scores, sources


def make_receipt(args):
    state, config, selection, history, scores, sources = inspect_closed(args.source_loop, args.resource_stop_review)
    require(type(args.owned_session_id) is int and args.owned_session_id > 0
            and re.fullmatch(r"[0-9a-f]+", args.owned_exit_chunk or "")
            and args.owned_exit_observed_at,
            "ROOT must supply an actually observed owned CLOSED exit0 tool receipt")
    observed = datetime.fromisoformat(args.owned_exit_observed_at.replace("Z", "+00:00"))
    finished = datetime.fromisoformat(state["finished_at_utc"].replace("Z", "+00:00"))
    require(observed.tzinfo is not None and observed >= finished,
            "Owned exit observation must follow the completed controller")
    if args.resource_stop_review is None:
        require(args.owned_exit_code == 0, "Ordinary controller must have an actual exit0")
    else:
        reviewed_exit = read(args.resource_stop_review)["owned_actual_exit"]
        require(args.owned_exit_code == reviewed_exit.get("exit_code") == 1
                and args.owned_session_id == reviewed_exit.get("session_id")
                and args.owned_exit_chunk == reviewed_exit.get("chunk_id", reviewed_exit.get("tool_chunk"))
                and observed == datetime.fromisoformat(reviewed_exit["observed_at_utc"].replace("Z", "+00:00")),
                "Reviewed actual controller exit1/chunk/session/time must remain truthful")
    require(re.fullmatch(r"[A-Za-z0-9_-]+", args.target_run_id or "") is not None,
            "Safe explicit target run ID required")
    from tsc_bpe_acquisition_transition_v1 import acquisition_declaration
    declaration = acquisition_declaration()
    original = {name: record(path) for name, path in sources.items()}
    require(read(sources["runtime_completion"]).get("fresh_runtime_sha256")
            == original["fresh_runtime"]["sha256"], "Native completed runtime bytes differ")
    for name in ("io", "cache", "memory"):
        if name in state["paths"]:
            value = read(state["paths"][name])
            require(value.get("status") == "complete" and value.get("sources_unchanged") is True
                    and not value.get("error"), "Runtime guard did not finish successfully: " + name)
    if args.validate_only:
        return {"kind": "ROOT_closed_source_archive_preflight_v1", "passed": True,
                "archive_created": False, "selected_step": selection["selected_step"],
                "completed_steps": selection["completed_steps"], "original_artifacts": original,
                "tensor_state_audited": False}
    archive = scoped(args.archive_dir)
    require(archive.is_relative_to(ROOT / "work" / "closed-sources") and not archive.exists(),
            "Use a fresh archive under work/closed-sources")
    require(args.receipt_output and not scoped(args.receipt_output).exists(), "Fresh receipt output required")
    archive.mkdir(parents=True, exist_ok=False)
    archived = {}
    for name, source in sources.items():
        destination = archive / (name + Path(source).suffix)
        shutil.copy2(source, destination)
        bound = record(destination)
        require((bound["sha256"], bound["bytes"]) == (original[name]["sha256"], original[name]["bytes"])
                and record(source) == original[name], "Source or archival copy changed: " + name)
        archived[name] = bound
    # Recheck closure metadata after archival; no new trainer is dispatched here.
    require(record(sources["controller_status"]) == original["controller_status"],
            "Controller source changed during archival")
    receipt = {
        "schema": 1, "kind": "ROOT_owned_closed_TSC_best_for_BPE512_acquisition_v1",
        "status": "actual_CLOSED_qualified_best", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_id": PARENT_RUN, "target_run_id": args.target_run_id,
        "scientific_config_identity_sha256": canonical(config["config_identity"]),
        "selected_step": selection["selected_step"], "completed_steps": selection["completed_steps"],
        "stop_reason": selection["stop_early_reason"], "source_stop_category": state["stop_category"],
        "retention_gate_passed_at_selected_step": True, "retention_max_word_errors": RETENTION_MAX,
        "selected_DEV_asr": scores, "allow_BPE512_row_only_acquisition": True,
        "best_and_latest_state_must_not_be_mixed": True,
        "expected_best_sampler_boundary_step": selection["selected_step"],
        "expected_latest_sampler_boundary_step": selection["completed_steps"],
        "artifacts": archived, "original_artifact_paths": {name: value["path"] for name, value in original.items()},
        "original_artifact_records": original,
        "owned_actual_exits": [{"session_id": args.owned_session_id, "exit_code": args.owned_exit_code,
                                "tool_chunk": args.owned_exit_chunk, "observed_at_utc": observed.isoformat(),
                                "role": "trainer_controller"}],
        "source_runtime_pid": read(sources["fresh_runtime"])["pid"],
        "source_runtime_ppid": read(sources["fresh_runtime"])["ppid"],
        "declared_acquisition": declaration,
        "receipt_creator": record(__file__), "archive_copies_byte_identical": True,
        "independent_checkpoint_state_audit_in_this_receipt": False,
        "torch_imported": "torch" in sys.modules, "actual_GPU_or_training_activation": False,
        "limits": "Owned exit is ROOT-supplied tool evidence. Future worker must independently audit actual207 tensors/Adam/sampler/RNG before expansion; this writer only binds closed metadata and byte-identical archival artifacts.",
    }
    if args.resource_stop_review is not None:
        receipt["resource_stop_review"] = archived["resource_stop_review"]
        receipt["source_status_remains_resource_review_required"] = True
    write_new(args.receipt_output, receipt)
    return {"receipt": record(args.receipt_output), "selected_step": selection["selected_step"],
            "completed_steps": selection["completed_steps"], "archive_created": True}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-loop", type=Path, required=True)
    result.add_argument("--archive-dir", type=Path)
    result.add_argument("--receipt-output", type=Path)
    result.add_argument("--target-run-id")
    result.add_argument("--owned-session-id", type=int)
    result.add_argument("--owned-exit-code", type=int, choices=(0, 1), default=0)
    result.add_argument("--owned-exit-chunk")
    result.add_argument("--owned-exit-observed-at")
    result.add_argument("--resource-stop-review", type=Path)
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
        result = {"kind": "ROOT_closed_source_archive_rejection_v1", "passed": False,
                  "error": str(error), "checkpoint_byte_reads": CHECKPOINT_BYTE_READS,
                  "torch_imported": "torch" in sys.modules,
                  "archive_or_receipt_admitted": False}
        if args.validation_report:
            write_new(args.validation_report, result)
        print(json.dumps(result, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
