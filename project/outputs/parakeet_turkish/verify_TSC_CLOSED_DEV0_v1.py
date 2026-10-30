"""Read-only external gate for a ROOT-observed CLOSED TSC DEV0 hold.

Compare all six fixed validation sets, model/evaluation fingerprints, raw
transcripts, emitted token/duration paths, independently rescored ASR counts
and all five original retention gates. No model/checkpoint/corpus/audio reads.
Only after this gate passes may ROOT choose SAME-run exact resume from step0.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import run_hf_TSC_DEV0_hold_v1 as hold
from metrics import score_transcriptions

C = hold.controller
HERE = Path(__file__).resolve().parent
LANGUAGES = ("tr", "en", "es", "it", "fr", "de")
IDENTITY_KEYS = ("source_id", "sentence_id", "language", "split", "reference",
                 "dataset_revision", "audio_sha256", "wav_sha256", "duration")
TOKEN_KEYS = ("token_id", "duration_id", "piece", "start_frame", "duration_frames")
METRICS_SHA = "433c31d9cbc4975c42d3a0fe659cd0bbd6c1caf2e1eb3d85169baafec7f61047"


def read_record(item):
    path = C.scoped(item["path"])
    C.require(C.record(path) == item, "Saved evidence changed: " + str(path))
    return C.read(path, {})


def rows(path):
    path = C.scoped(path)
    C.require(path.name.startswith("validation_") and path.suffix == ".jsonl",
              "Only saved validation predictions may be read")
    with path.open(encoding="utf-8") as stream:
        result = [json.loads(line) for line in stream if line.strip()]
    C.require(len(result) == 50, "Exactly 50 fixed DEV rows required")
    return result


def emission_path(row):
    confidence = row.get("confidence", {})
    C.require(confidence.get("text") == row.get("hypothesis")
              and isinstance(confidence.get("tokens"), list), "Actual emitted path/text missing")
    result = []
    for token in confidence["tokens"]:
        C.require(all(key in token for key in TOKEN_KEYS), "Complete token/duration path required")
        C.require(type(token["token_id"]) is int and 0 <= token["token_id"] < 8199
                  and token["token_id"] != 8192 and type(token["duration_id"]) is int
                  and 0 <= token["duration_id"] <= 4 and type(token["start_frame"]) is int
                  and token["start_frame"] >= 0 and type(token["duration_frames"]) is int
                  and token["duration_frames"] == token["duration_id"]
                  and isinstance(token["piece"], str), "Unsafe original8199 token/duration path")
        result.append(tuple(token[key] for key in TOKEN_KEYS))
    return result


def compare_language(source, candidate, source_asr, candidate_asr, language):
    C.require(len(source) == len(candidate) == 50, "Fixed six-by-50 DEV required")
    for label, items in (("source", source), ("candidate", candidate)):
        C.require(len({row["source_id"] for row in items}) == 50,
                  "Duplicate fixed source in " + label)
    token_count = 0
    for index, (before, after) in enumerate(zip(source, candidate)):
        C.require(all(key in before and key in after and before[key] == after[key] for key in IDENTITY_KEYS)
                  and before["language"] == language and before["split"] == "validation",
                  f"DEV ordered identity/reference differs: {language}/{index}")
        C.require(type(before.get("hypothesis")) is str and before["hypothesis"] == after.get("hypothesis"),
                  f"Raw DEV transcript differs: {language}/{index}")
        before_path, after_path = emission_path(before), emission_path(after)
        C.require(before_path == after_path, f"Emitted token/duration path differs: {language}/{index}")
        token_count += len(before_path)
    scored = [{key: value for key, value in score_transcriptions(items, language=language).items()
               if key != "details"} for items in (source, candidate)]
    C.require(scored[0] == scored[1] == source_asr == candidate_asr,
              "Independent ASR rescore/saved counts differ: " + language)
    return {"utterances": 50, "exact_transcripts": 50, "exact_emitted_paths": 50,
            "emitted_tokens_compared": token_count, "asr": scored[0]}


def retention(summary, baseline):
    result = {}
    for language in LANGUAGES[1:]:
        source = baseline["validation_" + language]["asr"]["wer"]
        current = summary["validation_" + language]["asr"]["wer"]
        limit = source + max(.005, .1 * source)
        C.require(current <= limit, "Original retention gate fails: " + language)
        result[language] = {"wer": current, "baseline_wer": source,
                            "maximum_allowed_wer": limit, "passed": True}
    return result


def verify(args):
    C.require(args.root_owned_controller_exit_confirmed,
              "ROOT must first observe the owned hold controller exit0")
    hold.frozen_sources()
    C.require(C.sha(HERE / "metrics.py") == METRICS_SHA, "Pinned scoring code changed")
    loop = C.scoped(args.source_loop)
    C.require(loop.parent == HERE / "runs", "Actual first-arm loop directory required")
    status_path = loop / "loop_status.json"; status_before = C.record(status_path)
    state = C.read(status_path, {}); hold.require_closed_zero(state)
    C.require(state["loop_id"] == loop.name and state["run_id"] == loop.name + "_TSC",
              "Actual first-loop/run namespace differs")
    request = read_record(state["ROOT_DEV0_hold"])
    C.require(Path(state["ROOT_DEV0_hold"]["path"]).resolve() == loop / hold.HOLD_NAME
        and request.get("kind") == "root_requested_TSC_DEV0_before_dispatch_hold_v1"
        and request.get("run_id") == state["run_id"] and request.get("loop_id") == state["loop_id"]
        and request.get("wrapper") == C.record(HERE / "run_hf_TSC_DEV0_hold_v1.py")
        and request.get("original_controller_sha256") == hold.CONTROLLER_SHA
        and request.get("routing_receipt") == C.record(C.CHAIN_PATH)
        and request.get("expected_completed_updates") == 0
        and request.get("hold_created_before_run_arm_dispatch") is True
        and request.get("original_worker_arguments_sha256") == C.guards.argument_sha(state["worker_arguments"]),
        "Actual before-dispatch ROOT hold provenance differs")
    C.require(request["stop_file"] == C.record(Path(state["paths"]["stop"]))
        and Path(state["paths"]["stop"]).read_text(encoding="utf-8") == hold.HOLD_TEXT,
        "Original hold stop file changed")
    closure = read_record(request["source_closure"])
    C.require(request["source_closure"] == state["source_identity"]["closure"]
        and request["source_checkpoint"] == state["source_identity"]["source_checkpoint"] == closure["source_checkpoint"]
        and closure.get("kind") == "root_owned_TSC_domain30_closed_source_v3"
        and closure.get("passed") is True and closure.get("root_observed_owned_session_completed") is True
        and closure.get("child_exit_code") == 0, "Original ROOT CLOSED qualifying source differs")
    source_selection = read_record(closure["source_selection"])
    C.require(source_selection.get("accepted") is True
        and source_selection.get("heldout_tests_evaluated") is False
        and source_selection.get("selection_split") == "FLEURS validation only"
        and source_selection.get("run_id") == closure["source_run_id"]
        and source_selection.get("selected_step") == state["source_identity"]["source_selected_step"],
        "Selected qualifying source identity differs")
    selection = read_record(state["selection"])
    C.require(selection.get("completed_steps") == 0 and selection.get("selected_step") == 0
        and selection.get("accepted") is True and selection.get("stop_early_reason") == "controller_stop_file",
        "No optimizer update permitted before equality")
    actual_runtime = C.verify_completion(state, selection)
    C.require(actual_runtime == state["actual_runtime_completion"], "Closed runtime records differ")
    run = HERE / "runs" / state["run_id"]
    C.require(C.read(run / "training_history.json", None) == [], "Zero-update history must be empty")
    config = C.read(run / "run_config.json", {})
    source_config = read_record(closure["source_run_config"])
    warm = config["warm_start"]
    C.require(warm.get("mode") == "new_optimizer_warm_start"
        and Path(warm["path"]).resolve() == Path(closure["source_checkpoint"]["path"]).resolve()
        and warm.get("sha256") == closure["source_checkpoint"]["sha256"]
        and warm.get("source_run_id") == closure["source_run_id"]
        and warm.get("source_step") == source_selection["selected_step"]
        and config["config_identity"]["extended_tokenizer_sha256"] == C.TOKENIZER_SHA
        and config["config_identity"]["lora_layout"] == source_config["config_identity"]["lora_layout"],
        "Actually loaded source run/step/layout/tokenizer differs")
    source_summary = read_record(closure["source_development_summary"])
    source_dir = Path(closure["source_development_summary"]["path"]).resolve().parent
    C.require(source_dir == HERE / "results" / (closure["source_run_id"] + "_step_" + str(source_selection["selected_step"])),
              "Bind actual selected source DEV folder")
    candidate_dir = HERE / "results" / (state["run_id"] + "_step_0")
    summary_path = candidate_dir / "summary.json"; candidate_summary = C.read(summary_path, {})
    metadata_paths = [folder / "metadata.json" for folder in (source_dir, candidate_dir)]
    metadata = [C.read(path, {}) for path in metadata_paths]
    C.require(metadata[0].get("schema_version") == metadata[1].get("schema_version") == 1
        and metadata[0].get("model_fingerprint") == metadata[1].get("model_fingerprint")
        and isinstance(metadata[0].get("model_fingerprint", {}).get("identity", {}).get("model_state_sha256"), str)
        and metadata[0].get("manifests") == metadata[1].get("manifests"),
        "Actual model/tokenizer/generation/evaluation/sample fingerprints differ")
    comparisons, prediction_records = {}, {}
    for language in LANGUAGES:
        key = "validation_" + language
        paths = [folder / (key + ".jsonl") for folder in (source_dir, candidate_dir)]
        saved = [rows(path) for path in paths]
        C.require(metadata[0]["manifests"][key]["source_ids"] == [row["source_id"] for row in saved[0]],
                  "Actual fixed source metadata/order differs: " + language)
        comparisons[language] = compare_language(*saved, source_summary[key]["asr"], candidate_summary[key]["asr"], language)
        prediction_records[language] = {"source": C.record(paths[0]), "DEV0": C.record(paths[1])}
    C.require(source_summary["validation_tr"]["asr"]["wer"] == source_selection["turkish_validation_wer"]
        == selection["turkish_validation_wer"] == C.read(run / "initial_development.json", {})["turkish_wer"],
        "Actual source/selected0/initial gate TR scores differ")
    baseline_path = HERE / "results/base_v3/summary.json"; baseline = C.read(baseline_path, {})
    gates = retention(candidate_summary, baseline); retention(source_summary, baseline)
    C.require(C.read(run / "initial_development.json", {})["retention_gate_passed"] is True,
              "Actual DEV0 gate must pass")
    C.require(C.record(status_path) == status_before, "Closed loop changed during read-only comparison")
    return {"schema": 1, "kind": "actual_CLOSED_TSC_DEV0_source_equality_verification_v1",
        "passed": True, "run_id": state["run_id"], "loop_id": state["loop_id"],
        "completed_updates": 0, "selected_step": 0, "compared_fixed_DEV_utterances": 300,
        "all_raw_transcripts_and_emitted_token_duration_paths_exact": True,
        "all_six_independently_rescored_ASR_metrics_exact": True,
        "model_evaluation_and_manifest_fingerprints_exact": True, "retention_gates": gates,
        "comparison": comparisons, "predictions": prediction_records,
        "provenance": {"verifier": C.record(__file__), "hold_request": state["ROOT_DEV0_hold"],
            "closed_loop_status": status_before, "source_closure": request["source_closure"],
            "source_selected_checkpoint": request["source_checkpoint"],
            "source_summary": closure["source_development_summary"], "DEV0_summary": C.record(summary_path),
            "source_metadata": C.record(metadata_paths[0]), "DEV0_metadata": C.record(metadata_paths[1]),
            "baseline_summary": C.record(baseline_path), "actual_runtime_completion": actual_runtime},
        "closure_evidence_scope": "ROOT confirms owned controller exit0; frozen controller records owned child exit0 before terminal state. Verifier does not independently query OS exits.",
        "checkpoint_or_corpus_or_audio_read": False, "model_or_GPU_execution_by_verifier": False,
        "confidence_float_equivalence_claimed": False, "heldout_TEST_evaluated": False,
        "objective_complete": False,
        "next_action": "ROOT may authorize reviewed SAME-run exact resume from saved0; keep original native-ready/config/Adam/sampler/RNG. No fresh initialization.",
        "verified_at_utc": C.now()}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-loop", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--root-owned-controller-exit-confirmed", action="store_true")
    return result


def main(args):
    output = C.scoped(args.output)
    C.require(not output.exists(), "Verification evidence must be new; no overwrites")
    try:
        result = verify(args); exit_code = 0
    except Exception as error:
        result = {"kind": "actual_CLOSED_TSC_DEV0_source_equality_verification_v1", "passed": False,
                  "error": repr(error), "source_loop": str(args.source_loop),
                  "verifier": C.record(__file__), "model_or_GPU_execution_by_verifier": False}; exit_code = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write("\n")
    print(json.dumps({"passed": result["passed"], "output": str(output),
                      "compared_fixed_DEV_utterances": result.get("compared_fixed_DEV_utterances", 0)}))
    return exit_code


if __name__ == "__main__":
    sys.exit(main(parser().parse_args()))
