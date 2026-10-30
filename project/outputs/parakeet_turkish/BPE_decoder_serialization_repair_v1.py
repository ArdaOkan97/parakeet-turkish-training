"""Narrow, explicit recovery of the observed integer-key journal failure.

This is not the normal CLOSED0 continuation path. The failed execution and its
uncheckpointed journal suffix remain evidence; only the saved model state resumes.
"""
from __future__ import annotations
import json
from pathlib import Path

from atomic_io_retry_v1 import read_json
from tsc_bpe_full_decoder_transition_v1 import STORAGE, SOURCE, checked, record, require

RUN = "adaptive_20261015_TR_BPE512_full_decoder_02"
KIND = "BPE_observed_JSON_key_failure_exact_checkpoint_repair_v1"
ALLOWED_CHANGES = {"train_lora_batched_tsc_bpe_full_decoder_v1.py",
                   "run_TSC_bpe_full_decoder_v1.py", Path(__file__).name}


def validate(path, *, run_id, checkpoint, current_pins):
    path = Path(path).resolve()
    require(path.is_relative_to(STORAGE.resolve()), "Repair evidence must remain on E:")
    value = read_json(path)
    require(value["kind"] == KIND and value["run_id"] == run_id == RUN
            and value["actual_owned_exit"] == {"session": 69509, "exit_code": 1, "tool_chunk": "56e322"}
            and value["saved_step"] == 62500 and value["observed_unsaved_step"] == 62750
            and value["discarded_uncheckpointed_updates"] == 250,
            "Only the actual observed JSON publication failure may use this recovery")
    failed = read_json(checked(value["failed_controller"]))
    native = read_json(checked(value["failed_native"]))
    require(failed["status"] == "failed" and failed["child_exit_code"] == 1
            and failed["GPU_running"] is False and failed["published_step"] == 62750
            and native["status"] == "failed" and native["sources_unchanged"] is True
            and native["error"] == "JournalError('Journal metrics require JSON string keys')",
            "Actual failed native/controller evidence differs; never synthesize CLOSED0")
    checked(value["saved_checkpoint"])
    require(record(checkpoint) == value["saved_checkpoint"], "Exact saved checkpoint bytes changed")
    prior = read_json(checked(value["original_run_config"]))
    require(value["CPU_checkpoint_facts"] == {"step": 62500, "weights": 207, "history_rows": 62500,
            "optimizer_updates": 250, "sampler_updates": 250, "retention_failures": 1,
            "selected_step": 62250, "CUDA_initialized": False}, "Saved-state CPU audit differs")
    require(prior["config_identity"]["source_checkpoint"] == failed["source_checkpoint"],
            "Original coherent parent changed")
    for name, digest in failed["source_sha256"].items():
        if name not in ALLOWED_CHANGES:
            require(current_pins[name] == digest, "Scientific dependency changed during serialization repair: " + name)
    for name, ref in value["original_sources"].items():
        require(record(checked(ref))["sha256"] == failed["source_sha256"][name], "Archived original source differs")
    require(set(current_pins) - set(failed["source_sha256"]) <= {Path(__file__).name},
            "Unexpected new runtime dependency")
    return value, prior


def rebind(path, args, context, own, config, source_pins):
    """After full old-identity validation, bind operational source changes only."""
    native = read_json(args.native_backend_provenance)
    value, original = validate(path, run_id=args.run_id, checkpoint=args.resume_exact,
                               current_pins=native["source_sha256"])
    require(config == original and own["step"] == 62500
            and own["optimizer"]["updates"] == 250
            and own["sampler"]["decoder_adaptation_updates"] == 250,
            "Operational repair cannot reset saved clocks or replace original admission")
    identity = dict(config["config_identity"])
    identity["source_sha256"] = source_pins
    identity["runtime_serialization_repair"] = record(path)
    result = dict(config)
    result["config_identity"] = identity
    result["runtime_serialization_repair"] = {
        "receipt": record(path), "original_run_config": value["original_run_config"],
        "saved_step": 62500, "unsaved_failed_frontier": 62750,
        "uncheckpointed_updates_to_replay": 250, "normal_CLOSED0_claimed": False,
        "weights_Adam_RNG_sampler_history_counters_preserved": True,
        "only_runtime_source_identity_rebound_after_original_state_validation": True}
    return result, identity
