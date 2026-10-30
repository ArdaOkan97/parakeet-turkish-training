"""Source-bound native completed-DEV warmstate helpers, without heavy imports.

This prepares model/Adam/Cosine reuse with NEW local Lightning loops. It does
not load tensors, restore old fit-loop counters or RNG, launch a process, or
modify a source checkpoint. A GPU owner must separately verify predecessor exit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE_ENTRY_SHA = "876772adb6377161f96e6d296cd2a67abc8fc0ed8513b2a69ffcfb46cab2bed4"
SOURCE_CONFIG_SHA = "b434edf6c69f745a5001d77a84029d26d15d12a1abc1406d3af51472b1755553"
TOTAL_RECORDINGS = 149776
HORIZON = 9361


def require(value, message):
    if not value:
        raise ValueError(message)


def scoped(value):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    require(path.is_relative_to(ROOT), "Workspace-only native boundary evidence")
    return path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _bound(path):
    path = scoped(path)
    require(path.is_file(), "Required native boundary artifact missing: " + str(path))
    return {"path": str(path), "sha256": sha(path)}


def validate_boundary_evidence(run_dir, cumulative_step, require_improved=True):
    """Describe a completed v10 DEV checkpoint; never infer it from a filename."""
    run_dir = scoped(run_dir)
    require(run_dir.parent == HERE / "runs", "Native source must be an owned workspace run")
    require(type(cumulative_step) is int and 100 < cumulative_step <= HORIZON, "Completed native cumulative step required")
    paths = {name: run_dir / name for name in ("resolved_trial_config.json", "preflight.json", "native_trial_provenance.json")}
    config, prepared, provenance = (read(paths[name]) for name in paths)
    require(config["kind"] == "full_nemo_lightning_turkish_clean_warmstart_v10"
            and prepared["run_id"] == provenance["run_id"] == run_dir.name
            and prepared["entry_sha256"] == SOURCE_ENTRY_SHA and prepared["config_sha256"] == SOURCE_CONFIG_SHA,
            "Only the explicitly frozen v10 source is supported by this helper")
    require(sha(HERE / "train_full_nemo_turkish_v10.py") == SOURCE_ENTRY_SHA
            and sha(HERE / "full_nemo_turkish_v10.json") == SOURCE_CONFIG_SHA,
            "Frozen source entry/config bytes changed")
    require(config == read(HERE / "full_nemo_turkish_v10.json"), "Source resolved recipe differs from the declared full v10 plan")
    require(prepared["data_proof_sha256"] == sha(scoped(config["data_proof"]))
            and prepared["tokenizer_manifest_sha256"] == sha(scoped(config["tokenizer_manifest"])),
            "Source clean data/tokenizer identity changed")
    require(provenance["native_Lightning_warmstart"]["source_step"] == 100
            and provenance["native_Lightning_warmstart"]["consumed_training_records"] == 1600
            and config["model"]["train_ds"]["batch_size"] == 1 and config["trainer"]["accumulate_grad_batches"] == 16,
            "Source v10 cumulative step convention changed")
    local_step = cumulative_step - 100
    gate_path = run_dir / f"development_gate_step_{cumulative_step}.json"
    marker_path = run_dir / f"recovery_checkpoint_step_{cumulative_step}.json"
    gate, marker = read(gate_path), read(marker_path)
    require(gate["step"] == marker["step"] == cumulative_step and gate["local_step"] == marker["local_step"] == local_step,
            "Native local/cumulative DEV and checkpoint metadata disagree")
    require(marker["phase"] == "completed_DEV_boundary" and marker["fresh_full_checkpoint_saved_in_this_run"] is True
            and marker["official_Lightning_checkpoint"] is True, "Use an actual post-training full DEV checkpoint")
    checkpoint = scoped(marker["path"])
    require(checkpoint.parent == run_dir and checkpoint.is_file() and sha(checkpoint) == marker["checkpoint_sha256"],
            "DEV checkpoint byte identity/scope changed")
    if require_improved:
        first_gate = read(run_dir / "development_gate_step_100.json")
        require(gate["retention_gate_passed"] is True and gate["turkish_wer"] < first_gate["turkish_wer"],
                "Quality continuation requires all five gates and improvement over trained source100")
    train = scoped(config["model"]["train_ds"]["manifest_filepath"])
    require(prepared["full_stream_audit"]["training_records"] == TOTAL_RECORDINGS
            and sha(train) == prepared["full_stream_audit"]["full_manifest_sha256"], "Source ordered full stream changed")
    consumed = 16 * cumulative_step
    remaining = TOTAL_RECORDINGS - consumed
    require(remaining >= 0 and remaining % 16 == 0 and remaining // 16 == HORIZON - cumulative_step,
            "Ordered full-pass cursor/horizon disagrees")
    artifacts = {name: _bound(path) for name, path in paths.items()}
    artifacts.update(development_gate=_bound(gate_path), checkpoint_marker=_bound(marker_path), checkpoint=_bound(checkpoint))
    return {"schema": 1, "kind": "native_completed_DEV_boundary_warmstate_v1", "passed": True,
            "source_run_id": run_dir.name, "source_run_dir": str(run_dir), "source_checkpoint_local_step": local_step,
            "source_cumulative_optimizer_step": cumulative_step, "source_scheduler_last_epoch": cumulative_step,
            "checkpoint": artifacts["checkpoint"], "source_artifacts": artifacts,
            "ordered_training_manifest": str(train), "ordered_training_manifest_sha256": sha(train),
            "first_unseen_source_index": consumed, "consumed_recordings_committed_to_optimizer": consumed,
            "remaining_recordings": remaining, "target_local_optimizer_update_limit": remaining // 16,
            "scheduler_horizon": HORIZON, "source_physical_batch": 1, "source_accumulation": 16,
            "effective_recordings_per_optimizer_update": 16, "all_five_retention_gates_passed": gate["retention_gate_passed"],
            "turkish_development_wer": gate["turkish_wer"], "require_improvement_over_source100": require_improved,
            "weights_Adam_Cosine_reuse": True, "source_fit_loops_or_RNG_restored": False,
            "exact_random_gradient_continuation_claimed": False, "source_worker_exit_must_be_verified_by_GPU_owner": True,
            "tensor_state_metadata_verification_pending_actual_load": True,
            "helper_sha256": sha(__file__), "prepared_at_utc": datetime.now(timezone.utc).isoformat()}


def verify_completed_checkpoint_state(checkpoint, boundary):
    """Call after an actual CPU load; accepts no old-loop restart assumptions."""
    local = boundary["source_checkpoint_local_step"]
    cumulative = boundary["source_cumulative_optimizer_step"]
    require(checkpoint["global_step"] == local and len(checkpoint["state_dict"]) == 917,
            "Expected native source model/local optimizer step metadata")
    require(len(checkpoint["optimizer_states"]) == len(checkpoint["lr_schedulers"]) == 1, "One full Adam/Cosine state required")
    optimizer, scheduler = checkpoint["optimizer_states"][0], checkpoint["lr_schedulers"][0]
    require([len(group["params"]) for group in optimizer["param_groups"]] == [828, 9, 6]
            and len(optimizer["state"]) == 207, "Native Adam ordering/state coverage changed")
    require(scheduler["max_steps"] == HORIZON and scheduler["warmup_steps"] == 25
            and scheduler["last_epoch"] == cumulative and scheduler["_step_count"] == cumulative + 1
            and [group["lr"] for group in optimizer["param_groups"]] == scheduler["_last_lr"], "Native Adam/Cosine position disagrees")
    for state in optimizer["state"].values():
        step = state["step"]
        step = step.item() if hasattr(step, "item") else step
        require(step == cumulative, "Adam trainable tensor step disagrees with source cumulative updates")
    progress = checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]
    committed_microbatches = local * 16
    for scope in ("total", "current"):
        require(all(progress[scope][field] == committed_microbatches for field in ("ready", "started", "processed", "completed")),
                "This DEV checkpoint contains incomplete accumulation or unexpected loader counters")
    return {"source_checkpoint_local_step": local, "source_cumulative_optimizer_step": cumulative,
            "state_tensor_count": 917, "Adam_state_count": 207, "scheduler_last_epoch": cumulative,
            "completed_source_microbatches": committed_microbatches, "checkpoint_contains_partial_accumulation": False,
            "first_unseen_source_index": boundary["first_unseen_source_index"], "old_fit_loops_used": False}


def restore_official_optimizer_scheduler(optimizer, scheduler, source_optimizer, source_scheduler, boundary):
    require([len(group["params"]) for group in optimizer.param_groups] == [828, 9, 6]
            and scheduler.max_steps == HORIZON and scheduler.warmup_steps == 25, "Official target optimizer ordering/horizon changed")
    optimizer.load_state_dict(source_optimizer)
    scheduler.load_state_dict(source_scheduler)
    require([group["lr"] for group in optimizer.param_groups] == source_scheduler["_last_lr"]
            and scheduler.last_epoch == boundary["source_cumulative_optimizer_step"], "Native optimizer/scheduler restore did not preserve position")
    return {"source_cumulative_optimizer_step": scheduler.last_epoch,
            "optimizer_group_learning_rates": [group["lr"] for group in optimizer.param_groups],
            "optimizer_or_scheduler_step_during_restore": False, "restored_at": "official on_train_start"}


def suffix_range(dataset_length, boundary, physical_batch=1):
    require(type(physical_batch) is int and physical_batch in (1, 2, 4, 8), "Declared physical B1/B2/B4/B8 only")
    require(dataset_length == TOTAL_RECORDINGS and boundary["first_unseen_source_index"] % 16 == 0,
            "Ordered full-pass boundary required")
    indices = range(boundary["first_unseen_source_index"], dataset_length)
    require(len(indices) == boundary["remaining_recordings"] and len(indices) % 16 == 0, "Suffix recording cursor changed")
    return indices, {"physical_batch": physical_batch, "accumulation": 16 // physical_batch,
                     "effective_recordings_per_update": 16, "optimizer_updates": len(indices) // 16,
                     "physical_mean_volume_grouping_changed": physical_batch != 1,
                     "B1_gradient_equivalence_claimed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--cumulative-step", type=int, default=350)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unqualified-diagnostic-source", action="store_true")
    args = parser.parse_args()
    report = validate_boundary_evidence(args.source_run_dir, args.cumulative_step,
                                        require_improved=not args.allow_unqualified_diagnostic_source)
    output = scoped(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"passed": True, "path": str(output), "source_cumulative_optimizer_step": args.cumulative_step,
                      "first_unseen_source_index": report["first_unseen_source_index"]}))


if __name__ == "__main__":
    main()
