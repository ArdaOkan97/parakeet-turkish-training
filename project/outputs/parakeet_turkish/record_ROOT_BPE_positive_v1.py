"""Record bounded current native evidence without reading historical metrics."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import struct
from pathlib import Path

from append_only_history_v1 import MAGIC, ROW
from atomic_io_retry_v1 import read_json, write_json
from tsc_bpe_full_decoder_transition_v1 import ROOT, SOURCE, STORAGE, record, require


def latest_row(directory):
    head = read_json(directory / "HEAD.json")
    ref = head["last_transaction"]
    path = directory / ref["name"]
    require(ref["bytes"] <= 41943040, "Only a bounded current journal transaction may be read")
    data = path.read_bytes()
    require(len(data) == ref["bytes"] and hashlib.sha256(data).hexdigest() == ref["sha256"]
            and data[:8] == MAGIC, "Current immutable transaction bytes differ")
    offset = 12 + struct.unpack("<I", data[8:12])[0]
    result = None
    while data[offset:offset + 4] == b"ROW1":
        tag, step, length, digest = ROW.unpack(data[offset:offset + ROW.size])
        offset += ROW.size
        payload = data[offset:offset + length]
        require(hashlib.sha256(payload).digest() == digest, "Current row digest differs")
        result = json.loads(payload)
        offset += length
    require(result is not None and result["step"] == head["max_step"], "Current row/frontier differ")
    return result, record(path)


def main(args):
    run = STORAGE / "runs" / args.run_id
    loop = args.recovery_loop or STORAGE / "runs" / (args.run_id + "_loop_01")
    state = read_json(loop / "loop_status.json")
    lease = read_json(ROOT / "work/gpu_training_owner.json")
    identity = read_json(loop / "worker_identity.json")
    config = read_json(run / "run_config.json")
    proof = read_json(run / "BPE_full_decoder_initial_source_equivalence.json")
    require(lease["owner"] == "/root" and lease["target_run_id"] == args.run_id
            and lease["running"] is True and state["status"] == "running"
            and state["published_step"] > 62250 and proof["passed"] is True
            and config["source207_audit"]["passed"] is True, "Actual ROOT restore and positive progress required")
    stages = [json.loads(line) for line in (run / "stage_memory.jsonl").read_text().splitlines()]
    updates = [value for value in stages if value["stage"] == "after_actual_decoder_optimizer"
               and value.get("actual_parameter_delta_evidence") and value["pid"] == identity["pid"]]
    require(updates, "Actual native parameter-delta proof required")
    first = updates[0]
    delta = first["actual_parameter_delta_evidence"]
    require(delta["actual_native_backward_optimizer_update"] is True and delta["frozen192_unchanged"] is True
            and len(delta["changed_mature_partitions"]) == 17
            and (len(delta["changed_fresh_partitions"]) == 3 if state["decoder_arm"] == "bpe512"
                 else not delta["changed_fresh_partitions"]), "Mature/fresh actual updates and frozen encoder required")
    row, transaction = latest_row(run / "history_journal")
    require(row["head_lr"] == 1e-5 and row["lora_lr"] == 0. and row["encoder_LoRA_active"] is False
            and row["new_rows_lr"] == (1e-4 if state["decoder_arm"] == "bpe512" else None),
            "Actual applied rates differ from the declared paired recipe")
    result = {"kind": "ROOT_BPE_actual_native_positive_v1", "at_utc": datetime.now(timezone.utc).isoformat(),
              "passed": True, "owner": "/root", "run_id": args.run_id,
              "actual_owned_session": args.owned_session, "actual_launcher_pid": state["owned_launcher_pid"],
              "actual_worker_pid": identity["pid"], "published_step": row["step"],
              "positive_updates": row["step"] - 62250,
              "source350": record(run / "BPE_full_decoder_initial_source_equivalence.json"),
              "run_config": record(run / "run_config.json"), "first_native_update": first,
              "actual_applied_LR_and_coverage": {key: row[key] for key in
                  ("step", "head_lr", "new_rows_lr", "lora_lr", "mature_Adam_clock", "fresh_row_Adam_clock",
                   "frozen_encoder_Adam_clock", "coverage", "source_ID_sequence_sha256")},
              "current_transaction": transaction, "all_new_outputs_on_E": True,
              "no_quality_improvement_claimed": True}
    if args.recovery_loop:
        restored = read_json(loop / "exact_restore.json")
        require(restored["passed"] is True and restored["saved_step"] == 62500
                and restored["sampler_updates"] == 62500 and restored["fresh_row_Adam_clock"] == 250
                and restored["retention_failures"] == 1 and first["step"] == 62501
                and row["step"] > 62500 and row["mature_Adam_clock"] == row["step"]
                and row["fresh_row_Adam_clock"] == row["step"] - 62250
                and restored["journal"]["preserved_orphan_branches"] >= 1,
                "Actual exact restoration, preserved failed suffix and new positive native updates required")
        result["actual_exact_restore"] = record(loop / "exact_restore.json")
        result["runtime_repair"] = restored["repair_receipt"]
        result["recovered_saved_step"] = 62500
        result["failed_uncheckpointed_updates_preserved_then_replayed"] = 250
        result["new_positive_updates_after_restore"] = row["step"] - 62500
    suffix = "_actual_repair_native_positive_v1.json" if args.recovery_loop else "_actual_native_positive_v1.json"
    target = STORAGE / "analysis/BPE_full_decoder_entry_v1" / (args.run_id + suffix)
    require(not target.exists(), "Never replace immutable first-positive evidence")
    write_json(target, result)
    ref = record(target)
    for filename in ("experiment_plan.json", "post_epoch_experiment_protocol.json"):
        path = SOURCE / filename
        value = read_json(path)
        value["current_GPU_owner"] = "/root"
        value["current_BPE_actual_native_positive"] = ref
        if args.recovery_loop: value["current_BPE_runtime_serialization_recovery"] = result["actual_exact_restore"]
        value["BPE_authoritative_E_queue"] = str(STORAGE / "experiment_queue_state.json")
        for item in value.get("ordered_next_work", []):
            if "BPE512 with full" in item.get("item", ""):
                item["status"] = ("ROOT_actual_native_positive_candidate_RUNNING_matched8199_next_after_actual_closure"
                                  if state["decoder_arm"] == "bpe512" else
                                  "ROOT_actual_native_positive_matched8199_RUNNING_candidate_complete")
                item["actual_native_positive"] = ref
        write_json(path, value)
    queue_path = SOURCE / "experiment_queue_state.json"
    archive = STORAGE / "analysis/BPE_full_decoder_entry_v1/prior_BASE_canonical_queue_v1.json"
    if not archive.exists():
        write_json(archive, read_json(queue_path))
    write_json(queue_path, {"kind": "E_authoritative_BPE_queue_pointer_v1", "owner": "/root",
        "current_run_id": args.run_id, "status": "running", "published_step": row["step"],
        "configured_steps": 64250, "authoritative_state_path": str(STORAGE / "experiment_queue_state.json"),
        "actual_native_positive": ref, "prior_completed_BASE_queue": record(archive),
        "updated_at_utc": result["at_utc"], "objective_complete": False})
    print(json.dumps({"passed": True, "receipt": ref, "step": row["step"],
                      "positive_updates": row["step"] - 62250}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--owned-session", type=int, required=True)
    p.add_argument("--recovery-loop", type=Path)
    main(p.parse_args())
