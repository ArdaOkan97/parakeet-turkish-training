"""Write parent-authorized exit receipt; stdlib only, no tensor/model loads."""
import datetime as dt
import hashlib
import json
from pathlib import Path

from diagnose_native_extension_stop_v1 import owned_pid

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
LOOP = OUT / "runs/adaptive_20261011_native_memory_fix_01"
RUN_ID = "adaptive_20261011_native_memory_fix_01_public_winner_extension"
RUN = OUT / "runs" / RUN_ID


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def main():
    state_path = LOOP / "loop_status.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    selection = json.loads((RUN / "selection.json").read_text(encoding="utf-8"))
    history = json.loads((RUN / "training_history.json").read_text(encoding="utf-8"))
    trial = next(t for t in state["trials"] if t["run_id"] == RUN_ID)
    assert state["status"] == "resource_review_required" and trial["exit_code"] == 0
    assert selection["completed_steps"] == trial["completed_steps"] == 358
    assert selection["selected_step"] == trial["selected_step"] == 0
    assert len(history) == 358 and [r["step"] for r in history] == list(range(1, 359))
    process_status = [owned_pid(29744), owned_pid(1116)]
    assert all(p.get("present") is False for p in process_status)
    paths = {"source_latest": selection["latest_weights"], "source_best": selection["weights"], "child_run_config": RUN / "run_config.json", "child_history": RUN / "training_history.json", "child_selection": RUN / "selection.json", "resource_review": OUT / "native_memory_fix_01_extension_stop_diagnosis_v1.json"}
    receipt = {"schema": 1, "kind": "root_owned_native_resource_review_exit_receipt_v2", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "root_owned_exec_session": 40244, "root_observed_owned_session_completed": True, "root_observed_exit_code": 1, "root_observed_completion_chunk": "91771a", "root_observation_authority": "Parent explicitly confirmed owned session40244 EXIT1/chunk91771a and authorized this receipt", "predecessor_loop_id": LOOP.name, "predecessor_run_id": RUN_ID, "predecessor_status": state["status"], "predecessor_status_path": str(state_path.resolve()), "predecessor_status_sha256": digest(state_path), "child_exit_code": 0, "child_durable_step": 358, "child_selected_step": 0, "child_stop_early_reason": selection["stop_early_reason"], "child_stop_reason": trial["cooperative_stop_reason"], "child_configured_steps": selection["configured_steps"], "history_contiguous_steps": 358, "checkpoint_tensor_schema_reloaded_in_this_receipt": False, "checkpoint_validation_scope": "Saved filenames/bytes/SHA plus exact closed selection/history; no Torch/checkpoint tensor deserialization", "owned_processes_absent_now": process_status, "resource_sample_at_utc": "2026-10-11T08:17:11.716716+00:00", "controller_stop_request_at_utc": trial["cooperative_stop_requested_at_utc"], "sample_to_stop_request_seconds": (dt.datetime.fromisoformat(trial["cooperative_stop_requested_at_utc"])-dt.datetime.fromisoformat("2026-10-11T08:17:11.716716+00:00")).total_seconds(), "resource_review_passed": True, "runtime_provenance": {key: record(trial[key]) for key in ("memory_provenance_path", "cache_provenance_path", "io_provenance_path", "worker_identity_path", "native_provenance_path")}, "selected_qualifying_development_wer": selection["turkish_validation_wer"], "heldout_tests_evaluated": False, "exports_created_by_this_stage": False, "root_only_future_GPU_execution": True, "original_sources_artifacts_reports_unchanged": True}
    for name, path in paths.items():
        value = record(path)
        receipt[name+"_path"] = value["path"]
        receipt[name+"_sha256"] = value["sha256"]
        receipt[name+"_bytes"] = value["bytes"]
    destination = OUT / "native_memory_fix_01_extension_exit_receipt_v1.json"
    assert not destination.exists()
    destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(destination), "sha256": digest(destination), "latest": receipt["source_latest_sha256"], "best": receipt["source_best_sha256"], "history": receipt["child_history_sha256"], "sample_to_request_seconds": receipt["sample_to_stop_request_seconds"]}))


if __name__ == "__main__":
    main()
