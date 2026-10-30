"""Verify actual coherent diagnostic21, archive metadata, retire owned features."""
from pathlib import Path
import json
import os
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "outputs/parakeet_turkish"))
from fastmss_clean_feature_window_v1 import CACHE, cleanup_window, read_bound, record, write_new, scoped
from resource_watchdog_v2 import sample_memory

ROOT = Path(r"E:\gpu_training_scratchpad")
AREA = ROOT / "analysis/ASR18_memory_diagnosis_v1/cleaner_precomputed_after20_01"
OUT = ROOT / "analysis/FastMSS_integration_v1"
CPU = OUT / "actual_cleaner_feature_window_01"


def guard():
    value = sample_memory(os.getpid(), "CPU_metadata_owned_clean_feature_retirement")
    worker, system = value["worker_private_memory"], value["system_memory"]
    assert worker["available"] and system["available"] and value["paired_sampling_within_one_second"]
    assert worker["private_bytes"] < 2_000_000_000
    assert system["commit_available_bytes"] >= 2_000_000_000
    assert system["commit_used_bytes"] / system["commit_limit_bytes"] < .90


def main():
    report = json.loads((CPU / "report.json").read_bytes())
    owner = read_bound(report["owner"], guard)
    assert owner["phase"] == "cleaner_data_fine_tuning" and owner["updates"] == 1
    assert report["manifest"]["sha256"] == "5db8d92d6bf1d26f901b6f55c497d632c8d266a34790f4dde3f11ae3a0f08b11"
    assert report["owner"]["sha256"] == "08337b0c9e0d0bcb4c08e3da2534b8e324f10dca39ee2546f4eafac4af2236ca"
    boundary_path = AREA / "final_coherent_clean_feature_cleanup_boundary_v1.json"
    assert record(boundary_path, guard)["sha256"] == "fd912cd36bfda51765036ac04dbdbd104e8b68ed16c7410ce90ef6e995fde2d1"
    boundary = json.loads(boundary_path.read_bytes())
    native = read_bound(boundary["native_actual_closed"], guard)
    actual_report = read_bound(boundary["native_report"], guard)
    restore = read_bound(boundary["final_exact_restore"], guard)
    summary = read_bound(boundary["actual_summary"], guard)
    latest = read_bound(boundary["verified_final_coherent_latest_pointer"], guard)
    state = read_bound(boundary["checkpoint"], guard)
    assert native["actual_tool_exit_code"] == native["actual_native_exit_code"] == 0
    assert native["owned_session"] == 46439 and native["native_PID"] == 36104
    assert native["actual_tool_chunk_id"] == "d97b18"
    assert native["actual_native_updates_this_child"] == 1 and native["coherent_chain_updates"] == 21
    assert native["report"] == boundary["native_report"] and actual_report["passed"] is True
    assert restore["passed"] is True
    assert all(actual_report["completion"].get(key) == value for key, value in restore.items())
    assert actual_report["completion"]["sources_unchanged"] is True
    assert restore["native_updates_this_child"] == 1 and restore["coherent_chain_updates"] == 21
    for name in (
        "all207_weights_three_Adam_Python_NumPy_Torch_CUDA_RNG_exact",
        "actual_cleaner_cursor_carried_replay_and_journal_prefix_restored",
        "original20_source_and_sampler_restored_before_next_exactdraw",
        "clean_feature_payload_exact_before_after_restore",
    ):
        assert restore[name] is True
    assert restore["checkpoint"] == boundary["checkpoint"]
    assert restore["feature_window"] == report["manifest"]
    assert restore["actual_applied_augmentation_count"] == 0
    assert restore["parent_frontend_augmenter_transport_invoked"] is False
    assert latest["committed_updates"] == 21 and latest["clocks"] == [64271, 2021, 62271]
    assert latest["writer_returned_after_fsync"] is True
    assert latest["checkpoint"] == boundary["checkpoint"]
    assert latest["payload"] == state["payload"] == boundary["checkpoint_payload"]
    assert record(boundary["checkpoint_payload"]["path"], guard) == boundary["checkpoint_payload"]
    assert summary["checkpoint"] == boundary["checkpoint"]
    assert summary["tensor_payload"] == boundary["checkpoint_payload"]
    assert summary["all_native_write_restore_forecasts_covered"] is True
    assert summary["production_ready"] is False and summary["diagnostic_only"] is True
    assert boundary["verified_all207_three_Adam_all_RNG_samplers_and_journal_prefix"] is True
    assert boundary["serial_DEV_uses_checkpoint_not_features"] is True
    assert boundary["phase"] == owner["phase"] and boundary["committed_window_updates"] == 1
    preserved = OUT / "actual_cleaner_feature_window_original_bindings_after_native21_v1.json"
    write_new(preserved, {
        "kind": "FastMSS_actual_cleaner_feature_original_metadata_snapshot_v1",
        "owner": report["owner"], "owner_contents": owner,
        "actual_CPU_preparation_handoff": record(OUT / "FastMSS_actual_clean_feature_CPU_handoff_v1.json", guard),
        "retirement_boundary": record(boundary_path, guard),
        "copied_small_metadata_only": True,
        "original_audio_source_checkpoint_journal_untouched": True,
    })
    closures = [entry["actual_feature_closure"] for entry in report["updates"]]
    result = cleanup_window(
        report["owner"], closures=closures, manifest_binding=report["manifest"], boundary=boundary,
        verify_boundary=lambda actual: actual == boundary,
        evidence_path=OUT / "actual_cleaner_feature_window_durable_metadata_after_native21_v1.json",
        guard=guard,
    )
    assert len(result["removed_exact_owned_files"]) == 12
    assert not scoped(report["owner"]["path"]).parent.exists() and not list(CACHE.iterdir())
    for key in ("native_actual_closed", "native_report", "final_exact_restore", "actual_summary",
                "verified_final_coherent_latest_pointer", "checkpoint"):
        assert record(boundary[key]["path"], guard) == boundary[key]
    result.update(
        kind="FastMSS_actual_final_coherent_clean_feature_window_retirement_v1", passed=True,
        actual_native_owned_exit=native, verified_checkpoint_payload=boundary["checkpoint_payload"],
        verified_final_restore=boundary["final_exact_restore"],
        preserved_original_metadata=record(preserved, guard), feature_files_only_deleted=True,
        original_audio_source_checkpoint_journal_untouched=True,
        native_qualification_or_production_approval=False,
    )
    receipt = OUT / "actual_cleaner_feature_window_actual_retirement_v1.json"
    write_new(receipt, result)
    print(json.dumps({"passed": True, "removed_exact_owned_files": len(result["removed_exact_owned_files"]),
                      "retirement": record(receipt, guard), "preserved_original_metadata": record(preserved, guard)}))


if __name__ == "__main__":
    main()
