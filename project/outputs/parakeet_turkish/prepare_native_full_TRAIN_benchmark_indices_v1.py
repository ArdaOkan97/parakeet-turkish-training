"""Small benchmark-only ordered TRAIN blocks; no normal training cursor changes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    full = ROOT / "work/nemo_direct_trial_v1/manifests_clean_v1/train_balanced_full_pass.jsonl"
    proof_path = OUT / "full_nemo_clean_TRAIN_continuation_v1.json"
    shapes_path = OUT / "native_clean_full_TRAIN_memory_shape_review_v1.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    shapes = json.loads(shapes_path.read_text(encoding="utf-8"))
    full_hash = sha(full)
    if proof["manifest_sha256"][str(full)] != full_hash or shapes["source_files_sha256"][str(full)] != full_hash:
        raise ValueError("Use unchanged verified clean fullTRAIN.")
    definitions = [
        ("first20_unseen_updates_after350", 5600, 5920, "20 consecutive effective16 updates, initial normalpass cursor5600"),
        ("worst_B8_and_B4_effective_update1476", 23600, 23616, "Actual fullordered B8/B4 maximum sum lattice block"),
        ("largest_singleton_effective_update1668", 26672, 26688, "Largest singleton at row26679 T302/U165"),
        ("second_singleton_effective_update6148", 98352, 98368, "Second largest singleton row98359 T319/U154"),
        ("third_singleton_effective_update8372", 133936, 133952, "Third largest singleton row133943 T362/U134"),
    ]
    for b in (4, 8):
        row_start = shapes["top8_by_sum_for_each_physical_batch"][str(b)][0]["clips"][0]["manifest_row_zero_based"]
        if row_start // 16 * 16 != 23600:
            raise ValueError("Worst physicalbatch row block doesnot match actual shape report.")
    indices = [i for _, start, end, _ in definitions for i in range(start, end)]
    if len(indices) != 384 or len(set(indices)) != 384:
        raise ValueError("Expected384 distinct ordered benchmark recordings.")
    selected = {}
    wanted = set(indices)
    with full.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i in wanted:
                row = json.loads(line)
                if row["training_role"] != "train" or row["dataset_split"] not in {"train", "validated"}:
                    raise ValueError("Only accepted TRAIN sources may enter benchmark.")
                selected[i] = row
            if i >= max(wanted):
                break
    if set(selected) != wanted:
        raise ValueError("Missing selectedsource row.")
    blocks = []
    for name, start, end, purpose in definitions:
        block_rows = [selected[i] for i in range(start, end)]
        for group_start in range(0, len(block_rows), 16):
            group = block_rows[group_start:group_start + 16]
            if [r["language"] for r in group[:8]] != ["tr"] * 8 or any(r["language"] == "tr" for r in group[8:]):
                raise ValueError("Expected original8TRthen8replay source-order within every effective16.")
        blocks.append({"name": name, "start_row_zero_based": start, "end_row_exclusive": end,
            "source_effective16_update_start_one_based": start // 16 + 1,
            "source_effective16_update_end_inclusive": end // 16, "purpose": purpose,
            "ordered_indices": list(range(start, end)), "rows": block_rows,
            "ordered_source_ids": [r["source_id"] for r in block_rows]})
    report = {"schema": 1, "kind": "native_full_TRAIN_benchmark_indices_v1", "passed": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "benchmark_only": True,
        "full_manifest_path": str(full), "full_manifest_sha256": full_hash,
        "source_cumulative_updates": 350, "source_normal_pass_cursor": 5600,
        "ordered_indices": indices, "rows": [selected[i] for i in indices],
        "blocks": blocks, "effective_batch": 16, "recordings": len(indices), "effective16_updates": len(indices) // 16,
        "full_proof_path": str(proof_path), "full_proof_sha256": sha(proof_path),
        "shape_proof_path": str(shapes_path), "shape_proof_sha256": sha(shapes_path),
        "tokenizer_path": proof["clean_TRAIN"]["tokenizer_path"], "tokenizer_sha256": proof["clean_TRAIN"]["tokenizer_sha256"],
        "source_ID_WAV_hash_clean_label_order_metadata_equal_full_manifest": True,
        "normal_quality_training_order": "range(5600,149776) unchanged; this benchmark index order is never a normalpass samplerstate",
        "normal_pass_cursor_advanced": False, "benchmark_trained_weights_eligible_for_quality_selection": False,
        "PCM_rehash_or_GPU_or_DEV_TEST_read": False, "script_sha256": sha(__file__)}
    target = OUT / "native_full_TRAIN_benchmark_indices_v1.json"
    if target.exists():
        raise FileExistsError("Keep existing benchmark proof immutable.")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(target), "sha256": sha(target), "recordings": len(indices),
        "full_manifest_sha256": full_hash, "blocks": [{k: v for k, v in b.items() if k not in {"rows", "ordered_source_ids", "ordered_indices"}} for b in blocks]}))


if __name__ == "__main__":
    main()
