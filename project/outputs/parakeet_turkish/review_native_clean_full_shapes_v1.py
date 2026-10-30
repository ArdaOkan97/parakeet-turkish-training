"""Whole clean TRAIN schedule lattice arithmetic; CPU/SP only, no PCM rehash."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def rank(heap, item, key, ordinal):
    record = (int(item[key]), ordinal, item)
    if len(heap) < 8:
        heapq.heappush(heap, record)
    elif record[:2] > heap[0][:2]:
        heapq.heapreplace(heap, record)


def main():
    import sentencepiece as spm
    started = time.perf_counter()
    model = ROOT / "work/nemo_direct_trial_v1/tokenizer/tokenizer.model"
    sp = spm.SentencePieceProcessor(model_file=str(model))
    if sp.get_piece_size() != 8198:
        raise ValueError("Use unchanged8198 native characterbaseline.")
    tr_metadata = ROOT / "work/data/tr_public_v3_frame_metadata_v1.jsonl"
    frames = {}
    formula_checks = 0
    for row in rows(tr_metadata):
        sid = row["source_id"]
        if sid in frames:
            raise ValueError("Repeated TR metadataID.")
        mel = int(row["actual_PCM_frames"]) // 160
        t = (mel + 7) // 8
        if t != row["encoder_frames"] or mel != row["valid_mel_frames"]:
            raise ValueError("Pinned PCM→mel→T formula differs from audited metadata.")
        frames[sid] = t
        formula_checks += 1
    if formula_checks != 74883:
        raise ValueError("Require complete accepted TR PCM-derived metadata.")
    prior_path = OUT / "native_pilot_lattice_memory_review_v1.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    old_us, tr_pilot_checks, replay_ids = {}, 0, set()
    for batch in prior["all_microbatch_estimates"]:
        for sid, shape in zip(batch["source_ids"], batch["clip_shapes"]):
            t, u = map(int, shape)
            if sid in frames and frames[sid] != t:
                raise ValueError("PCM metadata/native pilot encoderT mismatch.")
            if sid not in frames:
                frames[sid] = t
                replay_ids.add(sid)
            if sid in old_us and old_us[sid] != u:
                raise ValueError("Inconsistent source token length in original pilot.")
            old_us[sid] = u
    tr_pilot_checks = sum(sid not in replay_ids for sid in old_us)
    if tr_pilot_checks != 2000 or len(replay_ids) != 1595 or len(frames) != 76478:
        raise ValueError("All replay+2000TR actualpilot shapes required.")
    clean_dir = ROOT / "work/nemo_direct_trial_v1/manifests_clean_v1"
    lengths, unique_stats = {}, Counter()
    pilot_u_changes = Counter()
    for name in ("train_turkish_pool.jsonl", "train_replay_pool.jsonl"):
        for row in rows(clean_dir / name):
            sid = row["source_id"]
            if sid not in frames or sid in lengths:
                raise ValueError("Every canonical source must have knownframes and beunique.")
            u = len(sp.encode(row["text"]))
            if u <= 0:
                raise ValueError("Empty native targets.")
            lengths[sid] = u
            unique_stats[row["language"] + "_sources"] += 1
            if sid in old_us:
                pilot_u_changes["equal" if u == old_us[sid] else "shorter" if u < old_us[sid] else "longer"] += 1
    full = clean_dir / "train_balanced_full_pass.jsonl"
    proof_path = OUT / "full_nemo_clean_TRAIN_continuation_v1.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if sha(full) != proof["manifest_sha256"][str(full)]:
        raise ValueError("Cleanfullmanifest proof mismatch.")
    heaps = {b: [] for b in (1, 2, 4, 8)}
    buffers = {b: [] for b in heaps}
    seen = set()
    presentations = Counter()
    total = 0
    max_t = max_u = 0
    for i, row in enumerate(rows(full)):
        sid = row["source_id"]
        t, u = frames[sid], lengths[sid]
        max_t, max_u = max(max_t, t), max(max_u, u)
        seen.add(sid)
        presentations[row["language"]] += 1
        descriptor = {"manifest_row_zero_based": i, "source_id": sid, "language": row["language"],
            "dataset": row["dataset"], "duration_seconds": row["duration"], "encoder_T": t, "native_target_U": u,
            "joint_matrix_FP32_bytes": t * (u + 1) * 8204 * 4,
            "native_label_acts_or_gradient_FP32_bytes": t * (u + 1) * 8199 * 4,
            "audio_filepath": row["audio_filepath"], "wav_sha256": row["wav_sha256"]}
        for b in buffers:
            buffers[b].append(descriptor)
            if len(buffers[b]) == b:
                group = buffers[b]
                group_id = i // b
                item = {"physical_B": b, "microbatch_index_zero_based": group_id,
                    "effective16_optimizer_update_one_based": i // 16 + 1,
                    "source_ids": [x["source_id"] for x in group], "languages": [x["language"] for x in group],
                    "clip_shapes": [[x["encoder_T"], x["native_target_U"]] for x in group],
                    "sum_individual_FP32_joint_matrix_bytes": sum(x["joint_matrix_FP32_bytes"] for x in group),
                    "max_individual_FP32_joint_matrix_bytes": max(x["joint_matrix_FP32_bytes"] for x in group),
                    "sum_individual_native_label_acts_or_gradient_FP32_bytes": sum(x["native_label_acts_or_gradient_FP32_bytes"] for x in group),
                    "hypothetical_padded_full_B_joint_FP32_bytes": b * max(x["encoder_T"] for x in group) * (max(x["native_target_U"] for x in group) + 1) * 8204 * 4,
                    "clips": group}
                rank(heaps[b], item, "sum_individual_FP32_joint_matrix_bytes", group_id)
                buffers[b] = []
        total += 1
    if total != 149776 or len(seen) != 76478 or any(buffers.values()):
        raise ValueError("Expected complete/fullpreserved native TRAIN schedule and groupalignment.")
    report = {"schema": 1, "kind": "whole_clean_native_TRAIN_singleton_B2_B4_B8_lattice_arithmetic_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "CPU_only": True,
        "Torch_NeMo_imported": "torch" in sys.modules or "nemo" in sys.modules,
        "source_files_sha256": {str(p): sha(p) for p in (model, tr_metadata, prior_path, proof_path, full)},
        "presentations": total, "unique_sources": len(seen), "unique_source_counts_by_language": dict(unique_stats),
        "presentation_counts_by_language": dict(presentations), "max_encoder_T": max_t, "max_clean_native_U": max_u,
        "frame_formula": "auditedactualPCM samples/160 floor→ceil(mel/8); currentTR metadataexact74883; all1595 replayT reusedactualpilotWAVheader shapeaudit",
        "TR_formula_checks": formula_checks, "TR_formula_vs_original_native_pilot_tuples_checked_unique": tr_pilot_checks,
        "replay_source_actual_pilot_tuples_reused": len(replay_ids), "pilot_clean_U_vs_original_unique_counts": dict(pilot_u_changes),
        "joint_width": 8204, "label_width_including_blank": 8199,
        "formula": "FP32joint=T*(U+1)*8204*4; FP32labelacts/grad=T*(U+1)*8199*4",
        "top8_by_sum_for_each_physical_batch": {str(b): [r[2] for r in sorted(heaps[b], reverse=True)] for b in heaps},
        "ranking_and_limits": [
            "Actualorderedfullmanifestgroups, not re-bucketed or generatedstress: B1x16/B2x8/B4x4/B8x2 shareeffective16 recordings.",
            "Fusedjointsubbatch1 trims each loss toclipT/U, but retainsmultiplegraphs untilphysicalmicrobatchbackward; summedmatrices describepossiblegraph burden, not exactpeakVRAM.",
            "HypotheticalpaddedBjoint is provided only as arithmetic comparison; stockfusedsubbatch1 doesnot formthat fullpaddedB lattice.",
            "Excludesweights/optimizer/nativetemporarycopies/hidden/encoderactivations/frontend/compilerworkspace andallocatorfragmentation.",
            "ThisisnotactualGPUmemory/capacity/speed/qualityproof. Rootshouldstressactualwholeeffective16 containinglargest singletonwithallacceptedrows preserved.",
            "NoPCMread/rehash, dataedit, subsetdrop, cropping, orDEV/TESTread.",
        ], "elapsed_CPU_seconds": time.perf_counter() - started, "script_sha256": sha(__file__)}
    target = OUT / "native_clean_full_TRAIN_memory_shape_review_v1.json"
    if target.exists():
        raise FileExistsError("Preservepreviousshapeevidence.")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(target), "sha256": sha(target), "unique": len(seen), "presentations": total,
        "max_T": max_t, "max_U": max_u, "maxima": {str(b): report["top8_by_sum_for_each_physical_batch"][str(b)][0] for b in heaps},
        "elapsed_CPU_seconds": report["elapsed_CPU_seconds"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
