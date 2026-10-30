"""Bounded independent PCM/metadata audit of an imported TSC TRAIN pilot.

Reads at most500 accepted canonical TRAIN WAVs; no archive, model, GPU, network,
heldout audio/predictions or training adoption. Label/audio semantic quality
still needs listening: this audit verifies identities and reports signal flags.
"""
from __future__ import annotations
import argparse
from array import array
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import wave

import tsc_supplemental_data_policy_v1 as policy

HERE, ROOT = policy.OUTPUT, policy.ROOT


def require(value, message):
    if not value:
        raise ValueError(message)


def signal_evidence(path):
    """Whole canonical PCM, bounded65536-frame buffers, integer finite samples."""
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), "Workspace WAV required")
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    digest = hashlib.sha256()
    frames = zeros = clipped = peak = square_sum = 0
    with wave.open(str(path), "rb") as stream:
        require((stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype())
                == (16000, 1, 2, "NONE"), "Canonical16k mono PCM16 required")
        expected = stream.getnframes()
        require(8000 <= expected <= 480000, "Whole pilot recording must be0.5–30s")
        while True:
            raw = stream.readframes(65536)
            if not raw:
                break
            require(len(raw) % 2 == 0, "Complete PCM16 samples required")
            digest.update(raw)
            samples = array("h", raw)
            if sys.byteorder != "little":
                samples.byteswap()
            frames += len(samples)
            zeros += samples.count(0)
            clipped += sum(abs(sample) >= 32767 for sample in samples)
            peak = max(peak, max((abs(sample) for sample in samples), default=0))
            square_sum += sum(sample * sample for sample in samples)
    require(frames == expected, "Truncated WAV sample count")
    require(before == (path.stat().st_size, path.stat().st_mtime_ns), "WAV changed during audit")
    return {"actual_PCM_frames": frames, "duration_seconds": frames / 16000,
            "canonical_pcm_sha256": digest.hexdigest(), "wav_sha256": policy.sha256(path),
            "PCM16_samples_finite_by_representation": True, "all_zero": peak == 0,
            "zero_fraction": zeros / frames, "clipped_sample_fraction": clipped / frames,
            "peak_absolute_normalized": peak / 32768, "rms_normalized": math.sqrt(square_sum / frames) / 32768}


def quantiles(values):
    values = sorted(values)
    return {"min": values[0], "median": values[(len(values)-1)//2],
            "p95": values[math.ceil(.95*len(values))-1], "max": values[-1]}


def main(args):
    report = args.report.resolve()
    require(report.is_relative_to(ROOT) and not report.exists(), "NEW workspace report required")
    proof_path = args.import_proof.resolve()
    proof_sha = policy.sha256(proof_path)
    parent = policy.read_json(HERE / "tr_public_v3_preparation.json")
    require(policy.sha256(HERE / "prepare_TSC_TRAIN_v1.py") == policy.APPROVED_IMPORTER_SHA
            and policy.sha256(HERE / "tr_public_v3_preparation.json") == policy.PARENT_PROOF_SHA,
            "Frozen importer/parent source changed")
    proof = policy.read_json(proof_path)
    policy.validate_import_proof(proof, parent)
    accepted = policy.verified_path(proof["accepted_manifest_path"], proof["accepted_manifest_sha256"])
    rows = policy.read_rows(accepted)
    require(0 < len(rows) == proof["accepted_clips"] <= 500, "Whole bounded pilot of at most500 clips required")
    evidence = []
    for row in rows:
        policy.validate_tsc_row(row, accepted)
        result = signal_evidence(row["audio_filepath"])
        require(result["wav_sha256"] == row["wav_sha256"]
                and result["canonical_pcm_sha256"] == row["canonical_pcm_sha256"]
                and result["duration_seconds"] == row["duration"], "Importer PCM/WAV/duration identity differs")
        evidence.append({"source_id": row["source_id"], **result})
    require(policy.sha256(proof_path) == proof_sha
            and (not args.import_proof_sha256 or proof_sha == args.import_proof_sha256),
            "ROOT-supplied importer proof SHA changed")
    require(policy.sha256(accepted) == proof["accepted_manifest_sha256"], "Accepted manifest changed during audit")
    by_id = {item["source_id"]: item for item in evidence}
    # Deterministic TRAIN-only listening sample: duration extremes, digits and spread.
    ordered = sorted(rows, key=lambda row: (row["duration"], row["source_id"]))
    chosen = ordered[:3] + ordered[-3:] + [row for row in rows if any(c.isdigit() for c in row["text"])][:5]
    chosen += [rows[round(index * (len(rows)-1) / 11)] for index in range(12)]
    seen = set(); examples = []
    for row in chosen:
        if row["source_id"] in seen:
            continue
        seen.add(row["source_id"])
        examples.append({"source_id": row["source_id"], "audio_filepath": row["audio_filepath"],
                         "reference": row["text"], "target_tokens": row["target_tokens"],
                         "signal": by_id[row["source_id"]]})
        if len(examples) == 20:
            break
    result = {"schema": 1, "kind": "TSC_TRAIN_pilot_independent_PCM_metadata_audit_v1",
              "passed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "auditor_sha256": policy.sha256(__file__), "import_proof_path": str(proof_path),
              "import_proof_sha256": proof_sha, "accepted_manifest_path": str(accepted),
              "accepted_manifest_sha256": proof["accepted_manifest_sha256"], "clips_audited": len(rows),
              "actual_PCM_hours": math.fsum(row["duration"] for row in rows)/3600,
              "duration_seconds": quantiles([row["duration"] for row in rows]),
              "target_tokens": quantiles([row["target_tokens"] for row in rows]),
              "all_zero_clips": sum(item["all_zero"] for item in evidence),
              "clips_with_any_full_scale_samples": sum(item["clipped_sample_fraction"] > 0 for item in evidence),
              "max_clipped_sample_fraction": max(item["clipped_sample_fraction"] for item in evidence),
              "signal_evidence": evidence, "TRAIN_listening_examples": examples,
              "all_PCM_WAV_duration_identities_exact": True, "fresh_PCM_content_reads": True,
              "archive_rehashed_or_read": False, "heldout_audio_or_predictions_read": False,
              "model_GPU_or_training_used": False, "new_data_added_to_training": False,
              "semantic_transcription_quality_verified": False,
              "full_import_decision_requires_ROOT_listening_review": True,
              "limitations": ["Finite/nonzero PCM and exact hashes do not establish transcript correctness.",
                              "Clipping flags are descriptive and not an automatic corpus rejection.",
                              "Speaker/video/perceptual near-duplicate disjointness is not proven."]}
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    print(json.dumps({key: result[key] for key in ("passed", "clips_audited", "actual_PCM_hours", "all_zero_clips", "clips_with_any_full_scale_samples")}))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--import-proof", type=Path, required=True)
    p.add_argument("--import-proof-sha256")
    p.add_argument("--report", type=Path, required=True)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
