"""Offline TSC TRAIN candidate importer; never modifies existing pools.

The full pinned archive must pass size/SHA verification before import. A metadata
pass verifies all pairs, then a second pass stages only selected flat Train
WAV/TXT pairs without tar extraction APIs; creates a new separate
candidate manifest/proof. SoundFile/SciPy/Rust tokenizers load only when run,
not for --help. No Torch, Transformers, NeMo, model, GPU or network APIs.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
import unicodedata

from metrics import normalize_text

OUTPUT = Path(__file__).resolve().parent
WORKSPACE = OUTPUT.parent.parent
DATA = WORKSPACE / "work/data"
REPO = "issai/Turkish_Speech_Corpus"
REVISION = "d79ea01392882322e8f46657cb635236af25b6d4"
ARCHIVE_SHA = "3feaba1f911542b587622d49c968bef971cfd8a4b3908d73090b8b914d2b694f"
ARCHIVE_BYTES = 21366175820
PARENT_PROOF_SHA = "5c220d7fa4fa554b741afa7c565afe3f1500b3de8bfa99de0b1040a450cf4980"
PARENT_LABEL_SHA = "42aa606057ba78232ef930f7317675ef318dddb034bc302e4de2247eedd04e3a"
TOKENIZER_SHA = "9c9f537bf7190ae2f0e30448a6915c4f607fa4d011b4f62b6f1849064ef8afcd"
TOKENIZER_SEMANTIC_SHA = "b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75"
NORMALIZER_SHA = "433c31d9cbc4975c42d3a0fe659cd0bbd6c1caf2e1eb3d85169baafec7f61047"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def member_path(name):
    if "\\" in name:
        raise ValueError("Backslash archive path is unsupported")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or any(":" in part for part in path.parts):
        raise ValueError("Unsafe archive path: " + name)
    if not path.parts or path.parts[0] != "ISSAI_TSC_218":
        raise ValueError("Archive root differs from inspected author recipe: " + name)
    return path


def train_pair_key(path):
    if len(path.parts) < 2 or path.parts[1] != "Train":
        return None
    if len(path.parts) != 3 or path.suffix not in {".wav", ".txt"}:
        raise ValueError("Unexpected flat Train WAV/TXT layout: " + str(path))
    return path.stem


def configure_cpu_runtime():
    """Constrain this importer only, before importing numerical libraries."""
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
             "RAYON_NUM_THREADS")
    for name in names:
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    evidence = {"thread_environment": {name: os.environ[name] for name in names},
                "TOKENIZERS_PARALLELISM": "false", "priority_scope": "own_process_only",
                "Windows_below_normal_applied": False}
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.argtypes = []
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.GetPriorityClass.argtypes = [wintypes.HANDLE]
        kernel.GetPriorityClass.restype = wintypes.DWORD
        kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.SetPriorityClass.restype = wintypes.BOOL
        # GetCurrentProcess returns a pseudo-handle: it must not be closed.
        handle = kernel.GetCurrentProcess()
        evidence["priority_class_before"] = int(kernel.GetPriorityClass(handle))
        if not kernel.SetPriorityClass(handle, 0x4000):
            raise ctypes.WinError(ctypes.get_last_error())
        after = int(kernel.GetPriorityClass(handle))
        if after != 0x4000:
            raise RuntimeError("Own-process BelowNormal priority was not applied")
        evidence.update(Windows_below_normal_applied=True, priority_class_after=after)
    return evidence


def scan_archive_pairs(archive):
    """Read only bounded documents and pair metadata; retain no TarInfo cache."""
    pairs, split_counts, archive_text = {}, Counter(), []
    with tarfile.open(archive, "r|gz") as stream:
        while (member := stream.next()) is not None:
            try:
                path = member_path(member.name)
                if member.isdir():
                    continue
                if not member.isreg():
                    raise ValueError("Links/special archive members unsupported: " + member.name)
                split_counts[path.parts[1] if len(path.parts) > 1 else "root"] += 1
                if path.name.lower().startswith(("license", "licence", "copying", "readme")) and len(archive_text) < 8:
                    with stream.extractfile(member) as document:
                        value = document.read(min(member.size, 65536))
                    archive_text.append({"member": str(path), "truncated": len(value) < member.size,
                                         "text": value.decode("utf-8", errors="replace")})
                key = train_pair_key(path)
                if key is None:
                    continue
                pair = pairs.setdefault(key, {})
                if path.suffix in pair or member.size > (256 * 1024 * 1024 if path.suffix == ".wav" else 65536):
                    raise ValueError("Duplicate or oversized Train member: " + str(path))
                pair[path.suffix] = {"member": str(path), "size_bytes": member.size}
            finally:
                stream.members.clear()
    if not pairs or any(set(pair) != {".wav", ".txt"} for pair in pairs.values()):
        raise ValueError("Complete paired Train WAV/TXT set not observed")
    return pairs, split_counts, archive_text


def select_pair_keys(pairs, seed, candidate_limit):
    ordered = sorted(pairs, key=lambda key: hashlib.sha256(f"{seed}:TSC_train:{key}".encode()).digest())
    return ordered if candidate_limit is None else ordered[:candidate_limit]


def stage_selected_pairs(archive, metadata, selected, raw_dir):
    """Second stream writes exactly the selected pairs, never the full pool."""
    selected_set = set(selected)
    if len(selected_set) != len(selected) or not selected_set <= metadata.keys():
        raise ValueError("Selected pair keys must be unique and present in metadata")
    pairs, staged_files, staged_bytes = {}, 0, 0
    with tarfile.open(archive, "r|gz") as stream:
        while (member := stream.next()) is not None:
            try:
                path = member_path(member.name)
                if member.isdir():
                    continue
                if not member.isreg():
                    raise ValueError("Links/special archive members unsupported: " + member.name)
                key = train_pair_key(path)
                if key not in selected_set:
                    continue
                expected = metadata[key][path.suffix]
                if expected != {"member": str(path), "size_bytes": member.size} or path.suffix in pairs.get(key, {}):
                    raise ValueError("Selected archive metadata differs between passes: " + str(path))
                identity = hashlib.sha256(f"{REVISION}:{key}".encode()).hexdigest()[:24]
                target = raw_dir / (identity + path.suffix)
                with stream.extractfile(member) as source, target.open("xb") as out:
                    shutil.copyfileobj(source, out, length=65536)
                if target.stat().st_size != member.size:
                    raise ValueError("Staged selected member size differs: " + str(path))
                pairs.setdefault(key, {})[path.suffix] = (target, str(path))
                staged_files += 1
                staged_bytes += member.size
            finally:
                stream.members.clear()
    if set(pairs) != selected_set or any(set(pair) != {".wav", ".txt"} for pair in pairs.values()):
        raise ValueError("Selected complete pairs were not staged")
    return pairs, {"raw_Train_files_staged": staged_files, "raw_Train_bytes_staged": staged_bytes,
                   "Train_pairs_staged": len(pairs), "tar_member_cache_discarded": True,
                   "metadata_only_first_pass": True, "archive_stream_passes": 2}


def pcm_evidence(path, sf):
    digest = hashlib.sha256()
    with sf.SoundFile(path) as stream:
        if (stream.samplerate, stream.channels, stream.subtype) != (16000, 1, "PCM_16"):
            raise ValueError("Expected canonical16k mono PCM16: " + str(path))
        duration = stream.frames / stream.samplerate
        while True:
            signal = stream.read(65536, dtype="int16")
            if not signal.size:
                break
            digest.update(signal.astype("<i2", copy=False).tobytes())
    return digest.hexdigest(), duration


def canonical_audio(path, sf, np, resample_poly):
    with sf.SoundFile(path) as stream:
        if (stream.samplerate, stream.channels, stream.subtype) == (16000, 1, "PCM_16"):
            raw = stream.read(dtype="int16")
            if raw.size == 0:
                raise ValueError("Empty audio")
            digest = hashlib.sha256(raw.astype("<i2", copy=False).tobytes()).hexdigest()
            return Path(path).read_bytes(), digest, len(raw) / 16000
        signal = stream.read(dtype="float32", always_2d=True)
        rate = stream.samplerate
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError("Empty/nonfinite audio")
    signal = signal.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        signal = resample_poly(signal, 16000 // divisor, rate // divisor)
    buffer = io.BytesIO()
    sf.write(buffer, signal, 16000, format="WAV", subtype="PCM_16")
    payload = buffer.getvalue()
    pcm, rate = sf.read(io.BytesIO(payload), dtype="int16")
    return payload, hashlib.sha256(pcm.astype("<i2", copy=False).tobytes()).hexdigest(), len(pcm) / rate


def run(args):
    started = time.perf_counter()
    cpu_runtime = configure_cpu_runtime()
    archive = args.archive.resolve()
    archive_stat = (archive.stat().st_size, archive.stat().st_mtime_ns)
    if archive.stat().st_size != ARCHIVE_BYTES or sha256(archive) != ARCHIVE_SHA:
        raise ValueError("Full pinned archive size/SHA mismatch; refusing import")
    parent_path = OUTPUT / "tr_public_v3_preparation.json"
    labels_path = OUTPUT / "tr_public_v3_tokenizer_acceptance.json"
    tokenizer_path = OUTPUT / "checkpoint/tokenizer.json"
    normalizer_path = OUTPUT / "metrics.py"
    for path, expected in ((parent_path, PARENT_PROOF_SHA), (labels_path, PARENT_LABEL_SHA),
                           (tokenizer_path, TOKENIZER_SHA), (normalizer_path, NORMALIZER_SHA)):
        if sha256(path) != expected:
            raise ValueError("Pinned parent/helper identity changed: " + str(path))
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not labels["passed"] or labels["tokenizer_semantic_sha256"] != TOKENIZER_SEMANTIC_SHA:
        raise ValueError("Completed parent tokenizer/PCM proof required")
    current_path = Path(parent["CPU_tokenizer_accepted_manifest_path"])
    replay_path = DATA / "replay_train.jsonl"
    protected = {current_path: parent["CPU_tokenizer_accepted_manifest_sha256"],
                 replay_path: sha256(replay_path)}
    protected.update({Path(path): digest for path, digest in parent["heldout_manifests_sha256"].items()})
    for path, digest in protected.items():
        if sha256(path) != digest:
            raise ValueError("Existing manifest differs: " + str(path))
    stage = DATA / args.tag
    proof_path = OUTPUT / f"{args.tag}_preparation.json"
    if stage.exists() or proof_path.exists():
        raise FileExistsError("Fresh separate candidate namespace required")
    metadata_started = time.perf_counter()
    pair_metadata, split_counts, archive_text = scan_archive_pairs(archive)
    metadata_seconds = time.perf_counter() - metadata_started
    selected = select_pair_keys(pair_metadata, args.seed, args.candidate_limit)
    stage.mkdir()
    raw_dir, wav_dir = stage / "raw_train", stage / "audio"
    raw_dir.mkdir(); wav_dir.mkdir()
    staging_started = time.perf_counter()
    pairs, staging = stage_selected_pairs(archive, pair_metadata, selected, raw_dir)
    staging_seconds = time.perf_counter() - staging_started

    # Imports are intentionally delayed: offline --help/source checks are stdlib-only.
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    semantic = hashlib.sha256(json.dumps(json.loads(tokenizer.to_str()), ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if semantic != TOKENIZER_SEMANTIC_SHA or tokenizer.get_vocab_size() != 8199:
        raise ValueError("Exact current six-character HF tokenizer required")
    existing = read_rows(current_path) + read_rows(replay_path)
    heldout = list({row["source_id"]: row for path in parent["heldout_manifests_sha256"]
                    for row in read_rows(path)}.values())
    if any(row.get("dataset_split") not in {"validation", "test"} for row in heldout):
        raise ValueError("Fixed heldout split differs")
    forbidden_ids = {row["source_id"] for row in heldout}
    forbidden_text = {normalize_text(row["text"], language) for row in heldout for language in (row["language"], "tr")}
    forbidden_hashes = {row[key] for row in heldout for key in ("audio_sha256", "wav_sha256", "canonical_pcm_sha256") if row.get(key)}
    seen_ids = {row["source_id"] for row in existing}
    seen_hashes = {row[key] for row in existing for key in ("audio_sha256", "wav_sha256", "canonical_pcm_sha256") if row.get(key)}
    parent_pcm = {row["source_id"]: row["canonical_pcm_sha256"] for row in labels["accepted_source_labels_and_audio"]}
    if set(parent_pcm) != {row["source_id"] for row in read_rows(current_path)}:
        raise ValueError("Exact parent TRAIN canonical-PCM coverage required")
    seen_hashes.update(parent_pcm.values())
    for row in read_rows(replay_path):
        seen_hashes.add(pcm_evidence(row["audio_filepath"], sf)[0])
    for row in heldout:
        forbidden_hashes.add(pcm_evidence(row["audio_filepath"], sf)[0])

    excluded, diagnostics, prepared_count, accepted_count, seconds = Counter(), [], 0, 0, 0.0
    prepared_path, accepted_path = stage / "prepared_train.jsonl", stage / "accepted_train.jsonl"
    with prepared_path.open("x", encoding="utf-8", newline="\n") as prep, accepted_path.open("x", encoding="utf-8", newline="\n") as out:
        for key in selected:
            raw_wav, wav_member = pairs[key][".wav"]
            raw_txt, txt_member = pairs[key][".txt"]
            original = raw_txt.read_text(encoding="utf-8")
            text = unicodedata.normalize("NFC", original).strip()
            normal = normalize_text(text, "tr")
            source_id = f"{REPO}:default:train:None:{wav_member}"
            reason = "empty_normalized_reference" if not normal else None
            if source_id in forbidden_ids or normal in forbidden_text:
                reason = "fixed_heldout_source_or_normalized_reference"
            info = sf.info(raw_wav)
            source_duration = info.frames / info.samplerate
            if not .5 <= source_duration <= 30.:
                reason = "whole_recording_duration_outside_declared_bounds"
            if reason is None:
                source_hash = sha256(raw_wav)
                payload, canonical, duration = canonical_audio(raw_wav, sf, np, resample_poly)
                wav_hash = hashlib.sha256(payload).hexdigest()
                hashes = {source_hash, canonical, wav_hash}
                if hashes & forbidden_hashes:
                    reason = "fixed_heldout_audio_or_canonical_PCM"
                elif source_id in seen_ids or hashes & seen_hashes:
                    reason = "duplicate_existing_or_TSC_TRAIN_audio"
                elif not .5 <= duration <= 30.:
                    reason = "converted_whole_recording_duration_outside_declared_bounds"
            if reason:
                excluded[reason] += 1
                if len(diagnostics) < 12:
                    diagnostics.append({"source_id": source_id, "reason": reason, "raw_text": original[:512]})
                continue
            path = wav_dir / raw_wav.name
            with path.open("xb") as audio:
                audio.write(payload)
            row = {"dataset": REPO, "dataset_revision": REVISION, "dataset_config": "default", "dataset_split": "train",
                   "training_role": "train", "dataset_file": "ISSAI_TSC_218.tar.gz", "source_id": source_id,
                   "sentence_id": None, "original_audio_path": wav_member, "original_transcript_path": txt_member,
                   "speaker_id": None, "speaker_identity_verified": False, "language": "tr", "text": text,
                   "reference": text, "reference_raw_text": original, "reference_normalized": normal,
                   "source_transcript_sha256": sha256(raw_txt), "source_duration": source_duration,
                   "source_sample_rate": info.samplerate, "source_channels": info.channels,
                   "audio_filepath": str(path.resolve()), "duration": duration, "sample_rate": 16000,
                   "audio_sha256": source_hash, "wav_sha256": wav_hash, "canonical_pcm_sha256": canonical,
                   "data_expansion_role": "tsc_train", "archive_sha256": ARCHIVE_SHA}
            if pcm_evidence(path, sf) != (canonical, duration):
                raise ValueError("Converted WAV verification differs")
            prep.write(json.dumps(row, ensure_ascii=False) + "\n")
            prepared_count += 1
            seen_ids.add(source_id); seen_hashes.update(hashes)
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            decoded = tokenizer.decode(ids, skip_special_tokens=True)  # Rust BPE decode has no CTC grouping.
            if not ids or len(ids) > 180 or min(ids) < 0 or max(ids) >= 8199 or 8192 in ids:
                reason = "invalid_or_over180_current_HF_target"
            elif normalize_text(decoded, "tr") != normal:
                reason = "current_HF_ungrouped_normalized_roundtrip_mismatch"
            else:
                reason = None
            if reason:
                excluded[reason] += 1
                if len(diagnostics) < 12:
                    diagnostics.append({"source_id": source_id, "reason": reason, "raw_text": original[:512], "decoded": decoded[:512]})
                continue
            row.update(target_tokens=len(ids), target_max_id=max(ids),
                       label_ids_sha256=hashlib.sha256(json.dumps(ids, ensure_ascii=False, sort_keys=True).encode()).hexdigest())
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            accepted_count += 1; seconds += duration
            if accepted_count % 1000 == 0:
                print(f"TSC TRAIN candidate accepted {accepted_count}; processed {prepared_count}/{len(selected)}", flush=True)
    for path, digest in protected.items():
        if sha256(path) != digest:
            raise ValueError("Protected existing manifest changed during import")
    if (archive.stat().st_size, archive.stat().st_mtime_ns) != archive_stat:
        raise ValueError("Archive stat identity changed since full SHA verification")
    if accepted_count == 0:
        raise ValueError("No tokenizer-eligible TSC TRAIN candidate rows; refusing passed proof")
    proof = {"schema": 1, "kind": "TSC_TRAIN_candidate_import_v1", "passed": True,
             "status": "separate_CPU_audio_tokenizer_candidate_verified;future_source_policy_and_training_pending",
             "created_at_utc": datetime.now(timezone.utc).isoformat(), "importer_sha256": sha256(__file__),
             "archive_path": str(archive), "archive_revision": REVISION, "archive_sha256": ARCHIVE_SHA,
             "archive_size_bytes": ARCHIVE_BYTES, "full_archive_hash_verified": True,
             "archive_stat_unchanged_after_import": True,
             "archive_split_member_counts": dict(split_counts), "archive_license_or_readme_text_observed": archive_text,
             "Train_pairs_observed": len(pair_metadata), "Train_pairs_selected_for_candidate": len(selected),
             "staging": staging, "CPU_runtime": cpu_runtime,
             "archive_metadata_scan_seconds": metadata_seconds, "selected_pair_staging_seconds": staging_seconds,
             "full_Train_candidate_attempted": args.candidate_limit is None, "seed": args.seed,
             "prepared_clips": prepared_count, "accepted_clips": accepted_count, "accepted_actual_PCM_hours": seconds / 3600,
             "prepared_manifest_path": str(prepared_path), "prepared_manifest_sha256": sha256(prepared_path),
             "accepted_manifest_path": str(accepted_path), "accepted_manifest_sha256": sha256(accepted_path),
             "tokenizer_semantic_sha256": semantic, "max_target_tokens": 180, "blank_id": 8192,
             "TDT_decode_grouping": False, "duration_bounds_seconds": [.5, 30.],
             "parent_preparation_sha256": PARENT_PROOF_SHA, "parent_labels_sha256": PARENT_LABEL_SHA,
             "protected_existing_manifests_sha256": {str(path): value for path, value in protected.items()},
             "exclusions": dict(excluded), "finite_diagnostic_examples": diagnostics,
             "distinct_audio_repeated_TRAIN_text_retained": True, "speaker_identity_unknown": True,
             "fixed_heldout_predictions_read": False, "Dev_Test_archive_audio_or_text_imported": False,
             "existing_TRAIN_manifests_modified": False, "GPU_or_model_used": False,
             "new_data_added_to_training": False, "acoustic_feature_cache_or_TR_metadata_acceptance_pending": True,
             "limitations": ["Exact hash/PCM/text checks do not establish perceptual-near-duplicate or video/speaker disjointness.",
                             "Existing v3 source policy intentionally rejects TSC; future explicit policy/config is required.",
                             "No archive license scope is inferred from these metadata observations."],
             "elapsed_seconds": time.perf_counter() - started}
    with proof_path.open("x", encoding="utf-8") as stream:
        json.dump(proof, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: proof[key] for key in ("status", "accepted_clips", "accepted_actual_PCM_hours", "accepted_manifest_path")}, ensure_ascii=False))


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--archive", required=True, type=Path)
    value.add_argument("--tag", default="tsc_TRAIN_candidate_v1")
    value.add_argument("--seed", type=int, default=20261011)
    value.add_argument("--candidate-limit", type=int)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    if not arguments.tag.replace("_", "").isalnum() or (arguments.candidate_limit is not None and arguments.candidate_limit < 1):
        raise ValueError("Safe new tag and positive optional candidate limit required")
    run(arguments)
