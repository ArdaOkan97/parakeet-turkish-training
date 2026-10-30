"""Inactive whole-TRAIN target regeneration; Rust tokenizers, no audio/model reads.

Both approved inputs are streamed once. A failed baseline/geometry/roundtrip
check leaves only a clearly pending output and a failure receipt, never a
published complete metadata file. No original manifest or target is rewritten.
"""
from __future__ import annotations
import argparse
from collections import Counter
import ctypes
from datetime import datetime, timezone
import hashlib
import heapq
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
DATA = ROOT / "work/data"
KIND = "TSC_TR_frame_target_metadata_HF_BPE_512_v1"
EXPECTED_CLIPS = 253798
EXPECTED_GROUPS = {"fleurs": 2037, "cv": 72846, "tsc": 178915}
BASE_FILE_SHA = "9c9f537bf7190ae2f0e30448a6915c4f607fa4d011b4f62b6f1849064ef8afcd"
BASE_SEMANTIC_SHA = "b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75"
CANDIDATE_FILE_SHA = "6c656b9dc5f85e3dd2c2abe16983908e5d2293da1b2ae096e92e1dc5551df327"
CANDIDATE_SEMANTIC_SHA = "10b747471a6613f818445f4c5697d04bb64ca45c48eb1bde7184d33e28e68607"
NORMALIZER_SHA = "433c31d9cbc4975c42d3a0fe659cd0bbd6c1caf2e1eb3d85169baafec7f61047"
ADAPTER_SHA = "839194dddf83c3b08bf608d01cb540b2f160feee426a91373d14cfc426ab7494"
SOURCE_KEYS = ("audio_filepath", "duration", "sample_rate", "audio_sha256", "wav_sha256",
               "language", "dataset", "dataset_revision", "dataset_config", "dataset_split",
               "dataset_file", "text", "reference", "reference_normalized", "training_role",
               "sentence_id", "original_audio_path")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def semantic(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def label_sha(ids):
    # Exact existing frame-metadata convention; intentional default separators.
    return hashlib.sha256(json.dumps(ids, sort_keys=True).encode("utf-8")).hexdigest()


def asset(path, expected=None):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(expected is None or digest == expected, "Pinned asset changed: " + str(path))
    return raw, {"path": str(Path(path).resolve()), "sha256": digest, "size_bytes": len(raw)}


def histogram_summary(hist):
    n = sum(hist.values())
    require(n > 0, "Empty source-group histogram")
    ordered = sorted(hist.items())
    def value_at(index):
        cumulative = 0
        for value, count in ordered:
            cumulative += count
            if cumulative > index:
                return value
        raise AssertionError("Histogram index outside rows")
    total = sum(value * count for value, count in ordered)
    return {"clips": n, "sum": total, "min": ordered[0][0], "max": ordered[-1][0],
            "mean": total / n,
            "median": (value_at((n-1)//2) + value_at(n//2)) / 2,
            "p90": value_at(min(n-1, int(.9*n))),
            "p99": value_at(min(n-1, int(.99*n)))}


def source_group(row):
    require(row["language"] == "tr" and row["dataset_split"] in ("train", "validated"),
            "Only approved Turkish TRAIN-role rows are eligible")
    if row["dataset_split"] == "validated":
        require(row.get("training_role") == "train", "CV validated row lacks TRAIN role")
    if row["dataset"] == "google/fleurs":
        require(row["dataset_split"] == "train" and row["dataset_config"] == "tr_tr"
                and row["dataset_revision"] == "70bb2e84b976b7e960aa89f1c648e09c59f894dd",
                "FLEURS TRAIN source identity differs")
        return "fleurs"
    if row["dataset"] == "ysdede/commonvoice_17_tr_fixed":
        require(row["dataset_revision"] == "9d8a025f1ae7c5a2f5c43298c7c4910b03995a4b",
                "Common Voice source identity differs")
        return "cv"
    require(row["dataset"] == "issai/Turkish_Speech_Corpus" and row["dataset_split"] == "train"
            and row.get("training_role") == "train"
            and row["dataset_revision"] == "d79ea01392882322e8f46657cb635236af25b6d4"
            and "ISSAI_TSC_218/Train/" in row["source_id"] and "/Dev/" not in row["source_id"]
            and "/Test/" not in row["source_id"], "TSC TRAIN identity differs")
    return "tsc"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA / "TSC_BPE512_TRAIN_metadata_v1")
    parser.add_argument("--report", type=Path, default=OUT / "TSC_BPE512_TRAIN_target_metadata_preparation_v1.json")
    args = parser.parse_args()
    directory, report = args.output_dir.resolve(), args.report.resolve()
    require(directory.parent == DATA and not directory.exists() and report.parent == OUT and not report.exists(),
            "Fresh inactive metadata directory and report required")
    failure_report = report.with_name(report.stem + "_failure.json")
    require(not failure_report.exists(), "Preserve the prior failure receipt")
    os.environ.update(TOKENIZERS_PARALLELISM="false", RAYON_NUM_THREADS="1",
                      OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    priority = {"platform": os.name, "applied": False}
    if os.name == "nt":
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.GetCurrentProcess.restype = ctypes.c_void_p
        api.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        api.SetPriorityClass.restype = ctypes.c_int
        api.GetPriorityClass.argtypes = [ctypes.c_void_p]
        api.GetPriorityClass.restype = ctypes.c_uint32
        handle = api.GetCurrentProcess()
        require(bool(api.SetPriorityClass(handle, 0x4000)) and api.GetPriorityClass(handle) == 0x4000,
                "BelowNormal priority must be applied")
        priority.update(applied=True, priority_class=0x4000)
    started = time.perf_counter()
    # These modules are stdlib/Rust only; the training/frame helpers are never imported.
    import tokenizers
    sys.path.insert(0, str(OUT))
    from metrics import normalize_text
    require(not any(name in sys.modules for name in ("torch", "transformers", "nemo", "soundfile", "torchaudio")),
            "Model/audio libraries must not be imported")
    script_raw, script_record = asset(__file__)
    _, normalizer_record = asset(OUT / "metrics.py", NORMALIZER_SHA)
    adapter_raw, adapter_record = asset(OUT / "native_tdt_checkpointed_v3.py", ADAPTER_SHA)
    # Verify the actual source guard expression without importing its Torch/NeMo module.
    import ast
    adapter_tree = ast.parse(adapter_raw.decode("utf-8"))
    guarded = [node for node in ast.walk(adapter_tree) if isinstance(node, ast.Compare)
               and ast.unparse(node) == "t * (u + 1) * (vocab_size + len(durations)) * 4 <= max_joint_bytes"]
    require(len(guarded) == 1, "Actual native adapter joint-bound expression changed")
    proof_raw, proof_record = asset(OUT / "TSC_quality_inputs_preparation_v2.json")
    proof = json.loads(proof_raw)
    require(proof["kind"] == "TSC_supplemental_TR_frame_target_inputs_v2" and proof["passed"] is True
            and proof["clips"] == EXPECTED_CLIPS, "Complete approved combined input proof required")
    manifest = Path(proof["files"]["combined_TR"]["path"]).resolve()
    parent_metadata = Path(proof["files"]["TR_metadata"]["path"]).resolve()
    require(manifest == DATA / "TSC_quality_inputs_v2/combined_TR_train.jsonl"
            and parent_metadata == DATA / "TSC_quality_inputs_v2/combined_TR_frame_metadata.jsonl",
            "Only the current approved paired TRAIN inputs are admitted")
    manifest_expected, parent_expected = proof["files"]["combined_TR"]["sha256"], proof["files"]["TR_metadata"]["sha256"]
    before_stats = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in (manifest, parent_metadata)}
    old_raw, baseline_record = asset(ROOT / "work/models/lora_20261010_01_merged/tokenizer.json", BASE_FILE_SHA)
    new_raw, candidate_record = asset(ROOT / "work/hf_turkish_bpe_candidate_512_v1/tokenizer.json", CANDIDATE_FILE_SHA)
    old_json, new_json = json.loads(old_raw), json.loads(new_raw)
    require(semantic(old_json) == BASE_SEMANTIC_SHA and semantic(new_json) == CANDIDATE_SEMANTIC_SHA,
            "Frozen tokenizer semantic identity differs")
    require(len(old_json["model"]["vocab"]) == 8199 and len(new_json["model"]["vocab"]) == 8711
            and all(new_json["model"]["vocab"].get(piece) == identifier
                    for piece, identifier in old_json["model"]["vocab"].items()), "Original8199 IDs/pieces moved")
    require(new_json["model"]["merges"][:len(old_json["model"]["merges"])] == old_json["model"]["merges"]
            and {k:v for k,v in old_json.items() if k != "model"} == {k:v for k,v in new_json.items() if k != "model"},
            "Merge prefix/normalizer/pretokenizer/decoder/special settings differ")
    baseline = tokenizers.Tokenizer.from_str(old_raw.decode("utf-8"))
    candidate = tokenizers.Tokenizer.from_str(new_raw.decode("utf-8"))
    require(baseline.id_to_token(8192) == candidate.id_to_token(8192) == "<blank>"
            and baseline.get_vocab_size() == 8199 and candidate.get_vocab_size() == 8711,
            "Interior blank/vocabulary geometry changed")
    unk = baseline.token_to_id("<unk>")
    require(candidate.token_to_id("<unk>") == unk, "Original unknown ID changed")
    directory.mkdir()
    final_metadata = directory / "candidate_TR_frame_target_metadata.jsonl"
    pending = directory / "candidate_TR_frame_target_metadata.pending.jsonl"
    digests = {name: hashlib.sha256() for name in ("manifest", "parent_metadata", "output", "ordered_ids",
                                                "manifest_rows", "parent_geometry", "candidate_geometry")}
    groups = {name: {"counts": Counter(), "baseline_U": Counter(), "candidate_U": Counter(),
                     "new_piece_counts": Counter(), "PCM_frames": 0, "encoder_frames": 0,
                     "scored_characters": 0} for name in EXPECTED_GROUPS}
    joint_top = []
    joint_top_old = []
    examples = []
    current = {"row_index": None, "source_id": None}
    def add_peak(heap, value, index, details):
        entry = (value, index, details)
        if len(heap) < 8:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)
    try:
        with manifest.open("rb") as left, parent_metadata.open("rb") as right, pending.open("xb") as target:
            for index, (row_raw, parent_raw) in enumerate(zip_longest(left, right)):
                current = {"row_index": index, "source_id": None}
                require(row_raw is not None and parent_raw is not None and row_raw.strip() and parent_raw.strip(),
                        "Paired input row count/blank-line mismatch")
                digests["manifest"].update(row_raw); digests["parent_metadata"].update(parent_raw)
                row, parent = json.loads(row_raw), json.loads(parent_raw)
                current["source_id"] = row["source_id"]
                require(row["source_id"] == parent["source_id"], "Ordered source/metadata identity differs")
                group = source_group(row)
                require((index < 74883 and group in ("cv", "fleurs")) or (index >= 74883 and group == "tsc"),
                        "Original74883 prefix or quarantined TSC suffix order changed")
                require(row["sample_rate"] == 16000 and math.isfinite(row["duration"])
                        and .5 <= row["duration"] <= 30 and isinstance(row["text"], str), "Approved raw TRAIN dimensions differ")
                pcm, mel, frames = parent["actual_PCM_frames"], parent["valid_mel_frames"], parent["encoder_frames"]
                require(type(pcm) is int and type(mel) is int and type(frames) is int and pcm > 0
                        and int(round(row["duration"] * 16000)) == pcm and mel == pcm // 160
                        and frames == (mel + 7) // 8 and parent["teacher_feature_values_required_for_TR_preservation"] is False,
                        "Copied audited PCM/mel/encoder dimensions disagree")
                old = baseline.encode(row["text"], add_special_tokens=False).ids
                new = candidate.encode(row["text"], add_special_tokens=False).ids
                current.update(baseline_U=len(old), parent_U=parent["target_tokens"], candidate_U=len(new),
                               baseline_label_sha256=label_sha(old), parent_label_sha256=parent["label_ids_sha256"])
                require(1 <= len(old) <= 180 and min(old) >= 0 and max(old) < 8199 and 8192 not in old
                        and len(old) == parent["target_tokens"] and label_sha(old) == parent["label_ids_sha256"],
                        "Exact baseline audited U/label IDs mismatch; stop publication")
                require(1 <= len(new) <= 180 and min(new) >= 0 and max(new) < 8711 and 8192 not in new,
                        "Candidate cannot retain this accepted row within max180; no truncate/drop allowed")
                decoded_old = baseline.decode(old, skip_special_tokens=True)
                decoded_new = candidate.decode(new, skip_special_tokens=True)
                normalized = normalize_text(row["text"], "tr")
                require(normalize_text(decoded_old, "tr") == normalized and normalize_text(decoded_new, "tr") == normalized
                        and decoded_old == decoded_new, "Baseline/candidate raw-decode or normalized raw-reference roundtrip differs")
                require(old.count(unk) == new.count(unk), "Candidate changed unknown-token count")
                copied = {key:value for key,value in parent.items() if key not in ("target_tokens", "label_ids_sha256")}
                manifest_row_digest = hashlib.sha256(canonical(row)).hexdigest()
                item = {**parent, "target_tokens":len(new), "label_ids_sha256":label_sha(new),
                        "kind":KIND, "manifest_row_index":index, "label_ids":new,
                        "baseline_target_tokens":len(old), "baseline_label_ids_sha256":parent["label_ids_sha256"],
                        "training_manifest_row_sha256":manifest_row_digest,
                        "text_sha256":hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
                        **{key:row[key] for key in SOURCE_KEYS if key in row}}
                require(all(item[key] == value for key,value in copied.items()), "Non-target audited metadata changed")
                # All original metadata fields and source fields are copied; full original row is bound by its hash.
                digests["parent_geometry"].update(canonical(copied) + b"\n")
                digests["candidate_geometry"].update(canonical({key:item[key] for key in copied}) + b"\n")
                digests["ordered_ids"].update((row["source_id"] + "\n").encode("utf-8"))
                digests["manifest_rows"].update(canonical(row) + b"\n")
                encoded = canonical(item) + b"\n"
                target.write(encoded); digests["output"].update(encoded)
                stats = groups[group]; c = stats["counts"]
                c.update(clips=1, baseline_label_hashes_reproduced=1, candidate_roundtrip_passed=1,
                         baseline_unknown_tokens=old.count(unk), candidate_unknown_tokens=new.count(unk),
                         encoding_changed=int(old != new), target_length_changed=int(len(old) != len(new)),
                         shorter_target=int(len(new) < len(old)), longer_target=int(len(new) > len(old)),
                         rows_with_new512=int(any(i >= 8199 for i in new)),
                         raw_reference_exact_decode=int(decoded_new == row["text"]))
                stats["baseline_U"][len(old)] += 1; stats["candidate_U"][len(new)] += 1
                stats["new_piece_counts"].update(i for i in new if i >= 8199)
                stats["PCM_frames"] += pcm; stats["encoder_frames"] += frames
                stats["scored_characters"] += len(normalized)
                old_bound = frames * (len(old) + 1) * (8199 + 5) * 4
                new_bound = frames * (len(new) + 1) * (8711 + 5) * 4
                details = {"row_index":index, "source_id":row["source_id"], "source_group":group,
                           "duration":row["duration"], "encoder_frames":frames,
                           "baseline_U":len(old), "candidate_U":len(new),
                           "baseline_guard_joint_bytes":old_bound, "candidate_guard_joint_bytes":new_bound}
                add_peak(joint_top, new_bound, index, details); add_peak(joint_top_old, old_bound, index, details)
                if len(examples) < 6 and old != new:
                    examples.append({"source_id":row["source_id"], "source_group":group, "baseline_U":len(old),
                                     "candidate_U":len(new), "reference":row["text"], "candidate_ids":new})
                if (index + 1) % 25000 == 0:
                    print(json.dumps({"completed_rows":index+1,"elapsed_seconds":time.perf_counter()-started}), flush=True)
        counts = {group:stats["counts"]["clips"] for group,stats in groups.items()}
        require(counts == EXPECTED_GROUPS and sum(counts.values()) == EXPECTED_CLIPS, "Whole ordered pool count differs")
        require(digests["manifest"].hexdigest() == manifest_expected and digests["parent_metadata"].hexdigest() == parent_expected,
                "Whole streamed manifest/parent metadata SHA differs from existing proof")
        require(digests["manifest_rows"].hexdigest() == proof["identity"]["combined_TR_ordered_rows_sha256"],
                "Whole canonical manifest-row identity differs")
        require(digests["parent_geometry"].digest() == digests["candidate_geometry"].digest(),
                "Audited non-target metadata hash differs")
        require(all(before_stats[str(path)] == (path.stat().st_size, path.stat().st_mtime_ns)
                    for path in (manifest, parent_metadata)), "Input changed while streamed")
        require(not any(name in sys.modules for name in ("torch", "transformers", "nemo", "soundfile", "torchaudio")),
                "Model/audio libraries unexpectedly imported")
        # Same directory, complete-file publication only after every row/input hash passed.
        pending.rename(final_metadata)
    except Exception as exc:
        failure = {"schema":1, "kind":"TSC_BPE512_TRAIN_metadata_preparation_failure_v1", "passed":False,
                   "CPU_only":True, "complete_metadata_published":False, "candidate_activated":False,
                   "script":script_record, "source_proof":proof_record, "last_row":current,
                   "error_type":type(exc).__name__, "error":str(exc), "pending_metadata_path":str(pending),
                   "partial_counts":{g:dict(s["counts"]) for g,s in groups.items()},
                   "elapsed_seconds":time.perf_counter()-started}
        with failure_report.open("x", encoding="utf-8") as stream:
            json.dump(failure, stream, ensure_ascii=False, indent=2); stream.write("\n")
        print(json.dumps({"failure_report":str(failure_report), "error":str(exc)}, ensure_ascii=True), flush=True)
        raise
    aggregate_old, aggregate_new, aggregate_counts, aggregate_pieces = Counter(), Counter(), Counter(), Counter()
    result_groups = {}
    for name, stats in groups.items():
        old_summary, new_summary = histogram_summary(stats["baseline_U"]), histogram_summary(stats["candidate_U"])
        result_groups[name] = {"counts":dict(stats["counts"]), "baseline_target_tokens":old_summary,
                              "candidate_target_tokens":new_summary,
                              "target_token_sum_reduction_fraction":1-new_summary["sum"]/old_summary["sum"],
                              "actual_PCM_frames":stats["PCM_frames"], "actual_PCM_hours":stats["PCM_frames"]/16000/3600,
                              "encoder_frames_total":stats["encoder_frames"], "scored_characters":stats["scored_characters"],
                              "baseline_scored_characters_per_token":stats["scored_characters"]/old_summary["sum"],
                              "candidate_scored_characters_per_token":stats["scored_characters"]/new_summary["sum"],
                              "new512_distinct_pieces_used":len(stats["new_piece_counts"]),
                              "new512_token_occurrences":sum(stats["new_piece_counts"].values()),
                              "new512_piece_counts":dict(sorted(stats["new_piece_counts"].items()))}
        aggregate_old.update(stats["baseline_U"]); aggregate_new.update(stats["candidate_U"])
        aggregate_counts.update(stats["counts"]); aggregate_pieces.update(stats["new_piece_counts"])
    old_summary, new_summary = histogram_summary(aggregate_old), histogram_summary(aggregate_new)
    result = {"schema":1, "kind":"TSC_BPE512_whole_TRAIN_target_metadata_preparation_v1", "passed":True,
              "CPU_only":True, "created_at_utc":datetime.now(timezone.utc).isoformat(), "candidate_activated":False,
              "fit_performed":False, "DEV_or_TEST_inputs_read":False, "audio_files_or_APIs_read":False,
              "Torch_Transformers_NeMo_imported":False, "original_assets_or_labels_modified":False,
              "script":script_record, "source_proof":proof_record, "normalizer":normalizer_record,
              "baseline_tokenizer":{**baseline_record,"semantic_sha256":BASE_SEMANTIC_SHA},
              "candidate_tokenizer":{**candidate_record,"semantic_sha256":CANDIDATE_SEMANTIC_SHA},
              "manifest":{**proof["files"]["combined_TR"],"actual_streamed_sha256":digests["manifest"].hexdigest()},
              "parent_metadata":{**proof["files"]["TR_metadata"],"actual_streamed_sha256":digests["parent_metadata"].hexdigest()},
              "metadata_path":str(final_metadata), "metadata_sha256":digests["output"].hexdigest(),
              "metadata_size_bytes":final_metadata.stat().st_size, "metadata_kind":KIND, "clips":EXPECTED_CLIPS,
              "counts":dict(aggregate_counts), "per_source":result_groups,
              "baseline_target_tokens":old_summary, "candidate_target_tokens":new_summary,
              "target_token_sum_reduction_fraction":1-new_summary["sum"]/old_summary["sum"],
              "new512_distinct_pieces_used":len(aggregate_pieces), "new512_token_occurrences":sum(aggregate_pieces.values()),
              "ordered_source_ids_newline_sha256":digests["ordered_ids"].hexdigest(),
              "canonical_manifest_rows_sha256":digests["manifest_rows"].hexdigest(),
              "copied_non_target_parent_fields_sha256":digests["parent_geometry"].hexdigest(),
              "regenerated_non_target_parent_fields_sha256":digests["candidate_geometry"].hexdigest(),
              "source_ID_order_text_reference_audio_paths_PCM_mel_T_and_other_parent_fields_preserved":True,
              "clips_dropped_or_truncated":0, "whole_input_files_streamed_once":True, "output_hash_computed_while_written":True,
              "geometry":{"baseline_vocab":8199,"candidate_vocab":8711,"original8199_ID_piece_prefix_exact":True,
                          "blank_id_unchanged":8192,"six_glyph_IDs_unchanged":[8193,8199],
                          "new512_IDs":[8199,8711],"duration_IDs_after_expansion":[8711,8716],
                          "teacher_original_vocab_unchanged":8193,"max_target_tokens":180,
                          "normalizer_preprocessor_decoder_specials_and_original_merge_prefix_unchanged":True},
              "roundtrip_policy":{"raw_text_encoded_without_pre_normalizing":True,"add_special_tokens":False,
                                  "raw_backend_decoding_without_CTC_grouping":True,
                                  "candidate_raw_decode_equals_baseline_for_every_row":True,
                                  "both_normalized_decodes_equal_normalized_original_raw_TRAIN_text":True,
                                  "candidate_unknown_count_equals_baseline_per_row":True,
                                  "unknown_surface_punctuation_loss_not_claimed_absent":True},
              "native_joint_guard":{"adapter":adapter_record,"actual_expression_line":guarded[0].lineno,
                                    "formula":"encoder_frames * (target_tokens + 1) * (vocab_size + len(durations)) * 4",
                                    "candidate_width":8716,"durations":[0,1,2,3,4],
                                    "maximum_candidate_bytes":max(entry[0] for entry in joint_top),
                                    "largest_candidate_rows":[entry[2] for entry in sorted(joint_top,reverse=True)],
                                    "maximum_baseline_bytes":max(entry[0] for entry in joint_top_old),
                                    "largest_baseline_rows":[entry[2] for entry in sorted(joint_top_old,reverse=True)],
                                    "scope":"Exact native per-clip acceptance-guard arithmetic; not measured VRAM/temporary/backward/whole-batch fit. Adapter3 splits packed outputs before FP32 casting."},
              "finite_examples":examples,
              "runtime":{"elapsed_seconds":time.perf_counter()-started,"priority":priority,
                         "tokenizers_version":tokenizers.__version__,"TOKENIZERS_PARALLELISM":"false","RAYON_NUM_THREADS":"1"},
              "can_current_frozen_BPE_worker_consume_metadata":False,
              "integration_gaps":["Frozen BPE reader/helper/worker pin original74883-only kind/manifest/source admission; this whole combined253798 metadata requires explicit future admission, not a CLI override.",
                                  "Original replay1595 labels/cache/teacher8193 are external and untouched; no replay retokenization was performed.",
                                  "No model row expansion, optimizer conversion/acquisition, GPU initial300 equality or native loss execution performed."],
              "scope":"Inactive whole approved combined Turkish TRAIN target preparation only; no fit, ASR quality or GPU-fit claim."}
    with report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write("\n")
    print(json.dumps({"report":str(report), "clips":EXPECTED_CLIPS, "metadata_sha256":result["metadata_sha256"],
                      "token_reduction_fraction":result["target_token_sum_reduction_fraction"],
                      "max_candidate_joint_bytes":result["native_joint_guard"]["maximum_candidate_bytes"],
                      "elapsed_seconds":result["runtime"]["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
