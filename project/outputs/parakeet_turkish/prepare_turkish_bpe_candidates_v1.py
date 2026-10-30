"""INERT append-only native Turkish BPE candidates; SentencePiece/protobuf CPU only.

Never change the active tokenizer or load Torch/NeMo/model weights. New merges
are learned only from accepted Turkish TRAIN/validated refs. A deterministic
normalized-text holdout is excluded from learning. DEV is evaluation only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
import math
from pathlib import Path
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
SEED = 20261010
LANGUAGES = ("en", "es", "it", "fr", "de", "tr")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def lexical(text, lang):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("I", "ı").replace("İ", "i") if lang == "tr" else text
    text = text.lower().replace("i\u0307", "i")
    return " ".join("".join(" " if unicodedata.category(c)[0] in {"Z", "C"}
                             else "" if unicodedata.category(c)[0] == "P" else c
                             for c in text).split())


def holdout(normalized):
    # Duplicate normalized references stay in the same partition.
    digest = hashlib.sha256((str(SEED) + "|" + normalized).encode()).digest()
    return int.from_bytes(digest[:8], "big") % 10 == 0


def words(pieces):
    current = []
    for piece in pieces:
        if "▁" in piece and current:
            yield tuple(current)
            current = []
        current.append(piece)
    if current:
        yield tuple(current)


def learn_merges(word_counts, baseline_pieces, count):
    """Continue BPE on weighted base segmentations, leaving old ranks exact."""
    sequences = [list(w) for w in word_counts]
    frequencies = [word_counts[w] for w in word_counts]
    counts = Counter()
    members = defaultdict(set)

    def permitted(pair):
        piece = "".join(pair)
        letters = piece[1:] if piece.startswith("▁") else piece
        return (1 < len(letters) <= 24 and "▁" not in letters
                and piece not in baseline_pieces and letters.isalpha()
                and all("LATIN" in unicodedata.name(c, "") for c in letters))

    def pairs(seq):
        return Counter(pair for pair in zip(seq, seq[1:]) if permitted(pair))

    for i, sequence in enumerate(sequences):
        for pair, occurrences in pairs(sequence).items():
            counts[pair] += occurrences * frequencies[i]
            members[pair].add(i)
    heap = [(-frequency, "".join(pair), pair) for pair, frequency in counts.items()]
    heapq.heapify(heap)
    additions, selected = [], set()
    while len(additions) < count:
        while heap:
            negative, combined, pair = heapq.heappop(heap)
            if counts[pair] == -negative and counts[pair] > 0 and combined not in selected:
                break
        else:
            raise RuntimeError("Not enough eligible TRAIN-derived BPE merges.")
        learned_frequency = counts[pair]
        affected = sorted(members[pair])
        touched = set()
        for i in affected:
            old_pairs = pairs(sequences[i])
            for old, occurrences in old_pairs.items():
                counts[old] -= occurrences * frequencies[i]
                members[old].discard(i)
                touched.add(old)
            previous, merged = sequences[i], []
            j = 0
            while j < len(previous):
                if j + 1 < len(previous) and (previous[j], previous[j + 1]) == pair:
                    merged.append(combined)
                    j += 2
                else:
                    merged.append(previous[j])
                    j += 1
            sequences[i] = merged
            for new, occurrences in pairs(merged).items():
                counts[new] += occurrences * frequencies[i]
                members[new].add(i)
                touched.add(new)
        selected.add(combined)
        additions.append({"piece": combined, "left": pair[0], "right": pair[1],
                          "weighted_pair_frequency_at_learning": learned_frequency})
        for changed in touched:
            if counts[changed] > 0 and "".join(changed) not in selected:
                heapq.heappush(heap, (-counts[changed], "".join(changed), changed))
        if len(additions) % 128 == 0:
            print(json.dumps({"phase": "TRAIN_only_BPE_learning", "merges": len(additions)}), flush=True)
    return additions


def stats(values):
    values = sorted(values)
    n = len(values)
    return {"clips": n, "sum": sum(values), "mean": sum(values) / n if n else None,
            "median": values[n // 2] if n else None,
            "p90": values[min(n - 1, int(.9 * n))] if n else None,
            "p99": values[min(n - 1, int(.99 * n))] if n else None,
            "max": max(values) if n else None}


def evaluate(rows, baseline, candidate, language):
    old_lengths, new_lengths = [], []
    raw_match = normalized_match = same_decode = changed = unknown = 0
    regressions, differences = [], []
    for row in rows:
        text = str(row.get("text", row.get("reference", "")))
        old, new = baseline.encode(text), candidate.encode(text)
        old_decode, new_decode = baseline.decode(old), candidate.decode(new)
        old_lengths.append(len(old)); new_lengths.append(len(new))
        raw_match += new_decode == text
        normalized_match += lexical(new_decode, language) == lexical(text, language)
        same_decode += new_decode == old_decode
        changed += new != old
        unknown += candidate.unk_id() in new
        if new_decode != old_decode and len(regressions) < 5:
            regressions.append({"source_id": row["source_id"], "reference": text,
                                "baseline_decode": old_decode, "candidate_decode": new_decode})
        if new != old and len(differences) < 5:
            differences.append({"source_id": row["source_id"], "reference": text,
                                "old_U": len(old), "new_U": len(new),
                                "old_pieces": baseline.encode(text, out_type=str),
                                "new_pieces": candidate.encode(text, out_type=str)})
    return {"baseline_U": stats(old_lengths), "candidate_U": stats(new_lengths),
            "token_count_reduction_fraction": 1 - sum(new_lengths) / sum(old_lengths) if sum(old_lengths) else 0,
            "encoding_changed_clips": changed, "raw_exact_roundtrip_clips": raw_match,
            "raw_exact_roundtrip_mismatch_clips": len(old_lengths) - raw_match,
            "ASR_normalized_roundtrip_clips": normalized_match,
            "decoded_text_equals_baseline_clips": same_decode,
            "decoded_text_changes_from_baseline": len(old_lengths) - same_decode,
            "unknown_token_clips": unknown, "decoded_text_regression_examples": regressions,
            "encoding_change_examples": differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", default=str(ROOT / "work/turkish_bpe_candidates_v1"))
    parser.add_argument("--report", default=str(OUT / "turkish_bpe_candidates_v1.json"))
    args = parser.parse_args()
    destination, report_path = Path(args.destination), Path(args.report)
    if destination.exists() or report_path.exists():
        raise FileExistsError("Fresh inert candidate paths are required.")
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as pb

    started = time.perf_counter()
    base_path = ROOT / "work/nemo_direct_trial_v1/tokenizer/tokenizer.model"
    train_path = ROOT / "work/data/tr_public_v3_train.jsonl"
    pilot_path = ROOT / "work/nemo_direct_trial_v1/manifests/train_balanced_250.jsonl"
    base_bytes = base_path.read_bytes()
    proto = pb.ModelProto.FromString(base_bytes)
    if len(proto.pieces) != 8198 or proto.trainer_spec.model_type != 2:
        raise ValueError("Expected current native8198piece BPE with six character additions.")
    baseline = spm.SentencePieceProcessor(model_proto=base_bytes)
    heldout_count = learning_count = 0
    learned_source_digest, heldout_source_digest = hashlib.sha256(), hashlib.sha256()
    word_counts = Counter()
    for row in json_rows(train_path):
        if row["language"] != "tr" or row["dataset_split"] not in {"train", "validated"}:
            raise ValueError("Candidate learning must use Turkish TRAIN/validated only.")
        text = row["text"]
        if holdout(baseline.normalize(text)):
            heldout_count += 1
            heldout_source_digest.update((row["source_id"] + "\n").encode())
        else:
            learning_count += 1
            learned_source_digest.update((row["source_id"] + "\n").encode())
            word_counts.update(words(baseline.encode(text, out_type=str)))
    print(json.dumps({"phase": "TRAIN_partition_complete", "learning_clips": learning_count,
                      "heldout_TRAIN_clips": heldout_count, "weighted_word_types": len(word_counts)}), flush=True)
    merges = learn_merges(word_counts, {p.piece for p in proto.pieces}, 1024)
    del word_counts
    destination.mkdir(parents=True)
    partition = {"rule": "SHA256(seed|SentencePiece-normalized-TRAIN-text) first8bytes asbigendian modulo10; zero=heldout",
                 "seed": SEED, "learning_clips": learning_count, "heldout_TRAIN_clips": heldout_count,
                 "learning_source_ID_order_sha256": learned_source_digest.hexdigest(),
                 "heldout_TRAIN_source_ID_order_sha256": heldout_source_digest.hexdigest(),
                 "duplicate_normalized_texts_stay_in_same_partition": True,
                 "learning_input_path": str(train_path), "learning_input_sha256": sha(train_path),
                 "no_DEV_or_TEST_text_read_before_candidates_frozen": True}
    (destination / "learning_partition.json").write_text(json.dumps(partition, indent=2), encoding="utf-8")
    candidate_records = {}
    minimum = min(p.score for p in proto.pieces)
    for count in (512, 1024):
        model = pb.ModelProto(); model.CopyFrom(proto)
        added = []
        for i, merge in enumerate(merges[:count]):
            piece = model.pieces.add()
            piece.piece, piece.score, piece.type = merge["piece"], minimum - i - 1, pb.ModelProto.SentencePiece.NORMAL
            added.append(dict(merge, native_id=8198 + i, score=piece.score))
        model.trainer_spec.vocab_size = len(model.pieces)
        restored = pb.ModelProto(); restored.CopyFrom(model)
        del restored.pieces[8198:]
        restored.trainer_spec.vocab_size = proto.trainer_spec.vocab_size
        if restored.SerializeToString() != base_bytes:
            raise AssertionError("Original8198prefix/protobuf metadata changed.")
        directory = destination / f"add_{count}"
        directory.mkdir()
        model_path = directory / "tokenizer.model"
        model_path.write_bytes(model.SerializeToString())
        for source_name, lines in (("tokenizer.vocab", [f"{x['piece']}\t{x['score']:g}" for x in added]),
                                   ("vocab.txt", [x["piece"] for x in added])):
            original = (base_path.parent / source_name).read_bytes()
            prefix = original + (b"" if original.endswith(b"\n") else b"\n")
            (directory / source_name).write_bytes(prefix + ("\n".join(lines) + "\n").encode())
        (directory / "added_pieces.json").write_text(json.dumps(added, ensure_ascii=False, indent=2), encoding="utf-8")
        candidate_records[str(count)] = {"tokenizer_dir": str(directory), "vocabulary_size": len(model.pieces),
            "original8192_and_current6_piece_ID_score_type_prefix_bitexact": True,
            "all_other_proto_fields_exact_after_restoring_only_declared_vocab_size": True,
            "blank_id_for_future_native_model": len(model.pieces),
            "duration_rows_for_future_native_model": list(range(len(model.pieces) + 1, len(model.pieces) + 6)),
            "files_sha256": {str(p): sha(p) for p in directory.iterdir()},
            "extra_trainable_head_embedding_parameters": count * (640 + 640 + 1),
            "extra_FP32_parameters_plus_Adam_moments_bytes": count * (640 + 640 + 1) * 12,
            "new_head_embedding_rows_initialized_or_trained": False}
    (destination / "candidate_files_frozen_before_DEV_read.json").write_text(
        json.dumps({k: x["files_sha256"] for k, x in candidate_records.items()}, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "candidates_fixed", "DEV_or_TEST_used_for_learning": False}), flush=True)

    # Evaluation begins only after every candidate and merge priority is fixed.
    pilot = list(json_rows(pilot_path))
    old_shapes = json.loads((OUT / "native_pilot_lattice_memory_review_v1.json").read_text())
    frame_lengths = [t for b in old_shapes["all_microbatch_estimates"] for t, _ in b["clip_shapes"]]
    if sha(pilot_path) != old_shapes["manifest_sha256"] or len(frame_lengths) != len(pilot):
        raise ValueError("Pilot T/U provenance no longer matches.")
    evaluation_inputs = {str(pilot_path): sha(pilot_path)}
    for count, record in candidate_records.items():
        candidate = spm.SentencePieceProcessor(model_file=str(Path(record["tokenizer_dir"]) / "tokenizer.model"))
        record["accepted_TRAIN_all"] = evaluate(json_rows(train_path), baseline, candidate, "tr")
        record["heldout_TRAIN_text_disjoint"] = evaluate(
            (r for r in json_rows(train_path) if holdout(baseline.normalize(r["text"]))), baseline, candidate, "tr")
        record["pilot_Turkish"] = evaluate((r for r in pilot if r["language"] == "tr"), baseline, candidate, "tr")
        record["DEV_by_language"] = {}
        for lang in LANGUAGES:
            path = ROOT / f"work/nemo_direct_trial_v1/manifests/dev_{lang}.jsonl"
            evaluation_inputs[str(path)] = sha(path)
            record["DEV_by_language"][lang] = evaluate(json_rows(path), baseline, candidate, lang)
        old_costs, new_costs, matrix_rows = [], [], []
        for i, (row, t) in enumerate(zip(pilot, frame_lengths)):
            old_U, new_U = len(baseline.encode(row["text"])), len(candidate.encode(row["text"]))
            old_bytes = t * (old_U + 1) * 8204 * 4
            new_bytes = t * (new_U + 1) * (record["vocabulary_size"] + 6) * 4
            old_costs.append(old_bytes); new_costs.append(new_bytes)
            matrix_rows.append({"manifest_index": i, "source_id": row["source_id"], "language": row["language"],
                                "T": t, "baseline_U": old_U, "candidate_U": new_U,
                                "baseline_joint_FP32_bytes": old_bytes, "candidate_joint_FP32_bytes": new_bytes})
        record["pilot_all_languages_lattice_projection"] = {
            "method": "actual SentencePiece U and previously header-verified T, FP32 T*(U+1)*(new_vocab+blank+5durations)*4",
            "baseline_joint_matrix_costs": stats(old_costs), "candidate_joint_matrix_costs": stats(new_costs),
            "summed_matrix_byte_reduction_fraction": 1 - sum(new_costs) / sum(old_costs),
            "max_matrix_byte_reduction_fraction": 1 - max(new_costs) / max(old_costs),
            "top10_candidate_singletons": sorted(matrix_rows, key=lambda x: x["candidate_joint_FP32_bytes"], reverse=True)[:10],
            "not_measured_GPU_peak_memory_or_training_throughput": True}
        matrix_path = Path(record["tokenizer_dir"]) / "pilot_lattice_projection.jsonl"
        matrix_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in matrix_rows), encoding="utf-8")
        record["pilot_lattice_projection_path"] = str(matrix_path)
        record["pilot_lattice_projection_sha256"] = sha(matrix_path)
        if any(x["decoded_text_changes_from_baseline"] for x in [record["accepted_TRAIN_all"], *record["DEV_by_language"].values()]):
            record["roundtrip_preservation_passed"] = False
        else:
            record["roundtrip_preservation_passed"] = True
        print(json.dumps({"phase": "CPU_candidate_evaluated", "added": count,
                          "heldout_TRAIN_U_reduction": record["heldout_TRAIN_text_disjoint"]["token_count_reduction_fraction"],
                          "pilot_matrix_byte_reduction": record["pilot_all_languages_lattice_projection"]["summed_matrix_byte_reduction_fraction"]}), flush=True)
    for record in candidate_records.values():
        for path, expected in record["files_sha256"].items():
            if sha(path) != expected:
                raise AssertionError("Candidate changed during evaluation.")
    if sha(base_path) != hashlib.sha256(base_bytes).hexdigest():
        raise AssertionError("Active tokenizer changed.")
    result = {"schema": 1, "kind": "INERT_TRAIN_only_Turkish_BPE_append_candidates_v1", "CPU_only": True,
              "Torch_NeMo_imported": "torch" in sys.modules or "nemo" in sys.modules,
              "active_tokenizer_unchanged": True, "candidate_activation_authorized_or_performed": False,
              "baseline_tokenizer_path": str(base_path), "baseline_tokenizer_sha256": sha(base_path),
              "baseline_vocab_size": 8198, "learning_partition": partition,
              "learning_algorithm": "Weighted frequent adjacent base-piece BPE merges restricted to Latin alphabet inside word; score below every original score; prefix512 subset of1024",
              "scores_are_BPE_merge_priorities_not_unigram_log_probabilities": True,
              "evaluation_inputs_sha256": evaluation_inputs, "candidates": candidate_records,
              "research_review": {"v2_v3_vocabulary_comparison": str(OUT / "v2_v3_vocabulary_comparison.json"),
                                  "v2_v3_conclusion": "v2/v3 do not preserve IDs; this candidate deliberately preserves original8192 and current6"},
              "primary_sources": ["https://github.com/google/sentencepiece/blob/master/src/bpe_model.cc",
                                  "https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto",
                                  "https://github.com/google/sentencepiece/blob/master/doc/model_proto.md"],
              "limitations": ["No weights resized, no training, no WER claim, no optimizer migration; current active run unchanged.",
                              "Old IDs/scores preserved does not imply old-language encodings or normalized probabilities unchanged.",
                              "Future native blank and duration rows must move exactly, old/current lexical rows stay exact; new embeddings/output rows need explicit initialization and fresh optimizer state.",
                              "Header/T/U arithmetic excludes saved graph/contiguous/native gradient/encoder/optimizer storage; measured GPU stress and DEV retention gates remain necessary.",
                              "Raw normalization/unknown roundtrip limitations are reported rather than dropping refs; candidates preserve baseline decoded text if tests pass.",
                              "No TEST text was read; DEV is roundtrip/encoding diagnostic only and did not influence piece/rank learning."],
              "script_sha256": sha(__file__), "elapsed_CPU_seconds": time.perf_counter() - started}
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "candidate_directory": str(destination), "CPU_only": True}), flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
