"""Cached-text-only sensitivity audit; never changes an admission/scoring gate."""
from __future__ import annotations

import ast
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
STORE = Path(r"E:\gpu_training_scratchpad")
OUT = STORE / "analysis/CPU_validation_v5"
NORMALIZER = ROOT / "work/research/turkish_normalization_v1/evaluate.py"
EXPECTED_NORMALIZER_SHA = "52545e91070c9ab6dc80bd27760af278921d6cb5f9eb56f0d4943824c1932323"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "outputs/parakeet_turkish"))
sys.path.insert(0, str(STORE / "scratch/CPU_validation_v2_env/Lib/site-packages"))
import psutil
from metrics import edit_distance, normalize_text


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_spoken():
    """Execute exact frozen transform nodes, excluding old main/import paths."""
    assert sha(NORMALIZER) == EXPECTED_NORMALIZER_SHA
    vendor = NORMALIZER.parent / "vendor"
    receipt = json.loads((NORMALIZER.parent / "vendor_receipt.json").read_text(encoding="utf-8"))
    for relative, size, digest in receipt["files"]:
        source = vendor / relative
        assert source.stat().st_size == size and sha(source) == digest, relative
    sys.path.insert(0, str(vendor))
    from trnorm.num_to_text import convert_numbers_to_words_wrapper
    from trnorm.ordinals import num_to_text as ordinal_words
    names = {"NUMBER", "NUM_RE", "UNITS", "UNIT_RE", "EXPLICIT_ORDINAL", "RANK_ORDINAL", "REGNAL"}
    parsed = ast.parse(NORMALIZER.read_text(encoding="utf-8-sig"))
    nodes = [node for node in parsed.body
             if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))
             or (isinstance(node, ast.FunctionDef) and node.name in {"frozen", "spoken", "fixtures"})]
    namespace = {"re": re, "unicodedata": unicodedata,
                 "convert_numbers_to_words_wrapper": convert_numbers_to_words_wrapper,
                 "ordinal_words": ordinal_words}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NORMALIZER), "exec"), namespace)
    fixture_count = len(namespace["fixtures"]())
    return namespace["spoken"], fixture_count, receipt


def counts(ref, hyp):
    return {"word_errors": edit_distance(ref.split(), hyp.split()),
            "reference_words": len(ref.split()),
            "wer": edit_distance(ref.split(), hyp.split()) / max(1, len(ref.split())),
            "character_errors": edit_distance(ref, hyp), "reference_characters": len(ref),
            "cer": edit_distance(ref, hyp) / max(1, len(ref)),
            "nonspace_character_errors": edit_distance(ref.replace(" ", ""), hyp.replace(" ", "")),
            "reference_nonspace_characters": len(ref.replace(" ", ""))}


def boundary_diagnostic(ref, hyp):
    """Minimum residual token edits allowing exact-character splits/joins <=4.

    This is a text comparison diagnostic, not a normalization or WER replacement.
    No letter may change in a zero-cost split/join operation.
    """
    a, b = ref.split(), hyp.split()
    n, m = len(a), len(b)
    dp = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = ((0, 0), None)

    def offer(i, j, ni, nj, cost, op):
        previous = dp[i][j][0]
        key = (previous[0] + cost, previous[1] + (op == "exact_character_boundary"))
        if dp[ni][nj] is None or key < dp[ni][nj][0]:
            dp[ni][nj] = (key, (i, j, op))

    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i][j] is None:
                continue
            if i < n and j < m:
                offer(i, j, i + 1, j + 1, a[i] != b[j], "equal" if a[i] == b[j] else "substitute")
            if i < n:
                offer(i, j, i + 1, j, 1, "delete")
            if j < m:
                offer(i, j, i, j + 1, 1, "insert")
            for na in range(1, min(4, n - i) + 1):
                ar = a[i:i + na]
                for nb in range(1, min(4, m - j) + 1):
                    bh = b[j:j + nb]
                    if (na > 1 or nb > 1) and ar != bh and "".join(ar) == "".join(bh):
                        offer(i, j, i + na, j + nb, 0, "exact_character_boundary")
    diffs = []
    i, j = n, m
    while dp[i][j][1] is not None:
        pi, pj, op = dp[i][j][1]
        if op != "equal":
            diffs.append({"type": op, "source": " ".join(a[pi:i]), "checker": " ".join(b[pj:j])})
        i, j = pi, pj
    return {"residual_token_edits": dp[n][m][0][0],
            "exact_character_boundary_operations": dp[n][m][0][1],
            "differences": list(reversed(diffs))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    # This bounded text-only process uses <100MB. Observe the same commit guards
    # using the Win32 API, without importing project audio/ML modules.
    import ctypes
    from ctypes import wintypes

    class Performance(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("CommitTotal", ctypes.c_size_t),
                    ("CommitLimit", ctypes.c_size_t), ("CommitPeak", ctypes.c_size_t),
                    ("PhysicalTotal", ctypes.c_size_t), ("PhysicalAvailable", ctypes.c_size_t),
                    ("SystemCache", ctypes.c_size_t), ("KernelTotal", ctypes.c_size_t),
                    ("KernelPaged", ctypes.c_size_t), ("KernelNonpaged", ctypes.c_size_t),
                    ("PageSize", ctypes.c_size_t), ("HandleCount", wintypes.DWORD),
                    ("ProcessCount", wintypes.DWORD), ("ThreadCount", wintypes.DWORD)]

    def memory_guard():
        while True:
            p = Performance()
            p.cb = ctypes.sizeof(p)
            assert ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p), p.cb)
            ratio = p.CommitTotal / p.CommitLimit
            headroom = (p.CommitLimit - p.CommitTotal) * p.PageSize
            if ratio >= .95 or headroom < 2_000_000_000 or process.memory_info().rss > 2_000_000_000:
                raise RuntimeError("Own text job stopped at unchanged resource protection")
            if ratio < .90:
                return {"commit_ratio": ratio, "commit_headroom_bytes": headroom}
            time.sleep(5)

    memory = memory_guard()
    spoken, fixture_count, vendor_receipt = load_spoken()
    inputs = [STORE / "analysis/CPU_validation_v2/independent_QC_ledger_v2.jsonl",
              STORE / "analysis/CPU_validation_v3/new8_shards_independent_QC_ledger_v3.jsonl"]
    ledger, seen, raw_flags, nonidempotent = [], set(), collections.Counter(), []
    total, recognized, size_exclusions = 0, 0, 0
    for path in inputs:
        for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            recognized += bool(row.get("independent_recognizer_completed"))
            size_exclusions += not bool(row.get("independent_recognizer_completed"))
            if not row.get("triage_flags"):
                continue
            memory = memory_guard()
            identifier = row["segment_id"]
            assert identifier not in seen
            seen.add(identifier)
            raw_flags.update(row["triage_flags"])
            ref, hyp = row["transcript"], row["independent_hypothesis"]
            frozen_ref, frozen_hyp = normalize_text(ref, "tr"), normalize_text(hyp, "tr")
            official = counts(frozen_ref, frozen_hyp)
            assert abs(official["wer"] - row["source_vs_independent_WER"]) < 1e-12
            if "source_vs_independent_CER" in row:
                assert abs(official["cer"] - row["source_vs_independent_CER"]) < 1e-12
            sr, sh = spoken(ref), spoken(hyp)
            for side, value in (("source", sr), ("checker", sh)):
                if spoken(value) != value:
                    nonidempotent.append({"segment_id": identifier, "side": side,
                                          "one_pass": value, "second_pass": spoken(value)})
            diag = counts(sr, sh)
            boundary = boundary_diagnostic(sr, sh)
            number_format_changed = (sr, sh) != (frozen_ref, frozen_hyp)
            numeric_reduced_edits = diag["word_errors"] < official["word_errors"]
            boundary_reduced_edits = boundary["residual_token_edits"] < diag["word_errors"]
            if official["word_errors"] == 0:
                category = "no_text_disagreement_flagged_only_by_language"
            elif diag["word_errors"] == 0:
                category = "fully_explained_by_existing_spoken_form_diagnostic"
            elif boundary["residual_token_edits"] == 0:
                category = "fully_explained_by_spoken_form_plus_exact_character_boundaries"
            elif numeric_reduced_edits or boundary_reduced_edits:
                category = "mixed_number_format_or_boundary_effects_with_remaining_differences"
            else:
                category = "remaining_differences_without_detected_number_or_boundary_reduction"
            ledger.append({
                "segment_id": identifier, "recording_id": row["recording_id"], "shard": row["shard"],
                "cached_source_ledger": str(path), "cached_source_line": line_no,
                "source_transcript": ref, "small_checker_hypothesis": hyp,
                "raw_source_vs_independent_WER": row["source_vs_independent_WER"],
                "raw_source_vs_independent_CER_if_cached": row.get("source_vs_independent_CER"),
                "raw_triage_flags": row["triage_flags"], "raw_outcome": row["outcome"],
                "independent_language": row["independent_language"],
                "independent_language_probability": row["independent_language_probability"],
                "recomputed_frozen_score": official,
                "existing_spoken_form_diagnostic": {**diag, "source": sr, "checker": sh},
                "word_boundary_diagnostic_only": boundary,
                "spoken_form_changed_either_text": number_format_changed,
                "spoken_form_reduced_absolute_word_edits": numeric_reduced_edits,
                "exact_boundary_equivalences_reduce_residual_edits": boundary_reduced_edits,
                "text_diagnostic_category": category,
                "official_flags_unchanged": True, "audio_correctness_established": False,
                "accepted_TRAIN": False})
    assert (total, recognized, size_exclusions, len(ledger)) == (504, 496, 8, 106)
    assert raw_flags == {"source_independent_ASR_disagreement": 104,
                         "independent_LID_uncertain": 4, "independent_detected_non_Turkish": 1}
    destination = OUT / "cached_flagged_text_diagnostic_ledger_v5.jsonl"
    destination.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ledger), encoding="utf-8")

    def sums(key):
        c = collections.Counter()
        for row in ledger:
            c.update({k: v for k, v in row[key].items() if isinstance(v, int)})
        return {**c, "wer": c["word_errors"] / c["reference_words"],
                "cer": c["character_errors"] / c["reference_characters"]}

    raw_asr = [r for r in ledger if "source_independent_ASR_disagreement" in r["raw_triage_flags"]]
    diagnostic_asr = [r for r in ledger if r["existing_spoken_form_diagnostic"]["wer"] > .30]
    hypothetical_union = [r for r in ledger if r["existing_spoken_form_diagnostic"]["wer"] > .30 or
                          any(f != "source_independent_ASR_disagreement" for f in r["raw_triage_flags"])]
    crossed = [r for r in raw_asr if r["existing_spoken_form_diagnostic"]["wer"] <= .30]
    chosen = []
    for predicate in [lambda r: r["segment_id"].startswith("51e250"),
                      lambda r: r["text_diagnostic_category"].startswith("fully_explained"),
                      lambda r: r in crossed,
                      lambda r: r["spoken_form_reduced_absolute_word_edits"],
                      lambda r: r["exact_boundary_equivalences_reduce_residual_edits"],
                      lambda r: r["independent_language"] != "tr",
                      lambda r: r["recomputed_frozen_score"]["wer"] > .75]:
        for row in ledger:
            if predicate(row) and row not in chosen:
                chosen.append(row)
                break
    examples = [{k: row[k] for k in ("segment_id", "shard", "source_transcript", "small_checker_hypothesis",
                                    "raw_triage_flags", "recomputed_frozen_score", "existing_spoken_form_diagnostic",
                                    "word_boundary_diagnostic_only", "text_diagnostic_category",
                                    "cached_source_ledger", "cached_source_line")} for row in chosen]
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    report = {
        "kind": "cached_pilot_flagged_text_sensitivity_audit_v5", "completed_at_utc": now,
        "pid": os.getpid(), "scope": "106 previously flagged unique clips only; cached text pairs, no audio/model/TEST predictions",
        "source": {"audio_dataset": "espnet/yodas3", "audio_revision": "81d802f45a1c1e1f5331260484dacf1fbea81981",
                   "caption_annotation_dataset": "cloud0day3/alania-speech-captions-tr",
                   "annotation_revision": "8b0fdd976ed9668b64f5ee0696317a8f241ce39c",
                   "source_page": "https://huggingface.co/datasets/cloud0day3/alania-speech-captions-tr",
                   "audio_source_page": "https://huggingface.co/datasets/espnet/yodas3",
                   "independent_checker": "SYSTRAN/faster-whisper-small CPU int8; cached hypotheses only",
                   "checker_receipt": str(STORE / "analysis/CPU_validation_v2/independent_recognizer_receipt_v2.json")},
        "original_scope": {"declared_clips": total, "recognized": recognized, "size_exclusions_separate": size_exclusions,
                           "unique_triage_clips": len(ledger), "overlapping_reason_counts": dict(raw_flags)},
        "original_official_flags_preserved": True,
        "original_ASR_threshold": "Frozen-normalized word disagreement >0.30; a review cue, not proof source is wrong",
        "aggregate_over_106_original_flagged_pairs": {"frozen": sums("recomputed_frozen_score"),
                                                      "spoken_form_diagnostic_only": sums("existing_spoken_form_diagnostic")},
        "categories": dict(collections.Counter(r["text_diagnostic_category"] for r in ledger)),
        "number_format_changed_either_side_clips": sum(r["spoken_form_changed_either_text"] for r in ledger),
        "number_format_reduced_absolute_word_edits_clips": sum(r["spoken_form_reduced_absolute_word_edits"] for r in ledger),
        "exact_boundary_equivalences_reduce_residual_edits_clips": sum(r["exact_boundary_equivalences_reduce_residual_edits"] for r in ledger),
        "threshold_sensitivity_only": {"original_ASR_disagreement_flags": len(raw_asr),
                                       "existing_spoken_form_ASR_disagreement_over_point30": len(diagnostic_asr),
                                       "original_ASR_flags_at_or_below_point30_under_spoken_form": len(crossed),
                                       "hypothetical_unique_triage_with_original_LID_reasons_preserved": len(hypothetical_union),
                                       "actual_unique_triage_after_audit": len(ledger),
                                       "actual_flags_dropped_or_changed": 0,
                                       "new_TRAIN_hours_accepted": 0,
                                       "crossing_segment_ids": [r["segment_id"] for r in crossed]},
        "normalizer": {"path": str(NORMALIZER), "sha256": sha(NORMALIZER),
                       "method": "Existing frozen spoken(text) applied independently to both sides; exact original AST nodes, no original main executed",
                       "fixture_count_passed": fixture_count, "all212transforms_idempotent": not nonidempotent,
                       "one_pass_only_used_for_all_scores": True, "new_scope_nonidempotent_cases": nonidempotent,
                       "vendor_repository": vendor_receipt["repository"], "vendor_revision": vendor_receipt["revision"],
                       "vendor_all_file_bindings_verified": True},
        "word_boundary_method": "Diagnostic token-edit DP allows zero-cost exact-character groups of up to4 tokens on each side; letters/accents unchanged; no output rewriting or scorer replacement",
        "examples": examples,
        "limitations": ["No listening was performed; direction of any source/checker error is undetermined.",
                        "Punctuation and case were already excluded by frozen scoring, so they cannot alone explain the original flags.",
                        "The existing spoken-form normalizer handles conservative numbers/ordinals/percent/SI units; no acronym/spelling correction or arbitrary boundary merging.",
                        "Residual differences include lexical, suffix, name, orthographic or checker errors; these categories do not establish source-label defects.",
                        "Number expansion changes reference word counts; WER changes are reported with absolute errors and denominators.",
                        "The 106 flagged clips were selected by a checker; this audit does not estimate population defect rates.",
                        "All original ledgers/scores/flags, official scorer/gates, source labels,17 text-overlap/3 long-match quarantines and existing worker source remain unchanged."],
        "inputs": [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)} for p in inputs],
        "diagnostic_ledger": {"path": str(destination), "sha256": sha(destination)},
        "source_implementation": {"path": str(Path(__file__)), "sha256": sha(Path(__file__))},
        "runtime": {"CPU_only": True, "audio_read": False, "new_inference": False, "TEST_predictions_read": False,
                    "torch_imported": "torch" in sys.modules, "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
                    "own_RSS_bytes": process.memory_info().rss, "memory": memory}}
    report_path = OUT / "cached_flagged_text_diagnostic_report_v5.json"
    json_write(report_path, report)
    print(json.dumps({"report": str(report_path), "categories": report["categories"],
                      "threshold_sensitivity_only": report["threshold_sensitivity_only"],
                      "frozen": report["aggregate_over_106_original_flagged_pairs"]["frozen"],
                      "spoken": report["aggregate_over_106_original_flagged_pairs"]["spoken_form_diagnostic_only"],
                      "runtime": report["runtime"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
