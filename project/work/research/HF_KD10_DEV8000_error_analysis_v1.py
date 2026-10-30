"""Rescore only already-saved fixed DEV text. No ML, audio or checkpoints."""
from __future__ import annotations

import collections
import datetime
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0, str(OUT))
from analyze_turkish_dev_errors_v1 import FOLD, alignment, boundary_oracle, top
from metrics import score_transcriptions

RUN = "adaptive_20261011_hf_quality_window_01_stronger_kd"
STEPS = (5000, 6705, 8000)
DESTINATION = OUT / "HF_KD10_DEV8000_error_evidence_v1.json"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    # Discard the large confidence arrays immediately, retaining text/identity.
    keep = ("source_id", "sentence_id", "language", "split", "reference",
            "hypothesis", "dataset_revision", "audio_sha256", "wav_sha256",
            "duration", "alignment")
    result = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            result.append({key: value[key] for key in keep if key in value})
    return result


def compact_score(value):
    return {key: val for key, val in value.items() if key != "details"}


def correct_positions(operations):
    position = 0
    correct = set()
    for operation, ref, _ in operations:
        if operation == "equal":
            correct.add(position)
        if ref:
            position += 1
    return correct


def example(row, detail):
    return {"source_id": row["source_id"], "reference": detail["reference_normalized"],
            "hypothesis": detail["hypothesis_normalized"], "word_errors": detail["word_errors"],
            "character_errors": detail["character_errors"]}


def main():
    assert not DESTINATION.exists(), "Preserve prior analysis"
    fixed_path = ROOT / "work/data/validation_tr.jsonl"
    fixed = [json.loads(line) for line in fixed_path.read_text(encoding="utf-8").splitlines()]
    prior = OUT / "HF_KD10_DEV5000_error_evidence_v1.json"
    prior_sha = sha(prior)
    report = {"schema": 1, "kind": "saved_HF_KD10_DEV8000_text_error_comparison_v1",
              "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "run_id": RUN, "steps": list(STEPS), "CPU_only": True,
              "scope": "Saved fixed FLEURS DEV text only; no new inference, audio, checkpoint, TRAIN or TEST evaluation",
              "provenance": {"script_path": str(Path(__file__).resolve()), "script_sha256": sha(__file__),
                             "metrics_path": str(OUT / "metrics.py"), "metrics_sha256": sha(OUT / "metrics.py"),
                             "diagnostic_helper_sha256": sha(OUT / "analyze_turkish_dev_errors_v1.py"),
                             "fixed_DEV_manifest_path": str(fixed_path), "fixed_DEV_manifest_sha256": sha(fixed_path),
                             "preserved_DEV5000_report_path": str(prior), "preserved_DEV5000_report_sha256": prior_sha},
              "models": {}, "pairs": {}, "checks": 0}
    by_step = {}
    for step in STEPS:
        folder = OUT / "results" / f"{RUN}_step_{step}"
        prediction_path, summary_path = folder / "validation_tr.jsonl", folder / "summary.json"
        predictions_sha, summary_sha = sha(prediction_path), sha(summary_path)
        predictions = rows(prediction_path)
        assert len(predictions) == len(fixed) == 50
        for row, source in zip(predictions, fixed):
            assert row["source_id"] == source["source_id"] and row["reference"] == source["text"]
            for key in ("sentence_id", "dataset_revision", "audio_sha256", "wav_sha256", "duration"):
                assert row[key] == source[key], (step, key, row["source_id"])
                report["checks"] += 1
            assert row["split"] == "validation" and row["language"] == "tr"
            report["checks"] += 3
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        score = score_transcriptions(predictions, language="tr")
        for key, value in compact_score(score).items():
            assert value == summary["validation_tr"]["asr"][key], (step, key)
            report["checks"] += 1
        word_counts, saved_counts, char_counts, substitutions = (collections.Counter() for _ in range(4))
        diagnostics = collections.Counter()
        word_ops, correct, numeric, no_numeric, boundary_examples = [], [], [], [], []
        letter_stats = {letter: {"reference_occurrences": 0, "hypothesis_occurrences": 0,
                                 "correct_reference_character_positions": 0} for letter in "ğış"}
        for row, detail in zip(predictions, score["details"]):
            r, h = detail["reference_normalized"], detail["hypothesis_normalized"]
            operations = alignment(r.split(), h.split())
            word_ops.append(operations)
            correct.append(correct_positions(operations))
            assert sum(op != "equal" for op, _, _ in operations) == detail["word_errors"]
            report["checks"] += 1
            for operation, a, b in operations:
                word_counts[operation] += 1
                if operation == "sub":
                    substitutions[a, b] += 1
                if operation != "equal" and any(c.isdigit() for c in a):
                    diagnostics["traceback_errors_on_reference_digit_words"] += 1
            for key in ("hits", "substitutions", "deletions", "insertions"):
                saved_counts[key] += row["alignment"][key]
            for operation, a, b in alignment(r, h):
                char_counts[operation] += 1
                if operation != "equal" and " " in (a, b):
                    char_counts["space_involved_operations"] += 1
                if operation == "sub" and a.translate(FOLD) == b.translate(FOLD):
                    char_counts["diacritic_only_substitutions"] += 1
                if operation == "equal" and a in letter_stats:
                    letter_stats[a]["correct_reference_character_positions"] += 1
            for letter in letter_stats:
                letter_stats[letter]["reference_occurrences"] += r.count(letter)
                letter_stats[letter]["hypothesis_occurrences"] += h.count(letter)
            boundary = boundary_oracle(r.split(), h.split())
            diagnostics["exact_boundary_oracle_word_errors"] += boundary
            diagnostics["accent_folded_word_errors"] += score_transcriptions(
                [{"reference": r.translate(FOLD), "hypothesis": h.translate(FOLD)}], language="tr")["word_errors"]
            if detail["word_errors"] - boundary >= 2:
                boundary_examples.append(dict(example(row, detail), boundary_forgiven_edits=detail["word_errors"]-boundary))
            (numeric if any(c.isdigit() for c in r) else no_numeric).append(row)
        assert sum(saved_counts[key] for key in ("substitutions", "deletions", "insertions")) == score["word_errors"]
        assert word_counts["sub"] + word_counts["del"] + word_counts["ins"] == score["word_errors"]
        assert char_counts["sub"] + char_counts["del"] + char_counts["ins"] == score["character_errors"]
        assert predictions_sha == sha(prediction_path) and summary_sha == sha(summary_path)
        report["checks"] += 5
        diagnostics["boundary_forgiven_word_edits"] = score["word_errors"] - diagnostics["exact_boundary_oracle_word_errors"]
        diagnostics["accent_fold_forgiven_word_edits"] = score["word_errors"] - diagnostics["accent_folded_word_errors"]
        by_step[step] = {"rows": predictions, "score": score, "correct": correct}
        report["models"][str(step)] = {"predictions_path": str(prediction_path), "predictions_sha256": predictions_sha,
                                       "summary_path": str(summary_path), "summary_sha256": summary_sha,
                                       "standard_metrics": compact_score(score), "saved_word_alignment_counts": dict(saved_counts),
                                       "diagnostic_word_traceback_counts": dict(word_counts), "character_traceback_counts": dict(char_counts),
                                       "core_new_character_positions": letter_stats, "diagnostic_only": dict(diagnostics),
                                       "digit_reference_subset": compact_score(score_transcriptions(numeric, language="tr")),
                                       "no_digit_reference_subset": compact_score(score_transcriptions(no_numeric, language="tr")),
                                       "top_word_substitutions": top(substitutions, 12), "boundary_examples": boundary_examples[:4]}
    for first, last in ((5000, 6705), (6705, 8000), (5000, 8000)):
        a, b = by_step[first], by_step[last]
        changes, recovered, lost = [], collections.Counter(), collections.Counter()
        digit_delta = no_digit_delta = 0
        for old, new, old_detail, new_detail, old_correct, new_correct in zip(a["rows"], b["rows"], a["score"]["details"], b["score"]["details"], a["correct"], b["correct"]):
            assert old["source_id"] == new["source_id"]
            ref = old_detail["reference_normalized"].split()
            recovered.update(ref[i] for i in new_correct-old_correct)
            lost.update(ref[i] for i in old_correct-new_correct)
            delta = new_detail["word_errors"]-old_detail["word_errors"]
            if any(c.isdigit() for c in old_detail["reference_normalized"]):
                digit_delta += delta
            else:
                no_digit_delta += delta
            changes.append({"source_id": old["source_id"], "word_error_delta": delta,
                            "character_error_delta": new_detail["character_errors"]-old_detail["character_errors"],
                            "reference": old_detail["reference_normalized"], "before_hypothesis": old_detail["hypothesis_normalized"],
                            "after_hypothesis": new_detail["hypothesis_normalized"],
                            "new_correct_reference_positions": sorted(new_correct-old_correct),
                            "lost_correct_reference_positions": sorted(old_correct-new_correct),
                            "raw_transcript_changed": old["hypothesis"] != new["hypothesis"],
                            "normalized_transcript_changed": old_detail["hypothesis_normalized"] != new_detail["hypothesis_normalized"]})
        improved = sorted([item for item in changes if item["word_error_delta"] < 0], key=lambda item: item["word_error_delta"])
        worsened = sorted([item for item in changes if item["word_error_delta"] > 0], key=lambda item: -item["word_error_delta"])
        gains = sum(-item["word_error_delta"] for item in improved)
        losses = sum(item["word_error_delta"] for item in worsened)
        assert losses-gains == b["score"]["word_errors"]-a["score"]["word_errors"]
        report["checks"] += 51
        report["pairs"][f"{first}_to_{last}"] = {"word_error_delta": losses-gains,
            "character_error_delta": b["score"]["character_errors"]-a["score"]["character_errors"],
            "improved_clips": len(improved), "worsened_clips": len(worsened), "same_word_error_clips": 50-len(improved)-len(worsened),
            "raw_transcripts_changed": sum(item["raw_transcript_changed"] for item in changes),
            "normalized_transcripts_changed": sum(item["normalized_transcript_changed"] for item in changes),
            "gross_recovered_word_edits": gains, "gross_added_word_edits": losses,
            "largest_three_improved_clips_fraction_of_gross_gains": sum(-item["word_error_delta"] for item in improved[:3])/gains if gains else None,
            "digit_reference_clip_word_error_delta": digit_delta, "no_digit_reference_clip_word_error_delta": no_digit_delta,
            "newly_correct_reference_word_occurrences": sum(recovered.values()), "lost_correct_reference_word_occurrences": sum(lost.values()),
            "top_recovered_words": recovered.most_common(12), "top_lost_words": lost.most_common(12),
            "largest_improvements": improved[:5], "largest_regressions": worsened[:4]}
    baseline_path = OUT / "results/base_v3/summary.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    report["provenance"]["original_baseline_summary_sha256"] = sha(baseline_path)
    report["retention_8000"] = {}
    current = json.loads((OUT / "results" / f"{RUN}_step_8000/summary.json").read_text(encoding="utf-8"))
    for language in ("en", "es", "it", "fr", "de"):
        result_path = OUT / "results" / f"{RUN}_step_8000" / f"validation_{language}.jsonl"
        scored = score_transcriptions(rows(result_path), language=language)
        stored = current[f"validation_{language}"]["asr"]
        assert all(value == stored[key] for key, value in compact_score(scored).items())
        original = baseline[f"validation_{language}"]["asr"]
        limit = max(0.005, original["wer"]*0.10)
        max_errors = math.floor((original["wer"]+limit)*scored["reference_words"]+1e-10)
        report["retention_8000"][language] = {"asr": compact_score(scored), "predictions_sha256": sha(result_path),
            "baseline_word_errors": original["word_errors"], "gate_maximum_word_errors": max_errors,
            "remaining_word_error_margin": max_errors-scored["word_errors"], "passed": scored["wer"] <= original["wer"]+limit}
        report["checks"] += 10
    assert sha(prior) == prior_sha
    report["checks"] += 1
    report["limits"] = ["All comparisons reuse 50 selected DEV clips / 825 words. No new inference or TEST evidence.",
        "Saved S/D/I use the original alignment. Independent traceback categories/word recovery can differ on tied minimum alignments.",
        "Accent folding and exact concatenation are overlapping diagnostics, not replacement scores or a causal error partition.",
        "Digit-clip errors include all words on those clips; they do not attribute errors to digits alone.",
        "Sampling, TSC data and BPE recommendations are hypotheses for isolated future arms, not measured intervention effects."]
    with DESTINATION.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"path": str(DESTINATION), "sha256": sha(DESTINATION), "checks": report["checks"],
        "models": {k: {"metrics": v["standard_metrics"], "saved_edits": v["saved_word_alignment_counts"], "diagnostics": v["diagnostic_only"], "chars": v["core_new_character_positions"], "numeric":v["digit_reference_subset"]} for k, v in report["models"].items()},
        "pairs": {k: {key:val for key,val in value.items() if key not in ("largest_improvements", "largest_regressions")} for k,value in report["pairs"].items()},
        "retention": {k: {"errors":v["asr"]["word_errors"],"maximum":v["gate_maximum_word_errors"],"margin":v["remaining_word_error_margin"],"passed":v["passed"]} for k,v in report["retention_8000"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
