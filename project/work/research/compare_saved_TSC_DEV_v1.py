"""Rescore saved DEV text and describe a paired change; no ML/audio/checkpoints."""
from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0, str(OUT))
from metrics import paired_bootstrap_delta, score_transcriptions
from analyze_turkish_dev_errors_v1 import alignment, boundary_oracle, FOLD


def record(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def rows(path):
    keys = ("source_id", "sentence_id", "language", "split", "reference", "hypothesis",
            "dataset_revision", "audio_sha256", "wav_sha256", "duration")
    result = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                result.append({key: value[key] for key in keys if key in value})
    return result


def main(args):
    assert re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id)
    assert 0 < args.before < args.after
    destination = args.output.resolve()
    assert destination.is_relative_to(OUT) and not destination.exists()
    folders = {step: OUT / "results" / f"{args.run_id}_step_{step}"
               for step in (args.before, args.after)}
    summaries = {step: json.loads((folder / "summary.json").read_text(encoding="utf-8"))
                 for step, folder in folders.items()}
    report = {"schema": 1, "kind": "paired_saved_TSC_DEV_text_comparison_v1",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_id": args.run_id,
              "steps": [args.before, args.after], "CPU_only": True,
              "scope": "Existing saved fixed DEV predictions only; no new inference/TRAIN/TEST/weights/audio.",
              "sources": {f"summary_{step}": record(folder / "summary.json")
                          for step, folder in folders.items()}, "comparisons": {}}
    for language in ("tr", "en", "es", "it", "fr", "de"):
        predictions, scores = {}, {}
        for step, folder in folders.items():
            path = folder / f"validation_{language}.jsonl"
            predictions[step] = rows(path)
            assert len(predictions[step]) == 50
            report["sources"][f"{language}_{step}"] = record(path)
            scores[step] = score_transcriptions(predictions[step], language=language)
            assert {key: value for key, value in scores[step].items() if key != "details"} == summaries[step][f"validation_{language}"]["asr"]
        before, after = predictions[args.before], predictions[args.after]
        identity = lambda row: {key: value for key, value in row.items() if key != "hypothesis"}
        assert [identity(row) for row in before] == [identity(row) for row in after]
        changes = []
        for old, new, old_detail, new_detail in zip(before, after, scores[args.before]["details"], scores[args.after]["details"]):
            changes.append({"source_id": old["source_id"], "reference": old_detail["reference_normalized"],
                "before": old_detail["hypothesis_normalized"], "after": new_detail["hypothesis_normalized"],
                "before_word_errors": old_detail["word_errors"], "after_word_errors": new_detail["word_errors"],
                "delta_word_errors": new_detail["word_errors"] - old_detail["word_errors"]})
        compact = {str(step): {key: value for key, value in score.items() if key != "details"}
                   for step, score in scores.items()}
        item = {"ordered_identity_exact": True, "rescored_counts_match_saved_summaries": True,
                "metrics": compact, "utterances_improved": sum(row["delta_word_errors"] < 0 for row in changes),
                "utterances_worsened": sum(row["delta_word_errors"] > 0 for row in changes),
                "gross_word_errors_removed": sum(-min(row["delta_word_errors"], 0) for row in changes),
                "gross_word_errors_added": sum(max(row["delta_word_errors"], 0) for row in changes),
                "largest_gain": min(row["delta_word_errors"] for row in changes),
                "largest_regression": max(row["delta_word_errors"] for row in changes),
                "improvement_examples": sorted([row for row in changes if row["delta_word_errors"] < 0], key=lambda row: row["delta_word_errors"])[:5],
                "regression_examples": sorted([row for row in changes if row["delta_word_errors"] > 0], key=lambda row: -row["delta_word_errors"])[:5]}
        if language == "tr":
            item["paired_bootstrap"] = paired_bootstrap_delta(
                [row["reference"] for row in before], [row["hypothesis"] for row in before],
                [row["hypothesis"] for row in after], language=language,
                groups=[row.get("sentence_id") for row in before], resamples=2000)
            item["diagnostic_only"] = {}
            for step, score in scores.items():
                kinds, diacritics, oracle = collections.Counter(), 0, 0
                for detail in score["details"]:
                    ref, hyp = detail["reference_normalized"], detail["hypothesis_normalized"]
                    operations = alignment(ref.split(), hyp.split())
                    assert sum(op != "equal" for op, _, _ in operations) == detail["word_errors"]
                    kinds.update(op for op, _, _ in operations if op != "equal")
                    diacritics += sum(op == "sub" and a.translate(FOLD) == b.translate(FOLD)
                                      for op, a, b in alignment(ref, hyp))
                    oracle += boundary_oracle(ref.split(), hyp.split())
                item["diagnostic_only"][str(step)] = {"word_edit_counts": dict(kinds),
                    "diacritic_only_character_substitutions": diacritics,
                    "exact_concat_oracle_word_errors": oracle,
                    "spacing_sensitive_word_edits": score["word_errors"] - oracle}
        report["comparisons"][language] = item
    report["sources"]["scorer"] = record(OUT / "metrics.py")
    report["sources"]["alignment"] = record(OUT / "analyze_turkish_dev_errors_v1.py")
    report["limitations"] = ["Repeatedly selected small DEV set: bootstrap is descriptive and is not independent evidence of generalization.",
        "Edit traceback ties and diagnostic spacing/diacritic categories are not replacement benchmark metrics.",
        "No causal claim about tokenizer, confidence or training changes follows from a paired text comparison."]
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": record(destination), "TR": {key: value for key, value in report["comparisons"]["tr"].items()
                     if key not in ("improvement_examples", "regression_examples")}}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--before", type=int, required=True)
    parser.add_argument("--after", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
