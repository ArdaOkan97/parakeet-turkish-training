"""Text-only, fixed-step review of saved HF Turkish and Spanish DEV outputs."""
import collections
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

from analyze_turkish_dev_errors_v1 import alignment, boundary_oracle, FOLD, top
from metrics import edit_distance, normalize_text, score_transcriptions

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
CURRENT = "adaptive_20261011_hf_quality_window_01_stronger_kd"
CASES = {
    "initial_qualifying": CURRENT + "_step_0",
    "old_KD5_unqualified_1000": "adaptive_20261011_hf_return_01_step_1000",
    "current_KD10_1000": CURRENT + "_step_1000",
    "current_KD10_qualifying_1250": CURRENT + "_step_1250",
}


def read(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def predictions(tag, language):
    path = OUT / "results" / tag / ("validation_" + language + ".jsonl")
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    return rows, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def identity(rows):
    return [(r["source_id"], r["reference"], r["audio_sha256"]) for r in rows]


def characterize(rows, language):
    score = score_transcriptions(rows, language=language)
    word_edits, char_edits, substitutions, new_tokens = (collections.Counter() for _ in range(4))
    letters = {ch: collections.Counter() for ch in "ğış"}
    folded_words = boundary = 0
    for row, detail in zip(rows, score["details"]):
        ref, hyp = detail["reference_normalized"], detail["hypothesis_normalized"]
        operations = alignment(ref.split(), hyp.split())
        assert sum(op != "equal" for op, _, _ in operations) == detail["word_errors"]
        for op, left, right in operations:
            if op != "equal":
                word_edits[op] += 1
            if op == "sub":
                substitutions[left, right] += 1
        for op, left, right in alignment(ref, hyp):
            if op != "equal":
                char_edits[op] += 1
            if op == "sub" and left.translate(FOLD) == right.translate(FOLD):
                char_edits["diacritic_only_substitutions"] += 1
            if op != "equal" and " " in (left, right):
                char_edits["space_operations"] += 1
            for letter in letters:
                if left == letter:
                    letters[letter]["reference_occurrences"] += 1
                    letters[letter]["correct" if op == "equal" else op] += 1
                if right == letter:
                    letters[letter]["hypothesis_occurrences"] += 1
        if language == "tr":
            folded_words += edit_distance(ref.translate(FOLD).split(), hyp.translate(FOLD).split())
            boundary += boundary_oracle(ref.split(), hyp.split())
        for token in row["confidence"]["tokens"]:
            if 8193 <= token["token_id"] < 8199:
                new_tokens[token["piece"]] += 1
    details = score.pop("details")
    result = {"metrics": score, "word_edits_one_minimum_traceback": dict(word_edits),
              "character_edits_one_minimum_traceback": dict(char_edits),
              "new_character_alignment": {ch: dict(value) for ch, value in letters.items()},
              "new6_emitted_piece_counts": dict(new_tokens),
              "top_word_substitutions": top(substitutions, 16),
              "empty_hypotheses": sum(not normalize_text(r["hypothesis"], language) for r in rows)}
    if language == "tr":
        result["diagnostic_only"] = {"diacritic_folded_word_errors": folded_words,
                                      "word_boundary_oracle_errors": boundary,
                                      "word_errors_remaining_after_exact_spacing_forgiveness": boundary}
        for name, numeric in (("numeric_reference", True), ("nonnumeric_reference", False)):
            subset = [r for r in rows if any(ch.isdigit() for ch in r["reference"]) == numeric]
            metrics = score_transcriptions(subset, language="tr")
            metrics.pop("details")
            result[name + "_subset"] = metrics
    return result, details


def ref_matches(ref, hyp):
    matches, position = {}, 0
    for op, left, right in alignment(ref.split(), hyp.split()):
        if left:
            matches[position] = (op == "equal", left, right, op)
            position += 1
    return matches


def paired(old_rows, new_rows, language):
    assert identity(old_rows) == identity(new_rows)
    rows, recovered, lost = [], collections.Counter(), collections.Counter()
    for old, new in zip(old_rows, new_rows):
        reference = normalize_text(old["reference"], language)
        before, after = (normalize_text(row["hypothesis"], language) for row in (old, new))
        old_errors = edit_distance(reference.split(), before.split())
        new_errors = edit_distance(reference.split(), after.split())
        old_match, new_match = ref_matches(reference, before), ref_matches(reference, after)
        recovered_words, lost_words = [], []
        for index in old_match:
            if new_match[index][0] and not old_match[index][0]:
                recovered[old_match[index][1]] += 1
                recovered_words.append(old_match[index][1])
            if old_match[index][0] and not new_match[index][0]:
                lost[old_match[index][1]] += 1
                lost_words.append(old_match[index][1])
        rows.append({"source_id": old["source_id"], "sentence_id": old["sentence_id"],
                     "reference": old["reference"], "before": old["hypothesis"], "after": new["hypothesis"],
                     "raw_transcript_changed": old["hypothesis"] != new["hypothesis"],
                     "normalized_transcript_changed": before != after,
                     "word_errors_before": old_errors, "word_errors_after": new_errors,
                     "word_error_delta": new_errors - old_errors,
                     "reference_words_recovered": recovered_words, "reference_words_lost": lost_words})
    changed = [row for row in rows if row["normalized_transcript_changed"]]
    improved = sorted((row for row in rows if row["word_error_delta"] < 0), key=lambda r: (r["word_error_delta"], r["sentence_id"]))
    worsened = sorted((row for row in rows if row["word_error_delta"] > 0), key=lambda r: (-r["word_error_delta"], r["sentence_id"]))
    return {"raw_transcripts_changed": sum(row["raw_transcript_changed"] for row in rows),
            "normalized_transcripts_changed": len(changed),
            "improved_clips": len(improved), "worsened_clips": len(worsened),
            "equal_word_error_count_clips": sum(row["word_error_delta"] == 0 for row in rows),
            "changed_transcripts_with_equal_error_count": sum(row["word_error_delta"] == 0 for row in changed),
            "net_word_error_delta": sum(row["word_error_delta"] for row in rows),
            "reference_word_occurrences_recovered": sum(recovered.values()),
            "reference_word_occurrences_lost": sum(lost.values()),
            "frequent_recovered_words": recovered.most_common(14), "frequent_lost_words": lost.most_common(14),
            "largest_improvements": improved[:4], "largest_regressions": worsened[:3],
            "per_sentence_changes": rows}


def main():
    report = {"schema": 1, "kind": "HF_KD10_DEV1250_fixed_text_error_review_v1",
              "fixed_current_step": 1250, "CPU_only": True, "new_inference": False, "TEST_evaluated": False,
              "scope": "Saved fixed 50 TR DEV and 50 ES DEV recordings, with all-language saved gate counts and TRAIN history prefix through1250 only",
              "models": {}, "Turkish_pairs": {}, "Spanish_pairs": {}}
    all_rows, all_summaries = {}, {}
    for name, tag in CASES.items():
        summary_path = OUT / "results" / tag / "summary.json"
        summary, summary_sha = read(summary_path)
        all_summaries[name] = summary
        all_rows[name] = {}
        for language in ("tr", "es"):
            rows, evidence = predictions(tag, language)
            all_rows[name][language] = rows
            metrics, _ = characterize(rows, language)
            assert len(rows) == 50
            assert all(metrics["metrics"][key] == summary["validation_" + language]["asr"][key]
                       for key in ("wer", "cer", "word_errors", "reference_words", "character_errors", "reference_characters"))
            metrics["predictions"] = evidence
            report["models"].setdefault(name, {})[language] = metrics
        report["models"][name]["summary"] = {"path": str(summary_path), "sha256": summary_sha}
    initial_rows, initial_pin = predictions("adaptive_20261011_hf_return_01_step_0", "tr")
    assert identity(initial_rows) == identity(all_rows["initial_qualifying"]["tr"])
    assert [r["hypothesis"] for r in initial_rows] == [r["hypothesis"] for r in all_rows["initial_qualifying"]["tr"]]
    report["initial_equals_prior_qualifying_best_predictions"] = {"verified": True, **initial_pin}
    winner = "current_KD10_qualifying_1250"
    for before in ("initial_qualifying", "old_KD5_unqualified_1000", "current_KD10_1000"):
        for language, target in (("tr", "Turkish_pairs"), ("es", "Spanish_pairs")):
            report[target][before + "_to_1250"] = paired(all_rows[before][language], all_rows[winner][language], language)
    baseline, baseline_sha = read(OUT / "results/base_v3/summary.json")
    gates = {}
    for language in ("en", "es", "it", "fr", "de"):
        base, actual = baseline["validation_" + language]["asr"], all_summaries[winner]["validation_" + language]["asr"]
        limit = max(.005, .1 * base["wer"])
        permitted = base["wer"] + limit
        max_errors = math.floor(permitted * actual["reference_words"] + 1e-10)
        gates[language] = {"baseline_word_errors": base["word_errors"], "current_word_errors": actual["word_errors"],
                           "reference_words": actual["reference_words"], "baseline_wer": base["wer"], "current_wer": actual["wer"],
                           "maximum_permitted_wer": permitted, "maximum_permitted_integer_word_errors": max_errors,
                           "additional_word_errors_remaining_before_gate_failure": max_errors - actual["word_errors"],
                           "passed": actual["wer"] <= permitted}
    assert all(value["passed"] for value in gates.values())
    report["retention_at_1250"] = {"all_five_passed": True, "baseline_summary_sha256": baseline_sha, "languages": gates}
    history, history_sha = read(OUT / "runs" / CURRENT / "training_history.json")
    assert len(history) >= 1250 and all(row["step"] == index for index, row in enumerate(history[:1250], 1))
    history = history[:1250]
    tr_ids = {sid for row in history for sid in row["sampled_source_ids"]
              if not sid.startswith("google/fleurs:") or ":tr_tr:train:" in sid}
    assert len(tr_ids) == history[-1]["turkish_unique_clips_seen"] == 10000
    config, config_sha = read(OUT / "runs" / CURRENT / "run_config.json")
    report["TRAIN_exposure_through_1250"] = {
        "run_config_sha256": config_sha, "history_file_snapshot_sha256": history_sha,
        "frozen_1250_history_prefix_sha256": hashlib.sha256(json.dumps(history, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "history_rows_analyzed": 1250, "unique_TR_clips": len(tr_ids), "TR_pool_clips": config["turkish_clips_used"],
        "TR_unique_fraction": len(tr_ids) / config["turkish_clips_used"],
        "TR_domain_unique_clips": {"FLEURS": sum(sid.startswith("google/fleurs:") for sid in tr_ids),
                                   "CommonVoice": sum(not sid.startswith("google/fleurs:") for sid in tr_ids)},
        "TR_audio_hours_seen": history[-1]["turkish_audio_seen_seconds"] / 3600,
        "replay_audio_hours_seen": history[-1]["replay_audio_seen_seconds"] / 3600,
        "recorded_turkish_epoch": history[-1]["turkish_epoch"],
        "token_KD_coefficient": config["replay_kl"], "replay_clip_fraction": config["replay_fraction"],
        "mixed_variable_example_loss_diagnostic": {key: {"median_first100": statistics.median(row[key] for row in history[:100]),
                                                         "median_last100": statistics.median(row[key] for row in history[-100:])}
                                                    for key in ("loss", "tdt_loss", "replay_kl")},
    }
    report["limits"] = ["50 reused TR DEV clips/825 words are a small tuning set; no generalization or TEST claim",
                         "Substitution/diacritic/spacing categories use one tied minimum-edit traceback and overlap",
                         "History analysis stops at1250 even if live training later writes more points",
                         "No full Turkish pass has occurred; broad vocabulary/phrase coverage remains incomplete",
                         "Different checkpoints can have the same aggregate WER while changing many transcripts",
                         "New character counts establish emission, not adequate Turkish lexical recognition"]
    assert "torch" not in sys.modules and "nemo" not in sys.modules
    path = OUT / "HF_KD10_DEV1250_text_error_evidence_v1.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    compact = {"path": str(path), "models": {name: {lang: {"metrics": value[lang]["metrics"],
                  "word_edits": value[lang]["word_edits_one_minimum_traceback"],
                  "new_chars": value[lang]["new_character_alignment"],
                  "new_tokens": value[lang]["new6_emitted_piece_counts"],
                  "diagnostics": value[lang].get("diagnostic_only")} for lang in ("tr", "es")}
                  for name, value in report["models"].items()},
               "TR_pairs": {name: {key: value for key, value in pair.items() if key not in ("per_sentence_changes", "largest_improvements", "largest_regressions")}
                            for name, pair in report["Turkish_pairs"].items()},
               "ES_gate": gates["es"], "exposure": report["TRAIN_exposure_through_1250"]}
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
