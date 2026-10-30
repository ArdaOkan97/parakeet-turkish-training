"""Paired English v2/v3 confidence audit from saved FLEURS predictions.

No model inference occurs here. Match recordings by source_id; compare raw scores
on words correct in both systems at the same reference_word_index. Fit each
model's temperature on their common development recordings, then freeze it for
their common test recordings. Bootstrap recordings as paired sampling units.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from confidence import align_word_confidences, confidence_report, fit_temperature, normalize_text
from metrics import paired_bootstrap_delta, score_transcriptions


ROOT = Path(__file__).resolve().parent


def read_rows(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["source_id"])
            if key in rows and {k: v for k, v in rows[key].items() if k != "alignment"} != {k: v for k, v in row.items() if k != "alignment"}:
                raise ValueError(f"Different cached predictions have the same source_id: {key}")
            # Recompute alignment to add reference indices to older cached rows.
            row["alignment"] = align_word_confidences(row["reference"], row["confidence"], "en")
            rows[key] = row
    return rows


def common_recordings(v2: dict, v3: dict) -> tuple[list[str], dict]:
    ids = sorted(v2.keys() & v3.keys())
    if not ids:
        raise ValueError("There are no common recordings to compare.")
    for source_id in ids:
        if normalize_text(v2[source_id]["reference"], "en") != normalize_text(v3[source_id]["reference"], "en"):
            raise ValueError(f"References differ for shared source_id {source_id}")
    return ids, {"common_recordings": len(ids), "v2_recordings": len(v2), "v3_recordings": len(v3),
                 "v2_only_recordings": len(v2.keys() - v3.keys()), "v3_only_recordings": len(v3.keys() - v2.keys())}


def _word_observation(row: dict, word: dict) -> dict:
    token_logprobs = [row["confidence"]["tokens"][index]["logprob"] for index in word["token_indices"]]
    return {"confidence": word["confidence"],
            "conditional_nonblank_confidence": word.get("conditional_nonblank_confidence", word["confidence"]),
            "word_mean_logprob": sum(token_logprobs) / len(token_logprobs),
            "token_logprob_sum": sum(token_logprobs), "token_count": len(token_logprobs),
            "pieces": [row["confidence"]["tokens"][index]["piece"] for index in word["token_indices"]]}


def shared_correct_words(ids: list[str], v2: dict, v3: dict) -> dict[str, list[dict]]:
    grouped = {source_id: [] for source_id in ids}
    for source_id in ids:
        words_v2 = {word["reference_word_index"]: word for word in v2[source_id]["alignment"]["words"] if word["correct"]}
        words_v3 = {word["reference_word_index"]: word for word in v3[source_id]["alignment"]["words"] if word["correct"]}
        for reference_index in sorted(words_v2.keys() & words_v3.keys()):
            word_v2, word_v3 = words_v2[reference_index], words_v3[reference_index]
            if word_v2["word"] != word_v3["word"]:
                raise ValueError("Words at a shared correct reference index differ.")
            grouped[source_id].append({"source_id": source_id, "reference_word_index": reference_index,
                                       "word": word_v2["word"], "reference": v2[source_id]["reference"],
                                       "v2": _word_observation(v2[source_id], word_v2),
                                       "v3": _word_observation(v3[source_id], word_v3)})
    return grouped


def shared_word_statistics(observations: list[dict]) -> dict:
    if not observations:
        return {"shared_correct_word_count": 0}
    result = {"shared_correct_word_count": len(observations)}
    for name in ("v2", "v3"):
        scores = [word[name] for word in observations]
        mean_logprob = sum(score["word_mean_logprob"] for score in scores) / len(scores)
        result[name] = {
            "mean_word_confidence": sum(score["confidence"] for score in scores) / len(scores),
            "geometric_word_confidence": math.exp(mean_logprob),
            "mean_word_logprob": mean_logprob,
            "mean_emitted_token_logprob": sum(score["token_logprob_sum"] for score in scores) / sum(score["token_count"] for score in scores),
            "emitted_token_count": sum(score["token_count"] for score in scores),
            "mean_tokens_per_word": sum(score["token_count"] for score in scores) / len(scores),
            "mean_conditional_nonblank_word_confidence": sum(score["conditional_nonblank_confidence"] for score in scores) / len(scores),
            "geometric_conditional_nonblank_word_confidence": math.exp(sum(math.log(max(score["conditional_nonblank_confidence"], 1e-300)) for score in scores) / len(scores)),
            "correct_words_below_0_5_fraction": sum(score["confidence"] < 0.5 for score in scores) / len(scores),
            "correct_words_below_0_8_fraction": sum(score["confidence"] < 0.8 for score in scores) / len(scores),
        }
    result["delta_v3_minus_v2"] = {
        metric: result["v3"][metric] - result["v2"][metric]
        for metric in result["v2"] if metric != "emitted_token_count"
    }
    return result


def conditional_nonblank_diagnostics(rows: list[dict]) -> dict:
    words = [word for row in rows for word in row["alignment"]["words"]]
    result = {}
    for population, selected in [("all_emitted_words", words),
                                 ("correct_words", [word for word in words if word["correct"]]),
                                 ("incorrect_words", [word for word in words if not word["correct"]])]:
        result[population] = {
            "word_count": len(selected),
            "mean_raw_confidence": sum(word["confidence"] for word in selected) / len(selected) if selected else None,
            "mean_conditional_nonblank_confidence": sum(word.get("conditional_nonblank_confidence", word["confidence"]) for word in selected) / len(selected) if selected else None,
            "mean_tokens_per_word": sum(len(word["token_indices"]) for word in selected) / len(selected) if selected else None,
        }
    result["interpretation"] = "Conditioning on nonblank emission diagnoses lexical versus emission-timing uncertainty; it discards blank competition and is not assumed calibrated for word correctness."
    return result


def _quantile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = probability * (len(values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def paired_confidence_bootstrap(grouped: dict, resamples: int, seed: int) -> dict:
    ids = list(grouped)
    if resamples < 1:
        raise ValueError("resamples must be positive")
    point = shared_word_statistics([word for words in grouped.values() for word in words])
    if not point.get("shared_correct_word_count"):
        return {"resampling_unit": "paired recording", "resamples": resamples, "available": False}
    rng = random.Random(seed)
    samples = {metric: [] for metric in point["delta_v3_minus_v2"]}
    for _ in range(resamples):
        selected = [word for _ in ids for word in grouped[rng.choice(ids)]]
        statistics = shared_word_statistics(selected)
        if not statistics.get("shared_correct_word_count"):
            continue
        for metric, value in statistics["delta_v3_minus_v2"].items():
            samples[metric].append(value)
    return {
        "resampling_unit": "paired recording (all its shared correct words together)",
        "recording_count": len(ids), "resamples": resamples, "seed": seed, "confidence": 0.95,
        "delta_v3_minus_v2_ci": {metric: [_quantile(values, 0.025), _quantile(values, 0.975)]
                                  for metric, values in samples.items() if values},
    }


def compare(v2_dir: Path, v3_dir: Path, output_path: Path, resamples: int = 2000, seed: int = 20261010) -> dict:
    rows = {"v2": {}, "v3": {}}
    overlap = {}
    shared_ids = {}
    for split in ("validation", "test"):
        rows["v2"][split] = read_rows(v2_dir / f"{split}_en.jsonl")
        rows["v3"][split] = read_rows(v3_dir / f"{split}_en.jsonl")
        shared_ids[split], overlap[split] = common_recordings(rows["v2"][split], rows["v3"][split])
    models = {}
    for name in ("v2", "v3"):
        dev = [rows[name]["validation"][source_id] for source_id in shared_ids["validation"]]
        test = [rows[name]["test"][source_id] for source_id in shared_ids["test"]]
        calibration = fit_temperature([row["alignment"] for row in dev])
        asr = score_transcriptions(test, language="en")
        asr.pop("details")
        models[name] = {
            "english_asr_on_common_test_recordings": asr,
            "raw_test_word_confidence": confidence_report([row["alignment"] for row in test]),
            "conditional_nonblank_diagnostics": conditional_nonblank_diagnostics(test),
            "development_calibration": calibration,
            "calibrated_test_word_confidence": confidence_report([row["alignment"] for row in test], calibration["temperature"]),
        }
    test_ids = shared_ids["test"]
    grouped = shared_correct_words(test_ids, rows["v2"]["test"], rows["v3"]["test"])
    observations = [word for words in grouped.values() for word in words]
    lowest = sorted(observations, key=lambda word: (word["v3"]["confidence"], word["v3"]["confidence"] - word["v2"]["confidence"]))
    refs = [rows["v2"]["test"][source_id]["reference"] for source_id in test_ids]
    v2_hyps = [rows["v2"]["test"][source_id]["hypothesis"] for source_id in test_ids]
    v3_hyps = [rows["v3"]["test"][source_id]["hypothesis"] for source_id in test_ids]
    report = {
        "comparison": "Parakeet v3 minus v2 on fixed common English FLEURS recordings",
        "inputs": {"v2_dir": str(v2_dir.resolve()), "v3_dir": str(v3_dir.resolve())},
        "overlap": overlap,
        "models": models,
        "paired_english_asr": paired_bootstrap_delta(refs, v2_hyps, v3_hyps, language="en", seed=seed, resamples=resamples),
        "shared_correct_words": shared_word_statistics(observations),
        "shared_correct_word_uncertainty": paired_confidence_bootstrap(grouped, resamples, seed),
        "v3_correct_words_below_0_5_examples": [word for word in lowest if word["v3"]["confidence"] < 0.5][:25],
        "lowest_v3_shared_correct_word_examples": lowest[:25],
        "interpretation": {
            "score": "Raw score is the geometric mean of nonblank lexical token probabilities normalized over the full token vocabulary, including blank and excluding duration classes.",
            "paired_word_population": "Only words correct in both systems, keyed by source_id and zero-based reference_word_index. Other correct and incorrect emitted words still enter each model's calibration report.",
            "temperature": "Each model uses its own vocabulary-softmax temperature chosen by word Brier on the same common development recordings; test data never fits temperature.",
            "raw_token_comparison_limit": "Vocabularies and token segmentation differ, so token logprob shifts are descriptive rather than equivalent likelihoods. Word reliability and WER determine usefulness.",
            "uncertainty_limit": "Small FLEURS pilot; percentile bootstrap resamples recordings. Repeated sentences and out-of-domain speech may have additional uncertainty.",
            "calibration_limit": "Temperature changes reported confidence, not decoded words or WER. Higher scores alone do not establish better calibration.",
            "conditional_nonblank_limit": "Conditional nonblank scores remove timing/blank competition and are diagnostic only. The reliability metrics retain the full-token-vocabulary score.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output_path.resolve()), "overlap": overlap,
                      "shared_correct_words": report["shared_correct_words"],
                      "english_wer": {name: models[name]["english_asr_on_common_test_recordings"]["wer"] for name in models}}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-dir", type=Path, default=ROOT / "results" / "base_v2")
    parser.add_argument("--v3-dir", type=Path, default=ROOT / "results" / "base_v3")
    parser.add_argument("--output", type=Path, default=ROOT / "english_confidence_comparison.json")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    compare(args.v2_dir, args.v3_dir, args.output, args.resamples, args.seed)
