"""Documented, paired ASR scoring for the Turkish Parakeet experiment.

This normalizer is deliberately shared by baseline and adapted predictions. It
is not a claim of exact equivalence to FLEURS' original FST normalizer. Unicode
letters/accents are retained; punctuation and capitalization are excluded.
"""
from __future__ import annotations

import math
import random
import unicodedata
from typing import Any, Iterable, Sequence


def normalize_text(text: str, language: str = "tr") -> str:
    text = unicodedata.normalize("NFC", str(text or ""))
    if language.lower().split("_")[0].split("-")[0] == "tr":
        text = text.replace("I", "ı").replace("İ", "i")
    text = text.lower().replace("i\u0307", "i")
    # Punctuation is deleted (apostrophes stay inside the surrounding word).
    # Unicode separator/control characters become whitespace.
    text = "".join(
        " " if unicodedata.category(char)[0] in {"Z", "C"}
        else "" if unicodedata.category(char)[0] == "P"
        else char
        for char in text
    )
    return " ".join(text.split())


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Levenshtein edit count, using linear memory."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, 1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ref_token != hyp_token),
            ))
        previous = current
    return previous[-1]


def _rate(errors: int, total: int) -> float:
    return errors / total if total else (0.0 if errors == 0 else math.inf)


def score_transcriptions(
    rows: Iterable[dict[str, Any]],
    ref_key: str = "reference",
    hyp_key: str = "hypothesis",
    language: str | None = None,
) -> dict[str, Any]:
    """Return corpus WER/CER as fractions, plus counts and per-row errors.

    CER includes normalized spaces. Scores are computed from total edit counts,
    not an unweighted mean of utterance error rates. Per-row entries support
    paired bootstrap comparisons without redoing dynamic programming.
    """
    word_errors = character_errors = reference_words = reference_characters = 0
    details = []
    for index, row in enumerate(rows):
        lang = language or str(row.get("language", "tr"))
        reference = normalize_text(row.get(ref_key, ""), lang)
        hypothesis = normalize_text(row.get(hyp_key, ""), lang)
        words = reference.split()
        word_count, character_count = len(words), len(reference)
        we = edit_distance(words, hypothesis.split())
        ce = edit_distance(reference, hypothesis)
        word_errors += we
        character_errors += ce
        reference_words += word_count
        reference_characters += character_count
        details.append({
            "source_id": row.get("source_id", index),
            "sentence_id": row.get("sentence_id"),
            "language": lang,
            "reference_normalized": reference,
            "hypothesis_normalized": hypothesis,
            "word_errors": we,
            "reference_words": word_count,
            "character_errors": ce,
            "reference_characters": character_count,
        })
    return {
        "wer": _rate(word_errors, reference_words),
        "cer": _rate(character_errors, reference_characters),
        "word_errors": word_errors,
        "reference_words": reference_words,
        "character_errors": character_errors,
        "reference_characters": reference_characters,
        "utterances": len(details),
        "normalization": "NFC; Turkish-aware lowercase; delete Unicode punctuation; collapse whitespace; preserve accents; CER includes spaces",
        "details": details,
    }


def paired_bootstrap_delta(
    refs: Sequence[str],
    before: Sequence[str],
    after: Sequence[str],
    language: str = "tr",
    seed: int = 20261010,
    resamples: int = 2000,
    confidence: float = 0.95,
    groups: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Paired CI for adapted minus baseline WER/CER (percentage points).

    Supply FLEURS sentence IDs as groups to resample sentence clusters, keeping
    repeated recordings of the same sentence together. The same resampled units
    always enter both systems' scores.
    """
    if not (len(refs) == len(before) == len(after)):
        raise ValueError("Paired predictions and references must have identical lengths")
    if not refs:
        raise ValueError("Cannot bootstrap an empty evaluation set")
    if resamples < 1 or not 0 < confidence < 1:
        raise ValueError("resamples must be positive and confidence must be in (0, 1)")
    if groups is not None and len(groups) != len(refs):
        raise ValueError("groups must have one entry per utterance")
    baseline = score_transcriptions([
        {"reference": ref, "hypothesis": hyp} for ref, hyp in zip(refs, before)
    ], language=language)
    adapted = score_transcriptions([
        {"reference": ref, "hypothesis": hyp} for ref, hyp in zip(refs, after)
    ], language=language)
    cluster_indices: dict[Any, list[int]] = {}
    for i in range(len(refs)):
        group = groups[i] if groups is not None else i
        # None never inadvertently groups all utterances together.
        group = ("missing", i) if group is None else group
        cluster_indices.setdefault(group, []).append(i)
    clusters = list(cluster_indices.values())
    rng = random.Random(seed)
    samples = {"wer": [], "cer": []}
    for _ in range(resamples):
        indices = [i for _ in clusters for i in rng.choice(clusters)]
        for metric, numerator, denominator in [
            ("wer", "word_errors", "reference_words"),
            ("cer", "character_errors", "reference_characters"),
        ]:
            total = sum(baseline["details"][i][denominator] for i in indices)
            difference = sum(
                adapted["details"][i][numerator] - baseline["details"][i][numerator]
                for i in indices
            )
            samples[metric].append(100 * _rate(difference, total))

    def quantile(values: list[float], probability: float) -> float:
        values.sort()
        position = probability * (len(values) - 1)
        lower, upper = math.floor(position), math.ceil(position)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    tail = (1 - confidence) / 2
    return {
        "resamples": resamples,
        "seed": seed,
        "confidence": confidence,
        "resampling_unit": "sentence cluster" if groups is not None else "utterance",
        "units": len(clusters),
        **{
            metric: {
                "baseline": baseline[metric],
                "adapted": adapted[metric],
                "delta_percentage_points": 100 * (adapted[metric] - baseline[metric]),
                "ci_percentage_points": [
                    quantile(samples[metric], tail),
                    quantile(samples[metric], 1 - tail),
                ],
            }
            for metric in ("wer", "cer")
        },
    }
