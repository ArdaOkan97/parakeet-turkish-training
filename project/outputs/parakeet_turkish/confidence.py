"""Word confidence for greedy Hugging Face Parakeet TDT inference.

Token and duration logits are separate distributions. We normalize the complete
token vocabulary (including blank), drop blank emissions, and use the geometric
mean of lexical token probabilities for each word. This is an empirical score,
not a probability of word correctness until its reliability is measured.

The pinned Transformers Parakeet generate wrapper drops output_scores. A joint
network forward hook captures its *actual decoding path* without another forward
pass. The collector deliberately supports batch size one and plain greedy
decoding, so finished-sequence padding cannot contaminate the observations.
Temperature changes confidence only; it never changes the decoded transcript.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable


DEFAULT_TEMPERATURES = (0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0)


def _temperature_key(value: float) -> str:
    return format(float(value), ".8g")


def normalize_text(text: str, language: str | None = None) -> str:
    """Unicode lexical normalization, with locale-correct Turkish case folding."""
    text = unicodedata.normalize("NFC", text)
    if language and language.split("_")[0].split("-")[0] == "tr":
        text = text.replace("I", "ı").replace("İ", "i")
    text = text.lower().replace("i\u0307", "i")
    text = "".join(" " if unicodedata.category(c)[0] in {"Z", "C"}
                   else "" if unicodedata.category(c)[0] == "P" else c for c in text)
    return re.sub(r"\s+", " ", text).strip()


def _lexical(character: str) -> bool:
    # The shared WER normalizer retains symbols, including currency units.
    # A standalone "$" is consequently a scored reference/hypothesis unit.
    return unicodedata.category(character)[0] in {"L", "M", "N", "S"}


def _word_separator(character: str) -> bool:
    # Match normalize_text exactly, including invisible controls such as ZWSP.
    return character.isspace() or unicodedata.category(character)[0] in {"Z", "C"}


def _decode_tokens(tokenizer, tokens: list[dict], language: str | None) -> tuple[str, list[dict]]:
    from tokenizers.decoders import DecodeStream

    stream = DecodeStream(skip_special_tokens=True)
    text_chunks: list[str] = []
    words: list[dict] = []
    current_chars: list[str] = []
    current_indices: set[int] = set()
    pending_indices: list[int] = []

    def flush() -> None:
        surface = "".join(current_chars)
        normalized = normalize_text(surface, language)
        if normalized and current_indices:
            indices = sorted(current_indices)
            selected = [tokens[i] for i in indices]
            by_temperature = {
                key: math.exp(sum(token["logprob_by_temperature"][key] for token in selected) / len(selected))
                for key in selected[0]["logprob_by_temperature"]
            }
            words.append({
                "word": normalized,
                "surface": surface,
                "token_indices": indices,
                "confidence": by_temperature["1"],
                "confidence_by_temperature": by_temperature,
                "conditional_nonblank_confidence": math.exp(
                    sum(token["conditional_nonblank_logprob"] for token in selected) / len(selected)
                ),
                "mean_normalized_entropy": sum(token["normalized_entropy"] for token in selected) / len(selected),
            })
        current_chars.clear()
        current_indices.clear()

    for index, token in enumerate(tokens):
        pending_indices.append(index)
        chunk = stream.step(tokenizer._tokenizer, token["token_id"])
        if chunk is None:
            continue
        text_chunks.append(chunk)
        # Byte fallback tokens may take several emissions before producing text.
        # Every emission in that chunk contributes to its lexical word.
        emitted_indices = pending_indices[:]
        pending_indices.clear()
        for character in chunk:
            if _word_separator(character):
                flush()
            else:
                current_chars.append(character)
                if _lexical(character):
                    current_indices.update(emitted_indices)
    flush()
    text = "".join(text_chunks)
    normalized_words = normalize_text(text, language).split()
    if normalized_words != [word["word"] for word in words]:
        raise ValueError("Decoded text and scored lexical words differ; do not silently misalign confidence. "
                         f"Decoded={text!r}; normalized_words={normalized_words!r}; "
                         f"scored_words={[word['word'] for word in words]!r}")
    return text, words


def generate_with_confidence(
    model,
    tokenizer,
    model_inputs: dict,
    *,
    language: str | None = None,
    temperatures: Iterable[float] = DEFAULT_TEMPERATURES,
    **generate_kwargs,
) -> tuple[object, dict]:
    """Return (normal Parakeet generation output, JSON-serializable confidence).

    Inputs must contain exactly one example. Do not use beam search, sampling,
    custom logits processors, compilation, or simultaneous calls on this model.
    The hook is always removed, including when generation fails.
    """
    import torch

    if model.training:
        raise ValueError("Call model.eval() before measuring confidence.")
    if model_inputs["input_features"].shape[0] != 1:
        raise ValueError("Confidence collection requires batch size 1.")
    if generate_kwargs.get("do_sample", False) or generate_kwargs.get("num_beams", 1) != 1:
        raise ValueError("Confidence collection requires plain greedy decoding.")
    if generate_kwargs.get("logits_processor"):
        raise ValueError("Custom logits processors are unsupported by the confidence collector.")
    values = sorted(set(float(value) for value in temperatures) | {1.0})
    if any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("Temperatures must be finite positive values.")
    keys = [_temperature_key(value) for value in values]
    vocab_size = int(model.config.vocab_size)
    blank_id = int(model.config.blank_token_id)
    captured: list[dict] = []

    def capture_joint(_module, _args, output) -> None:
        if output.shape[0] != 1 or output.numel() != output.shape[-1]:
            raise ValueError("Expected one Parakeet joint distribution per greedy step.")
        logits = output.reshape(-1).detach().float()
        token_logits = logits[:vocab_size]
        token_id = int(token_logits.argmax())
        logprob = token_logits.log_softmax(-1)
        selected_logprob = float(logprob[token_id])
        blank_probability = float(logprob[blank_id].exp())
        step = {
            "token_id": token_id,
            "logprob": selected_logprob,
            "blank_probability": blank_probability,
        }
        if token_id != blank_id:
            temp_tensor = torch.tensor(values, device=logits.device, dtype=torch.float32)
            scaled = token_logits.unsqueeze(0) / temp_tensor.unsqueeze(1)
            scaled_selected = scaled[:, token_id] - torch.logsumexp(scaled, -1)
            step["logprob_by_temperature"] = dict(zip(keys, scaled_selected.cpu().tolist()))
            step["conditional_nonblank_logprob"] = selected_logprob - math.log1p(-min(blank_probability, 1 - 1e-12))
            step["normalized_entropy"] = float(-(logprob.exp() * logprob).sum() / math.log(vocab_size))
            if logits.numel() > vocab_size:
                duration_logits = logits[vocab_size:]
                duration_id = int(duration_logits.argmax())
                step["duration_id"] = duration_id
                step["duration_logprob"] = float(duration_logits.log_softmax(-1)[duration_id])
        captured.append(step)

    hook = model.joint.register_forward_hook(capture_joint)
    try:
        with torch.inference_mode():
            generated = model.generate(**model_inputs, **generate_kwargs)
    finally:
        hook.remove()
    sequence = generated.sequences[0].detach().cpu().tolist()
    durations = generated.durations[0].detach().cpu().tolist()
    if len(sequence) != len(captured) + 1 or len(durations) != len(sequence):
        raise ValueError("Generation output and captured decoding steps have different lengths.")
    frame = 0
    tokens = []
    special_ids = set(tokenizer.all_special_ids) | {blank_id}
    for index, step in enumerate(captured):
        if sequence[index + 1] != step["token_id"]:
            raise ValueError("A logits processor changed the raw greedy token; confidence would be invalid.")
        duration = int(durations[index + 1])
        if step["token_id"] not in special_ids:
            step.update({
                "piece": tokenizer.convert_ids_to_tokens(step["token_id"]),
                "start_frame": frame,
                "duration_frames": duration,
            })
            tokens.append(step)
        frame += duration
    text, words = _decode_tokens(tokenizer, tokens, language)
    return generated, {
        "text": text,
        "language": language,
        "tokens": tokens,
        "words": words,
        "blank_steps": sum(step["token_id"] == blank_id for step in captured),
        "total_steps": len(captured),
        "temperatures": values,
        "aggregation": "geometric mean of nonblank lexical token probabilities",
        "token_normalizer": "full token vocabulary, including blank, excluding duration classes",
    }


def align_word_confidences(reference: str, confidence: dict, language: str | None = None) -> dict:
    """Label emitted words by minimum-edit alignment; report deletions separately."""
    import jiwer

    language = language or confidence.get("language")
    ref = normalize_text(reference, language)
    words = [dict(word) for word in confidence["words"]]
    hyp = " ".join(word["word"] for word in words)
    result = jiwer.process_words(ref, hyp)
    for word in words:
        word["correct"] = False
        word["operation"] = "unassigned"
        word["reference_word_index"] = None
    for chunk in result.alignments[0]:
        for index in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
            words[index]["correct"] = chunk.type == "equal"
            words[index]["operation"] = chunk.type
            if chunk.type == "equal":
                words[index]["reference_word_index"] = chunk.ref_start_idx + index - chunk.hyp_start_idx
    if any(word["operation"] == "unassigned" for word in words):
        raise ValueError("jiwer alignment failed to label a hypothesis word.")
    return {
        "words": words,
        "language": language,
        "hits": result.hits,
        "substitutions": result.substitutions,
        "insertions": result.insertions,
        "deletions": result.deletions,
        "reference_word_count": len(ref.split()),
        "wer": result.wer,
    }


def _auc(probabilities: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    pairs = sorted(zip(probabilities, labels))
    rank_sum = 0.0
    start = 0
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and pairs[end][0] == pairs[start][0]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        rank_sum += average_rank * sum(label for _, label in pairs[start:end])
        start = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def confidence_report(alignments: Iterable[dict], temperature: float = 1.0, bins: int = 10) -> dict:
    """Reliability metrics for emitted words, never treating deletions as scored."""
    if bins < 1:
        raise ValueError("bins must be positive.")
    alignments = list(alignments)
    words = [word for alignment in alignments for word in alignment["words"]]
    key = _temperature_key(temperature)
    probabilities = [word["confidence_by_temperature"][key] for word in words]
    labels = [int(word["correct"]) for word in words]
    if not words:
        return {"word_count": 0, "temperature": temperature, "ece": None, "brier": None, "nll": None}
    count = len(words)
    accuracy = sum(labels) / count
    reliability = []
    ece = 0.0
    for index in range(bins):
        indices = [i for i, probability in enumerate(probabilities) if min(int(probability * bins), bins - 1) == index]
        if indices:
            avg_confidence = sum(probabilities[i] for i in indices) / len(indices)
            bin_accuracy = sum(labels[i] for i in indices) / len(indices)
            ece += len(indices) / count * abs(bin_accuracy - avg_confidence)
        else:
            avg_confidence = bin_accuracy = None
        reliability.append({"lower": index / bins, "upper": (index + 1) / bins,
                            "count": len(indices), "mean_confidence": avg_confidence, "accuracy": bin_accuracy})
    correct = [probability for probability, label in zip(probabilities, labels) if label]
    incorrect = [probability for probability, label in zip(probabilities, labels) if not label]
    clipped = [min(max(probability, 1e-12), 1 - 1e-12) for probability in probabilities]
    return {
        "word_count": count,
        "temperature": float(temperature),
        "emitted_word_accuracy": accuracy,
        "mean_confidence": sum(probabilities) / count,
        "underconfidence_gap": accuracy - sum(probabilities) / count,
        "ece": ece,
        "brier": sum((probability - label) ** 2 for probability, label in zip(probabilities, labels)) / count,
        "nll": -sum(label * math.log(probability) + (1 - label) * math.log1p(-probability)
                    for probability, label in zip(clipped, labels)) / count,
        "correctness_auroc": _auc(probabilities, labels),
        "correct_word_mean_confidence": sum(correct) / len(correct) if correct else None,
        "incorrect_word_mean_confidence": sum(incorrect) / len(incorrect) if incorrect else None,
        "correct_words_below_0_5_fraction": sum(p < 0.5 for p in correct) / len(correct) if correct else None,
        "incorrect_words_above_0_8_fraction": sum(p >= 0.8 for p in incorrect) / len(incorrect) if incorrect else None,
        "unscored_deletions": sum(alignment.get("deletions", 0) for alignment in alignments),
        "reference_word_count": sum(alignment.get("reference_word_count", 0) for alignment in alignments),
        "reliability_bins": reliability,
    }


def fit_temperature(dev_alignments: Iterable[dict]) -> dict:
    """Select vocabulary-softmax temperature by word Brier on development only."""
    dev_alignments = list(dev_alignments)
    words = [word for alignment in dev_alignments for word in alignment["words"]]
    if not words:
        raise ValueError("Cannot fit calibration without development words.")
    common_keys = set.intersection(*(set(word["confidence_by_temperature"]) for word in words))
    candidates = [confidence_report(dev_alignments, float(key)) for key in common_keys]
    selected = min(candidates, key=lambda report: (report["brier"], abs(math.log(report["temperature"]))))
    return {
        "temperature": selected["temperature"],
        "objective": "minimum emitted-word Brier on development set",
        "dev_word_count": len(words),
        "dev_report": selected,
        "candidate_brier": {str(report["temperature"]): report["brier"] for report in
                            sorted(candidates, key=lambda report: report["temperature"])},
        "test_policy": "freeze this temperature before evaluating held-out test",
        "scope": "post-hoc confidence calibration; decoded words and WER are unchanged",
    }
