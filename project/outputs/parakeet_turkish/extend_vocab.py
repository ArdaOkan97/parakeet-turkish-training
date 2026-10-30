"""Append missing Turkish characters to HF Parakeet TDT without resetting weights.

Call before constructing the optimizer. The helper mutates the supplied processor
and model in memory, returns a JSON-serializable audit, and writes no files.
Save both objects with save_pretrained after training.

This is an HF checkpoint extension, not a NeMo .nemo tokenizer conversion.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Iterable


CHARACTER_DONORS = {"ğ": "g", "Ğ": "G", "ı": "i", "İ": "I", "ş": "s", "Ş": "S"}
NEW_ROW_BIAS_PENALTY = 2.0
RETENTION_TEXTS = (
    "Hello world. I've called 42 people; their names are Alice and Michael.",
    "¿Cómo estás? El niño pidió una canción y café para mañana.",
    "L'Italia è bella: perché abbiamo già un'altra città?",
    "Bonjour à tous. Noël, cœur, français, été et garçon: 123.",
    "Die Straße führt über Köln. Öl, Größe, grüße und zwölf Wörter.",
    "çÇöÖüÜ abc ABC xyz XYZ 0123456789 ,.!?'\"-:/()[]",
    "",
)
TURKISH_TEXTS = (
    "Işık İstanbul şehri, dağların aşağısında.",
    "Türkçe çığ, yağış, çağrı, ışık ve İĞŞ: ğğıİşŞ.",
    "Ğğ Iı İi Şş Çç Öö Üü: Türkiye'de konuşuyoruz.",
)


def _hash_json(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _backend(tokenizer):
    backend = getattr(tokenizer, "backend_tokenizer", None)
    return backend if backend is not None else tokenizer._tokenizer


def _make_tokenizer(tokenizer, backend):
    # Preserve the tokenizer class and its HF settings; modify the native BPE
    # vocabulary rather than registering substring-matching AddedTokens.
    kwargs = copy.deepcopy(tokenizer.init_kwargs)
    for key in ("tokenizer_file", "tokenizer_object"):
        kwargs.pop(key, None)
    kwargs.update(copy.deepcopy(tokenizer.special_tokens_map))
    kwargs["tokenizer_object"] = backend
    return tokenizer.__class__(**kwargs)


def _text_statistics(backend, texts):
    unknown_id = backend.token_to_id("<unk>")
    tokens = 0
    unknown = 0
    words = 0
    rows = 0
    for text in texts:
        encoding = backend.encode(text, add_special_tokens=False)
        tokens += len(encoding.ids)
        unknown += encoding.ids.count(unknown_id)
        words += len(text.split())
        rows += 1
    return {
        "rows": rows,
        "tokens": tokens,
        "unknown_tokens": unknown,
        "tokens_per_word": tokens / words if words else None,
    }


def extend_processor_and_model(processor, model, training_texts: Iterable[str] | None = None):
    """Add six missing Turkish characters while retaining original token IDs.

    Blank keeps its original ID (8192 for the official v3 HF checkpoint). New
    tokens follow blank. Original token/blank head rows and embedding rows remain
    identical; duration rows and their generation suppression IDs move to the
    end of the expanded head. New rows copy
    ASCII donor rows with a 2-logit bias penalty. Therefore a new row cannot beat
    its original donor in initial greedy decoding. Repeated invocation is safe.
    """
    import torch
    from tokenizers import Tokenizer

    old_tokenizer = processor.tokenizer
    old_backend = _backend(old_tokenizer)
    original = json.loads(old_backend.to_str())
    if original["model"]["type"] != "BPE":
        raise ValueError("This extension requires the official BPE tokenizer.")
    old_vocab = old_backend.get_vocab()
    old_size = int(model.config.vocab_size)
    blank_id = int(model.config.blank_token_id)
    if set(old_vocab.values()) != set(range(old_size)):
        raise ValueError("Tokenizer IDs must exactly match the contiguous model vocabulary.")
    if old_backend.token_to_id(processor.blank_token) != blank_id:
        raise ValueError("Processor and model blank IDs disagree.")
    additions = [char for char in CHARACTER_DONORS if char not in old_vocab]
    if not additions:
        return {
            "changed": False,
            "reason": "All six Turkish characters already exist.",
            "vocab_size": old_size,
            "blank_token_id": blank_id,
        }

    updated = copy.deepcopy(original)
    core = updated["model"]["vocab"]
    # The official blank is an added special token outside the core BPE vocab.
    # Include its SAME ID in the core before appending, avoiding a sparse ID map.
    for token, token_id in sorted(old_vocab.items(), key=lambda item: item[1]):
        if token not in core:
            core[token] = token_id
    for index, char in enumerate(additions, start=old_size):
        core[char] = index
    new_size = old_size + len(additions)
    new_backend = Tokenizer.from_str(json.dumps(updated, ensure_ascii=False))
    new_tokenizer = _make_tokenizer(old_tokenizer, new_backend)
    final_backend = _backend(new_tokenizer)
    if len(new_tokenizer) != new_size:
        raise AssertionError("Unexpected tokenizer size after extension.")
    for token, token_id in old_vocab.items():
        if final_backend.token_to_id(token) != token_id:
            raise AssertionError(f"An original token changed ID: {token!r}.")
    unchanged_samples = []
    for text in RETENTION_TEXTS:
        old_ids = old_backend.encode(text, add_special_tokens=False).ids
        new_ids = final_backend.encode(text, add_special_tokens=False).ids
        if old_ids != new_ids:
            raise AssertionError(f"Original-language tokenization changed: {text!r}.")
        unchanged_samples.append(text)
    for text in TURKISH_TEXTS:
        ids = final_backend.encode(text, add_special_tokens=False).ids
        if final_backend.token_to_id("<unk>") in ids:
            raise AssertionError(f"Turkish test contains an unknown token: {text!r}.")
        if final_backend.decode(ids, skip_special_tokens=True) != text:
            raise AssertionError(f"Turkish test failed its exact round trip: {text!r}.")
    final_json = json.loads(final_backend.to_str())
    for key, value in original.items():
        if key != "model" and final_json[key] != value:
            raise AssertionError(f"Tokenizer metadata changed unexpectedly: {key}.")
    for key, value in original["model"].items():
        if key != "vocab" and final_json["model"][key] != value:
            raise AssertionError(f"BPE settings or merge order changed: {key}.")

    old_embedding = model.decoder.embedding
    old_head = model.joint.head
    duration_count = len(model.config.durations)
    generation_config = getattr(model, "generation_config", None)
    suppression_audit = {}
    # GenerationMixin treats the joint output as a single vocabulary. TDT's
    # duration classes must stay suppressed there even though the TDT frame
    # updater still reads their raw logits. Their IDs move with the head rows.
    for field in ("suppress_tokens", "begin_suppress_tokens"):
        before = getattr(generation_config, field, None) if generation_config is not None else None
        after = None if before is None else [
            new_size + (token_id - old_size)
            if old_size <= token_id < old_size + duration_count else token_id
            for token_id in before
        ]
        suppression_audit[field] = {"before": copy.deepcopy(before), "after": after}
    if old_embedding.num_embeddings != old_size:
        raise ValueError("Unexpected decoder embedding size.")
    if old_head.out_features != old_size + duration_count or old_head.bias is None:
        raise ValueError("Unexpected TDT joint head layout.")
    new_embedding = torch.nn.Embedding(
        new_size,
        old_embedding.embedding_dim,
        padding_idx=old_embedding.padding_idx,
        device=old_embedding.weight.device,
        dtype=old_embedding.weight.dtype,
    )
    new_head = torch.nn.Linear(
        old_head.in_features,
        new_size + duration_count,
        bias=True,
        device=old_head.weight.device,
        dtype=old_head.weight.dtype,
    )
    donor_audit = []
    with torch.no_grad():
        new_embedding.weight[:old_size].copy_(old_embedding.weight)
        new_head.weight[:old_size].copy_(old_head.weight[:old_size])
        new_head.bias[:old_size].copy_(old_head.bias[:old_size])
        new_head.weight[new_size:].copy_(old_head.weight[old_size:])
        new_head.bias[new_size:].copy_(old_head.bias[old_size:])
        for char in additions:
            new_id = final_backend.token_to_id(char)
            donor_char = CHARACTER_DONORS[char]
            donor_id = old_backend.token_to_id(donor_char)
            if donor_id is None or donor_id == old_backend.token_to_id("<unk>"):
                raise ValueError(f"Missing donor character {donor_char!r}.")
            new_embedding.weight[new_id].copy_(old_embedding.weight[donor_id])
            new_head.weight[new_id].copy_(old_head.weight[donor_id])
            new_head.bias[new_id].copy_(old_head.bias[donor_id] - NEW_ROW_BIAS_PENALTY)
            donor_audit.append({"token": char, "id": new_id, "donor": donor_char, "donor_id": donor_id})

    preserved = {
        "embedding_rows": torch.equal(new_embedding.weight[:old_size], old_embedding.weight),
        "token_head_weights": torch.equal(new_head.weight[:old_size], old_head.weight[:old_size]),
        "token_head_biases": torch.equal(new_head.bias[:old_size], old_head.bias[:old_size]),
        "duration_head_weights": torch.equal(new_head.weight[new_size:], old_head.weight[old_size:]),
        "duration_head_biases": torch.equal(new_head.bias[new_size:], old_head.bias[old_size:]),
    }
    if not all(preserved.values()):
        raise AssertionError("An original embedding or output row changed.")
    new_embedding.weight.requires_grad_(old_embedding.weight.requires_grad)
    new_head.weight.requires_grad_(old_head.weight.requires_grad)
    new_head.bias.requires_grad_(old_head.bias.requires_grad)
    new_embedding.train(old_embedding.training)
    new_head.train(old_head.training)

    texts = list(training_texts) if training_texts is not None else list(TURKISH_TEXTS)
    audit = {
        "changed": True,
        "original_vocab_size": old_size,
        "extended_vocab_size": new_size,
        "blank_token_id": blank_id,
        "original_core_vocab_size": len(original["model"]["vocab"]),
        "original_byte_fallback_flag": original["model"].get("byte_fallback", False),
        "original_byte_piece_count": sum(token.startswith("<0x") for token in old_vocab),
        "added_characters": donor_audit,
        "new_row_bias_penalty": NEW_ROW_BIAS_PENALTY,
        "duration_rows_original": [old_size, old_size + duration_count],
        "duration_rows_extended": [new_size, new_size + duration_count],
        "generation_suppression": suppression_audit,
        "original_ids_preserved": True,
        "original_merge_order_preserved": True,
        "weights_preserved": preserved,
        "retention_tokenization_checks": unchanged_samples,
        "turkish_roundtrip_checks": list(TURKISH_TEXTS),
        "text_audit_before": _text_statistics(old_backend, texts),
        "text_audit_after": _text_statistics(final_backend, texts),
        "original_tokenizer_sha256": _hash_json(original),
        "extended_tokenizer_sha256": _hash_json(final_json),
    }
    # Commit only after the tokenizer and both replacement modules pass audits.
    processor.tokenizer = new_tokenizer
    model.decoder.embedding = new_embedding
    model.joint.head = new_head
    model.joint.vocab_size = new_size
    model.config.vocab_size = new_size
    if hasattr(model.decoder, "vocab_size"):
        model.decoder.vocab_size = new_size
    if generation_config is not None:
        if hasattr(generation_config, "vocab_size"):
            generation_config.vocab_size = new_size
        for field, values in suppression_audit.items():
            if hasattr(generation_config, field):
                setattr(generation_config, field, values["after"])
    return audit
