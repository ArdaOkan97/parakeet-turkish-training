"""Ordered clean-phase audio protection; never an augmentation or cache producer.

Caller supplies independent full targets and clean PCM identity. This wrapper
returns the original ndarray unchanged. Sampler/Adam/journal commit is external.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
import struct
import threading

from fastmss_ordered_draw_binding_v1 import AugmentationDrawContext, file_binding
from fastmss_train_augmentation_v1 import canonical, require

PHASE = "cleaner_data_fine_tuning"
KIND = "FastMSS_explicit_clean_phase_protection_v1"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def waveform_hash(waveform, guard):
    view = memoryview(waveform).cast("B")
    value = hashlib.sha256()
    for offset in range(0, len(view), 1024 * 1024):
        guard()
        value.update(view[offset:offset + 1024 * 1024])
    return value.hexdigest()


def row_contract(row, context, target_ids, audio):
    require(isinstance(context, AugmentationDrawContext) and context.phase == PHASE and context.split == "train",
            "Explicit cleaner TRAIN context required; no weak/evaluation default")
    require(row["source_id"] == context.source_id and row["language"] == context.language and
            canonical(row.get("audio_draw_context")) == canonical(asdict(context)),
            "Exact row/source/language/epoch/draw/phase binding required")
    role = row.get("augmentation_role")
    require(role in ("student_tr", "replay", "teacher") and (role != "student_tr" or context.language == "tr") and
            (role != "replay" or context.language != "tr"),
            "Explicit cleaner student/replay/teacher role required")
    require(isinstance(row["text"], str) and type(row["duration"]) in (int, float) and
            math.isfinite(row["duration"]) and row["duration"] > 0 and
            type(row["encoder_frames"]) is int and row["encoder_frames"] > 0, "Exact clean source metadata required")
    require(type(target_ids) in (list, tuple) and target_ids and all(type(v) is int for v in target_ids),
            "Caller must provide every full target ID, never a prefix or count alone")
    ceiling = 8711 if context.language == "tr" else 8193
    require(all(0 <= v < ceiling and v != 8192 for v in target_ids), "Full declared target vocabulary differs")
    require(type(row["target_tokens"]) is int and row["target_tokens"] == len(target_ids), "Full target count mismatch")
    if context.language == "tr":
        require(type(row.get("tr_metadata_labels")) in (tuple, list) and
                all(type(v) is int for v in row["tr_metadata_labels"]) and
                tuple(row["tr_metadata_labels"]) == tuple(target_ids), "Full cleaner TR target IDs differ")
    else:
        require(type(row.get("clean_target_ids")) in (tuple, list) and
                all(type(v) is int for v in row["clean_target_ids"]) and
                tuple(row["clean_target_ids"]) == tuple(target_ids), "Full unchanged replay/teacher target IDs differ")
    require(type(audio) is dict and set(audio) == {"sample_rate", "samples", "waveform_sha256"} and
            audio["sample_rate"] == 16000 and type(audio["samples"]) is int and audio["samples"] > 0 and
            isinstance(audio["waveform_sha256"], str) and len(audio["waveform_sha256"]) == 64 and
            all(v in "0123456789abcdef" for v in audio["waveform_sha256"]), "Independent exact clean PCM identity required")
    encoded_ids = hashlib.sha256()
    for value in target_ids:
        encoded_ids.update(struct.pack("<I", value))
    return {"draw": asdict(context), "role": role, "raw_text_sha256": digest(row["text"].encode("utf-8")),
            "duration_metadata": row["duration"], "encoder_frames": row["encoder_frames"],
            "target_tokens": len(target_ids), "target_ids_u32_sha256": encoded_ids.hexdigest(),
            "target_vocabulary": ceiling, "clean_PCM": dict(audio)}


class CleanPhaseOrderedAudioReader:
    """One serial ordered batch; direct clean reads, no augmentation transport."""
    def __init__(self, binding, rows, contexts, *, batch_id, clean_reader, expected_target_ids,
                 expected_audio, guard, record_prepared):
        rows, contexts = tuple(rows), tuple(contexts)
        targets, audio = tuple(expected_target_ids), tuple(expected_audio)
        require(0 < len(rows) == len(contexts) == len(targets) == len(audio) <= 64,
                "Exact bounded clean rows/contexts/full-target/audio cardinality required")
        require(isinstance(batch_id, str) and 0 < len(batch_id) <= 200, "Bounded explicit batch ID required")
        require(all(isinstance(v, AugmentationDrawContext) for v in contexts) and
                len({(v.epoch, v.draw_index) for v in contexts}) == len(contexts), "Distinct immutable durable draws required")
        require(binding.augmenter.policy.phases == ("weak_label_continued_training",),
                "Frozen broad-only policy required; cleaner never adds an augmentation phase")
        self._identity = {"kind": KIND, "wrapper": file_binding(__file__), "phase": PHASE,
                          "frozen_broad_augmentation": binding.identity, "waveform_cache_created": False,
                          "augmentation_or_transport_invoked": False}
        self._contracts = tuple(canonical(row_contract(r, c, t, a)) for r, c, t, a in zip(rows, contexts, targets, audio))
        self._contexts, self._targets = contexts, tuple(tuple(v) for v in targets)
        self._audio = tuple(dict(v) for v in audio)
        self._batch_id, self._reader, self._guard, self._record = batch_id, clean_reader, guard, record_prepared
        self._cursor, self._failed, self._events, self._lock = 0, False, [], threading.Lock()
        self.transport_binding = {"kind": KIND, "identity": self.identity, "batch_id": batch_id,
                                  "ordered_contracts_sha256": digest(canonical([v.decode("utf-8") for v in self._contracts])),
                                  "prepared_is_committed": False, "cache_manifest": None}

    @property
    def identity(self):
        import json
        return json.loads(canonical(self._identity))

    def __call__(self, row):
        import json
        import numpy as np
        with self._lock:
            require(not self._failed and self._cursor < len(self._contexts), "Clean batch failed/exhausted/reused")
            index = self._cursor
            try:
                self._guard()
                def actual_contract():
                    return canonical(row_contract(row, self._contexts[index], self._targets[index], self._audio[index]))
                require(actual_contract() == self._contracts[index], "Actual ordered row or full target identity changed")
                wave = self._reader(row)
                require(isinstance(wave, np.ndarray) and wave.dtype == np.float32 and wave.ndim == 1 and
                        wave.flags.c_contiguous and len(wave) == self._audio[index]["samples"] and np.isfinite(wave).all(),
                        "Original clean reader PCM geometry/dtype/finite contract differs")
                actual_sha = waveform_hash(wave, self._guard)
                require(actual_sha == self._audio[index]["waveform_sha256"] and actual_contract() == self._contracts[index],
                        "Clean source PCM/metadata/targets changed")
                event = {"state": "prepared_not_committed", "batch_id": self._batch_id, "batch_row": index,
                         "draw": asdict(self._contexts[index]), "augmentation": {
                             "source_id": row["source_id"], "policy_sha256": self._identity["frozen_broad_augmentation"]["policy_sha256"],
                             "applied": False, "reason": "explicit_cleaner_phase_direct_original_audio",
                             "label_policy": "unchanged_single_target"},
                         "clean_protection": json.loads(self._contracts[index]), "clean_waveform_sha256": actual_sha}
                encoded = canonical(event)
                self._record(json.loads(encoded)); self._events.append(encoded); self._cursor += 1
                return wave  # Exact same object/bytes; no copy, clipping, resampling or transform.
            except BaseException:
                self._failed = True
                raise

    def prepared_payload(self):
        import json
        require(not self._failed and self._cursor == len(self._contexts), "Incomplete/failed clean batch cannot prepare commit")
        events = [json.loads(value) for value in self._events]
        return {"kind": "FastMSS_prepared_batch_v1", "protection_kind": KIND,
                "state": "prepared_not_committed", "batch_id": self._batch_id, "identity": self.identity,
                "rows": events, "payload_sha256": digest(canonical(events)), "committed_frontier_advanced": False,
                "commit_owner": "native caller only after actual positive optimizer commit"}
