"""NEW opt-in production hook: immutable per-row draw contexts, caller-owned commit.

No sampler/optimizer/RNG, disk journal writer, decoder or GPU work lives here.
Prepared receipt payloads cannot advance a committed frontier. The native caller
owns optimizer commit, journal prefix and checkpoint transactions together.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import threading

import fastmss_train_augmentation_v1 as adapter_module
from fastmss_train_augmentation_v1 import AugmentationPolicy, canonical, require, storage_path


def file_binding(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    require(len(raw) <= 256 * 1024, "Only bounded source/policy metadata binding is supported")
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


@dataclass(frozen=True)
class AugmentationDrawContext:
    source_id: str
    language: str
    epoch: int
    draw_index: int
    phase: str = "weak_label_continued_training"
    split: str = "train"

    def __post_init__(self):
        require(isinstance(self.source_id, str) and self.source_id, "Exact source ID required")
        require(self.language in ("tr", "pt", "en", "es", "it", "fr", "de"), "Declared training/replay language required")
        require(type(self.epoch) is int and type(self.draw_index) is int and
                self.epoch >= 0 and self.draw_index >= 0, "Nonnegative durable integer sampler position required")
        require(self.split == "train", "This production reader binding cannot be used for DEV/TEST/teacher")
        require(self.phase in ("weak_label_continued_training", "cleaner_data_fine_tuning"), "Declared training phase required")

    def adapter_context(self):
        return {"split": self.split, "phase": self.phase,
                "role": "student_tr" if self.language == "tr" else "replay",
                "epoch": self.epoch, "draw_index": self.draw_index}


class PreparedOrderedBatchReader:
    """One ordered batch only. Never reuse this object for another/prefetched batch."""
    def __init__(self, augmenter, original_reader, contexts, *, batch_id, record_prepared):
        contexts = tuple(contexts)
        require(0 < len(contexts) <= 64 and all(isinstance(v, AugmentationDrawContext) for v in contexts),
                "One bounded ordered batch of immutable draw contexts required")
        require(len({(v.epoch, v.draw_index) for v in contexts}) == len(contexts),
                "Each actual row needs a distinct declared draw position, including duplicate source IDs")
        require(isinstance(batch_id, str) and 0 < len(batch_id) <= 200, "Bounded durable batch ID required")
        self._contexts = contexts
        self._batch_id = batch_id
        self._original_reader = original_reader
        self._augmenter = augmenter
        self._record = record_prepared
        self._receipts = []  # <=64 small immutable serialized receipts, no waveform cache
        self._cursor = 0
        self._failed = False
        self._lock = threading.Lock()

    def __call__(self, row):
        # Actual collate audio_reader is serial. A shared/prefetched global
        # mutable draw pointer is not used; each batch owns its own contexts.
        with self._lock:
            require(not self._failed and self._cursor < len(self._contexts), "Prepared batch failed, exhausted or reused")
            context = self._contexts[self._cursor]
            try:
                require(row.get("source_id") == context.source_id and row.get("language") == context.language,
                        "Actual row order/source/language differs from durable sampler context")
                waveform = self._original_reader(row)
                transformed, receipt = self._augmenter.apply(waveform, row, **context.adapter_context())
                event = {"state": "prepared_not_committed", "batch_id": self._batch_id,
                         "batch_row": self._cursor, "draw": asdict(context), "augmentation": receipt}
                encoded = canonical(event)
                self._record(json.loads(encoded))
                self._receipts.append(encoded)
                self._cursor += 1
                return transformed
            except BaseException:
                # Failed/prepared input work must be rebuilt from its original
                # durable sampler state; never silently consume a draw on retry.
                self._failed = True
                raise

    def prepared_payload(self):
        require(not self._failed and self._cursor == len(self._contexts), "Incomplete/failed batch cannot prepare a commit payload")
        draws = [json.loads(value) for value in self._receipts]
        return {"kind": "FastMSS_prepared_batch_v1", "state": "prepared_not_committed",
                "batch_id": self._batch_id, "rows": draws,
                "payload_sha256": hashlib.sha256(canonical(draws)).hexdigest(),
                "committed_frontier_advanced": False,
                "commit_owner": "native caller only after actual positive optimizer commit"}


class FastMSSOrderedDrawBinding:
    def __init__(self, augmenter, *, policy_path, adapter_path):
        # Read the enabled/disabled config itself, not a silent in-memory override.
        require(AugmentationPolicy.from_json(policy_path) == augmenter.policy,
                "Actual augmenter differs from the declared policy file")
        require(Path(adapter_path).resolve(strict=True) == Path(adapter_module.__file__).resolve(strict=True),
                "Bound adapter path differs from the actual imported module")
        self.augmenter = augmenter
        self._identity = {
            "kind": "FastMSS_production_draw_binding_v1",
            "adapter": file_binding(adapter_path), "policy_file": file_binding(policy_path),
            "binding_helper": file_binding(__file__), "policy_sha256": augmenter.policy_sha256,
            "registry_sha256": augmenter.registry_sha256,
            "numpy_version": package_version("numpy"), "scipy_version": package_version("scipy"),
            "draw_owner": "existing durable sampler", "commit_owner": "existing native optimizer/journal/checkpoint caller"}

    @property
    def identity(self):
        return json.loads(canonical(self._identity))  # callers cannot mutate the stored declaration

    def prepare_batch(self, original_reader, contexts, *, batch_id, record_prepared):
        return PreparedOrderedBatchReader(self.augmenter, original_reader, contexts,
            batch_id=batch_id, record_prepared=record_prepared)

    def checkpoint_binding(self, *, committed_journal):
        journal = dict(committed_journal)
        require(set(journal) == {"path", "offset", "sha256"}, "Exact caller-owned committed journal prefix required")
        path = storage_path(journal["path"])
        require(type(journal["offset"]) is int and 0 <= journal["offset"] <= path.stat().st_size,
                "Committed journal byte offset out of range")
        digest = journal["sha256"]
        require(isinstance(digest, str) and len(digest) == 64 and
                all(char in "0123456789abcdef" for char in digest), "Committed journal SHA256 required")
        journal["path"] = str(path)
        return {"identity": self.identity, "committed_journal": journal,
                "draw_frontier": "canonical sampler state in same native checkpoint; no independent augmentation counter"}

    def verify_restore(self, saved, *, verify_journal_prefix):
        require(saved["identity"] == self.identity, "Adapter/policy/registry/environment binding differs at resume")
        current = self.checkpoint_binding(committed_journal=saved["committed_journal"])
        require(current == saved, "Augmentation checkpoint contract differs at resume")
        # The parent journal already owns bounded prefix verification. Do not
        # reread arbitrarily large journals or consume an uncheckpointed tail.
        require(verify_journal_prefix(dict(saved["committed_journal"])) is True,
                "Native caller rejected committed journal prefix")
        return self.identity
