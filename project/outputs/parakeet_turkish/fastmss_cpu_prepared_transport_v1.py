"""CPU-only augmentation transport, actual-exit gate and owned transient cache.

Producer runs inside the bounded decoder child. Consumer never calls apply().
Native optimizer/journal/checkpoint/sampler commit remains entirely caller-owned.
No Torch/model/GPU imports. Frozen augmentation modules are reused unchanged.
"""
from __future__ import annotations
from dataclasses import asdict
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import threading
import time

from fastmss_train_augmentation_v1 import canonical, require, storage_path
from fastmss_ordered_draw_binding_v1 import AugmentationDrawContext, file_binding

STORAGE = Path(r"E:\gpu_training_scratchpad")
CACHE_ROOT = STORAGE / "temp/ASR18_FastMSS_prepared_v1"
MAX_CACHE_BYTES = 128 * 1024 * 1024
MAX_JOBS = 2
MAX_MANIFEST_BYTES = 256 * 1024


@contextmanager
def capacity_lock():
    """One tiny Windows metadata lock; never a GPU/data-download permission gate."""
    import msvcrt
    lock = CACHE_ROOT.with_suffix(".capacity.lock").resolve()
    require(lock.is_relative_to(STORAGE.resolve(strict=True)), "Capacity lock must stay on E")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as stream:
        if stream.seek(0, 2) == 0:
            stream.write(b"0");stream.flush()
        stream.seek(0)
        deadline = time.monotonic() + 2
        while True:
            try:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                require(time.monotonic() < deadline, "Owned capacity metadata busy; no cache/draw consumed")
                time.sleep(0.02)
        try:
            yield
        finally:
            stream.seek(0);msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def digest_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def stream_hash(path, guard):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            guard()
            digest.update(chunk)
    return digest.hexdigest()


def record(path, guard):
    path = storage_path(path)
    return {"path": str(path), "sha256": stream_hash(path, guard), "bytes": path.stat().st_size}


def read_bound(binding, guard, *, limit=MAX_MANIFEST_BYTES):
    path = storage_path(binding["path"])
    require(0 < path.stat().st_size <= limit, "Bounded artifact size required")
    require(record(path, guard) == binding, "Artifact SHA/size/path mismatch")
    return json.loads(path.read_bytes())


def write_new_json(path, value):
    raw = canonical(value)
    require(len(raw) <= MAX_MANIFEST_BYTES, "Bounded metadata exceeded")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def scoped_job(path):
    path = storage_path(path)
    require(path.parent == CACHE_ROOT.resolve(strict=True) and
            re.fullmatch(r"[A-Za-z0-9_-]{1,80}", path.name), "Exact dedicated owned cache job required")
    return path


def _capacity_usage_unlocked():
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    root = storage_path(CACHE_ROOT)
    jobs = list(root.iterdir())
    require(len(jobs) <= MAX_JOBS, "Too many outstanding owned cache jobs")
    total = 0
    for job in jobs:
        job = scoped_job(job)
        files = list(job.iterdir())
        require(len(files) <= 16, "Unexpected cache cardinality")
        actual = 0
        for path in files:
            resolved = storage_path(path)
            require(resolved.parent == job and path.is_file(), "Cache contains unexpected directory/link")
            actual += path.stat().st_size
        owner = job / "owner.json"
        if owner.exists():
            require(owner.stat().st_size <= MAX_MANIFEST_BYTES, "Oversized cache owner metadata")
            reserved = json.loads(owner.read_bytes())["reserved_bytes"]
            require(type(reserved) is int and 0 < reserved <= MAX_CACHE_BYTES, "Invalid capacity reservation")
            total += max(actual, reserved)
        else:
            total += MAX_CACHE_BYTES  # incomplete creation fails closed, never a free slot
    return total, len(jobs)


def capacity_usage():
    with capacity_lock():
        return _capacity_usage_unlocked()


def waveform_hash(waveform, guard):
    view = memoryview(waveform).cast("B")
    digest = hashlib.sha256()
    for offset in range(0, len(view), 1024 * 1024):
        guard()
        digest.update(view[offset:offset + 1024 * 1024])
    return digest.hexdigest()


def target_contract(row):
    require(row["language"] == "tr" and row.get("augmentation_role") == "student_tr", "Prepared audio target must explicitly be student TR")
    ids = row["tr_metadata_labels"]
    require(isinstance(ids, (list, tuple)) and ids and
            all(type(v) is int and 0 <= v < 8711 and v != 8192 for v in ids), "Exact full BPE target IDs required")
    require(type(row["target_tokens"]) is int and row["target_tokens"] == len(ids), "Target count mismatch")
    require(type(row["encoder_frames"]) is int and row["encoder_frames"] > 0, "Positive exact frame metadata required")
    require(isinstance(row["text"], str) and math.isfinite(row["duration"]) and row["duration"] > 0,
            "Raw text and actual decoded duration required")
    samples = round(row["duration"] * 16000)
    require(samples > 0 and abs(samples / 16000 - row["duration"]) < 1e-9, "Exact decoded sample count required")
    target_hash = hashlib.sha256()
    for value in ids:
        target_hash.update(struct.pack("<I", value))
    return {"source_id": row["source_id"], "language": "tr", "augmentation_role": "student_tr", "sample_rate": 16000,
            "samples": samples, "encoder_frames": row["encoder_frames"], "target_tokens": len(ids),
            "target_ids_u32_sha256": target_hash.hexdigest(), "raw_text_sha256": digest_bytes(row["text"].encode("utf-8"))}


def eligible(context):
    return context.language == "tr" and context.phase == "weak_label_continued_training" and context.split == "train"


def producer_closed(identity):
    import psutil
    try:
        actual = psutil.Process(identity["pid"])
        return abs(actual.create_time() - identity["create_time"]) > 0.001
    except psutil.NoSuchProcess:
        return True


def prepare_TR_transport(binding, rows, contexts, *, batch_id, job_id, clean_reader, guard):
    """Call inside one bounded decoder CHILD; returns bindings, never commits."""
    import numpy as np
    import psutil
    rows, contexts = tuple(rows), tuple(contexts)
    require(0 < len(rows) == len(contexts) <= 14, "One bounded TR update required")
    require(all(isinstance(c, AugmentationDrawContext) and eligible(c) for c in contexts),
            "Only weak TRAIN/TR may be prepared; evaluation/teacher/replay/clean phase protected")
    require(len({(c.epoch, c.draw_index) for c in contexts}) == len(contexts), "Duplicate durable draw position")
    require(binding.augmenter.policy.phases == ("weak_label_continued_training",), "Broad-only policy required")
    require(re.fullmatch(r"[A-Za-z0-9_-]{1,80}", job_id), "Safe unique job ID required")
    contracts = [target_contract(r) for r in rows]
    require(all(r["source_id"] == c.source_id and r["language"] == c.language for r, c in zip(rows, contexts)),
            "Declared ordered draw identity differs from source rows")
    required = sum(c["samples"] * 4 + 512 for c in contracts) + 1024 * 1024
    guard()
    folder = CACHE_ROOT / job_id
    process = psutil.Process()
    producer = {"pid": os.getpid(), "ppid": os.getppid(), "create_time": process.create_time()}
    names = [f"audio_{i:03}.npy" for i in range(len(rows))]
    owner = {"kind": "FastMSS_owned_transient_cache_v1", "job_id": job_id, "batch_id": batch_id,
             "producer": producer, "transport_source": file_binding(__file__),
             "owned_names": ["owner.json", "manifest.json", *names], "prepared_is_committed": False,
             "reserved_bytes": required}
    with capacity_lock():
        used, jobs = _capacity_usage_unlocked()
        require(jobs < MAX_JOBS and used + required <= MAX_CACHE_BYTES, "Fixed 128MiB/two-job cache capacity reached; no draw consumed")
        folder.mkdir(exist_ok=False)
        write_new_json(folder / "owner.json", owner)
    clean = []

    def observed_read(row):
        wave = clean_reader(row)
        contract = contracts[len(clean)]
        require(isinstance(wave, np.ndarray) and wave.dtype == np.float32 and wave.ndim == 1 and
                wave.flags.c_contiguous and len(wave) == contract["samples"] and np.isfinite(wave).all(),
                "Clean waveform sample/rate/dtype contract mismatch")
        clean.append(waveform_hash(wave, guard))
        return wave

    reader = binding.prepare_batch(observed_read, contexts, batch_id=batch_id, record_prepared=lambda _: None)
    entries = []
    for index, row in enumerate(rows):
        wave = reader(row)
        path = folder / names[index]
        # No permanent source rewrite; only this exact owned transient .npy.
        with path.open("xb") as stream:
            np.save(stream, wave, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        entries.append({"draw": asdict(contexts[index]), "target": contracts[index],
            "clean_waveform_sha256": clean[index], "prepared_waveform_sha256": waveform_hash(wave, guard),
            "cache": record(path, guard)})
        del wave
        guard()
    payload = reader.prepared_payload()
    for entry, event in zip(entries, payload["rows"]):
        entry["prepared_event"] = event
    semantic = {"identity": binding.identity, "batch_id": batch_id,
                "transport_source": owner["transport_source"],
                "entries": [{k: v for k, v in entry.items() if k != "cache"} for entry in entries]}
    manifest = {"kind": "FastMSS_CPU_prepared_transport_v1", "job_id": job_id, "batch_id": batch_id,
        "owner": record(folder / "owner.json", guard), "producer": producer,
        "identity": binding.identity, "transport_source": owner["transport_source"], "entries": entries,
        "semantic_sha256": digest_bytes(canonical(semantic)), "prepared_is_committed": False,
        "sample_rate": 16000, "capacity_bytes": MAX_CACHE_BYTES}
    write_new_json(folder / "manifest.json", manifest)
    require(capacity_usage()[0] <= MAX_CACHE_BYTES, "Actual cache capacity exceeded")
    return {"manifest": record(folder / "manifest.json", guard), "owner": manifest["owner"],
            "semantic_sha256": manifest["semantic_sha256"], "producer": producer,
            "sampler_journal_frontier_advanced": False}


def observe_closed_child(child, *, owner_record, manifest_record, proof_path, guard, timeout=10):
    """Parent observes its OWN Popen exit. No closure is inferred from a file."""
    code = child.wait(timeout=timeout)
    require(child.poll() is not None, "Owned CPU child is still running")
    owner = read_bound(owner_record, guard)
    producer = owner["producer"]
    require(producer["pid"] == child.pid or producer["ppid"] == child.pid, "CPU producer is not the owned child")
    require(producer_closed(producer), "Actual CPU producer is still alive")
    proof = {"kind": "FastMSS_owned_CPU_actual_closure_v1", "actual_exit_code": code,
        "owned_handle_pid": child.pid, "producer": producer, "owner": owner_record,
        "manifest": manifest_record, "job_id": owner["job_id"], "batch_id": owner["batch_id"],
        "parent_observed_wait": True, "observer_source": file_binding(__file__)}
    target = Path(proof_path).resolve()
    require(target.is_relative_to(STORAGE.resolve(strict=True)) and not target.is_relative_to(scoped_job(Path(owner_record["path"]).parent)),
            "Closure proof must be E-only outside the transient cache")
    target.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(target, proof)
    return record(target, guard)


def verify_closure(closure_record, owner_record, manifest_record, guard, *, success):
    closure = read_bound(closure_record, guard)
    owner = read_bound(owner_record, guard)
    require(closure["kind"] == "FastMSS_owned_CPU_actual_closure_v1" and closure["parent_observed_wait"] is True and
            closure["owner"] == owner_record and closure["manifest"] == manifest_record and
            closure["producer"] == owner["producer"] and closure["job_id"] == owner["job_id"] and
            closure["batch_id"] == owner["batch_id"] and producer_closed(owner["producer"]), "Stale/wrong/live CPU closure")
    require(not success or closure["actual_exit_code"] == 0, "CPU preparation did not actually close0")
    require(closure["observer_source"] == file_binding(__file__) and owner["transport_source"] == file_binding(__file__),
            "Transport implementation changed")
    return owner


class VerifiedPreparedAudioReader:
    """GPU parent consumes prepared TR ONCE; no transform/apply/SciPy call here."""
    def __init__(self, binding, expected_rows, contexts, *, batch_id, manifest_record, closure_record,
                 protected_reader, record_prepared, guard, expected_clean_hashes):
        self._rows, self._contexts = tuple(expected_rows), tuple(contexts)
        require(0 < len(self._rows) == len(self._contexts) <= 64, "Exact ordered batch cardinality required")
        require(all(isinstance(c, AugmentationDrawContext) for c in self._contexts), "Immutable contexts required")
        require(all(r["source_id"] == c.source_id and r["language"] == c.language for r, c in zip(self._rows, self._contexts)),
                "Expected row/context identity differs")
        require(all(r.get("augmentation_role") in ("student_tr", "replay", "teacher") for r in self._rows),
                "Explicit student/replay/teacher role required")
        require(binding.augmenter.policy.phases == ("weak_label_continued_training",), "Broad-only policy required")
        require(len({(c.epoch, c.draw_index) for c in self._contexts}) == len(self._contexts), "Duplicate durable draw position")
        self._binding, self._batch_id, self._guard = binding, batch_id, guard
        self._protected_reader, self._record = protected_reader, record_prepared
        self._cursor, self._failed, self._events, self._lock = 0, False, [], threading.Lock()
        manifest = read_bound(manifest_record, guard)
        require(manifest["kind"] == "FastMSS_CPU_prepared_transport_v1" and manifest["batch_id"] == batch_id and
                manifest["identity"] == binding.identity and manifest["sample_rate"] == 16000 and
                manifest["prepared_is_committed"] is False and manifest["capacity_bytes"] == MAX_CACHE_BYTES and
                manifest["transport_source"] == file_binding(__file__),
                "Stale/wrong policy, source, batch, rate or identity")
        folder = scoped_job(Path(manifest_record["path"]).parent)
        require(Path(manifest_record["path"]).name == "manifest.json", "Exact producer manifest filename required")
        owner = verify_closure(closure_record, manifest["owner"], manifest_record, guard, success=True)
        require(owner["job_id"] == manifest["job_id"] == folder.name and owner["batch_id"] == batch_id and
                {p.name for p in folder.iterdir()} == set(owner["owned_names"]), "Stale/missing/extra owned cache files")
        selected = [(r, c) for r, c in zip(self._rows, self._contexts) if eligible(c) and r["augmentation_role"] == "student_tr"]
        require(len(selected) == len(manifest["entries"]) and selected, "Missing/extra/duplicate prepared TR entries")
        require(len(expected_clean_hashes) == len(selected), "Independent clean waveform SHA declarations required")
        self._entries = {}
        names = set()
        for index, ((row, context), entry, clean_hash) in enumerate(zip(selected, manifest["entries"], expected_clean_hashes)):
            require(entry["draw"] == asdict(context) and entry["target"] == target_contract(row) and
                    entry["clean_waveform_sha256"] == clean_hash, "Ordered draw/target/text/sample/clean SHA mismatch")
            path = storage_path(entry["cache"]["path"])
            require(path.parent == folder and path.name == f"audio_{index:03}.npy" and path.name not in names,
                    "Cache path/order/duplicate mismatch")
            names.add(path.name)
            require(entry["cache"]["bytes"] <= entry["target"]["samples"] * 4 + 512 and
                    record(path, guard) == entry["cache"], "Prepared cache SHA/size mismatch")
            event = entry["prepared_event"]
            require(event["state"] == "prepared_not_committed" and event["batch_id"] == batch_id and
                    event["batch_row"] == index and event["draw"] == asdict(context) and
                    event["augmentation"]["policy_sha256"] == binding.identity["policy_sha256"] and
                    event["augmentation"]["source_id"] == context.source_id, "Prepared receipt identity mismatch")
            self._entries[(context.epoch, context.draw_index)] = entry
        semantic = {"identity": manifest["identity"], "batch_id": batch_id, "transport_source": manifest["transport_source"],
                    "entries": [{k: v for k, v in entry.items() if k != "cache"} for entry in manifest["entries"]]}
        require(digest_bytes(canonical(semantic)) == manifest["semantic_sha256"], "Semantic transport hash mismatch")
        self.transport_binding = {"manifest": manifest_record, "closure": closure_record,
                                  "semantic_sha256": manifest["semantic_sha256"]}

    def __call__(self, row):
        import numpy as np
        with self._lock:
            require(not self._failed and self._cursor < len(self._contexts), "Prepared batch failed/exhausted/reused")
            context = self._contexts[self._cursor]
            expected = self._rows[self._cursor]
            try:
                require(row["source_id"] == context.source_id == expected["source_id"] and
                        row["language"] == context.language == expected["language"], "Out-of-order actual source row")
                require(row.get("augmentation_role") == expected["augmentation_role"], "Actual student/teacher/replay role changed")
                require(all(row.get(k) == expected.get(k) for k in
                            ("text", "duration", "encoder_frames", "target_tokens")), "Actual row metadata/targets changed")
                if eligible(context) and row["augmentation_role"] == "student_tr":
                    entry = self._entries[(context.epoch, context.draw_index)]
                    require(target_contract(row) == entry["target"], "Actual row targets/text/samples changed")
                    path = storage_path(entry["cache"]["path"])
                    require(record(path, self._guard) == entry["cache"], "Prepared cache changed before consume")
                    wave = np.load(path, allow_pickle=False)
                    require(wave.dtype == np.float32 and wave.ndim == 1 and wave.flags.c_contiguous and
                            len(wave) == entry["target"]["samples"] and np.isfinite(wave).all() and
                            waveform_hash(wave, self._guard) == entry["prepared_waveform_sha256"], "Prepared waveform SHA/geometry mismatch")
                    event = json.loads(canonical(entry["prepared_event"]))
                    event["batch_row"] = self._cursor
                else:
                    wave = self._protected_reader(row)
                    reason = "protected_split_phase_role_or_language" if self._binding.augmenter.policy.enabled else "disabled"
                    event = {"state": "prepared_not_committed", "batch_id": self._batch_id,
                        "batch_row": self._cursor, "draw": asdict(context), "augmentation": {
                            "policy_sha256": self._binding.identity["policy_sha256"], "applied": False,
                            "source_id": context.source_id, "label_policy": "unchanged_single_target", "reason": reason}}
                encoded = canonical(event)
                self._record(json.loads(encoded));self._events.append(encoded);self._cursor += 1
                return wave
            except BaseException:
                self._failed = True
                raise

    def prepared_payload(self):
        require(not self._failed and self._cursor == len(self._contexts), "Incomplete/failed transport cannot prepare commit")
        events = [json.loads(raw) for raw in self._events]
        return {"kind": "FastMSS_prepared_batch_v1", "state": "prepared_not_committed", "batch_id": self._batch_id,
                "rows": events, "payload_sha256": digest_bytes(canonical(events)), "committed_frontier_advanced": False,
                "commit_owner": "native caller only after actual positive optimizer commit"}


def cleanup_owned_cache(*, owner_record, manifest_record, closure_record, boundary, verify_boundary, guard):
    """Delete only this exact owned cache after a caller-verified durable boundary."""
    owner = verify_closure(closure_record, owner_record, manifest_record, guard, success=False)
    require(boundary.get("kind") in ("durable_native_boundary", "discard_uncommitted_preparation") and
            boundary.get("batch_id") == owner["batch_id"] and boundary.get("owner_sha256") == owner_record["sha256"] and
            verify_boundary(dict(boundary)) is True, "No verified caller-owned durable/discard boundary; keep cache")
    folder = scoped_job(Path(owner_record["path"]).parent)
    names = owner["owned_names"]
    require(names[:2] == ["owner.json", "manifest.json"] and len(names) <= 16 and len(set(names)) == len(names) and
            all(name in ("owner.json", "manifest.json") or re.fullmatch(r"audio_\d{3}\.npy", name) for name in names),
            "Invalid exact owned-file inventory")
    actual = list(folder.iterdir())
    require(all(p.is_file() and storage_path(p).parent == folder and p.name in names for p in actual),
            "Unknown file/directory in cache; delete nothing")
    if manifest_record is not None:
        manifest = read_bound(manifest_record, guard)
        require(manifest["owner"] == owner_record, "Cache owner changed")
        for entry in manifest["entries"]:
            path = storage_path(entry["cache"]["path"])
            require(path.parent == folder and path.name in names and record(path, guard) == entry["cache"], "Cache changed; delete nothing")
    elif boundary["kind"] != "discard_uncommitted_preparation":
        raise ValueError("Incomplete producer may only be discarded without committed draw")
    removed = []
    # Exact paths were resolved and inventoried above. No glob/recursive delete,
    # original assets, manifests, checkpoints or another job are ever traversed.
    with capacity_lock():
        require(all(p.is_file() and storage_path(p).parent == folder and p.name in names for p in folder.iterdir()),
                "Cache inventory changed; delete nothing")
        for name in names[2:] + names[1:2] + names[:1]:
            path = folder / name
            if path.exists():
                require(path.resolve().parent == folder, "Owned cleanup path changed")
                path.unlink();removed.append(str(path))
        folder.rmdir()
    return {"removed_exact_owned_files": removed, "job_id": owner["job_id"], "batch_id": owner["batch_id"],
            "owner_sha256": owner_record["sha256"], "manifest": manifest_record,
            "verified_boundary_sha256": digest_bytes(canonical(boundary)), "sampler_journal_frontier_advanced": False}
