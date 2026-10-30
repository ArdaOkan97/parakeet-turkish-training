"""NEW inactive phase-aware owned feature sink; actual bridge4 only.

Imports and validation-only finalization are NumPy-free. Native row loads and CPU producer saves retain their original lazy NumPy path.
No sampler, Adam, Torch, frontend implementation or PCM cache producer lives
here. Each row is decoded/transformed/featurized serially through callbacks;
there is never an admission requirement to retain14 TR waveforms together.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import threading
import time
from types import SimpleNamespace

import fastmss_BPE9223_plan_bridge_v7 as bridge
import fastmss_stdlib_feature_validation_v1 as stdlib_arrays
from fastmss_clean_phase_reader_v1 import CleanPhaseOrderedAudioReader, KIND as CLEAN_PROTECTION
from fastmss_ordered_draw_binding_v1 import AugmentationDrawContext

ROOT, PROJECT = bridge.ROOT, bridge.PROJECT
CACHE = ROOT / "temp/BPE9223_original_CPU_feature_engine_v1"
CAPACITY, METADATA = bridge.CAPACITY, bridge.METADATA
KIND = "FastMSS_owned_phase_aware_feature_window_v1"
MANIFEST = "FastMSS_closed_phase_aware_feature_window_v1"
BRIDGE_SHA = "ee85f8af3ed68a6aa3ad2930b42b9fb59ddb1aaa15bf3c509d63a4c2faaa833a"
FRONTEND_ARGS = {"sampling_rate": 16000, "return_tensors": "pt", "padding": "longest", "return_attention_mask": True, "subsampling_factor": 8}
canonical, digest, same, require = bridge.canonical, bridge.digest, bridge.same, bridge.require
scoped, read_bound, integer = bridge.scoped, bridge.read_bound, bridge.integer


def serial():
    require(threading.current_thread() is threading.main_thread(), "Serial feature protocol only")


def record(path, guard=lambda: None):
    path = scoped(path, source=True)
    value = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            guard(); value.update(block)
    require((before.st_size, before.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns), "Bound file changed during hash")
    return {"path": str(path), "bytes": before.st_size, "sha256": value.hexdigest()}


def write_new(path, value):
    raw = canonical(value)
    require(len(raw) <= 2 * METADATA, "Bounded metadata required; never truncate")
    with scoped(path).open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


@contextmanager
def _lock(path):
    import msvcrt
    path = scoped(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.seek(0, 2) == 0:
            stream.write(b"0"); stream.flush()
        until = time.monotonic() + 2
        while True:
            try:
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1); break
            except OSError:
                require(time.monotonic() < until, "Owned feature capacity lock busy")
                time.sleep(.02)
        try:
            yield
        finally:
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def capacity_lock():
    return _lock(CACHE.with_suffix(".capacity.lock"))


def _load_bridge(pin, guard):
    require(record(bridge.__file__, guard)["sha256"] == BRIDGE_SHA, "Exact frozen bridge4 source required")
    value = read_bound(pin, guard)
    bridge.validate_bridge_scope(value)
    require(value["bridge_source_sha256"] == BRIDGE_SHA and value["maximum_feature_window_bytes"] == CAPACITY and value["raw_PCM_capacity_bytes"] == bridge.PCM_CAPACITY and value["raw_PCM_maximum_jobs"] == 2, "Exact bounded source-typed bridge7 protocol required; no diagnostic alias")
    return value


def _frontend(value, guard):
    require(value["Torch"] == "2.7.1+cpu" and type(value["torch_threads"]) is int and value["torch_threads"] == 4 and same(value["arguments"], FRONTEND_ARGS), "Exact original CPU frontend defaults required")
    for key in ("source", "processor_config"):
        bridge.verify_file(value[key], guard, maximum=256 * 1024)
    proof = read_bound(value["CPU_GPU_byte_equivalence_proof"], guard)
    require(proof["passed"] is True, "Actual original frontend byte-equivalence evidence required")
    scoped(value["base_path"], source=True)


def create_window(bridge_binding, *, window_id, frontend, guard, verify_prior_boundary=None):
    serial()
    value = _load_bridge(bridge_binding, guard)
    rebuilt = bridge.bridge_plan(value["coordinator_plan"], DATA_evidence=value["DATA_evidence"], session_binding=value["session"], window_index=value["window_index"], prior_window_boundary=value["prior_window_boundary"], verify_prior_boundary=verify_prior_boundary, guard=guard)
    require(same(rebuilt, value), "Actual DATA/plan/session/source bridge must revalidate before ownership")
    require(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", window_id) is not None, "Fresh safe window ID required")
    _frontend(frontend, guard)
    names = ["owner.json", "manifest.json"]
    for i, update in enumerate(value["updates"]):
        names += [f"active_{i:02}.json", f"update_{i:02}.json"]
        for j in range(len(update["rows"])):
            names += [f"u{i:02}_r{j:02}_features.npy", f"u{i:02}_r{j:02}_mask.npy"]
    owner = {"kind": KIND, "source": record(__file__, guard), "window_id": window_id, "bridge": bridge_binding,
        "session": value["session"], "window_index": value["window_index"], "phase": value["phase"],
        "updates": len(value["updates"]), "owned_names": names, "capacity_bytes": CAPACITY,
        "augmentation_identity": value["augmentation_identity"], "frontend": frontend,
        "raw_PCM_capacity_bytes": bridge.PCM_CAPACITY, "raw_PCM_maximum_jobs": 2,
        "raw_processing": "serial_per_row_not_full14_PCM_bank", "prepared_is_committed": False,
        "sampler_journal_frontier_advanced": False, "production_enabled": False}
    guard()
    with capacity_lock():
        CACHE.mkdir(parents=True, exist_ok=True)
        require(not list(CACHE.iterdir()), "One shared512MiB owned window; preserve any outstanding files")
        folder = CACHE / window_id; folder.mkdir(exist_ok=False)
        write_new(folder / "owner.json", owner)
    return record(folder / "owner.json", guard)


def owner_folder(pin, guard):
    value = read_bound(pin, guard)
    folder = scoped(pin["path"]).parent
    require(value["kind"] == KIND and same(value["source"], record(__file__, guard)) and folder.parent == CACHE.resolve() and folder.name == value["window_id"] and Path(pin["path"]).name == "owner.json", "Exact NEW actual owned window/source required; synthetic owner forbidden")
    require(value["capacity_bytes"] == CAPACITY and value["production_enabled"] is False and value["raw_PCM_capacity_bytes"] == bridge.PCM_CAPACITY and value["raw_PCM_maximum_jobs"] == 2, "Fixed feature/raw bounds required")
    plan = _load_bridge(value["bridge"], guard)
    require(value["phase"] == plan["phase"] and same(value["session"], plan["session"]) and value["window_index"] == plan["window_index"] and value["updates"] == len(plan["updates"]) and same(value["augmentation_identity"], plan["augmentation_identity"]), "Window session/phase/policy differs from exact bridge")
    return value, folder, plan


class _Store:
    """Owned file engine. Private synthetic tests use a different E namespace."""
    def __init__(self, folder, owner, guard, lock):
        self.folder, self.owner, self.guard, self.lock = scoped(folder), owner, guard, lock
        names = self.owner["owned_names"]
        require(type(names) is list and len(names) == len(set(names)) <= 704 and all(type(v) is str and Path(v).name == v and v not in (".", "..") for v in names), "Exact safe finite owned inventory required")
        self._names_object, self._names_tuple = names, tuple(names)
        self._names_set = frozenset(names)
        self._directory_identity = self._directory_stat()

    def _directory_stat(self):
        value = os.stat(self.folder, follow_symlinks=False)
        require(stat.S_ISDIR(value.st_mode) and not value.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT,
                "Owned inventory root must remain an ordinary no-reparse directory")
        require(type(value.st_dev) is int and type(value.st_ino) is int and value.st_ino > 0,
                "Available stable owned-directory device/inode identity required")
        return value.st_dev, value.st_ino

    def inventory(self):
        names = self.owner["owned_names"]
        require(names is self._names_object and type(names) is list and tuple(names) == self._names_tuple,
                "Owned inventory name list changed; preserve all")
        require(self._directory_stat() == self._directory_identity, "Owned inventory directory replaced; preserve all")
        count, total = 0, 0
        # Keep every original preallocation/postwrite full scan. DirEntry uses
        # no-follow metadata and constant-time exact-name membership, avoiding
        # per-file Windows path resolution and repeated linear list searches.
        with os.scandir(self.folder) as entries:
            for entry in entries:
                count += 1
                require(count <= len(self._names_tuple) and entry.name in self._names_set,
                        "Unknown file/directory; preserve all")
                value = entry.stat(follow_symlinks=False)
                require(stat.S_ISREG(value.st_mode) and not value.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT,
                        "Unknown directory/reparse/nonregular member; preserve all")
                total += value.st_size
                require(total <= CAPACITY, "Actual whole-window512MiB cap exceeded")
        require(self._directory_stat() == self._directory_identity, "Owned inventory directory replaced during scan; preserve all")
        return total

    def write_metadata(self, name, value):
        require(name in self.owner["owned_names"], "Metadata name not owned")
        with self.lock():
            require(self.inventory() + len(canonical(value)) <= CAPACITY, "Metadata cannot exceed whole feature capacity")
            write_new(self.folder / name, value)
        return record(self.folder / name, self.guard)

    def save(self, name, array):
        import numpy as np
        require(name in self.owner["owned_names"] and re.fullmatch(r"u\d{2}_r\d{2}_(features|mask)\.npy", name), "Exact owned array name required")
        require(isinstance(array, np.ndarray) and array.flags.c_contiguous and array.dtype in (np.float32, np.bool_) and not array.dtype.hasobject, "Contiguous numeric feature/mask only; no pickle")
        with self.lock():
            require(self.inventory() + array.nbytes + 1024 <= CAPACITY, "Actual aggregate feature capacity admission required; no row exclusion")
            with (self.folder / name).open("xb") as stream:
                np.save(stream, array, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
            require(self.inventory() <= CAPACITY, "Actual NPY file capacity differs")
        return {"file": record(self.folder / name, self.guard), "shape": list(array.shape), "dtype": str(array.dtype), "array_sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest()}

    def validate_arrays(self, entry):
        return stdlib_arrays.validate_feature_arrays(entry, folder=self.folder,
            owned_names=self.owner["owned_names"], guard=self.guard)

    def arrays(self, entry):
        import numpy as np
        result = {}
        for key, dtype in (("input_features", "float32"), ("attention_mask", "bool")):
            item = entry[key]; path = scoped(item["file"]["path"])
            require(path.parent == self.folder and path.name in self.owner["owned_names"] and same(record(path, self.guard), item["file"]), "Feature file ownership/hash differs")
            require(type(item["shape"]) is list and len(item["shape"]) in (2, 3) and all(type(v) is int and v > 0 for v in item["shape"]), "Positive declared shape required before NPY load")
            with path.open("rb") as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), "Owned np.save format required")
                header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, actual_dtype = header(stream, max_header_size=1024)
                require(not fortran and not actual_dtype.hasobject and str(actual_dtype) == dtype and list(shape) == item["shape"] and stream.tell() + math.prod(shape) * actual_dtype.itemsize == path.stat().st_size <= CAPACITY, "Actual NPY header/physical bytes differ; reject before allocating")
            value = np.load(path, allow_pickle=False)
            require(str(value.dtype) == dtype == item["dtype"] and value.flags.c_contiguous and list(value.shape) == item["shape"] and hashlib.sha256(memoryview(value).cast("B")).hexdigest() == item["array_sha256"], "Exact NPY shape/dtype/content required")
            result[key] = value
        features, mask = result["input_features"], result["attention_mask"]
        require(features.ndim == 3 and features.shape[0] == 1 and features.shape[2] == 128 and mask.shape == features.shape[:2] and mask.any() and np.isfinite(features).all() and np.array_equal(mask, np.arange(mask.shape[1])[None, :] < mask.sum(1)[:, None]) and not features[~mask].any() and (int(mask.sum()) + 7) // 8 == entry["encoder_frames"], "Exact original finite/right-mask/zero-pad/full encoder geometry required")
        return result


def process_identity():
    import psutil
    process = psutil.Process()
    return {"pid": process.pid, "ppid": process.ppid(), "create_time": process.create_time()}


def producer_closed(value):
    import psutil
    try:
        return abs(psutil.Process(value["pid"]).create_time() - value["create_time"]) > .001
    except psutil.NoSuchProcess:
        return True


def verify_feature_closure(pin, owner_binding, *, success, guard):
    proof = read_bound(pin, guard)
    require(proof["kind"] == "FastMSS_phase_aware_feature_child_actual_closure_v1" and same(proof["owner"], owner_binding) and proof["parent_observed_wait"] is True and producer_closed(proof["producer"]) and same(proof["observer_source"], record(__file__, guard)), "Actual closed owned phase-aware child required")
    require(type(proof["actual_exit_code"]) is int and (not success or proof["actual_exit_code"] == 0 and proof["update_record"] is not None), "Successful actual frontend exit required")
    active = read_bound(proof["active_record"], guard)
    require(same(active["owner"], owner_binding) and same(active["producer"], proof["producer"]) and active["update_index"] == proof["update_index"], "Active producer/closure/order differs")
    monitor = read_bound(proof["guard_closure"], guard)
    require(monitor["pid"] == proof["producer"]["pid"] and monitor["actual_exit_code"] == proof["actual_exit_code"] and monitor["monitor_joined"] is True and (not success or monitor["monitor_clean"] is True and monitor["observed"]["monitor_error"] is None and monitor["observed"]["samples"] >= 3), "Actual frontend guard closure differs")
    if proof["update_record"] is not None:
        item = read_bound(proof["update_record"], guard)
        require(same(item["active_record"], proof["active_record"]) and same(item["producer"], proof["producer"]) and item["update_index"] == proof["update_index"], "Published update belongs to another actual child")
    return proof


def _wave_hash(value, guard):
    raw, sha = memoryview(value).cast("B"), hashlib.sha256()
    for start in range(0, len(raw), 1024**2):
        guard(); sha.update(raw[start:start + 1024**2])
    return sha.hexdigest()


class _RowProtocol:
    """No admission: internal serial numerical/file engine used by guarded tests."""
    def __init__(self, update, references, augmentation_identity, store, *, clean_reader, weak_transform, transform_identity, frontend, guard):
        require(all(callable(v) for v in (clean_reader, frontend, guard)), "Explicit serial CPU callbacks required")
        self.update, self.references, self.identity = deepcopy(update), deepcopy(references), deepcopy(augmentation_identity)
        self.store, self.clean_reader, self.weak_transform, self.frontend, self.guard = store, clean_reader, weak_transform, frontend, guard
        require(same(transform_identity, augmentation_identity), "Transform callback must use exact frozen policy/registry/source identity")
        self.rows = []
        for raw, context in zip(update["coordinator_update"]["rows"], update["coordinator_update"]["contexts"]):
            row = deepcopy(raw); row["audio_draw_context"] = deepcopy(context)
            row["augmentation_role"] = "student_tr" if row["language"] == "tr" else "replay"
            self.rows.append(row)
        self.cursor, self.failed, self.entries, self.events = 0, False, [], []
        self.clean_wrapper = None
        if update["phase"] == bridge.CLEANER:
            binding = SimpleNamespace(identity=deepcopy(augmentation_identity), augmenter=SimpleNamespace(policy=SimpleNamespace(phases=(bridge.WEAK,))))
            self.clean_wrapper = CleanPhaseOrderedAudioReader(binding, self.rows,
                [AugmentationDrawContext(**v) for v in update["coordinator_update"]["contexts"]],
                batch_id=update["batch_id"], clean_reader=clean_reader,
                expected_target_ids=[v["full_target_ids"] for v in references], expected_audio=[v["expected_audio"] for v in references], guard=guard, record_prepared=lambda event: None)

    def prepare_row(self, index):
        import numpy as np
        try:
            serial(); integer(index, 0, len(self.rows) - 1)
            require(not self.failed and index == self.cursor, "Row failed/reused/out of order")
            row, expected, reference = self.rows[index], self.update["rows"][index], self.references[index]
            row_sha = digest(row)
            require(same(reference["context"], expected["context"]) and same(reference["full_target_ids"], expected["full_target_ids"]), "Independent ordered full-target/context proof differs")
            self.guard()
            wave = self.clean_wrapper(row) if self.clean_wrapper is not None else self.clean_reader(row)
            require(isinstance(wave, np.ndarray) and wave.dtype == np.float32 and wave.ndim == 1 and wave.flags.c_contiguous and np.isfinite(wave).all() and len(wave) == reference["expected_audio"]["samples"], "Whole original finite mono float32 PCM required")
            original_sha = _wave_hash(wave, self.guard)
            require(original_sha == reference["expected_audio"]["waveform_sha256"] and reference["expected_audio"]["sample_rate"] == 16000, "Whole original independent clean PCM hash differs")
            if self.update["phase"] == bridge.WEAK and expected["role"] == "student_tr":
                require(callable(self.weak_transform), "Frozen weak transform callback required only for TR")
                prepared, augmentation = self.weak_transform(wave, row, deepcopy(expected["context"]))
            else:
                prepared = wave
                augmentation = {"policy_sha256": self.identity["policy_sha256"], "source_id": row["source_id"], "applied": False, "label_policy": "unchanged_single_target", "reason": "protected_split_phase_role_or_language"}
            require(isinstance(prepared, np.ndarray) and prepared.dtype == np.float32 and prepared.ndim == 1 and prepared.flags.c_contiguous and len(prepared) == len(wave) and np.isfinite(prepared).all() and _wave_hash(wave, self.guard) == original_sha, "Transform mutated clean source or sample geometry")
            require(augmentation["policy_sha256"] == self.identity["policy_sha256"] and augmentation["source_id"] == row["source_id"] and type(augmentation["applied"]) is bool and augmentation["label_policy"] == "unchanged_single_target", "Actual frozen augmentation receipt/source/label policy differs")
            prepared_sha = _wave_hash(prepared, self.guard)
            require(augmentation["applied"] or prepared_sha == original_sha, "Unapplied transform cannot silently change audio")
            if augmentation["applied"]:
                require(type(augmentation["operations"]) is list and augmentation["operations"] and all(v["kind"] in ("rir", "noise") for v in augmentation["operations"]) and augmentation["speed_or_speaker_mixing"] is False and type(augmentation["samples_unchanged"]) is int and augmentation["samples_unchanged"] == len(wave) and augmentation["sample_rate"] == 16000 and augmentation["epoch"] == expected["context"]["epoch"] and augmentation["draw_index"] == expected["context"]["draw_index"], "Actual frozen single-target/no-speed operation receipt required")
            protected = self.update["phase"] == bridge.CLEANER or expected["role"] == "replay" or len(wave) > 120 * 16000
            require(not protected or augmentation["applied"] is False and prepared_sha == original_sha, "Protected cleaner/replay/long clip must remain byte exact and retained")
            arrays = self.frontend(prepared)
            require(type(arrays) is dict and set(arrays) == {"input_features", "attention_mask"} and _wave_hash(prepared, self.guard) == prepared_sha and _wave_hash(wave, self.guard) == original_sha and digest(row) == row_sha, "Frontend/transform changed source metadata/targets or PCM bytes")
            entry = {"source_id": row["source_id"], "language": row["language"], "context": deepcopy(expected["context"]), "role": expected["role"],
                "original_row_sha256": expected["original_row_sha256"], "raw_text_sha256": expected["raw_text_sha256"],
                "cleaner_original_ordinal": row.get("cleaner_original_ordinal"), "full_target_ids": deepcopy(expected["full_target_ids"]),
                "target_ids_u32_sha256": expected["target_ids_u32_sha256"], "target_tokens": len(expected["full_target_ids"]),
                "encoder_frames": row["encoder_frames"], "sample_rate": 16000, "samples": len(wave),
                "clean_PCM_sha256": original_sha, "prepared_PCM_sha256": prepared_sha}
            for key, suffix in (("input_features", "features"), ("attention_mask", "mask")):
                entry[key] = self.store.save(f"u{self.update['update_index']:02}_r{index:02}_{suffix}.npy", arrays[key])
            self.store.arrays(entry)
            event = {"state": "prepared_not_committed", "batch_id": self.update["batch_id"], "batch_row": index, "draw": deepcopy(expected["context"]), "augmentation": augmentation}
            self.entries.append(json.loads(canonical(entry))); self.events.append(json.loads(canonical(event)))
            self.cursor += 1
            return deepcopy(entry)
        except BaseException:
            self.failed = True; raise

    def prepared(self):
        require(not self.failed and self.cursor == len(self.rows), "Every ordered row must finish before payload")
        payload = self.clean_wrapper.prepared_payload() if self.clean_wrapper is not None else {"kind": "FastMSS_prepared_batch_v1", "state": "prepared_not_committed", "batch_id": self.update["batch_id"], "rows": deepcopy(self.events), "payload_sha256": digest(self.events), "committed_frontier_advanced": False, "commit_owner": "native caller only after actual positive optimizer commit"}
        require([v["draw"] for v in payload["rows"]] == self.update["coordinator_update"]["contexts"] and digest(payload["rows"]) == payload["payload_sha256"] and payload["committed_frontier_advanced"] is False, "Actual complete copied prepared payload differs")
        if self.update["phase"] == bridge.CLEANER:
            require(payload["protection_kind"] == CLEAN_PROTECTION and all(v["augmentation"]["applied"] is False for v in payload["rows"]), "Cleaner explicit direct protection required")
        return deepcopy(self.entries), json.loads(canonical(payload))


class PhaseAwareUpdateSink:
    def __init__(self, owner_binding, update_index, input_binding, *, prior_closures, clean_reader, weak_transform, transform_identity, frontend, guard):
        serial(); self.owner_binding, self.guard = owner_binding, guard
        self.owner, self.folder, plan = owner_folder(owner_binding, guard)
        integer(update_index, 0, self.owner["updates"] - 1)
        require(len(prior_closures) == update_index, "One next ordered frontend child at a time")
        for i, pin in enumerate(prior_closures):
            require(verify_feature_closure(pin, owner_binding, success=True, guard=guard)["update_index"] == i, "Prior child order differs")
        evidence = read_bound(input_binding, guard)
        expected = bridge.bind_verified_update_inputs(self.owner["bridge"], update_index, independent_PCM_proof=evidence["independent_clean_PCM"], actual_closed0=evidence["independent_actual0"], guard=guard)
        require(same(expected, evidence), "Actual independent clean PCM source/full-target proof required before frontend child")
        proof = read_bound(evidence["independent_clean_PCM"], guard)
        self.index, self.input_binding = update_index, input_binding
        self.update = plan["updates"][update_index]
        self.store = _Store(self.folder, self.owner, guard, capacity_lock)
        self.active = {"kind": "FastMSS_phase_aware_feature_child_active_v1", "owner": owner_binding, "update_index": update_index, "producer": process_identity(), "prepared_is_committed": False}
        self.active_record = self.store.write_metadata(f"active_{update_index:02}.json", self.active)
        self.protocol = _RowProtocol(self.update, proof["references"], self.owner["augmentation_identity"], self.store, clean_reader=clean_reader, weak_transform=weak_transform, transform_identity=transform_identity, frontend=frontend, guard=guard)

    def prepare_row(self, row_index):
        return self.protocol.prepare_row(row_index)

    def finish(self):
        require(same(self.active["producer"], process_identity()), "Only actual owned child may finish its update")
        rows, payload = self.protocol.prepared()
        original = self.update["coordinator_update"]
        item = {"kind": "FastMSS_provisional_phase_aware_feature_update_v1", "owner": self.owner_binding, "active_record": self.active_record,
            "producer": self.active["producer"], "update_index": self.index, "global_update": original["global_update"], "phase": original["phase"],
            "batch_id": original["batch_id"], "native_transaction_sha256": original["native_transaction_sha256"],
            "before_sampler_sha256": original["before_sampler_sha256"], "after_sampler_sha256": original["after_sampler_sha256"],
            "coordinator_update_sha256": self.update["coordinator_update_sha256"], "rows": rows, "prepared_payload": payload,
            "input_evidence": self.input_binding, "frontend": self.owner["frontend"], "augmentation_identity": self.owner["augmentation_identity"],
            "prepared_is_committed": False, "sampler_journal_frontier_advanced": False, "actual_CPU_child_exit_observed": False,
            "raw_processing": "serial_per_row_no_full_update_PCM_bank"}
        return self.store.write_metadata(f"update_{self.index:02}.json", item)


def observe_feature_child(child, *, owner_binding, update_index, guard_closure, proof_path, guard, timeout=10):
    serial(); require(isinstance(child, subprocess.Popen), "Actual owned Popen handle required")
    code = child.wait(timeout=timeout)
    owner, folder, _ = owner_folder(owner_binding, guard)
    active_record = record(folder / f"active_{update_index:02}.json", guard)
    active = read_bound(active_record, guard)
    producer = active["producer"]
    require(producer["pid"] == child.pid or producer["ppid"] == child.pid, "Producer is not owned child/guard shim")
    require(child.poll() is not None and producer_closed(producer), "Frontend child is still active")
    monitor = read_bound(guard_closure, guard)
    require(monitor["pid"] == producer["pid"] and monitor["actual_exit_code"] == code, "Actual guard/native exit differs")
    update_path = folder / f"update_{update_index:02}.json"
    proof = {"kind": "FastMSS_phase_aware_feature_child_actual_closure_v1", "owner": owner_binding, "producer": producer,
        "owned_handle_pid": child.pid, "actual_exit_code": code, "parent_observed_wait": True, "update_index": update_index,
        "active_record": active_record, "update_record": record(update_path, guard) if update_path.exists() else None,
        "guard_closure": guard_closure, "observer_source": record(__file__, guard)}
    path = scoped(proof_path)
    require(not path.is_relative_to(CACHE.resolve()), "Preserve actual closure outside transient cache")
    path.parent.mkdir(parents=True, exist_ok=True); write_new(path, proof)
    return record(path, guard)


def _validate_update(owner_binding, item, planned, store):
    require(item["kind"] == "FastMSS_provisional_phase_aware_feature_update_v1" and same(item["owner"], owner_binding) and item["phase"] == planned["phase"] and item["update_index"] == planned["update_index"] and item["coordinator_update_sha256"] == planned["coordinator_update_sha256"] and item["prepared_is_committed"] is False and item["sampler_journal_frontier_advanced"] is False, "Exact native-compatible closed phase-aware update required")
    original = planned["coordinator_update"]
    require(same(item["frontend"], store.owner["frontend"]) and same(item["augmentation_identity"], store.owner["augmentation_identity"]), "Actual original frontend/policy differs")
    evidence = read_bound(item["input_evidence"], store.guard)
    bound = bridge.bind_verified_update_inputs(store.owner["bridge"], planned["update_index"], independent_PCM_proof=evidence["independent_clean_PCM"], actual_closed0=evidence["independent_actual0"], guard=store.guard)
    require(same(evidence, bound), "Actual independent PCM/input evidence differs")
    references = read_bound(evidence["independent_clean_PCM"], store.guard)["references"]
    require(all(same(item[k], original[k]) for k in ("global_update", "batch_id", "native_transaction_sha256", "before_sampler_sha256", "after_sampler_sha256")) and len(item["rows"]) == len(planned["rows"]), "Closed source/global/full-row transaction differs")
    payload = item["prepared_payload"]
    require(payload["state"] == "prepared_not_committed" and payload["committed_frontier_advanced"] is False and digest(payload["rows"]) == payload["payload_sha256"] and len(payload["rows"]) == len(item["rows"]), "Full copied uncommitted payload required")
    for entry, expected, event, reference in zip(item["rows"], planned["rows"], payload["rows"], references):
        require(all(same(entry[k], expected[k]) for k in ("source_id", "language", "context", "role", "full_target_ids", "target_ids_u32_sha256", "raw_text_sha256", "original_row_sha256")) and same(event["draw"], expected["context"]), "Closed immutable full target/context/source/payload differs")
        require(event["augmentation"]["policy_sha256"] == store.owner["augmentation_identity"]["policy_sha256"], "Closed policy differs")
        require(entry["clean_PCM_sha256"] == reference["expected_audio"]["waveform_sha256"] and type(entry["samples"]) is int and entry["samples"] == reference["expected_audio"]["samples"] and type(entry["sample_rate"]) is int and entry["sample_rate"] == 16000 and bridge.sha(entry["prepared_PCM_sha256"]), "Actual clean/prepared whole PCM identity differs")
        protected = planned["phase"] == bridge.CLEANER or expected["role"] == "replay" or entry["samples"] > 120 * 16000
        require(not protected or event["augmentation"]["applied"] is False and entry["clean_PCM_sha256"] == entry["prepared_PCM_sha256"], "Closed protected audio was changed")
        store.validate_arrays(entry)
    require(planned["phase"] != bridge.CLEANER or payload["protection_kind"] == CLEAN_PROTECTION, "Cleaner native-v2 explicit protection missing")


def finalize_window(owner_binding, closures, *, guard):
    serial(); owner, folder, plan = owner_folder(owner_binding, guard)
    require(len(closures) == owner["updates"], "All frontend children must actually close")
    store, parts = _Store(folder, owner, guard, capacity_lock), []
    for index, pin in enumerate(closures):
        proof = verify_feature_closure(pin, owner_binding, success=True, guard=guard)
        require(proof["update_index"] == index, "Actual closure sequence differs")
        item = read_bound(proof["update_record"], guard)
        _validate_update(owner_binding, item, plan["updates"][index], store)
        parts.append({"update": proof["update_record"], "closure": pin})
    manifest = {"kind": MANIFEST, "owner": owner_binding, "bridge": owner["bridge"], "session": owner["session"], "phase": owner["phase"], "updates": parts,
        "capacity_bytes": CAPACITY, "prepared_is_committed": False, "sampler_journal_frontier_advanced": False, "augmentation_identity": owner["augmentation_identity"], "frontend": owner["frontend"], "validation_only_array_verifier": {"source": record(stdlib_arrays.__file__, guard), "NumPy_required": False, "maximum_payload_read_bytes": stdlib_arrays.CHUNK_BYTES, "native_take_row_numpy_behavior_changed": False}}
    with capacity_lock():
        used = store.inventory(); manifest["aggregate_files_bytes"] = used
        manifest["aggregate_file_count"] = len(list(folder.iterdir())) + 1
        for _ in range(10):
            total = used + len(canonical(manifest))
            if total == manifest["aggregate_files_bytes"]: break
            manifest["aggregate_files_bytes"] = total
        require(used + len(canonical(manifest)) == manifest["aggregate_files_bytes"] <= CAPACITY, "Final actual manifest exceeds whole-window capacity")
        write_new(folder / "manifest.json", manifest)
    return record(folder / "manifest.json", guard)


class _ReadCursor:
    """No admission: consume-once file protocol shared with synthetic controls."""
    def __init__(self, store, update_binding, *, updates, guard, start=0):
        self.store, self._update_binding, self.guard = store, update_binding, guard
        self.index, self.row_index, self.failed, self.updates = start, 0, False, updates

    def update_record(self, index):
        integer(index, 0, self.updates - 1)
        return deepcopy(self._update_binding(index))

    def take_row(self, update_index, row_index, *, context, native_transaction_sha256, before_sampler_sha256):
        try:
            serial(); require(not self.failed and type(update_index) is int and type(row_index) is int and (update_index, row_index) == (self.index, self.row_index), "Feature row failed/reused/out of order")
            item = read_bound(self.update_record(update_index), self.guard)
            require(item["native_transaction_sha256"] == native_transaction_sha256 and item["before_sampler_sha256"] == before_sampler_sha256, "Actual native transaction/source frontier differs")
            context = asdict(context) if isinstance(context, AugmentationDrawContext) else context
            require(same(item["rows"][row_index]["context"], context), "Actual immutable phase/source/draw context differs")
            result = self.store.arrays(item["rows"][row_index])
            self.row_index += 1
            if self.row_index == len(item["rows"]):
                self.index += 1; self.row_index = 0
            return result
        except BaseException:
            self.failed = True; raise

    def prepared_payload(self, index):
        require(not self.failed and type(index) is int and 0 <= index < self.index, "Only fully consumed rows may expose their uncommitted payload")
        return deepcopy(read_bound(self.update_record(index), self.guard)["prepared_payload"])


class PhaseAwareFeatureWindowReader(_ReadCursor):
    def __init__(self, owner_binding, manifest_binding, *, guard, resume_binding=None, verify_checkpoint=None):
        serial(); self.owner, self.folder, self.plan = owner_folder(owner_binding, guard)
        self.manifest = read_bound(manifest_binding, guard)
        require(self.manifest["kind"] == MANIFEST and same(self.manifest["owner"], owner_binding) and same(self.manifest["bridge"], self.owner["bridge"]) and self.manifest["phase"] == self.owner["phase"] and len(self.manifest["updates"]) == self.owner["updates"] and self.manifest["prepared_is_committed"] is False, "Actual closed full window required")
        count = 0
        if resume_binding is not None:
            result = bridge.validate_resume(self.owner["bridge"], resume_binding, verify_checkpoint=verify_checkpoint, guard=guard)
            count = result["next_update_index"]
        super().__init__(_Store(self.folder, self.owner, guard, capacity_lock), self._verified_update, updates=self.owner["updates"], guard=guard, start=count)

    def _verified_update(self, index):
        integer(index, 0, self.owner["updates"] - 1)
        part = self.manifest["updates"][index]
        proof = verify_feature_closure(part["closure"], self.manifest["owner"], success=True, guard=self.guard)
        require(same(proof["update_record"], part["update"]), "Independent actual child/update differs")
        return deepcopy(part["update"])



def _retire_files(store, *, archive, evidence_path):
    """No boundary admission: exact engine operation after public checks."""
    actual = list(store.folder.iterdir()); store.inventory()
    snapshot = {p.name: record(p, store.guard) for p in actual}
    evidence = scoped(evidence_path)
    require(not evidence.is_relative_to(store.folder), "Durable archive must survive feature retirement")
    archive = deepcopy(archive)
    archive["metadata"] = {p.name: json.loads(p.read_bytes()) for p in actual if p.suffix == ".json"}
    evidence.parent.mkdir(parents=True, exist_ok=True); write_new(evidence, archive)
    archive_binding = record(evidence, store.guard)
    removed = []
    with store.lock():
        store.inventory()
        require(snapshot == {p.name: record(p, store.guard) for p in store.folder.iterdir()}, "Owned inventory/content changed after archive; preserve all")
        require(same(record(evidence, store.guard), archive_binding), "Archive changed before any deletion")
        for path in actual:
            require(path.resolve().parent == store.folder and path.name in store.owner["owned_names"], "Cleanup escapes exact owned window")
            path.unlink(); removed.append(str(path))
        store.folder.rmdir()
    return removed, archive_binding


def cleanup_window(owner_binding, *, closures, manifest_binding, boundary, verify_boundary, evidence_path, guard):
    serial(); owner, folder, plan = owner_folder(owner_binding, guard)
    require(boundary["kind"] in ("coherent_checkpoint_boundary", "discard_uncommitted_feature_window") and boundary["owner_sha256"] == owner_binding["sha256"] and same(boundary["session"], owner["session"]) and boundary["phase"] == owner["phase"] and callable(verify_boundary), "Exact caller-owned actual coherent/discard boundary required")
    if boundary["kind"] == "coherent_checkpoint_boundary":
        require(manifest_binding is not None and type(boundary["committed_window_updates"]) is int and boundary["committed_window_updates"] == owner["updates"] and boundary["window_manifest_sha256"] == manifest_binding["sha256"], "Intermediate checkpoint retains all future arrays")
        pointer = read_bound(boundary["coherent_checkpoint_pointer"], guard)
        require(pointer["all_window_updates_committed"] is True and pointer["committed_window_updates"] == owner["updates"] and pointer["identity"]["window"]["plan_sha256"] == plan["coordinator_plan_sha256"] and pointer["saved_semantic_state_verified"]["passed"] is True and same(pointer["checkpoint"], boundary["checkpoint"]), "Final returned source/window checkpoint required")
        metadata = read_bound(boundary["checkpoint"], guard)
        require(same(metadata["payload"], pointer["payload"]), "Final checkpoint payload binding differs")
        require(("actual_native_closed" in boundary) != ("resident_native_boundary" in boundary), "Exactly one truthful actual exit or live coherent idle boundary")
        if "resident_native_boundary" in boundary:
            from fastmss_owned_process_closure_v2 import verify_resident_idle
            verify_resident_idle(read_bound(boundary["resident_native_boundary"], guard), pointer=pointer, proof_binding=boundary["resident_native_boundary"], manifest=manifest_binding, phase=owner["phase"], guard=guard)
        else:
            bridge.observed_native0(read_bound(boundary["actual_native_closed"], guard), guard)
    else:
        require(type(boundary["native_work_started"]) is bool, "Truthful launched/never-launched discard scope required")
        if boundary["native_work_started"]:
            native = read_bound(boundary["actual_native_closed"], guard)
            require(native["owner"] == "/root/memory_diagnosis" and type(native["actual_native_exit_code"]) is int and type(native["actual_tool_exit_code"]) is int and type(native["actual_tool_chunk_id"]) is str and native["actual_tool_chunk_id"], "Actual failed-and-closed native proof required before discard")
    require(verify_boundary(deepcopy(boundary)) is True, "Native owner must verify actual all-state/payload/prefix or truthful discard; preserve on mismatch")
    by_index = {}
    for pin in closures:
        proof = verify_feature_closure(pin, owner_binding, success=False, guard=guard)
        require(proof["update_index"] not in by_index, "Duplicate frontend child closure")
        by_index[proof["update_index"]] = proof
    store = _Store(folder, owner, guard, capacity_lock)
    actual = list(folder.iterdir()); store.inventory()
    for index in range(owner["updates"]):
        require(not (folder / f"active_{index:02}.json").exists() or index in by_index, "Any outstanding/failed child must actually close before cleanup")
    if manifest_binding is not None:
        manifest = read_bound(manifest_binding, guard)
        require(same(manifest["owner"], owner_binding), "Cleanup manifest owner differs")
        for part in manifest["updates"]:
            for entry in read_bound(part["update"], guard)["rows"]:
                store.validate_arrays(entry)
    else:
        require(boundary["kind"] == "discard_uncommitted_feature_window", "Incomplete window only supports truthful discard")
    evidence = scoped(evidence_path)
    require(not evidence.is_relative_to(CACHE.resolve()), "Metadata archive outside transient window mandatory")
    archive = {"kind": "FastMSS_phase_aware_feature_window_durable_metadata_v1", "owner_binding": owner_binding, "manifest_binding": manifest_binding, "boundary": boundary, "closures": closures}
    removed, archived = _retire_files(store, archive=archive, evidence_path=evidence)
    return {"kind": "FastMSS_phase_aware_feature_window_owned_retirement_v1", "passed": True, "owner": owner_binding, "manifest": manifest_binding, "removed_exact_owned_files": removed, "preserved_metadata": archived, "boundary_sha256": digest(boundary), "feature_files_only_deleted": True, "original_audio_source_checkpoint_journal_untouched": True, "sampler_journal_frontier_advanced": False}
