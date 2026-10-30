"""Provisional CPU feature windows; no optimizer, sampler or GPU work.

PCM production remains in the frozen bounded decoder. Each frontend producer
must actually close before its features can join a finalized window. A window
contains at most 20 updates / 512 MiB, and is never a committed training state.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import time

from fastmss_train_augmentation_v1 import canonical, require
from fastmss_ordered_draw_binding_v1 import AugmentationDrawContext, file_binding

ROOT = Path(r"E:\gpu_training_scratchpad")
PROJECT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "temp/ASR18_CPU_feature_windows_v1"
CAPACITY = 512 * 1024**2
MAX_UPDATES = 20
MAX_METADATA = 8 * 1024**2
FRONTEND_ARGS = {"sampling_rate": 16000, "return_tensors": "pt", "padding": "longest",
                 "return_attention_mask": True, "subsampling_factor": 8}


def scoped(path, *, existing_input=False):
    value = Path(path).resolve()
    require(value.is_relative_to(ROOT.resolve()) or (existing_input and value.is_relative_to(PROJECT)),
            "New feature artifacts E-only; existing project inputs only")
    return value


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def record(path, guard=lambda: None):
    path = scoped(path, existing_input=True)
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            guard(); value.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": value.hexdigest()}


def read_bound(binding, guard=lambda: None):
    path = scoped(binding["path"], existing_input=True)
    require(0 < path.stat().st_size <= MAX_METADATA and record(path, guard) == binding, "Bounded exact metadata binding required")
    return json.loads(path.read_bytes())


def write_new(path, value):
    raw = canonical(value)
    require(len(raw) <= MAX_METADATA, "Bounded feature metadata exceeded")
    with scoped(path).open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


@contextmanager
def capacity_lock():
    import msvcrt
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.with_suffix(".capacity.lock").open("a+b") as stream:
        if stream.seek(0, 2) == 0:
            stream.write(b"0"); stream.flush()
        deadline = time.monotonic() + 2
        while True:
            try:
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1); break
            except OSError:
                require(time.monotonic() < deadline, "Feature capacity lock busy; no draw consumed")
                time.sleep(.02)
        try:
            yield
        finally:
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def owner_folder(binding, guard):
    owner = read_bound(binding, guard)
    folder = scoped(binding["path"]).parent
    require(folder.parent == CACHE.resolve() and folder.name == owner["window_id"] and
            owner["kind"] == "FastMSS_owned_provisional_feature_window_v1" and
            owner["source"] == file_binding(__file__) and owner["capacity_bytes"] == CAPACITY,
            "Exact owned feature window/source required")
    return owner, folder


def inventory(owner, folder):
    actual = list(folder.iterdir())
    require(len(actual) <= len(owner["owned_names"]) <= 704 and
            all(p.is_file() and p.resolve().parent == folder and p.name in owner["owned_names"] for p in actual),
            "Unknown feature file/directory; preserve all")
    total = sum(p.stat().st_size for p in actual)
    require(total <= CAPACITY, "Feature window hard512MiB bound exceeded")
    return total


def load_plan(binding, guard):
    plan = read_bound(binding, guard)
    updates = plan["updates"]
    require(plan["diagnostic_only"] is True and plan["production_enabled"] is False and
            plan["phase"] == "weak_label_continued_training" and
            plan["actual_optimizer_updates"] == 0 and plan["actual_broad_draws_consumed"] == 0 and
            plan["actual_replay_frontiers_advanced"] is False and
            0 < len(updates) <= MAX_UPDATES and plan["maximum_feature_window_bytes"] == CAPACITY,
            "Only bounded provisional weak-label draw plan; no committed training claim")
    for index, update in enumerate(updates):
        contexts = [AugmentationDrawContext(**v) for v in update["contexts"]]
        require(update["update"] == index + 1 and update["batch_id"] == f"update_{index+1}" and
                update["virtual_only"] is True and 0 < len(contexts) <= 16 and
                len(contexts) == len(update["TR_draws"]) + len(update["replay_rows"]) and
                len({(v.epoch, v.draw_index) for v in contexts}) == len(contexts) and
                all(v.phase == plan["phase"] and v.split == "train" for v in contexts),
                "Exact ordered provisional updates/contexts required")
    return plan


def create_window(plan_binding, *, window_id, augmentation_identity, frontend, guard):
    plan = load_plan(plan_binding, guard)
    require(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", window_id), "Safe unique window ID required")
    require(frontend["arguments"] == FRONTEND_ARGS and frontend["Torch"] == "2.7.1+cpu" and
            frontend["torch_threads"] == 4, "Pinned exact original frontend execution policy required")
    names = ["owner.json", "manifest.json"]
    for i, update in enumerate(plan["updates"]):
        names.extend([f"active_{i:02}.json", f"update_{i:02}.json"])
        for j in range(len(update["contexts"])):
            names.extend([f"u{i:02}_r{j:02}_features.npy", f"u{i:02}_r{j:02}_mask.npy"])
    owner = {"kind": "FastMSS_owned_provisional_feature_window_v1", "window_id": window_id,
             "source": file_binding(__file__), "plan": plan_binding, "updates": len(plan["updates"]),
             "capacity_bytes": CAPACITY, "owned_names": names, "frontend": frontend,
             "augmentation_identity": augmentation_identity, "prepared_is_committed": False,
             "sampler_journal_frontier_advanced": False}
    guard()
    with capacity_lock():
        CACHE.mkdir(parents=True, exist_ok=True)
        require(not list(CACHE.iterdir()), "One512MiB feature window at a time; preserve outstanding window")
        folder = CACHE / window_id; folder.mkdir(exist_ok=False)
        write_new(folder / "owner.json", owner)
    return record(folder / "owner.json", guard)


def process_identity():
    import psutil
    process = psutil.Process()
    return {"pid": os.getpid(), "ppid": os.getppid(), "create_time": process.create_time()}


def producer_closed(identity):
    import psutil
    try:
        return abs(psutil.Process(identity["pid"]).create_time() - identity["create_time"]) > .001
    except psutil.NoSuchProcess:
        return True


def verify_feature_closure(binding, owner_binding, *, success, guard):
    proof = read_bound(binding, guard)
    require(proof["kind"] == "FastMSS_feature_child_actual_closure_v1" and proof["owner"] == owner_binding and
            proof["parent_observed_wait"] is True and producer_closed(proof["producer"]) and
            proof["observer_source"] == file_binding(__file__) and
            (not success or (proof["actual_exit_code"] == 0 and proof["update_record"] is not None)),
            "Actual closed feature producer proof required")
    active = read_bound(proof["active_record"], guard)
    require(active["producer"] == proof["producer"] and active["owner"] == owner_binding and
            active["update_index"] == proof["update_index"], "Feature child closure identity differs")
    if proof["update_record"] is not None:
        item = read_bound(proof["update_record"], guard)
        require(item["active_record"] == proof["active_record"] and item["producer"] == proof["producer"] and
                item["update_index"] == proof["update_index"], "Feature output belongs to another child/update")
    return proof


def begin_update(owner_binding, index, prior_closures, guard):
    owner, folder = owner_folder(owner_binding, guard)
    plan = load_plan(owner["plan"], guard)
    require(type(index) is int and 0 <= index < owner["updates"] and len(prior_closures) == index,
            "Exactly next ordered provisional update required")
    for i, closure in enumerate(prior_closures):
        require(verify_feature_closure(closure, owner_binding, success=True, guard=guard)["update_index"] == i,
                "Prior feature child closure/order differs")
    active = {"kind": "FastMSS_feature_child_active_v1", "owner": owner_binding, "update_index": index,
              "producer": process_identity(), "prepared_is_committed": False}
    with capacity_lock():
        inventory(owner, folder)
        require(not (folder / f"active_{index:02}.json").exists(), "Feature update reused or producer still outstanding")
        write_new(folder / f"active_{index:02}.json", active)
    return owner, folder, plan["updates"][index], record(folder / f"active_{index:02}.json", guard)


def save_array(owner, folder, name, array, guard):
    import numpy as np
    require(name in owner["owned_names"] and not (folder / name).exists(), "Exact fresh owned feature array required")
    guard()
    with capacity_lock():
        require(inventory(owner, folder) + array.nbytes + 1024 <= CAPACITY,
                "Feature window full; no exclusion or draw consumption allowed")
        with (folder / name).open("xb") as stream:
            np.save(stream, array, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
    return {"file": record(folder / name, guard), "shape": list(array.shape), "dtype": str(array.dtype),
            "array_sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest()}


def finish_update(owner_binding, index, active_record, rows, payload, *, input_evidence, frontend, guard):
    owner, folder = owner_folder(owner_binding, guard)
    active = read_bound(active_record, guard)
    update = load_plan(owner["plan"], guard)["updates"][index]
    require(active["producer"] == process_identity() and active["owner"] == owner_binding and
            active["update_index"] == index and len(rows) == len(update["contexts"]) == len(payload["rows"]) and
            payload["state"] == "prepared_not_committed" and payload["committed_frontier_advanced"] is False,
            "Complete exact feature rows/payload required before publishing provisional update")
    require([v["context"] for v in rows] == update["contexts"] and
            [v["draw"] for v in payload["rows"]] == update["contexts"] and
            digest(payload["rows"]) == payload["payload_sha256"], "Copied prepared payload/order hash differs")
    item = {"kind": "FastMSS_provisional_feature_update_v1", "owner": owner_binding, "active_record": active_record,
            "producer": active["producer"], "update_index": index, "batch_id": update["batch_id"],
            "native_transaction_sha256": update["expected_native_transaction_sha256"],
            "before_sampler_sha256": update["expected_before_sampler_sha256"],
            "after_sampler_sha256": update["expected_after_sampler_sha256"], "rows": rows,
            "prepared_payload": payload, "input_evidence": input_evidence, "frontend": frontend,
            "augmentation_identity": owner["augmentation_identity"], "prepared_is_committed": False,
            "sampler_journal_frontier_advanced": False}
    raw_size = len(canonical(item))
    with capacity_lock():
        require(inventory(owner, folder) + raw_size <= CAPACITY, "Feature metadata would exceed hard capacity")
        write_new(folder / f"update_{index:02}.json", item)
    return record(folder / f"update_{index:02}.json", guard)


def observe_feature_child(child, *, owner_binding, index, proof_path, guard, timeout=10):
    code = child.wait(timeout=timeout)
    require(child.poll() is not None, "Feature child must actually exit")
    owner, folder = owner_folder(owner_binding, guard)
    active_record = record(folder / f"active_{index:02}.json", guard)
    active = read_bound(active_record, guard); producer = active["producer"]
    require(producer["pid"] == child.pid or producer["ppid"] == child.pid, "Feature producer is not owned child")
    require(producer_closed(producer), "Actual feature producer is still live")
    update_path = folder / f"update_{index:02}.json"
    proof = {"kind": "FastMSS_feature_child_actual_closure_v1", "actual_exit_code": code,
             "owner": owner_binding, "owned_handle_pid": child.pid, "producer": producer, "update_index": index,
             "active_record": active_record, "update_record": record(update_path, guard) if update_path.exists() else None,
             "parent_observed_wait": True, "observer_source": file_binding(__file__)}
    path = scoped(proof_path)
    require(not path.is_relative_to(CACHE.resolve()), "Actual closure proof outside transient feature cache required")
    path.parent.mkdir(parents=True, exist_ok=True); write_new(path, proof)
    return record(path, guard)


def validate_feature_entry(entry, folder, guard):
    import numpy as np
    values = {}
    for key, suffix, dtype in (("input_features", "features", "float32"), ("attention_mask", "mask", "bool")):
        item = entry[key]; path = scoped(item["file"]["path"])
        require(path.parent == folder and record(path, guard) == item["file"] and item["dtype"] == dtype,
                "Feature file SHA/path/dtype differs")
        value = np.load(path, allow_pickle=False)
        require(str(value.dtype) == dtype and list(value.shape) == item["shape"] and value.flags.c_contiguous and
                hashlib.sha256(memoryview(value).cast("B")).hexdigest() == item["array_sha256"], "Feature array bytes/geometry differ")
        values[key] = value
    features, mask = values["input_features"], values["attention_mask"]
    require(features.ndim == 3 and features.shape[0] == 1 and features.shape[2] == 128 and
            mask.shape == features.shape[:2] and mask.any() and np.isfinite(features).all() and
            np.array_equal(mask, np.arange(mask.shape[1])[None, :] < mask.sum(1)[:, None]) and
            not features[~mask].any() and int((int(mask.sum()) + 7) // 8) == entry["encoder_frames"],
            "Original finite/right-padded/zero-masked frontend geometry differs")
    return values


def finalize_window(owner_binding, closures, *, guard):
    owner, folder = owner_folder(owner_binding, guard)
    require(len(closures) == owner["updates"], "Every feature update needs actual closure")
    updates = []
    for i, closure in enumerate(closures):
        proof = verify_feature_closure(closure, owner_binding, success=True, guard=guard)
        require(proof["update_index"] == i, "Feature closure sequence differs")
        item = read_bound(proof["update_record"], guard)
        for entry in item["rows"]:
            validate_feature_entry(entry, folder, guard)
        updates.append({"update": proof["update_record"], "closure": closure})
    manifest = {"kind": "FastMSS_closed_provisional_feature_window_v1", "owner": owner_binding,
                "plan": owner["plan"], "updates": updates, "capacity_bytes": CAPACITY,
                "prepared_is_committed": False, "sampler_journal_frontier_advanced": False,
                "augmentation_identity": owner["augmentation_identity"], "frontend": owner["frontend"]}
    with capacity_lock():
        used = inventory(owner, folder)
        manifest["aggregate_files_bytes"] = used
        manifest["aggregate_file_count"] = len(list(folder.iterdir())) + 1
        for _ in range(8):
            total = used + len(canonical(manifest))
            if total == manifest["aggregate_files_bytes"]:
                break
            manifest["aggregate_files_bytes"] = total
        require(used + len(canonical(manifest)) == manifest["aggregate_files_bytes"] <= CAPACITY,
                "Final manifest exceeds feature capacity")
        write_new(folder / "manifest.json", manifest)
    return record(folder / "manifest.json", guard)


class FeatureWindowReader:
    """Serial one-row NumPy read; native consumer/caller owns commit and devices."""
    def __init__(self, owner_binding, manifest_binding, *, guard, resume=None, verify_resume=None):
        self.owner, self.folder = owner_folder(owner_binding, guard)
        self.manifest = read_bound(manifest_binding, guard)
        require(self.manifest["owner"] == owner_binding and self.manifest["kind"] == "FastMSS_closed_provisional_feature_window_v1" and
                self.manifest["prepared_is_committed"] is False and len(self.manifest["updates"]) == self.owner["updates"],
                "Exact finalized but uncommitted feature window required")
        self.plan = load_plan(self.owner["plan"], guard)
        self.guard, self.index, self.row_index, self.failed = guard, 0, 0, False
        if resume is not None:
            require(type(resume) is dict and set(resume) == {"checkpoint", "committed_window_updates"} and callable(verify_resume),
                    "Blind start/skip prohibited; actual coherent checkpoint verifier required")
            count = resume["committed_window_updates"]
            require(type(count) is int and 0 < count <= self.owner["updates"] and
                    record(resume["checkpoint"]["path"], guard) == resume["checkpoint"], "Exact coherent checkpoint binding/count required")
            previous = self.plan["updates"][count-1]
            next_before = self.plan["updates"][count]["expected_before_sampler_sha256"] if count < self.owner["updates"] else previous["expected_after_sampler_sha256"]
            expected = {"passed":True, "checkpoint":resume["checkpoint"], "feature_window_manifest":manifest_binding,
                "committed_window_updates":count, "committed_transaction_sha256":previous["expected_native_transaction_sha256"],
                "committed_after_sampler_sha256":previous["expected_after_sampler_sha256"], "next_before_sampler_sha256":next_before,
                "augmentation_identity":self.owner["augmentation_identity"], "native_checkpoint_state_verified":True,
                "actual_restored_sampler_verified":True, "committed_journal_prefix_verified":True}
            require(verify_resume(dict(resume)) == expected,
                    "Restored checkpoint/window/transaction/sampler/journal binding differs; no arbitrary skip")
            self.index = count

    def take_row(self, update_index, row_index, *, context, native_transaction_sha256, before_sampler_sha256):
        try:
            require(not self.failed and (update_index, row_index) == (self.index, self.row_index), "Feature row reused/out of order")
            part = self.manifest["updates"][update_index]
            proof = verify_feature_closure(part["closure"], self.manifest["owner"], success=True, guard=self.guard)
            require(proof["update_record"] == part["update"], "Actual feature closure/update binding differs")
            item = read_bound(part["update"], self.guard)
            require(item["native_transaction_sha256"] == native_transaction_sha256 and
                    item["before_sampler_sha256"] == before_sampler_sha256 and
                    item["rows"][row_index]["context"] == asdict(context), "Actual native transaction/frontier/context differs")
            result = validate_feature_entry(item["rows"][row_index], self.folder, self.guard)
            self.row_index += 1
            if self.row_index == len(item["rows"]):
                self.index += 1; self.row_index = 0
            return result
        except BaseException:
            self.failed = True; raise

    def prepared_payload(self, update_index):
        require(not self.failed and type(update_index) is int and 0 <= update_index < self.index,
                "Every feature row must be consumed once before returning prepared receipt")
        part = self.manifest["updates"][update_index]
        item = read_bound(part["update"], self.guard)
        return json.loads(canonical(item["prepared_payload"]))


def cleanup_window(owner_binding, *, closures, manifest_binding, boundary, verify_boundary, evidence_path, guard):
    owner, folder = owner_folder(owner_binding, guard)
    require(boundary.get("kind") in ("coherent_checkpoint_boundary", "discard_uncommitted_feature_window") and
            boundary.get("owner_sha256") == owner_binding["sha256"] and verify_boundary(dict(boundary)) is True,
            "Caller-verified coherent/commit/discard boundary required; preserve features")
    if boundary["kind"] == "coherent_checkpoint_boundary":
        require(manifest_binding is not None and boundary.get("committed_window_updates") == owner["updates"] and
                boundary.get("window_manifest_sha256") == manifest_binding["sha256"] and
                record(boundary["checkpoint"]["path"], guard) == boundary["checkpoint"],
                "Keep all future features through intermediate checkpoint/commit; only final coherent window checkpoint may evict")
    by_index = {}
    for closure in closures:
        proof = verify_feature_closure(closure, owner_binding, success=False, guard=guard)
        require(proof["update_index"] not in by_index, "Duplicate feature closure")
        by_index[proof["update_index"]] = proof
    actual = list(folder.iterdir()); inventory(owner, folder)
    for i in range(owner["updates"]):
        if (folder / f"active_{i:02}.json").exists():
            require(i in by_index, "Missing actual closure for an active/failed feature child; delete nothing")
    if manifest_binding is not None:
        manifest = read_bound(manifest_binding, guard)
        require(manifest["owner"] == owner_binding, "Feature manifest owner differs")
        for part in manifest["updates"]:
            item = read_bound(part["update"], guard)
            for entry in item["rows"]:
                for key in ("input_features", "attention_mask"):
                    require(record(entry[key]["file"]["path"], guard) == entry[key]["file"], "Feature changed; delete nothing")
    else:
        require(boundary["kind"] == "discard_uncommitted_feature_window", "Partial window may only be discarded")
    evidence = scoped(evidence_path)
    require(not evidence.is_relative_to(CACHE.resolve()), "Preserve small feature evidence outside transient cache")
    evidence.parent.mkdir(parents=True, exist_ok=True)
    archive = {"kind": "FastMSS_feature_window_durable_metadata_v1", "owner_binding": owner_binding,
               "manifest_binding": manifest_binding, "boundary": boundary, "closures": closures,
               "metadata": {p.name: json.loads(p.read_bytes()) for p in actual if p.suffix == ".json"}}
    write_new(evidence, archive)
    removed = []
    with capacity_lock():
        inventory(owner, folder)
        require({p.name for p in actual} == {p.name for p in folder.iterdir()}, "Owned inventory changed; delete nothing")
        for path in actual:
            require(path.resolve().parent == folder and path.name in owner["owned_names"], "Cleanup escaped exact owned feature files")
            path.unlink(); removed.append(str(path))
        folder.rmdir()
    return {"removed_exact_owned_files": removed, "owner": owner_binding, "manifest": manifest_binding,
            "preserved_metadata": record(evidence, guard),
            "boundary_sha256": digest(boundary), "sampler_journal_frontier_advanced": False}
