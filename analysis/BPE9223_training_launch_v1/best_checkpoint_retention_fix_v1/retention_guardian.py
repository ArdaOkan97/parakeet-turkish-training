"""Preserve complete DEV checkpoints outside a running trainer's pruning tree.

The trainer is never modified, stopped, or imported. Copies are byte verified,
published atomically, and never pruned by this service. Lowest Turkish DEV and
the stricter qualified-winner policy are deliberately separate.
"""
from __future__ import annotations

import argparse
import base64
import copy
import ctypes
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

META_LIMIT = 16 * 1024 * 1024
PAYLOAD_LIMIT = 512 * 1024 * 1024
COPY_CHUNK = 2 * 1024 * 1024
FILES = ("pre_save.json", "next_draw.json", "prepared_metadata.json.gz",
         "checkpoint.json", "pointer.json", "persistence_verified.json",
         "checkpoint.tensors.bin")
BOUNDARIES = {187937, 375874, 401072}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("xb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def pin(path, raw):
    return {"path": str(path), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def checked_bytes(path, maximum=META_LIMIT):
    path = Path(path)
    if path.stat().st_size > maximum:
        raise RuntimeError("Input exceeds its declared bound: " + str(path))
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise RuntimeError("Input grew beyond its declared bound")
    return raw


class Resources:
    """Unchanged 603 admission, runtime floors, and 180s active-copy bound.

    This is a new, explicitly authorized persistent preservation service.
    Its idle lifetime is separate from the bounded copy/metadata operations.
    The existing full 512MiB Root reservation remains charged.
    """
    def __init__(self, folder):
        import psutil
        self.process = psutil.Process()
        self.process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        self.folder = Path(folder)
        self.operation_start = None

    def sample(self):
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ("totalphysical", "physical", "commitlimit", "commit",
                 "totalvirtual", "virtual", "extended")]
        value = Memory()
        value.length = ctypes.sizeof(value)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            raise OSError("GlobalMemoryStatusEx failed")
        return dict(time=time.time(), pid=self.process.pid,
                    ctime=self.process.create_time(), ppid=self.process.ppid(),
                    commit=value.commit, physical=value.physical,
                    private=self.process.memory_info().private,
                    E_free=shutil.disk_usage("E:/gpu_training_scratchpad").free)

    def admit(self):
        samples = []
        for name in ("initial", "fresh"):
            value = self.sample()
            atomic_json(self.folder / (name + "_BEFORE_predicate.json"), value)
            samples.append(value)
        for value in samples:
            self.check(value, 6028384896, 3278384896)
        return samples

    @staticmethod
    def check(value, commit, physical):
        if not (value["commit"] >= commit and value["physical"] >= physical
                and value["private"] <= 536870912
                and value["E_free"] >= 100000000000):
            raise RuntimeError("Original resource guard refused preservation")

    def begin(self):
        self.operation_start = time.monotonic()
        self.guard()

    def guard(self):
        self.check(self.sample(), 3509949440, 3009949440)
        if (self.operation_start is not None and
                time.monotonic() - self.operation_start >= 180):
            raise TimeoutError("180-second preservation operation limit")


class Retention:
    def __init__(self, run, archive, guard=lambda: None):
        self.run = Path(run).resolve()
        self.archive = Path(archive).resolve()
        self.guard = guard
        if self.archive == self.run or self.run in self.archive.parents:
            raise ValueError("Archive must be outside the trainer's run tree")
        self.archive.mkdir(parents=True, exist_ok=True)
        self.audit = self.archive / "audit"
        self.audit.mkdir(exist_ok=True)
        self.scores = self.load("evaluated_scores.json", {})
        self.preserved = self.load("preserved.json", {})

    def load(self, name, default):
        path = self.archive / name
        return json.loads(checked_bytes(path)) if path.exists() else default

    def capture(self, path, maximum=META_LIMIT):
        self.guard()
        raw = checked_bytes(path, maximum)
        digest = hashlib.sha256(raw).hexdigest()
        receipt = self.audit / (digest + ".json")
        if not receipt.exists():
            atomic_json(receipt, dict(source=str(path), bytes=len(raw),
                        sha256=digest, base64=base64.b64encode(raw).decode("ascii")))
        return raw

    def checkpoint_folder(self, update):
        if type(update) is not int or not 0 <= update <= 401072:
            raise ValueError("Unexpected checkpoint step")
        return self.run / "checkpoints" / ("update_" + str(update).zfill(9))

    def copy_file(self, source, destination, expected=None, payload=False):
        self.guard()
        source = Path(source)
        size = source.stat().st_size
        if size > (PAYLOAD_LIMIT if payload else META_LIMIT):
            raise RuntimeError("Copy exceeds declared file bound")
        digest = hashlib.sha256()
        count = 0
        with source.open("rb") as reader, destination.open("xb") as writer:
            while True:
                self.guard()
                chunk = reader.read(COPY_CHUNK)
                if not chunk:
                    break
                writer.write(chunk)
                digest.update(chunk)
                count += len(chunk)
                if count > size:
                    raise RuntimeError("Source changed during copy")
            writer.flush()
            os.fsync(writer.fileno())
        actual = dict(path=str(destination), bytes=count, sha256=digest.hexdigest())
        if count != size or (expected and any(actual[k] != expected[k]
                                             for k in ("bytes", "sha256"))):
            raise RuntimeError("Checkpoint copy checksum/size mismatch")
        return actual

    def preserve(self, update, reason="DEV checkpoint"):
        key = str(update)
        final = self.archive / ("update_" + str(update).zfill(9))
        if key in self.preserved:
            manifest = json.loads(checked_bytes(final / "manifest.json"))
            payload = Path(manifest["portable_pointer"]["payload"]["path"])
            if not payload.is_file() or payload.stat().st_size != manifest["payload"]["bytes"]:
                raise RuntimeError("Previously protected payload is missing")
            return manifest
        source = self.checkpoint_folder(update)
        raw_pointer = self.capture(source / "pointer.json", 131072)
        pointer = json.loads(raw_pointer)
        if (pointer["committed_updates"] != update or pointer["diagnostic_only"]
                or not pointer["saved_semantic_state_verified"]
                or not pointer["writer_returned_and_payload_verified"]):
            raise RuntimeError("Checkpoint is not a verified production save")
        checkpoint_path = Path(pointer["checkpoint"]["path"])
        payload_path = Path(pointer["payload"]["path"])
        if (checkpoint_path.resolve() != (source / "checkpoint.json").resolve()
                or payload_path.resolve() != (source / "checkpoint.tensors.bin").resolve()):
            raise RuntimeError("Checkpoint paths escape the exact save directory")
        raw_checkpoint = self.capture(checkpoint_path)
        checkpoint = json.loads(raw_checkpoint)
        if (pin(checkpoint_path, raw_checkpoint) != pointer["checkpoint"]
                or checkpoint["payload"] != pointer["payload"]):
            raise RuntimeError("Pointer and checkpoint metadata do not match")
        proof = json.loads(self.capture(source / "persistence_verified.json", 131072))
        if not (proof["passed"] and proof["optimizer_updates"] == update
                and proof["weights_verified"] == 207 and proof["Adam_states_verified"] == 215
                and proof["named_tensors_verified"] == 854
                and proof["payload"] == pointer["payload"]
                and proof["checkpoint"] == pointer["checkpoint"]):
            raise RuntimeError("Original full-state persistence proof does not match")
        # The namespace is new; no incomplete staging directory is ever accepted.
        stage = self.archive / (".staging_" + str(update) + "_" + uuid.uuid4().hex)
        original = stage / "original"
        original.mkdir(parents=True)
        names = os.listdir(source)
        atomic_json(self.audit / ("names_" + str(update) + "_" + uuid.uuid4().hex + ".json"),
                    dict(path=str(source), names_original_order=names))
        if set(FILES) - set(names):
            raise RuntimeError("Complete checkpoint companion files are required")
        copied = {}
        for name in FILES:
            expected = pointer["payload"] if name == "checkpoint.tensors.bin" else None
            copied[name] = self.copy_file(source / name, original / name, expected,
                                         payload=name == "checkpoint.tensors.bin")
            copied[name]["path"] = str(final / "original" / name)
        # Preserve serialized state/identity exactly. Only external storage paths
        # at the top level change; the checkpoint tree and pointer identity do not.
        portable_checkpoint = copy.deepcopy(checkpoint)
        portable_payload = dict(pointer["payload"],
                         path=str(final / "original" / "checkpoint.tensors.bin"))
        portable_checkpoint["payload"] = portable_payload
        raw_portable_checkpoint = encoded(portable_checkpoint)
        atomic_json(stage / "checkpoint.json", portable_checkpoint)
        portable_pointer = copy.deepcopy(pointer)
        portable_pointer["payload"] = portable_payload
        portable_pointer["checkpoint"] = pin(final / "checkpoint.json", raw_portable_checkpoint)
        atomic_json(stage / "pointer.json", portable_pointer)
        manifest = dict(kind="ASR18_immutable_protected_full_checkpoint_v1", update=update,
            reason=reason, original_pointer=pin(source / "pointer.json", raw_pointer),
            original_checkpoint=pointer["checkpoint"], payload=pointer["payload"],
            original_files=copied, portable_pointer=portable_pointer,
            portable_pointer_record=pin(final / "pointer.json", encoded(portable_pointer)),
            checkpoint_tree_unchanged=portable_checkpoint["tree"] == checkpoint["tree"],
            pointer_identity_unchanged=portable_pointer["identity"] == pointer["identity"],
            full_payload_copy_sha256_verified=True, GPU_restore_performed=False,
            no_training_or_science_state_changed=True, retained_without_pruning=True,
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        atomic_json(stage / "manifest.json", manifest)
        self.guard()
        os.replace(stage, final)
        self.preserved[key] = dict(update=update, manifest=str(final / "manifest.json"),
                                  pointer=str(final / "pointer.json"), reason=reason)
        atomic_json(self.archive / "preserved.json", self.preserved)
        self.publish_best()
        return manifest

    def record_score(self, update, score):
        errors = score["word_errors"]["tr"]
        words = score["reference_words"]["tr"]
        if not (type(errors) is int and errors >= 0 and type(words) is int and words > 0):
            raise ValueError("Invalid Turkish word-error counts")
        self.scores[str(update)] = dict(update=update, errors=errors, words=words,
            percent=100 * errors / words,
            all_six_passed=bool(score.get("strict_all_six_passed", False)))
        atomic_json(self.archive / "evaluated_scores.json", self.scores)
        self.publish_best()

    def publish_best(self):
        values = list(self.scores.values())
        order = lambda v: (Fraction(v["errors"], v["words"]), v["update"])
        if values:
            observed = min(values, key=order)
            atomic_json(self.archive / "best_observed.json", dict(observed,
                protected_copy_exists=str(observed["update"]) in self.preserved))
        kept = [v for v in values if str(v["update"]) in self.preserved]
        if kept:
            best = min(kept, key=order)
            atomic_json(self.archive / "best_retained.json", dict(best,
                checkpoint=self.preserved[str(best["update"])],
                selection="Lowest Turkish DEV WER, independent of all-six qualification"))


def run_service(run, archive, session, interval=15):
    session = Path(session)
    session.mkdir(parents=True, exist_ok=False)
    resources = Resources(session)
    samples = resources.admit()
    atomic_json(session / "scope_BEFORE_training_metadata_stats.json", dict(
        kind="HUMAN_AUTHORIZED_NEW_PRODUCTION_RETENTION_SERVICE", run=str(run),
        archive=str(archive), payload_operation="byte copy plus checksum only; no tensor decoding",
        maximum_metadata_bytes=META_LIMIT, maximum_payload_bytes=PAYLOAD_LIMIT,
        active_operation_seconds=180, self_private_limit=536870912,
        original_resource_floors_and_reserves_unchanged=True,
        native_training_changes=False, initial_and_fresh_samples=samples))
    resources.begin()
    retention = Retention(run, archive, resources.guard)
    run = Path(run)
    # One authoritative bootstrap. On later changes only newly appended DEV
    # records are processed; unchanged files are never rescanned.
    periodic = json.loads(retention.capture(run / "periodic_DEV.json"))
    for row in periodic["completed_evaluations"]:
        retention.record_score(row["optimizer_updates"], row["score"])
    last_eval = max([int(k) for k in retention.scores] or [0])
    latest = json.loads(retention.capture(run / "latest_checkpoint.json", 131072))
    retention.preserve(latest["committed_updates"], "Verified available fallback at repair activation")
    # Inventory physical payloads once to identify any surviving earlier best.
    checkpoint_root = run / "checkpoints"
    names = os.listdir(checkpoint_root)
    atomic_json(session / "checkpoint_names_RAW_BEFORE_selection.json", names)
    if len(names) > 4096:
        raise RuntimeError("Checkpoint inventory exceeds declared bound")
    available = []
    for name in names:
        if name.startswith("update_") and name[7:].isdigit():
            path = checkpoint_root / name / "checkpoint.tensors.bin"
            if path.is_file():
                available.append(dict(update=int(name[7:]), bytes=path.stat().st_size,
                                      path=str(path)))
    atomic_json(session / "available_payload_inventory.json", available)
    for row in available:
        if str(row["update"]) in retention.scores:
            retention.preserve(row["update"], "Surviving evaluated checkpoint at repair activation")
    retention.publish_best()
    stamps = {}
    started = time.monotonic()
    while True:
        resources.operation_start = None
        resources.guard()
        if time.monotonic() - started > 86400:
            raise TimeoutError("Preservation service maximum lifetime is 24 hours")
        for filename in ("latest_checkpoint.json", "periodic_DEV.json", "lifecycle.json"):
            path = run / filename
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stamps.get(filename) == stamp:
                continue
            resources.begin()
            value = json.loads(retention.capture(path))
            if filename == "latest_checkpoint.json":
                update = value["committed_updates"]
                if update % 4000 == 0 or update in BOUNDARIES:
                    retention.preserve(update)
            elif filename == "periodic_DEV.json":
                fresh = [row for row in value["completed_evaluations"]
                         if row["optimizer_updates"] > last_eval]
                for row in fresh:
                    update = row["optimizer_updates"]
                    retention.preserve(update)
                    retention.record_score(update, row["score"])
                    last_eval = max(last_eval, update)
            else:
                for row in value["completed_boundaries"]:
                    pointer = row["checkpoint_pointer"]
                    update = int(Path(pointer["path"]).parent.name.removeprefix("update_"))
                    if str(update) not in retention.scores:
                        if (retention.checkpoint_folder(update) / "checkpoint.tensors.bin").is_file():
                            retention.preserve(update, "Full-pass DEV")
                        retention.record_score(update, row["score"])
            stamps[filename] = stamp
        status = dict(kind="ACTIVE_PROTECTED_DEV_ARCHIVER", time=time.time(), pid=os.getpid(),
            ctime=resources.process.create_time(), preserved=sorted(map(int, retention.preserved)),
            best_observed=retention.load("best_observed.json", None),
            best_retained=retention.load("best_retained.json", None),
            latest_evaluation_seen=last_eval, error=None,
            active_copy_timeout_seconds=180, original_resource_guards_unchanged=True)
        atomic_json(session / "status.json", status)
        if 401072 in map(int, retention.preserved) and "401072" in retention.scores:
            atomic_json(session / "complete.json", status)
            return
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    root = Path("E:/gpu_training_scratchpad").resolve()
    for value in (args.run, args.archive, args.session):
        if not Path(value).resolve().is_relative_to(root):
            raise ValueError("All preservation paths must remain on designated E storage")
    try:
        run_service(args.run, args.archive, args.session)
    except BaseException as error:
        atomic_json(Path(args.session) / "failure.json", dict(error=repr(error),
                    time=time.time(), pid=os.getpid(), no_automatic_retry=True))
        raise


if __name__ == "__main__":
    main()
