"""Inactive bounded BPE target sidecar; no ML imports or audio reads.

Sampling always uses the immutable source catalog's original T/U metadata.
Only the at-most-eight TR dictionaries passed to the model are copied with
candidate labels. Replay dictionaries and cached original8193 labels pass through.
Every new sidecar, journal, temporary file and receipt is confined to E: storage.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import struct

STORAGE_ROOT = Path(r"E:\gpu_training_scratchpad")
OLD_VOCAB, NEW_VOCAB, BLANK, DURATIONS = 8199, 8711, 8192, 5
KIND = "TSC_BPE512_same_ID_sampling_target_sidecar_v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def scoped(path):
    root = STORAGE_ROOT.resolve(strict=True)
    path = Path(path).resolve()
    require(path != root and path.is_relative_to(root), "New BPE storage must be inside E:\\gpu_training_scratchpad")
    require(os.environ.get("GPU_TRAINING_STORAGE_ROOT") is not None
            and Path(os.environ["GPU_TRAINING_STORAGE_ROOT"]).resolve() == root,
            "Apply the authorized process-local E: storage environment before preparation")
    return path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}


def labels_sha(labels):
    # This matches the original TRAIN metadata producer's exact label encoding.
    return hashlib.sha256(json.dumps(list(labels), sort_keys=True).encode()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path, value):
    path = scoped(path)
    temp = path.with_name(path.name + ".pending")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def _ids(tokenizer, text):
    value = tokenizer.encode(text, add_special_tokens=True)
    return list(value.ids if hasattr(value, "ids") else value)


def prepare_label_catalog(rows, output_directory, baseline_tokenizer, candidate_tokenizer, *,
                          source_catalog_record, baseline_tokenizer_record,
                          candidate_tokenizer_record, max_full_joint_bytes=1700000000):
    """Stream one admitted catalog; a failure admits no partial sidecar or rows."""
    directory = scoped(output_directory)
    require(not directory.exists(), "Fresh E: label-sidecar directory required")
    require(max_full_joint_bytes == 1700000000, "Unchanged native per-clip1.7GB ceiling required")
    require(rows.receipt.get("complete") is True and rows.receipt.get("rows") == len(rows)
            and rows.receipt.get("provenance", {}).get("training_admission_complete") is True
            and rows.receipt.get("provenance", {}).get("all_rows_verified") is True,
            "Existing complete admitted indexed source catalog required")
    for item in (source_catalog_record, baseline_tokenizer_record, candidate_tokenizer_record):
        require(record(item["path"]) == item, "Pinned source/tokenizer record changed")
    directory.mkdir(parents=True)
    database = directory / "labels.sqlite3.building"
    connection = sqlite3.connect(database)
    complete = False
    count = old_total = new_total = new_piece_uses = maximum_joint = 0
    maximum_old_u = maximum_new_u = 0
    ordered = hashlib.sha256()
    try:
        # Incremental primary/unique B-trees avoid a whole-pool sorting operation.
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA cache_size=-2048")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("CREATE TABLE labels (ordinal INTEGER PRIMARY KEY, source_id TEXT NOT NULL UNIQUE, "
                           "text_sha TEXT NOT NULL, frames INTEGER NOT NULL, old_u INTEGER NOT NULL, "
                           "old_labels_sha TEXT NOT NULL, new_u INTEGER NOT NULL, new_labels_sha TEXT NOT NULL, "
                           "label_ids BLOB NOT NULL, full_joint_bytes INTEGER NOT NULL)")
        for ordinal in range(len(rows)):
            row = rows[ordinal]
            require(row["language"] == "tr" and row.get("training_input_type") == "TR_frame_target_metadata_v1"
                    and "cache_path" not in row, "Source TRAIN metadata only; no replay/audio/teacher cache")
            old = list(row["tr_metadata_labels"])
            actual_old = _ids(baseline_tokenizer, row["text"])
            candidate = _ids(candidate_tokenizer, row["text"])
            require(old == actual_old and len(old) == row["target_tokens"]
                    and 1 <= len(old) <= 180 and all(type(v) is int and 0 <= v < OLD_VOCAB and v != BLANK for v in old),
                    "Original exact labels/target length changed: " + row["source_id"])
            require(1 <= len(candidate) <= 180
                    and all(type(v) is int and 0 <= v < NEW_VOCAB and v != BLANK for v in candidate),
                    "Candidate empty/blank/out-of-range/overlong target; reject whole sidecar")
            require(baseline_tokenizer.decode(old) == candidate_tokenizer.decode(candidate),
                    "Candidate changed the exact decoded training target")
            frames = row["encoder_frames"]
            require(type(frames) is int and frames > 0 and math.isfinite(float(row["duration"]))
                    and row["duration"] > 0, "Exact source frame/duration metadata required")
            joint = frames * (len(candidate) + 1) * (NEW_VOCAB + DURATIONS) * 4
            require(joint <= max_full_joint_bytes, "Candidate target exceeds unchanged native geometry ceiling")
            old_sha, new_sha = labels_sha(old), labels_sha(candidate)
            connection.execute("INSERT INTO labels VALUES (?,?,?,?,?,?,?,?,?,?)", (
                ordinal, row["source_id"], text_sha(row["text"]), frames, len(old), old_sha,
                len(candidate), new_sha, struct.pack("<" + "H" * len(candidate), *candidate), joint))
            ordered.update(json.dumps([ordinal, row["source_id"], frames, old_sha, new_sha],
                                      ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
            count += 1; old_total += len(old); new_total += len(candidate)
            new_piece_uses += sum(v >= OLD_VOCAB for v in candidate)
            maximum_old_u = max(maximum_old_u, len(old)); maximum_new_u = max(maximum_new_u, len(candidate))
            maximum_joint = max(maximum_joint, joint)
            if count % 512 == 0:
                connection.commit()
        connection.commit()
        require(count == len(rows) and connection.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == count,
                "Whole source TRAIN catalog must be present exactly once")
        require(connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "Label catalog integrity failed")
        connection.close(); connection = None
        with database.open("r+b") as stream:
            os.fsync(stream.fileno())
        target = directory / "labels.sqlite3"
        os.replace(database, target)
        evidence = {"kind": KIND, "complete": True, "rows": count,
                    "source_catalog": source_catalog_record, "source_fingerprint": rows.fingerprint,
                    "source_ids_sha256": rows.ids_sha,
                    "baseline_tokenizer": baseline_tokenizer_record, "candidate_tokenizer": candidate_tokenizer_record,
                    "all_original8199_targets_exact": True, "all_candidate_targets_decode_identically": True,
                    "replay_retokenized": False, "source_sampling_T_U_IDs_order_RNG_mutated": False,
                    "sampling_uses_original_target_lengths": True, "model_targets_are_separate": True,
                    "blank_id": BLANK, "vocabulary": NEW_VOCAB,
                    "native_joint_width": NEW_VOCAB + DURATIONS, "native_per_clip_limit": max_full_joint_bytes,
                    "maximum_full_joint_bytes": maximum_joint,
                    "source_target_tokens": old_total, "candidate_target_tokens": new_total,
                    "source_maximum_U": maximum_old_u, "candidate_maximum_U": maximum_new_u,
                    "candidate_added_piece_occurrences": new_piece_uses,
                    "ordered_row_target_mapping_sha256": ordered.hexdigest(),
                    "bounded_source_row_materialization": 1, "label_cache_cap_rows": 128,
                    "data": record(target), "audio_read": False, "model_or_Torch_imported": False}
        write_json(directory / "label_catalog_provenance.json", evidence)
        complete = True
        return evidence
    finally:
        if connection is not None:
            connection.close()
        if not complete:
            # Failed files remain explicit uncommitted evidence; never auto-delete.
            failure = directory / "FAILED.json"
            if not failure.exists():
                write_json(failure, {"complete": False, "rows_written_before_failure": count,
                                     "partial_sidecar_is_not_training_admitted": True})


class BPETrainingLabels:
    def __init__(self, directory, source_catalog, *, cache_size=128):
        self.directory = scoped(directory)
        self.receipt = json.loads((self.directory / "label_catalog_provenance.json").read_text(encoding="utf-8"))
        require(cache_size == 128 and self.receipt.get("kind") == KIND and self.receipt.get("complete") is True
                and self.receipt["rows"] == len(source_catalog)
                and self.receipt["source_fingerprint"] == source_catalog.fingerprint
                and self.receipt["source_ids_sha256"] == source_catalog.ids_sha,
                "Whole immutable same-ID source/label catalog admission required")
        database = self.directory / "labels.sqlite3"
        require(record(database) == self.receipt["data"], "Committed label sidecar bytes changed")
        self.connection = sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)
        self.cache = OrderedDict()
        self.cache_size = cache_size

    def _get(self, source_id):
        if source_id in self.cache:
            self.cache.move_to_end(source_id)
            return self.cache[source_id]
        value = self.connection.execute("SELECT text_sha,frames,old_u,old_labels_sha,new_u,new_labels_sha,label_ids "
                                        "FROM labels WHERE source_id=?", (source_id,)).fetchone()
        require(value is not None, "Unadmitted TRAIN source ID")
        self.cache[source_id] = value
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return value

    def rewrite_batch(self, rows):
        require(len(rows) <= 8, "Only one physical at-most-eight batch may be materialized")
        result = []
        for row in rows:
            if row["language"] != "tr":
                result.append(row)
                continue
            text_digest, frames, old_u, old_sha, new_u, new_sha, blob = self._get(row["source_id"])
            require(text_digest == text_sha(row["text"]) and frames == row["encoder_frames"]
                    and old_u == row["target_tokens"] and old_sha == labels_sha(row["tr_metadata_labels"]),
                    "Sampling source row differs from the immutable target-sidecar input")
            labels = list(struct.unpack("<" + "H" * new_u, blob))
            require(labels_sha(labels) == new_sha, "Candidate label payload changed")
            candidate = dict(row)
            candidate.update(original_target_tokens=old_u, sampling_target_tokens=old_u,
                             target_tokens=new_u, tr_metadata_labels=labels,
                             source_tr_frame_metadata_sha256=row.get("tr_frame_metadata_sha256"),
                             tr_frame_metadata_sha256=self.receipt["data"]["sha256"],
                             BPE_target_catalog_kind=KIND,
                             BPE_target_catalog_sha256=self.receipt["data"]["sha256"])
            result.append(candidate)
        return result

    def close(self):
        self.cache.clear()
        self.connection.close()
