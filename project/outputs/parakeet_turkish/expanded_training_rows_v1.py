"""Immutable disk-backed TRAIN rows; bounded decoded metadata, no ML imports.

The durable carrier preserves complete JSON values and ordering. Offset/length
columns are mmap-backed; source-ID lookup and sorted fingerprints use SQLite
on disk. Sampling implementations must retain indices, never list(rows).
"""
from __future__ import annotations
from collections import OrderedDict, Counter
from collections.abc import Sequence, Mapping
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import sqlite3
import struct
import tempfile

KIND = "indexed_bounded_expanded_TRAIN_rows_v1"
COL = struct.Struct("<IIdB")
OFF = struct.Struct("<Q")
GROUPS = ("fleurs", "cv", "tsc", "worldspeech")
FINGERPRINT_KEYS = ("source_id", "dataset", "dataset_revision", "dataset_split", "training_role",
                    "language", "audio_sha256", "text", "encoder_frames", "target_tokens", "label_ids_sha256", "duration")


def require(ok, message):
    if not ok: raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            require(line.strip(), "Blank TRAIN/metadata record")
            value = json.loads(line)
            require(type(value) is dict, "Object TRAIN record required")
            yield value


def group_of(row):
    if row.get("dataset") == "issai/Turkish_Speech_Corpus": return "tsc"
    if row.get("dataset") == "google/fleurs": return "fleurs"
    if row.get("dataset") == "ysdede/commonvoice_17_tr_fixed": return "cv"
    if row.get("dataset") == "disco-eth/WorldSpeech":
        require(row.get("dataset_revision") == "f389569345a04eb8ac8fa9405b0d5048ad74b8b3"
                and row.get("dataset_config") == "tr_tr" and row.get("dataset_split") == "train",
                "Only pinned Turkish WorldSpeech TRAIN is eligible")
        return "worldspeech"
    raise ValueError("Unrecognized accepted Turkish source group")


class RowIndexView(Sequence):
    def __init__(self, rows, indices): self.parent, self.indices = rows, indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, index):
        if isinstance(index, slice): return RowIndexView(self.parent, self.indices[index])
        return self.parent[self.indices[index]]


class _Column(Sequence):
    def __init__(self, rows, position): self.rows, self.position = rows, position
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        if isinstance(index, slice): return (self[i] for i in range(*index.indices(len(self))))
        value = COL.unpack_from(self.rows._columns, self.rows._index(index) * COL.size)[self.position]
        return GROUPS[value] if self.position == 3 else value


class _SourceIds(Sequence):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        if isinstance(index, slice): return (self[i] for i in range(*index.indices(len(self))))
        return self.rows._db.execute("SELECT source_id FROM catalog WHERE ordinal=?", (self.rows._index(index),)).fetchone()[0]
    def __iter__(self):
        for (source,) in self.rows._db.execute("SELECT source_id FROM catalog ORDER BY ordinal"): yield source


class _SourceIndex(Mapping):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __iter__(self): return iter(self.rows.source_ids)
    def __getitem__(self, source):
        found = self.rows._db.execute("SELECT ordinal FROM catalog WHERE source_id=?", (source,)).fetchone()
        if found is None: raise KeyError(source)
        return found[0]


class IndexedTrainingRows(Sequence):
    def __init__(self, directory, *, cache_size=256, verify=True):
        require(type(cache_size) is int and 0 <= cache_size <= 256, "Cache cap is256 rows")
        self.directory = Path(directory).resolve()
        self.receipt = json.loads((self.directory / "index_provenance.json").read_text(encoding="utf-8"))
        require(self.receipt.get("kind") == KIND and self.receipt.get("complete") is True, "Complete immutable row index required")
        self._count = self.receipt["rows"]
        require(type(self._count) is int and self._count > 0, "Positive complete row count required")
        for name, item in self.receipt["files"].items():
            path = self.directory / name
            require(path.parent == self.directory and path.stat().st_size == item["bytes"], "Index file/size changed")
            if verify: require(sha256(path) == item["sha256"], "Index file hash changed: " + name)
        require((self.directory / "offsets.u64").stat().st_size == (self._count + 1) * OFF.size
                and (self.directory / "columns.bin").stat().st_size == self._count * COL.size, "Exact compact column geometry required")
        self._offset_file = (self.directory / "offsets.u64").open("rb")
        self._column_file = (self.directory / "columns.bin").open("rb")
        self._offsets = mmap.mmap(self._offset_file.fileno(), 0, access=mmap.ACCESS_READ)
        self._columns = mmap.mmap(self._column_file.fileno(), 0, access=mmap.ACCESS_READ)
        self._row_file = (self.directory / "rows.jsonl").open("rb")
        self._db = sqlite3.connect((self.directory / "catalog.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True)
        self._db.execute("PRAGMA cache_size=-2048")
        require(self._db.execute("SELECT COUNT(*) FROM catalog").fetchone()[0] == self._count, "Catalog count changed")
        self.cache_size, self._cache = cache_size, OrderedDict()
        self.source_ids, self.index_by_source_id = _SourceIds(self), _SourceIndex(self)
        self.encoder_frames, self.target_tokens = _Column(self, 0), _Column(self, 1)
        self.durations, self.groups = _Column(self, 2), _Column(self, 3)
        self.fingerprint = self.receipt["fingerprint"]
        self.ids_sha = self.receipt["source_ids_sha256"]
        self.ordered_rows_sha = self.receipt["ordered_rows_sha256"]
    def __len__(self): return self._count
    def _index(self, value):
        require(type(value) is int, "Integer row index required")
        if value < 0: value += len(self)
        if not 0 <= value < len(self): raise IndexError(value)
        return value
    def __getitem__(self, index):
        if isinstance(index, slice): return RowIndexView(self, range(*index.indices(len(self))))
        index = self._index(index)
        payload = self._cache.pop(index, None)
        if payload is None:
            start = OFF.unpack_from(self._offsets, index * 8)[0]
            end = OFF.unpack_from(self._offsets, (index + 1) * 8)[0]
            require(0 <= start < end <= self.receipt["files"]["rows.jsonl"]["bytes"], "Invalid immutable row offsets")
            self._row_file.seek(start); payload = self._row_file.read(end - start)
            require(len(payload) == end - start and payload.endswith(b"\n"), "Truncated row carrier")
        if self.cache_size:
            self._cache[index] = payload
            while len(self._cache) > self.cache_size: self._cache.popitem(last=False)
        # Cache bytes, not mutable dictionaries: callers cannot poison later rows.
        row = json.loads(payload)
        if "tr_canonical_WAV_stat_at_preparation" in row:
            stat = Path(row["audio_filepath"]).stat()
            require(row["tr_canonical_WAV_stat_at_preparation"] ==
                    {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
                    "Actual admitted candidate WAV stat changed before batch loading")
        return row
    def __iter__(self):
        for index in range(len(self)): yield self[index]
    def length_key(self, index):
        t, u = self.encoder_frames[index], self.target_tokens[index]
        return (t.bit_length() - 1, u.bit_length() - 1, t, u, self.source_ids[index])
    @property
    def cache_rows(self): return len(self._cache)
    def close(self):
        self._cache.clear()
        self._db.close(); self._row_file.close(); self._offsets.close(); self._columns.close()
        self._offset_file.close(); self._column_file.close()


def prepare_indexed_rows(rows_iter, output_dir, *, expected_count, provenance, transform=None, cache_size=256, validate_result=None):
    """Consume one accepted row at a time and publish only a complete NEW carrier."""
    require(type(expected_count) is int and expected_count > 0, "Expected full pool count required")
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(output / "catalog.sqlite")
    db.execute("PRAGMA cache_size=-2048"); db.execute("PRAGMA temp_store=FILE")
    db.execute("CREATE TABLE catalog(ordinal INTEGER PRIMARY KEY, source_id TEXT UNIQUE NOT NULL, fingerprint BLOB NOT NULL)")
    ordered = hashlib.sha256(); counts = Counter(); duration_values = []
    count, offset, hours_sum, worst = 0, 0, 0., {"required_joint_bytes": 0}
    try:
        with (output / "rows.jsonl").open("xb") as data, (output / "offsets.u64").open("xb") as offsets, (output / "columns.bin").open("xb") as columns:
            for original in rows_iter:
                require(type(original) is dict, "Complete row dictionary required")
                row = original if transform is None else transform(original)
                require(type(row) is dict and row.get("language") == "tr" and type(row.get("source_id")) is str,
                        "Only complete accepted Turkish row dictionaries allowed")
                t, u, duration = row["encoder_frames"], row["target_tokens"], row["duration"]
                require(type(t) is int and type(u) is int and min(t, u) > 0 and type(duration) in (int, float)
                        and math.isfinite(duration) and duration > 0, "Finite authentic positive lengths required")
                group = group_of(row); counts[group] += 1
                payload = json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
                offsets.write(OFF.pack(offset)); data.write(payload); offset += len(payload)
                columns.write(COL.pack(t, u, duration, GROUPS.index(group)))
                ordered.update(canonical(row) + b"\n")
                db.execute("INSERT INTO catalog VALUES(?,?,?)", (count, row["source_id"], canonical({key: row.get(key) for key in FINGERPRINT_KEYS}) + b"\n"))
                count += 1; require(count <= expected_count, "Extra source rows; no subset/drop accepted")
                # Bounded chunked fsum avoids a full duration list and keeps roundoff small.
                duration_values.append(float(duration))
                if len(duration_values) == 256: hours_sum += math.fsum(duration_values); duration_values.clear()
                joint = t * (u + 1) * 8204 * 4
                if joint > worst["required_joint_bytes"]: worst = {"source_id": row["source_id"], "encoder_frames": t, "target_tokens": u, "required_joint_bytes": joint}
                if count % 10000 == 0: db.commit()
            offsets.write(OFF.pack(offset))
        require(count == expected_count, "Missing accepted rows; no short pool allowed")
        hours_sum += math.fsum(duration_values); db.commit()
        ids, fingerprint = hashlib.sha256(), hashlib.sha256()
        for index, (source, value) in enumerate(db.execute("SELECT source_id,fingerprint FROM catalog ORDER BY source_id COLLATE BINARY")):
            if index: ids.update(b"\n")
            ids.update(source.encode("utf-8")); fingerprint.update(value)
        db.close()
        result = {"schema": 1, "kind": KIND, "complete": True, "rows": count, "cache_cap_rows": 256,
                  "provenance": provenance, "source_ids_sha256": ids.hexdigest(), "fingerprint": fingerprint.hexdigest(),
                  "ordered_rows_sha256": ordered.hexdigest(), "source_group_counts": dict(counts), "hours": hours_sum / 3600,
                  "max_full_joint": worst, "full_original_row_values_and_key_order_preserved": True,
                  "files": {name: record(output / name) for name in ("rows.jsonl", "offsets.u64", "columns.bin", "catalog.sqlite")}}
        if validate_result is not None: validate_result(result)
        with (output / "index_provenance.json").open("x", encoding="utf-8") as stream: json.dump(result, stream, ensure_ascii=False, indent=2)
        return IndexedTrainingRows(output, cache_size=cache_size, verify=False)
    except BaseException:
        try: db.close()
        except Exception: pass
        # Partial files are diagnostic only; no completed receipt is published.
        raise


def iter_frame_annotated_rows(manifest, metadata, *, encode, metadata_sha256, check_wav_stat=True):
    """Exact per-row v3 annotator checks, without retaining either source array."""
    items = iter(iter_jsonl(metadata))
    for row in iter_jsonl(manifest):
        item = next(items, None)
        require(item is not None, "Missing ordered frame metadata")
        mel = item["actual_PCM_frames"] // 160; frames = (mel + 7) // 8
        require(row["language"] == "tr" and row["source_id"] == item["source_id"]
                and int(round(row["duration"] * 16000)) == item["actual_PCM_frames"]
                and item["valid_mel_frames"] == mel and item["encoder_frames"] == frames,
                "Approved ordered PCM/frame metadata changed")
        labels = encode(row["text"])
        require(type(labels) is list and 1 <= len(labels) <= 180 and len(labels) == item["target_tokens"]
                and all(type(value) is int and 0 <= value < 8199 and value != 8192 for value in labels)
                and hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest() == item["label_ids_sha256"],
                "Original8199 actual labels/U/hash changed")
        if check_wav_stat and "canonical_WAV_stat_at_preparation" in item:
            stat = Path(row["audio_filepath"]).stat()
            require(item["canonical_WAV_stat_at_preparation"] == {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
                    "Canonical WAV stat changed; fresh verification required")
        yield {**row, "encoder_frames": frames, "target_tokens": len(labels), "training_input_type": "TR_frame_target_metadata_v1",
               "tr_metadata_labels": labels, "tr_frame_metadata_sha256": metadata_sha256,
               "tr_canonical_WAV_stat_at_preparation": item["canonical_WAV_stat_at_preparation"]}
    require(next(items, None) is None, "Extra frame metadata beyond complete source")


# Frozen original storage/annotator implementation; only the new group/kind above
# differ. Approval below binds actual preparation, independent verification and
# the actual TRAIN-only sanity review; generic carriers remain assessment only.
ORIGINAL_GROUPS = {"cv": 72846, "fleurs": 2037, "tsc": 178915}
CANDIDATE_COUNT = 98970
ORIGINAL_COUNT = 253798
PREPARATION_SHA = "8aac435e3472d3e304a88c4ac952756f1a01682682ef1d8cc82beedbc694e9e7"
VERIFICATION_SHA = "f4e1fe9b487d1a8fc8da25e9ed0faf9f8ff35cf293bcc87ace4b084fa0b55e3f"
SANITY_REVIEW_SHA = "19b1469269dfca2c428c5da06e28f4c1fb7088f793050e9160982262ff7264b3"


def checked_expansion_inputs(preparation_path, verification_path, sanity_review_path):
    inputs = [record(path) for path in (preparation_path, verification_path, sanity_review_path)]
    require([x["sha256"] for x in inputs] == [PREPARATION_SHA, VERIFICATION_SHA, SANITY_REVIEW_SHA],
            "Exact completed candidate/verification/TRAIN-sanity records required")
    preparation, verification, sanity = [json.loads(Path(x["path"]).read_text(encoding="utf-8-sig")) for x in inputs]
    require(preparation["status"] == "complete" and preparation["prepared_candidate_clips"] == CANDIDATE_COUNT
            and verification["passed"] is True and verification["preparation"] == inputs[0]
            and verification["actual_all_record_label_frame_stat_and_heldout_hash_checks"] == CANDIDATE_COUNT
            and verification["raw_human_labels_preserved"] is True
            and verification["duplicate_source_or_embedded_or_PCM_or_WAV_found"] is False
            and verification["exact_existing_TRAIN_replay_audio_overlap_found"] is False
            and verification["fixed_DEV_TEST_ID_audio_normalized_text_overlap_found"] is False,
            "Complete exact source eligibility evidence required")
    require(sanity["kind"] == "WorldSpeech_TRAIN_sanity_review_v1" and sanity["passed"] is True
            and sanity["approved_for_controlled_training_admission"] is True
            and sanity["actual_owned_exit"] == {"session_id": 57592, "exit_code": 0, "tool_chunk": "4e543f"}
            and sanity["source_refs_unchanged"] is True and sanity["optimizer_updates"] == 0,
            "Actual bounded TRAIN sanity review required; no manual-label or model-improvement claim")
    for item in preparation["exports"]:
        require(record(item["path"]) == item, "Candidate export bytes changed")
    return preparation, verification, sanity, inputs


def prepare_approved_expanded_TR_rows(original_rows, output_dir, *, encode, preparation_path,
                                      verification_path, sanity_review_path,
                                      native_max_joint_bytes=1_850_000_000, cache_size=256):
    """Stream old approved rows then actual verified candidate rows into a NEW index."""
    preparation, verification, sanity, inputs = checked_expansion_inputs(
        preparation_path, verification_path, sanity_review_path)
    original = original_rows.receipt
    provenance = original.get("provenance", {})
    require(original["kind"] == "indexed_bounded_TRAIN_rows_v1" and original["complete"] is True
            and len(original_rows) == original["rows"] == ORIGINAL_COUNT
            and original["source_group_counts"] == ORIGINAL_GROUPS
            and provenance.get("training_admission_complete") is True
            and provenance.get("all_rows_verified") is True and provenance.get("verified_clips") == ORIGINAL_COUNT,
            "Unchanged original whole-pool approved carrier required first")
    require(native_max_joint_bytes == 1_850_000_000, "Unchanged declared native bound required")
    manifests = {Path(x["path"]).name: x for x in preparation["exports"]}
    manifest = manifests["candidate_train.jsonl"]; metadata = manifests["candidate_frame_metadata.jsonl"]
    counts = {"original": 0, "worldspeech": 0}
    def rows():
        for row in original_rows:
            require(group_of(row) in ORIGINAL_GROUPS, "Original source group changed")
            counts["original"] += 1
            yield row
        for row in iter_frame_annotated_rows(manifest["path"], metadata["path"], encode=encode,
                                             metadata_sha256=metadata["sha256"], check_wav_stat=True):
            require(group_of(row) == "worldspeech" and row["target_tokens"] <= 180
                    and row["encoder_frames"] * (row["target_tokens"] + 1) * 8204 * 4 <= native_max_joint_bytes,
                    "Verified original8199 candidate native geometry required")
            counts["worldspeech"] += 1
            yield row
        require(counts == {"original": ORIGINAL_COUNT, "worldspeech": CANDIDATE_COUNT}, "Whole union counts changed")
    def validate(result):
        require(result["source_group_counts"] == {**ORIGINAL_GROUPS, "worldspeech": CANDIDATE_COUNT}
                and result["max_full_joint"]["required_joint_bytes"] <= native_max_joint_bytes
                and result["rows"] == ORIGINAL_COUNT + CANDIDATE_COUNT,
                "Complete four-source union and actual lattice scan required")
    return prepare_indexed_rows(rows(), output_dir, expected_count=ORIGINAL_COUNT + CANDIDATE_COUNT,
        cache_size=cache_size, validate_result=validate,
        provenance={"kind": "verified_WorldSpeech_and_original_TRAIN_union_v1",
            "training_admission_complete": True, "all_rows_verified": True,
            "verified_clips": ORIGINAL_COUNT + CANDIDATE_COUNT,
            "original_carrier": record(original_rows.directory / "index_provenance.json"),
            "original_carrier_fingerprint": original_rows.fingerprint,
            "candidate_preparation": inputs[0], "independent_CPU_verification": inputs[1],
            "actual_TRAIN_sanity_review": inputs[2], "expanded_helper": record(__file__),
            "original_rows_preserved_value_for_value_in_same_prefix_order": True,
            "candidate_human_labels_replaced": False, "heldout_predictions_used": False,
            "speaker_or_perceptual_disjointness_proven": False,
            "native_training_memory_fit_proved": False,
            "native_max_joint_bytes": native_max_joint_bytes})


def reopen_approved_expanded_TR_rows(directory, *, cache_size=256):
    store = IndexedTrainingRows(directory, cache_size=cache_size, verify=True)
    try:
        value = store.receipt; proof = value["provenance"]
        require(value["rows"] == len(store) == ORIGINAL_COUNT + CANDIDATE_COUNT
                and value["source_group_counts"] == {**ORIGINAL_GROUPS, "worldspeech": CANDIDATE_COUNT}
                and proof["training_admission_complete"] is True and proof["all_rows_verified"] is True
                and proof["verified_clips"] == len(store) and proof["expanded_helper"] == record(__file__),
                "Exact completed four-source admission/helper required")
        checked_expansion_inputs(proof["candidate_preparation"]["path"],
                                 proof["independent_CPU_verification"]["path"],
                                 proof["actual_TRAIN_sanity_review"]["path"])
        require(record(proof["original_carrier"]["path"]) == proof["original_carrier"], "Original approved index receipt changed")
        return store
    except BaseException:
        store.close()
        raise
