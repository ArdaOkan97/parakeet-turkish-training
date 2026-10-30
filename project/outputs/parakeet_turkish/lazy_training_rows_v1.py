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

KIND = "indexed_bounded_TRAIN_rows_v1"
COL = struct.Struct("<IIdB")
OFF = struct.Struct("<Q")
GROUPS = ("fleurs", "cv", "tsc")
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
        return json.loads(payload)
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
               "tr_metadata_labels": labels, "tr_frame_metadata_sha256": metadata_sha256}
    require(next(items, None) is None, "Extra frame metadata beyond complete source")


def iter_array_field(path, field):
    """Stream one object-array field in a proof; other proof fields stay small."""
    decoder = json.JSONDecoder(parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite proof JSON")))
    with Path(path).open(encoding="utf-8-sig") as stream:
        buffer, position, ended = "", 0, False
        def fill():
            nonlocal buffer, position, ended
            chunk = stream.read(65536)
            buffer = buffer[position:] + chunk; position = 0
            if not chunk: ended = True
        def white():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace(): position += 1
                if position < len(buffer) or ended: return
                fill()
        def token(expected):
            nonlocal position
            white(); require(buffer[position:position+1] == expected, "Invalid streamed proof separator")
            position += 1
        def value():
            nonlocal position
            white()
            while True:
                try:
                    result, end = decoder.raw_decode(buffer, position)
                    if end == len(buffer) and not ended and type(result) in (int, float): fill(); continue
                    position = end; return result
                except json.JSONDecodeError:
                    require(not ended and len(buffer) < 4_000_000, "Truncated/oversized individual proof value")
                    fill()
        token("{"); seen = set(); found = False
        while True:
            white()
            if buffer[position:position+1] == "}": token("}"); break
            key = value(); require(type(key) is str and key not in seen, "Unique proof keys required")
            seen.add(key); token(":")
            if key == field:
                found = True; token("["); white()
                if buffer[position:position+1] != "]":
                    while True:
                        row = value(); require(type(row) is dict, "Proof array record must be an object")
                        yield row; white()
                        if buffer[position:position+1] == "]": break
                        token(",")
                token("]")
            else: value()
            white()
            if buffer[position:position+1] == "}": token("}"); break
            token(",")
        require(found, "Required proof array field absent")
        white(); require(not buffer[position:].strip() and not stream.read(1), "Trailing proof content")


def prepare_approved_TR_rows(args, processor, output_dir, *, cache_size=256, heldout_paths=None):
    """Stream unchanged v2 policy, raw prefix/suffix, PCM dedup and exact labels.

    Frozen preparation/import/quarantine are revalidated by hashes. Complete
    source rows and parent-label evidence are streamed, not decoded as arrays.
    No WAV payload or archive is read; canonical stat checks remain unchanged.
    """
    import tsc_training_inputs_v2 as inputs
    policy = inputs.policy; base = policy.parent
    proof = policy.read_json(args.tsc_input_proof); refs = proof["references"]
    require(proof.get("kind") == "TSC_supplemental_TR_frame_target_inputs_v2" and proof.get("passed") is True
            and proof.get("CPU_only") is True and proof.get("helper_sha256") == sha256(inputs.__file__)
            and sha256(policy.__file__) == inputs.SUPPLEMENT_POLICY_SHA
            and proof.get("clips") == 253798 and proof.get("original_TR_clips") == 74883
            and proof.get("TSC_clips") == 178915 and proof.get("training_launch_eligible") is True
            and proof["max_full_joint_bytes"] <= args.native_max_joint_bytes and args.max_tokens == 180,
            "Exact whole original8199/quarantined253798 input proof required")
    checked = {}
    def verify(item):
        path = Path(item["path"]).resolve()
        require(path.is_relative_to(policy.ROOT), "Workspace proof-bound source required")
        token = (path.stat().st_size, path.stat().st_mtime_ns)
        if path not in checked:
            require(sha256(path) == item["sha256"], "Immutable admission source changed: " + str(path))
            checked[path] = token
        else: require(checked[path] == token, "Admission source changed while streamed")
        return path
    for item in [*refs.values(), *proof["files"].values()]: verify(item)
    for flag, key in (("tr_manifest", "combined_TR"), ("replay_manifest", "original_replay"), ("tr_frame_metadata", "TR_metadata")):
        require(Path(getattr(args, flag)).resolve() == verify(proof["files"][key]), "Exact prepared source path required")
    require(Path(args.tr_frame_metadata_proof).resolve() == Path(args.tsc_input_proof).resolve(), "Same exact frame/input proof required")
    inputs.approve(args.tsc_training_approval, refs["import_proof"]["path"], training=True,
                   quarantine_proof_path=refs["quarantine_proof"]["path"])
    parent = policy.read_json(verify({"path": policy.OUTPUT / "tr_public_v3_preparation.json", "sha256": policy.PARENT_PROOF_SHA}))
    labels_path = verify({"path": policy.OUTPUT / "tr_public_v3_tokenizer_acceptance.json", "sha256": policy.PARENT_LABEL_SHA})
    for name, digest in (("prepare_TSC_TRAIN_v1.py", policy.APPROVED_IMPORTER_SHA), ("public_turkish_data_policy_v3.py", policy.V3_POLICY_SHA),
                         ("metrics.py", policy.NORMALIZER_SHA), ("checkpoint/tokenizer.json", policy.TOKENIZER_FILE_SHA)):
        verify({"path": policy.OUTPUT / name, "sha256": digest})
    imported = policy.read_json(refs["import_proof"]["path"])
    base.validate_import_proof(imported, parent, require_full_train_attempt=True)
    for path, digest in imported["protected_existing_manifests_sha256"].items(): verify({"path": path, "sha256": digest})
    original_path = verify({"path": parent["CPU_tokenizer_accepted_manifest_path"], "sha256": parent["CPU_tokenizer_accepted_manifest_sha256"]})
    quarantine = policy.read_json(refs["quarantine_proof"]["path"])
    require(quarantine.get("kind") == "TSC_TRAIN_uncertainty_quarantine_v2" and quarantine.get("passed") is True
            and quarantine.get("helper_sha256") == policy.QUARANTINE_HELPER_SHA
            and sha256(policy.OUTPUT / "prepare_TSC_quarantine_v2.py") == policy.QUARANTINE_HELPER_SHA
            and sha256(base.__file__) == policy.PARENT_POLICY_SHA, "Exact completed quarantine sources required")
    for item in quarantine["references"].values(): verify(item)
    retained_path = verify(quarantine["retained_manifest"])
    original_tsc_path = verify({"path": imported["accepted_manifest_path"], "sha256": imported["accepted_manifest_sha256"]})
    require(quarantine["references"]["full_import_proof"] == refs["import_proof"]
            and quarantine["references"]["original_accepted_manifest"] == {"path": str(original_tsc_path), "sha256": imported["accepted_manifest_sha256"]}
            and quarantine["original_clips"] == imported["accepted_clips"] == 178918
            and quarantine["retained_clips"] == 178915 and quarantine["quarantined_clips"] == 3
            and quarantine["combined_TR_clips"] == 253798 and quarantine["kept_manifest_rows_byte_identical"] is True
            and quarantine["original_labels_of_retained_rows_preserved"] is True, "Actual quarantine/full-import binding changed")
    closure = policy.read_json(quarantine["references"]["import_closure"]["path"])
    require(closure.get("kind") == "root_owned_TSC_full_import_closure_v2" and closure.get("passed") is True
            and closure.get("root_observed_owned_session_completed") is True and closure.get("child_exit_code") == 0
            and closure["import_proof"] == refs["import_proof"] and closure["accepted_manifest"] == quarantine["references"]["original_accepted_manifest"],
            "Actual full importer closure required")
    cases, sanity = policy.review_cases(quarantine["references"]["ROOT_Whisper_review"]["path"])
    require(sanity == quarantine["references"]["actual_Whisper_TRAIN_sanity"], "Quarantine actual reviewed evidence changed")
    semantic = hashlib.sha256(json.dumps(json.loads(processor.tokenizer._tokenizer.to_str()), ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    require(semantic == policy.TOKENIZER_SEMANTIC_SHA, "Exact original8199 ungrouped tokenizer required")
    replay = list(iter_jsonl(args.replay_manifest))  # The bounded original1595 is intentionally resident.
    require(len(replay) == 1595 and policy.ids_sha(replay) == policy.REPLAY_SOURCE_SET_SHA
            and policy.ordered_rows_sha(replay) == proof["identity"]["replay_ordered_rows_sha256"], "Original replay rows/cache/order changed")
    heldout = [row for path, digest in parent["heldout_manifests_sha256"].items()
               for row in iter_jsonl(verify({"path": path, "sha256": digest}))]
    if heldout_paths:
        for path in heldout_paths: heldout.extend(iter_jsonl(path))
    heldout_ids = {row["source_id"] for row in heldout}
    heldout_sentences = {row["sentence_id"] for row in heldout if row.get("dataset") == policy.FL_REPO and row.get("sentence_id") is not None}
    heldout_hashes = {row[key] for row in heldout for key in ("audio_sha256", "wav_sha256", "canonical_pcm_sha256") if row.get(key)}
    references = {(row["language"], policy.parent.normalize_text(row["text"], row["language"])) for row in heldout}
    tr_refs = {policy.parent.normalize_text(row["text"], language) for row in heldout for language in (row["language"], "tr")}
    def isolation(row):
        require(row["source_id"] not in heldout_ids and not (row.get("dataset") == policy.FL_REPO and row.get("sentence_id") in heldout_sentences)
                and not any(row.get(key) in heldout_hashes for key in ("audio_sha256", "wav_sha256", "canonical_pcm_sha256") if row.get(key))
                and (row["language"], base.normalize_text(row["text"], row["language"])) not in references
                and not (row["language"] == "tr" and base.normalize_text(row["text"], "tr") in tr_refs), "Fixed heldout overlap: " + row["source_id"])
    raw_digest = hashlib.sha256(); prefix_digest = hashlib.sha256(); retained_digest = hashlib.sha256()
    raw_count = 0
    with tempfile.TemporaryDirectory(prefix="lazy_admission_", dir=policy.ROOT / "work") as scratch:
        guard = sqlite3.connect(Path(scratch) / "guards.sqlite")
        guard.execute("PRAGMA cache_size=-2048"); guard.execute("PRAGMA temp_store=FILE")
        guard.execute("CREATE TABLE hashes(value TEXT PRIMARY KEY)")
        guard.execute("CREATE TABLE parent_pcm(source_id TEXT PRIMARY KEY,value TEXT NOT NULL)")
        for item in iter_array_field(labels_path, "accepted_source_labels_and_audio"):
            guard.execute("INSERT INTO parent_pcm VALUES(?,?)", (item["source_id"], item["canonical_pcm_sha256"]))
            guard.execute("INSERT OR IGNORE INTO hashes VALUES(?)", (item["canonical_pcm_sha256"],))
        require(guard.execute("SELECT COUNT(*) FROM parent_pcm").fetchone()[0] == 74883, "Complete original PCM source evidence required")
        for row in replay:
            base.validate_original_source(row); isolation(row)
            for key in ("audio_sha256", "wav_sha256", "canonical_pcm_sha256"):
                if row.get(key): guard.execute("INSERT OR IGNORE INTO hashes VALUES(?)", (row[key],))
        def vetted_raw():
            nonlocal raw_count
            original, kept = iter(iter_jsonl(original_path)), iter(iter_jsonl(retained_path))
            for row in iter_jsonl(args.tr_manifest):
                raw_digest.update(canonical(row) + b"\n"); isolation(row)
                if raw_count < 74883:
                    require(next(original, None) == row, "Original74883 prefix value/order changed")
                    base.validate_original_source(row); prefix_digest.update(canonical(row) + b"\n")
                    require(guard.execute("SELECT value FROM parent_pcm WHERE source_id=?", (row["source_id"],)).fetchone() is not None,
                            "Original canonical PCM/source identity changed")
                    guard.execute("DELETE FROM parent_pcm WHERE source_id=?", (row["source_id"],))
                else:
                    require(next(kept, None) == row, "Exact retained TSC suffix/order changed")
                    base.validate_tsc_row(row, original_tsc_path); retained_digest.update(canonical(row) + b"\n")
                    require(all(guard.execute("SELECT 1 FROM hashes WHERE value=?", (value,)).fetchone() is None
                                for value in {row[key] for key in ("audio_sha256", "wav_sha256", "canonical_pcm_sha256")}),
                            "TSC audio/PCM duplicates existing TRAIN or previous TSC")
                for key in ("audio_sha256", "wav_sha256", "canonical_pcm_sha256"):
                    if row.get(key): guard.execute("INSERT OR IGNORE INTO hashes VALUES(?)", (row[key],))
                raw_count += 1
                if raw_count % 10000 == 0: guard.commit()
                yield row
            require(raw_count == 253798 and next(original, None) is None and next(kept, None) is None
                    and guard.execute("SELECT COUNT(*) FROM parent_pcm").fetchone()[0] == 0,
                    "Complete exact prefix/suffix source frontier required")
            require(raw_digest.hexdigest() == proof["identity"]["combined_TR_ordered_rows_sha256"]
                    and retained_digest.hexdigest() == quarantine["retained_ordered_rows_sha256"], "Raw accepted pool canonical identity changed")
        def annotated():
            metadata = iter(iter_jsonl(args.tr_frame_metadata))
            for row in vetted_raw():
                item = next(metadata, None); require(item is not None, "Missing authentic metadata")
                mel = item["actual_PCM_frames"] // 160; frames = (mel + 7) // 8
                require(row["source_id"] == item["source_id"] and int(round(row["duration"]*16000)) == item["actual_PCM_frames"]
                        and item["valid_mel_frames"] == mel and item["encoder_frames"] == frames, "Authentic PCM/frame alignment changed")
                labels = processor.tokenizer(row["text"])["input_ids"]
                require(1 <= len(labels) <= 180 and len(labels) == item["target_tokens"]
                        and all(type(v) is int and 0 <= v < 8199 and v != 8192 for v in labels)
                        and hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest() == item["label_ids_sha256"], "Actual labels/U changed")
                if "canonical_WAV_stat_at_preparation" in item:
                    stat = Path(row["audio_filepath"]).stat()
                    require(item["canonical_WAV_stat_at_preparation"] == {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}, "Canonical WAV stat changed")
                yield {**row, "encoder_frames": frames, "target_tokens": len(labels), "training_input_type": inputs.KIND,
                       "tr_metadata_labels": labels, "tr_frame_metadata_sha256": proof["metadata_sha256"]}
            require(next(metadata, None) is None, "Extra authentic metadata")
            for path, token in checked.items(): require(token == (path.stat().st_size, path.stat().st_mtime_ns), "Pinned source changed during admission")
        def validate_result(result):
            require(result["source_ids_sha256"] == proof["identity"]["combined_TR_source_ids_sha256"]
                    and abs(result["hours"] - proof["identity"]["combined_TR_actual_PCM_hours"]) <= 1e-10
                    and result["source_group_counts"] == {"cv": 72846, "fleurs": 2037, "tsc": 178915}
                    and result["max_full_joint"]["required_joint_bytes"] <= args.native_max_joint_bytes,
                    "Whole source IDs/hours/counts/native bound changed before index publication")
        store = prepare_indexed_rows(annotated(), output_dir, expected_count=253798, cache_size=cache_size, validate_result=validate_result,
            provenance={"training_admission_complete": True, "all_rows_verified": True, "verified_clips": 253798,
                        "lazy_helper": record(__file__),
                        "source_input_proof": record(args.tsc_input_proof), "ROOT_training_approval": record(args.tsc_training_approval),
                        "raw_ordered_rows_sha256": proof["identity"]["combined_TR_ordered_rows_sha256"],
                        "immutable_manifest_and_metadata": proof["files"], "complete_importer_PCM_checks_reused": True})
        guard.close()
    require(store.ids_sha == proof["identity"]["combined_TR_source_ids_sha256"]
            and abs(store.receipt["hours"] - proof["identity"]["combined_TR_actual_PCM_hours"]) <= 1e-10,
            "Actual complete source IDs/hours differ")
    identity = {**proof["identity"], "combined_TR_manifest_path": str(Path(args.tr_manifest).resolve()),
                "combined_TR_manifest_sha256": proof["files"]["combined_TR"]["sha256"], "accepted_clips": 253798,
                "accepted_actual_PCM_hours": store.receipt["hours"], "frame_target_preparation_sha256": sha256(args.tsc_input_proof),
                "ROOT_preparation_approval_sha256": refs["ROOT_approval"]["sha256"], "ROOT_training_approval": inputs.file_record(args.tsc_training_approval),
                "replay_sampling_stream_bitwise_identity_claimed": False}
    return store, {"schema": 1, "kind": "TSC_quarantined_streamed_metadata_input_audit_v1", "passed": True,
                   "identity": identity, "accepted_source_ids_sha256": store.ids_sha,
                   "heldout_source_ids_sha256": policy.ids_sha(heldout), "replay_original_ordered_rows_sha256": policy.ordered_rows_sha(replay),
                   "actual_TR_metadata_accepted_clips": len(store), "actual_TR_metadata_accepted_hours": store.receipt["hours"],
                   "turkish_training_representation": inputs.KIND, "TR_frozen_encoder_feature_cache_required": False,
                   "TR_metadata_acceptance_pending": False, "new_TR_frame_target_metadata_acceptance_pending": False,
                   "acoustic_cache_acceptance_pending": "not_applicable_to_TR_metadata", "no_model_GPU_or_heldout_predictions_used": True,
                   "actual_audio_read_in_this_audit": False, "complete_importer_heldout_PCM_checks_reused": True,
                   "current_WAV_payload_mutation_not_independently_excluded": True,
                   "speaker_video_and_perceptual_near_duplicate_disjointness_not_proven": True,
                   "indexed_carrier": record(Path(output_dir) / "index_provenance.json")}


def reopen_approved_TR_rows(args, directory, public_audit, *, cache_size=256):
    """Reopen an original immutable admission; no re-tokenization/new carrier."""
    import tsc_training_inputs_v2 as inputs
    require(type(public_audit) is dict and public_audit.get("passed") is True
            and public_audit.get("actual_TR_metadata_accepted_clips") == 253798,
            "Original scientific config's admitted compact audit required")
    require(public_audit.get("indexed_carrier") == record(Path(directory) / "index_provenance.json"), "Original indexed receipt changed")
    store = IndexedTrainingRows(directory, cache_size=cache_size, verify=True)
    try:
        provenance = store.receipt["provenance"]
        require(provenance.get("training_admission_complete") is True and provenance.get("all_rows_verified") is True
                and provenance.get("verified_clips") == len(store) == 253798 and provenance.get("lazy_helper") == record(__file__)
                and provenance.get("source_input_proof") == record(args.tsc_input_proof)
                and provenance.get("ROOT_training_approval") == record(args.tsc_training_approval), "Exact admitted source/helper/approval changed")
        proof = inputs.policy.read_json(args.tsc_input_proof)
        require(provenance["immutable_manifest_and_metadata"] == proof["files"]
                and store.ids_sha == proof["identity"]["combined_TR_source_ids_sha256"]
                and provenance["raw_ordered_rows_sha256"] == proof["identity"]["combined_TR_ordered_rows_sha256"]
                and public_audit["accepted_source_ids_sha256"] == store.ids_sha, "Exact source/pool identity changed")
        for flag, key in (("tr_manifest", "combined_TR"), ("replay_manifest", "original_replay"), ("tr_frame_metadata", "TR_metadata")):
            item = proof["files"][key]; path = Path(getattr(args, flag)).resolve()
            require(path == Path(item["path"]).resolve() and sha256(path) == item["sha256"], "Original source input changed")
        inputs.approve(args.tsc_training_approval, proof["references"]["import_proof"]["path"], training=True,
                       quarantine_proof_path=proof["references"]["quarantine_proof"]["path"])
        return store, public_audit
    except BaseException:
        store.close(); raise


def _profile_main():
    import argparse
    import gc
    import sys
    import time
    parser = argparse.ArgumentParser(description="CPU-only bounded real TRAIN metadata profile; not new training admission")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=253798)
    args = parser.parse_args()
    require(not args.report.exists(), "Fresh profile output required")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
    priority = {"platform": os.name, "below_normal_applied": False}
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel.SetPriorityClass.restype = ctypes.c_int
        priority.update(below_normal_applied=bool(kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000)))
    from resource_watchdog_v2 import sample_memory
    observations = []; last_sample = 0.; peak = 0
    def observe(phase, force=False):
        nonlocal last_sample, peak
        now = time.monotonic()
        if not force and now - last_sample < 2.: return
        last_sample = now; value = sample_memory(os.getpid(), phase)
        observations.append(value)
        worker = value.get("worker_private_memory", {})
        peak = max(peak, worker.get("private_bytes", 0) or 0)
        system = value.get("system_memory", {})
        require(system.get("available") is True and system.get("commit_fraction", 1.) < .90
                and system.get("commit_available_bytes", 0) >= 2_000_000_000,
                "CPU metadata profile stopped for real system commit/headroom guard")
    observe("before_tokenizer", True)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    require(tokenizer.get_vocab_size() == 8199 and tokenizer.token_to_id("<blank>") == 8192, "Pinned original8199 tokenizer required")
    tokenizer_semantic = hashlib.sha256(json.dumps(json.loads(tokenizer.to_str()), ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    require(tokenizer_semantic == "b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75", "Tokenizer semantic identity changed")
    inputs = {name: record(path) for name, path in (("manifest", args.manifest), ("metadata", args.metadata), ("tokenizer", args.tokenizer))}
    start = time.perf_counter()
    def rows():
        for index, row in enumerate(iter_frame_annotated_rows(args.manifest, args.metadata,
                encode=lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
                metadata_sha256=inputs["metadata"]["sha256"], check_wav_stat=False)):
            if index % 1000 == 0: observe("stream_stage")
            yield row
    store = prepare_indexed_rows(rows(), args.output_dir, expected_count=args.expected_count,
        provenance={"input_records": inputs, "tokenizer_semantic_sha256": tokenizer_semantic,
                    "actual_original8199_target_ids_and_frames_verified": True,
                    "training_admission_complete": False, "profile_only": True,
                    "WAV_stat_checks_deferred_to_training_admission": True})
    elapsed = time.perf_counter() - start
    # Exercise a spread across the entire real pool, then prove cache stays bounded.
    for index in range(min(1024, len(store))): store[(index * 997) % len(store)]
    observe("indexed_open_with_256_row_cache", True)
    count, receipt, cached = len(store), store.receipt, store.cache_rows
    store.close(); del store, tokenizer; gc.collect(); observe("after_close_and_gc", True)
    require(all(inputs[name] == record(path) for name, path in (("manifest", args.manifest), ("metadata", args.metadata), ("tokenizer", args.tokenizer))),
            "Original immutable metadata inputs changed during profile")
    require("torch" not in sys.modules and not any(name.startswith("nemo") for name in sys.modules), "No Torch/NeMo import allowed")
    result = {"schema": 1, "kind": "actual_CPU_indexed_TRAIN_metadata_profile_v1", "passed": True,
              "actual_rows_staged": count, "all_original8199_target_ids_and_frames_verified": True,
              "training_admission_complete": False, "no_original_manifest_mutation": True,
              "Torch_imported": False, "GPU_used": False, "audio_content_read": False,
              "large_full_source_arrays_loaded": False, "decoded_row_cache_count": cached,
              "stage_seconds": elapsed, "own_private_peak_bytes_observed": peak, "priority": priority,
              "index_provenance": record(args.output_dir / "index_provenance.json"),
              "carrier": receipt, "memory_observations": observations,
              "limitation": "Measured only bounded staging/cache process. No full-resident baseline was allocated; actual GPU/worker lifetime remains unmeasured."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream: json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ("passed", "actual_rows_staged", "stage_seconds", "own_private_peak_bytes_observed")}))


if __name__ == "__main__": _profile_main()
