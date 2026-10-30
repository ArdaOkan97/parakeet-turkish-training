"""Share only immutable Python objects in an already audited training carrier.

The list, row dictionaries, label lists, types, values, and orders stay intact.
No corpus/ML/RNG is read here. The caller supplies actual admitted rows and
owns host samples. Estimates count per-row distinct redundant references and
can overcount pre-existing sharing across rows; they are not measured RAM.
"""
from __future__ import annotations

import hashlib
import json
import sys

VOCAB_SIZE = 8711
MAX_SHARED_KEYS = 256
MAX_METADATA_VALUES_PER_FIELD = 128
MAX_METADATA_STRING_LENGTH = 256
METADATA_FIELDS = frozenset(("dataset", "dataset_revision", "dataset_config", "dataset_split", "dataset_file",
    "language", "gender", "training_role", "source", "training_input_type", "tr_frame_metadata_sha256",
    "bpe512_TR_metadata_kind", "bpe512_TR_report_sha256", "bpe512_candidate_semantic_sha256"))


def require(ok, message):
    if not ok: raise ValueError(message)


def canonical(row):
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def compact_training_rows(rows, *, expected_rows=253798):
    """Return evidence after one in-place row stream, with no retained second pool."""
    require(type(rows) is list and type(expected_rows) is int and expected_rows > 0
        and len(rows) == expected_rows, "Complete already-admitted row list required")
    keys, values = {}, {field:{} for field in METADATA_FIELDS}
    integers = tuple(range(VOCAB_SIZE))
    used_new_integers = set()
    before_digest, after_digest = hashlib.sha256(), hashlib.sha256()
    changed_keys = changed_labels = changed_metadata = shared_text = 0
    eliminated_reference_bytes = 0
    for index, row in enumerate(rows):
        require(type(row) is dict and row.get("bpe512_TR_admission_complete") is True,
            "Every actual candidate TRAIN row must already pass admission")
        labels = row.get("tr_metadata_labels")
        require(type(labels) is list and all(type(value) is int and 0 <= value < VOCAB_SIZE and value != 8192 for value in labels),
            "Existing candidate label LIST with safe lexical int IDs required")
        row_id, label_id = id(row), id(labels)
        key_order = tuple(row)
        before = hashlib.sha256(canonical(row)).digest()
        before_digest.update(before)
        # A small one-row set avoids double-counting aliases within a row.
        # It does NOT claim global distinct-object accounting across all rows.
        eliminated_this_row = set()
        def eliminated(value):
            nonlocal eliminated_reference_bytes
            identifier=id(value)
            if identifier not in eliminated_this_row:
                eliminated_this_row.add(identifier)
                eliminated_reference_bytes += sys.getsizeof(value)
        rebuilt = {}
        for key, value in row.items():
            require(type(key) is str, "Existing JSON row dictionary keys must be strings")
            if key not in keys:
                require(len(keys) < MAX_SHARED_KEYS, "Bounded audited row schema key count exceeded")
                keys[key] = key
            shared_key = keys[key]
            if key is not shared_key:
                eliminated(key); changed_keys += 1
            if key in METADATA_FIELDS and type(value) is str and len(value) <= MAX_METADATA_STRING_LENGTH:
                pool=values[key]
                if value in pool:
                    replacement=pool[value]
                    if value is not replacement:
                        eliminated(value); changed_metadata += 1
                        value=replacement
                elif len(pool) < MAX_METADATA_VALUES_PER_FIELD:
                    pool[value]=value
            rebuilt[shared_key]=value
        # Reconstruct only one tiny dictionary; preserve the original row and
        # its insertion order for sampler/loader references already held.
        row.clear(); row.update(rebuilt)
        for position, value in enumerate(labels):
            replacement=integers[value]
            if value is not replacement:
                eliminated(value); changed_labels += 1
                if value > 256: used_new_integers.add(value)
                labels[position]=replacement
        text=row.get("text")
        if type(text) is str:
            for field in ("reference", "reference_normalized"):
                value=row.get(field)
                if type(value) is str and value == text and value is not text:
                    eliminated(value); shared_text += 1; row[field]=text
        after=hashlib.sha256(canonical(row)).digest()
        require(before == after and tuple(row) == key_order and id(row) == row_id and id(labels) == label_id
            and row["tr_metadata_labels"] is labels and rows[index] is row,
            "Compaction changed row/label object, value/type or key order; discard partial run")
        after_digest.update(after)
    require(before_digest.digest()==after_digest.digest() and len(rows)==expected_rows,
        "Complete ordered per-row canonical SHA changed")
    new_integer_bytes=sum(sys.getsizeof(integers[value]) for value in used_new_integers)
    return {"kind":"audited_training_row_object_compaction_v1", "passed":True, "rows_verified":expected_rows,
        "same_row_list_and_row_dict_objects":True, "same_label_list_objects":True,
        "all_rows_canonical_equal":True, "all_row_key_orders_equal":True,
        "ordered_row_canonical_sha256_before":before_digest.hexdigest(),
        "ordered_row_canonical_sha256_after":after_digest.hexdigest(), "no_RNG_drawn":True,
        "label_vocab_size":VOCAB_SIZE, "shared_key_count":len(keys),
        "bounded_metadata_value_counts":{field:len(pool) for field,pool in sorted(values.items())},
        "metadata_unique_values_cap_per_field":MAX_METADATA_VALUES_PER_FIELD,
        "metadata_max_string_length":MAX_METADATA_STRING_LENGTH,
        "shared_key_references":changed_keys, "shared_label_integer_references":changed_labels,
        "shared_metadata_string_references":changed_metadata, "shared_equal_text_reference_fields":shared_text,
        "estimated_duplicate_object_bytes_eliminated":max(0,eliminated_reference_bytes-new_integer_bytes),
        "estimated_per_row_distinct_redundant_reference_bytes":eliminated_reference_bytes,
        "new_shared_integer_object_bytes":new_integer_bytes,
        "estimate_may_overcount_preexisting_cross_row_sharing":True,
        "estimated_bytes_are_measured_process_memory":False,
        "row_fields_discarded":False, "arrays_used":False, "model_or_RNG_mutated":False}
