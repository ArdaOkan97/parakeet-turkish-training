"""Small stdlib-only BPE partition/sidecar fixture. All new output is on E:."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "outputs/parakeet_turkish"))
import bpe_full_decoder_optimizer_v1 as opt
import bpe_training_label_catalog_v1 as labels


def main(output):
    started = time.perf_counter()
    output = labels.scoped(output)
    assert not output.exists(), "Fresh E: fixture output required"
    output.mkdir(parents=True)
    checks, rejects = [], []

    def check(name, condition):
        assert condition, name
        checks.append(name)

    def reject(name, function):
        try:
            function()
        except (ValueError, AssertionError, sqlite3.IntegrityError):
            rejects.append(name)
        else:
            raise AssertionError("Expected rejection: " + name)

    heads = ("encoder_projector.weight", "encoder_projector.bias", "decoder.embedding.weight",
             "decoder.lstm.weight_ih_l0", "decoder.lstm.weight_hh_l0", "decoder.lstm.bias_ih_l0", "decoder.lstm.bias_hh_l0",
             "decoder.lstm.weight_ih_l1", "decoder.lstm.weight_hh_l1", "decoder.lstm.bias_ih_l1", "decoder.lstm.bias_hh_l1",
             "decoder.decoder_projector.weight", "decoder.decoder_projector.bias", "joint.head.weight", "joint.head.bias")
    adapters = tuple(f"encoder.synthetic_{i}.lora_{letter}" for i in range(96) for letter in ("A", "B"))
    # Deliberately use encoder-first dictionary insertion order, unlike Adam group order.
    names = adapters + heads
    for vocabulary, fresh_count in ((8199, 0), (8711, 3)):
        actual_heads, actual_lora, mature, fresh = opt.partition_plan(names, vocabulary)
        check(f"{vocabulary}: actual15 head and192 frozen names", actual_heads == heads and actual_lora == adapters)
        check(f"{vocabulary}: disjoint17 mature partitions", len(mature) == 17)
        check(f"{vocabulary}: fresh partition count", len(fresh) == fresh_count)
        check(f"{vocabulary}: source IDs follow actual head Adam order",
              all(part["source_id"] == heads.index(part["name"]) for part in mature))
        for name in opt.TABLES:
            selected = [part for part in (*mature, *fresh) if part["name"] == name]
            visits = [index for part in selected for index in range(*part["target_rows"])]
            count = vocabulary + (0 if name == opt.TABLES[0] else 5)
            check(f"{vocabulary}: {name} covered once", sorted(visits) == list(range(count)))
            check(f"{vocabulary}: {name} original blank8192 and six8193:8199 mature",
                  any(part["part"] == "lexical" and part["source_rows"] == [0, 8199] for part in selected))
        check(f"{vocabulary}: duration values explicitly remap",
              all(p["source_rows"] == [8199, 8204] and p["target_rows"] == [vocabulary, vocabulary + 5]
                  for p in mature if p["part"] == "duration"))
    reject("unsupported vocabulary", lambda: opt.partition_plan(names, 9000))
    reject("missing source tensor", lambda: opt.partition_plan(names[:-1], 8711))
    reject("duplicate source tensor", lambda: opt.partition_plan(names[:-1] + (names[0],), 8711))

    class Encoding:
        def __init__(self, values): self.ids = values

    class Tokenizer:
        def __init__(self, candidate=False, empty=False, mismatch=False):
            self.candidate, self.empty, self.mismatch = candidate, empty, mismatch
        def encode(self, text, add_special_tokens=True):
            if self.empty: return Encoding([])
            i = int(text.removeprefix("demo"))
            return Encoding(([8199] if self.candidate else [12, 13]) + [20 + i])
        def decode(self, values):
            return "mismatch" if self.mismatch else "demo" + str(values[-1] - 20)

    class Rows:
        def __init__(self):
            self.values = [dict(source_id=f"synthetic_TRAIN_{i}", language="tr", text=f"demo{i}",
                                training_input_type="TR_frame_target_metadata_v1", tr_metadata_labels=[12, 13, 20+i],
                                target_tokens=3, encoder_frames=4+i, duration=1., tr_frame_metadata_sha256="fixture")
                           for i in range(8)]
            self.fingerprint = "synthetic_eight_row_fingerprint"
            self.ids_sha = "synthetic_eight_ordered_IDs"
            self.receipt = dict(complete=True, rows=8, provenance=dict(training_admission_complete=True, all_rows_verified=True))
        def __len__(self): return len(self.values)
        def __getitem__(self, i): return self.values[i]

    rows = Rows()
    records = []
    for name in ("source_index.json", "baseline_tokenizer.json", "candidate_tokenizer.json"):
        p = output / name
        p.write_text(json.dumps({"synthetic_fixture_only": name}), encoding="utf-8")
        records.append(labels.record(p))
    kwargs = dict(source_catalog_record=records[0], baseline_tokenizer_record=records[1], candidate_tokenizer_record=records[2])
    evidence = labels.prepare_label_catalog(rows, output / "valid", Tokenizer(), Tokenizer(True), **kwargs)
    check("all8 source rows admitted once", evidence["rows"] == 8 and evidence["complete"])
    check("source24 candidate16 target counts", evidence["source_target_tokens"] == 24 and evidence["candidate_target_tokens"] == 16)
    check("sampling source T/U/IDs unchanged", evidence["source_sampling_T_U_IDs_order_RNG_mutated"] is False)
    carrier = labels.BPETrainingLabels(output / "valid", rows)
    selected = carrier.rewrite_batch(rows.values)
    check("physical8TR labels use candidate only", all(x["tr_metadata_labels"] == [8199, 20+i] for i,x in enumerate(selected)))
    check("all source dictionaries and original labels untouched", all(x["target_tokens"] == 3 for x in rows.values))
    check("model and sampling U are separate", all(x["target_tokens"] == 2 and x["sampling_target_tokens"] == 3 for x in selected))
    replay_en = dict(source_id="synthetic_RP_EN", language="en", cache_path="unchanged")
    replay_pt = dict(source_id="synthetic_RP_PT", language="pt", cache_path="unchanged")
    mixed = carrier.rewrite_batch(rows.values[:6] + [replay_en, replay_pt])
    check("6TR+oldfive+PT replay objects unchanged", mixed[-2] is replay_en and mixed[-1] is replay_pt)
    check("bounded label cache", len(carrier.cache) <= 128)
    reject("physical batch larger than8", lambda: carrier.rewrite_batch(rows.values + rows.values[:1]))
    changed = dict(rows[0], target_tokens=4)
    reject("sampling target metadata mutation", lambda: carrier.rewrite_batch([changed]))
    carrier.close()
    reject("empty candidate targets", lambda: labels.prepare_label_catalog(rows, output / "empty", Tokenizer(), Tokenizer(True, empty=True), **kwargs))
    reject("candidate changed decoded text", lambda: labels.prepare_label_catalog(rows, output / "mismatch", Tokenizer(), Tokenizer(True, mismatch=True), **kwargs))
    reject("reuse committed target directory", lambda: labels.prepare_label_catalog(rows, output / "valid", Tokenizer(), Tokenizer(True), **kwargs))
    reject("scratch escape to C project", lambda: labels.scoped(ROOT / "forbidden_scratch"))
    database = output / "valid" / "labels.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE labels SET text_sha='corrupt' WHERE ordinal=0")
    reject("committed sidecar hash corruption", lambda: labels.BPETrainingLabels(output / "valid", rows))
    check("no Torch or model imported", "torch" not in sys.modules and "transformers" not in sys.modules)
    result = dict(kind="BPE_full_decoder_small_partition_and_label_sidecar_CPU_verification_v1", passed=True,
                  checks=checks, positive_checks=len(checks), rejections=rejects, rejected_cases=len(rejects),
                  elapsed_seconds=time.perf_counter()-started, synthetic_rows=8, real_corpus_or_audio_read=False,
                  actual_Torch_storage_alias_or_optimizer_numerics_executed=False,
                  GPU_or_native_training_executed=False,
                  required_future_runtime_proof="Actual view aliases, named source Adam transfer, source350, frozen192 digest, native backward and old/new row deltas remain GPU admission requirements.",
                  source_files={name:labels.record(ROOT/'outputs/parakeet_turkish'/name) for name in
                                ('bpe_full_decoder_optimizer_v1.py','bpe_training_label_catalog_v1.py')})
    report = output / "verification.json"
    labels.write_json(report, result)
    print(json.dumps(dict(passed=True, positive_checks=len(checks), rejections=len(rejects), report=labels.record(report))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    main(parser.parse_args().output_directory)
