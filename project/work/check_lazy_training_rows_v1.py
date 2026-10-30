"""Small stdlib carrier/consumer guards; no actual corpus/model/Trackio reads."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "outputs/parakeet_turkish/lazy_training_rows_v1.py"


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    if args.output.exists(): raise ValueError("Fresh fixture output required")
    spec = importlib.util.spec_from_file_location("lazy_fixture", SOURCE)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    checks = []; rejects = []
    def check(ok, name):
        if not ok: raise AssertionError(name)
        checks.append(name)
    def reject(fn, name):
        try: fn()
        except (ValueError, IndexError, KeyError, json.JSONDecodeError): rejects.append(name); return
        raise AssertionError("Accepted: " + name)
    with tempfile.TemporaryDirectory(prefix="lazy_rows_fixture_", dir=ROOT / "work") as temp:
        base = Path(temp)
        rows = [{"source_id": "sid" + str(i), "dataset": "google/fleurs" if i % 2 else "ysdede/commonvoice_17_tr_fixed",
                 "language": "tr", "text": "Türkçe İı " + str(i), "duration": 1.125 + i/16,
                 "encoder_frames": i + 2, "target_tokens": i + 1, "tr_metadata_labels": [8193, 21, i],
                 "null": None, "nested": {"flag": True, "order": [1, 0]}} for i in range(10)]
        store = m.prepare_indexed_rows(iter(rows), base / "carrier", expected_count=10, provenance={"fixture_only": True}, cache_size=3)
        check(list(store) == rows, "all values/types/order preserved")
        check(list(store.source_ids) == [r["source_id"] for r in rows], "source-ID sequence is original order")
        check(store.index_by_source_id["sid7"] == 7, "source ID lookup")
        check(store[0:10:2].indices == range(0, 10, 2) and list(store[0:10:2]) == rows[::2], "slice is index-only view")
        check(store[-1] == rows[-1], "negative row index")
        check(store.cache_rows <= 3, "cache cap")
        altered = store[3]; altered["text"] = "changed"; altered["tr_metadata_labels"].append(10)
        check(store[3] == rows[3], "returned dictionary cannot poison carrier/cache")
        check([store.groups[i] for i in range(10)] == ["cv" if i % 2 == 0 else "fleurs" for i in range(10)], "mmap source groups")
        check(store.length_key(3) == (2, 2, 5, 4, "sid3"), "exact reservoir length key")
        ids = hashlib.sha256("\n".join(sorted(r["source_id"] for r in rows)).encode()).hexdigest()
        fp = hashlib.sha256(b"".join(m.canonical({k: r.get(k) for k in m.FINGERPRINT_KEYS}) + b"\n"
                                    for r in sorted(rows, key=lambda r: r["source_id"]))).hexdigest()
        check(store.ids_sha == ids and store.fingerprint == fp, "exact legacy sorted ID/fingerprint conventions")
        check(store.ordered_rows_sha == hashlib.sha256(b"".join(m.canonical(r) + b"\n" for r in rows)).hexdigest(), "exact ordered-row digest")
        # Same RNG calls and reservoir lengths; row storage cannot change sampling.
        indices = list(range(10)); resident = rows.copy(); a, b = random.Random(4), random.Random(4)
        a.shuffle(indices); b.shuffle(resident)
        for start in range(0, 10, 3):
            indices[start:start+3] = sorted(indices[start:start+3], key=store.length_key)
            resident[start:start+3] = sorted(resident[start:start+3], key=lambda r: (r["encoder_frames"].bit_length()-1,
                r["target_tokens"].bit_length()-1, r["encoder_frames"], r["target_tokens"], r["source_id"]))
        check([store.source_ids[i] for i in indices] == [r["source_id"] for r in resident] and a.getstate() == b.getstate(), "identical shuffle/reservoir order and RNG")
        reject(lambda: store[10], "out of bounds")
        reject(lambda: store.index_by_source_id["absent"], "unknown source")
        store.close()
        opened = m.IndexedTrainingRows(base / "carrier", cache_size=0)
        check(opened[2] == rows[2] and opened.cache_rows == 0, "verified immutable reopening")
        opened.close()
        with (base / "carrier/rows.jsonl").open("ab") as out: out.write(b"\n")
        reject(lambda: m.IndexedTrainingRows(base / "carrier"), "changed carrier bytes")
        reject(lambda: m.prepare_indexed_rows(iter(rows[:-1]), base / "short", expected_count=10, provenance={}), "short full pool")
        check(not (base / "short/index_provenance.json").exists(), "failed staging never publishes complete receipt")
        reject(lambda: m.prepare_indexed_rows(iter(rows), base / "extra", expected_count=9, provenance={}), "extra full pool")
        reject(lambda: m.prepare_indexed_rows(iter(rows), base / "bad-admission", expected_count=10, provenance={},
            validate_result=lambda result: m.require(False, "reject")), "admission fails before completed publication")
        check(not (base / "bad-admission/index_provenance.json").exists(), "invalid admission has no complete receipt")
        proof = base / "proof.json"; proof.write_text(json.dumps({"value": 6, "target": rows, "tail": True}, ensure_ascii=False), encoding="utf-8")
        check(list(m.iter_array_field(proof, "target")) == rows, "stream nested parent evidence array")
        proof.write_text('{"target":[{"x":1}', encoding="utf-8")
        reject(lambda: list(m.iter_array_field(proof, "target")), "truncated proof")
        proof.write_text('{"target":[]} trailing', encoding="utf-8")
        reject(lambda: list(m.iter_array_field(proof, "target")), "trailing proof junk")
    check("torch" not in sys.modules, "no Torch imported")
    result = {"schema": 1, "kind": "indexed_TRAIN_rows_tiny_stdlib_verification_v1", "passed": True,
              "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(), "checks": checks, "rejections": rejects,
              "actual_corpus_or_checkpoint_read": False, "Torch_or_GPU_used": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "checks": len(checks), "rejections": len(rejects)}))


if __name__ == "__main__": main()
