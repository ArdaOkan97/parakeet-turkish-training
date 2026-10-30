"""Independently recompute all saved baseline metrics without GPU use."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0, str(OUTPUT))
from confidence import confidence_report, fit_temperature
from metrics import normalize_text, score_transcriptions

parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="base_v3")
parser.add_argument("--languages", nargs="+", default=["en", "es", "it", "fr", "de", "tr"])
args = parser.parse_args()
folder = OUTPUT / "results" / args.tag
summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
assert metadata["schema_version"] == 1
for filename, expected in metadata["model_fingerprint"]["identity"]["evaluation_code_sha256"].items():
    assert hashlib.sha256((OUTPUT / filename).read_bytes()).hexdigest() == expected, ("evaluation_code_changed", filename)

report = []
for language in args.languages:
    all_rows = {}
    for split, size in [("validation", 50), ("test", 100)]:
        key = f"{split}_{language}"
        manifest_path = ROOT / f"work/data/{key}.jsonl"
        manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
        expected = {row["source_id"]: row for row in manifest}
        assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == metadata["manifests"][key]["file_sha256"]
        rows = [json.loads(line) for line in (folder / f"{key}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == size and len({r["source_id"] for r in rows}) == size
        assert set(expected) == {r["source_id"] for r in rows}
        assert [r["source_id"] for r in manifest] == metadata["manifests"][key]["source_ids"]
        for row in rows:
            source = expected[row["source_id"]]
            assert row["reference"] == source["text"] and row["sentence_id"] == source["sentence_id"]
            assert row["dataset_revision"] == source["dataset_revision"] and row["wav_sha256"] == source["wav_sha256"]
            assert row["language"] == language and row["split"] == split
            assert row["hypothesis"].strip() == row["confidence"]["text"].strip()
            assert normalize_text(row["hypothesis"], language).split() == [w["word"] for w in row["confidence"]["words"]]
            assert all("logprob" in t and "duration_logprob" in t for t in row["confidence"]["tokens"])
        metrics = score_transcriptions(rows, language=language)
        metrics.pop("details")
        assert metrics == summary[key]["asr"], key
        raw = confidence_report([r["alignment"] for r in rows])
        assert raw == summary[key]["raw_confidence"], key
        all_rows[split] = rows
        report.append({"split": key, **metrics})
    dev_fit = fit_temperature([r["alignment"] for r in all_rows["validation"]])
    assert dev_fit == summary[f"calibration_{language}"], language
    calibrated_test = confidence_report([r["alignment"] for r in all_rows["test"]], dev_fit["temperature"])
    assert calibrated_test == summary[f"test_{language}"]["calibrated_confidence"], language

result = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "device": "CPU only", "tag": args.tag,
          "model_fingerprint_sha256": metadata["model_fingerprint"]["sha256"],
          "checks": ["Complete fixed development/test source-ID sets and immutable manifest hashes", "Source provenance and references match every prediction", "All metric summaries independently recomputed exactly", "Every decoded normalized word matches scored confidence units", "Raw/calibrated confidence summaries independently reproduced", "Every temperature fit independently reproduced from development rows only", "Evaluation code hashes still match the recorded baseline fingerprint"],
          "utterances": sum(item["utterances"] for item in report), "results": report, "passed": True}
(OUTPUT / f"{args.tag}_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({"tag": args.tag, "utterances": result["utterances"], "passed": True}, indent=2))
