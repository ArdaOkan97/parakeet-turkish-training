"""Recompute a completed LoRA development evaluation and retention gates on CPU."""
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0, str(OUTPUT))
from confidence import confidence_report, fit_temperature
from metrics import normalize_text, score_transcriptions, paired_bootstrap_delta


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


parser = argparse.ArgumentParser()
parser.add_argument("--step", type=int, default=100)
args = parser.parse_args()
config_path = OUTPUT / "run_config.json"
config = load(config_path)
tag = f"{config['run_id']}_step_{args.step}"
folder = OUTPUT / "results" / tag
baseline_folder = OUTPUT / "results/base_v3"
summary = load(folder / "summary.json")
baseline_summary = load(baseline_folder / "summary.json")
metadata = load(folder / "metadata.json")
baseline_metadata = load(baseline_folder / "metadata.json")
history_path = OUTPUT / "training_history.json"
history = load(history_path)
assert config["run_id"] == "lora_20261010_01"
assert len(history) == history[-1]["step"] and [row["step"] for row in history] == list(range(1,len(history)+1))
assert all(math.isfinite(row[key]) for row in history for key in ("tdt_loss", "replay_kl", "replay_duration_kl", "replay_feature_distillation", "gradient_norm"))
record = next(row for row in history if row["step"] == args.step)
assert "validation" in record
assert metadata["schema_version"] == 1
assert metadata["model_fingerprint"]["sha256"] != baseline_metadata["model_fingerprint"]["sha256"]
identity = metadata["model_fingerprint"]["identity"]
for name, expected in identity["evaluation_code_sha256"].items():
    assert sha(OUTPUT / name) == expected, name
for name, expected in config["code_sha256"].items():
    assert sha(OUTPUT / name) == expected, name
for name in ("metrics.py", "confidence.py", "runtime.py"):
    assert identity["evaluation_code_sha256"][name] == baseline_metadata["model_fingerprint"]["identity"]["evaluation_code_sha256"][name]
assert identity["inference_dtype"] == baseline_metadata["model_fingerprint"]["identity"]["inference_dtype"] == "CUDA bf16 autocast"
assert not list(folder.glob("test_*.jsonl"))
assert set(metadata["manifests"]) == {"validation_"+language for language in ("en", "es", "it", "fr", "de", "tr")}
report = {}
input_files = [config_path, folder / "metadata.json", folder / "summary.json", baseline_folder / "metadata.json", baseline_folder / "summary.json"]
for language in ("en", "es", "it", "fr", "de", "tr"):
    key = "validation_"+language
    path = ROOT / "work/data" / (key+".jsonl")
    manifest = read_rows(path)
    predictions_path = folder / (key+".jsonl")
    predictions = read_rows(predictions_path)
    baseline_path = baseline_folder / (key+".jsonl")
    baseline_predictions = read_rows(baseline_path)
    input_files.extend([path, predictions_path, baseline_path])
    expected = {row["source_id"]: row for row in manifest}
    assert len(manifest) == len(predictions) == len(baseline_predictions) == 50
    assert len({row["source_id"] for row in predictions}) == 50
    assert [row["source_id"] for row in predictions] == [row["source_id"] for row in baseline_predictions] == [row["source_id"] for row in manifest]
    assert metadata["manifests"][key] == baseline_metadata["manifests"][key]
    assert sha(path) == metadata["manifests"][key]["file_sha256"]
    assert metadata["manifests"][key]["source_ids"] == [row["source_id"] for row in manifest]
    assert metadata["manifests"][key]["utterances"] == 50 and metadata["manifests"][key]["limit"] == 0
    for row in predictions:
        source = expected[row["source_id"]]
        assert row["reference"] == source["text"]
        assert source["dataset_split"] == "validation"
        assert all(row[name] == source[name] for name in ("sentence_id", "dataset_revision", "audio_sha256", "wav_sha256", "duration", "language"))
        assert row["language"] == language and row["split"] == "validation"
        assert row["hypothesis"].strip() == row["confidence"]["text"].strip()
        assert normalize_text(row["hypothesis"], language).split() == [word["word"] for word in row["confidence"]["words"]]
        alignment = row["alignment"]
        assert alignment["reference_word_count"] == len(normalize_text(row["reference"], language).split())
        assert alignment["substitutions"]+alignment["insertions"]+alignment["deletions"] == score_transcriptions([row], language=language)["word_errors"]
    measured = score_transcriptions(predictions, language=language)
    measured.pop("details")
    baseline = score_transcriptions(baseline_predictions, language=language)
    baseline.pop("details")
    assert measured == summary[key]["asr"] and baseline == baseline_summary[key]["asr"]
    assert confidence_report([row["alignment"] for row in predictions]) == summary[key]["raw_confidence"]
    assert fit_temperature([row["alignment"] for row in predictions]) == summary["calibration_"+language]
    delta = measured["wer"]-baseline["wer"]
    tolerance = max(.005,.1*baseline["wer"])
    bootstrap = paired_bootstrap_delta([row["reference"] for row in predictions], [row["hypothesis"] for row in baseline_predictions], [row["hypothesis"] for row in predictions], language=language, groups=[row["sentence_id"] for row in predictions])
    report[language] = {"baseline": baseline, "candidate": measured, "wer_delta_percentage_points": delta*100, "cer_delta_percentage_points": 100*(measured["cer"]-baseline["cer"]),
                        "retention_tolerance_percentage_points": None if language=="tr" else 100*tolerance,
                        "retention_ceiling_wer": None if language=="tr" else baseline["wer"]+tolerance,
                        "retention_gate_passed": None if language=="tr" else delta <= tolerance,
                        "alignment_counts": {name: sum(row["alignment"][name] for row in predictions) for name in ("hits", "substitutions", "insertions", "deletions")},
                        "baseline_alignment_counts": {name: sum(row["alignment"][name] for row in baseline_predictions) for name in ("hits", "substitutions", "insertions", "deletions")},
                        "exploratory_dev_paired_sentence_cluster_bootstrap": bootstrap}
    if language != "tr":
        assert delta == record["validation"]["retention_wer_deltas"][language]
retention_passed = all(report[language]["retention_gate_passed"] for language in report if language != "tr")
assert retention_passed == record["validation"]["retention_gate_passed"]
assert report["tr"]["candidate"]["wer"] == record["validation"]["turkish_wer"]
result = {"verified_at_utc":datetime.now(timezone.utc).isoformat(),"passed":True,"device":"CPU only; no model/GPU loaded","tag":tag,"run_id":config["run_id"],"step":args.step,
          "model_fingerprint_sha256":metadata["model_fingerprint"]["sha256"],"retention_gate_passed":retention_passed,
          "turkish_dev_improved":report["tr"]["candidate"]["wer"] < report["tr"]["baseline"]["wer"],
          "training_snapshot":{"steps":len(history),"all_recorded_losses_kls_gradnorms_finite":True,"step_under_audit":record,"last_step":history[-1]},
          "input_file_sha256":{str(path):sha(path) for path in input_files},"languages":report,
          "checks":["Every fixed development set has all50 original paired source IDs, identical manifest SHA256/provenance/reference and no test prediction files","ASR corpus edit counts, raw confidence and development-only temperature fit independently reproduced","Every retention delta and max(.5pp,10%relative) gate independently reproduced","All current run/evaluator code hashes match recorded provenance; normalization/confidence/runtime/AMP match unchanged baseline","Training history has contiguous steps and finite recorded losses/KLs/gradient norms"],
          "notes":["Development is used for model selection; exploratory paired bootstrap intervals on this reused small set are not heldout quality evidence.","Recorded model fingerprint cannot be independently recomputed without reconstructing the exact model; this CPU audit validates saved-result/manifest/code provenance and does not rerun GPU inference.","A passed first development gate supports a provisional selection only; merged standard-HF export/save-load and separate final heldout ASR/confidence remain outstanding."]}
destination = OUTPUT / (tag+"_verification.json")
destination.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"passed":True,"tag":tag,"retention_gate_passed":retention_passed,"languages":{language:{key:value for key,value in results.items() if key not in ("baseline_alignment_counts", "exploratory_dev_paired_sentence_cluster_bootstrap")} for language,results in report.items()}},indent=2))
