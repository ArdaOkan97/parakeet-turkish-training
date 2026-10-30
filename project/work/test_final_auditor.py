"""CPU regression checks for final-auditor helpers using real frozen baseline data."""
import copy
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0,str(OUTPUT))
import verify_final_results as auditor

folder = OUTPUT / "results/base_v3"
metadata = auditor.json_load(folder / "metadata.json")
summary = auditor.json_load(folder / "summary.json")
check = auditor.Checks()
for language in auditor.LANGUAGES:
    for split in ("validation","test"):
        auditor.verify_split(folder,metadata,summary,language,split,check)
base = Path((ROOT / "work/base_model_path.txt").read_text().strip())
vocabulary = auditor.json_load(OUTPUT / "vocabulary_audit.json")
auditor.verify_export_fingerprint(base,metadata,{"extended_tokenizer_sha256":vocabulary["original_tokenizer_sha256"],
                                  "evaluation_config_name_or_path":str(base)},check,require_merged=False)

fixture = ROOT / "work/verify_final_fixture"
fixture.mkdir(exist_ok=True)
file = fixture / "validation_en.jsonl"
original_rows = auditor.rows_load(folder / file.name)


def negative(label,mutation):
    rows = copy.deepcopy(original_rows)
    fixture_summary = copy.deepcopy(summary)
    mutation(rows,fixture_summary)
    file.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows),encoding="utf-8")
    try:
        auditor.verify_split(fixture,metadata,fixture_summary,"en","validation",auditor.Checks())
    except auditor.AuditError as error:
        return {"test":label,"rejected":True,"failure":str(error)}
    raise AssertionError("Corruption was accepted: "+label)


tests = []
tests.append(negative("duplicate source ID",lambda rows,summary:rows.__setitem__(1,rows[0])))
tests.append(negative("mutated human reference",lambda rows,summary:rows[0].__setitem__("reference","wrong reference")))
tests.append(negative("mutated WAV hash",lambda rows,summary:rows[0].__setitem__("wav_sha256","0"*64)))
tests.append(negative("corrupted ASR counter",lambda rows,summary:summary["validation_en"]["asr"].__setitem__("word_errors",999)))
tests.append(negative("corrupted word correctness",lambda rows,summary:rows[0]["alignment"]["words"][0].__setitem__("correct",not rows[0]["alignment"]["words"][0]["correct"])))
tests.append(negative("corrupted token probability",lambda rows,summary:rows[0]["confidence"]["tokens"][0].__setitem__("blank_probability",1.5)))
tests.append(negative("missing recording",lambda rows,summary:rows.pop()))

tr_source = auditor.rows_load(ROOT / "work/data/validation_tr.jsonl")[0]
tr_prediction = auditor.rows_load(folder / "validation_tr.jsonl")[0]
calibration = {language:summary["calibration_"+language] for language in auditor.LANGUAGES}
temperature = calibration["tr"]["temperature"]
temperature_key = format(float(temperature),".8g")
smoke = {"transcript":tr_prediction["hypothesis"],"confidence_temperature":temperature,"confidence_language":"tr",
         "words":[{"word":word["surface"],"raw_confidence":word["confidence"],"calibrated_confidence":word["confidence_by_temperature"][temperature_key],
                   "conditional_nonblank_confidence":word["conditional_nonblank_confidence"]} for word in tr_prediction["confidence"]["words"]]}
reference = {"source_id":tr_source["source_id"],"reference":tr_source["text"],"audio_filepath":tr_source["audio_filepath"],"wav_sha256":tr_source["wav_sha256"]}
smoke_path = fixture / "transcription_smoke.json"
reference_path = fixture / "transcription_smoke_reference.json"
report = {"artifacts":{"transcription_smoke":str(smoke_path)}}
smoke_path.write_text(json.dumps(smoke,ensure_ascii=False),encoding="utf-8")
reference_path.write_text(json.dumps(reference,ensure_ascii=False),encoding="utf-8")
auditor.verify_transcription_smoke(report,fixture,folder,calibration,check)


def negative_smoke(label,mutation):
    damaged = copy.deepcopy(smoke)
    damaged_reference = copy.deepcopy(reference)
    mutation(damaged,damaged_reference)
    smoke_path.write_text(json.dumps(damaged,ensure_ascii=False),encoding="utf-8")
    reference_path.write_text(json.dumps(damaged_reference,ensure_ascii=False),encoding="utf-8")
    try:
        auditor.verify_transcription_smoke(report,fixture,folder,calibration,auditor.Checks())
    except auditor.AuditError as error:
        return {"test":label,"rejected":True,"failure":str(error)}
    raise AssertionError("Smoke corruption was accepted: "+label)


tests.append(negative_smoke("smoke wrong source",lambda smoke,reference:reference.__setitem__("source_id","wrong")))
tests.append(negative_smoke("smoke different decoded transcript",lambda smoke,reference:smoke.__setitem__("transcript","wrong transcript")))
tests.append(negative_smoke("smoke wrong language",lambda smoke,reference:smoke.__setitem__("confidence_language","en")))
tests.append(negative_smoke("smoke wrong fitted temperature",lambda smoke,reference:smoke.__setitem__("confidence_temperature",temperature+1)))
tests.append(negative_smoke("smoke inconsistent raw confidence",lambda smoke,reference:smoke["words"][0].__setitem__("raw_confidence",smoke["words"][0]["raw_confidence"]*.5)))
tests.append(negative_smoke("smoke inconsistent calibrated confidence",lambda smoke,reference:smoke["words"][0].__setitem__("calibrated_confidence",smoke["words"][0]["calibrated_confidence"]*.5)))
tests.append(negative_smoke("smoke missing decoded word",lambda smoke,reference:smoke["words"].pop()))
result = {"passed":True,"device":"CPU only; no GPU inference","scope":"All900 real original baseline result/provenance/metric/confidence helper checks plus streamed original safetensors model-state/config/tokenizer/generation fingerprint and a synthetic CLI-shape smoke matching the actual first fixed Turkish dev prediction. Fourteen corruption cases fail as intended. No actual exported-model CLI inference was run by this regression; complete final-report integration remains pending actual finalization.",
          "positive_checks":len(check.items),"negative_cases":tests,"verifier_sha256":auditor.file_sha(OUTPUT / "verify_final_results.py")}
(OUTPUT / "verify_final_results_cpu_checks.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
