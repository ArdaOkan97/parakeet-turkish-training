"""Re-score the three immutable saved DEV prediction sets without model/audio use."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(OUTPUT))
from metrics import score_transcriptions

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]

report_path = OUTPUT / 'HF_KD10_DEV5000_error_evidence_v1.json'
report = json.loads(report_path.read_text(encoding='utf-8'))
assert sha(report_path) == '8e94d5b23a1f826274743b0f21097ff16c18f4e02bf39575753400ba90452376'
manifest = ROOT / 'work/data/validation_tr.jsonl'
assert sha(manifest) == report['fixed_DEV_manifest_sha256']
references = rows(manifest)
assert len(references) == 50
reference_map = {row['source_id']:row for row in references}
assert len(reference_map) == 50
checks = {}
for step in ('0', '2500', '5000'):
    model = report['models'][step]
    path = Path(model['predictions_path']).resolve()
    assert path.is_relative_to(OUTPUT) and path.name == 'validation_tr.jsonl'
    assert sha(path) == model['predictions_sha256']
    predictions = rows(path)
    assert len(predictions) == 50 and {row['source_id'] for row in predictions} == set(reference_map)
    assert all(row['reference'] == reference_map[row['source_id']]['text'] for row in predictions)
    score = score_transcriptions(predictions, language='tr')
    expected = model['standard_metrics']
    for key in ('wer', 'cer', 'word_errors', 'reference_words', 'character_errors', 'reference_characters', 'utterances'):
        assert score[key] == expected[key], (step, key)
    alignment = model['saved_word_alignment_counts']
    assert sum(alignment[key] for key in ('substitutions', 'deletions', 'insertions')) == score['word_errors']
    checks[step] = {key:score[key] for key in expected if key != 'normalization'}

target = OUTPUT / 'HF_KD10_DEV5000_error_ROOT_rescore_v1.json'
result = {'kind':'ROOT_saved_DEV_error_rescore_v1', 'passed':True,
          'diagnosis_path':str(report_path), 'diagnosis_sha256':sha(report_path),
          'fixed_DEV_manifest_sha256':sha(manifest), 'scores':checks,
          'scope':'Same50 DEV references and independent canonical scorer recalculation; S/D/I sum checked, alignment tie categories and boundary oracle not independently recomputed',
          'model_audio_GPU_TEST_used':False, 'script_sha256':sha(__file__)}
with target.open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps({'passed':True, 'scores':checks, 'receipt':str(target)}))
