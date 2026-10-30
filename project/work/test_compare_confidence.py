import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path('outputs/parakeet_turkish').resolve()))
from compare_confidence import compare


root = Path('work/compare_fixture')
for name in ['v2', 'v3']:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for source_id, ref in [('a', 'one two three'), ('b', 'four five')]:
        hyp = 'one two wrong' if name == 'v3' and source_id == 'a' else ref
        probability = 0.3 if name == 'v3' else 0.8
        words = [{'word': word, 'token_indices': [index], 'confidence': probability,
                  'confidence_by_temperature': {'1': probability, '0.1': 0.9}}
                 for index, word in enumerate(hyp.split())]
        conf = {'text': hyp, 'language': 'en', 'words': words,
                'tokens': [{'logprob': math.log(probability), 'piece': word} for word in hyp.split()]}
        rows.append({'source_id': source_id, 'reference': ref, 'hypothesis': hyp,
                     'language': 'en', 'confidence': conf})
    for split in ['validation', 'test']:
        (folder / f'{split}_en.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')

report = compare(root / 'v2', root / 'v3', root / 'report.json', resamples=100)
assert report['shared_correct_words']['shared_correct_word_count'] == 4
assert abs(report['shared_correct_words']['delta_v3_minus_v2']['mean_word_confidence'] + 0.5) < 1e-12
assert report['models']['v2']['english_asr_on_common_test_recordings']['wer'] == 0
assert report['models']['v3']['english_asr_on_common_test_recordings']['wer'] == 0.2
interval = report['shared_correct_word_uncertainty']['delta_v3_minus_v2_ci']['mean_word_confidence']
assert all(abs(value + 0.5) < 1e-12 for value in interval)
assert len(report['v3_correct_words_below_0_5_examples']) == 4
assert report['models']['v3']['development_calibration']['temperature'] == 0.1
print('Paired comparison fixture checks passed.')
