"""CPU checks for deterministic variants and reference-word pairing."""
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'outputs/parakeet_turkish'))
import confidence_robustness as check

# Signal-level check independent of inference; ensure original samples survive.
audio = (0.15 * np.sin(2 * np.pi * 317 * np.arange(16000) / 16000)).astype(np.float32)
original = audio.copy()
a = check.perturb(audio, 'fixture', 20261010)
b = check.perturb(audio, 'fixture', 20261010)
c = check.perturb(audio, 'other_recording', 20261010)
assert np.array_equal(audio, original)
assert all(np.array_equal(a[k][0], b[k][0]) for k in check.VARIANTS)
assert not np.array_equal(a['noise_snr_50_db'][0], c['noise_snr_50_db'][0])
gain = a['gain_minus_0_2_db'][0].astype(float)
noise = a['noise_snr_50_db'][0].astype(float) - original.astype(float)
assert abs(20 * math.log10(np.linalg.norm(gain) / np.linalg.norm(original.astype(float))) + 0.2) < 1e-5
assert abs(20 * math.log10(np.linalg.norm(original.astype(float)) / np.linalg.norm(noise)) - 50) < 1e-5
assert a['silence_20_ms_each_side'][0].size == audio.size + 640
assert not a['silence_20_ms_each_side'][0][:320].any()
assert not a['silence_20_ms_each_side'][0][-320:].any()
assert np.array_equal(a['silence_20_ms_each_side'][0][320:-320], original)

# Ten real prepared development waveforms: no Torch or CUDA imports.
rows, _, plan = check.prepare(root / 'work/data/validation_en.jsonl', 10, 20261010)
assert len(rows) == 10 and 'torch' not in sys.modules

# Real baseline alignments supply realistic labels. All copied variants retain
# transcripts; modify one shared-correct word and exclude another reference
# position to test exact conditioning rather than order-based word matching.
saved = [json.loads(line) for line in (root / 'outputs/parakeet_turkish/results/base_v3/validation_en.jsonl').read_text(encoding='utf-8').splitlines()][:10]
records = []
total_correct = sum(sum(w['correct'] for w in r['alignment']['words']) for r in saved)
first = next(w for w in saved[0]['alignment']['words'] if w['correct'])
target_index = first['reference_word_index']
for row in saved:
    for variant in check.VARIANTS:
        item = copy.deepcopy(row)
        item['variant'] = variant
        if variant != 'clean' and item['source_id'] == saved[0]['source_id']:
            word = next(w for w in item['alignment']['words'] if w['reference_word_index'] == target_index)
            word['confidence'] -= 0.125
            word['conditional_nonblank_confidence'] -= 0.0625
        records.append(item)
summary = check.summarize(records)
for value in summary['perturbations'].values():
    assert value['raw_transcripts_changed'] == value['normalized_transcripts_changed'] == 0
    assert value['wer_delta_percentage_points'] == 0
    assert value['shared_correct_word_confidence_delta']['count'] == total_correct
    assert abs(value['shared_correct_word_confidence_delta']['mean'] + 0.125 / total_correct) < 1e-14
    assert abs(value['shared_correct_word_confidence_delta']['absolute_max'] - 0.125) < 1e-14
copied = copy.deepcopy(summary)
for value in copied['perturbations'].values():
    for word in value['paired_words']:
        word['confidence_delta'] *= 0.5
        word['conditional_nonblank_delta'] *= 0.5
pairwise = check.compare_sensitivity({'a': {'summary': summary}, 'b': {'summary': copied}})
for value in pairwise['a_vs_b'].values():
    assert value['shared_correct_words_all_four_decodes'] == total_correct
    assert abs(value['right_minus_left_absolute_raw_change']['mean'] + 0.0625 / total_correct) < 1e-14
error_fixture = copy.deepcopy(records[1])
first_correct = next(w for w in error_fixture['alignment']['words'] if w['correct'])
first_correct['correct'] = False
first_correct['reference_word_index'] = None
assert len(check.paired_word_deltas(records[0], error_fixture)) == sum(w['correct'] for w in records[0]['alignment']['words']) - 1
audit = {'status': 'passed', 'gpu_used': False, 'torch_imported': 'torch' in sys.modules,
    'actual_prepared_recordings': 10, 'checks': ['original waveform unchanged', 'noise determinism and source-specific seed',
    'independent amplitude gain and noise SNR', 'padding preserves every original sample',
    'prepared WAV hash verification', 'actual development split selected', 'reference-index pairing',
    'ASR and transcript separation', 'exact expected confidence deltas', 'matched model-pair sensitivity',
    'words becoming incorrect excluded from confidence pair']}
check.write_json(root / 'outputs/parakeet_turkish/confidence_robustness_cpu_verification.json', audit)
print(json.dumps(audit, indent=2))
