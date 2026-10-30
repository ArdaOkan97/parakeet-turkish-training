import json
import math
import random
import unicodedata
from pathlib import Path

import jiwer


root = Path('outputs/parakeet_turkish')
report = json.loads((root / 'english_confidence_comparison.json').read_text(encoding='utf-8'))
rows = {}
checks = []


def normalize(text):
    text = unicodedata.normalize('NFC', text).lower().replace('i\u0307', 'i')
    return ' '.join(''.join(' ' if unicodedata.category(c)[0] in {'Z', 'C'}
                           else '' if unicodedata.category(c)[0] == 'P' else c for c in text).split())


def close(description, actual, expected, tolerance=1e-11):
    if abs(actual - expected) > tolerance:
        raise AssertionError((description, actual, expected))
    checks.append({'check': description, 'difference': abs(actual - expected), 'passed': True})


def reliability(words, temperature):
    key = format(temperature, '.8g')
    p = [word['confidence_by_temperature'][key] for word in words]
    y = [int(word['correct']) for word in words]
    positive = [score for score, label in zip(p, y) if label]
    negative = [score for score, label in zip(p, y) if not label]
    ece = 0.0
    for bucket in range(10):
        selected = [(score, label) for score, label in zip(p, y) if min(9, int(score * 10)) == bucket]
        if selected:
            ece += abs(sum(score for score, _ in selected) - sum(label for _, label in selected)) / len(words)
    return {'brier': sum((score - label) ** 2 for score, label in zip(p, y)) / len(words),
            'ece': ece,
            'correctness_auroc': sum((a > b) + 0.5 * (a == b) for a in positive for b in negative) / (len(positive) * len(negative)),
            'correct_word_mean_confidence': sum(positive) / len(positive)}


def distance(a, b):
    previous = list(range(len(b) + 1))
    for index, token in enumerate(a, 1):
        current = [index]
        for other_index, other in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[other_index] + 1, previous[other_index - 1] + (token != other)))
        previous = current
    return previous[-1]


for name in ['v2', 'v3']:
    rows[name] = {}
    for split in ['validation', 'test']:
        loaded = [json.loads(line) for line in (root / f'results/base_{name}/{split}_en.jsonl').read_text(encoding='utf-8').splitlines() if line]
        rows[name][split] = {str(row['source_id']): row for row in loaded}
        assert len(loaded) == len(rows[name][split])
        for row in loaded:
            ref = normalize(row['reference']).split()
            hyp = normalize(row['hypothesis']).split()
            assert hyp == [word['word'] for word in row['alignment']['words']]
            expected = jiwer.process_words(' '.join(ref), ' '.join(hyp))
            labels = [False] * len(hyp)
            for chunk in expected.alignments[0]:
                if chunk.type == 'equal':
                    for index in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                        labels[index] = True
            assert labels == [word['correct'] for word in row['alignment']['words']]
            for word in row['alignment']['words']:
                if word['correct']:
                    assert ref[word['reference_word_index']] == word['word']
                else:
                    assert word['reference_word_index'] is None
    dev_words = [word for row in rows[name]['validation'].values() for word in row['alignment']['words']]
    candidates = [(float(key), reliability(dev_words, float(key))['brier']) for key in dev_words[0]['confidence_by_temperature']]
    best = min(candidates, key=lambda pair: (pair[1], abs(math.log(pair[0]))))[0]
    close(name + ' dev-only minimum-Brier temperature', best, report['models'][name]['development_calibration']['temperature'])
    test_words = [word for row in rows[name]['test'].values() for word in row['alignment']['words']]
    for field, temperature in [('raw_test_word_confidence', 1.0), ('calibrated_test_word_confidence', best)]:
        for metric, value in reliability(test_words, temperature).items():
            close(name + ' ' + field + ' ' + metric, value, report['models'][name][field][metric])

for split in ['validation', 'test']:
    assert rows['v2'][split].keys() == rows['v3'][split].keys()
assert set(rows['v2']['validation']).isdisjoint(rows['v2']['test'])
ids = sorted(rows['v2']['test'])
groups = []
for source_id in ids:
    assert normalize(rows['v2']['test'][source_id]['reference']) == normalize(rows['v3']['test'][source_id]['reference'])
    paired = {}
    for name in ['v2', 'v3']:
        row = rows[name]['test'][source_id]
        paired[name] = {word['reference_word_index']: word for word in row['alignment']['words'] if word['correct']}
    indices = paired['v2'].keys() & paired['v3'].keys()
    group = {'count': len(indices)}
    for name in ['v2', 'v3']:
        selected = [paired[name][index] for index in indices]
        group[name] = {'raw_sum': sum(word['confidence'] for word in selected),
                       'conditional_sum': sum(word['conditional_nonblank_confidence'] for word in selected),
                       'token_count': sum(len(word['token_indices']) for word in selected)}
    reference = normalize(rows['v2']['test'][source_id]['reference']).split()
    group['reference_count'] = len(reference)
    group['word_error_difference'] = distance(reference, normalize(rows['v3']['test'][source_id]['hypothesis']).split()) - distance(reference, normalize(rows['v2']['test'][source_id]['hypothesis']).split())
    groups.append(group)


def summaries(indices):
    selected = [groups[index] for index in indices]
    count = sum(group['count'] for group in selected)
    return {
        'mean_word_confidence': sum(group['v3']['raw_sum'] - group['v2']['raw_sum'] for group in selected) / count,
        'mean_conditional_nonblank_word_confidence': sum(group['v3']['conditional_sum'] - group['v2']['conditional_sum'] for group in selected) / count,
        'mean_tokens_per_word': sum(group['v3']['token_count'] - group['v2']['token_count'] for group in selected) / count,
        'wer_delta_pp': 100 * sum(group['word_error_difference'] for group in selected) / sum(group['reference_count'] for group in selected),
    }


assert sum(group['count'] for group in groups) == report['shared_correct_words']['shared_correct_word_count']
point = summaries(range(len(groups)))
for metric in ['mean_word_confidence', 'mean_conditional_nonblank_word_confidence', 'mean_tokens_per_word']:
    close('independent shared-correct point ' + metric, point[metric], report['shared_correct_words']['delta_v3_minus_v2'][metric])
close('independent WER delta pp', point['wer_delta_pp'], report['paired_english_asr']['wer']['delta_percentage_points'])
rng = random.Random(20261010)
samples = {metric: [] for metric in point}
for _ in range(2000):
    estimates = summaries([rng.randrange(len(groups)) for _ in groups])
    for metric, value in estimates.items():
        samples[metric].append(value)


def quantile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower) if lower != upper else ordered[lower]


for metric, values in samples.items():
    actual = [quantile(values, 0.025), quantile(values, 0.975)]
    expected = report['paired_english_asr']['wer']['ci_percentage_points'] if metric == 'wer_delta_pp' else report['shared_correct_word_uncertainty']['delta_v3_minus_v2_ci'][metric]
    for index in [0, 1]:
        close('independent paired-recording CI ' + metric + ' ' + str(index), actual[index], expected[index])

audit = {'passed': True, 'common_dev_recordings': 50, 'common_test_recordings': 100,
         'shared_correct_words': sum(group['count'] for group in groups),
         'independent_checks': checks,
         'method': 'Raw JSON review; independent ECE/Brier/pairwise AUROC; direct reference-index validation; independent Levenshtein counts and paired recording bootstrap from per-recording sufficient statistics.',
         'remaining_work': 'Candidate post-training confidence and calibration evaluation; small-perturbation stability experiment.'}
(root / 'english_confidence_independent_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
print(json.dumps({'passed': True, 'check_count': len(checks), 'max_difference': max(check['difference'] for check in checks), 'shared_correct_words': audit['shared_correct_words']}, indent=2))
