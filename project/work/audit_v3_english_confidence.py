import json
import math
from pathlib import Path

root = Path('outputs/parakeet_turkish')
folder = root / 'results/base_v3'
rows = [json.loads(line) for line in (folder / 'test_en.jsonl').read_text(encoding='utf-8').splitlines() if line]
summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
max_word_score_difference = 0.0
max_token_temperature_one_difference = 0.0
nonfinite_values = invalid_token_probabilities = invalid_entropy_values = invalid_conditional_values = nonmonotonic_temperatures = 0
special_step_residuals = []
all_words = []
total_tokens = 0
for row in rows:
    captured = row['confidence']
    special_step_residuals.append(captured['total_steps'] - captured['blank_steps'] - len(captured['tokens']))
    for token in captured['tokens']:
        total_tokens += 1
        logprob = token['logprob']
        nonfinite_values += not math.isfinite(logprob)
        invalid_token_probabilities += not (-math.log(8193) - 1e-5 <= logprob <= 1e-5)
        invalid_entropy_values += not (-1e-6 <= token['normalized_entropy'] <= 1 + 1e-6)
        conditional = math.exp(token['conditional_nonblank_logprob'])
        invalid_conditional_values += not (math.exp(logprob) - 1e-6 <= conditional <= 1 + 1e-6)
        max_token_temperature_one_difference = max(max_token_temperature_one_difference, abs(logprob - token['logprob_by_temperature']['1']))
        ordered = sorted((float(key), value) for key, value in token['logprob_by_temperature'].items())
        nonmonotonic_temperatures += any(first[1] + 1e-5 < second[1] for first, second in zip(ordered, ordered[1:]))
    for word in row['alignment']['words']:
        logs = [captured['tokens'][index]['logprob'] for index in word['token_indices']]
        derived = math.exp(sum(logs) / len(logs))
        max_word_score_difference = max(max_word_score_difference, abs(derived - word['confidence']))
        all_words.append({'source_id': row['source_id'], 'word': word['word'],
                          'correct': word['correct'], 'operation': word['operation'],
                          'reference_word_index': word.get('reference_word_index'),
                          'confidence': word['confidence'],
                          'conditional_nonblank_confidence': word['conditional_nonblank_confidence'],
                          'token_count': len(word['token_indices']),
                          'tokens': [{'piece': captured['tokens'][index]['piece'],
                                     'logprob': captured['tokens'][index]['logprob'],
                                     'blank_probability': captured['tokens'][index]['blank_probability']}
                                    for index in word['token_indices']],
                          'reference': row['reference'], 'hypothesis': row['hypothesis']})
correct = [word for word in all_words if word['correct']]
incorrect = [word for word in all_words if not word['correct']]
raw = summary['test_en']['raw_confidence']
calibrated = summary['test_en']['calibrated_confidence']
audit = {
    'recordings': len(rows), 'word_count': len(all_words), 'correct_words': len(correct),
    'incorrect_words': len(incorrect), 'nonblank_token_count': total_tokens,
    'numeric_checks': {
        'nonfinite_token_logprobs': nonfinite_values,
        'invalid_argmax_token_probability_bounds': invalid_token_probabilities,
        'invalid_normalized_entropy_values': invalid_entropy_values,
        'invalid_conditional_nonblank_values_tolerance_1e_6': invalid_conditional_values,
        'nonmonotonic_temperature_arrays_tolerance_1e_5': nonmonotonic_temperatures,
        'max_raw_token_logprob_vs_temperature_one_difference': max_token_temperature_one_difference,
        'max_word_geometric_mean_reconstruction_difference': max_word_score_difference,
        'max_uncounted_special_emissions': max(special_step_residuals),
    },
    'correct_word_mean_raw_confidence': sum(word['confidence'] for word in correct) / len(correct),
    'correct_word_mean_conditional_nonblank_confidence': sum(word['conditional_nonblank_confidence'] for word in correct) / len(correct),
    'incorrect_word_mean_raw_confidence': sum(word['confidence'] for word in incorrect) / len(incorrect),
    'incorrect_word_mean_conditional_nonblank_confidence': sum(word['conditional_nonblank_confidence'] for word in incorrect) / len(incorrect),
    'correct_words_below_0_5': [word for word in correct if word['confidence'] < 0.5],
    'high_confidence_incorrect_examples': sorted(incorrect, key=lambda word: -word['confidence'])[:12],
    'development_fitted_temperature': summary['calibration_en']['temperature'],
    'heldout_calibration': {metric: {'raw': raw[metric], 'calibrated': calibrated[metric]}
                            for metric in ['ece', 'brier', 'nll', 'correctness_auroc', 'incorrect_words_above_0_8_fraction']},
    'interpretation': 'Broad low confidence on correct words is not reproduced on this clean FLEURS English sample. Rare near-0.5 words exist; confident substitutions/insertions are more common. Development-fitted T>1 reduces overconfidence on held-out test without changing WER.',
}
(root / 'base_v3_english_confidence_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps({key: value for key, value in audit.items() if key not in ['correct_words_below_0_5', 'high_confidence_incorrect_examples']}, indent=2))
print('Low-confidence correct words:', ascii(audit['correct_words_below_0_5']))
print('Confident wrong examples:', ascii(audit['high_confidence_incorrect_examples'][:3]))
