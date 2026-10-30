import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path('outputs/parakeet_turkish').resolve()))
from confidence import (align_word_confidences, confidence_report, fit_temperature,
                        generate_with_confidence, normalize_text)


tokenizer = AutoTokenizer.from_pretrained('work/research', local_files_only=True)
blank = tokenizer.convert_tokens_to_ids('<blank>')
vocab_size = blank + 1


class FixtureModel(torch.nn.Module):
    def __init__(self, text):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size, blank_token_id=blank)
        self.joint = torch.nn.Identity()
        self.ids = tokenizer(text, add_special_tokens=False)['input_ids']

    def generate(self, **inputs):
        sequence = [blank]
        durations = [0]
        for token_id in self.ids:
            for selected, duration in [(blank, 1), (token_id, 0)]:
                logits = torch.full((1, 1, 1, vocab_size + 5), -9.0)
                logits[..., selected] = 1.0
                if selected != blank:
                    logits[..., blank] = 0.8
                logits[..., vocab_size + duration] = 3.0
                self.joint(logits)
                sequence.append(selected)
                durations.append(duration)
        return SimpleNamespace(sequences=torch.tensor([sequence]), durations=torch.tensor([durations]))


inputs = {'input_features': torch.zeros(1, 10, 128)}
for language, text in [('en', "It's a test, isn't it?"), ('tr', 'Merhaba ve dünya güzel!'),
                       ('it', 'Il costo è 5 $ e 100 $.'), ('en', 'The cost is £ 5 and € 10.')]:
    model = FixtureModel(text).eval()
    generated, captured = generate_with_confidence(model, tokenizer, inputs, language=language)
    decoded_expected = tokenizer.decode(model.ids, group_tokens=False, skip_special_tokens=True)
    assert normalize_text(decoded_expected, language) == ' '.join(word['word'] for word in captured['words'])
    assert len(captured['tokens']) == len(model.ids)
    assert captured['blank_steps'] == len(model.ids)
    assert captured['tokens'][0]['start_frame'] == 1
    assert all(word['confidence'] < 0.6 for word in captured['words'])
    assert all(word['confidence_by_temperature']['0.1'] > word['confidence'] for word in captured['words'])
    assert all(word['conditional_nonblank_confidence'] > word['confidence'] for word in captured['words'])
    alignment = align_word_confidences(decoded_expected, captured, language)
    assert [word['reference_word_index'] for word in alignment['words']] == list(range(len(captured['words'])))
    report = confidence_report([alignment])
    assert report['emitted_word_accuracy'] == 1.0
    assert report['correctness_auroc'] is None
    calibration = fit_temperature([alignment])
    assert calibration['temperature'] == 0.1
    # A deletion has no emitted confidence; it must be counted separately.
    deleted = align_word_confidences(decoded_expected + ' extra', captured, language)
    assert deleted['deletions'] == 1
    assert confidence_report([deleted])['word_count'] == report['word_count']
    print(language, ascii(captured['text']), report['word_count'], round(report['brier'], 4), calibration['temperature'])

assert normalize_text('I İ IĞDIR İSTANBUL', 'tr') == 'ı i ığdır istanbul'
assert len(model.joint._forward_hooks) == 0
known = [{'words': [
    {'confidence_by_temperature': {'1': 0.8}, 'correct': True},
    {'confidence_by_temperature': {'1': 0.1}, 'correct': False},
]}]
metrics = confidence_report(known)
assert abs(metrics['brier'] - 0.025) < 1e-10
assert abs(metrics['ece'] - 0.15) < 1e-10
assert metrics['correctness_auroc'] == 1.0
try:
    generate_with_confidence(model, tokenizer, {'input_features': torch.zeros(2, 10, 128)})
except ValueError:
    pass
else:
    raise AssertionError('batch >1 must be rejected')
print('Confidence fixture checks passed.')
