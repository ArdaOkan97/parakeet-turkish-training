import json
import sys
import unicodedata
from pathlib import Path
from types import SimpleNamespace

from tokenizers import Tokenizer, decoders, models
from transformers import AutoTokenizer

sys.path.insert(0, str(Path('outputs/parakeet_turkish').resolve()))
import confidence

root = Path('outputs/parakeet_turkish')
completed = [json.loads(line) for line in (root / 'results/base_v3/test_it.jsonl').read_text(encoding='utf-8').splitlines() if line]
seen = {row['source_id'] for row in completed}
data = [json.loads(line) for line in Path('work/data/test_it.jsonl').read_text(encoding='utf-8').splitlines() if line]
pending = next(row for row in data if row['source_id'] not in seen)
tokenizer = AutoTokenizer.from_pretrained('work/research', local_files_only=True)
ids = tokenizer(pending['text'], add_special_tokens=False)['input_ids']
tokens = [{'token_id': token_id, 'logprob_by_temperature': {'1': -0.1},
           'conditional_nonblank_logprob': -0.1, 'normalized_entropy': 0.1}
          for token_id in ids if token_id not in tokenizer.all_special_ids]
fixed_lexical = confidence._lexical
confidence._lexical = lambda character: unicodedata.category(character)[0] in {'L', 'M', 'N'}
try:
    confidence._decode_tokens(tokenizer, tokens, 'it')
except ValueError as error:
    old_error = str(error)
else:
    raise AssertionError('The old scorer must reproduce the dropped-currency error')
finally:
    confidence._lexical = fixed_lexical
text, words = confidence._decode_tokens(tokenizer, tokens, 'it')
assert [word['word'] for word in words] == confidence.normalize_text(text, 'it').split()
assert len([word for word in words if word['word'] == '$']) == 2

backend = Tokenizer(models.WordLevel({'a\u200bb': 0}, unk_token='a\u200bb'))
backend.decoder = decoders.Fuse()
control_text, control_words = confidence._decode_tokens(SimpleNamespace(_tokenizer=backend), [
    {'token_id': 0, 'logprob_by_temperature': {'1': -0.1},
     'conditional_nonblank_logprob': -0.1, 'normalized_entropy': 0.1}], 'en')
assert [word['word'] for word in control_words] == ['a', 'b']
audit = {
    'failing_recording_source_id': pending['source_id'],
    'completed_test_it_before_failure': len(completed),
    'reproducer': 'CPU tokenizer replay of the next failed recording reference, using synthetic logprobs; no model inference',
    'old_error': old_error,
    'cause': 'The shared WER normalizer retains Unicode symbols, including standalone currency $, while the confidence scorer only counted letters, marks, and numbers.',
    'fix': 'Score symbol tokens as retained units; split whitespace, separators, and invisible controls consistently with the shared normalizer. Continue requiring exact decoded/scored word equality.',
    'decoded_text': text,
    'normalized_words': [word['word'] for word in words],
    'scored_currency_units': 2,
    'invisible_control_fixture': {'decoded': control_text, 'words': [word['word'] for word in control_words]},
    'verified': True,
}
(root / 'confidence_word_boundary_fix.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps({key: value for key, value in audit.items() if key not in ['old_error', 'decoded_text', 'normalized_words']}, indent=2))
