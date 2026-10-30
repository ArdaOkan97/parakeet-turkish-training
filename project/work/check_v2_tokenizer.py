import json
import random
from pathlib import Path

import pyarrow.parquet as pq
import sentencepiece as spm
from transformers import AutoProcessor

root = Path('outputs/parakeet_turkish')
native = spm.SentencePieceProcessor(model_file=str(next(Path('work/v2-extracted').glob('*.model'))))
fast = AutoProcessor.from_pretrained('work/models/parakeet-v2', local_files_only=True).tokenizer
source = next(Path('work/data').glob('**/en_us/validation-*.parquet'))
texts = pq.read_table(source, columns=['raw_transcription']).column('raw_transcription').to_pylist()
selected = random.Random(2102026).sample(texts, min(100, len(texts)))
mismatches = []
id_matches = 0
same_policy_decode_matches = 0
for index, text in enumerate(selected):
    native_ids = native.encode(text, out_type=int)
    fast_ids = fast(text, add_special_tokens=False)['input_ids']
    native_text = native.decode(native_ids)
    fast_text = fast.decode(fast_ids, group_tokens=False, skip_special_tokens=True)
    id_matches += native_ids == fast_ids
    native_skipped = native.decode([token_id for token_id in native_ids if token_id not in fast.all_special_ids])
    same_policy_decode_matches += native_skipped == fast_text
    if native_ids != fast_ids or native_text != fast_text:
        mismatches.append({'index': index, 'text': text,
                           'native_ids': native_ids, 'fast_ids': fast_ids,
                           'native_decoded': native_text, 'fast_decoded': fast_text})
audit = {
    'source': 'FLEURS en_us validation raw_transcription',
    'sample_count': len(selected),
    'random_seed': 2102026,
    'native_tokenizer': 'SentencePiece 0.2.2',
    'fast_tokenizer': 'pinned Transformers ParakeetConverter',
    'exact_token_ids_and_decode_matches': len(selected) - len(mismatches),
    'exact_token_ids_matches': id_matches,
    'decode_matches_with_identical_skipped_special_policy': same_policy_decode_matches,
    'native_unk_display_note': 'Native SentencePiece displays unknown as Unicode U+2047; the benchmark skips special tokens, including <unk>.',
    'mismatch_count': len(mismatches),
    'mismatches': mismatches,
}
(root / 'v2_tokenizer_parity.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
conversion_path = root / 'v2_conversion.json'
conversion = json.loads(conversion_path.read_text(encoding='utf-8'))
conversion['native_sentencepiece_parity'] = {key: value for key, value in audit.items() if key != 'mismatches'}
conversion_path.write_text(json.dumps(conversion, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in audit.items() if key != 'mismatches'}, indent=2))
if id_matches != len(selected) or same_policy_decode_matches != len(selected):
    print('First mismatch:', ascii(mismatches[0]))
    raise SystemExit(1)
