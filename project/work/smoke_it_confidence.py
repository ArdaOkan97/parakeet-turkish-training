import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path('outputs/parakeet_turkish').resolve()))
from confidence import generate_with_confidence, normalize_text
from runtime import inputs_for, load_model

root = Path('outputs/parakeet_turkish')
audit_path = root / 'confidence_word_boundary_fix.json'
audit = json.loads(audit_path.read_text(encoding='utf-8'))
rows = [json.loads(line) for line in Path('work/data/test_it.jsonl').read_text(encoding='utf-8').splitlines() if line]
row = next(row for row in rows if row['source_id'] == audit['failing_recording_source_id'])
model, processor = load_model()
model.eval()
with torch.autocast('cuda', dtype=torch.bfloat16):
    generated, conf = generate_with_confidence(model, processor.tokenizer, inputs_for(processor, row), language='it')
hypothesis = processor.decode(generated.sequences[0], skip_special_tokens=True)
assert hypothesis.strip() == conf['text'].strip()
assert normalize_text(hypothesis, 'it').split() == [word['word'] for word in conf['words']]
verification = {'source_id': row['source_id'], 'hypothesis': hypothesis,
                'scored_word_count': len(conf['words']),
                'currency_units': [word['word'] for word in conf['words'] if word['word'] == '$'],
                'decoded_and_scored_word_lists_match': True, 'success': True}
audit['actual_gpu_inference_verification'] = verification
audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(verification, ensure_ascii=True, indent=2), flush=True)
del model
torch.cuda.empty_cache()
