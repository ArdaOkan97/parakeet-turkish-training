"""Small actual-data CPU checks for the standalone large-batch profiler."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from tokenizers import Tokenizer
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(OUT))
import profile_large_batches as p
from extend_vocab import CHARACTER_DONORS, _make_tokenizer
from batched_lora import collate_training_batch
torch.set_num_threads(4)
checks = []
def check(name, value):
    if not value:
        raise AssertionError(name)
    checks.append(name)

hashes = p.source_hashes()
pool = json.loads((OUT / 'runs/adaptive_20261010_01_screen_A/training_pool.json').read_text(encoding='utf-8'))
base = (ROOT / 'work/base_model_path.txt').read_text().strip()
processor = AutoProcessor.from_pretrained(base, local_files_only=True)
old = processor.tokenizer
size = len(old)
updated = copy.deepcopy(json.loads(old._tokenizer.to_str()))
for token, index in old._tokenizer.get_vocab().items():
    updated['model']['vocab'].setdefault(token, index)
for index, char in enumerate((c for c in CHARACTER_DONORS if c not in old._tokenizer.get_vocab()), start=size):
    updated['model']['vocab'][char] = index
processor.tokenizer = _make_tokenizer(old, Tokenizer.from_str(json.dumps(updated, ensure_ascii=False)))
cases = {}
for family in ('typical', 'long'):
    rows = p.choose_rows(pool, family)
    items, specs = p.inspect_rows(rows)
    check(family + ':16_unique_actual_train_ids', len({r['source_id'] for r in rows}) == 16)
    check(family + ':eight_turkish_eight_replay', sum(r['language'] == 'tr' for r in rows) == 8)
    check(family + ':all_five_retention_languages', {r['language'] for r in rows if r['language'] != 'tr'} == set(p.LANGUAGES))
    for batch_size in (1, 2, 4, 8, 16):
        groups = p.grouping(batch_size)
        check(f'{family}:B{batch_size}:all16_once', sorted(i for g in groups for i in g) == list(range(16)))
        check(f'{family}:B{batch_size}:effective16', len(groups) * batch_size == 16 and all(len(g) == batch_size for g in groups))
        check(f'{family}:B{batch_size}:clip_mean_equal', abs(sum(sum(range(16)[i] for i in g) / batch_size / len(groups) for g in groups) - 7.5) < 1e-12)
        padding = p.padding_from_rows(rows, groups)
        check(f'{family}:B{batch_size}:padding_valid', padding['lattice_padding_factor'] >= 1 and padding['encoder_padding_frames'] >= 0 and padding['decoder_padding_tokens'] >= 0)
        if batch_size in (2, 4, 8):
            check(f'{family}:B{batch_size}:homogeneous_groups', all(len({rows[i]['language'] == 'tr' for i in g}) == 1 for g in groups))
    batch = collate_training_batch(processor, rows, items, blank_id=8192, old_size=size, device='cpu')
    inv = p.batch_invariants(batch, list(range(16)))
    check(family + ':actual_frontend_masks_padding', inv['input_padding_zero'] and inv['cache_padding_zero'] and inv['target_padding_blank'] and inv['right_padded_mask'])
    check(family + ':actual_teacher_lengths', batch.frame_lengths.tolist() == [r['encoder_frames'] for r in rows])
    check(family + ':actual_target_lengths', batch.target_lengths.tolist() == [r['target_tokens'] for r in rows])
    check(family + ':replay_indices', batch.replay_indices == tuple(range(1, 16, 2)))
    for i, row in enumerate(rows):
        check(family + ':labels:' + row['source_id'], items[i]['labels'].tolist() == processor.tokenizer(row['text'])['input_ids'])
    bad = copy.copy(batch)
    bad.labels = batch.labels.clone()
    padded_row = next(i for i, length in enumerate(batch.target_lengths.tolist()) if length < batch.labels.shape[-1])
    bad.labels[padded_row, int(batch.target_lengths[padded_row])] = 1
    try:
        p.batch_invariants(bad, list(range(16)))
    except ValueError:
        checks.append(family + ':bad_padding_rejected')
    else:
        raise AssertionError('corrupt target padding was accepted')
    cases[family] = {'source_ids': [r['source_id'] for r in rows], 'B16_actual_invariants': inv,
                     'audio_seconds_per_update': sum(r['duration'] for r in rows)}
check('all_frozen_source_hashes_unchanged', p.source_hashes() == hashes)
parsed = p.parser().parse_args(['--mode', 'timing', '--batch-size', '16', '--backend', 'default', '--gpu-context', 'solo',
                              '--exploratory-timing', '--updates', '10', '--warmup', '2'])
check('CLI_required_root_options_parse', parsed.batch_size == 16 and parsed.exploratory_timing and parsed.gpu_context == 'solo')
report = {'passed': True, 'checks': len(checks), 'scope': 'CPU-only actual32 TRAIN clips, tokenizer/cache/WAV identities, B1/2/4/8/16 grouping, B16 actual frontend/mask padding and corruption guard; no model-weight load or GPU.',
          'source_sha256': hashes, 'cases': cases}
p.write(OUT / 'large_batch_profiler_cpu_verification.json', report)
print(json.dumps({'passed': True, 'checks': len(checks), 'families': list(cases)}))
