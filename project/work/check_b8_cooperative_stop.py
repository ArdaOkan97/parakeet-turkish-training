"""Read-only CPU checkpoint and DEV audit for the completed B8 trial."""
import io
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(OUT))
from atomic_io_retry_v1 import read_bytes_shared
from metrics import score_transcriptions
import torch

torch.set_num_threads(2)
RID = 'adaptive_20261010_recovery_b8_01_b8_quality'
LOOP = 'adaptive_20261010_recovery_b8_01'
RUN = OUT / 'runs' / RID
REPORT = OUT / 'b8_cooperative_stop_verification.json'
checks = 0


def check(ok, message):
    global checks
    checks += 1
    if not ok:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(read_bytes_shared(path)).hexdigest()


def read(path):
    return json.loads(read_bytes_shared(path).decode('utf-8'))


preserved = [OUT / 'b8_memory_slowdown_diagnosis.json',
             OUT / 'b8_runtime_resource_observation.json',
             OUT / 'sleep_recovery_quality_verification.json',
             RUN / 'selection.json', RUN / 'run_config.json', RUN / 'training_history.json']
preserved += sorted((OUT / 'results' / (RID + '_step_801')).glob('*'))
before = {str(p): sha(p) for p in preserved if p.is_file()}
selection = read(RUN / 'selection.json')
config = read(RUN / 'run_config.json')
history = read(RUN / 'training_history.json')
check(selection['accepted'] is True, 'selected candidate not accepted')
check(selection['completed_steps'] == 907, 'completed step changed')
check(selection['selected_step'] == 801, 'best step changed')
check(selection['stop_early_reason'] == 'controller_stop_file', 'stop was not cooperative')
check(selection['heldout_tests_evaluated'] is False, 'trainer used TEST')
check([r['step'] for r in history] == list(range(1, 908)), 'history is not contiguous')
check(history[-1]['validation']['retention_gate_passed'] is True, 'final retention failed')
code = {}
for name, expected in config['code_sha256'].items():
    actual = sha(OUT / name)
    check(actual == expected, 'source changed: ' + name)
    code[name] = actual
provenance_path = OUT / 'runs' / LOOP / (RID + '_io_retry_provenance.json')
provenance = read(provenance_path)
check(provenance['status'] == 'complete', 'launcher did not complete')
check(provenance['training_semantics_changed'] is False, 'I/O wrapper changed training semantics')
for key, path in [('worker', OUT / 'train_lora_batched.py'),
                  ('launcher', OUT / 'launch_with_io_retry_v1.py'),
                  ('helper', OUT / 'atomic_io_retry_v1.py')]:
    actual = sha(path)
    check(provenance[key + '_sha256'] == provenance[key + '_sha256_after'] == actual,
          key + ' source changed')

checkpoints = {}
required = {'schema', 'run_id', 'step', 'weights', 'lora_metadata', 'optimizer', 'sampler',
            'python_rng', 'torch_rng', 'cuda_rng', 'config_identity', 'history', 'best_score',
            'selected_step', 'patience', 'retention_failures', 'next_eval_epoch',
            'train_wall_seconds', 'audio_seconds', 'language_audio_seconds',
            'evaluation_wall_seconds', 'initial_gate', 'stop_reason'}
for kind, step in [('best', 801), ('latest', 907)]:
    path = ROOT / 'work' / (RID + '_' + kind + '.pt')
    data = read_bytes_shared(path)
    digest = hashlib.sha256(data).hexdigest()
    state = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    check(set(state) == required and state['schema'] == 2, kind + ' incomplete schema')
    check(state['run_id'] == RID and state['step'] == step, kind + ' identity/step')
    check(state['selected_step'] == 801 and state['best_score'] == 570/825, kind + ' selected score')
    check(state['config_identity'] == config['config_identity'], kind + ' configuration identity')
    check(state['lora_metadata'] == config['lora_layout'], kind + ' adapter layout')
    check(len(state['weights']) == 207, kind + ' weight count')
    for name, tensor in state['weights'].items():
        check(tensor.device.type == 'cpu' and torch.isfinite(tensor).all().item(), kind + ' nonfinite weight ' + name)
    optimizer = state['optimizer']
    check(len(optimizer['state']) == 207 and len(optimizer['param_groups']) == 2, kind + ' Adam count')
    ids = [i for group in optimizer['param_groups'] for i in group['params']]
    check(len(ids) == len(set(ids)) == 207 and set(ids) == set(optimizer['state']), kind + ' optimizer parameter identity')
    for index, value in optimizer['state'].items():
        check(set(value) == {'step', 'exp_avg', 'exp_avg_sq'}, kind + ' Adam state schema')
        check(float(value['step']) == step, kind + ' Adam step')
        check(value['exp_avg'].shape == value['exp_avg_sq'].shape, kind + ' Adam moments shape')
        check(torch.isfinite(value['exp_avg']).all().item() and torch.isfinite(value['exp_avg_sq']).all().item(), kind + ' Adam finite')
        check((value['exp_avg_sq'] >= 0).all().item(), kind + ' Adam second moment negative')
    sampler = state['sampler']
    check(sampler['batch_size'] == 8 and sampler['accumulation'] == 2 and sampler['replay_per_update'] == 8, kind + ' sampler effective batch')
    local_rng = random.Random(); local_rng.setstate(state['python_rng'])
    sampler_rng = random.Random(); sampler_rng.setstate(sampler['rng'])
    cpu_generator = torch.Generator(device='cpu'); cpu_generator.set_state(state['torch_rng'])
    check(isinstance(state['cuda_rng'], list) and len(state['cuda_rng']) == 1, kind + ' CUDA RNG list')
    check(all(x.device.type == 'cpu' and x.dtype == torch.uint8 and x.numel() > 0 for x in state['cuda_rng']), kind + ' CUDA RNG serialized state')
    check([r['step'] for r in state['history']] == list(range(1, step + 1)), kind + ' history contiguity')
    if kind == 'best':
        check(digest == 'ebec01271d1300b615e869ee7317d54990a290d86713f0f8eab12041a0d80b7a', 'best801 changed since pre-stop read')
        check(state['stop_reason'] is None, 'best801 metadata rewritten')
    else:
        check(state['history'] == history, 'latest history differs from durable JSON')
        check(state['stop_reason'] == 'controller_stop_file', 'latest stop reason')
        check(state['audio_seconds'] == selection['audio_seen_seconds'], 'latest audio accounting')
        check(state['train_wall_seconds'] == selection['train_wall_seconds'], 'latest training wall accounting')
    checkpoints[kind] = {'path': str(path), 'sha256': digest, 'size_bytes': len(data),
                         'step': step, 'selected_step': state['selected_step'],
                         'best_score': state['best_score'], 'weights': 207,
                         'adam_parameter_states': 207, 'adam_step': step,
                         'optimizer_groups': 2, 'history_entries': len(state['history']),
                         'sampler_batch_size': sampler['batch_size'], 'sampler_accumulation': sampler['accumulation'],
                         'stop_reason': state['stop_reason'], 'python_and_cpu_rng_roundtrip': True,
                         'cuda_rng_serialization_validated_without_cuda_initialization': True}
    del state, data

base = read(OUT / 'results/base_v3/summary.json')
metrics = {}
for step in [801, 907]:
    folder = OUT / 'results' / (RID + '_step_' + str(step))
    summary = read(folder / 'summary.json')
    scores = {}
    for lang in ['tr', 'en', 'es', 'it', 'fr', 'de']:
        key = 'validation_' + lang
        rows = [json.loads(x) for x in read_bytes_shared(folder / (key + '.jsonl')).decode('utf-8').splitlines()]
        baseline_rows = [json.loads(x) for x in read_bytes_shared(OUT / 'results/base_v3' / (key + '.jsonl')).decode('utf-8').splitlines()]
        check(len(rows) == 50, key + ' coverage')
        check([(r['source_id'],r['reference']) for r in rows] == [(r['source_id'],r['reference']) for r in baseline_rows], key + ' source/reference identity')
        score = score_transcriptions(rows, language=lang)
        for field in ['wer','cer','word_errors','reference_words','character_errors','reference_characters','utterances']:
            check(score[field] == summary[key]['asr'][field], key + ' score mismatch ' + field)
        old = base[key]['asr']['wer']; limit = max(.005,.1*old)
        gate = score['wer'] <= old + limit
        if lang != 'tr': check(gate, key + ' retention gate')
        scores[lang] = {'wer':score['wer'], 'cer':score['cer'], 'word_errors':score['word_errors'],
                        'reference_words':score['reference_words'], 'original_wer':old,
                        'delta_percentage_points':100*(score['wer']-old),
                        'retention_limit_percentage_points':100*limit,
                        'retention_gate_passed':gate if lang != 'tr' else None}
    metrics[str(step)] = scores
check(metrics['801']['tr']['wer'] == selection['turkish_validation_wer'], 'selected DEV score')
check(metrics['907']['tr']['wer'] == history[-1]['validation']['turkish_wer'], 'final DEV score')
check(torch.cuda.is_initialized() is False, 'audit initialized CUDA')
check(not list((ROOT/'work').glob(RID + '*.tmp*')), 'uncommitted checkpoint temporary')
for path, digest in before.items(): check(sha(Path(path)) == digest, 'prior evidence mutated: ' + path)
report = {'schema':1,'status':'complete','passed':True,'checks':checks,
          'created_at_utc':datetime.now(timezone.utc).isoformat(),'run_id':RID,'loop_id':LOOP,
          'scope':'CPU-only cooperative-stop checkpoint/source/DEV verification; no GPU or TEST inference',
          'completed_step':907,'selected_step':801,'best801_unchanged_from_pre_stop_sha256':True,
          'checkpoints':checkpoints,'development':metrics,'selection':selection,
          'source_sha256':code,'trainer_io_provenance_path':str(provenance_path),
          'trainer_io_provenance_sha256':sha(provenance_path),'trainer_io_status':provenance['status'],
          'trainer_io_retry_events':provenance['retry_events'],'preserved_evidence_sha256':before,
          'cuda_initialized':torch.cuda.is_initialized(),
          'limitations':['Serialized CUDA RNG validity checked on CPU; RNG not restored into a CUDA generator.',
                         'This verifies durable trainer state and DEV scores; merged export and heldout TEST require the separate final auditor.']}
with REPORT.open('x',encoding='utf-8') as f: json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps({'passed':True,'checks':checks,'report':str(REPORT),'report_sha256':sha(REPORT),
                  'checkpoints':checkpoints,'cuda_initialized':False},ensure_ascii=False))
