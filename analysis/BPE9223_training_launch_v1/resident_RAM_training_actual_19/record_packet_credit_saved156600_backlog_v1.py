import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

P = Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
S = P / 'outputs' / 'parakeet_turkish'
A = Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1\resident_RAM_training_actual_19')
TASK = S / 'backlog' / 'tasks' / 'asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
MARKER = 'BPE9223_PACKET_CREDIT_SAVED156400_RECOVERED156600_20261023'
STATUS = A / 'actual_source24_saved156600_recovery_status.json'
ROOT = A.parent / 'ROOT_owned_packet_credit_saved156400_recovered156600_actual_verification_v1.json'
NOTE = A / 'packet_credit_saved156600_milestone_note.txt'
ATTEMPT = A / 'packet_credit_saved156600_backlog_attempt.json'
RESULT = A / 'actual_packet_credit_saved156600_backlog_result.json'

def pin(path):
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2)
        f.write('\n')

assert os.environ.get('GPU_TRAINING_STORAGE_ROOT', '').lower() == r'E:\gpu_training_scratchpad'.lower()
assert os.environ.get('TORCH_EXTENSIONS_DIR', '').lower() == r'E:\gpu_training_scratchpad\cache\torch_extensions'.lower()
assert pin(STATUS) == {'path': str(STATUS), 'bytes': 12670, 'sha256': 'cd565e2636ba59961f92263c8b5fa2f0a864543ed0a393f6be7458ed4d538ead'}
assert pin(ROOT) == {'path': str(ROOT), 'bytes': 13548, 'sha256': '72aa00ddc8640affea638248bd570c193f817594cd81adfa41a6e768fc2b80d5'}
accepted = json.loads(ROOT.read_text(encoding='utf-8'))
assert accepted['accepted'] is True and accepted['Backlog_marker'] == MARKER
assert not RESULT.exists() and not ATTEMPT.exists() and not NOTE.exists()
assert TASK.read_text(encoding='utf-8-sig').count(MARKER) == 0

note = (
    datetime.datetime.now(datetime.timezone.utc).isoformat() + ': ' + MARKER + ' — '
    'The separate added-buffer scheduling timeout at actual156537 and exact saved156400 recovery are independently accepted. '
    'The source22 pre-update gate timed out after300s: last available commit4,676,124,672B was below its unchanged4,768,435,456B threshold. '
    'The original memory Monitor completed cleanly with no error; separate historical native/CPU/wrapper numeric OS exits remain unknown. '
    'Saved156400 and all137 later positive attempts156401..156537 remain preserved as historical evidence. '
    'Reviewed source24/CPU10 uses one owned cross-process logical packet credit, acquired before preparation and returned only after consumed payload references are dropped; cancellation, bounded waits and handle ownership are explicit. '
    'Transient producer/receiver physical copies remain possible. Full2.5GB added native buffer, full256MiB packet/queue/prefetch reservations and all original memory/CUDA/storage guards are retained. '
    'The original numerical/native/commit/DEV bodies, labels, sampler order, three Adam groups, four RNGs, LR, checkpoint200 and periodic4000 policies remain unchanged. '
    'The finite synthetic kernel/transport proof is not an audio/frontend golden or a measured memory-saving, speed, allocation, failure-prevention or future-fit guarantee. '
    'Fresh156400 restore matched all207 weights/threeAdam/fourRNG/bothsamplers/journal/nextdraw BEFORE native; only the saved316802360-byte prefix was copied AFTER restore into activejournal18 with0native/draw. '
    'ROOT verified138 new ordered actual positive events156401..156538, genuinely redoing all137 historical attempts and passing the old frontier. '
    'Standard156600 persistence verified854 named tensors/207 weights/215 Adam states with PRE/SAVED and semantic equality; this does not claim a second fresh GPU restore from156600. '
    'Accepted owner status156640 and independent ROOT156680/new280 showed exact three clocks, both poison flags false and allfive runtime identities alive. '
    'Trackio reader26892 caught up to journal18/newnative42304/workertrue/errorNone/durable156600 without restart. '
    'Same exclusive session96057/source24/CPU10/helper1/helper3 and full401072/adminNone continue unchanged. '
    'The first actual helper3 normal periodic production phase remains pending160000; no new quality, inference, export or benchmark is claimed. '
    'Accepted status ' + str(STATUS) + ' SHAcd565e26; independent ROOT ' + str(ROOT) + ' SHA72aa00dd.'
)
with NOTE.open('x', encoding='utf-8', newline='\n') as f:
    f.write(note + '\n')
write(ATTEMPT, {'kind': 'ASR18_one_packet_credit_Backlog_attempt_v1', 'marker': MARKER, 'canonical_marker_count_before': 0, 'note': pin(NOTE), 'state': 'before_task_edit'})

edit = subprocess.run([str(S / 'tools' / 'backlog.exe'), 'task', 'edit', 'ASR-18', '--append-notes', note], cwd=str(S), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(A / 'packet_credit_saved156600_backlog_edit.stdout.txt').write_bytes(edit.stdout)
(A / 'packet_credit_saved156600_backlog_edit.stderr.txt').write_bytes(edit.stderr)
write(A / 'packet_credit_saved156600_backlog_edit_result.json', {'actual_exit_code': edit.returncode, 'canonical_marker_count_after_edit': TASK.read_text(encoding='utf-8-sig').count(MARKER)})
assert edit.returncode == 0, 'Actual Backlog edit failed; do not rerun ambiguously'
assert TASK.read_text(encoding='utf-8-sig').count(MARKER) == 1

refresh = subprocess.run([str(P / 'work' / 'training-env' / 'Scripts' / 'python.exe'), str(S / 'refresh_backlog.py')], cwd=str(P), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(A / 'packet_credit_saved156600_backlog_refresh.stdout.txt').write_bytes(refresh.stdout)
(A / 'packet_credit_saved156600_backlog_refresh.stderr.txt').write_bytes(refresh.stderr)
write(A / 'packet_credit_saved156600_backlog_refresh_result.json', {'actual_exit_code': refresh.returncode})
assert refresh.returncode == 0, 'Actual refresh failed; edit already completed, do not repeat it'
count = TASK.read_text(encoding='utf-8-sig').count(MARKER)
assert count == 1
write(RESULT, {
    'kind': 'ASR18_actual_packet_credit_saved156600_Backlog_milestone_v1',
    'marker': MARKER,
    'actual_task_edit_exit_code': edit.returncode,
    'actual_refresh_exit_code': refresh.returncode,
    'canonical_marker_count_after_edit_and_refresh': count,
    'status': pin(STATUS),
    'ROOT_actual_verification': pin(ROOT),
    'note': pin(NOTE),
    'training_changes': False,
    'duplicate_prior_note': False,
})
print(json.dumps({'passed': True, 'result': pin(RESULT), 'actual_task_edit_exit_code': edit.returncode, 'actual_refresh_exit_code': refresh.returncode, 'canonical_marker_count': count}))
