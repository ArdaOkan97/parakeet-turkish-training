"""ONE ROOT-accepted ninth file-specific repair/recovery milestone and normal refresh."""
import datetime, hashlib, json, os, subprocess, sys
from pathlib import Path

P = Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
S = P / 'outputs/parakeet_turkish'
F = Path(__file__).resolve().parent
A = F.parent / 'resident_RAM_training_actual_27'
OUT = A / 'backlog_ninth_saved186400_recovered186600_01'
RESULT = A / 'actual_ninth_saved186400_recovered186600_backlog_result_v1.json'
TASK = S / 'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
MARKER = 'BPE9223_SOURCE37_NINTH_FULLCLIP_SAVED186400_REDONE163_RECOVERED186600_20261024'

def pin(path):
    raw = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())

assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
assert not OUT.exists() and not RESULT.exists(), 'One fresh operation; no ambiguous retry'
before = TASK.read_text(encoding='utf-8').count(MARKER)
assert before == 0, ('Canonical marker already present; no duplicate edit or refresh', before)
inputs = {
    'owner_actual_recovery_status': (
        A / 'actual_dynamic_natural_recovery_01/actual_recovery_status.json',
        '15cfab5e74cd2f501682f5c95eca8eac7893734b5deec486fe5246e6ce85f9a0'),
    'ROOT_actual_recovery_review': (
        F / 'ROOT_source37_saved186400_redone163_repaired186564_natural186600_actual_recovery_independent_verification_v1.json',
        '48b493164975cd9ecc4d579303de15b561d9e7af2a553308589ac9634383663a')}
pins = {}
for key, (path, digest) in inputs.items():
    pins[key] = pin(path)
    assert pins[key]['sha256'] == digest, (key, pins[key])

note = f'''{MARKER}
Source36 stopped after 186563 positive optimizer updates, with the fully saved checkpoint at 186400. Its final credit-release exception masked an original interval-decoder failure; the exact recorded candidate was localized to row5 of plan389/index3 (global186564). All163 unsaved positive events were preserved as historical evidence. Separate native/CPU/wrapper numeric OS exits remain UNKNOWN and are not inferred from the outer TOOL1.
The ninth correction is restricted to the exact original row/member/annotation and preserves all eight earlier file-specific corrections. The independent complete-member observation used all630840 physical native samples once, yielding210280 clean samples; full66 targets, encoder T165 and valid-mask length1314 remain unchanged. The original larger630864 native/210288 clean capacities and full901306112-byte numerical reservation remain charged. The genuine CPU14 ingress and original frontend produced finite features[1,1315,128] with the original mask and strict source/row/target/geometry negatives. This does not establish a generic EOF cause or a corpus-wide correction.
Source37 genuinely restored full186400 model/three Adam groups/four RNG/both samplers/nextdraw BEFORE native work, then copied only the saved journal prefix with zero native updates or authority draws. All163 historical positives were genuinely redone, followed by the repaired186564 positive update. ROOT verified the same164-event segment and the new186600 archive's complete200 ordered records, repaired row5/full66/native metadata and original16 integer route-key digest semantics. Natural186600 full854 named tensors/207 weights/215 Adam states passed original semantic and independent PRE-SAVED equality checks, followed by positive training. The unchanged reader caught up to journal26/native19320 and remained healthy. All recovery/closure audits are accepted and closed.
The original180k once-only event is inherited as consumed BEFORE the source37 loop; the new owner remains completed=false/attempted=false with operations[], and no second eligible/ineligible event, capture, collect, freeze, unfreeze or extra synchronization is performed. Sole-owned session26493 continues full401072/adminNone. Original science, LR, native cleanup, save200, 2GB commit/1.5GB physical floors,2250000000B cushion, FULL packet/queue/prefetch reservations, CPU2.5GB reserve/private8GB, MAX2/LIVE1 and the full16MiB event bookkeeping reserve remain unchanged.
The recurring throughput-fix request remains OPEN. This accepted file-specific repair/recovery makes no durable-speed, fit, memory-saving or generic-EOF/prevention claim. No old quality or diagnostic milestone is repeated.
Evidence: owner recovery status {pins['owner_actual_recovery_status']['sha256']}; ROOT accepted recovery {pins['ROOT_actual_recovery_review']['sha256']}.
'''
OUT.mkdir()
with (OUT / 'milestone_note.txt').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(note)
    stream.flush()
    os.fsync(stream.fileno())
write(OUT / 'entry.json', {
    'kind': 'ASR18_source37_ninth_saved186400_recovered186600_Backlog_entry_v1',
    'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'marker': MARKER, 'marker_count_before': before, 'inputs': pins,
    'task_before': pin(TASK), 'one_edit_only': True})
result = {
    'kind': 'ASR18_source37_ninth_saved186400_recovered186600_Backlog_result_v1',
    'marker': MARKER, 'marker_count_before': before, 'inputs': pins,
    'CLI_edit_invocations': 0, 'actual_CLI_edit_exit': None, 'actual_refresh_exit': None,
    'passed': False, 'error': None,
    'refresh_interpreter': str(P / 'work/training-env/Scripts/python.exe')}
try:
    result['CLI_edit_invocations'] = 1
    with (OUT / 'edit.stdout.log').open('xb') as out, (OUT / 'edit.stderr.log').open('xb') as err:
        done = subprocess.run([str(S / 'tools/backlog.exe'), 'task', 'edit', 'ASR-18', '--append-notes', note],
                              cwd=S, stdout=out, stderr=err, timeout=60, check=False)
    result['actual_CLI_edit_exit'] = done.returncode
    assert done.returncode == 0
    assert TASK.read_text(encoding='utf-8').count(MARKER) == 1
    with (OUT / 'refresh.stdout.log').open('xb') as out, (OUT / 'refresh.stderr.log').open('xb') as err:
        done = subprocess.run([result['refresh_interpreter'], '-Xutf8', '-B', str(S / 'refresh_backlog.py')],
                              cwd=S, stdout=out, stderr=err, timeout=60, check=False)
    result['actual_refresh_exit'] = done.returncode
    assert done.returncode == 0
    result['marker_count_after'] = TASK.read_text(encoding='utf-8').count(MARKER)
    assert result['marker_count_after'] == 1
    result['passed'] = True
except BaseException as error:
    result['error'] = {'type': type(error).__name__, 'message': str(error), 'ambiguous_retry_forbidden': True}
finally:
    result['marker_count_after'] = TASK.read_text(encoding='utf-8').count(MARKER)
    result['task_after'] = pin(TASK)
    result['entry'] = pin(OUT / 'entry.json')
    result['note'] = pin(OUT / 'milestone_note.txt')
    result['source'] = pin(Path(__file__))
    result['logs'] = {path.name: pin(path) for path in sorted(OUT.glob('*.log'))}
    result['input_pins_unchanged'] = all(pin(path) == pins[key] for key, (path, _) in inputs.items())
    result['no_training_reader_source_environment_or_policy_change'] = True
    write(RESULT, result)
print(json.dumps({'result': pin(RESULT), 'passed': result['passed'], 'edit': result['actual_CLI_edit_exit'],
                  'refresh': result['actual_refresh_exit'], 'marker_before': before,
                  'marker_after': result['marker_count_after'], 'error': result['error']}), flush=True)
sys.exit(0 if result['passed'] else 1)
