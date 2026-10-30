"""One actual tiny WAV fixture; no corpus scan/model imports."""
from pathlib import Path
import ast, hashlib, importlib.util, json, sys, tempfile, wave
from unittest.mock import patch
import soundfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/parakeet_turkish/training_input_audit_reuse_v2.py'
spec = importlib.util.spec_from_file_location('audit_reuse_v2_fixture', SOURCE)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
checks = []


def checked(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)


def rejected(name, fn, expected):
    try:
        fn()
    except (ValueError, TypeError) as error:
        checked(name, expected in str(error))
    else:
        raise AssertionError(name + ': unexpectedly accepted')


with tempfile.TemporaryDirectory(prefix='reuse_v2_', dir=ROOT / 'work/research') as directory:
    folder = Path(directory)
    audio = folder / 'tiny.wav'
    with wave.open(str(audio), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b'\0\0' * 400)
    arguments = []
    argument_sha = hashlib.sha256(json.dumps(arguments, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    identity = {'expected_run_id': 'tiny_fixture', 'pid': 123, 'frozen_worker_sha256': helper.WORKER_SHA,
                'data_policy_sha256': helper.POLICY_SHA, 'worker_arguments_sha256': argument_sha}
    io = {'arguments': arguments, 'worker_sha256': helper.WORKER_SHA}
    snapshots = {}
    for key, value in [('owned_identity', identity), ('IO_provenance', io),
                       ('owned_worker_log', 'public v3 PCM/input verification 70000/74883\nacoustic cache 1/74883\n')]:
        path = folder / (key + '.json' if isinstance(value, dict) else key + '.log')
        path.write_text(json.dumps(value) if isinstance(value, dict) else value, encoding='utf-8')
        snapshots[key] = {'path': str(path), 'sha256': helper.sha(path)}
    manifest = folder / 'tiny.jsonl'
    manifest.write_text(json.dumps({'audio_filepath': str(audio)}) + '\n', encoding='utf-8')
    stat = audio.stat()
    fingerprints = folder / 'fingerprints.jsonl'
    fingerprints.write_text(json.dumps({'path': str(audio), 'size': stat.st_size,
                                       'mtime_ns': stat.st_mtime_ns, 'sample_rate': 16000,
                                       'duration': 400 / 16000}) + '\n', encoding='utf-8')
    receipt = {'passed': True, 'kind': 'observed_full_input_audit_reuse_receipt_v1',
               'observed_completed_full_audit': True, 'original_returned_audit_JSON_separately_available': False,
               'new_PCM_byte_hashes_computed': False, 'source_snapshots': snapshots,
               'source_run_id': 'tiny_fixture', 'actual_owned_worker_pid': 123,
               'worker_sha256': helper.WORKER_SHA, 'data_policy_sha256': helper.POLICY_SHA,
               'worker_arguments_sha256': argument_sha,
               'input_files_sha256': {str(manifest): helper.sha(manifest)},
               'fingerprints_path': str(fingerprints), 'fingerprints_sha256': helper.sha(fingerprints),
               'unique_TRAIN_replay_heldout_WAV_files': 1,
               'fingerprint_limitation': 'Synthetic fixture; metadata fingerprints do not hash audio contents.'}
    receipt_path = folder / 'receipt.json'
    receipt_path.write_text(json.dumps(receipt), encoding='utf-8')
    with patch.object(soundfile, 'info', wraps=soundfile.info) as info:
        checked('default mode accepts actual header and exact fixture set', helper.validate_reuse_receipt(receipt_path)[1] == 1)
        checked('default mode opens exactly one WAV header', info.call_count == 1)
    with patch.object(soundfile, 'info', side_effect=AssertionError('Header read in stat-only mode')):
        checked('explicit False verifies exact set without opening header', helper.validate_reuse_receipt(receipt_path, check_audio_headers=False)[1] == 1)
    fields = helper.reuse_verification_fields(receipt, 1, check_audio_headers=False)
    checked('stat-only evidence fresh fingerprints one', fields['fresh_WAV_fingerprint_files'] == 1)
    checked('stat-only evidence fresh headers and legacy combined count zero',
            fields['fresh_WAV_header_files'] == fields['fresh_WAV_fingerprint_and_header_files'] == 0)
    checked('stat-only evidence reused headers one and explicit mode',
            fields['reused_audited_WAV_header_files'] == 1 and not fields['WAV_headers_checked_this_audit']
            and fields['audio_verification_mode'].endswith('_reused_headers'))
    checked('default evidence preserves fresh combined count', helper.reuse_verification_fields(receipt, 1)['fresh_WAV_fingerprint_and_header_files'] == 1)
    checked('evidence JSON serializable', bool(json.dumps(fields)))
    rejected('explicit boolean required', lambda: helper.validate_reuse_receipt(receipt_path, check_audio_headers='false'), 'explicit boolean')
    rejected('stat-only without receipt rejected before policy import',
             lambda: helper.audit_public_training_inputs_reused(None, None, None, check_audio_headers=False), 'requires an explicit')
    with audio.open('ab') as stream:
        stream.write(b'changed')
    for mode in (True, False):
        rejected('changed WAV rejects mode ' + str(mode),
                 lambda mode=mode: helper.validate_reuse_receipt(receipt_path, check_audio_headers=mode), 'run full PCM audit')
    receipt['source_snapshots']['owned_identity']['sha256'] = '0' * 64
    receipt_path.write_text(json.dumps(receipt), encoding='utf-8')
    rejected('corrupt receipt fails without full fallback',
             lambda: helper.validate_reuse_receipt(receipt_path, check_audio_headers=False), 'Immutable completion snapshot changed')

tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
wrapper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'audit_public_training_inputs_reused')
calls = [node for node in ast.walk(wrapper) if isinstance(node, ast.Call)
         and isinstance(node.func, ast.Name) and node.func.id == 'validate_reuse_receipt']
checked('wrapper forwards explicit header mode to validator', len(calls) == 1
        and any(keyword.arg == 'check_audio_headers' and isinstance(keyword.value, ast.Name)
                and keyword.value.id == 'check_audio_headers' for keyword in calls[0].keywords))
report = {'schema': 1, 'kind': 'training_input_audit_reuse_v2_tiny_WAV_CPU_verification',
          'passed': True, 'checks': checks, 'checks_passed': len(checks),
          'helper_path': str(SOURCE), 'helper_sha256': helper.sha(SOURCE),
          'scope': {'actual_audio_fixture_files': 1, 'frames': 400, 'sample_rate': 16000,
                    'full_corpus_scan': False, 'model_or_NeMo_imports': False,
                    'Torch_imported': 'torch' in sys.modules, 'GPU_work': False,
                    'routing_changed': False, 'wrapper_full_policy_integration_tested': False}}
(ROOT / 'outputs/parakeet_turkish/training_input_audit_reuse_v2_cpu_verification.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'passed': True, 'checks': len(checks), 'helper_sha256': helper.sha(SOURCE),
                  'Torch_imported': 'torch' in sys.modules}))
