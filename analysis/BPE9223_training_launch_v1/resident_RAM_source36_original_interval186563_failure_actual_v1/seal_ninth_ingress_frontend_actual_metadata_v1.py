"""Seal only the new admitted ninth ingress proof metadata; no candidate import or subprocess."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import psutil

S = Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20/outputs/parakeet_turkish')
sys.path.insert(0, str(S))
from resource_watchdog_v2 import sample_memory

F = Path(__file__).parent
R = F / 'ROOT_ninth_CPU14_one_original_ingress_frontend_exact_ONCE_execution_scope_v1.json'
E = F / 'ninth_ingress_frontend_external_result_v1.json'
REPORT = F / 'ninth_ingress_frontend_actual_01/report.json'
TOOL = F / 'ninth_ingress_frontend_actual_TOOL_result_v1.json'
OUT = F / 'ninth_ingress_frontend_actual_metadata_completion_v1.json'
PREFLIGHT = F / 'ninth_ingress_frontend_manual_prelaunch_gate_v1.json'
INITIAL = F / 'ninth_ingress_frontend_manual_prelaunch_numeric_sample_v1.json'


def require(value, text):
    if not value:
        raise ValueError(text)


def identity(p):
    return {'pid': p.pid, 'create_time': p.create_time(), 'ppid': p.ppid()}


def same_alive(w):
    try:
        return abs(psutil.Process(w['pid']).create_time() - w['create_time']) < .001
    except psutil.NoSuchProcess:
        return False


def pin(p):
    p = Path(p).resolve()
    require(p.stat().st_size <= 8388608, 'Compact metadata/source only')
    b = p.read_bytes()
    return {'path': str(p), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}


def write(p, v):
    with p.open('x', encoding='utf-8', newline='\n') as h:
        json.dump(v, h, indent=2)
        h.write('\n')
        h.flush()
        os.fsync(h.fileno())


def main():
    began = time.monotonic()
    me = psutil.Process()
    me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    mine = identity(me)
    parent = identity(me.parent())
    require(not OUT.exists() and not TOOL.exists(), 'Fresh immutable metadata completion')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
            and os.environ.get('PYTHONUTF8') == '1'
            and all(os.environ.get(k) == '1' for k in
                    ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
            and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),
            'Original process-local CPU metadata environment')
    samples = [sample_memory(p.pid, 'ninth_ingress_metadata_completion_initial') for p in (me, me.parent())]
    numeric = {'samples': samples, 'sampled_family_private_bytes':
               sum(s['worker_private_memory']['private_bytes'] for s in samples),
               'E_free_bytes': shutil.disk_usage('E:/').free,
               'prospective_metadata_bytes': 536870912,
               'commit_threshold_bytes': 5055306368, 'physical_threshold_bytes': 2305306368}
    write(F / 'ninth_ingress_metadata_completion_initial_numeric_sample_v1.json', numeric)
    require(all(s['paired_sampling_within_one_second']
                and s['system_memory']['commit_available_bytes'] >= 5055306368
                and s['system_memory']['physical_available_bytes'] >= 2305306368 for s in samples)
            and numeric['E_free_bytes'] >= 100000000000
            and numeric['sampled_family_private_bytes'] <= 536870912, 'Full metadata admission before reads')
    require(pin(R)['sha256'] == '986900fc21771b9ea5b472dc0a330d27273462214fcd63deed53552a41d9203a', 'Actual exact ROOT scope')
    review = json.loads(R.read_text(encoding='utf-8'))
    external = json.loads(E.read_text(encoding='utf-8'))
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    manual = json.loads(PREFLIGHT.read_text(encoding='utf-8'))
    initial = json.loads(INITIAL.read_text(encoding='utf-8'))
    require(external['report'] == pin(REPORT), 'Actual report pin')
    require(external['passed'] is True and external['Popen_returncode'] == 0
            and external['external_guard_error'] is None and external['required_external_cleanup'] == [],
            'Actual external clean outcome')
    require(report['passed'] is True and report['guard_error'] is None
            and report['input_pins_before_after_exact'] is True
            and report['input_pins_before'] == report['input_pins_after'] == review['input_pins']
            and external['input_pins_before'] == external['input_pins_after'] == review['input_pins']
            and len(review['input_pins']) == 51, 'Actual all51 before/after equality')
    require(manual['passed'] is True and initial['admission_predicate_evaluated'] is False
            and initial['candidate_imported'] is False, 'Original pre-predicate sample retained')
    require(len(report['negative_controls']) == 8
            and all(v['rejected'] is True for v in report['negative_controls'])
            and report['all_stubs_restored'] is True, 'Actual prescribed strict negatives and restoration')
    require(report['actual_geometry'] == {'clean16k_samples': 210280, 'valid_mel_frames': 1314,
                                        'physical_feature_frames': 1315, 'encoder_frames': 165}
            and len(report['reference']['full_target_ids']) == 66
            and report['raw_buffers_released_and_metadata_handles_closed'] is True,
            'Actual geometry/full66 and source-bound raw/DB closure receipt')
    identities = {}
    for w in [*external['recorded_identities'], external['external_identity'], external['launcher_identity'],
              external['original_Popen_identity'], report['producer'], manual['identity'], manual['launcher_identity']]:
        identities[(w['pid'], w['create_time'])] = w
    closure = [{'identity': w, 'exact_dead': not same_alive(w)} for w in identities.values()]
    require(all(c['exact_dead'] for c in closure), 'All observed proof/manual identities exact dead')
    training = [{'pid': 21912, 'create_time': 1792780981.1092248},
                {'pid': 37108, 'create_time': 1792780981.1024935},
                {'pid': 29600, 'create_time': 1792780979.698954},
                {'pid': 39856, 'create_time': 1792790240.3032615},
                {'pid': 35440, 'create_time': 1792790240.2869232}]
    readers = [{'pid': 26892, 'create_time': 1792652904.3359847},
               {'pid': 30128, 'create_time': 1792652904.283631}]
    td = [{'identity': w, 'exact_dead': not same_alive(w)} for w in training]
    ra = [{'identity': w, 'exact_alive': same_alive(w)} for w in readers]
    require(all(v['exact_dead'] for v in td) and all(v['exact_alive'] for v in ra), 'Protected identities unchanged')
    reader = Path('E:/gpu_training_scratchpad/analysis/BPE9223_Trackio_live_v1/current_status.json')
    require(reader.stat().st_size <= 65536, 'Only compact current reader status')
    reader_compact = {'pin': pin(reader), 'content': json.loads(reader.read_text(encoding='utf-8'))}
    outputs = [pin(p) for p in (REPORT, E, INITIAL, PREFLIGHT,
               F / 'ninth_ingress_frontend_stdout01.log', F / 'ninth_ingress_frontend_stderr01.log',
               Path(report['configuration']['path']))]
    tool = {'kind': 'Actual_owned_ninth_ingress_frontend_TOOL_result_v1', 'sole_owner': '/root/memory_diagnosis',
            'exec_session_id': 19814, 'initial_chunk_id': '3a9af9',
            'yield_chunks': ['fa6b2e', '7bd333'], 'terminal_chunk_id': 'df6a9c',
            'TOOL_exit_code': 0, 'actual_external_stdout_terminal':
            {'external_result': pin(E), 'passed': True, 'Popen_returncode': 0,
             'guard_error': None, 'cleanup': [], 'sampled_family_peak': 457441280},
            'separate_individual_OS_exit_codes': 'UNKNOWN unless actual Popen return explicitly persisted',
            'no_retry_or_production_action': True}
    write(TOOL, tool)
    final_sample = sample_memory(me.pid, 'ninth_ingress_metadata_completion_final')
    require(time.monotonic() - began < 180
            and final_sample['worker_private_memory']['private_bytes'] +
            sample_memory(parent['pid'], 'ninth_ingress_metadata_completion_parent')['worker_private_memory']['private_bytes'] <= 536870912
            and final_sample['system_memory']['commit_available_bytes'] >= 2268435456
            and final_sample['system_memory']['physical_available_bytes'] >= 1768435456
            and shutil.disk_usage('E:/').free >= 100000000000, 'Metadata final runtime bounds')
    result = {'kind': 'NINTH_ingress_frontend_actual_metadata_completion_v1', 'passed': True,
              'ROOT_execution_scope': pin(R), 'actual_TOOL_result': pin(TOOL), 'report': pin(REPORT),
              'external_result': pin(E), 'new_output_metadata_pins': outputs,
              'actual_TOOL_exit_code': 0, 'actual_Popen_returncode': 0,
              'inner_guard_error': report['guard_error'], 'outer_guard_error': external['external_guard_error'],
              'external_cleanup': external['required_external_cleanup'],
              'input_pins_before_after_exact': True, 'input_pin_count': 51,
              'observed_job_and_manual_identity_closure_records': closure,
              'all_recorded_job_and_manual_identities_exact_dead': True,
              'completion_identity': mine, 'completion_launcher_identity': parent,
              'completion_pair_alive_at_write': [same_alive(mine), same_alive(parent)],
              'completion_pair_exact_deaths_require_after_return_check': True,
              'original_pre_predicate_numeric_sample': pin(INITIAL),
              'completion_initial_numeric_sample': pin(F / 'ninth_ingress_metadata_completion_initial_numeric_sample_v1.json'),
              'completion_final_resource_sample': final_sample,
              'actual_geometry': report['actual_geometry'], 'full_target_count': 66,
              'prescribed_negative_count': 8, 'all_prescribed_negatives_rejected': True,
              'scientific_and_method_limits': 'One actual original ingress/frontend and strict controls; delegation/geometry-loader METHOD_STUBs labelled in report. No actual transport test, production restoration, fit or speed proof.',
              'method_close_evidence': 'Actual successful source-bound raw ledger/reader closure receipt; process deaths alone are not method closure proof.',
              'protected_training_exact_deaths': td, 'protected_reader_exact_alive': ra,
              'current_compact_reader_status': reader_compact,
              'no_old_journal_payload_audio_or_closed_study_read': True,
              'production_remains_stopped_no_source37_prepared': True,
              'sealer_source': pin(__file__), 'sealer_wall_seconds': time.monotonic() - began}
    write(OUT, result)
    print(json.dumps({'completion': pin(OUT), 'TOOL_receipt': pin(TOOL), 'report': pin(REPORT),
                      'external_result': pin(E), 'observed_exact_dead_count': len(closure),
                      'completion_identity': mine, 'completion_launcher_identity': parent}), flush=True)


if __name__ == '__main__':
    main()
