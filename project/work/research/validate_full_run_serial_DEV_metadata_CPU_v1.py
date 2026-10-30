"""Inactive DEV glue syntax/import/negative metadata controls, no evaluation."""
import argparse
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT/'outputs/parakeet_turkish'
sys.path.insert(0, str(SOURCE))
import asr18_full_run_serial_DEV_v1 as dev
from fastmss_row_serial_cpu_v4 import environment, resource_guard


def main(output):
    environment(); resource_guard()
    output = dev.new_E(output); dev.require(not output.exists(), 'Fresh E-only metadata fixture'); output.mkdir()
    checks = []
    def check(name, condition): dev.require(condition, name); checks.append(name)
    def reject(name, function):
        try: function()
        except (ValueError, KeyError, TypeError, FileNotFoundError): checks.append(name); return
        raise AssertionError('Expected inactive fail-closed rejection: '+name)
    source = SOURCE/'asr18_full_run_serial_DEV_v1.py'
    parsed = ast.parse(source.read_text(encoding='utf-8'))
    check('New glue parses without runtime execution', isinstance(parsed, ast.Module))
    check('Import does not load numerical/frontend/model/audio runtime', not any(name in sys.modules for name in
        ('torch','numpy','scipy','transformers','pyarrow','soundfile','av')))
    check('Worker admission is dynamic separate argument; no future PID/request circular binding',
          'owned_worker_admission' in inspect.signature(dev.evaluate_boundary_checkpoint).parameters)
    check('Concrete fixed original evaluator and gate sources are pinned',
          dev.sources()['metrics.py']['sha256'] == dev.SCORER_SHA
          and dev.sources()['tsc_bpe_full_decoder_transition_v1.py']['sha256'] == dev.GATE_SHA)
    manifests = {}
    for language, path in dev.canonical_DEV_paths().items():
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        check('Original '+language+' manifest metadata is exactly50 validation unique IDs',
              len(rows) == 50 and len({row['source_id'] for row in rows}) == 50
              and all(row['dataset_split'] == 'validation' and row['language'] == language for row in rows))
        manifests[language] = dev.record(path)
    check('Portuguese uses original separate retention directory',
          Path(manifests['pt']['path']).parent.name == 'portuguese_retention_v1')
    pointer = {'kind':'ASR18_production_window_completed_checkpoint_pointer_v2',
        'diagnostic_only':True,'writer_returned_and_payload_verified':True,
        'all_window_updates_committed':True,'committed_updates':20}
    pin = dev.write_new(output/'explicit_synthetic_diagnostic_pointer.json', pointer)
    reject('Finite diagnostic20 checkpoint cannot enter full-run DEV', lambda:
        dev.checked_checkpoint(pin,identity={},expected_updates=20,boundary={}))
    bool_pointer = {**pointer,'diagnostic_only':False,'committed_updates':True}
    pin_bool = dev.write_new(output/'explicit_synthetic_boolean_pointer.json', bool_pointer)
    reject('Boolean optimizer count rejected before checkpoint open', lambda:
        dev.checked_checkpoint(pin_bool,identity={},expected_updates=1,boundary={}))
    stale = {**pointer,'diagnostic_only':False,'committed_updates':2}
    pin_stale = dev.write_new(output/'explicit_synthetic_stale_pointer.json', stale)
    reject('Valid older pointer count cannot describe newer boundary', lambda:
        dev.checked_checkpoint(pin_stale,identity={},expected_updates=3,boundary={}))
    live = dev.write_new(output/'explicit_synthetic_live_idle_not_process0.json',
                         {'kind':'ASR18_resident_native_coherent_idle_boundary_v1'})
    reject('Live coherent idle cannot stand in for actual native exit before DEV', lambda:
        dev.checked_native_boundary(live,request={},pointer={}))
    reject('Missing native process0 record is not an actual closure', lambda:
        dev.checked_native_boundary({'path':str(output/'missing.json'),'bytes':10,'sha256':'0'*64},request={},pointer={}))
    reject('Unknown request kind rejected', lambda: dev.checked_request({'kind':'synthetic_not_execution'}))
    request = {'kind':dev.REQUEST_KIND,'owner':dev.OWNER,'production_enabled':True}
    reject('Armed production request rejected by inactive adapter', lambda: dev.checked_request(request))
    base = {'kind':dev.REQUEST_KIND,'owner':dev.OWNER,'production_enabled':False,
        'action':{'kind':'checkpoint_then_serial350DEV'},'split':'validation','clips':350,'TEST_used':False,
        'native_updates_in_DEV':0}
    reject('Source identity/ROOT policy cannot be omitted', lambda: dev.checked_request(base))
    for name, delta in [('TEST split',{'split':'test'}),('incomplete349',{'clips':349}),
                        ('TEST use',{'TEST_used':True}),('one native update',{'native_updates_in_DEV':1}),
                        ('boolean zero native count',{'native_updates_in_DEV':False})]:
        reject(name+' rejected before any worker/artifact action', lambda delta=delta:
            dev.checked_request({**base,**delta}))
    # This is explicitly an UNACCEPTED draft, never a forged positive ROOT gate.
    policy = {'official_scorer':dev.record(SOURCE/'metrics.py'),
        'integer_gate_contract':dev.record(SOURCE/'tsc_bpe_full_decoder_transition_v1.py'),
        'reference_words':deepcopy(dev.REFERENCES),'retention_limits':deepcopy(dev.LIMITS),'DEV_manifests':manifests}
    policy_pin = dev.write_new(output/'draft_canonical_policy_NOT_ACCEPTED.json',policy)
    review_pin = dev.write_new(output/'draft_ROOT_review_NOT_ACCEPTED.json',
        {'kind':'ROOT_full_run_serial350DEV_policy_and_sources_review_v1','accepted':False,
         'fixture_only':True,'production_approval_claimed':False})
    reject('Real manifest counts alone never substitute for independent ROOT policy review', lambda:
        dev.checked_policy(policy_pin,review_pin,identity={},execution={}))
    for invalid in (True,False,-1,0):
        reject('Exact positive frontier rejects '+repr(invalid), lambda invalid=invalid:dev.integer(invalid,1))
    check('Typed canonical equality distinguishes boolean identity drift', not dev.same({'updates':True},{'updates':1}))
    check('Metadata controls wrote no DEV predictions or native checkpoint',
          not any(path.name.startswith(('validation_','after_update')) for path in output.iterdir()))
    check('Import/control still has no numerical/model/audio runtime', not any(name in sys.modules for name in
        ('torch','numpy','scipy','transformers','pyarrow','soundfile','av')))
    resource_guard()
    report = {'kind':'ASR18_full_run_serial_DEV_metadata_negative_controls_v1','passed':True,
        'check_count':len(checks),'checks':checks,'source':dev.record(__file__),'implementation':dev.record(source),
        'dependencies':dev.sources(),'canonical_DEV_manifest_metadata_read_only':manifests,
        'actual_serial350DEV_executed':False,'all207_tensor_restore_tested':False,
        'actual_owner_worker_or_process0_positive_fixture_forged':False,
        'source_runtime_policy_and_durable_lifecycle_integration_remain_required':True,
        'native_GPU_audio_frontend_or_CACHE_job_started':False,'production_approval_claimed':False}
    dev.write_new(output/'report.json',report)
    print(json.dumps({'passed':True,'checks':len(checks),'DEV_executed':False}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True)
    main(Path(parser.parse_args().output))
