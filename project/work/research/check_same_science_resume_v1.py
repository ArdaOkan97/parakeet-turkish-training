"""Tiny stdlib fixtures only: fresh preload ordering and closed-source guards."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUT))
import launch_same_science_resume_v1 as launcher
import resume_hf_same_science_v1 as controller
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def check(name,ok):
    if not ok:raise AssertionError(name)
    checks.append(name)
base_args=argparse.Namespace(source_loop=OUT/'runs/adaptive_20261011_hf_quality_window_01',
    run_id='adaptive_20261011_hf_quality_window_01_stronger_kd',output_dir=OUT/'runs/adaptive_20261011_hf_exact_followon_01',
    source_closed_confirmed=False,window_hours=4.5,poll_seconds=5.,prepare_only=True)
plan=controller.source_plan(base_args,False)
check('prepare_only_no_checkpoint_archive',not Path(plan['archive_path']).exists())
check('prepare_only_actual_frozen_science_and_ready_unchanged',plan['max_retention_failures_unchanged']==2
    and plan['original_ready']['path']==controller.quality.runtime.option(plan['worker_arguments'],'--native-backend-provenance'))
check('no_sampler_or_optimizer_reset',plan['optimizer_sampler_RNG_counters_reset'] is False)
check('same_run_id_preserved',controller.quality.runtime.option(plan['worker_arguments'],'--run-id')==base_args.run_id)
check('exact_flag_replaces_warm_flag','--resume-exact' in plan['worker_arguments'] and '--resume-weights' not in plan['worker_arguments'])
check('no_Torch_or_NeMo_imports','torch' not in sys.modules and 'nemo' not in sys.modules)

with tempfile.TemporaryDirectory(dir=ROOT/'work/research',prefix='same_science_resume_stdlib_') as folder:
    folder=Path(folder)
    fake_parent=folder/'running_parent';fake_parent.mkdir()
    (fake_parent/'loop_status.json').write_text(json.dumps({'status':'running','trials':[{'run_id':base_args.run_id,
        'worker_arguments':plan['worker_arguments'],'child_exit_code':None,'status':'running'}]}),encoding='utf-8')
    bad=copy.copy(base_args);bad.source_loop=fake_parent;bad.output_dir=folder/'closed_only';bad.source_closed_confirmed=True
    try:controller.source_plan(bad,True)
    except ValueError:checks.append('running_parent_rejected_before_checkpoint_copy')
    else:raise AssertionError('A running parent was accepted')
    check('live_checkpoint_not_copied',not bad.output_dir.exists())
    bad.source_closed_confirmed=False
    try:controller.source_plan(bad,True)
    except ValueError:checks.append('root_closure_assertion_required')
    else:raise AssertionError('Root confirmation omitted')
    original=Path(plan['original_ready']['path']);original_sha=sha(original)
    values=plan['command'][plan['command'].index('--')+1:]
    old_preload=sys.modules.get('preload_nemo_linker_v1');preload=types.ModuleType('preload_nemo_linker_v1')
    seen=[];old_run_path=launcher.runpy.run_path;original_argv=list(sys.argv)
    def fake_preload():
        check('actual_function_invoked_before_any_Torch_import','torch' not in sys.modules)
        seen.append('preload')
        return {'kind':'coherent_native_cuda12_9_preload_v1','module_sha256':launcher.PINS['preload_nemo_linker_v1.py'],
            'actual_binding_linker':{'version':[12,9]},'cuda_initialized':False,'CPU_fixture_not_hardware_proof':True}
    preload.preload=fake_preload;sys.modules['preload_nemo_linker_v1']=preload
    def fake_dispatch(path,run_name):
        check('preload_precedes_memory15_dispatch',seen==['preload'])
        check('dispatch_keeps_frozen_memory15',Path(path)==OUT/'launch_with_memory_guard_v15.py' and run_name=='__main__')
        check('original_ready_stays_worker_identity_token',controller.quality.runtime.option(sys.argv,'--native-backend-provenance')==str(original))
        seen.append('dispatch')
    launcher.runpy.run_path=fake_dispatch
    args=argparse.Namespace(original_native_provenance=original,original_native_provenance_sha256=original_sha,
        fresh_runtime_provenance=folder/'fresh_success.json',memory_arguments=values,validate_only=False)
    try:
        launcher.main(args)
        proof=json.loads(args.fresh_runtime_provenance.read_text(encoding='utf-8'))
        completion=json.loads(args.fresh_runtime_provenance.with_suffix('.completion.json').read_text(encoding='utf-8'))
        check('fresh_runtime_records_current_self_PID',proof['pid']==os.getpid() and proof['ppid']==os.getppid())
        check('old_PID_explicitly_original_not_fresh',proof['original_first_start_ready']['original_first_start_pid']
            ==json.loads(original.read_text(encoding='utf-8')).get('pid') and proof['original_ready_scope'].startswith('Immutable scientific'))
        check('fresh_proof_separate_and_original_ready_unchanged',args.fresh_runtime_provenance!=original and sha(original)==original_sha)
        check('actual_fresh_completion_crossbinds_proof',completion['status']=='complete' and completion['sources_unchanged']
            and completion['fresh_runtime_proof_sha256']==sha(args.fresh_runtime_provenance))
        check('argv_restored_after_success',sys.argv==original_argv)
        controller.validate_fresh_runtime(proof,plan,{'pid':os.getpid()})
        checks.append('controller_validates_actual_fresh_PID_proof')
        tampered=copy.deepcopy(proof);tampered['pid']+=1
        try:controller.validate_fresh_runtime(tampered,plan,{'pid':os.getpid()})
        except ValueError:checks.append('old_or_wrong_runtime_PID_rejected')
        else:raise AssertionError('Wrong fresh PID accepted')
        def failure(path,run_name):raise RuntimeError('CPU fixture dispatch failure')
        launcher.runpy.run_path=failure;seen.clear();args.fresh_runtime_provenance=folder/'fresh_failed.json'
        try:launcher.main(args)
        except RuntimeError as e:check('original_dispatch_exception_preserved',str(e)=='CPU fixture dispatch failure')
        else:raise AssertionError('Dispatch error swallowed')
        failure_completion=json.loads(args.fresh_runtime_provenance.with_suffix('.completion.json').read_text(encoding='utf-8'))
        check('failed_completion_retained',failure_completion['status']=='failed' and failure_completion['sources_unchanged'])
        check('argv_restored_after_failure',sys.argv==original_argv)
        def preload_failure():raise RuntimeError('CPU fixture preload failure')
        preload.preload=preload_failure;args.fresh_runtime_provenance=folder/'preload_failed.json'
        try:launcher.main(args)
        except RuntimeError as e:check('original_preload_exception_preserved',str(e)=='CPU fixture preload failure')
        else:raise AssertionError('Preload failure swallowed')
        check('startup_failure_has_separate_durable_proof',json.loads(args.fresh_runtime_provenance.read_text())['status']=='failed'
            and json.loads(args.fresh_runtime_provenance.with_suffix('.completion.json').read_text())['status']=='failed')
        check('original_ready_immutable_even_after_errors',sha(original)==original_sha)
    finally:
        launcher.runpy.run_path=old_run_path
        if old_preload is None:sys.modules.pop('preload_nemo_linker_v1',None)
        else:sys.modules['preload_nemo_linker_v1']=old_preload
        sys.argv=original_argv
report={'schema':1,'kind':'HF_same_science_resume_stdlib_CPU_preparation_v1','passed':True,'checks':checks,
    'check_count':len(checks),'source_sha256':{'launcher':sha(Path(launcher.__file__)),'controller':sha(Path(controller.__file__))},
    'GPU_launch':False,'actual_fresh_preload_executed':False,'actual_optimizer_restore_verified':False,
    'live_checkpoint_copied':False,'scope':'Actual stdlib launcher/controller functions with mocked preload/memory dispatch; no Torch/NeMo imports or model/optimizer loading.'}
with (OUT/'HF_same_science_resume_CPU_verification_v1.json').open('x',encoding='utf-8') as stream:
    json.dump(report,stream,indent=2)
print(json.dumps({'passed':True,'checks':len(checks),'GPU_launch':False,'Torch_imported':'torch' in sys.modules}))
