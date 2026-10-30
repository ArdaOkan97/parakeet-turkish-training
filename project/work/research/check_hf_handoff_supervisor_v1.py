"""Tiny stdlib handoff guards, plus real OWNED CPU-Python exits0/7; no ML/GPU."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(OUT))
import supervise_hf_same_science_v1 as s


def main(fixture_args):
    checks=[]
    def check(name,value):
        if not value: raise AssertionError(name)
        checks.append(name)
    check('no_ML_imports',all(name not in sys.modules for name in ('torch','nemo','transformers')))
    scratch=ROOT/'work/research'/('handoff_fixture_'+uuid.uuid4().hex);scratch.mkdir()
    actual=[]
    for code in (0,7):
        child=subprocess.Popen([sys.executable,'-c',f'import time,sys;time.sleep(.3);sys.exit({code})'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        handle=s.OwnedProcess(child.pid,'tiny_owned_CPU_fixture')
        try:
            check(f'actual_live_handle_{code}',handle.poll() is None and handle.parent_pid>0 and handle.creation_ticks>0)
            check(f'actual_exit_{code}',child.wait(timeout=10)==code and handle.poll()==code)
            actual.append(handle.evidence())
        finally:handle.close()
    run_id='CPU_handoff_fixture_'+uuid.uuid4().hex
    run_dir=OUT/'runs'/run_id;run_dir.mkdir()
    (run_dir/'run_config.json').write_text(json.dumps({'steps':2,'config_identity':{'training_arguments':{'steps':3}}}))
    folder=scratch/'window01';folder.mkdir()
    worker_sha=s.sha(OUT/'train_lora_batched_data_v11.py')
    def write_window(path,completed,reason='soft_max_wall_seconds',status='train_window_ended'):
        path.mkdir(exist_ok=True)
        paths={key:path/(key+'.json') for key in ('fresh_runtime','io','cache','memory','identity')}
        identity={'kind':'owned_training_worker_identity_v1','pid':902,'ppid':901,'expected_run_id':run_id,
            'frozen_worker_sha256':worker_sha,'process_creation_identity':{'available':True,'pid':902,'creation_filetime_100ns':200}}
        paths['identity'].write_text(json.dumps(identity))
        for key in ('io','cache','memory'):paths[key].write_text(json.dumps({'status':'complete'}))
        paths['fresh_runtime'].write_text(json.dumps({'status':'ready','run_id':run_id,'pid':902}))
        paths['fresh_runtime'].with_suffix('.completion.json').write_text(json.dumps({'kind':'HF_same_science_native_preload_resume_completion_v1',
            'status':'complete','run_id':run_id,'pid':902,'sources_unchanged':True,'fresh_runtime_proof_sha256':s.sha(paths['fresh_runtime']),
            'launcher_sha256_after':s.PRELOADER_SHA}))
        selection_path=run_dir/'selection.json'
        selection_path.write_text(json.dumps({'accepted':True,'run_id':run_id,'completed_steps':completed,'configured_steps':2,'selected_step':0,'stop_early_reason':reason}))
        state={'kind':'HF_same_run_exact_science_continuation_v1','run_id':run_id,'status':status,'child_exit_code':0,
            'completed_steps':completed,'selection':s.record(selection_path),'resource_review_required':False,
            'controller_sha256':s.HELPER_SHA,'new_preloader':{'sha256':s.PRELOADER_SHA},
            'runtime_source_sha256':{'train_lora_batched_data_v11.py':worker_sha},'owned_launcher_pid':901,
            'paths':{key:str(value) for key,value in paths.items()},'scientific_arguments_unchanged':True,'optimizer_sampler_RNG_counters_reset':False}
        s.write(path/'loop_status.json',state)
        return state
    class MockHandle:
        def __init__(self,pid,role):
            self.pid,self.role=pid,role;self.parent_pid={900:899,901:900,902:901}[pid]
            self.creation_ticks={900:100,901:150,902:200}[pid];self.exit=None;self.closed=False
        def poll(self):return self.exit
        def evidence(self):return {'pid':self.pid,'role':self.role,'exit_code_observed_from_retained_handle':self.exit,'explicit_fixture_mock':True}
        def close(self):self.closed=True
    write_window(folder,1,status='running')
    handles,attachment=s.attach_window(folder,run_id,expected_launcher=901,expected_worker=902,process_factory=MockHandle)
    check('ownership_attaches_all3_scoped_handles',len(handles)==3 and attachment['controller_parent_derived_only_from_owned_launcher'])
    for handle in handles:handle.exit=0
    write_window(folder,1)
    check('softwall_all0_eligible',s.closed_window(folder,run_id,handles)['automatic_budget_handoff_allowed'])
    for reason in ('consecutive_retention_gate_failures','development_plateau','development_target_met','configured_steps_complete','unknown'):
        write_window(folder,1,reason)
        check('no_bypass_'+reason,not s.closed_window(folder,run_id,handles)['automatic_budget_handoff_allowed'])
    write_window(folder,1)
    handles[0].exit=7
    try:s.closed_window(folder,run_id,handles)
    except ValueError:check('nonzero_real_exit_blocks_closure',True)
    else:raise AssertionError('failed exit accepted')
    handles[0].exit=0
    state=write_window(folder,1,status='resource_review_required')
    try:s.closed_window(folder,run_id,handles)
    except ValueError:check('resource_stop_blocks_closure',True)
    else:raise AssertionError('resource stop accepted')
    write_window(folder,1)
    for handle in handles:handle.close()
    command=s.successor_command(folder,scratch/'new02',run_id,4.5)
    check('exact_pinned_existing_helper_flags',command[1]==str(s.HELPER) and '--source-closed-confirmed' in command and '--gpu-window-authorized' in command
        and '--resume-weights' not in command and '--source-loop' in command)
    # Actual supervisor state machine, with explicit mocked CPU-only child.
    launched=[]; original_attach=s.attach_window;original_popen=s.subprocess.Popen
    def mock_attach(path,run,**kwargs):
        state=write_window(path,1 if Path(path)==folder else 2)
        values=[MockHandle(pid,role) for pid,role in ((901,'launcher'),(902,'worker'),(900,'controller'))]
        for value in values:value.exit=0
        return values,{'explicit_fixture_mock':True,'run_id':run,'source_loop':str(path)}
    class MockChild:
        pid=777
        def __init__(self,cmd,**kwargs):
            launched.append(cmd);target=Path(cmd[cmd.index('--output-dir')+1]);write_window(target,2,status='running')
        def poll(self):return None
        def wait(self,timeout=None):return 0
    s.attach_window=mock_attach;s.subprocess.Popen=MockChild
    args=SimpleNamespace(source_loop=folder,run_id=run_id,supervisor_dir=scratch/'supervisor',launcher_pid=901,worker_pid=902,
        successor_prefix='same_science_fixture',window_hours=4.5,max_hours=12,poll_seconds=5,attach_only=False,gpu_window_authorized=True)
    try:
        check('mocked_supervised_handoff_exit0',s.main(args)==0)
        status=s.read(args.supervisor_dir/'supervisor_status.json',{})
        check('exactly1_successor_horizon_no_goal_claim',len(launched)==1 and status['status']=='next_experiment_required'
            and status['objective_complete'] is False and status['stop_reason']=='configured_horizon')
        check('closed_handles_receipts_before_successor',len(status['windows'])==2 and launched[0][launched[0].index('--source-loop')+1]==str(folder))
        check('effective_horizon_distinct_from_declared_cap',status['configured_optimizer_horizon_unchanged']==2
            and status['original_scientific_argument_step_cap_unchanged']==3)
    finally:s.attach_window=original_attach;s.subprocess.Popen=original_popen
    check('no_ML_imports_after',all(name not in sys.modules for name in ('torch','nemo','transformers')))
    report={'kind':'same_science_supervisor_stdlib_CPU_verification_v1','passed':True,'checks':checks,'check_count':len(checks),
        'actual_owned_CPU_process_exit_proofs':actual,'actual_GPU_or_model_or_next_training_launch':False,
        'actual_current_worker_attachment':'pending separate ROOT attach-only route','mocked_next_helper_lifecycle':True,
        'fixture_directory':str(scratch),'supervisor_sha256':s.sha(s.__file__),'fixture_sha256':s.sha(__file__)}
    path=(fixture_args.output or OUT/('HF_same_science_supervisor_CPU_verification_'+uuid.uuid4().hex+'.json')).resolve()
    assert path.is_relative_to(ROOT) and not path.exists()
    with path.open('x',encoding='utf-8') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'checks':len(checks),'report':str(path),'source_sha256':report['supervisor_sha256']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    main(parser.parse_args())
