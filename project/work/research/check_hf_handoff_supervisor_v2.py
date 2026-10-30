"""Focused stdlib-only bootstrap/handoff guards; no ML/GPU/controller launch."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(OUT))
import supervise_hf_same_science_v2 as s


def main(args):
    checks=[]
    def check(name, value):
        assert value, name
        checks.append(name)
    def rejects(name, fn):
        try: fn()
        except ValueError: check(name, True)
        else: raise AssertionError(name)
    scratch=ROOT/'work/research'/('handoff_v2_fixture_'+uuid.uuid4().hex); scratch.mkdir()
    actual=[]
    for code in (0, 7):
        child=subprocess.Popen([sys.executable, '-c', f'import time,sys;time.sleep(.15);sys.exit({code})'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        check('actual_owned_live_Popen_'+str(code), child.poll() is None)
        check('actual_owned_exit_'+str(code), child.wait(timeout=10) == code and child.poll() == code)
        actual.append({'pid':child.pid,'exit_code':child.returncode,'scope':'Actual owned CPU-Python Popen, no training'})
    order=s.controller_order_proof()
    check('real_pinned_child_close_publish_source_order', list(order['source_lines'].values()) == sorted(order['source_lines'].values()))
    run_id='CPU_handoff_v2_'+uuid.uuid4().hex; run=OUT/'runs'/run_id; run.mkdir()
    (run/'run_config.json').write_text(json.dumps({'steps':2,'config_identity':{'training_arguments':{'steps':3}}}))
    original=scratch/'original_ready.json'; original.write_text('{}')
    latest=scratch/'latest.pt'; latest.write_bytes(b'explicit tiny mock full state')
    best=scratch/'best.pt'; best.write_bytes(b'explicit tiny mock qualified state')
    worker_sha=s.sha(OUT/'train_lora_batched_data_v11.py'); argv=['--run-id',run_id]
    def window(folder, steps=1, status='running', reason='soft_max_wall_seconds'):
        folder.mkdir(exist_ok=True)
        paths={key:folder/(key+'.json') for key in ('fresh_runtime','identity','io','cache','memory')}
        identity={'pid':902,'ppid':901,'expected_run_id':run_id,'frozen_worker_sha256':worker_sha,
            'process_creation_identity':{'available':True,'pid':902,'creation_filetime_100ns':200}}
        ready={'kind':'HF_same_science_native_preload_resume_v1','status':'ready','run_id':run_id,'pid':902,'ppid':901,
            'launcher_sha256':s.base.PRELOADER_SHA,'source_sha256':{'train_lora_batched_data_v11.py':worker_sha},
            'worker_arguments_sha256':hashlib.sha256(json.dumps(argv,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
            'original_first_start_ready':s.record(original)}
        for key,value in [('identity',identity),('fresh_runtime',ready)]:paths[key].write_text(json.dumps(value))
        for key in ('io','cache','memory'):paths[key].write_text(json.dumps({'status':'complete'}))
        paths['fresh_runtime'].with_suffix('.completion.json').write_text(json.dumps({'kind':'HF_same_science_native_preload_resume_completion_v1',
            'status':'complete','run_id':run_id,'pid':902,'sources_unchanged':True,
            'fresh_runtime_proof_sha256':s.sha(paths['fresh_runtime']),'launcher_sha256_after':s.base.PRELOADER_SHA}))
        (run/'selection.json').write_text(json.dumps({'run_id':run_id,'accepted':True,'completed_steps':steps,'configured_steps':2,
            'selected_step':0,'stop_early_reason':reason,'latest_weights':str(latest),'weights':str(best)}))
        state={'kind':'HF_same_run_exact_science_continuation_v1','run_id':run_id,'status':status,'child_exit_code':0,
            'controller_sha256':s.base.HELPER_SHA,'new_preloader':{'sha256':s.base.PRELOADER_SHA},
            'runtime_source_sha256':{'train_lora_batched_data_v11.py':worker_sha},'scientific_arguments_unchanged':True,
            'optimizer_sampler_RNG_counters_reset':False,'owned_launcher_pid':901,'paths':{k:str(v) for k,v in paths.items()},
            'fresh_runtime':ready,'worker_arguments':argv,'original_ready':s.record(original),'updated_at_utc':s.now(),
            'completed_steps':steps,'selection':s.record(run/'selection.json'),'resource_review_required':False}
        s.write(folder/'loop_status.json',state); return state
    folder=scratch/'window01'; window(folder)
    first=s.bootstrap(folder,run_id,901,902)
    check('initial_live_metadata_identity_not_OS_claim',first['initial_independent_OS_exit_observation'] is False)
    rejects('running_metadata_cannot_authorize_handoff',lambda:s.closure(folder,run_id,initial_bootstrap=first))
    window(folder,status='train_window_ended')
    closed=s.closure(folder,run_id,initial_bootstrap=first)
    check('clean_controller_evidence_budget_eligible',closed['automatic_budget_handoff_allowed'] and closed['actual_supervisor_owned_controller_exit_code'] is None)
    check('actual_saved_checkpoint_hashes_recorded',closed['saved_checkpoint_bytes']['full_latest']==s.record(latest))
    for reason in ('consecutive_retention_gate_failures','development_plateau','development_target_met','configured_steps_complete','unknown'):
        window(folder,status='train_window_ended',reason=reason)
        check('never_bypass_'+reason,not s.closure(folder,run_id)['automatic_budget_handoff_allowed'])
    window(folder,status='train_window_ended')
    class Child:
        pid=777
        def __init__(self, code):self.code=code
        def poll(self):return self.code
        def wait(self,timeout=None):return self.code
    rejects('owned_child_live_blocks_terminal_metadata',lambda:s.closure(folder,run_id,owned_child=Child(None)))
    rejects('owned_child_exit7_blocks_terminal_metadata',lambda:s.closure(folder,run_id,owned_child=Child(7)))
    check('owned_child_exit0_observed',s.closure(folder,run_id,owned_child=Child(0))['actual_supervisor_owned_controller_exit_code']==0)
    state=window(folder,status='train_window_ended');path=Path(state['paths']['cache']);path.write_text('{"status":"failed"}')
    rejects('unfinished_runtime_blocks_handoff',lambda:s.closure(folder,run_id))
    state=window(folder,status='resource_review_required')
    rejects('resource_terminal_blocks_handoff',lambda:s.closure(folder,run_id))
    # Exercise actual main and pinned command constructor, with explicit mocked training children.
    window(folder,status='running');launched=[];orig_popen=s.subprocess.Popen;orig_sleep=s.time.sleep
    class MockChild(Child):
        def __init__(self,command,**kwargs):
            super().__init__(0);launched.append(command);target=Path(command[command.index('--output-dir')+1]);window(target,steps=2,status='train_window_ended')
    def sleep_mock(seconds):window(folder,status='train_window_ended')
    s.subprocess.Popen=MockChild;s.time.sleep=sleep_mock
    try:
        ns=SimpleNamespace(source_loop=folder,run_id=run_id,supervisor_dir=scratch/'supervisor',launcher_pid=901,worker_pid=902,
            successor_prefix='same_science_v2_fixture',window_hours=4.5,max_hours=12,poll_seconds=5,bootstrap_only=False,gpu_window_authorized=True)
        check('actual_main_mocked_bootstrap_handoff',s.main(ns)==0)
        status=s.read(ns.supervisor_dir/'supervisor_status.json',{})
        check('one_successor_realclosure_beforelaunch_no_goal',len(launched)==1 and status['status']=='next_experiment_required'
            and status['objective_complete'] is False and len(status['windows'])==2 and status['windows'][1]['actual_supervisor_owned_controller_exit_code']==0)
        check('effective_horizon_and_original_cap_separate',status['configured_optimizer_horizon_unchanged']==2
            and status['original_scientific_argument_step_cap_unchanged']==3)
    finally:s.subprocess.Popen=orig_popen;s.time.sleep=orig_sleep
    check('no_ML_imports_or_GPU',all(name not in sys.modules for name in ('torch','nemo','transformers')))
    report={'kind':'same_science_supervisor_v2_stdlib_CPU_verification','passed':True,'check_count':len(checks),'checks':checks,
        'actual_owned_CPU_child_exits':actual,'mocked_training_lifecycle':True,'actual_GPU_or_next_training_launch':False,
        'supervisor_sha256':s.sha(s.__file__),'fixture_sha256':s.sha(__file__),'fixture_directory':str(scratch)}
    path=(args.output or OUT/('HF_same_science_supervisor_v2_CPU_'+uuid.uuid4().hex+'.json')).resolve()
    assert path.is_relative_to(ROOT) and not path.exists()
    with path.open('x',encoding='utf-8') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'checks':len(checks),'source_sha256':report['supervisor_sha256'],'report':str(path)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);main(p.parse_args())
