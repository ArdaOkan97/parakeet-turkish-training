"""CPU-only recovery lifecycle guards; every child/inference is mocked."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUTPUT))
import run_experiment_loop_recovery as controller
import launch_with_io_retry_v1 as launcher
import benchmark  # Complete library imports before scoped import-module mocks.
from atomic_io_retry_v1 import write_json, read_json


def main():
    checks=[]
    def require(value,name):
        assert value,name
        checks.append(name)
    args=controller.parser().parse_args([])
    require(not hasattr(args,'hours'),'No duration reset option')
    require(args.audit_worker=='verify_recovery_results_v1.py','Versioned recovery auditor default')
    require(args.strong_test_worker=='benchmark_strong_tr_test_v2.py','Versioned longform TEST worker default')
    require(args.finalization_cap_seconds==1800,'Post-selection cap remains finite')
    require('False' in (OUTPUT/'run_experiment_loop_recovery.py').read_text(), 'No numerical equivalence assertion introduced')
    # Dispatch the launcher through a fake worker. No model or GPU call occurs.
    scratch=ROOT/'work/research/recovery_launcher_cpu'
    scratch.mkdir(exist_ok=True)
    dispatch=[]
    class Worker:
        def parser(self):
            result=argparse.ArgumentParser();result.add_argument('--run-id');return result
        def run(self,parsed):dispatch.append(parsed.run_id)
        def save_training_state(self,path,*values,**keywords):Path(path).write_bytes(b'complete-checkpoint')
    for name in ('train_lora_batched.py','finalize_lora.py'):
        fake=Worker()
        arguments=launcher.parser().parse_args(['--worker',name,'--provenance',str(scratch/(name+'.json')),'--','--run-id','cpu-only'])
        with patch.object(launcher.importlib,'import_module',return_value=fake):launcher.main(arguments)
        evidence=read_json(arguments.provenance,{})
        require(evidence['status']=='complete',name+' mocked dispatch completes')
        require(evidence['worker_sha256']==evidence['worker_sha256_after'],name+' frozen worker SHA unchanged')
        require(evidence['helper_sha256']==evidence['helper_sha256_after'],name+' helper SHA unchanged')
        if name.startswith('train_'):
            target=scratch/'checkpoint.pt';fake.save_training_state(target)
            require(target.read_bytes()==b'complete-checkpoint','Wrapped original checkpoint serialization retained')
    require(dispatch==['cpu-only','cpu-only'],'Both scoped worker APIs use run(args) correctly')
    # Exercise controller C -> B2 -> inherited-config extension -> final stages.
    stamp=datetime.now(timezone.utc).strftime('%H%M%S%f')
    loop_id='cpu_recovery_guard_'+stamp
    args=controller.parser().parse_args(['--loop-id',loop_id,
        '--strong-test-worker','benchmark_strong_tr_test.py','--audit-worker','verify_adaptive_results.py'])
    pilot_weights=scratch/'pilot.pt';pilot_weights.write_bytes(b'cpu-fake-weights')
    real_validate=controller.validate_selection
    selections={}
    commands=[]
    provenance={'selected_step':1200,'weights_sha256':controller.sha(pilot_weights),'code_sha256':{},'input_sha256':{}}
    def selection(run_id):
        if run_id in selections:return selections[run_id],dict(provenance,selected_step=selections[run_id]['selected_step'])
        return {'accepted':True,'run_id':run_id,'selected_step':1200,'weights':str(pilot_weights),
                'turkish_validation_wer':.824 if run_id==controller.PILOT_RUN else .7272727272727273},provenance
    def io_record(command):
        worker=command[command.index('--worker')+1];path=Path(command[command.index('--provenance')+1])
        evidence={'status':'complete','training_semantics_changed':False,
                  'arguments':command[command.index('--')+1:]}
        for key,name in [('worker_sha256',worker),('launcher_sha256','launch_with_io_retry_v1.py'),('helper_sha256','atomic_io_retry_v1.py')]:
            evidence[key]=evidence[key+'_after']=controller.sha(OUTPUT/name)
        write_json(path,evidence)
    class Child:
        def __init__(self,command,**keywords):
            commands.append(command);self.pid=999999;self.returncode=0
            run_id=command[command.index('--run-id')+1]
            run_dir=OUTPUT/'runs'/run_id;run_dir.mkdir()
            name=run_id.removeprefix(loop_id+'_')
            score={'screen_C':.74,'screen_B2':.70,'extend_1':.69,'extend_2':.691}[name]
            epochs=float(command[command.index('--epochs')+1])
            selections[run_id]={'accepted':True,'run_id':run_id,'selected_step':3204,'completed_steps':3204,
                'weights':str(pilot_weights),'turkish_validation_wer':score,'turkish_epochs_completed':epochs,
                'selection_split':'FLEURS validation only','heldout_tests_evaluated':False}
            write_json(run_dir/'selection.json',selections[run_id]);io_record(command)
        def poll(self):return self.returncode
    def run(command,**keywords):
        commands.append(command)
        freeze=read_json(OUTPUT/'runs'/loop_id/'frozen_final_selection.json',{})
        if '--worker' in command:
            winner=freeze['run_id'];selections[winner].update(merged=True,heldout_tests_evaluated=True,
                merged_development_gate={'passed':True,'turkish_wer':.69})
            write_json(OUTPUT/'runs'/winner/'selection.json',selections[winner]);io_record(command)
        elif '--selected-run-id' in command:
            write_json(OUTPUT/'adaptive_strong_baseline_test_comparison_v2.json',{
                'status':'complete','passed':True,'run_id':freeze['run_id'],
                'frozen_selection_sha256':controller.sha(OUTPUT/'runs'/loop_id/'frozen_final_selection.json')})
        else:
            write_json(OUTPUT/'runs'/loop_id/'adaptive_results_verification.json',{
                'passed':True,'run_id':freeze['run_id'],'loop_id':loop_id,'selected_step':freeze['selected_step'],
                'frozen_selection_sha256':controller.sha(OUTPUT/'runs'/loop_id/'frozen_final_selection.json')})
        return SimpleNamespace(returncode=0)
    # Guard creates only synthetic new-run artifacts. Preserve output-root files.
    saved={path:path.read_bytes() if path.exists() else None for path in
           [OUTPUT/'adaptive_strong_baseline_test_comparison_v2.json',OUTPUT/'adaptive_experiment_recovery_report.json']}
    try:
        with patch.object(controller,'validate_selection',side_effect=selection), \
             patch.object(controller,'subprocess',SimpleNamespace(Popen=Child,run=run,CREATE_NO_WINDOW=0,TimeoutExpired=TimeoutError,STDOUT=-2)), \
             patch.object(controller,'keep_awake',return_value={'supported':False,'enabled':False}), \
             patch.object(controller,'backlog_note',return_value=None):
            controller.main(args)
        state=read_json(OUTPUT/'runs'/loop_id/'loop_status.json',{})
        require(state['status']=='complete','Mocked complete lifecycle reaches final independent audit')
        origin=read_json(OUTPUT/'runs'/args.original_loop_id/'loop_status.json',{})
        require(state['deadline_at_utc']==origin['deadline_at_utc'],'Original absolute deadline remains exact')
        b2=[entry for entry in state['trials'] if entry['name']=='screen_B2'][0]
        require(b2['effective_batch_size']==4 and b2['selected_configuration']['batch_size']==2,'Physical B2 keeps effective4')
        require(b2['batch_quality_experiment'] and b2['equal_requested_exposure_completed'],'B2 experimental status and equal exposure explicit')
        for entry in state['trials']:
            if entry['name'].startswith('extend_'):
                require(entry['selected_configuration']['batch_size']==2 and entry['selected_configuration']['accumulation']==2,
                        entry['name']+' inherits DEV-qualified B2 configuration')
        require(state['best_development_candidate']['run_id'].endswith('extend_1'),'DEV score alone selects winner')
        require(state['selected_configuration']['batch_size']==1,'Measured B1 profile remains distinct from experimental B2')
        require(all('--resume-weights' in command for command in commands if '--epochs' in command),'Fresh recovery optimizer policy explicit; no fake exact resume')
        require(state['finalizer_io_retry_provenance_sha256'],'Actual finalizer I/O provenance retained')
        require(all(entry.get('batch_numerical_equivalence_proven') is False for entry in state['trials']),
                'Neither B1 nor B2 is relabeled as strict numerical equivalence')
    finally:
        for path,payload in saved.items():
            if payload is None:
                if path.exists():path.unlink()
            else:path.write_bytes(payload)
    report={'passed':True,'check_count':len(checks),'checks':checks,
            'scope':'CPU synthetic lifecycle; every trainer/finalizer/TEST/audit child mocked; no real training or TEST inference',
            'controller_sha256':controller.sha(OUTPUT/'run_experiment_loop_recovery.py'),
            'launcher_sha256':controller.sha(OUTPUT/'launch_with_io_retry_v1.py'),
            'helper_sha256':controller.sha(OUTPUT/'atomic_io_retry_v1.py'),
            'synthetic_loop_id':loop_id,'guard_script_sha256':controller.sha(__file__)}
    write_json(OUTPUT/'recovery_controller_cpu_verification.json',report)
    print(json.dumps({'passed':True,'check_count':len(checks)}))


if __name__=='__main__':main()
