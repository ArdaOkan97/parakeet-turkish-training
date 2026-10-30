"""Small CPU transition/B8 lifecycle guard; every child/inference is mocked."""
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2];OUTPUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUTPUT))
import run_experiment_loop_recovery_v2 as controller
from atomic_io_retry_v1 import write_json,read_json


def main():
    checks=[]
    def require(value,name):assert value,name;checks.append(name)
    args=controller.parser().parse_args([])
    require((args.quality_batch_size,args.quality_accumulation,args.warmup_steps)==(8,2,25),'Authorized B8/effective16/warmup25 defaults')
    require(args.epochs==3 and args.eval_per_epoch==4,'Finite three-epoch cap and requested DEV cadence')
    require(not hasattr(args,'hours'),'No original-deadline reset flag')
    manifest,prior,path=controller.validate_transition(args)
    require(prior['failure_type']=='RuntimeError' and manifest['intentional_transition'],'Intentional interlock transition validated')
    real_read=controller.read
    for field,value in [('prior_controller_exited',False),('user_authorized_throughput_priority',False),
                        ('intentional_transition',False),('target_recovery_loop_id','wrong'),('prior_controller_sha256','0'*64)]:
        changed=copy.deepcopy(manifest);changed[field]=value
        def bad_read(candidate,default=None):return changed if Path(candidate).resolve()==path else real_read(candidate,default)
        with patch.object(controller,'read',side_effect=bad_read):
            try:controller.validate_transition(args)
            except RuntimeError:checks.append('Reject changed transition '+field)
            else:raise AssertionError(field)
    stamp=datetime.now(timezone.utc).strftime('%H%M%S%f');loop_id='cpu_b8_guard_'+stamp
    scratch=ROOT/'work/research/recovery_v2_cpu';scratch.mkdir(exist_ok=True)
    synthetic=copy.deepcopy(manifest);synthetic['target_recovery_loop_id']=loop_id
    synthetic_manifest=scratch/(loop_id+'.json');write_json(synthetic_manifest,synthetic)
    args=controller.parser().parse_args(['--loop-id',loop_id,'--transition-manifest',str(synthetic_manifest)])
    real_validate=controller.validate_selection
    selection_map={};commands=[]
    dummy_weights=scratch/'dummy.pt';dummy_weights.write_bytes(b'CPU mock, no model')
    def validate(run_id):
        if run_id not in selection_map:return real_validate(run_id)
        row=selection_map[run_id]
        return row,{'selected_step':row['selected_step'],'weights_sha256':controller.sha(dummy_weights),'code_sha256':{},'input_sha256':{}}
    def io_record(command):
        worker=command[command.index('--worker')+1];target=Path(command[command.index('--provenance')+1])
        value={'status':'complete','training_semantics_changed':False,'arguments':command[command.index('--')+1:]}
        for key,name in [('worker_sha256',worker),('launcher_sha256','launch_with_io_retry_v1.py'),('helper_sha256','atomic_io_retry_v1.py')]:
            value[key]=value[key+'_after']=controller.sha(OUTPUT/name)
        write_json(target,value)
    class Child:
        def __init__(self,command,**keywords):
            self.pid=999999;self.returncode=0;commands.append(command)
            def arg(key):return command[command.index(key)+1]
            require((arg('--batch-size'),arg('--accumulation'),arg('--warmup-steps'))==('8','2','25'),'Actual child argv uses B8/effective16/matched warmup sample exposure')
            require(arg('--epochs')=='3.0' and arg('--eval-per-epoch')=='4.0','Actual child argv keeps finite epoch cap/cadence')
            require((arg('--head-lr'),arg('--lora-lr'),arg('--new-token-lr-multiplier'))==('5e-05','0.0003','6'),'Actual B8 LR stays prior DEV-winning B parameters')
            require(Path(arg('--resume-weights')).resolve()==(ROOT/'work/adaptive_20261010_01_screen_B_best.pt').resolve(),'Actual B8 starts best qualifying DEV weights')
            require('--resume-exact' not in command,'Effective-batch change uses a fresh optimizer explicitly')
            run_id=arg('--run-id');directory=OUTPUT/'runs'/run_id;directory.mkdir()
            row={'accepted':True,'run_id':run_id,'selected_step':50,'completed_steps':50,'weights':str(dummy_weights),
                 'turkish_validation_wer':.70,'turkish_epochs_completed':.0156,'selection_split':'FLEURS validation only','heldout_tests_evaluated':False}
            selection_map[run_id]=row;write_json(directory/'selection.json',row);io_record(command)
        def poll(self):return self.returncode
    def run(command,**keywords):
        commands.append(command);directory=OUTPUT/'runs'/loop_id
        frozen=read_json(directory/'frozen_final_selection.json',{})
        if '--worker' in command:
            winner=frozen['run_id'];selection_map[winner].update(merged=True,heldout_tests_evaluated=True,
                merged_development_gate={'passed':True,'turkish_wer':.70})
            write_json(OUTPUT/'runs'/winner/'selection.json',selection_map[winner]);io_record(command)
        elif '--selected-run-id' in command:
            write_json(OUTPUT/'adaptive_strong_baseline_test_comparison_v2.json',{'status':'complete','passed':True,
                       'run_id':frozen['run_id'],'frozen_selection_sha256':controller.sha(directory/'frozen_final_selection.json')})
        else:
            write_json(directory/'adaptive_results_verification.json',{'passed':True,'run_id':frozen['run_id'],
                       'loop_id':loop_id,'selected_step':frozen['selected_step'],
                       'frozen_selection_sha256':controller.sha(directory/'frozen_final_selection.json')})
        return SimpleNamespace(returncode=0)
    protected_paths=[OUTPUT/'adaptive_strong_baseline_test_comparison_v2.json',OUTPUT/'adaptive_experiment_recovery_v2_report.json']
    saved={p:p.read_bytes() if p.exists() else None for p in protected_paths}
    old_hashes={p:controller.sha(p) for p in [OUTPUT/'run_experiment_loop_recovery.py',OUTPUT/'launch_with_io_retry_v1.py',
              OUTPUT/'atomic_io_retry_v1.py',OUTPUT/'runs/adaptive_20261010_recovery_01/loop_status.json',
              OUTPUT/'runs/adaptive_20261010_recovery_01_screen_C/selection.json']}
    try:
        with patch.object(controller,'validate_selection',side_effect=validate), \
             patch.object(controller,'subprocess',SimpleNamespace(Popen=Child,run=run,STDOUT=-2,CREATE_NO_WINDOW=0,TimeoutExpired=TimeoutError)), \
             patch.object(controller,'keep_awake',return_value={'supported':False,'enabled':False}), \
             patch.object(controller,'backlog_note',return_value=None):controller.main(args)
        state=read_json(OUTPUT/'runs'/loop_id/'loop_status.json',{})
        require(state['status']=='complete','Mocked full B8 lifecycle reaches final independent audit')
        require(state['deadline_at_utc']==manifest['original_deadline_at_utc'],'Original absolute deadline remains unchanged')
        active=[row for row in state['trials'] if row['name']=='b8_quality']
        require(len(active)==1,'Only one continuous B8 optimizer trial launched')
        row=active[0]
        require(row['effective_batch_size']==16 and row['batch_numerical_equivalence_proven'] is False,'Effective batch change and strict numeric failure explicit')
        require(row['warmup_sample_exposure']=={'current':400,'prior_B':400},'Warmup matches prior sample exposure')
        require(row['epochs_completed']<.25 and 'selection_provenance' in row,'B8 partial final-stop DEV can qualify; no false matched-quarter requirement')
        require(state['best_development_candidate']['selected_configuration']['batch_size']==8,'DEV improvement selects actual B8 configuration')
        require(state['recovery']['backend_flags_documented']['trainer_fresh_defaults']['cudnn_allow_tf32'] is True
                and state['recovery']['backend_flags_documented']['B8_profile']['cudnn_allow_tf32'] is False,'Backend difference retained without timing-transfer claim')
        require(len([r for r in state['trials'] if r.get('carried_from_prior_recovery')])==1,'Stopped partial C carried read-only')
    finally:
        for path,data in saved.items():
            if data is None:
                if path.exists():path.unlink()
            else:path.write_bytes(data)
    for path,digest in old_hashes.items():require(controller.sha(path)==digest,'Original/prior artifact unchanged '+path.name)
    report={'passed':True,'check_count':len(checks),'checks':checks,'controller_sha256':controller.sha(OUTPUT/'run_experiment_loop_recovery_v2.py'),
            'scope':'CPU transition guards and mocked full B8 lifecycle only; all trainer/finalizer/TEST/audit children mocked; no GPU work',
            'synthetic_loop_id':loop_id,'guard_script_sha256':controller.sha(__file__)}
    write_json(OUTPUT/'recovery_controller_v2_cpu_verification.json',report)
    print(json.dumps({'passed':True,'check_count':len(checks),'controller_sha256':report['controller_sha256']}))


if __name__=='__main__':main()
