"""One mocked candidate child/guarded-state lifecycle; stdlib only, no GPU."""
from __future__ import annotations
import copy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUT))
import run_hf_bpe512_arm_v1 as c


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def main(report):
    require=lambda ok,msg:ok or (_ for _ in ()).throw(AssertionError(msg))
    require(not any(name in sys.modules for name in ('torch','nemo','transformers','tokenizers')),'No ML imports')
    source=json.loads((OUT/'runs/adaptive_20261011_hf_quality_window_01_stronger_kd/run_config.json').read_text())
    recipe=source['config_identity']['training_arguments']
    argv=c.recipe_arguments(recipe)
    require(argv[argv.index('--replay-kl')+1]=='10.0' and argv[argv.index('--head-lr')+1]=='5e-05'
        and argv[argv.index('--lora-lr')+1]=='0.0003','Actual source recipe fields copied')
    for bad in ({**recipe,'replay_kl':float('nan')},{**recipe,'batch_size':4},{**recipe,'train_limit':1}):
        try:c.recipe_arguments(bad)
        except ValueError:pass
        else:raise AssertionError('Altered unsafe recipe was accepted')
    chain=c.chain_identity()
    with tempfile.TemporaryDirectory(prefix='bpe_controller_mock_',dir=ROOT/'work') as temporary:
        base=Path(temporary);fake=base/'outputs/parakeet_turkish';fake.mkdir(parents=True)
        loop_id='fixture_bpe_loop';run_id=loop_id+'_bpe512';loop=fake/'runs'/loop_id
        mocked_source=copy.deepcopy(source);mocked_source['run_id']='closed_fixture_source'
        mocked_source['reuse_input_audit']=None
        mocked_source['input_sha256']={}
        for key in ('tr_manifest','replay_manifest','data_v3_proof'):
            path=base/(key+'.json');save(path,{'mocked_metadata':True})
            mocked_source[key]=str(path);mocked_source['input_sha256'][str(path)]=c.sha(path)
        for name in mocked_source['code_sha256']:shutil.copyfile(OUT/name,fake/name)
        config_path=base/'source_config.json';save(config_path,mocked_source)
        parent_checkpoint=base/'mock_parent_weights.pt';parent_checkpoint.write_bytes(b'mocked checkpoint; never Torch loaded')
        converted_checkpoint=base/'mock_converted_weights.pt';converted_checkpoint.write_bytes(b'mocked enlarged weights; never Torch loaded')
        selection_path=base/'source_selection.json'
        save(selection_path,{'accepted':True,'selected_step':5000,'run_id':'closed_fixture_source','turkish_validation_wer':.56242424})
        closure_path=base/'source_closure.json'
        closure={'kind':'root_owned_BPE512_closed_source_v1','passed':True,'root_observed_owned_session_completed':True,
            'child_exit_code':0,'source_run_id':'closed_fixture_source','source_checkpoint':c.record(parent_checkpoint),
            'source_selection':c.record(selection_path),'source_run_config':c.record(config_path)}
        save(closure_path,closure)
        conversion_path=base/'conversion.json'
        save(conversion_path,{'kind':'HF_Turkish_BPE_512_weights_only_warm_start_v1','status':'complete','passed':True,
            'new_optimizer_required':True,'exact_training_resume':False,'target_run_id':run_id,
            'output_checkpoint':str(converted_checkpoint),'output_checkpoint_sha256':c.sha(converted_checkpoint),
            'source':{**c.record(parent_checkpoint),'run_id':'closed_fixture_source','selected_step':5000,
                'selection_sha256':c.sha(selection_path)}})
        arguments=SimpleNamespace(loop_id=loop_id,window_hours=1.,poll_seconds=5.,source_checkpoint=parent_checkpoint,
            source_selection=selection_path,source_closure=closure_path,source_run_config=config_path,
            conversion_report=conversion_path,converted_weights=converted_checkpoint,
            candidate_dir=ROOT/'work/hf_turkish_bpe_candidate_512_v1')
        with patch.object(c,'HERE',fake),patch.object(c,'chain_identity',lambda:chain):
            plan=c.prepare(arguments)
            require(plan['copied_source_recipe']==recipe,'Every original source recipe field is identical')
            require('launch_native_training_bpe512_v1.py' in plan['command'][1]
                and '--bpe-source-closure' in plan['worker_arguments'] and '--resume-exact' not in plan['worker_arguments'],
                'Actual new outer command and fresh warmstart arguments')
            closure['root_observed_owned_session_completed']=False;save(closure_path,closure)
            try:c.prepare(arguments)
            except ValueError:pass
            else:raise AssertionError('Live/unconfirmed source accepted')
            closure['root_observed_owned_session_completed']=True;save(closure_path,closure)
        loop.mkdir(parents=True)
        run=fake/'runs'/run_id;run.mkdir(parents=True)
        paths={k:Path(v) for k,v in plan['paths'].items()}
        selection={'run_id':run_id,'heldout_tests_evaluated':False,'accepted':True,'selected_step':1,
            'completed_steps':1,'stop_early_reason':'configured_training_complete','turkish_validation_wer':.56}
        save(run/'selection.json',selection)
        save(run/'training_history.json',[{'step':1,'current_reserved_vram_gb':4.,'audio_seen_seconds':100.,
            'train_wall_seconds':2.,'turkish_audio_seen_seconds':50.,'validation':{'turkish_wer':.56,'retention_gate_passed':True}}])
        save(paths['io'],{'status':'complete'});save(paths['memory'],{'status':'complete'})
        save(paths['cache'],{'status':'complete','cache_reclamation':{'eligible_update_calls':1}})
        save(paths['native'].with_suffix('.completion.json'),{'status':'complete','sources_unchanged':True})
        proof={'kind':'HF_BPE512_actual_unsuppressed_GPU_source_equivalence_v1','passed':True,'actual_GPU_decode':True,
            'before_optimizer_creation':True,'decoded_clips':300,'raw_transcript_changes':[],'token_duration_emission_changes':[]}
        save(run/'BPE512_actual_GPU_source_equivalence.json',proof)
        stamp=datetime.now(timezone.utc).isoformat();pid=123456
        names={'memory_launcher_sha256':'launch_with_memory_guard_bpe512_v1.py',
            'cache_launcher_sha256':'launch_with_cache_reclaim_bpe512_v1.py','cache_helper_sha256':'cuda_cache_reclaim_v2.py',
            'resource_helper_sha256':'resource_watchdog_v2.py','resource_telemetry_sha256':'paired_windows_commit_telemetry_v1.py',
            'frozen_worker_sha256':'train_lora_batched_bpe512_v1.py','data_policy_sha256':'public_turkish_data_policy_v3.py'}
        identity={'kind':'owned_training_worker_identity_v1','pid':pid,'ppid':1,'expected_run_id':run_id,
            'worker_arguments_sha256':c.guards.argument_sha(plan['worker_arguments']),'started_at_utc':stamp,
            'process_creation_identity':{'available':True,'pid':pid,'creation_unix_seconds':datetime.now(timezone.utc).timestamp()},
            **{k:chain['source_sha256'][v] for k,v in names.items()}}
        save(paths['identity'],identity)
        sample={'observed_at_utc':stamp,'observed_monotonic':c.time.monotonic(),
            'system_memory':{'available':True,'observed_at_utc':stamp,'commit_fraction':.4,'commit_used_bytes':20e9,
                'commit_available_bytes':30e9,'commit_limit_bytes':50e9},'worker_private_memory':{'pid':pid,'available':True},
            'calling_process_commit_memory':{'pid':pid,'available':True,'ullAvailPageFile_bytes':30e9}}
        save(Path(str(paths['identity'])+'.live_memory.json'),{'kind':'owned_worker_live_memory_v1','pid':pid,'run_id':run_id,
            'worker_identity_sha256':c.sha(paths['identity']),'process_creation_identity':identity['process_creation_identity'],
            'sample':sample,**{k:identity[k] for k in names}})
        class MockChild:
            def __init__(self,*a,**kw):
                require(a[0]==plan['command'],'Actual prepared isolated command dispatched')
                self.pid=pid;self.returncode=0;self.polls=0
            def poll(self):self.polls+=1;return None if self.polls==1 else 0
            def wait(self,**kw):return 0
        with patch.object(c,'HERE',fake),patch.object(c,'chain_identity',lambda:chain),patch.object(c.subprocess,'Popen',MockChild),\
             patch.object(c,'sample_memory',lambda *a:copy.deepcopy(sample)),patch.object(c,'publish_note',lambda *a:None),\
             patch.object(c.time,'sleep',lambda seconds:None):
            status=c.run_arm(plan,c.time.monotonic()+10000,SimpleNamespace(poll_seconds=5.))
            require(status==0 and plan['status']=='quality_window_ended' and plan['completed_steps']==1,'Mock child ended with durable state')
            require(plan['improved_qualifying_DEV_over_closed_source'] is True and plan['objective_complete'] is False,'Quality gain distinct from goal completion')
            require((loop/'memory_observations.jsonl').exists() and not paths['stop'].exists(),'Actual guard observation / no invented stop')
            proof['actual_GPU_decode']=False;save(run/'BPE512_actual_GPU_source_equivalence.json',proof)
            try:c.verify_completion(plan,selection)
            except ValueError:pass
            else:raise AssertionError('CPU-only source proof falsely accepted as GPU')
            proof['actual_GPU_decode']=True;proof['raw_transcript_changes']=[{'source_id':'fixture'}]
            save(run/'BPE512_actual_GPU_source_equivalence.json',proof)
            try:c.verify_completion(plan,selection)
            except ValueError:pass
            else:raise AssertionError('Changed transcript preflight falsely accepted')
            altered={**identity,'worker_arguments_sha256':'0'*64}
            try:c.validate_identity(altered,plan,pid,stamp)
            except ValueError:pass
            else:raise AssertionError('Unowned argument identity falsely accepted')
    require(not any(name in sys.modules for name in ('torch','nemo','transformers','tokenizers')),'Still no ML imports')
    result={'kind':'isolated_HF_BPE512_controller_mocked_CPU_verification_v1','passed':True,
        'controller_sha256':c.sha(OUT/'run_hf_bpe512_arm_v1.py'),'routing_receipt_sha256':c.sha(c.CHAIN_PATH),
        'checks':['Actual source KD10/LR/B8x2 recipe scalar copying','Reject NaN/batch change/subset',
            'Real pinned chain identity','Actual prepare command + exact recipe; source weights/receipt are mocked',
            'Reject live/unconfirmed source','Mocked owned child + actual resource/progress/self-sidecar guards',
            'Durable DEV-only complete state','Goal remains incomplete','Reject CPU-only/changed-transcript GPU proof',
            'Reject wrong owned argv identity'],'GPU_launched':False,'actual_GPU_decode':False,
        'Torch_NeMo_models_audio_imported':False,'full_TR_pool_rescanned':False}
    with report.open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'report':str(report),'controller_sha256':result['controller_sha256']}))


if __name__=='__main__':
    main(Path(sys.argv[1]) if len(sys.argv)>1 else OUT/'HF_BPE512_controller_mocked_CPU_verification_v1.json')
