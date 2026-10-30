"""One focused stdlib first-controller integration using explicit fake source files.

Executes actual prepare/chain/outer-validate/identity/completion functions.
Does not create a GPU child, read real data/checkpoints, or import ML modules.
"""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(HERE))
import run_hf_bpe512_arm_v2 as controller
import hf_bpe512_training_v2 as helper
import launch_native_training_bpe512_v2 as outer

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main(args):
    output=(args.output or HERE/('HF_BPE512_first_domain_controller_CPU_verification_v2_'+uuid.uuid4().hex+'.json')).resolve()
    if not output.is_relative_to(ROOT) or output.exists():raise ValueError('Fresh output required')
    checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def reject(name,call):
        try:call()
        except ValueError:checks.append(name);return
        raise AssertionError(name)
    actual_chain=controller.chain_identity()
    check('actual_frozen21_source_route',len(actual_chain['source_sha256'])==21)
    prior=json.loads((HERE/'HF_BPE512_domain_admission_CPU_verification_v2.json').read_text(encoding='utf-8'))
    prior_scratch=Path(prior['synthetic_fixture_path'])
    scratch=ROOT/'work/research'/('bpe_first_controller_fixture_'+uuid.uuid4().hex);scratch.mkdir()
    def rewrite(value):
        if isinstance(value,str):return value.replace(str(prior_scratch),str(scratch))
        if isinstance(value,list):return [rewrite(item) for item in value]
        if isinstance(value,dict):return {key:rewrite(item) for key,item in value.items()}
        return value
    def write(name,value):
        path=scratch/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8');return path
    def rec(path):return {'path':str(path.resolve()),'sha256':sha(path)}
    # Only small synthetic JSON + source bytes are copied; no actual .pt/model.
    for name in ('source_checkpoint_FAKE.json','converted_weights_FAKE.json','source_selection.json','source_config.json','source_history.json',
                 'results/synthetic_closed_domain_step_1/summary.json','results/base_v3/summary.json'):
        write(name,rewrite(json.loads((prior_scratch/name).read_text(encoding='utf-8'))))
    for name in actual_chain['source_sha256']:
        (scratch/name).write_bytes((HERE/name).read_bytes())
    for name in (*helper.DOMAIN_SOURCES,'run_experiment_loop_continuation_v8.py','launch_nemo_linker_v4.py'):
        if not (scratch/name).exists():(scratch/name).write_bytes((HERE/name).read_bytes())
    source_config=json.loads((scratch/'source_config.json').read_text(encoding='utf-8'))
    source_config['config_identity']['native_loss']={'sigma':0.,'omega':0.,'caller_reduction':'raw per-clip NLL / each target length, then physical-batch mean','max_joint_bytes':1700000000}
    source_config['input_sha256']={}
    for key in ('tr_manifest','replay_manifest','data_v3_proof'):
        p=write('fake_'+key+'.json',{'synthetic':True});source_config[key]=str(p)
        source_config['input_sha256'][str(p.resolve())]=sha(p)
    write('source_config.json',source_config)
    closure=rewrite(json.loads((prior_scratch/'closure.json').read_text(encoding='utf-8')))
    for key in ('source_checkpoint','source_selection','source_run_config','source_training_history','source_development_summary'):
        closure[key]=rec(Path(closure[key]['path']))
    closure_path=write('closure.json',closure)
    loop_id='synthetic_BPE_first_'+uuid.uuid4().hex;run_id=loop_id+'_bpe512'
    conversion=rewrite(json.loads((prior_scratch/'conversion.json').read_text(encoding='utf-8')))
    conversion.update(target_run_id=run_id,output_checkpoint_sha256=sha(scratch/'converted_weights_FAKE.json'))
    conversion['source']['selection_sha256']=sha(scratch/'source_selection.json')
    conversion_path=write('conversion.json',conversion)
    originals=(controller.HERE,helper.HERE,outer.HERE)
    controller.HERE=helper.HERE=outer.HERE=scratch
    a=SimpleNamespace(loop_id=loop_id,source_checkpoint=scratch/'source_checkpoint_FAKE.json',source_selection=scratch/'source_selection.json',
        source_closure=closure_path,source_run_config=scratch/'source_config.json',conversion_report=conversion_path,
        converted_weights=scratch/'converted_weights_FAKE.json',candidate_dir=ROOT/'work/hf_turkish_bpe_candidate_512_v1',
        window_hours=3.,poll_seconds=5.,validate_only=True,preflight_output=None,gpu_window_authorized=False)
    try:
        plan=controller.prepare(a)
        check('actual_prepare_keeps_all35_domain_args',plan['copied_source_recipe']==source_config['config_identity']['training_arguments'])
        check('same_domain_RP_policy',plan['source_sampling_policy']['sampling_policy_changed'] is False
            and plan['source_sampling_policy']['target_replay_RNG_independent'] is True)
        outer_args=SimpleNamespace(native_provenance=Path(plan['paths']['native']),memory_launcher=scratch/actual_chain['memory_launcher'],
            memory_launcher_sha256=actual_chain['source_sha256'][actual_chain['memory_launcher']],
            trainer_sha256=actual_chain['worker_sha256'],memory_arguments=plan['command'][plan['command'].index('--')+1:])
        _,_,_,worker,actual_run,_,_=outer.validate(outer_args)
        check('actual_outer_validate_generated_command',actual_run==run_id and worker==plan['worker_arguments'])
        check('only_weights_initial_not_exact', '--resume-weights' in worker and '--resume-exact' not in worker)
        check('candidate_native_bound_declared',worker[worker.index('--native-max-joint-bytes')+1]=='1800000000')
        names={'memory_launcher_sha256':'launch_with_memory_guard_bpe512_v2.py','cache_launcher_sha256':'launch_with_cache_reclaim_bpe512_v2.py',
            'cache_helper_sha256':'cuda_cache_reclaim_v2.py','resource_helper_sha256':'resource_watchdog_v2.py',
            'resource_telemetry_sha256':'paired_windows_commit_telemetry_v1.py','frozen_worker_sha256':'train_lora_batched_bpe512_v2.py',
            'data_policy_sha256':'public_turkish_data_policy_v3.py'}
        now=datetime.now(timezone.utc);started=now.isoformat()
        identity={'kind':'owned_training_worker_identity_v1','expected_run_id':run_id,'worker_arguments_sha256':controller.guards.argument_sha(worker),
            'pid':111,'ppid':100,'started_at_utc':started,'process_creation_identity':{'available':True,'pid':111,'creation_unix_seconds':now.timestamp()}}
        identity.update({key:plan['source_sha256'][name] for key,name in names.items()})
        check('realistic_actual_identity_validator',controller.validate_identity(identity,plan,100,started)['pid']==111)
        wrong=copy.deepcopy(identity);wrong['frozen_worker_sha256']='0'*64
        reject('reject_worker_identity_source_change',lambda:controller.validate_identity(wrong,plan,100,started))
        paths={key:Path(value) for key,value in plan['paths'].items()}
        for key in ('io','memory'):write(paths[key].relative_to(scratch),{'status':'complete'})
        write(paths['cache'].relative_to(scratch),{'status':'complete','cache_reclamation':{'eligible_update_calls':3}})
        write(paths['native'].with_suffix('.completion.json').relative_to(scratch),{'status':'complete','sources_unchanged':True})
        equiv_path=write('runs/'+run_id+'/BPE512_actual_GPU_source_equivalence.json',{'kind':'HF_BPE512_actual_unsuppressed_GPU_source_equivalence_v1',
            'passed':True,'actual_GPU_decode':True,'before_optimizer_creation':True,'decoded_clips':300,'raw_transcript_changes':[],'token_duration_emission_changes':[]})
        selected={'run_id':run_id,'heldout_tests_evaluated':False,'completed_steps':3}
        check('actual_completed_state_check_with_synthetic_evidence',controller.verify_completion(plan,selected)==rec(equiv_path))
        write(paths['cache'].relative_to(scratch),{'status':'complete','cache_reclamation':{'eligible_update_calls':2}})
        reject('reject_missing_completed_update_cache_hook',lambda:controller.verify_completion(plan,selected))
        stale=copy.copy(a);stale.converted_weights=scratch/'source_checkpoint_FAKE.json'
        reject('reject_stale_or_other_conversion_weights',lambda:controller.prepare(stale))
    finally:controller.HERE,helper.HERE,outer.HERE=originals
    check('no_ML_no_actual_process',all(name not in sys.modules for name in ('torch','nemo','transformers','tokenizers')))
    report={'kind':'HF_BPE512_first_domain_controller_CPU_verification_v2','passed':True,'checks':len(checks),'assertions':checks,
        'controller_sha256':sha(HERE/'run_hf_bpe512_arm_v2.py'),'routing_receipt':rec(controller.CHAIN_PATH),
        'synthetic_fixture_path':str(scratch),'GPU_child_created':False,'actual_checkpoint_data_or_audio_read':False,
        'scope':'Actual source functions/21pins/command validation/identity/completion, synthetic source files and runtime records only; actual GPU preservation/fit pending'}
    with output.open('x',encoding='utf-8') as out:out.write(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'passed':True,'checks':len(checks),'report':str(output),'controller_sha256':report['controller_sha256']}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);main(p.parse_args())
