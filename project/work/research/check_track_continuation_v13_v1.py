"""Small actual-function/producer-shape fixture; no mirror/DB/history imports."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
import argparse
import ast
import copy
import hashlib
import json
import math
import os
import re
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
source=HERE/'track_continuation_v13.py';old=HERE/'track_continuation_v12.py'
tree=ast.parse(source.read_text(encoding='utf-8'));previous=ast.parse(old.read_text(encoding='utf-8'))
def read_json(path,default=None):return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else default
def write_json(path,value):Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')
ns={'Path':Path,'ROOT':ROOT,'math':math,'json':json,'hashlib':hashlib,'os':os,'re':re,'time':time,
    'datetime':datetime,'timezone':timezone,'read_json':read_json,'write_json':write_json,'__file__':str(source),
    'original':SimpleNamespace(metrics_for=lambda record:{'train/loss':record.get('loss',1.)})}
for node in tree.body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        try:ns[node.targets[0].id]=ast.literal_eval(node.value)
        except (ValueError,TypeError):pass
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
nodes += [n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('RunShardStore','StreamRunShardStore')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
checks=[];rejects=[]
def ok(value,label):
    if not value:raise AssertionError(label)
    checks.append(label)
def bad(call,label):
    try:call()
    except (ValueError,KeyError):checks.append(label);rejects.append(label);return
    raise AssertionError('Unexpected acceptance '+label)
def record(path):return {k:ns['file_record'](path)[k] for k in ('path','sha256','bytes')}

def main(args):
    assert not args.output.exists(),'Fresh proof output required'
    legacy={n.name:n for n in previous.body if isinstance(n,ast.FunctionDef)}
    changed={'follow','parser','main','publish_cycle_memory','start_controls'}
    preserved=[name for name in legacy if name not in changed]
    ok(all(ast.dump(legacy[name],include_attributes=False)==ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False) for name in preserved),
       'All inherited V12 metric/context/math functions AST unchanged')
    with tempfile.TemporaryDirectory(prefix='v13_tiny_',dir=ROOT/'work/research') as folder:
        folder=Path(folder);gates={lang:{'word_errors':67 if lang=='pt' else 60,'allowed_word_errors':66 if lang=='pt' else 70,'passed':lang!='pt'} for lang in ('en','es','it','fr','de','pt')}
        recipe={'replay_fraction':.125,'replay_kl':40.,'duration_kl':2.,'feature_distillation':4.,'head_lr':1e-5,'lora_lr':3e-4,'new_token_lr_multiplier':6}
        prior=8968;budget=17488;end=25109
        full={'source_step':6500,'transition_source_step':7621,'prior_uniform_draws':prior,'additional_update_budget':budget,'expected_new_unique_TR_clips':253798,'end_step':end}
        parent={'run_id':'fixture_parent','phase':ns['UNIFORM_MAJORITY_PHASE'],'source_step':7621,'inherited_history_steps':7621,'uniform_origin_step':6500,
            'uniform_full_pass':full,'remaining_uniform_pass_updates':budget,'steps':end,'trainable_parameter_tensors':207,'replay_clips_used':1914,
            'evaluation_languages':['tr','en','es','it','fr','de','pt'],'Portuguese_recovery':{'grace_new_updates':1000},
            'config_identity':{'training_arguments':recipe,'domain_sampling_identity':{'quota':'7621'},'tr_pool_sha256':'a','replay_pool_sha256':'b','extended_tokenizer_sha256':'c',
                'majority_transition':{'declared':{'source_step':7621,'prior_uniform_draws':prior,'TR_per_update':14,'replay_per_update':2,'configured_end_step':end,'renewed_Portuguese_grace':False,
                    'preservation_coefficients':{'replay_kl':40.,'duration_kl':2.,'feature_distillation':4.}}}},
            'initial_source_equivalence':{'passed':True,'actual_GPU_decode':True,'source_selected_step':7621,'before_first_majority_update':True,'compared_fixed_DEV_utterances':350,
                'all_seven_ASR_counts_exact':True,'source_retention_gates':gates,'all_six_retention_gates_passed':False}}
        parent_path=folder/'parent_config.json';write_json(parent_path,parent)
        receipt_path=folder/'source_receipt.json';write_json(receipt_path,{'chosen_step':7750,'completed_steps':7750,'artifacts':{'run_config':record(parent_path)}})
        declared={'kind':'declared_lower_encoder_LR_full_state_v1','source_step':7750,'quota_boundary_step':7621,'uniform_origin_step':6500,'configured_end_step':end,
            'source_LoRA_lr':3e-4,'active_LoRA_lr':1e-4,'head_lr':1e-5,'preservation_coefficients':[40,2,4],'TR_per_update':14,'replay_per_update':2,
            'all207_Adam_moments_and_global_clocks_preserved':True,'fresh_optimizer_or_sampler':False,'renewed_Portuguese_grace':False,
            'parent_failure_counter_preserved':True,'all_six_required_for_every_BEST':True,'only_new_scientific_variable':'encoder_LoRA_group_LR'}
        config={**parent,'run_id':'fixture_lower','phase':ns['LOWER_LORA_PHASE'],'source_step':7750,'inherited_history_steps':7750,'majority_source_step':7621,
            'remaining_uniform_pass_updates':end-7750,'encoder_learning_rate_transition':declared,
            'config_identity':{**parent['config_identity'],'training_arguments':{**recipe,'lora_lr':1e-4},'encoder_lr_transition':{
                'kind':'TSC_uniform_majority_LoRA_LR_state_transition_v1','declared':declared,'source_receipt':record(receipt_path),
                'parent_config_identity_sha256':ns['_mirror_config_sha'](parent['config_identity'])}},
            'initial_source_equivalence':{**parent['initial_source_equivalence'],'source_selected_step':7750,'all_raw_transcripts_and_emitted_paths_exact':True,'source_receipt':record(receipt_path)}}
        origin={'step':6500,'turkish_unique_clips_seen':38437}
        boundary={'step':7621,'audio_seen_seconds':20000.,'turkish_audio_seen_seconds':8000.,'replay_audio_seen_seconds':12000.,'train_wall_seconds':2000.,
            'turkish_clips_seen':7621*8,'replay_clips_seen':7621*8,'turkish_unique_clips_seen':43000,
            'Turkish_domain_coverage':{'uniform_pass':{'source_step':6500,'draws':prior,'unique_clips_seen':prior}}}
        def row(step):
            m=step-7621;draws=prior+14*m;pool=253798
            return {'step':step,'majority_updates':m,'majority_source_step':7621,'uniform_origin_step':6500,'uniform_full_updates':step-6500,
                'head_lr':1e-5,'lora_lr':3e-4 if step<=7750 else 1e-4,'new_token_update_multiplier':6,'replay_kl_coefficient':40.,'duration_kl_coefficient':2.,'feature_distillation_coefficient':4.,
                'turkish_clips_seen':7621*8+14*m,'replay_clips_seen':7621*8+2*m,'turkish_unique_clips_seen':46000,
                'audio_seen_seconds':20000+100*m,'turkish_audio_seen_seconds':8000+70*m,'replay_audio_seen_seconds':12000+30*m,'train_wall_seconds':2000+10*m,
                'lora_lr_fork_updates':step-7750,'lora_lr_fork_source_step':7750,
                'Turkish_domain_coverage':{'uniform_pass':{'source_step':6500,'local_updates':step-6500,'pool_clips':pool,'draws':draws,'unique_clips_seen':draws,'repeat_visits':0,
                    'fraction_unique_seen':draws/pool,'epoch_clock':draws/pool,'full_pass_complete':False,'quota_segments':[
                        {'first_step':6501,'last_step':7621,'updates':1121,'TR_per_update':8,'replay_per_update':8,'TR_draws':prior,'replay_draws':prior},
                        {'first_step':7622,'last_step':step,'updates':m,'TR_per_update':14,'replay_per_update':2,'TR_draws':14*m,'replay_draws':2*m}]},
                    'majority_transition':{'local_updates':m,'source_step':7621}},
                'validation':{'retention_gate_passed':False,'old_five_retention_gate_passed':True,'Portuguese_grace_active_for_stop_only':False,'stop_retention_gate_passed':False}}
        source_row=row(7750);contexts=ns['loralr_phase_contexts_for'](config,[origin,boundary,source_row]);values=ns['loralr_phase_value_for_record'](row(8000),contexts,{})
        ok(values['phase/lora_lr_fork_updates']==250 and values['phase/majority_updates']==379 and values['phase/uniform_full_updates']==1500,'Fork7750/quota7621/uniform6500 clocks distinct')
        ok(values['resources/post_lower_lora_audio_minutes_per_wall_minute']==10 and values['resources/post_lower_lora_turkish_audio_minutes_per_wall_minute']==7,
           'New-phase rates subtract actual7750 counters')
        ok(values['data/post_majority_audio_seen_seconds']==37900 and values['data/post_lower_lora_audio_seen_seconds']==25000,'Wholemajority and newLR raw/audio rates kept separate')
        ok(values['phase/lora_lr']==1e-4 and values['phase/expected_full207_Adam_clock']==8000 and values['retention/Portuguese_stop_grace_active']==0,'Actual LR/expected clock/no renewed PT grace')
        dev={7750:{'dev/wer_'+lang+'_percent':40. for lang in ('tr','en','es','it','fr','de','pt')}}
        initial=ns['loralr_phase_value_for_record'](source_row,contexts,dev)
        ok(initial['phase/lora_lr_fork_updates']==0 and initial['retention/all_six_passed']==0 and initial['retention/original_five_passed']==1,'Actual initial7/sourcePTfailure not fabricated champion')
        old_values=ns['loralr_phase_value_for_record'](row(7740),contexts,{7740:{'dev/wer_tr_percent':99.}})
        ok(not any(key.startswith(('dev/','retention/')) for key in old_values) and old_values['phase/inherited_history']==1,'Inherited history cannot fabricate newrun DEV')
        ok(ns['loralr_phase_baseline_steps'](config)=={6500,7621,7750} and ns['loralr_phase_initial_development_step'](config)==7750,'Only three small baseline rows retained')
        selection={'run_id':config['run_id'],'source_step':7750,'majority_source_step':7621,'completed_steps':8000,'lora_lr_fork_updates':250,'majority_updates':379,
            'uniform_origin_step':6500,'uniform_full_updates':1500,'accepted':False,'selected_step':None,'turkish_validation_wer':None,'full_new_uniform_pass_complete':False,'stop_early_reason':'soft_max_wall_seconds'}
        ok(ns['loralr_selection_values'](config,selection,8000)['selection/published_lora_lr_fork_updates']==250 and not ns['loralr_rejected_run_terminal'](config,selection),'Budget with no all6 BEST stays resumable')
        bad(lambda:ns['loralr_phase_value_for_record']({**row(8000),'lora_lr':3e-4},contexts,{}),'Changed activeLR rejected')
        bad(lambda:ns['loralr_phase_value_for_record']({**row(8000),'lora_lr_fork_updates':251},contexts,{}),'Wrong forkclock rejected')
        bad(lambda:ns['lower_lora_context']({**config,'majority_source_step':7750},[origin,boundary,source_row]),'Quota cannot reset to new LR boundary')
        ns['start_controls']=lambda args:{'path':None};ns['follow']=lambda args,control:'cooperative_stop_file';finished=[];ns['finish_controls']=lambda *args:finished.append(args)
        current=SimpleNamespace(value=None,get=lambda:current.value)
        fake=SimpleNamespace(context_vars=SimpleNamespace(current_run=current),calls=0)
        def finish():fake.calls+=1;current.value=None
        fake.finish=finish;ns['trackio']=fake
        args_main=SimpleNamespace(state_path=folder/'v13.json',source_state_path=folder/'v12.json')
        ok(ns['main'](args_main)==0 and fake.calls==0 and finished[-1][-1] is True,'Already-finished finalizer succeeds without duplicatefinish')
        current.value=object();ok(ns['main'](args_main)==0 and fake.calls==1,'Active run finalizer calls actual API once')
        current.value=object();fake.finish=lambda:(_ for _ in ()).throw(ValueError('Other finish failure'))
        bad(lambda:ns['main'](args_main),'Other finalizer failure propagates')
        ok(finished[-1][1]=='failed' and finished[-1][-1] is False,'Other finish failure accurately reported')
    report={'schema':1,'kind':'inactive_Trackio_v13_lowerLR_stdlib_verification_v1','passed':True,'checks':checks,'assertions':len(checks),'rejections':len(rejects),
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'preserved_V12_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'legacy_functions_AST_unchanged':preserved,
        'scope':'Actual extracted metric/finalizer functions with synthetic hash-bound parent config/receipt, producer field shapes and three baseline records. No actual mirror/index/history/DB/ML/GPU read or execution.',
        'actual_V12_exit0_claimed':False,'actual_mirror_handoff_performed':False}
    with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({'passed':True,'checks':len(checks),'rejects':len(rejects),'source_sha256':report['source_sha256'],'proof_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),'proof':str(args.output)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'Trackio_v13_lowerLR_stdlib_verification_v1.json');main(parser.parse_args())
