"""Focused stdlib BPE v2 source admission, exact branch, and sampler checks.

All checkpoint contents and DEV summaries below are explicitly synthetic.
No model, Torch/NeMo, audio, saved checkpoint, or actual corpus is loaded.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import uuid

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(HERE))
import hf_bpe512_training_v2 as helper
from turkish_domain_sampler_v2 import DomainBalancedBucketSampler
from turkish_domain_sampler_v1 import FLEURS_REVISION,CV_REVISION

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def functions(path):return {n.name:n for n in ast.parse(Path(path).read_text(encoding='utf-8')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def dumped(node):return ast.dump(node,include_attributes=False)
def main(args):
    target=(args.output or HERE/('HF_BPE512_domain_admission_CPU_verification_v2_'+uuid.uuid4().hex+'.json')).resolve()
    if not target.is_relative_to(ROOT) or target.exists():raise ValueError('NEW workspace output required')
    checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def reject(name,call):
        try:call()
        except (ValueError,KeyError,TypeError):checks.append(name);return
        raise AssertionError('Expected rejection: '+name)
    old=functions(HERE/'hf_bpe512_training_v1.py');new=functions(HERE/'hf_bpe512_training_v2.py')
    for name in ('suppression_ids','candidate_assets','candidate_tr_rows','confirm_candidate_acceptance',
                 'install_candidate_student','native_joint_bound','validate_converted_weights','compare_development_predictions'):
        check('unchanged_'+name,dumped(old[name])==dumped(new[name]))
    worker=functions(HERE/'train_lora_batched_bpe512_v2.py')
    prior=functions(HERE/'train_lora_batched_bpe512_v1.py')
    domain=functions(HERE/'train_lora_batched_domain30_v1.py')
    for name in ('BucketCycler','save_training_state','lr_factor','next_evaluation_epoch','development_gate',
                 'native_loss_identity','native_loss_function'):
        check('unchanged_worker_'+name,dumped(worker[name])==dumped(prior[name]))
    check('exact_original_domain_cycler',dumped(worker['BucketCycler'])==dumped(domain['BucketCycler']))
    ns={'argparse':argparse,'Path':Path,'DATA':ROOT/'work/data','OUTPUT':HERE,'ROOT':ROOT,'__doc__':'fixture'}
    exec(compile(ast.Module(body=[worker['parser']],type_ignores=[]),'actual_v2_parser','exec'),ns)
    parsed=ns['parser']().parse_args(['--run-id','fixture','--sampling-mode','domain30','--replay-seed','902',
        '--bpe-source-checkpoint','x','--bpe-source-selection','y','--bpe-source-closure','z','--bpe-conversion-report','c'])
    check('actual_parser_domain_args',parsed.sampling_mode=='domain30' and parsed.replay_seed==902 and parsed.fleurs_fraction==.3)
    run_source=ast.get_source_segment((HERE/'train_lora_batched_bpe512_v2.py').read_text(encoding='utf-8'),worker['run'])
    check('actual_worker_exact_guard', 'if args.resume_exact is None:' in run_source and 'established_source_equivalence' in run_source)
    check('exact_inputs_keep_original_conversion', 'Path(source_identity["converted_weights_path"])' in run_source)
    check('fullstate_restore_remains',all(s in run_source for s in ('optimizer.load_state_dict(resume["optimizer"])',
        'sampler.load_state_dict(resume["sampler"])','random.setstate(resume["python_rng"])',
        'torch.set_rng_state(resume["torch_rng"])','torch.cuda.set_rng_state_all(resume["cuda_rng"])')))
    # One small synthetic training pool tests replay invariance despite shorter BPE targets.
    pool=[]
    for domain_name,size in (('fleurs',13),('cv',31)):
        for i in range(size):
            pool.append({'source_id':domain_name+str(i),'language':'tr','dataset':'google/fleurs' if domain_name=='fleurs' else 'ysdede/commonvoice_17_tr_fixed',
                'dataset_revision':FLEURS_REVISION if domain_name=='fleurs' else CV_REVISION,'dataset_config':'tr_tr' if domain_name=='fleurs' else None,
                'dataset_split':'train','encoder_frames':10+i,'target_tokens':8+i,'duration':1+i/10,'text':str(i),'audio_sha256':'fixture'})
    replay=[{'source_id':lang+str(i),'language':lang,'encoder_frames':10+i,'target_tokens':3+i,'duration':1.0}
            for lang in ('en','es','it','fr','de') for i in range(5)]
    cycns={'math':math,'random':random,'ids_sha':lambda rows:hashlib.sha256('\n'.join(sorted(r['source_id'] for r in rows)).encode()).hexdigest()}
    exec(compile(ast.Module(body=[worker['BucketCycler']],type_ignores=[]),'actual_BucketCycler','exec'),cycns)
    def sampler(rows):return DomainBalancedBucketSampler(rows,replay,seed=17,replay_seed=902,sampling_mode='domain30',
        batch_size=8,accumulation=2,replay_fraction=.5,fleurs_fraction=.3,bucket_size=8,
        cycler_factory=cycns['BucketCycler'],cycler_source_sha256=sha(HERE/'train_lora_batched_data_v11.py'))
    shorter=[{**row,'target_tokens':max(1,row['target_tokens']//2)} for row in pool]
    left,right=sampler(pool),sampler(shorter)
    for _ in range(12):
        a,b=left.next_batches(),right.next_batches()
        check('independent_original_replay_ids_'+str(left.updates),[r['source_id'] for g in a for r in g if r['language']!='tr']==[r['source_id'] for g in b for r in g if r['language']!='tr'])
    check('30percent_floor_quota',right.domain_draws['fleurs']==(12*8*3)//10)
    check('min_domain_clock',right.turkish_epoch==min(right.domain_draws[d]/right.domain_pool_sizes[d] for d in ('fleurs','cv')))
    saved=right.state_dict();restored=sampler(shorter);restored.load_state_dict(saved)
    check('exact_sampler_suffix',[[r['source_id'] for r in g] for g in right.next_batches()]==[[r['source_id'] for r in g] for g in restored.next_batches()])
    altered=copy.deepcopy(saved);altered['identity']['replay_seed']+=1
    reject('reject_changed_replay_seed',lambda:restored.load_state_dict(altered))
    # Execute real admission functions using only small synthetic files.
    scratch=ROOT/'work/research'/('bpe_domain_admission_fixture_'+uuid.uuid4().hex);scratch.mkdir()
    original_here=helper.HERE;helper.HERE=scratch
    def write(name,value):
        path=scratch/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8');return path
    def rec(path):return {'path':str(path.resolve()),'sha256':sha(path)}
    for name in helper.DOMAIN_SOURCES:(scratch/name).write_bytes((HERE/name).read_bytes())
    recipe={k:None for k in helper.COMMON_TRAINING_KEYS}
    recipe.update(epochs=1.,steps=13009,batch_size=8,accumulation=2,replay_fraction=.5,bucket_size=128,seed=17,
        max_tokens=180,head_lr=5e-5,lora_lr=3e-4,new_token_lr_multiplier=6.,replay_kl=10.,duration_kl=.5,
        feature_distillation=1.,rank=8,alpha=16.,chunk_size=16,tdt_implementation='nemo',warmup_steps=25,schedule='constant',
        min_lr_ratio=.1,eval_per_epoch=4.,eval_epoch_fraction=.026708331664062605,eval_every_updates=250,min_improvement=.002,
        no_encoder_checkpoint=False,phonetic_sh_donors=False,early_stop_patience=3,max_retention_failures=2,
        sampling_mode='domain30',fleurs_fraction=.3,replay_seed=902)
    identity={'sampling_mode':'domain30','replay_seed':902,'turkish_seed':17,'replay_rng_independent_of_turkish':True,
        'fleurs_fraction_numerator':3,'fleurs_fraction_denominator':10,'batch_size':8,'accumulation':2,'replay_per_update':8}
    source_run='synthetic_closed_domain';target_run='synthetic_BPE_target'
    source_checkpoint=write('source_checkpoint_FAKE.json',{'synthetic':True})
    converted=write('converted_weights_FAKE.json',{'synthetic':True})
    selection=write('source_selection.json',{'accepted':True,'selected_step':1,'completed_steps':2,'run_id':source_run,
        'turkish_validation_wer':.4,'heldout_tests_evaluated':False,'selection_split':'FLEURS validation only','weights':str(source_checkpoint)})
    source_config=write('source_config.json',{'run_id':source_run,'base_revision':'541d1f99c6b0c3cd0b11a95167540bb8edefd82b',
        'turkish_clips_used':74883,'replay_clips_used':1595,'config_identity':{'training_arguments':recipe,'domain_sampling_identity':identity,
            'extended_tokenizer_sha256':helper.BASELINE_SEMANTIC_SHA,'tr_pool_sha256':'synthetic_TR_pool','replay_pool_sha256':helper.REPLAY_IDS_SHA},
        'code_sha256':helper.DOMAIN_SOURCES})
    history=write('source_history.json',[{'step':1,'validation':{'retention_gate_passed':True,'turkish_wer':.4}},{'step':2}])
    dev={ 'validation_'+lang:{'asr':{'wer':.4 if lang=='tr' else .03,'utterances':50}} for lang in ('tr','en','es','it','fr','de') }
    devpath=write('results/'+source_run+'_step_1/summary.json',dev)
    write('results/base_v3/summary.json',dev)
    closure={'kind':'root_owned_BPE512_closed_source_v2','passed':True,'root_observed_owned_session_completed':True,'child_exit_code':0,
        'latest_closed_qualified_source_confirmed':True,'source_choice_rationale':'Fixture latest closed qualifying domain BEST, no uniform fallback',
        'allow_BPE512_vocabulary_stage':True,'source_sampling_metadata':identity,'source_run_id':source_run,
        'source_checkpoint':rec(source_checkpoint),'source_selection':rec(selection),'source_run_config':rec(source_config),
        'source_training_history':rec(history),'source_development_summary':rec(devpath)}
    closure_path=write('closure.json',closure)
    conversion=write('conversion.json',{'kind':'HF_Turkish_BPE_512_weights_only_warm_start_v1','status':'complete','passed':True,
        'new_optimizer_required':True,'exact_training_resume':False,'target_run_id':target_run,
        'source':{'path':str(source_checkpoint),'sha256':sha(source_checkpoint),'run_id':source_run,'selected_step':1,'selection_sha256':sha(selection)},
        'output_checkpoint':str(converted),'output_checkpoint_sha256':sha(converted)})
    namespace=SimpleNamespace(**recipe,run_id=target_run,bpe_source_checkpoint=source_checkpoint,bpe_source_selection=selection,
        bpe_source_closure=closure_path,bpe_conversion_report=conversion,resume_weights=converted,resume_exact=None)
    first=helper.closed_source_identity(namespace)
    check('actual_admission_synthetic_closed_domain',first['source_scientific_arguments']==recipe)
    exact=copy.copy(namespace);exact.resume_weights=None;exact.resume_exact=scratch/'immutable_latest_FAKE.pt'
    check('exact_admission_keeps_initial_source_identity',helper.closed_source_identity(exact)==first)
    check('same_sampling_policy_explicit',first['source_sampling_transition']['sampling_policy_changed'] is False
          and first['source_sampling_transition']['target_replay_RNG_independent'] is True)
    changed=copy.copy(namespace);changed.head_lr=2e-5
    reject('reject_changed_source_LR',lambda:helper.closed_source_identity(changed))
    changed=copy.copy(namespace);changed.replay_seed=903
    reject('reject_changed_replay_seed_recipe',lambda:helper.closed_source_identity(changed))
    bad=copy.deepcopy(closure);bad['allow_BPE512_vocabulary_stage']=False;write('closure.json',bad)
    reject('reject_missing_ROOT_vocabulary_permission',lambda:helper.closed_source_identity(namespace));write('closure.json',closure)
    bad=copy.deepcopy(closure);bad['latest_closed_qualified_source_confirmed']=False;write('closure.json',bad)
    reject('reject_older_source_fallback',lambda:helper.closed_source_identity(namespace));write('closure.json',closure)
    old_source=read_source=copy.deepcopy(json.loads(source_config.read_text(encoding='utf-8')))
    old_source['config_identity']['training_arguments']={k:v for k,v in recipe.items() if k in helper.COMMON_TRAINING_KEYS}
    reject('reject_uniform_source_without_domain_identity',lambda:helper.common_source_recipe(old_source,closure))
    config_equiv=write(target_run+'/run_config.json',{'config_identity':{'BPE512':{'source_equivalence_sha256':'pending'}}})
    equivalence=write(target_run+'/BPE512_actual_GPU_source_equivalence.json',{'kind':'HF_BPE512_actual_unsuppressed_GPU_source_equivalence_v1',
        'passed':True,'actual_GPU_decode':True,'before_optimizer_creation':True,'decoded_clips':300,'run_id':target_run,
        'source_identity':first,'student_geometry':helper.GEOMETRY,'converted_weights_sha256':sha(converted),
        'trainer_sha256':sha(HERE/'train_lora_batched_bpe512_v2.py'),'candidate_runtime_sha256':sha(HERE/'hf_bpe512_training_v2.py'),
        'candidate_tokenizer_sha256':helper.TOKENIZER_SEMANTIC_SHA,'raw_transcript_changes':[],'token_duration_emission_changes':[],
        'source_gate':{'retention_gate_passed':True},'candidate_gate':{'retention_gate_passed':True}})
    write(target_run+'/run_config.json',{'config_identity':{'BPE512':{'source_equivalence_sha256':sha(equivalence)}}})
    ev_args=dict(trainer_path=HERE/'train_lora_batched_bpe512_v2.py',runtime_path=HERE/'hf_bpe512_training_v2.py')
    check('established_actual_proof_schema_synthetic',helper.established_source_equivalence(scratch/target_run,target_run,first,**ev_args)['decoded_clips']==300)
    ev=json.loads(equivalence.read_text(encoding='utf-8'));ev['actual_GPU_decode']=False;write(target_run+'/BPE512_actual_GPU_source_equivalence.json',ev)
    reject('reject_CPU_readiness_as_GPU_equivalence',lambda:helper.established_source_equivalence(scratch/target_run,target_run,first,**ev_args))
    helper.HERE=original_here
    check('no_real_ML_imported',all(name not in sys.modules for name in ('torch','nemo','transformers','tokenizers')))
    report={'kind':'isolated_HF_BPE512_closed_domain_admission_CPU_verification_v2','passed':True,'checks':len(checks),'assertions':checks,
        'source_sha256':{name:sha(HERE/name) for name in ('hf_bpe512_training_v2.py','train_lora_batched_bpe512_v2.py','turkish_domain_sampler_v2.py')},
        'actual_GPU_source_equivalence_executed':False,'checkpoint_or_model_or_actual_corpus_read':False,'synthetic_fixture_path':str(scratch),
        'pending':['Actual latest CLOSED domain ROOT receipt','Refresh conversion from that qualifying BEST','Actual300 unsuppressed GPU equality before Adam','Widened joint resource fit','New runtime routes and exact-window supervisor integration']}
    with target.open('x',encoding='utf-8') as out:out.write(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'passed':True,'checks':len(checks),'report':str(target),'source_sha256':report['source_sha256']}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);main(p.parse_args())
