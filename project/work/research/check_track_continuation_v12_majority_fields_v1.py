"""Focused extracted-stdlib V12 fields/shard fixture; no Trackio or real state."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import argparse,ast,copy,hashlib,json,math,os,re,tempfile,time
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
source=HERE/'track_continuation_v12.py';old=HERE/'track_continuation_v11.py'
tree=ast.parse(source.read_text());prior=ast.parse(old.read_text())
def read_json(path,default=None):return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else default
def write_json(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')
metric_node=next(n for n in ast.parse((HERE/'track_experiment.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='metrics_for')
metric_ns={};exec(compile(ast.Module(body=[metric_node],type_ignores=[]),'frozen original metrics','exec'),metric_ns)
ns={'Path':Path,'ROOT':ROOT,'math':math,'json':json,'hashlib':hashlib,'os':os,'re':re,'time':time,
    'datetime':datetime,'timezone':timezone,'read_json':read_json,'write_json':write_json,'__file__':str(source),
    'original':SimpleNamespace(metrics_for=metric_ns['metrics_for'])}
for node in tree.body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        try:ns[node.targets[0].id]=ast.literal_eval(node.value)
        except (ValueError,TypeError):pass
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name not in ('follow','main','parser','development_from_steps','publish_cycle_memory')]
classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('RunShardStore','StreamRunShardStore')]
exec(compile(ast.Module(body=functions+classes,type_ignores=[]),str(source),'exec'),ns)
checks=[];rejections=[]
def ok(condition,label):
    if not condition:raise AssertionError(label)
    checks.append(label)
def bad(call,label):
    try:call()
    except (ValueError,KeyError,TypeError):checks.append(label);rejections.append(label);return
    raise AssertionError(label)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
for node in prior.body:
    if isinstance(node,ast.FunctionDef) and node.name not in ('follow','parser'):
        changed=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==node.name)
        ok(ast.dump(node,include_attributes=False)==ast.dump(changed,include_attributes=False),'unchanged V11 function AST '+node.name)
h=7621;origin=6500;prior_draws=8968;pool=253798;budget=17488
gates={lang:{'passed':lang!='pt','word_errors':99 if lang=='pt' else 1,'allowed_word_errors':66 if lang=='pt' else 5} for lang in ('en','es','it','fr','de','pt')}
proof={'passed':True,'actual_GPU_decode':True,'source_selected_step':h,'before_first_majority_update':True,
    'compared_fixed_DEV_utterances':350,'all_seven_ASR_counts_exact':True,'all_six_retention_gates_passed':False,'source_retention_gates':gates}
declared={'source_step':h,'prior_uniform_draws':prior_draws,'TR_per_update':14,'replay_per_update':2,'configured_end_step':h+budget,
    'renewed_Portuguese_grace':False,'preservation_coefficients':{'replay_kl':40.,'duration_kl':2.,'feature_distillation':4.}}
config={'run_id':'synthetic_majority','phase':ns['UNIFORM_MAJORITY_PHASE'],'source_step':h,'uniform_origin_step':origin,'steps':h+budget,
    'inherited_history_steps':h,'trainable_parameter_tensors':207,'replay_clips_used':1914,'remaining_uniform_pass_updates':budget,
    'uniform_full_pass':{'source_step':origin,'transition_source_step':h,'prior_uniform_draws':prior_draws,'expected_new_unique_TR_clips':pool,'additional_update_budget':budget,'end_step':h+budget},
    'initial_source_equivalence':proof,'config_identity':{'training_arguments':{'replay_fraction':.125,'replay_kl':40.,'duration_kl':2.,'feature_distillation':4.,
    'head_lr':1e-5,'lora_lr':3e-4,'new_token_lr_multiplier':6},'majority_transition':{'declared':declared}}}
baseline6500={'step':origin,'turkish_unique_clips_seen':38000}
baseline={'step':h,'audio_seen_seconds':100000.,'train_wall_seconds':10000.,'turkish_audio_seen_seconds':25000.,'replay_audio_seen_seconds':75000.,
    'turkish_unique_clips_seen':46000,'turkish_clips_seen':h*8,'replay_clips_seen':h*8,
    'Turkish_domain_coverage':{'uniform_pass':{'source_step':origin,'draws':prior_draws,'unique_clips_seen':prior_draws}},
    'tdt_loss':1.,'replay_kl':.01,'gradient_norm':.5,'elapsed_seconds':100.,'peak_allocated_vram_gb':4.,'validation':{'retention_gate_passed':False,'retention_wer_deltas':{'pt':.03}}}
context=ns['uniform_majority_context'](config,[baseline6500,baseline]);contexts=ns['majority_phase_contexts_for'](config,[baseline6500,baseline])
def row(n):
    draws=prior_draws+14*n;unique=min(draws,pool)
    return {**baseline,'step':h+n,'majority_updates':n,'majority_source_step':h,'uniform_origin_step':origin,'uniform_full_updates':h+n-origin,
        'head_lr':1e-5,'lora_lr':3e-4,'new_token_update_multiplier':6,'replay_kl_coefficient':40.,'duration_kl_coefficient':2.,'feature_distillation_coefficient':4.,
        'audio_seen_seconds':100000+100*n,'train_wall_seconds':10000+2*n,'turkish_audio_seen_seconds':25000+70*n,'replay_audio_seen_seconds':75000+30*n,
        'turkish_unique_clips_seen':max(38000,unique),'turkish_clips_seen':h*8+14*n,'replay_clips_seen':h*8+2*n,
        'Turkish_domain_coverage':{'uniform_pass':{'source_step':origin,'local_updates':h+n-origin,'pool_clips':pool,'draws':draws,'unique_clips_seen':unique,
        'repeat_visits':draws-unique,'fraction_unique_seen':unique/pool,'epoch_clock':draws/pool,'full_pass_complete':unique==pool,
        'quota_segments':[{'first_step':6501,'last_step':h,'updates':h-origin,'TR_per_update':8,'replay_per_update':8,'TR_draws':prior_draws,'replay_draws':prior_draws},
        {'first_step':h+1,'last_step':h+n,'updates':n,'TR_per_update':14,'replay_per_update':2,'TR_draws':14*n,'replay_draws':2*n}]},
        'majority_transition':{'local_updates':n,'source_step':h}},
        'validation':{'retention_gate_passed':False,'retention_wer_deltas':{'pt':.03},'old_five_retention_gate_passed':True,'Portuguese_grace_active_for_stop_only':False,'stop_retention_gate_passed':False}}
value=ns['uniform_majority_values_for'](row(100),context)
ok(value['resources/post_majority_audio_minutes_per_wall_minute']==50 and value['resources/post_majority_turkish_audio_minutes_per_wall_minute']==35
    and value['resources/post_majority_replay_audio_minutes_per_wall_minute']==15,'postH cumulative subtraction excludes prior6500 plus8/8 audio')
ok(value['phase/majority_updates']==100 and value['phase/uniform_full_updates']==1221 and value['phase/expected_full207_Adam_clock']==7721,'M versus whole-pass/global207 clocks distinct')
ok(value['data/uniform_local_TR_draws']==10368 and value['data/uniform_local_unique_TR_clips']==10368,'original8968 queue frontier plus14M preserved')
ok(value['phase/expected_RP_replay_KL_mean_density']==5 and value['phase/expected_RP_duration_KL_mean_density']==.25 and value['phase/expected_RP_feature_mean_density']==.5,'actual40/2/4 declared density distinct from same-gradient claim')
ok(ns['uniform_majority_values_for']({**row(100),'elapsed_seconds':.01,'process_wall_seconds':.1},context)==value,'exact-window process timestamps do not reset H rates')
final=ns['uniform_majority_values_for'](row(budget),context)
ok(final['data/uniform_local_unique_TR_clips']==pool and final['data/uniform_local_repeat_visits']==2 and final['data/uniform_full_pass_complete']==1,'all253798 unique authority and two whole-update tail repeats')
ok(ns['majority_phase_baseline_steps'](config)=={6500,h} and ns['majority_phase_initial_development_step'](config)==h,'bounded two-row source baseline retains origin coverage only')
dev={h:{'dev/wer_'+language+'_percent':50. for language in ('tr','en','es','it','fr','de','pt')},7500:{'dev/wer_tr_percent':99.}}
actual=ns['majority_phase_value_for_record'](baseline,contexts,dev)
ok(actual['retention/all_six_passed']==0 and actual['retention/Portuguese_stop_grace_active']==0 and actual['phase/majority_updates']==0,'sourceH failedall6 truth and elapsed PTgrace reproduced without qualifiedalias')
inherited=ns['majority_phase_value_for_record']({**baseline,'step':7500},contexts,dev)
ok(inherited['phase/inherited_history']==1 and not any(k.startswith(('dev/','retention/')) for k in inherited),'all<=H inherited oldDEV suppressed exceptactual sourceH7')
ok(not ns['majority_phase_actual_development_row'](config,baseline) and ns['majority_phase_actual_development_row'](config,row(100)),'new DEV candidates require actual newM row')
selected={'run_id':config['run_id'],'source_step':h,'uniform_origin_step':6500,'majority_updates':100,'uniform_full_updates':1221,'completed_steps':7721,
    'accepted':False,'selected_step':None,'turkish_validation_wer':None,'full_new_uniform_pass_complete':False,'stop_early_reason':'soft_max_wall_seconds'}
ok(ns['majority_selection_values'](config,selected,7721)['selection/all_six_eligible_best_available']==0 and not ns['majority_rejected_run_terminal'](config,selected),'None all6 BEST and budget continuation retained')
oldbest={**selected,'accepted':True,'selected_step':7500,'turkish_validation_wer':.38}
ok(ns['majority_selection_values'](config,oldbest,7721)['selection/eligible_BEST_inherited_at_or_before_H']==1,'parent qualifying BEST pointer remains inherited separate from ownbranch gain')
for key,value in [('majority_updates',101),('uniform_full_updates',101),('head_lr',5e-5),('replay_kl_coefficient',10),('turkish_clips_seen',7721*8),('replay_clips_seen',7721*8)]:
    bad(lambda key=key,value=value:ns['uniform_majority_values_for']({**row(100),key:value},context),'reject actual majority mismatch '+key)
broken=row(100);broken['validation']['Portuguese_grace_active_for_stop_only']=True
bad(lambda:ns['uniform_majority_values_for'](broken,context),'reject renewed PTgrace')
bad(lambda:ns['uniform_majority_context']({**config,'source_step':7000},[baseline6500,baseline]),'reject wrong sourceH')
bad(lambda:ns['majority_selection_values'](config,{**selected,'selected_step':6500},7721),'reject None borrowing oldfivechampion')
with tempfile.TemporaryDirectory(dir=ROOT/'work/research',prefix='v12_tiny_shards_') as folder:
    folder=Path(folder);old_dir=folder/'old_v11_shards';v10=folder/'old_v10_shards';v9=folder/'old_v9_shards'
    for directory in (old_dir,v10,v9):directory.mkdir()
    runs={}
    for name,directory in [('old41',v9),('PT04',old_dir),('oldV10',v10)]:
        payload={'reference_base_v3':True,'unknown':{'unicode':'İ','flags':[1,False]},'7':{'train/loss':1.},'finished':name=='old41'}
        data=json.dumps(payload,ensure_ascii=False).encode();digest=hashlib.sha256(data).hexdigest();path=directory/(digest+'.json');path.write_bytes(data)
        runs[name]={'shard':ns['file_record'](path),'finished_truthy':payload['finished']}
    index={'schema':1,'kind':'Trackio_run_shard_index_v11','source_import':{'complete':True,'source_archive_directories':[str(v10),str(v9)]},
        'runs':runs,'live_versions':{},'obsolete_live_shards':[],'unknown_index_payload':{'old41':True}}
    origin_path=folder/'old_v11.json';write_json(origin_path,index)
    store=ns['StreamRunShardStore'](folder/'new_v12.json')
    bad(lambda:store.migrate(origin_path),'reject unobserved liveV11 migration')
    store.migrate(origin_path,source_closed_confirmed=True)
    ok(store.index['source_import']['source_V11_index_payload']==index,'entire V11 payload/all unknownflags preserved')
    for name in runs:ok(store.activate(name)['unknown']=={'unicode':'İ','flags':[1,False]},'preserved archive map '+name)
    store.set_flag('new_majority_flag',True);store.flush()
    ok(all(Path(item['shard']['path']).exists() for item in runs.values()),'own pruning never deletes V11/V10/V9 source archives')
    ok(read_json(origin_path)==index,'sourceindex remains immutable after ownflags')
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
report={'kind':'Trackio_v12_majority_field_stdlib_verification_v1','passed':True,'checks_passed':len(checks),'checks':checks,
    'rejections':rejections,'source':{'path':str(source),'sha256':sha(source)},'source_V11_sha256':sha(old),
    'actual_state_migration_Trackio_database_history_or_GPU_read':False,'all_training_fields_and_archive_maps_explicit_tiny_fixtures':True}
with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
print(json.dumps({'passed':True,'checks':len(checks),'rejections':len(rejections),'output':str(args.output)}))
