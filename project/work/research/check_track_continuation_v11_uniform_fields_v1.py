"""Extracted stdlib metric and tiny shard fixtures; no real mirror/state/DB."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import ast
import copy
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
source=HERE/'track_continuation_v11.py';raw=source.read_bytes();tree=ast.parse(raw.decode())
old=ast.parse((HERE/'track_continuation_v10.py').read_text(encoding='utf-8'))
legacy={'finite_number','resume_context','values_for','acquisition_context','acquisition_values_for',
    'continued_acquisition_context','continued_acquisition_values_for','scheduled_row_lr','scheduled_acquisition_context',
    'scheduled_acquisition_values_for','adaptation_context','adaptation_values_for','shared_head_context','shared_head_values_for',
    'baseline_steps','initial_development_step','actual_development_row','contexts_for','value_for_record','cycle_rows','development_cycle',
    'portuguese_reference_record','required_eval_languages','portuguese_base_reference'}
new={'uniform_full_context','uniform_full_values_for','full_phase_baseline_steps','full_phase_initial_development_step',
    'full_phase_actual_development_row','full_phase_contexts_for','full_phase_value_for_record','uniform_full_selection_values','rejected_run_terminal','file_record'}
nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in legacy|new]
nodes += [node for node in tree.body if isinstance(node,ast.ClassDef) and node.name in ('RunShardStore','StreamRunShardStore')]
def read_json(path,default=None):return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else default
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')
metrics=next(n for n in ast.parse((HERE/'track_experiment.py').read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='metrics_for')
metric_ns={};exec(compile(ast.Module(body=[metrics],type_ignores=[]),'original metrics','exec'),metric_ns)
ns={'Path':Path,'ROOT':ROOT,'math':math,'json':json,'hashlib':hashlib,'os':os,'re':re,'time':time,
    'datetime':datetime,'timezone':timezone,'read_json':read_json,'write_json':write_json,
    '__file__':str(source),'original':SimpleNamespace(metrics_for=metric_ns['metrics_for'])}
for node in tree.body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        try:value=ast.literal_eval(node.value)
        except (ValueError,TypeError):continue
        ns[node.targets[0].id]=value
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
checks=[];rejections=[]
def ok(value,label):
    if not value:raise AssertionError(label)
    checks.append(label)
def bad(call,label):
    try:call()
    except (ValueError,KeyError,TypeError):checks.append(label);rejections.append(label);return
    raise AssertionError('unexpected acceptance '+label)
for name in legacy:
    a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
    b=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    ok(ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),'unchanged legacy AST '+name)
initial={'passed':True,'actual_GPU_decode':True,'source_selected_step':6500,'before_first_uniform_update':True,
    'compared_fixed_DEV_utterances':300,'Portuguese_fixed_DEV_measured_utterances':50,
    'all_five_original_retention_gates_passed':True,'all_six_retention_gates_passed':False,
    'Portuguese_source_word_errors':96,'Portuguese_allowed_word_errors':66}
config={'run_id':'fixture_uniform','phase':ns['UNIFORM_FULL_PHASE'],'source_step':6500,'inherited_history_steps':6500,'steps':38225,
    'uniform_full_pass':{'source_step':6500,'additional_update_budget':31725,'end_step':38225,'expected_new_unique_TR_clips':253798},
    'trainable_parameter_tensors':207,'replay_clips_used':1914,'evaluation_languages':['tr','en','es','it','fr','de','pt'],
    'initial_source_equivalence':initial,'Portuguese_recovery':{'grace_new_updates':1000,'all_six_required_for_every_eligible_BEST':True,'source_6500_all_six_qualified':False},
    'config_identity':{'training_arguments':{'sampling_mode':'uniform','head_lr':1e-5,'lora_lr':3e-4,'new_token_lr_multiplier':6},
        'uniform_full_transition':{'declared':{'all207_weights_and_Adam_states_retained':True,'fresh_optimizer_or_warmup':False,
            'source_selected_step':6500,'full_TR_clips':253798,'new_uniform_pass_updates':31725}}}}
baseline={'step':6500,'audio_seen_seconds':10000.,'train_wall_seconds':1000.,'turkish_audio_seen_seconds':3000.,'replay_audio_seen_seconds':7000.,
    'turkish_unique_clips_seen':40000,'tdt_loss':1.,'replay_kl':.01,'gradient_norm':.5,'elapsed_seconds':123.,'peak_allocated_vram_gb':4.,
    'validation':{'retention_gate_passed':True,'retention_wer_deltas':{'en':0.}}}
context=ns['uniform_full_context'](config,[baseline]);contexts=ns['full_phase_contexts_for'](config,[baseline])
def row(n):
    count=min(8*n,253798)
    return {**baseline,'step':6500+n,'uniform_full_updates':n,'head_lr':1e-5,'lora_lr':3e-4,'new_token_update_multiplier':6,
        'audio_seen_seconds':10000+100*n,'train_wall_seconds':1000+10*n,'turkish_audio_seen_seconds':3000+30*n,'replay_audio_seen_seconds':7000+70*n,
        'turkish_unique_clips_seen':max(40000,count),'turkish_clips_seen':(6500+n)*8,
        'Turkish_domain_coverage':{'uniform_pass':{'source_step':6500,'local_updates':n,'pool_clips':253798,'draws':8*n,'unique_clips_seen':count,
            'repeat_visits':8*n-count,'fraction_unique_seen':count/253798,'epoch_clock':8*n/253798,'full_pass_complete':count==253798}},
        'validation':{'retention_gate_passed':False,'retention_wer_deltas':{'pt':.03},'old_five_retention_gate_passed':True,
            'Portuguese_grace_active_for_stop_only':n<1000,'stop_retention_gate_passed':n<1000}}
value=ns['uniform_full_values_for'](row(250),context)
ok(value['phase/uniform_full_updates']==250 and value['phase/expected_full207_Adam_clock']==6750,'localN distinct from declared global207 clock')
ok(value['resources/post_uniform_audio_minutes_per_wall_minute']==10. and value['resources/post_uniform_turkish_audio_minutes_per_wall_minute']==3.
    and value['resources/post_uniform_replay_audio_minutes_per_wall_minute']==7.,'actual post6500 mixed/TR/replay rates exclude inherited counters')
changed={**row(250),'elapsed_seconds':.2,'process_wall_seconds':.1}
ok(ns['uniform_full_values_for'](changed,context)==value,'fresh exact process/window timestamps do not reset phase-throughput baseline')
ok(value['data/uniform_local_unique_TR_clips']==2000 and value['data/cumulative_TR_unique_union']==40000,'local uniform and historical unique union kept separate')
final=ns['uniform_full_values_for'](row(31725),context)
ok(final['data/uniform_full_pass_complete']==1 and final['data/uniform_local_unique_TR_clips']==253798 and final['data/uniform_local_repeat_visits']==2,'whole fullpass and final two nextcycle repeats honest')
ok(ns['uniform_full_values_for'](row(1000),context)['retention/Portuguese_stop_grace_active']==0,'PT stop grace ends at1000 without qualifying-best waiver')
history_old={**baseline,'step':6250};dev={6250:{'dev/wer_tr_percent':99.},6500:{'dev/wer_'+language+'_percent':39. for language in ('tr','en','es','it','fr','de','pt')}}
inherited=ns['full_phase_value_for_record'](history_old,contexts,dev)
ok(inherited['phase/inherited_history']==1 and not any(k.startswith(('dev/','retention/')) for k in inherited),'inherited DEV and old5 gates never alias newphase')
source_values=ns['full_phase_value_for_record'](baseline,contexts,dev)
ok(source_values['phase/uniform_full_updates']==0 and source_values['retention/all_dev_gates_passed']==0 and source_values['dev/wer_pt_percent']==39.,'actual initial7 source summary keeps PT failure distinct from inherited old5 pass')
ok(ns['full_phase_initial_development_step'](config)==6500 and ns['full_phase_baseline_steps'](config)=={6500},'bounded context retains only source6500 baseline')
ok(not ns['full_phase_actual_development_row'](config,history_old) and ns['full_phase_actual_development_row'](config,row(250)),'only newuniform validation rows are new DEV candidates')
selection={'run_id':'fixture_uniform','source_step':6500,'uniform_full_updates':250,'completed_steps':6750,'accepted':False,
    'selected_step':None,'turkish_validation_wer':None,'full_new_uniform_pass_complete':False,'stop_early_reason':'soft_max_wall_seconds'}
ok(ns['uniform_full_selection_values'](config,selection,6750)['selection/all_six_eligible_best_available']==0,'no all6 BEST stays None without fabricated zero checkpoint')
ok(not ns['rejected_run_terminal'](config,selection),'budget recovery closure does not finish SAME-ID mirror')
ok(ns['rejected_run_terminal'](config,{**selection,'stop_early_reason':'consecutive_retention_gate_failures'}),'actual scientific retention closure may finish rejected run')
for key,value in [('uniform_full_updates',251),('head_lr',5e-5),('turkish_unique_clips_seen',1999),('audio_seen_seconds',10000.),('new_token_update_multiplier',1)]:
    bad(lambda key=key,value=value:ns['uniform_full_values_for']({**row(250),key:value},context),'reject inconsistent actual uniform row '+key)
for field,value in [('replay_clips_used',1595),('trainable_parameter_tensors',3),('source_step',7000)]:
    bad(lambda field=field,value=value:ns['uniform_full_context']({**config,field:value},[baseline]),'reject unapproved uniform source '+field)
bad(lambda:ns['uniform_full_selection_values'](config,{**selection,'selected_step':6500},6750),'no all6 winner cannot borrow old-five source')
with tempfile.TemporaryDirectory(prefix='v11_shards_tiny_',dir=ROOT/'work/research') as directory:
    directory=Path(directory);v9=directory/'source_v9_shards';v10=directory/'source_v10_shards';v9.mkdir();v10.mkdir()
    def shard(folder,payload):
        data=json.dumps(payload).encode();digest=hashlib.sha256(data).hexdigest();path=folder/(digest+'.json');path.write_bytes(data)
        return ns['file_record'](path)
    a={'unknown':{'flag':[1,'İ']},'reference_base_v3':True,'7':{'train/loss':1.},'finished':'actual-old'}
    b={'sameproject':'parakeet-turkish','reference_base_pt_verified_v1':{'sha256':'a'*64},'finished':False}
    old_index={'schema':1,'kind':'Trackio_run_shard_index_v10','source_import':{'complete':True,'source_archive_directory':str(v9)},
        'runs':{'A':{'shard':shard(v9,a),'finished_truthy':True},'B':{'shard':shard(v10,b),'finished_truthy':False}},
        'live_versions':{},'obsolete_live_shards':[],'unknown_index_field':'preserved'}
    origin=directory/'source_v10.json';write_json(origin,old_index)
    # Match producer's source-directory naming without touching any real state.
    old_index['runs']['B']['shard']['path']=str(v10/Path(old_index['runs']['B']['shard']['path']).name)
    store=ns['StreamRunShardStore'](directory/'new_v11.json')
    bad(lambda:store.migrate(origin),'live/unconfirmed V10 handoff rejected')
    store.migrate(origin,source_closed_confirmed=True)
    ok(store.index['source_import']['source_V10_index_payload']==old_index,'entire old index unknown fields preserved')
    ok(store.activate('A')==a and store.is_finished('A'),'inherited V9 shard unknown fields/flags preserved')
    ok(store.activate('B')==b,'V10 own shard PT flags preserved')
    store.set_flag('new_fulluniform_flag',True);store.flush()
    ok(all(Path(v['shard']['path']).exists() for v in old_index['runs'].values()),'source V9/V10 shards never pruned')
    ok(read_json(origin)==old_index,'source V10 index never rewritten')
ok('trackio' not in sys.modules and 'torch' not in sys.modules,'no Trackio/ML imported')
report={'schema':1,'kind':'inactive_Trackio_v11_uniform_field_stdlib_verification_v1','passed':True,'source_sha256':hashlib.sha256(raw).hexdigest(),
    'checks':checks,'assertions':len(checks),'rejections':len(rejections),
    'scope':'Actual AST-extracted functions and tiny temporary synthetic shard maps; no real history/state/Trackio/DB/GPU or current source admission.',
    'full207_Adam_clock':'Expected from completed global worker updates; no tensor clocks inspected',
    'actual_V10_handoff_performed':False,'actual_fullrun_measured':False}
output=HERE/'Trackio_v11_uniform_field_stdlib_verification_v1.json'
with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
print(json.dumps({'passed':True,'checks':len(checks),'rejections':len(rejections),'source_sha256':report['source_sha256'],
    'report_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))
