"""One focused PT producer binding + tiny shard import; no live state or DB."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import argparse,ast,hashlib,json,math,os,re,tempfile,time
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
source=HERE/'track_continuation_v14.py';parent=HERE/'track_continuation_v13.py'
tree=ast.parse(source.read_text(encoding='utf-8'));old=ast.parse(parent.read_text(encoding='utf-8'))
def read_json(p,default=None):return json.loads(Path(p).read_text(encoding='utf-8')) if Path(p).exists() else default
def write_json(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,ensure_ascii=False),encoding='utf-8')
ns={'Path':Path,'ROOT':ROOT,'json':json,'math':math,'hashlib':hashlib,'os':os,'re':re,'time':time,'datetime':datetime,'timezone':timezone,
    '__file__':str(source),'read_json':read_json,'write_json':write_json,'original':SimpleNamespace(metrics_for=lambda r:{})}
for n in tree.body:
    if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
        try:ns[n.targets[0].id]=ast.literal_eval(n.value)
        except (ValueError,TypeError):pass
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) or isinstance(n,ast.ClassDef) and n.name in ('RunShardStore','StreamRunShardStore')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
checks=[];rejects=[]
def ok(v,label):
    if not v:raise AssertionError(label)
    checks.append(label)
def bad(call,label):
    try:call()
    except (ValueError,KeyError):checks.append(label);rejects.append(label);return
    raise AssertionError('Unexpected acceptance '+label)
def record(p):return {k:ns['file_record'](p)[k] for k in ('path','sha256','bytes')}
def main(args):
    assert not args.output.exists(),'Fresh proof output required'
    exempt={'follow','parser','main','publish_cycle_memory','start_controls'}
    names=[n.name for n in old.body if isinstance(n,ast.FunctionDef) and n.name not in exempt]
    ok(all(ast.dump(n,include_attributes=False)==ast.dump(next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==n.name),include_attributes=False) for n in old.body if isinstance(n,ast.FunctionDef) and n.name in names),'Every inherited V13 metric/math function AST unchanged')
    with tempfile.TemporaryDirectory(prefix='v14_PT_tiny_',dir=ROOT/'work/research') as d:
        d=Path(d);recipe={'head_lr':1e-5,'lora_lr':1e-4,'replay_fraction':.125,'replay_kl':40.,'duration_kl':2.,'feature_distillation':4.}
        prior_counts={'pt':100,'en':180,'es':180,'it':180,'fr':180,'de':180}
        focus={'source_step':8000,'quota_boundary_step':7621,'uniform_origin_step':6500,'configured_end_step':25109,'Portuguese_per_update':1,'old_five_per_update':1,'prior_Portuguese_draws':100,'prior_old_five_draws':900}
        parent_config={'phase':ns['LOWER_LORA_PHASE'],'source_step':7750,'steps':25109,'config_identity':{'training_arguments':recipe,'domain_sampling_identity':{'parent':'samequota'}},
            'uniform_full_pass':{'quota_boundary':7621},'Portuguese_recovery':{'grace_new_updates':1000},'evaluation_languages':['tr','en','es','it','fr','de','pt']}
        parent_path=d/'parent.json';write_json(parent_path,parent_config);receipt_path=d/'receipt.json';write_json(receipt_path,{'chosen_step':8000,'completed_steps':8000,'artifacts':{'run_config':record(parent_path)}})
        declared={'source_step':8000,'quota_boundary_step':7621,'uniform_origin_step':6500,'configured_end_step':25109,'head_lr':1e-5,'lora_lr':1e-4,
            'all207_Adam_moments_and_global_clocks_preserved':True,'fresh_optimizer_or_sampler':False,'renewed_Portuguese_grace':False,'parent_failure_counter_preserved':True,'all_six_required_for_every_BEST':True,
            'declared_change':{'PT_replay_clips_per_update':1,'original_five_replay_clips_per_update':1,'PT_preservation_coefficients':[40,2,4],'original_five_preservation_coefficients':[40*(5/3),2*(5/3),4*(5/3)]}}
        gates={l:{'passed':l!='pt'} for l in ('en','es','it','fr','de','pt')}
        config={**parent_config,'phase':ns['PT_FOCUSED_PHASE'],'source_step':8000,'majority_source_step':7621,'uniform_origin_step':6500,'inherited_history_steps':8000,'remaining_uniform_pass_updates':17109,'trainable_parameter_tensors':207,'replay_clips_used':1914,
            'PT_replay_policy_transition':declared,'config_identity':{**parent_config['config_identity'],'domain_sampling_identity':{**parent_config['config_identity']['domain_sampling_identity'],'sampler_family':'TSC_uniform_majority_PT_focus_six_replay_sampler_v1','PT_focus_transition':focus},
                'PT_replay_transition':{'kind':'TSC_uniform_majority_PT_replay_state_transition_v1','declared':declared,'source_receipt':record(receipt_path),'parent_config_identity_sha256':ns['_mirror_config_sha'](parent_config['config_identity'])}},
            'initial_source_equivalence':{'passed':True,'actual_GPU_decode':True,'source_selected_step':8000,'compared_fixed_DEV_utterances':350,'all_raw_transcripts_and_emitted_paths_exact':True,'all_seven_ASR_counts_exact':True,
                'source_receipt':record(receipt_path),'source_retention_gates':gates,'all_six_retention_gates_passed':False}}
        base={'step':8000,'audio_seen_seconds':20000.,'turkish_audio_seen_seconds':14000.,'replay_audio_seen_seconds':6000.,'train_wall_seconds':1000.,'replay_language_clip_counts':prior_counts}
        # Old math is AST-identical and previously tested; stub only its context
        # admission/delegation here to isolate the new producer contract.
        ns['loralr_phase_contexts_for']=lambda c,h:{'legacy':'unchanged'}
        ns['loralr_phase_value_for_record']=lambda r,c,d:{'phase/majority_updates':r['step']-7621,'phase/lora_lr_fork_updates':r['step']-7750,'dev/inherited':1}
        contexts=ns['ptfocus_phase_contexts_for'](config,[base])
        row={**base,'step':8250,'PT_focus_updates':250,'PT_focus_source_step':8000,'audio_seen_seconds':25000.,'turkish_audio_seen_seconds':17500.,'replay_audio_seen_seconds':7500.,'train_wall_seconds':1100.,
            'replay_language_clip_counts':{**prior_counts,'pt':350,'en':430},'Turkish_domain_coverage':{'PT_focus_transition':{'source_step':8000,'local_updates':250,'Portuguese_draws_after_transition':250,'old_five_draws_after_transition':250}},
            'preservation_coefficients_by_language':{'pt':[40,2,4],'old_five':[40*5/3,2*5/3,4*5/3]}}
        values=ns['ptfocus_phase_value_for_record'](row,contexts,{})
        ok(values['resources/post_PT_focus_audio_minutes_per_wall_minute']==50 and values['resources/post_PT_focus_turkish_audio_minutes_per_wall_minute']==35 and values['phase/PT_focus_updates']==250,'Actual8000 phase rates/N separate from majority7621 andLR7750')
        ok(values['phase/PT_expected_KL_mean_density']==2.5 and math.isclose(values['phase/each_old_five_expected_KL_mean_density'],5/6) and values['phase/PT_expected_preservation_density_multiplier']==3,'ExpectedPTdensitytriples while eacholdfive densityretained; no gradientclaim')
        dev={8000:{'dev/wer_'+l+'_percent':40. for l in ('tr','en','es','it','fr','de','pt')}}
        initial=ns['ptfocus_phase_value_for_record'](base,contexts,dev)
        ok(initial['phase/PT_focus_updates']==0 and initial['retention/all_six_passed']==0 and initial['retention/Portuguese_stop_grace_active']==0,'Only actual8000 sevenDEV initial; noPTgrace/BEST fabrication')
        bad(lambda:ns['ptfocus_phase_value_for_record']({**row,'replay_language_clip_counts':{**row['replay_language_clip_counts'],'pt':349}},contexts,{}),'Wrong PT drawclock rejected')
        archive=d/'source_v13_shards';archive.mkdir();payload={'unknown':['retained','İ'],'finished':False,'reference_base_pt_verified_v1':{'sha256':'a'*64}}
        raw=json.dumps(payload).encode();digest=hashlib.sha256(raw).hexdigest();shard=archive/(digest+'.json');shard.write_bytes(raw)
        source_index=d/'source_v13.json';old_index={'schema':1,'kind':'Trackio_run_shard_index_v13','source_import':{'complete':True,'source_archive_directories':[str(archive)]},'runs':{'old':{'shard':record(shard),'finished_truthy':False}},'unknown_index_flag':{'exit1_review_preserved':True}}
        write_json(source_index,old_index);store=ns['StreamRunShardStore'](d/'new_v14.json');store.migrate(source_index,source_closed_confirmed=True)
        ok(store.index['source_import']['source_V13_index_payload']==old_index and store.activate('old')==payload,'Full V13 index/map/unknownflags imported unchanged')
        store.set_flag('new_flag',True);store.flush();ok(shard.exists() and read_json(source_index)==old_index,'Sourcearchives andsourceindex never rewritten/deleted')
    report={'schema':1,'kind':'inactive_Trackio_v14_PTfocus_stdlib_verification_v1','passed':True,'checks':checks,'assertions':len(checks),'rejections':len(rejects),
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'preserved_V13_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),'legacy_functions_AST_unchanged':names,
        'scope':'Actual new PT context/value functions, synthetic hash-bound parentconfig/receipt and tiny actualshards. Old admission/value delegation explicitly stubbed; every oldmath AST unchanged. No actualmirror/history/DB/ML/GPU.'}
    with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'passed':True,'checks':len(checks),'source_sha256':report['source_sha256'],'proof_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'Trackio_v14_PTfocus_stdlib_verification_v1.json');main(p.parse_args())
