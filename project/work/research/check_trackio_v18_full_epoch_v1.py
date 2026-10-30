"""One producer-shaped stdlib fixture; no actual state, DB, model or GPU."""
import ast,hashlib,json,math,re,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
source=HERE/'track_continuation_v18.py';old=HERE/'track_continuation_v17.py'
tree=ast.parse(source.read_text(encoding='utf-8'));previous=ast.parse(old.read_text(encoding='utf-8'))
checks=[]
def check(value,label):
    if not value:raise AssertionError(label)
    checks.append(label)
def read_json(path,default=None):return json.loads(Path(path).read_text(encoding='utf-8'))
def record(path):
    path=path.resolve();return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
def write(path,value):path.write_text(json.dumps(value),encoding='utf-8');return record(path)
ns={'Path':Path,'ROOT':ROOT,'json':json,'hashlib':hashlib,'math':math,'re':re,'read_json':read_json}
for node in tree.body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        try:ns[node.targets[0].id]=ast.literal_eval(node.value)
        except (ValueError,TypeError):pass
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),ns)
exempt={'follow','parser','main','publish_cycle_memory','start_controls'}
check(all(ast.dump(n,include_attributes=False)==ast.dump(next(x for x in functions if x.name==n.name),include_attributes=False)
    for n in previous.body if isinstance(n,ast.FunctionDef) and n.name not in exempt),'Every inheritedV17 math function ASTunchanged')
with tempfile.TemporaryDirectory(prefix='v18_full_epoch_',dir=ROOT/'work/research') as directory:
    d=Path(directory);recipe={'lora_lr':1e-4,'head_lr':1e-5,'feature_distillation':20.,'replay_kl':40.,'duration_kl':2.,'portuguese_token_kl_multiplier':5.}
    parent={'phase':ns['PT_FOCUSED_PHASE'],'source_step':8750,'steps':25109,'evaluation_languages':['tr','en','es','it','fr','de','pt'],
        'config_identity':{'training_arguments':recipe,'domain_sampling_identity':{'same':'queue'},'retention_constraint_transition':{'feature':20},
            'PT_token_retention_constraint':{'attempted':5},'PT_replay_transition':{'source_step':8000},'retention_numeric_correction':{'source_step':8250}},
        'retention_constraint_transition':{'feature':20},'PT_token_retention_constraint':{'attempted':5},'PT_replay_policy_transition':{'source_step':8000},
        'retention_numeric_correction':{'source_step':8250},'uniform_full_pass':{'pool':253798},'Portuguese_recovery':{'grace':1000}}
    parent_record=write(d/'parent.json',parent)
    change={'source_portuguese_token_kl_multiplier':5.,'active_portuguese_token_kl_multiplier':1.,'PT_token_KL_before':200.,'PT_token_KL_after':40.,
        'active_retention_stop_policy':'monitor_only','strict_all_six_integer_champion_gates_unchanged':True,'memory_nonfinite_crash_protections_unchanged':True,
        'no_plateau_target_or_retention_stop_before_complete_unique_pass':True,'normal_budget_windows_automatically_exact_resume_despite_retention_failures':True}
    review={'kind':'ROOT_USER_TSC_full_epoch_policy_review_v1','passed':True,'chosen_step':9000,'completed_steps':9000,'chosen_checkpoint_role':'latest',
        'allow_full_epoch_monitor_only_continuation':True,'source_counter_preserved':3,'source_selected_BEST_preserved':8250,'source_unqualified_latest_used':True,
        'no_hybrid_training_state_used':True,'explicit_coverage_rewind_clips':0,'renewed_Portuguese_grace':False,
        'source_weights_and_Adam_from_same_complete_checkpoint':True,'source_moments_sampler_RNG_history_preserved':True,'declared_change':change}
    review_record=write(d/'review.json',review)
    receipt_record=write(d/'receipt.json',{'kind':'ROOT_owned_closed_TSC_uniform_full_epoch_v1','chosen_step':9000,'completed_steps':9000,
        'chosen_checkpoint_role':'latest','ROOT_review':review_record,'artifacts':{'run_config':parent_record}})
    declaration={**change,'source_retention_stop_policy':'enforced','retention_gates_for_champion_selection_unchanged':True,
        'retention_failure_counters_and_history_preserved':True,'retention_early_stops_disabled_before_actual_full_unique_pass':True,
        'resource_and_nonfinite_protections_unchanged':True,'preservation_coefficients_by_language':{'pt':[40,2,20],'old_five':[40*5/3,2*5/3,20*5/3]},
        'source_actual_retention_failures':3,'source_actual_selected_step':8250,'new_counter_migration':False,'rewound_TR_clips':0,
        'full_pass_requires_actual_unique_TR_clips':253798,'ROOT_review':review_record}
    transition={'kind':'TSC_uniform_USER_full_epoch_monitor_only_full_state_v1','source_step':9000,'source_receipt':receipt_record,
        'parent_config_identity_sha256':ns['_mirror_config_sha'](parent['config_identity']),'declared':declaration}
    gates={'pt':{'word_errors':71,'passed':False},'de':{'word_errors':71,'passed':False}}
    config={**parent,'source_step':9000,'PT_focus_source_step':8000,'inherited_history_steps':9000,'remaining_uniform_pass_updates':16109,
        'majority_source_step':7621,'uniform_origin_step':6500,'trainable_parameter_tensors':207,'replay_clips_used':1914,
        'portuguese_token_kl_multiplier':1.,'retention_stop_policy':'monitor_only','full_epoch_policy_transition':transition,
        'config_identity':{**parent['config_identity'],'full_epoch_policy_transition':transition,
            'training_arguments':{**recipe,'portuguese_token_kl_multiplier':1.,'retention_stop_policy':'monitor_only'}},
        'initial_source_equivalence':{'passed':True,'actual_GPU_decode':True,'source_selected_step':9000,'compared_fixed_DEV_utterances':350,
            'all_raw_transcripts_and_emitted_paths_exact':True,'all_seven_ASR_counts_exact':True,'all_six_retention_gates_passed':False,
            'source_receipt':receipt_record,'source_retention_gates':gates}}
    captured=[]
    ns['PT_token_phase_contexts_for']=lambda c,h:{'legacy':'unchanged'}
    def previous_values(row,contexts,development):
        captured.append(row);return {'phase/PT_focus_updates':row['step']-8000,'phase/PT_token_constraint_updates':row['step']-8750,
            'retention/all_six_passed':0,'retention/original_five_passed':0,'retention/stop_guard_passed':0}
    ns['PT_token_phase_value_for_record']=previous_values
    base={'step':9000,'audio_seen_seconds':40000.,'turkish_audio_seen_seconds':28000.,'replay_audio_seen_seconds':12000.,'train_wall_seconds':1400.}
    contexts=ns['full_epoch_phase_contexts_for'](config,[base]);row={**base,'step':9010,'audio_seen_seconds':40200.,'turkish_audio_seen_seconds':28140.,
        'replay_audio_seen_seconds':12060.,'train_wall_seconds':1410.,'full_epoch_continuation_updates':10,'PT_token_constraint_updates':260,
        'Portuguese_token_KL_multiplier':1.,'retention_stop_policy':'monitor_only','preservation_coefficients_by_language':declaration['preservation_coefficients_by_language'],
        'validation':{'retention_early_stop_active':False}}
    raw=json.dumps(row,sort_keys=True);values=ns['full_epoch_phase_value_for_record'](row,contexts,{})
    check(raw==json.dumps(row,sort_keys=True) and captured[-1]['Portuguese_token_KL_multiplier']==5 and captured[-1]['preservation_coefficients_by_language']['pt'][0]==200,
        'Rawhistory preserved; only temporaryoldvalidator view projectsrollback')
    check(values['phase/PT_active_replay_KL_coefficient']==40 and values['phase/PT_expected_KL_mean_density']==2.5
        and values['phase/full_epoch_continuation_updates']==10 and values['phase/PT_focus_updates']==1010
        and values['resources/post_full_epoch_audio_minutes_per_wall_minute']==20,'ActualPT40/currentN/post9000rates distinctfromlineage')
    check(values['retention/all_six_passed']==0 and values['retention/original_five_passed']==0
        and values['retention/training_monitor_only_policy']==1 and values['phase/source_actual_retention_failure_counter']==3
        and values['phase/inherited_qualified_BEST_step']==8250,'Strictfalsegates/counter3/BEST8250 retained separatelyfromtrainingpolicy')
    check(not ns['full_epoch_rejected_run_terminal'](config,{'stop_early_reason':'consecutive_retention_gate_failures','full_new_uniform_pass_complete':False})
        and ns['full_epoch_rejected_run_terminal'](config,{'full_new_uniform_pass_complete':True}),
        'Failedgate/budget cannotfinish beforeactualfullunique epoch')
    try:ns['full_epoch_phase_value_for_record']({**row,'retention_stop_policy':'enforced'},contexts,{})
    except ValueError:checks.append('Wrongactivepolicy rejected')
    else:raise AssertionError('wrongpolicy accepted')
report={'schema':1,'kind':'Trackio_V18_USER_full_epoch_focused_stdlib_verification_v1','passed':True,'checks':checks,'assertions':len(checks),
    'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'preserved_V17_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),
    'scope':'Actual newwrapper functions withsynthetic hash-bound parentconfig/USERreview/receipt. Priorancestorcontexts/values explicitlystubbed; everyinheritedmath function ASTunchanged. No actualstate/history/Trackio/DB/ML/GPU.'}
output=HERE/'Trackio_v18_USER_full_epoch_focused_stdlib_verification_v1.json'
with output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'passed':True,'checks':len(checks),'source_sha256':report['source_sha256'],'proof_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))
