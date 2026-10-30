"""Measured single-LR repair with honest complete-state BEST coverage rewind."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,ast
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def record(p):
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p.resolve()),'sha256':h,'bytes':p.stat().st_size}
closurepath=HERE/'feature5x_actual_CLOSED8750_retention_review_v1.json';closed=read(closurepath)
failedpath=HERE/'PT_token5x_actual_CLOSED9000_retention_review_v1.json';failed=read(failedpath)
assert closed['actual_owned_exit']=={'session_id':98758,'exit_code':0,'tool_chunk':'0848c0'} and failed['consecutive_retention_failures']==3
a=dict(closed['artifacts']);a['chosen_checkpoint']=a['source_best_checkpoint'];a.pop('source_best_checkpoint');a.pop('source_best_DEV_summary')
dev=HERE/'results'/'adaptive_20261012_TR_uniform_featureKD5x_02_step_8250'
for key,name in [('chosen_DEV_summary','summary.json'),('chosen_DEV_metadata','metadata.json'),('chosen_DEV_metadata_pt','metadata_pt.json')]:a[key]=record(dev/name)
for lang in ('tr','en','es','it','fr','de','pt'):a['chosen_DEV_'+lang]=record(dev/('validation_'+lang+'.jsonl'))
review={'schema':1,'kind':'ROOT_delegated_measured_TSC_feature_LoRA_LR_review_v1','passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),
    'decision_actor':'/root/benchmark_design under ROOT explicit delegated actual repair authority','ROOT_independent_tensor_review_claimed':False,
    'source_run_id':closed['run_id'],'chosen_step':8250,'completed_steps':8750,'chosen_checkpoint_role':'best',
    'source_actual_owned_exit':closed['actual_owned_exit'],'actual_closed_source_review':record(closurepath),
    'recent_failed_PT_token_review':record(failedpath),'actual_diagnostic':record(HERE/'retention_components_8750_actual_GPU_v3.json'),
    'paired_drift':record(HERE/'PT_token5x_DE_PT8750_to9000_saved_changes_v1.json'),
    'decision':'Reduce encoder update size from a qualified coherent full state. The measured encoder hybrid carried larger conditional DE/PT drift than heads; feature20 helped German and Turkish, but PTtoken200 failed. Smaller LoRA LR is a new one-factor constraint, not a guaranteed recovery. Unqualified latest9000 drift cannot be assumed reversible simply by reducing its future LR.',
    'declared_change':{'parameter':'lora_lr','source_value':1e-4,'active_value':3e-5,'feature_distillation_unchanged':20.,
        'token_KL_unchanged':40.,'duration_KL_unchanged':2.,'head_LR_unchanged':1e-5,'TR_per_update':14,
        'PT_replay_per_update':1,'old_five_replay_per_update':1,'old_five_language_weight_multiplier':5/3,'fixed_integer_gate_limits_unchanged':True},
    'allow_declared_feature_LoRA_LR_fork':True,'source_counter_preserved':0,'source_selected_BEST_preserved':8250,
    'source_unqualified_latest_used':False,'source_chosen_is_all_six_count_qualified':True,'no_hybrid_training_state_used':True,
    'source_weights_and_Adam_from_same_complete_checkpoint':True,'source_moments_sampler_RNG_history_preserved':True,
    'source_Adam_audit_before_LR_change':True,'renewed_Portuguese_grace':False,'initial_source350_required':True,
    'chosen_uniform_unique_clips':17774,'failed_latest_uniform_unique_clips':28274,'explicit_coverage_rewind_clips':10500,
    'failed_branch_history_preserved':True,'new_queue_shuffle_or_optimizer_reset':False,
    'uniform_origin_step':6500,'quota_boundary_step':7621,'lora_boundary_step':7750,'PT_focus_boundary_step':8000,
    'configured_end_step':25109,'remaining_updates':16859,'goal_complete':False,'artifacts':a}
path=HERE/'TSC_feature_loralr_measured_repair_review_v1.json'
with path.open('x',encoding='utf-8') as f:json.dump(review,f,indent=2)
def once(s,a,b):assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
w=(HERE/'train_lora_batched_tsc_uniform_constraint_v2.py').read_text(encoding='utf-8')
w=w.replace('train_lora_batched_tsc_uniform_constraint_v2.py','train_lora_batched_tsc_uniform_feature_loralr_v1.py').replace('run_TSC_uniform_constraint_v2.py','run_TSC_uniform_feature_loralr_v1.py').replace('tsc_uniform_retention_constraint_transition_v1','tsc_uniform_feature_loralr_transition_v1')
w=once(w,'restore_parent_sampler, constraint_declaration)','restore_parent_sampler, lora_lr_declaration)')
w=once(w,'TSC_uniform_feature_retention_constraint_full_state_v1','TSC_uniform_feature_lower_LoRA_full_state_v1')
w=once(w,'"declared": constraint_declaration(context)','"declared": lora_lr_declaration(context)')
w=once(w,'        "retention_constraint_transition":transition_identity(args,context,sampler_transition),','        "retention_constraint_transition":context["parent_identity"]["retention_constraint_transition"],\n        "feature_LoRA_LR_transition":transition_identity(args,context,sampler_transition),')
w=once(w,'        "retention_constraint_transition":config_identity["retention_constraint_transition"],','        "retention_constraint_transition":config_identity["retention_constraint_transition"],\n        "feature_LoRA_LR_transition":config_identity["feature_LoRA_LR_transition"],')
w=once(w,'    p.add_argument("--lora-lr",type=float,default=1e-4)','    p.add_argument("--lora-lr",type=float,default=3e-5)')
w=once(w,'or args.lora_lr!=1e-4 or args.replay_fraction','or args.lora_lr!=3e-5 or args.replay_fraction')
w=once(w,'                  "PT_replay_clips_per_update":1,"old_five_replay_clips_per_update":1,','                  "feature_LoRA_LR_updates":step-source_step,"feature_LoRA_LR_source_step":source_step,\n                  "PT_replay_clips_per_update":1,"old_five_replay_clips_per_update":1,')
wp=HERE/'train_lora_batched_tsc_uniform_feature_loralr_v1.py'
with wp.open('x',encoding='utf-8') as f:f.write(w)
tree=ast.parse(w);run=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run')
calls=[x for x in ast.walk(run) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and ast.unparse(x.func)=='torch.cuda.synchronize'];assert len(calls)==1
c=(HERE/'run_TSC_uniform_constraint_v2.py').read_text(encoding='utf-8').replace('train_lora_batched_tsc_uniform_constraint_v2.py',wp.name).replace('tsc_uniform_retention_constraint_transition_v1','tsc_uniform_feature_loralr_transition_v1')
c=once(c,"WORKER_SHA='f6433bd9cf34a1560f0a0093630659ef365f151e8bad68f6bfc4b9357b7731fe'","WORKER_SHA='"+hashlib.sha256(wp.read_bytes()).hexdigest()+"'")
c=once(c,"TRANSITION_SHA='ccdef7862d6fec68d28265199514ebdaa4772352be47cc4cfec0230e8370bfba'","TRANSITION_SHA='PENDING'")
c=once(c,'EXPECTED_SYNC_LINE=736',f'EXPECTED_SYNC_LINE={calls[0].lineno}')
c=once(c,"float(option(argv,'--lora-lr'))==.0001","float(option(argv,'--lora-lr'))==.00003")
c=once(c,"('--lora-lr',.0001)","('--lora-lr',.00003)")
cp=HERE/'run_TSC_uniform_feature_loralr_v1.py'
with cp.open('x',encoding='utf-8') as f:f.write(c)
compile(w,str(wp),'exec');compile(c,str(cp),'exec')
lp=ROOT/'work/gpu_training_owner.json';lease=read(lp);assert lease['owner']=='/root/benchmark_design' and not lease['running']
lease.update(status='reserved_for_measured_lower_encoder_LR_qualified8250_continuation',updated_at_utc=review['created_at_utc'],
    target_run_id='adaptive_20261012_TR_uniform_feature_LoRA_slow_01',source_frontier=8250,source_failure_counter_preserved=0,
    source_unique_uniform_clips=17774,explicit_coverage_rewind_clips=10500,actual_measured_constraint_review=record(path))
t=lp.with_suffix('.tmp');t.write_text(json.dumps(lease,indent=2),encoding='utf-8');t.replace(lp)
print(json.dumps({'review':record(path),'worker':record(wp),'sync':calls[0].lineno,'controller_helper_pin_pending':str(cp)}))
