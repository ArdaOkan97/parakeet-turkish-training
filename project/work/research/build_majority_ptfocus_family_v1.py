"""NEW PT-only replay-policy derivatives; no ML/runtime execution."""
from pathlib import Path
import ast,hashlib
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def change(t,a,b,n=1):
    if t.count(a)!=n:raise ValueError((a[:100],t.count(a),n))
    return t.replace(a,b)
assert sha(HERE/'train_lora_batched_tsc_uniform_majority_loralr_v1.py')=='1b2089821aac2923a0e516253610961b4dc5a25f35ced77a49947e7908151114'
assert sha(HERE/'run_TSC_uniform_majority_loralr_v1.py')=='0684422301c926e0037bfa0c1423230ce39dbc95cac538521882db9a68f7746e'
w=(HERE/'train_lora_batched_tsc_uniform_majority_loralr_v1.py').read_text(encoding='utf-8')
w=w.replace('tsc_uniform_majority_loralr_transition_v1','tsc_uniform_majority_ptfocus_transition_v1')
w=w.replace('run_TSC_uniform_majority_loralr_v1.py','run_TSC_uniform_majority_ptfocus_v1.py')
w=w.replace('loralr_declaration','ptfocus_declaration')
w=w.replace('TSC_uniform_majority_LoRA_LR_state_transition_v1','TSC_uniform_majority_PT_replay_state_transition_v1')
w=change(w,'from tsc_uniform_majority_sampler_v1 import TSCUniformMajoritySampler',
    'from tsc_uniform_majority_ptfocus_sampler_v1 import TSCUniformMajorityPTFocusSampler')
w=change(w,'sampler=TSCUniformMajoritySampler(', 'sampler=TSCUniformMajorityPTFocusSampler(')
w=change(w,'"tsc_uniform_majority_sampler_v1.py"','"tsc_uniform_majority_ptfocus_sampler_v1.py"',2)
w=change(w,'"encoder_lr_transition":transition_identity(args,context,sampler_transition)',
    '"PT_replay_transition":transition_identity(args,context,sampler_transition)')
w=change(w,'"HF_TSC_8199_uniform_Turkish_majority_lower_LoRA_v1"','"HF_TSC_8199_uniform_Turkish_majority_PT_focused_v1"')
w=change(w,'"encoder_learning_rate_transition":ptfocus_declaration(context)',
    '"PT_replay_policy_transition":ptfocus_declaration(context)')
w=change(w,'"method":"Full207 state-preserving same14/2 uniform pass; encoder LoRA LR3e-4 to1e-4 only"',
    '"method":"Full207 same14/2 uniform pass; onePT+oneoldfive replay with declared language-weighted preservation"')
w=change(w,'"lora_lr_fork_updates":step-source_step,"lora_lr_fork_source_step":source_step',
    '"PT_focus_updates":step-source_step,"PT_focus_source_step":source_step,\n                  "lora_lr_fork_updates":step-context_lora_origin,"lora_lr_fork_source_step":context_lora_origin')
w=change(w,'source_step=context["source_step"];uniform_origin=context["uniform_origin_step"];quota_boundary=context["quota_boundary_step"]',
    'source_step=context["source_step"];uniform_origin=context["uniform_origin_step"];quota_boundary=context["quota_boundary_step"];context_lora_origin=context["lora_boundary_step"]')
w=change(w,'"lora_lr_fork_updates":completed-source_step,',
    '"lora_lr_fork_updates":completed-context_lora_origin,"PT_focus_updates":completed-source_step,')
w=change(w,'"replay_kl_coefficient":args.replay_kl,"duration_kl_coefficient":args.duration_kl,"feature_distillation_coefficient":args.feature_distillation,',
    '"replay_kl_coefficient":args.replay_kl,"duration_kl_coefficient":args.duration_kl,"feature_distillation_coefficient":args.feature_distillation,\n                  "preservation_coefficients_by_language":{"pt":[args.replay_kl,args.duration_kl,args.feature_distillation],"old_five":[args.replay_kl*5/3,args.duration_kl*5/3,args.feature_distillation*5/3]},\n                  "PT_replay_clips_per_update":1,"old_five_replay_clips_per_update":1,')
w=w.replace('TSC_uniform_majority_loralr_fresh_preload_v1','TSC_uniform_majority_ptfocus_fresh_preload_v1')
addition='''
def preservation_row_scales(rows):
    """Exact per-clip density declaration; TR preservation remains literal0."""
    scales=[]
    for row in rows:
        language=row["language"]
        if language=="tr":scales.append(0.)
        elif language=="pt":scales.append(1.)
        elif language in ("en","es","it","fr","de"):scales.append(5/3)
        else:raise ValueError("Unapproved preservation language: "+str(language))
    return scales


def pt_focused_batch_objective(model,teacher,batch,*,replay_kl,duration_kl,feature_distillation,**kwargs):
    # Frozen native forward, per-targetU TDT and full-vocab teacher losses run
    # exactly once. Reweight their differentiable per-clip outputs, mean16 via
    # the unchanged physical-batch mean/accumulation2 convention.
    from dataclasses import replace
    result=batch_objective(model,teacher,batch,replay_kl=0.,duration_kl=0.,feature_distillation=0.,**kwargs)
    scale=torch.tensor(preservation_row_scales(batch.rows),dtype=torch.float32,device=result.per_clip["tdt"].device)
    total=(result.per_clip["tdt"]+replay_kl*scale*result.per_clip["token_kl"]
        +duration_kl*scale*result.per_clip["duration_kl"]+feature_distillation*scale*result.per_clip["feature_distillation"])
    return replace(result,total=total.mean(),per_clip={**result.per_clip,"total":total})

'''
w=change(w,'\ndef run(args):', '\n'+addition+'\ndef run(args):')
w=change(w,'result = batch_objective(model, teacher, batch, old_size=old_size, replay_kl=args.replay_kl,',
    'result = pt_focused_batch_objective(model, teacher, batch, old_size=old_size, replay_kl=args.replay_kl,')
w=change(w,'INACTIVE declared LoRA LR3e-4 to1e-4 full-state fork of a CLOSED majority frontier.',
    'INACTIVE PT-focused replay full-state fork of a CLOSED lower-LoRA majority frontier.')
w=change(w,'Quota, coefficients and pass horizon carry; only live encoder LoRA LR changes.',
    'Quota/LRs/pass carry; onePT+oneoldfive replay and per-language preservation are declared.')
ast.parse(w);p=HERE/'train_lora_batched_tsc_uniform_majority_ptfocus_v1.py'
if p.exists():raise ValueError('NEW worker exists')
p.write_text(w,encoding='utf-8',newline='\n');print(sha(p))
c=(HERE/'run_TSC_uniform_majority_loralr_v1.py').read_text(encoding='utf-8')
c=c.replace('train_lora_batched_tsc_uniform_majority_loralr_v1.py','train_lora_batched_tsc_uniform_majority_ptfocus_v1.py')
c=c.replace('tsc_uniform_majority_loralr_transition_v1','tsc_uniform_majority_ptfocus_transition_v1')
c=c.replace('guarded_TSC_8199_uniform_majority_loralr_fullpass_controller_v1','guarded_TSC_8199_uniform_majority_ptfocus_fullpass_controller_v1')
c=c.replace('TSC_uniform_majority_loralr_','TSC_uniform_majority_ptfocus_')
# Actual source350 comparison reuses the base generic equality producer.
c=c.replace('TSC_uniform_majority_ptfocus_before_update_source_equivalence_v1','TSC_uniform_majority_before_update_source_equivalence_v1')
c=change(c,"'tsc_uniform_majority_sampler_v1.py':SAMPLER_SHA", "'tsc_uniform_majority_ptfocus_sampler_v1.py':SAMPLER_SHA")
c=change(c,"WORKER_SHA='1b2089821aac2923a0e516253610961b4dc5a25f35ced77a49947e7908151114'","WORKER_SHA='PENDING_WORKER'")
c=change(c,"TRANSITION_SHA='e24eb0dda449fd42e7204c8e354020dd54caed208d4eb4c5784b44b3ef2dfea9'","TRANSITION_SHA='PENDING_HELPER'")
c=change(c,"SAMPLER_SHA='19adad00ed5d2c00ad3c07e9d31c221c8d171e1202b1ea83348176e276d61976'","SAMPLER_SHA='PENDING_SAMPLER'")
c=change(c,'EXPECTED_SYNC_LINE=680','EXPECTED_SYNC_LINE=0')
c=change(c,"selection['lora_lr_fork_updates']==selection['completed_steps']-plan['source_selected_step']", "selection['PT_focus_updates']==selection['completed_steps']-plan['source_selected_step']")
c=change(c,"'actual_lora_lr_fork_updates':max(0,state.get('published_step',state['resume_step'])-state['source_selected_step'])", "'actual_PT_focus_updates':max(0,state.get('published_step',state['resume_step'])-state['source_selected_step'])")
c=change(c,"'declared_optimization_change':'CLOSED207 majority frontier retained; only live encoder LoRA LR3e-4 to1e-4 after original moment audit/load; no reset'",
    "'declared_optimization_change':'CLOSED207 lowerLR frontier retained; onePT+oneoldfive replay withPT40/2/4 andoldfive5/3 compensation only; no reset'")
ast.parse(c);p=HERE/'run_TSC_uniform_majority_ptfocus_v1.py'
if p.exists():raise ValueError('NEW controller exists')
p.write_text(c,encoding='utf-8',newline='\n');print(sha(p))
