"""Minimal NEW LR-only source derivatives; no model/runtime execution."""
from pathlib import Path
import ast,hashlib
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def change(text,old,new,count=1):
    if text.count(old)!=count:raise ValueError((old[:120],text.count(old),count))
    return text.replace(old,new)
assert sha(HERE/'train_lora_batched_tsc_uniform_majority_v1.py')=='b076b4a637f331736db0462da14fc5e5ddfc3059cb57b82dc95887f78e835680'
assert sha(HERE/'run_TSC_uniform_majority_v1.py')=='6361896ba05893c07baf26644895ee68539a9471c539b46f1cdd74dd0db91105'
w=(HERE/'train_lora_batched_tsc_uniform_majority_v1.py').read_text()
w=w.replace('tsc_uniform_majority_transition_v1','tsc_uniform_majority_loralr_transition_v1')
w=w.replace('run_TSC_uniform_majority_v1.py','run_TSC_uniform_majority_loralr_v1.py')
w=change(w,'proof_path, majority_declaration)','proof_path, loralr_declaration, restore_parent_sampler)')
w=change(w,'"TSC_uniform_majority_207_state_transition_v1"','"TSC_uniform_majority_LoRA_LR_state_transition_v1"')
w=change(w,'majority_declaration(context["source_step"],context["prior_uniform_draws"])','loralr_declaration(context)')
w=change(w,'sampler_transition=sampler.migrate_closed_state(source["sampler"],context["parent_identity"]["domain_sampling_identity"])',
    'sampler_transition=restore_parent_sampler(sampler,source,context)')
w=change(w,'source_step=context["source_step"];uniform_origin=context["uniform_origin_step"]',
    'source_step=context["source_step"];uniform_origin=context["uniform_origin_step"];quota_boundary=context["quota_boundary_step"]')
w=change(w,'sampler.remaining_update_budget!=context["remaining_update_budget"]',
    'configured_steps-source_step!=context["remaining_update_budget"]')
w=change(w,'sampler.quota_transition["prior_uniform_draws"]!=context["prior_uniform_draws"]',
    'sampler.turkish_streams["uniform"].draws!=context["prior_uniform_draws"] or sampler.transition_source_step!=quota_boundary')
w=change(w,'"majority_transition":transition_identity(args,context,sampler_transition)',
    '"encoder_lr_transition":transition_identity(args,context,sampler_transition)')
w=change(w,'optimizer.load_state_dict(source["optimizer"])','optimizer.load_state_dict(source["optimizer"])\n        optimizer.param_groups[1]["lr"]=args.lora_lr')
w=change(w,'"remaining_uniform_pass_updates":sampler.remaining_update_budget',
    '"remaining_uniform_pass_updates":context["remaining_update_budget"],"majority_source_step":quota_boundary')
w=change(w,'"phase":"HF_TSC_8199_uniform_Turkish_majority_with_Portuguese_v1"',
    '"phase":"HF_TSC_8199_uniform_Turkish_majority_lower_LoRA_v1"')
w=change(w,'"transition_source_step":source_step','"transition_source_step":quota_boundary')
w=change(w,'"prior_uniform_draws":context["prior_uniform_draws"]',
    '"prior_uniform_draws":sampler.quota_transition["prior_uniform_draws"]')
w=change(w,'"retention_density_normalization":majority_declaration(source_step,context["prior_uniform_draws"])',
    '"retention_density_normalization":context["config"]["retention_density_normalization"],"encoder_learning_rate_transition":loralr_declaration(context)')
w=change(w,'"method":"Full207 state-preserving existing uniform-pass14TR/2replay allocation; retention-density40/2/4"',
    '"method":"Full207 state-preserving same14/2 uniform pass; encoder LoRA LR3e-4 to1e-4 only"')
w=change(w,'"majority_updates":step-source_step,"majority_source_step":source_step',
    '"majority_updates":step-quota_boundary,"majority_source_step":quota_boundary,\n                  "lora_lr_fork_updates":step-source_step,"lora_lr_fork_source_step":source_step')
w=change(w,'"majority_updates":completed-source_step,',
    '"majority_updates":completed-quota_boundary,"majority_source_step":quota_boundary,\n                 "lora_lr_fork_updates":completed-source_step,')
w=change(w,'TSC_uniform_majority_fresh_preload_v1','TSC_uniform_majority_loralr_fresh_preload_v1')
w=change(w,'p.add_argument("--lora-lr",type=float,default=3e-4)','p.add_argument("--lora-lr",type=float,default=1e-4)')
w=change(w,'args.lora_lr!=3e-4','args.lora_lr!=1e-4')
w=change(w,'INACTIVE declared14TR/2replay continuation of a CLOSED uniform frontier.',
    'INACTIVE declared LoRA LR3e-4 to1e-4 full-state fork of a CLOSED majority frontier.')
ast.parse(w);dest=HERE/'train_lora_batched_tsc_uniform_majority_loralr_v1.py'
if dest.exists():raise ValueError('NEW worker exists')
dest.write_text(w,encoding='utf-8',newline='\n');print(sha(dest))
c=(HERE/'run_TSC_uniform_majority_v1.py').read_text()
c=c.replace('train_lora_batched_tsc_uniform_majority_v1.py','train_lora_batched_tsc_uniform_majority_loralr_v1.py')
c=c.replace('tsc_uniform_majority_transition_v1','tsc_uniform_majority_loralr_transition_v1')
c=c.replace('guarded_TSC_8199_uniform_majority_fullpass_controller_v1','guarded_TSC_8199_uniform_majority_loralr_fullpass_controller_v1')
c=c.replace('TSC_uniform_majority_','TSC_uniform_majority_loralr_')
# Sampler name is a frozen implementation dependency, not an LR derivative.
c=c.replace('tsc_uniform_majority_loralr_sampler_v1.py','tsc_uniform_majority_sampler_v1.py')
c=change(c,"WORKER_SHA='b076b4a637f331736db0462da14fc5e5ddfc3059cb57b82dc95887f78e835680'","WORKER_SHA='PENDING_WORKER'")
c=change(c,"TRANSITION_SHA='173664a651ae1529f2981b7150acae402410ee8708ad96c98735e0d30fb03738'","TRANSITION_SHA='PENDING_HELPER'")
c=change(c,'EXPECTED_SYNC_LINE=679','EXPECTED_SYNC_LINE=0')
c=change(c,"float(option(argv,'--lora-lr'))==.0003","float(option(argv,'--lora-lr'))==.0001")
c=change(c,"('--sampling-mode','uniform'),('--steps',context['configured_end_step']),('--replay-fraction',.125)",
    "('--sampling-mode','uniform'),('--steps',context['configured_end_step']),('--lora-lr',.0001),('--replay-fraction',.125)")
c=change(c,"'uniform_origin_step':context['uniform_origin_step'],'prior_uniform_draws':context['prior_uniform_draws']",
    "'uniform_origin_step':context['uniform_origin_step'],'quota_boundary_step':context['quota_boundary_step'],'prior_uniform_draws':context['prior_uniform_draws']")
c=change(c,"'window_hours':args.window_hours,\n        'worker_arguments'",
    "'window_hours':args.window_hours,\n        'same_policy_previous':{'peak_observed_owned_private_bytes':source_status['peak_observed_owned_private_bytes'],'finished_at_utc':source_status['finished_at_utc']},\n        'worker_arguments'")
c=change(c,"proof['kind']=='TSC_uniform_majority_loralr_before_update_source_equivalence_v1'",
    "proof['kind']=='TSC_uniform_majority_loralr_before_update_source_equivalence_v1'")
c=change(c,"selection['majority_updates']==selection['completed_steps']-plan['source_selected_step']",
    "selection['majority_updates']==selection['completed_steps']-plan['quota_boundary_step']\n        and selection['lora_lr_fork_updates']==selection['completed_steps']-plan['source_selected_step']")
c=change(c,"'actual_majority_updates':max(0,state.get('published_step',state['resume_step'])-state['source_selected_step'])",
    "'actual_majority_updates':max(0,state.get('published_step',state['resume_step'])-state['quota_boundary_step']),\n        'actual_lora_lr_fork_updates':max(0,state.get('published_step',state['resume_step'])-state['source_selected_step'])")
c=change(c,'initial_admission=current_memory_admission(first_measurement=args.measure_new_allocation_policy)',
    'initial_admission=current_memory_admission(previous=plan["same_policy_previous"])')
c=change(c,"    p.add_argument('--measure-new-allocation-policy',action='store_true',help='ROOT-reviewed FIRST lazy-runtime measurement only: reserve8GB commit+physical; retain every live guard')\n",'')
c=change(c,"'declared_optimization_change':'CLOSED207 frontierH retained; same TR pass14/2 allocation and density40/2/4; PT grace and all6 gates retained'",
    "'declared_optimization_change':'CLOSED207 majority frontier retained; only live encoder LoRA LR3e-4 to1e-4 after original moment audit/load; no reset'")
ast.parse(c);dest=HERE/'run_TSC_uniform_majority_loralr_v1.py'
if dest.exists():raise ValueError('NEW controller exists')
dest.write_text(c,encoding='utf-8',newline='\n');print(sha(dest))
