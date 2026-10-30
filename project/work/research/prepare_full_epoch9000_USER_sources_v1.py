"""Explicit USER full-pass policy; preserve latest9000 and all failure history."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,ast,subprocess
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def record(p):
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p.resolve()),'sha256':h,'bytes':p.stat().st_size}
def atomic(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8');t.replace(p)
closedpath=HERE/'PT_token5x_actual_CLOSED9000_retention_review_v1.json';closed=read(closedpath);a=dict(closed['artifacts'])
a['chosen_checkpoint']=a['latest_checkpoint'];a['source_best_checkpoint']=a.pop('inherited_qualified_BEST')
for key in tuple(a):
    if key.startswith('DEV9000_'):a['chosen_DEV_'+key[len('DEV9000_'):]]=a.pop(key)
a['source_best_DEV_summary']=record(HERE/'results'/'adaptive_20261012_TR_uniform_featureKD5x_02_step_8250'/'summary.json')
now=datetime.now(timezone.utc).isoformat()
review={'schema':1,'kind':'ROOT_USER_TSC_full_epoch_policy_review_v1','passed':True,'created_at_utc':now,
    'decision_actor':'/root/benchmark_design under explicit USER steering relayed by ROOT and sole-GPU delegation',
    'USER_instruction':'Do not stop full training again until full epoch results. DEV retention failures are monitoring/champion selection only, never training early-stop. Resource/nonfinite/crash protections remain unchanged.',
    'source_run_id':closed['run_id'],'chosen_step':9000,'completed_steps':9000,'chosen_checkpoint_role':'latest',
    'source_actual_owned_exit':closed['actual_owned_exit'],'actual_closed_source_review':record(closedpath),
    'allow_full_epoch_monitor_only_continuation':True,'source_counter_preserved':3,'source_selected_BEST_preserved':8250,
    'source_unqualified_latest_used':True,'source_chosen_is_all_six_count_qualified':False,'no_hybrid_training_state_used':True,
    'source_weights_and_Adam_from_same_complete_checkpoint':True,'source_moments_sampler_RNG_history_preserved':True,
    'renewed_Portuguese_grace':False,'initial_source350_required':True,'explicit_coverage_rewind_clips':0,
    'chosen_uniform_unique_clips':28274,'failed_branch_history_preserved':True,'new_queue_shuffle_or_optimizer_reset':False,
    'declared_change':{'source_portuguese_token_kl_multiplier':5.,'active_portuguese_token_kl_multiplier':1.,
        'PT_token_KL_before':200.,'PT_token_KL_after':40.,'source_retention_stop_policy':'enforced','active_retention_stop_policy':'monitor_only',
        'base_token_KL':40.,'feature_distillation_unchanged':20.,'duration_KL_unchanged':2.,'head_LR_unchanged':1e-5,
        'LoRA_LR_unchanged':1e-4,'TR_per_update':14,'PT_replay_per_update':1,'old_five_replay_per_update':1,
        'strict_all_six_integer_champion_gates_unchanged':True,'memory_nonfinite_crash_protections_unchanged':True,
        'no_plateau_target_or_retention_stop_before_complete_unique_pass':True,'normal_budget_windows_automatically_exact_resume_despite_retention_failures':True},
    'uniform_origin_step':6500,'quota_boundary_step':7621,'lora_boundary_step':7750,'PT_focus_boundary_step':8000,
    'configured_end_step':25109,'remaining_updates':16109,'full_pass_complete_only_actual_unique_clips':253798,
    'goal_complete':False,'artifacts':a}
rp=HERE/'TSC_full_epoch9000_USER_policy_review_v1.json'
with rp.open('x',encoding='utf-8') as f:json.dump(review,f,indent=2)
def once(s,a,b):assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
w=(HERE/'train_lora_batched_tsc_uniform_pttoken_v1.py').read_text(encoding='utf-8')
w=w.replace('train_lora_batched_tsc_uniform_pttoken_v1.py','train_lora_batched_tsc_uniform_full_epoch_v1.py').replace('run_TSC_uniform_pttoken_v1.py','run_TSC_uniform_full_epoch_v1.py').replace('tsc_uniform_pttoken_constraint_transition_v1','tsc_uniform_full_epoch_transition_v1')
w=w.replace('token_constraint_declaration','full_epoch_declaration').replace('TSC_uniform_PT_token_retention_constraint_full_state_v1','TSC_uniform_USER_full_epoch_monitor_only_full_state_v1')
w=once(w,'        "PT_token_retention_constraint":transition_identity(args,context,sampler_transition),','        "PT_token_retention_constraint":context["parent_identity"]["PT_token_retention_constraint"],\n        "full_epoch_policy_transition":transition_identity(args,context,sampler_transition),')
w=once(w,'        "PT_token_retention_constraint":config_identity["PT_token_retention_constraint"],','        "PT_token_retention_constraint":config_identity["PT_token_retention_constraint"],\n        "full_epoch_policy_transition":config_identity["full_epoch_policy_transition"],')
w=once(w,'"target_turkish_wer","portuguese_token_kl_multiplier")','"target_turkish_wer","portuguese_token_kl_multiplier","retention_stop_policy")')
w=once(w,'if (source_step!=8750 or start_step!=8750 or retention_failures!=2 or selected_step!=8250','if (source_step!=9000 or start_step!=9000 or retention_failures!=3 or selected_step!=8250')
w=once(w,'or source_gate["retention_gate_passed"] or not all(source_gate["retention_language_passed"][lang] for lang in ("en","es","it","fr","de"))','or source_gate["retention_gate_passed"] or source_gate["retention_language_passed"]["de"] or source_gate["retention_language_passed"]["pt"]')
w=w.replace('Reviewed coherent latest8750 must carry genuine counter2','USER-reviewed coherent latest9000 must carry genuine counter3')
w=once(w,'        gate["stop_retention_gate_passed"] = stop_gate_pass','        gate["stop_retention_gate_passed"] = stop_gate_pass\n        gate["retention_early_stop_active"] = False\n        gate["retention_stop_policy"] = "monitor_only"')
w=once(w,'            if args.max_retention_failures and retention_failures >= args.max_retention_failures:\n                stop_reason = "consecutive_retention_gate_failures"\n                break','            # USER full-pass policy: integer retention gates select champions\n            # and update the carried failure counter; they never stop this pass.')
w=once(w,'"resource_and_nonfinite_stops_active":True,"old_five_retention_guard_always_active":True,','"resource_and_nonfinite_stops_active":True,"old_five_retention_guard_always_active":False,\n            "retention_stop_policy":"monitor_only","strict_all_six_integer_champion_gates_unchanged":True,')
w=once(w,'    p.add_argument("--portuguese-token-kl-multiplier",type=float,default=5.)','    p.add_argument("--portuguese-token-kl-multiplier",type=float,default=1.)\n    p.add_argument("--retention-stop-policy",choices=("monitor_only",),default="monitor_only")')
w=once(w,'args.portuguese_token_kl_multiplier!=5 or args.schedule','args.portuguese_token_kl_multiplier!=1 or args.retention_stop_policy!="monitor_only" or args.schedule')
w=w.replace('baseKD40/PTtoken200-duration2-feature20','baseKD40/PTtoken40-duration2-feature20/retention-monitor-only')
w=once(w,'"PT_token_constraint_updates":step-source_step,','"PT_token_constraint_updates":step-8750,"full_epoch_continuation_updates":step-source_step,"retention_stop_policy":"monitor_only",')
wp=HERE/'train_lora_batched_tsc_uniform_full_epoch_v1.py'
with wp.open('x',encoding='utf-8') as f:f.write(w)
tree=ast.parse(w);run=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run')
calls=[x for x in ast.walk(run) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and ast.unparse(x.func)=='torch.cuda.synchronize'];assert len(calls)==1
c=(HERE/'run_TSC_uniform_pttoken_v1.py').read_text(encoding='utf-8').replace('train_lora_batched_tsc_uniform_pttoken_v1.py',wp.name).replace('tsc_uniform_pttoken_constraint_transition_v1','tsc_uniform_full_epoch_transition_v1')
c=once(c,"WORKER_SHA='41c15b88156a449c81ab5c8f6049812d77721bf410b6cd3d30d29978d3e479a0'","WORKER_SHA='"+hashlib.sha256(wp.read_bytes()).hexdigest()+"'")
c=once(c,"TRANSITION_SHA='e5afc8f1a639893e8c5460e724fc3557e9d21765594286ea89b8337dbd64acfb'","TRANSITION_SHA='PENDING'")
c=once(c,'EXPECTED_SYNC_LINE=743',f'EXPECTED_SYNC_LINE={calls[0].lineno}')
c=once(c,"float(option(argv,'--portuguese-token-kl-multiplier'))==5","float(option(argv,'--portuguese-token-kl-multiplier'))==1 and option(argv,'--retention-stop-policy')=='monitor_only'")
c=once(c,"values['portuguese_token_kl_multiplier']=float(option(argv,'--portuguese-token-kl-multiplier'))","values['portuguese_token_kl_multiplier']=float(option(argv,'--portuguese-token-kl-multiplier'))\n    values['retention_stop_policy']=option(argv,'--retention-stop-policy')")
c=once(c,"('--portuguese-token-kl-multiplier',5.)","('--portuguese-token-kl-multiplier',1.),('--retention-stop-policy','monitor_only')")
c=once(c,"'Portuguese_token_kl':200.","'Portuguese_token_kl':40.")
c=once(c,"'replay_kl':(200+40*5/3)/16","'replay_kl':(40+40*5/3)/16")
c=once(c,"and selection['consecutive_retention_failures']<2,","and type(selection['consecutive_retention_failures']) is int,")
cp=HERE/'run_TSC_uniform_full_epoch_v1.py'
with cp.open('x',encoding='utf-8') as f:f.write(c)
compile(w,str(wp),'exec');compile(c,str(cp),'exec')
lp=ROOT/'work/gpu_training_owner.json';lease=read(lp);assert lease['owner']=='/root/benchmark_design' and not lease['running']
lease.update(status='reserved_for_USER_full_epoch9000_monitor_only_continuation',updated_at_utc=now,target_run_id='adaptive_20261012_TR_uniform_full_epoch_01',
    source_frontier=9000,source_failure_counter_preserved=3,source_unique_uniform_clips=28274,explicit_coverage_rewind_clips=0,actual_measured_constraint_review=record(rp),
    USER_full_epoch_priority=True,retention_early_stop_active=False)
atomic(lp,lease)
pp=HERE/'experiment_plan.json';plan=read(pp);plan.update(updated_at_utc=now,objective_complete=False,
    status_authority='NEW USER priority: complete actual253798-unique full pass from coherent latest9000/28274unique, zero rewind. Retention failures are monitoring and strict all-six champion selection only, never training early-stop. PTtoken200 to40 rollback; other science and resource protections unchanged. Sole GPU reserved pending minimal real admission.')
item=next(x for x in plan['ordered_experiments'] if x['id']=='04');item.update(status='USER_full_epoch9000_continuation_preparing',next_run_id=lease['target_run_id'],
    next_declared_repair=review['declared_change'],retention_early_stop_active=False,champion_integer_gates_unchanged=True,
    explicit_failed_branch_TR_clip_rewind=0,current_failure_counter=3,current_all6_selected_BEST_step=8250)
item['pending']=['Minimal archivedlatest9000 admission and actual sole launch','Automatic exact budget windows through full253798 unique pass','Complete end DEV and strict all-six champion selection']
plan['automatic_execution_policy']['PT_focused_replay']='USER override: retention never stops the remaining full pass; all six integer gates still select champions and all failure history/counters persist. Coherent latest9000/PTtoken rollback200 to40/no rewind. Resource/nonfinite protections unchanged; complete budget closures autoexact despite retention failures.'
atomic(pp,plan)
note='NEW USER priority overrides retention-stop policy: finish the actual full253798-clip uniform pass from coherent latest9000/full207 Adam/RNG/sampler/history/28274unique, no rewind. Revert ineffective PTtoken200 to40; feature20/head1e-5/LoRA1e-4/14TR1PT1old-five/native resources unchanged. Retention failures update real counters and exclude champions under unchanged integer gates, but NEVER stop training before full pass. Preserve genuine counter3/BEST8250/failed branches; no grace/reset. All budget windows must automatically resume exact complete state despite failed retention. LowerLoRA8250 draft was not launched and is deferred. Minimal source admission/actual sole GPU launch now.'
for task in ('ASR-15','ASR-15.1'):
    subprocess.run([str(HERE/'tools/backlog.exe'),'task','edit',task,'--status','In Progress','--append-notes',note],cwd=HERE,check=True,capture_output=True,encoding='utf-8')
subprocess.run([str(ROOT/'work/training-env/Scripts/python.exe'),str(HERE/'refresh_backlog.py')],cwd=ROOT,check=True,capture_output=True,encoding='utf-8')
print(json.dumps({'review':record(rp),'worker':record(wp),'sync':calls[0].lineno,'controller_pending':str(cp),'plan_Backlog_lease_updated':True}))
