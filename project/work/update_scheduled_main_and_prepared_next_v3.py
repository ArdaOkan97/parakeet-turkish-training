import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'outputs/parakeet_turkish'
p=HERE/'experiment_plan.json'
plan=json.loads(p.read_text(encoding='utf-8'))
state=json.loads((HERE/'experiment_queue_state.json').read_text(encoding='utf-8'))
assert state['queue_id']=='20261012_row_schedule_memory_03' and state['current_stage']=='schedule_main_train' and state['status']=='running'
smoke=json.loads((ROOT/'work/experiment_queues/20261012_row_schedule_memory_03/schedule_smoke_review.json').read_text(encoding='utf-8'))
assert smoke['passed'] and smoke['updates_completed']==1 and smoke['last_retention_passed']
i=next(x for x in plan['ordered_experiments'] if x['id']=='03')
i.update(status='running',actual_smoke_passed=True,current_family_smoke_passed=True,
 actual_smoke_review='work/experiment_queues/20261012_row_schedule_memory_03/schedule_smoke_review.json',
 measured_smoke_host_private_release_bytes=298307584,actual_main_controller_pid=state['owned_child_pid'],
 actual_GPU_trial_completed=False)
i=next(x for x in plan['ordered_experiments'] if x['id']=='05')
i.update(status='prepared_waiting_for_preceding_trial',code_ready=True,
 prepared_worker='train_lora_batched_tsc_bpe_shared_head_memory_v2.py',
 prepared_controller='run_TSC_bpe_shared_head_memory_v2.py',
 prepared_boundary='tsc_bpe_shared_head_memory_v2.py',
 real_source300_and_native_smoke_pending=True,actual_GPU_trial_completed=False)
i=next(x for x in plan['ordered_experiments'] if x['id']=='06')
i.update(status='audit_complete_quality_control_untested',
 source_and_saved_scalar_audit='BPE_target_length_loss_scaling_saved_control_audit_v1.json',
 audit_finding='NLL/candidateU once and effective mean16 are correct; shorterU nominally strengthens TR weighting. Scalar KD10 is0.84-1.98% of control loss; no gradient dominance conclusion. A baselineU divisor control is conditional and untested.',
 actual_GPU_trial_completed=False)
p.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'plan_updated':True,'main_running':state['current_run_id'],'prepared_shared_heads_are_untested':True}))
