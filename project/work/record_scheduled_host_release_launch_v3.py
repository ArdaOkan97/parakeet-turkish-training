from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'outputs/parakeet_turkish'
p=HERE/'experiment_plan.json'
plan=json.loads(p.read_text(encoding='utf-8'))
item=next(x for x in plan['ordered_experiments'] if x['id']=='03')
item.update(status='running_smoke',run_id='adaptive_20261011_hf_TSC_BPE512_row_schedule_memory_03_main',
 smoke_run_id='adaptive_20261011_hf_TSC_BPE512_row_schedule_memory_03_smoke',
 queue_executor='run_TSC_row_schedule_memory_queue_v3.py',
 actual_memory02_review='TSC_row_schedule_memory02_resource_DEV_host_audit_v1.json',
 actual_memory02_final_TR_WER=322/825,actual_memory02_retention_passed=True,
 actual_memory02_new_BPE_emissions=0,
 memory_policy='Same cap8/endcache3/native4/750MB/90x3/95/2GB; compact shared immutable row keys/metadata/token integer objects while retaining exact all253798 row values/order/labels, then release startup-only aliases. Actual host relief recorded by fresh smoke.',
 actual_ROOT_queue_session=42301,actual_ROOT_launch_tool_chunk='69e3cc',
 current_family_smoke_passed=False,actual_GPU_trial_completed=False,
 full1000_claimed=False)
p.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'plan_updated':True,'actual_queue_session':42301,'GPU_smoke_awaiting_actual_completion':True}))
