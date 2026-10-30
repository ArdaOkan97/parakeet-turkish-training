"""Small source-only memory-policy handoff; no actual state/ML/GPU action."""
from pathlib import Path
import ast
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'outputs/parakeet_turkish'
old_names=['tsc_bpe_row_schedule_transition_v1.py','train_lora_batched_tsc_bpe_row_schedule_v1.py','run_TSC_bpe_row_schedule_v1.py']
new_names=['tsc_bpe_row_schedule_memory_v2.py','train_lora_batched_tsc_bpe_row_schedule_memory_v2.py','run_TSC_bpe_row_schedule_memory_v2.py']
def record(name):
    p=HERE/name
    return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def normalize(text):
    for old,new in sorted(zip(old_names,new_names), key=lambda pair:len(pair[1]),reverse=True):text=text.replace(new,old)
    return text
before=ast.parse((HERE/old_names[1]).read_text(encoding='utf-8'))
after=ast.parse(normalize((HERE/new_names[1]).read_text(encoding='utf-8')))
bf={n.name:n for n in before.body if isinstance(n,ast.FunctionDef)}
af={n.name:n for n in after.body if isinstance(n,ast.FunctionDef)}
assert set(bf)==set(af)
for name in bf:
    if name!='native_loss_identity':assert ast.dump(bf[name],include_attributes=False)==ast.dump(af[name],include_attributes=False),name
assert ast.dump(ast.parse((HERE/old_names[0]).read_text(encoding='utf-8')),include_attributes=False)==ast.dump(ast.parse(normalize((HERE/new_names[0]).read_text(encoding='utf-8'))),include_attributes=False)
proof='TSC_BPE_row_schedule_memory_controller_dispatch_CPU_verification_v2.json'
assert json.loads((HERE/proof).read_text(encoding='utf-8'))['passed'] is True
cli='work/training-env/Scripts/python.exe outputs/parakeet_turkish/run_TSC_bpe_row_schedule_memory_v2.py --source-loop outputs/parakeet_turkish/runs/adaptive_20261011_hf_TSC_BPE512_acquire_01_loop_01 --source-receipt NEW_ROOT_SCHEDULE_RECEIPT --run-id FRESH_RUN --output-dir outputs/parakeet_turkish/runs/FRESH_LOOP --peak-row-lr 0.0003 --acquisition-max-updates 1000 --window-hours 3'
value={'kind':'inactive_scheduled_row_end_cache3GB_source_handoff_v2','status':'source_pinned_pending_ROOT_actual_gates',
 'new_sources':{name:record(name) for name in new_names},'frozen_predecessor_sources':{name:record(name) for name in old_names},
 'focused_dispatch_proof':record(proof),'all_worker_functions_except_native_identity_AST_exact_after_route_normalization':True,
 'whole_helper_AST_exact_after_route_normalization':True,'worker_run_sync_line_unchanged':471,
 'resource_policy_only':{'end_completed_update_cache_threshold_old_bytes':4000000000,'new_bytes':3000000000,
 'allocator_cap_bytes':8000000000,'native_inactive_threshold_bytes':750000000,'native_backend':'native4',
 'overall_memory_recurrence_prevention_claimed':False},
 'schedule_utility':record('bpe_newrows_scheduled_acquisition_v1.py'),'original_creator_still_valid':record('record_TSC_BPE_row_schedule_source_v1.py'),
 'receipt_declaration_phase_and_proof_schema_unchanged':True,
 'preflight_cli':cli+' --validate-only --preflight-output FRESH_REPORT_OUTSIDE_LOOP',
 'smoke_cli':cli+' --smoke-updates 1 --gpu-window-authorized',
 'main_cli':'Separate fresh run/loop/receipt; '+cli+' --gpu-window-authorized',
 'source300_and_actual_native_fit_measured_by_agent':False,'Torch_imported_or_GPU_launched':False,'activation_ready_claimed':False}
output=HERE/'TSC_BPE_row_schedule_memory_source_handoff_v2.json'
with output.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps({'passed':True,'handoff':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))
