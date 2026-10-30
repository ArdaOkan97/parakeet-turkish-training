"""Record frozen source-only handoff; never admit source state or launch."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'outputs/parakeet_turkish'
names = ['tsc_bpe_row_schedule_transition_v1.py', 'train_lora_batched_tsc_bpe_row_schedule_v1.py',
         'run_TSC_bpe_row_schedule_v1.py', 'bpe_newrows_scheduled_acquisition_v1.py',
         'record_TSC_BPE_row_schedule_source_v1.py']
def record(name):
    path=HERE/name
    return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
proofs=['TSC_BPE_row_schedule_controller_dispatch_CPU_verification_v1.json',
        'TSC_BPE_row_schedule_boundary_CPU_verification_v1.json',
        'BPE_scheduled_acquisition_stdlib_verification_v1.json']
proof_records=[]
for name in proofs:
    if (HERE/name).exists():
        proof=json.loads((HERE/name).read_text(encoding='utf-8'))
        assert proof['passed'] is True
        proof_records.append(record(name))
worker=ast.parse((HERE/names[1]).read_text(encoding='utf-8'))
run=next(n for n in worker.body if isinstance(n,ast.FunctionDef) and n.name=='run')
sync=[n.lineno for n in ast.walk(run) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.cuda.synchronize']
assert len(sync)==1
prefix='work/training-env/Scripts/python.exe outputs/parakeet_turkish/'
cli=prefix+'run_TSC_bpe_row_schedule_v1.py --source-loop outputs/parakeet_turkish/runs/adaptive_20261011_hf_TSC_BPE512_acquire_01_loop_01 --source-receipt NEW_ROOT_SCHEDULE_RECEIPT --run-id FRESH_RUN --output-dir outputs/parakeet_turkish/runs/FRESH_LOOP --peak-row-lr 0.0003 --acquisition-max-updates 1000 --window-hours 3'
report={'kind':'inactive_TSC_BPE512_scheduled_row_frozen_source_handoff_v1','status':'source_pinned_pending_ROOT_actual_gates',
 'source_records':{name:record(name) for name in names},'focused_CPU_proofs':proof_records,'worker_optimizer_end_sync_line':sync[0],
 'phase':'HF_TSC_BPE512_new_rows_scheduled_acquisition_v1','source_anchor':{'original_step_S':6500,'phase1_chosen_step':7000,'source_row_Adam_clock_K':500,
 'actual_new_source_admitted_by_agent':False,'bad_phase2_or_fixed_control_latest_used':False},
 'schedule':{'kind':'linear_warmup_cosine_floor_v1','warmup_updates':50,'horizon_updates':1000,'min_lr_ratio':0.1,
 'allowed_peaks':[0.001,0.0003],'clock':'local_N','source_Adam_K_retained':True,
 'initial_group_LR_before_first_update':0.0001,'first_applied_LR':'peak/50','last_applied_LR':'0.1*peak'},
 'state':{'full207_weights_and_three_Adam_saved':True,'frozen204_and_old_six_blank_duration_rows':True,
 'localN_distinct_from_AdamKplusN':True,'applied_LR_and_next_update_saved':True,
 'external_exact_resume_supported':False,'unfreeze_supported':False},
 'guards':{'cap_bytes':8000000000,'end_cache_bytes':4000000000,'native_inactive_threshold_bytes':750000000,
 'native_max_joint_bytes':1850000000,'arithmetic_bound_is_actual_VRAM_proof':False},
 'preflight_cli':cli+' --validate-only --preflight-output NEW_REPORT_OUTSIDE_FRESH_LOOP',
 'future_ROOT_smoke_cli':cli+' --smoke-updates 1 --gpu-window-authorized',
 'future_ROOT_main_cli':'Use another fresh run/loop and matching new ROOT receipt; '+cli+' --gpu-window-authorized',
 'actual_initial300_proof':'runs/<RUN>/TSC_BPE_row_schedule_initial_source_equivalence.json',
 'actual_smoke_proof':'runs/<RUN>/acquisition_actual_one_update_smoke.json',
 'pending_actual_gates':['ROOT immutable source receipt/preflight','actual source300 transcript/token/duration/blank paths before updates',
 'one real scheduled native-TDT update/full207-three-Adam save-load/protected bytes/applied LR','separate fresh1000 DEV250 screen'],
 'Torch_or_GPU_executed':False,'actual_checkpoint_or_model_read':False,'current_activated_files_modified':False,'activation_ready_claimed':False}
out=HERE/'TSC_BPE_row_schedule_source_handoff_v1.json'
with out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'handoff':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'sync':sync[0]}))
