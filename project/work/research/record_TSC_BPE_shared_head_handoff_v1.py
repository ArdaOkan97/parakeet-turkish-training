"""NEW source-only shared-head handoff; no state/model admission or launch."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'outputs/parakeet_turkish'
def record(name):
    p=HERE/name
    return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
files=['tsc_bpe_shared_head_transition_v1.py','train_lora_batched_tsc_bpe_shared_head_v1.py',
 'run_TSC_bpe_shared_head_v1.py','bpe_newrows_shared_head_adaptation_v1.py','record_TSC_BPE_shared_head_source_v1.py']
proofs=['TSC_BPE_shared_head_controller_dispatch_CPU_verification_v1.json','TSC_BPE_shared_head_boundary_CPU_verification_v1.json']
for name in proofs:assert json.loads((HERE/name).read_text(encoding='utf-8'))['passed'] is True
cli='work/training-env/Scripts/python.exe outputs/parakeet_turkish/run_TSC_bpe_shared_head_v1.py --source-loop outputs/parakeet_turkish/runs/adaptive_20261011_hf_TSC_BPE512_acquire_01_loop_01 --source-receipt NEW_ROOT_SHARED_HEAD_RECEIPT --run-id FRESH_RUN --output-dir outputs/parakeet_turkish/runs/FRESH_LOOP --row-lr 0.0003 --shared-lr 0.00001 --adaptation-max-updates 1000 --window-hours 3'
value={'kind':'inactive_TSC_BPE512_shared_heads_frozen_source_handoff_v1','status':'source_pinned_pending_ROOT_actual_gates',
 'source_records':{name:record(name) for name in files},'focused_CPU_proofs':[record(name) for name in proofs],
 'phase':'HF_TSC_BPE512_new_rows_plus_shared_heads_v1','worker_optimizer_end_sync_line':501,
 'source_anchor':{'original_step_S':6500,'actual_chosen_phase1_step':7000,'row_Adam_K':500,
 'bad_LoRA_phase2_latest_or_control_latest_used':False,'actual_source_admitted_by_agent':False},
 'changes':{'active_tensors':15,'new_BPE_table_tensors':3,'active_shared_heads':12,'frozen_encoder_LoRA_tensors':192,
 'acoustic_encoder_no_grad':True,'encoder_projector_outside_no_grad':True,'shared_decoder_differentiable':True,
 'admitted_row_LRs':[.001,.0003],'admitted_shared_LRs':[.00001,.000003],
 'row_source_LR_before_copy':.0001,'shared_source_LR_before_copy':.00001,'row_decay':0.,'retained_shared_source_decay':.01,
 'row_clock':'K+N','shared_clock':'S+N','cumulative_frontier':'S+K+N','single_active15_clip_norm':1.,
 'original3_resized_table_and192_LoRA_moments_active':False,'all207_weights_saved':True,'two_Adam_saved':True,
 'external_exact_resume_supported':False,'further_unfreeze_supported':False},
 'unchanged':{'B':8,'accumulation':2,'sampling':'domain30 / independent replay','replay1595_teacher8193':True,
 'KD':10,'native_backend':'native_tdt_checkpointed_v4','native_joint_bound':1850000000,
 'per_target_U_normalization_then_physical_batch_mean':True,'DEV_every_local_updates':250,
 'cap8GB_end4GB_inactive750MB_resource_progress_guards':True},
 'preflight_cli':cli+' --validate-only --preflight-output NEW_OUTPUT_OUTSIDE_FRESH_LOOP',
 'future_ROOT_smoke_cli':cli+' --smoke-updates 1 --gpu-window-authorized',
 'future_ROOT_main_cli':'Use separate fresh target/receipt/loop; '+cli+' --gpu-window-authorized',
 'source300_proof':'runs/<RUN>/TSC_BPE_shared_head_initial_source_equivalence.json',
 'source300_proof_kind':'TSC_BPE512_shared_heads_before_update_source_equivalence_v1',
 'source300_before_update_flag':'before_shared_head_updates',
 'smoke_proof':'runs/<RUN>/adaptation_actual_one_update_smoke.json',
 'smoke_kind':'TSC_BPE_shared_heads_actual_one_update_smoke_v1',
 'smoke_required':['actual_native_ASR_backward_and_two_Adam','all3_new_row_deltas_nonzero',
 'shared12_gradients_finite_and_all_deltas_nonzero','protected_old_six_blank_duration_and_frozen192_encoder_LoRA_rawbytes_exact',
 'complete207_two_optimizer_phase_save_load_passed','all15_Adam_moments_clocks_and_groups_exact','source300 equality'],
 'Torch_or_GPU_executed':False,'real_checkpoint_model_audio_or_live_data_read':False,'activation_ready_claimed':False,
 'pending_actual_gates':['ROOT chosen rowLR after matched scheduled result','fresh source receipt/preflight',
 'actual source300 path/ASR equality','CPU two-Adam utility proof and actual1update smoke','separate fresh1000 screen'],
 'future_mirror_phase_admission_required':True,'current_activated_files_or_logging_modified':False}
out=HERE/'TSC_BPE_shared_head_source_handoff_v1.json'
with out.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({'handoff':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
