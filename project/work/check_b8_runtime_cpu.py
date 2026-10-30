"""CPU-only B8 benchmark/runtime binding; no CUDA initialization or tensor loads."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent.parent
OUTPUT=ROOT/'outputs/parakeet_turkish'
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
checks=[]
def check(name,value):
    if not value:raise AssertionError(name)
    checks.append(name)
speed_path=OUTPUT/'large_batch_speed_report.json'
speed=read(speed_path)
audit_path=Path(speed['verification_path'])
check('speed_report_binds_complete_independent_audit',speed['integrity_passed'] is True and sha(audit_path)==speed['verification_sha256'])
audit=read(audit_path)
check('independent_matrix_audit_passed',audit['passed'] is True and audit['status']=='complete' and len(audit['checks'])==108939)
matrix=read(OUTPUT/'large_batch_matrix.json')
actual_defaults={'tf32_matmul_allowed':torch.backends.cuda.matmul.allow_tf32,
    'tf32_cudnn_allowed':torch.backends.cudnn.allow_tf32,'cudnn_enabled':torch.backends.cudnn.enabled,
    'cudnn_benchmark':torch.backends.cudnn.benchmark,
    'sdpa_flash_enabled':torch.backends.cuda.flash_sdp_enabled(),'sdpa_math_enabled':torch.backends.cuda.math_sdp_enabled(),
    'sdpa_mem_efficient_enabled':torch.backends.cuda.mem_efficient_sdp_enabled(),'sdpa_cudnn_enabled':torch.backends.cuda.cudnn_sdp_enabled()}
check('fresh_defaults_read_without_CUDA_initialization',torch.cuda.is_initialized() is False)
profiles=[]
runtime={'batch_size':8,'accumulation':2,'effective_batch':16,'precision':'bf16','tdt_implementation':'v2',
    'backend':'default','encoder_checkpointing':True,'chunk_size':16}
for family in ('typical','long'):
    case_id='large_batch_'+family+'_timing_default_b8'
    path=OUTPUT/(case_id+'.json');report=read(path)
    case=next(c for c in matrix['cases'] if c['case_id']==case_id)
    check(family+'_frozen_case_report_sha',sha(path)==case['report_sha256'])
    check(family+'_exact_selected_runtime',all(report[k]==v for k,v in runtime.items()))
    check(family+'_finite_timing_status',report['status']=='measured_with_numerical_gate_failure')
    timing=report['timing']['end_to_end']
    check(family+'_complete_warmed_updates',timing['updates']==(10 if family=='typical' else 3) and timing['warmup_updates']==2)
    row=next(r for r in speed['table'] if r['case_id']==case_id)
    for field in ('audio_min_per_wall_min','peak_allocated_vram_gb','peak_reserved_vram_gb'):
        check(family+'_'+field+'_exact_speed_report',row[field]==timing[field])
    check(family+'_preserved_strict_failure',row['fp32_math_oracle_gate_passed'] is True
        and row['same_shape_bf16_gate_passed'] is False and row['adoption_eligible'] is False
        and row['exploratory_despite_numerical_failure'] is True)
    check(family+'_preserved_BF16_backend_difference',report['backend_flags']['tf32_cudnn_allowed'] is False
        and actual_defaults['tf32_cudnn_allowed'] is True)
    profiles.append({'family':family,'case_id':case_id,'path':str(path),'sha256':sha(path),'runtime':runtime,
        'benchmark_backend_flags':report['backend_flags'],'audio_min_per_wall_min':row['audio_min_per_wall_min'],
        'peak_allocated_vram_gb':row['peak_allocated_vram_gb'],'peak_reserved_vram_gb':row['peak_reserved_vram_gb'],
        'gain_vs_matched_effective16_B1':row['gain_vs_matched_b1'],'lattice_padding_factor':row['lattice_padding_factor'],
        'fp32_math_oracle_gate_passed':True,'strict_same_shape_bf16_gate_passed':False,'adoption_eligible':False,
        'maximum_sampled_encoder_frames':max(r['encoder_frames'] for r in report['source_clips']),
        'maximum_sampled_target_tokens':max(r['target_tokens'] for r in report['source_clips']),
        'benchmark_optimizer':report['optimizer'],'benchmark_initial_trainable_sha256':report['initial_trainable_sha256']})
rows=[r for r in speed['table'] if r['family']=='typical']
check('B8_fastest_audited_typical_case',max(rows,key=lambda r:r['audio_min_per_wall_min'])['batch_size']==8)
sources={name:sha(OUTPUT/name) for name in ('train_lora_batched.py','batched_lora.py','runtime.py','launch_with_io_retry_v1.py')}
for name in ('train_lora_batched.py','runtime.py','launch_with_io_retry_v1.py'):
    source=(OUTPUT/name).read_text(encoding='utf-8')
    check(name+'_does_not_override_TF32_defaults','allow_tf32' not in source)
check('clip_exposure_warmup_matches',25*16==100*4==400)
report={'schema':1,'passed':True,'cpu_only':True,'gpu_initialized':torch.cuda.is_initialized(),
    'created_at_utc':datetime.now(timezone.utc).isoformat(),'checks_passed':len(checks),'checks':checks,
    'speed_report_sha256':sha(speed_path),'matrix_sha256':sha(OUTPUT/'large_batch_matrix.json'),
    'independent_matrix_audit_sha256':sha(audit_path),'source_sha256':sources,
    'fresh_training_environment_backend_defaults':actual_defaults,'profiles':profiles,
    'backend_difference':{'field':'cudnn.allow_tf32','benchmark':False,'unmodified_training_default':True,
        'interpretation':'Explicitly recorded for the new BF16 quality experiment; no exact backend-equivalence or timing-transfer claim.'},
    'new_trial_plan':{'physical_batch_size':8,'gradient_accumulation':2,'effective_optimizer_batch':16,
        'previous_effective_optimizer_batch':4,'replay_fraction_by_clips':.5,'Turkish_clips_per_update':8,
        'replay_clips_per_update':8,'warmup_updates':25,'warmup_clip_presentations':400,
        'previous_warmup_updates':100,'previous_warmup_clip_presentations':400,
        'optimizer_policy':'fresh optimizer warm-started from best DEV-qualified weights; optimizer batch and update frequency changed',
        'selection_policy':'fixed DEV only with five old-language retention gates; TEST reporting only after frozen final choice'},
    'scope_limits':['Existing timing used the immutable pilot1200 weights and fixed representative TRAIN clips; the next trial has different trained weights and learning parameters.',
        'Only sampled typical/long batches were measured; long sample maxT326/U126 does not prove a memory ceiling for every TRAIN batch up to targetcap180.',
        'Strict BF16 gradient gates stay failed. The user-authorized B8 trial must establish actual throughput and DEV/retention quality separately.']}
target=OUTPUT/'b8_runtime_cpu_review.json'
target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'checks_passed':len(checks),'gpu_initialized':report['gpu_initialized'],'path':str(target)}))
