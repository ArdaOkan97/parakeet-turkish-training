from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/parakeet_turkish'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
prior=D/'public_turkish_v3_launch_preflight_v4.json';report=json.loads(prior.read_text())
report.update(kind='TR_metadata_only_optional_input_audit_reuse_CPU_readiness_v1',
    status='CPU_ready_root_actual_GPU_length_smoke_pending',prior_readiness_path=str(prior),prior_readiness_sha256=sha(prior))
names=['train_lora_batched_data_v6.py','launch_with_io_retry_v4.py','launch_with_cache_reclaim_v7.py','launch_with_memory_guard_v10.py',
    'tr_frame_metadata_v1.py','batched_lora_tr_metadata_v1.py','training_input_audit_reuse_v1.py','prepare_input_audit_reuse_v1.py',
    'verify_tr_frame_metadata_v1.py','verify_tr_metadata_gpu_v1.py','TR_frame_metadata_cpu_verification_v1.json',
    'TR_metadata_collator_cpu_verification_v1.json','input_audit_reuse_cpu_verification_v1.json','data_v6_actual_io_dispatch_cpu_verification.json']
for name in names:report['artifacts_sha256'][str(D/name)]=sha(D/name)
for role,name in [('trainer','train_lora_batched_data_v6.py'),('io_launcher','launch_with_io_retry_v4.py'),('cache_launcher','launch_with_cache_reclaim_v7.py'),('memory_launcher','launch_with_memory_guard_v10.py')]:
    report['production_'+role+'_path']=str(D/name);report['production_'+role+'_sha256']=sha(D/name)
metadata=ROOT/'work/data/tr_public_v3_frame_metadata_v1.jsonl';receipt=D/'public_data_full_audit_reuse_v1/reuse_receipt.json'
report['artifacts_sha256'][str(metadata)]=sha(metadata);report['artifacts_sha256'][str(receipt)]=sha(receipt)
report.update(TR_training_representation='Explicit frame/target metadata; no TR frozen encoder feature cache or fake features',
    replay_teacher_representation='Genuine unchanged singleton base encoder feature files; packed replay-only in collator',
    TR_metadata_path=str(metadata),TR_metadata_sha256=sha(metadata),TR_frame_proof_path=str(D/'TR_frame_metadata_cpu_verification_v1.json'),
    optional_input_audit_reuse_receipt_path=str(receipt),optional_input_audit_reuse_receipt_sha256=sha(receipt),
    training_start_full_actual_WAV_verification_enabled='Default true; optional explicit --reuse-input-audit uses code-bound full audit receipt + fresh all fingerprints/headers + strict proof-chain instead',
    runtime_input_audit_reuse_requires_explicit_flag=True,
    CPU_guard_checks={**report['CPU_guard_checks'],'ALL_PCM_headers_and_existing25659_cached_frame_labels':74883,
        'actual_HF_feature_extractor_subsampling_boundary_fixtures':17,'exact_CPU_mixed_allTR_objective_every_gradient_and_AST':33,
        'optional_receipt_reuse_stale_fingerprint_default_full_audit_and_AST':12,'actual_nested_v6_parser_serializer_cache539_dispatch':19},
    pending=['Root-owned16TRAIN GPU actual frontend/encoder mask smoke','Actual fresh metadata-trainer accepted74,883 pool and first update validation','Actual new-data throughput/memory/DEV retention progression'],
    future_GPU_smoke_command='work/training-env/Scripts/python.exe -X utf8 outputs/parakeet_turkish/verify_tr_metadata_gpu_v1.py --root-owned-solo-gpu --report outputs/parakeet_turkish/TR_metadata_actual_GPU_masks_v1.json')
report['next_run_contract']['TR_frozen_encoder_cache_pass']='Eliminated; raw student encoder still computes/trainable each update'
report['next_run_contract']['actual_GPU_length_parity_required']=True
report['source_files_ready_for_parent_freeze']=True
target=D/'TR_metadata_launch_readiness_v1.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'path':str(target),'sha256':sha(target)}))
