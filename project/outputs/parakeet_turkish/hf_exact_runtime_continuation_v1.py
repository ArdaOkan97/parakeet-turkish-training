"""Narrow already-native3 HF continuation identity check; no tensor operations."""
import copy
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def require(ok,message):
    if not ok:
        raise ValueError(message)


def exact_identity_matches(saved,args,current,output,native_identity):
    old=saved['config_identity']
    path=args.resume_runtime_migration.resolve()
    receipt=json.loads(path.read_text(encoding='utf-8'))
    require(receipt.get('kind')=='HF_already_native3_exact_runtime_continuation_v1'
        and receipt.get('passed') is True and receipt.get('status')=='CPU_prepared'
        and receipt.get('target_run_id')==args.run_id
        and receipt.get('source_step')==saved['step']
        and receipt.get('inherited_selected_step')==saved['selected_step']
        and receipt.get('source_config_identity_sha256')==canonical(old), 'Exact HF continuation receipt differs')
    fork=receipt['fork_latest']
    require(Path(fork['path']).resolve()==args.resume_exact.resolve()
        and sha(args.resume_exact)==fork['sha256'] and fork['step']==saved['step'], 'Immutable full-state fork differs')
    source=receipt['source_run_config']
    require(sha(source['path'])==source['sha256'], 'Closed source configuration changed')
    parent=json.loads(Path(source['path']).read_text(encoding='utf-8'))
    require(parent['run_id']==receipt['source_run_id'] and parent['config_identity']==old,
        'Closed source run/config identity differs')
    mappings={'train_lora_batched_data_v10.py':'train_lora_batched_data_v11.py',
              'launch_native_training_v4.py':'launch_native_training_v5.py'}
    require(set(receipt['approved_source_changes'])==set(mappings), 'Only identity-routing source substitutions allowed')
    expected=copy.deepcopy(old)
    for previous,new in mappings.items():
        item=receipt['approved_source_changes'][previous]
        require(item['new_filename']==new and item['old_sha256']==old['source_sha256'].get(previous)
            and sha(output/previous)==item['old_sha256'] and sha(output/new)==item['new_sha256'],
            'Closed/new identity-routing source changed')
        expected['source_sha256'].pop(previous)
        expected['source_sha256'][new]=item['new_sha256']
    added=receipt['added_identity_validator']
    require(added['filename']=='hf_exact_runtime_continuation_v1.py'
        and sha(output/added['filename'])==added['sha256'] and added['filename'] not in expected['source_sha256'],
        'Only explicit continuation validator may be added')
    expected['source_sha256'][added['filename']]=added['sha256']
    previous_migration=str(Path(parent['resume_runtime_migration']).resolve())
    require(receipt['replaced_previous_migration']['path']==previous_migration
        and expected['input_sha256'].get(previous_migration)==receipt['replaced_previous_migration']['sha256']
        and sha(previous_migration)==receipt['replaced_previous_migration']['sha256'], 'Previous migration receipt changed')
    expected['input_sha256'].pop(previous_migration)
    backend=receipt['native_backend_transition']
    old_ready=str(Path(parent['native_backend_provenance']).resolve())
    new_ready=str(args.native_backend_provenance.resolve())
    require(str(Path(backend['old_path']).resolve())==old_ready
        and str(Path(backend['new_path']).resolve())==new_ready
        and expected['input_sha256'].get(old_ready)==backend['old_sha256']
        and sha(old_ready)==backend['old_sha256'], 'Old/new native ready provenance differs')
    ready_sha=sha(new_ready)
    expected['input_sha256'].pop(old_ready)
    require(str(path) not in expected['input_sha256'] and new_ready not in expected['input_sha256'], 'New receipt/ready namespace required')
    expected['input_sha256'][new_ready]=ready_sha
    expected['input_sha256'][str(path)]=sha(path)
    expected['native_loss']['native_backend_provenance_sha256']=ready_sha
    require(old['native_loss']['kind']=='native_nemo_checkpointed_per_clip_v3'
        and expected==current, 'Undeclared scientific/data/loss/layout/LR/KD change in exact HF continuation')
    ready=json.loads(Path(new_ready).read_text(encoding='utf-8'))
    policy={'cap_gb':8,'end_cache_gb':4,'native_inactive_threshold_bytes':750000000}
    require(receipt['declared_allocator_policy']==policy and ready['declared_allocator_policy']==policy,
        'Preserve proven cap8/end4/native750MB policy')
    native_identity(args)
    return True
