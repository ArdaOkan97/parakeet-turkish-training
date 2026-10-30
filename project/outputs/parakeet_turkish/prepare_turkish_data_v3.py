"""Recompute future TRAIN eligibility with explicit ungrouped TDT decoding.

Reuses unchanged v2 prepared rows/audio and completed v2 full PCM audit. Only new
eligible recordings are reread; no download, audio conversion, old artifact write,
DEV/TEST prediction, model inference or vocabulary change occurs.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
import os
import time

from atomic_io_retry_v1 import write_json
from metrics import normalize_text
from public_turkish_data_policy_v2 import sha256, audio_evidence
from resource_watchdog_v1 import sample_memory

OUTPUT=Path(__file__).resolve().parent;ROOT=OUTPUT.parents[1];DATA=ROOT/'work/data'
start=time.perf_counter()
target=DATA/'tr_public_v3_train.jsonl';proof_path=OUTPUT/'tr_public_v3_preparation.json';label_path=OUTPUT/'tr_public_v3_tokenizer_acceptance.json'
if any(path.exists() for path in (target,proof_path,label_path)):raise FileExistsError('New v3 outputs required; prior artifacts stay immutable')
v2_path=OUTPUT/'tr_public_v2_preparation.json';v2=json.loads(v2_path.read_text(encoding='utf-8'))
full_audit_path=OUTPUT/'tr_public_v2_input_verification.json';full=json.loads(full_audit_path.read_text(encoding='utf-8'))
if not full['passed'] or full['audit']['identity']['accepted_manifest_sha256']!=v2['CPU_tokenizer_accepted_manifest_sha256']:
    raise ValueError('Completed v2 actual-audio verification required')
prepared=Path(v2['prepared_manifest_path']);v2_manifest=Path(v2['CPU_tokenizer_accepted_manifest_path'])
if sha256(prepared)!=v2['prepared_manifest_sha256'] or sha256(v2_manifest)!=v2['CPU_tokenizer_accepted_manifest_sha256']:
    raise ValueError('Pinned existing manifests differ')
old_ids=set()
with v2_manifest.open(encoding='utf-8') as stream:
    old_ids={json.loads(line)['source_id'] for line in stream if line.strip()}
old_labels=json.loads((OUTPUT/'tr_public_v2_tokenizer_acceptance.json').read_text(encoding='utf-8'))
canonical={row['source_id']:row['canonical_pcm_sha256'] for row in old_labels['accepted_source_labels_and_audio']}
before=sample_memory(os.getpid(),'CPU_v3_TDT_eligibility_before_tokenizer_import')
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(OUTPUT/'checkpoint',local_files_only=True)
tokenizer_sha=hashlib.sha256(json.dumps(json.loads(tokenizer._tokenizer.to_str()),ensure_ascii=False,sort_keys=True).encode()).hexdigest()
if tokenizer_sha!=v2['tokenizer_semantic_sha256']:raise ValueError('Exact unchanged tokenizer required')
counts=Counter();excludes=Counter();grouping_recovery=Counter();evidence=[];accepted=[];new_ids=[];new_hours=[];examples=[];newly_recovered_audio_hashes=[]
with prepared.open(encoding='utf-8') as stream:
    for line in stream:
        row=json.loads(line);ids=tokenizer(row['text'])['input_ids']
        decoded=tokenizer.decode(ids,skip_special_tokens=True,group_tokens=False)
        reference=normalize_text(row['text'],'tr');normal=normalize_text(decoded,'tr')
        reason=None
        if not ids:reason='empty_target'
        elif len(ids)>180:reason='target_exceeds180'
        elif min(ids)<0 or max(ids)>=len(tokenizer) or 8192 in ids:reason='invalid_blank_or_OOV_target'
        elif normal!=reference:reason='ungrouped_TDT_lexical_roundtrip_mismatch'
        if reason:
            excludes[reason]+=1
            if len(examples)<8:examples.append({'source_id':row['source_id'],'reference':row['text'],'ungrouped_decode':decoded,'reason':reason})
            continue
        source=row.get('data_expansion_role','preserved_existing')
        if row['source_id'] not in old_ids:
            expected=row.get('canonical_pcm_sha256')
            measured=audio_evidence(row,expected)
            canonical[row['source_id']]=measured;new_ids.append(row['source_id']);new_hours.append(row['duration'])
            grouping_recovery[source]+=1
            newly_recovered_audio_hashes.append({'source_id':row['source_id'],'wav_sha256':row['wav_sha256'],'canonical_pcm_sha256':measured})
        accepted.append(row);counts[source]+=1
        evidence.append({'source_id':row['source_id'],'target_tokens':len(ids),'target_max_id':max(ids),
            'label_ids_sha256':hashlib.sha256(json.dumps(ids,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
            'canonical_pcm_sha256':canonical[row['source_id']]})
if not old_ids.issubset({row['source_id'] for row in accepted}):raise ValueError('Previously accepted recordings unexpectedly lost under TDT decoding')
with target.open('x',encoding='utf-8',newline='\n') as stream:
    for row in accepted:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
for path,digest in v2['protected_original_files_sha256'].items():
    if sha256(path)!=digest:raise ValueError('Protected original artifact changed')
result={**v2,'schema':1,'kind':'public_turkish_data_v3','status':'CPU_TDT_tokenizer_verified;future_acousticcache_and_optimizer_exposure_pending',
    'created_at_utc':datetime.now(timezone.utc).isoformat(),'preparation_script_sha256':sha256(__file__),
    'tokenizer_decode_group_tokens':False,'TDT_decode_policy':'Explicit False; CTC grouping is inappropriate for autoregressive TDT labels',
    'CPU_tokenizer_accepted_clips':len(accepted),'CPU_tokenizer_accepted_hours':math.fsum(row['duration'] for row in accepted)/3600,
    'CPU_tokenizer_accepted_source_counts':dict(counts),'lexical_exclusions':dict(excludes),
    'CPU_tokenizer_accepted_manifest_path':str(target.resolve()),'CPU_tokenizer_accepted_manifest_sha256':sha256(target),
    'prior_v2_preparation_path':str(v2_path.resolve()),'prior_v2_preparation_sha256':sha256(v2_path),
    'prior_v2_manifest_path':str(v2_manifest.resolve()),'prior_v2_manifest_sha256':sha256(v2_manifest),
    'reused_full_v2_audio_audit_path':str(full_audit_path.resolve()),'reused_full_v2_audio_audit_sha256':sha256(full_audit_path),
    'reused_verified_audio_clips':len(old_ids),'newly_recovered_clips':len(new_ids),'newly_recovered_hours':math.fsum(new_hours)/3600,
    'newly_recovered_by_source_role':dict(grouping_recovery),'newly_recovered_actual_audio_evidence':newly_recovered_audio_hashes,
    'remaining_small_TRAIN_examples':examples,'before_owned_memory':before,
    'after_owned_memory':sample_memory(os.getpid(),'CPU_v3_TDT_eligibility_complete'),
    'CUDA_initialized':torch.cuda.is_initialized(),'vocabulary_changed':False,
    'original_actual_accepted_source_set_preserved_as_subset':True,
    'no_DEV_or_TEST_predictions_or_references_read_for_diagnosis':True,
    'elapsed_seconds':time.perf_counter()-start}
write_json(label_path,{'schema':1,'passed':True,'tokenizer_semantic_sha256':tokenizer_sha,'max_tokens':180,
    'group_tokens':False,'existing_actual_trainer_source_set_preserved':True,'accepted_source_labels_and_audio':evidence})
write_json(proof_path,result)
print(json.dumps({key:result[key] for key in ('CPU_tokenizer_accepted_clips','CPU_tokenizer_accepted_hours','newly_recovered_clips','newly_recovered_hours','lexical_exclusions','elapsed_seconds')}),flush=True)
