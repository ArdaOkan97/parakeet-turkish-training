"""Explicit TRAIN Turkish lengths/labels; no Turkish frozen-encoder features."""
from pathlib import Path
import hashlib, json, math
import torch

KIND='TR_frame_target_metadata_v1'
EXPECTED_MANIFEST_SHA='d2d0f6df539b1e5cfe9488960a5859bc6ab1870a1ffc634b5b707bc2cc2f5704'
EXPECTED_TOKENIZER_SHA='b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75'

def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()

def annotate_tr_frame_metadata(processor, rows, metadata_path, proof_path, max_tokens):
    proof=json.loads(Path(proof_path).read_text(encoding='utf-8'))
    if not(proof.get('passed') is True and proof.get('CPU_only') is True and proof.get('clips')==74883
        and proof.get('manifest_sha256')==EXPECTED_MANIFEST_SHA and proof.get('metadata_sha256')==sha(metadata_path)
        and proof.get('cache_files_written')==0 and proof.get('can_current_frozen_trainer_consume_metadata') is False):
        raise ValueError('Exact completed CPU frame proof required')
    tokenizer_sha=hashlib.sha256(json.dumps(json.loads(processor.tokenizer._tokenizer.to_str()),ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    if tokenizer_sha!=EXPECTED_TOKENIZER_SHA:raise ValueError('Exact unchanged tokenizer required')
    metadata=[json.loads(line) for line in Path(metadata_path).read_text(encoding='utf-8').splitlines() if line]
    if len(rows)!=len(metadata) or len(rows)!=74883:raise ValueError('Whole public TRAIN metadata pool required')
    result=[]
    for row,item in zip(rows,metadata):
        if row['language']!='tr' or row['source_id']!=item['source_id']:raise ValueError('Exact ordered TR metadata required')
        mel=int(item['actual_PCM_frames'])//160;frames=(mel+7)//8
        if (int(round(row['duration']*16000))!=item['actual_PCM_frames'] or item['valid_mel_frames']!=mel or item['encoder_frames']!=frames):raise ValueError('Vetted PCM/frame lengths disagree')
        labels=processor.tokenizer(row['text'])['input_ids']
        if (not 1<=len(labels)<=max_tokens or len(labels)!=item['target_tokens'] or min(labels)<0 or max(labels)>=8199 or 8192 in labels
            or hashlib.sha256(json.dumps(labels,sort_keys=True).encode()).hexdigest()!=item['label_ids_sha256']):raise ValueError('Metadata exact target IDs differ')
        result.append({**row,'encoder_frames':frames,'target_tokens':len(labels),
            'training_input_type':KIND,'tr_metadata_labels':labels,'tr_frame_metadata_sha256':proof['metadata_sha256']})
    return result

def confirm_tr_metadata_acceptance(rows,audit):
    # The policy hashes ordered IDs using its canonical implementation.
    from public_turkish_data_policy_v3 import ids_sha
    if ids_sha(rows)!=audit['accepted_source_ids_sha256'] or len(rows)!=audit['identity']['accepted_clips']:raise ValueError('Whole vetted TR pool required')
    audit.update(turkish_training_representation=KIND,TR_frozen_encoder_feature_cache_required=False,
        actual_TR_metadata_accepted_clips=len(rows),actual_TR_metadata_accepted_hours=math.fsum(row['duration'] for row in rows)/3600,
        TR_metadata_acceptance_pending=False,acoustic_cache_acceptance_pending='not_applicable_to_TR_metadata')

def load_training_item(row):
    if row['language']=='tr':
        if row.get('training_input_type')!=KIND or 'cache_path' in row:raise ValueError('Turkish training requires explicit metadata, no teacher feature cache')
        return {'kind':KIND,'source_id':row['source_id'],'labels':torch.tensor(row['tr_metadata_labels'],dtype=torch.long),
            'encoder_frames':row['encoder_frames']}
    return torch.load(Path(row['cache_path']),map_location='cpu',weights_only=True)
