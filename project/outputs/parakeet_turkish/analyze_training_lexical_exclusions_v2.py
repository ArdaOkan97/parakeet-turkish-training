"""CPU TRAIN-only diagnosis of the already excluded v2 lexical targets."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import re
from datetime import datetime,timezone
from metrics import normalize_text
from resource_watchdog_v1 import sample_memory
from atomic_io_retry_v1 import write_json

ROOT=Path(__file__).resolve().parents[2];OUTPUT=ROOT/'outputs/parakeet_turkish';DATA=ROOT/'work/data'
def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()
target=OUTPUT/'turkish_v2_lexical_roundtrip_diagnosis.json'
if target.exists():raise FileExistsError(target)
before=sample_memory(os.getpid(),'TRAIN_lexical_diagnosis_before_Torch_tokenizer_import')
if before['system_memory'].get('commit_fraction',1)>.85:
    raise RuntimeError('Defer bounded tokenizer diagnostic until system commit<=85percent')
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(OUTPUT/'checkpoint',local_files_only=True)
tokenizer_sha=hashlib.sha256(json.dumps(json.loads(tokenizer._tokenizer.to_str()),ensure_ascii=False,sort_keys=True).encode()).hexdigest()
if tokenizer_sha!='b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75':raise ValueError('Pinned tokenizer differs')
with (DATA/'tr_public_v2_train.jsonl').open(encoding='utf-8') as stream:
    accepted={json.loads(line)['source_id'] for line in stream if line.strip()}
categories=Counter();source_counts=Counter();patterns=Counter();examples=[];excluded=0;unchanged_sources=0
character_changes=Counter();digit_count=0;new_character_targets=0
with (DATA/'tr_public_v2_prepared.jsonl').open(encoding='utf-8') as stream:
    for line in stream:
        row=json.loads(line)
        if row['source_id'] in accepted:continue
        if row['dataset_split'] not in ('train','validated'):raise ValueError('Only prepared TRAIN-role material allowed')
        ids=tokenizer(row['text'])['input_ids'];decoded=tokenizer.decode(ids,skip_special_tokens=True)
        left=normalize_text(row['text'],'tr');right=normalize_text(decoded,'tr')
        if left==right:raise ValueError('An excluded target unexpectedly round-trips')
        excluded+=1;source_counts[row.get('data_expansion_role','preserved_existing')]+=1
        if any(c.isdigit() for c in row['text']):digit_count+=1
        if any(i>=8193 for i in ids):new_character_targets+=1
        if left.replace(' ','')==right.replace(' ',''):
            kind='word_boundary_change_only'
            if re.search(r'\d\s+\d',left) or re.search(r'\d\s+\d',right):patterns['number_group_boundary']+=1
        elif ''.join(c for c in left if c.isalnum())==''.join(c for c in right if c.isalnum()):
            kind='other_separator_change_only'
        else:kind='lexical_character_change'
        categories[kind]+=1
        for character in 'ğĞıİşŞçÇöÖüÜ':
            delta=right.count(character)-left.count(character)
            if delta:character_changes[character]+=delta
        if len(examples)<15:
            examples.append({'source_id':row['source_id'],'dataset_split':row['dataset_split'],
                'data_expansion_role':row.get('data_expansion_role','preserved_existing'),
                'category':kind,'training_reference':row['text'],'normalized_reference':left,
                'tokenizer_decode':decoded,'normalized_decode':right,'target_tokens':len(ids),
                'appended_character_ids':sorted({i for i in ids if i>=8193})})
expected=json.loads((OUTPUT/'tr_public_v2_preparation.json').read_text(encoding='utf-8'))
if excluded!=sum(expected['lexical_exclusions'].values()):raise ValueError('Excluded source count differs')
after=sample_memory(os.getpid(),'TRAIN_lexical_diagnosis_complete')
report={'schema':1,'passed':True,'kind':'TRAIN_only_lexical_roundtrip_diagnosis_v2','created_at_utc':datetime.now(timezone.utc).isoformat(),
    'CPU_only':True,'CUDA_initialized':torch.cuda.is_initialized(),'tokenizer_semantic_sha256':tokenizer_sha,
    'prepared_manifest_sha256':sha(DATA/'tr_public_v2_prepared.jsonl'),'accepted_manifest_sha256':sha(DATA/'tr_public_v2_train.jsonl'),
    'diagnostic_source_sha256':sha(__file__),'excluded_clips':excluded,'source_role_counts':dict(source_counts),
    'categories':dict(categories),'boundary_patterns':dict(patterns),'excluded_targets_with_digits':digit_count,
    'excluded_targets_using_appended_character_ids':new_character_targets,'Turkish_character_count_deltas':dict(character_changes),
    'small_TRAIN_examples':examples,'before_owned_memory':before,'after_owned_memory':after,
    'no_DEV_or_TEST_references_or_predictions_read':True,
    'recommendation_limits':['Tokenizer subword coverage changes should be tested on TRAIN text first, preserve original IDs and initial multilingual behavior, and select quality on fixed DEV.',
        'Do not alter fixed scoring normalization to improve measured WER. These exclusions are tokenizer/reference fidelity checks, not observed ASR prediction errors.',
        'Restoring excluded clips requires deliberate transcript/tokenization policy; no new data is silently accepted by this diagnostic.']}
write_json(target,report)
print(json.dumps({'passed':True,'excluded':excluded,'categories':dict(categories),'patterns':dict(patterns),'source_role_counts':dict(source_counts)}))
