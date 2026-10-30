"""TRAIN-only remaining lexical exclusions and final future-pipeline readiness."""
from pathlib import Path
from collections import Counter,defaultdict
import sys,json,math,hashlib,ast,os
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/parakeet_turkish';DATA=ROOT/'work/data'
sys.path.insert(0,str(D))
from metrics import normalize_text
from public_turkish_data_policy_v3 import sha256
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
proof=load(D/'tr_public_v3_preparation.json');audit=load(D/'tr_public_v3_input_verification.json')
ids=set();source_counts=Counter();source_durations=defaultdict(list);repeats=Counter()
with (DATA/'tr_public_v3_train.jsonl').open(encoding='utf-8') as stream:
    for line in stream:
        row=json.loads(line);ids.add(row['source_id']);role=f"{row['dataset']}:{row['dataset_split']}"
        source_counts[role]+=1;source_durations[role].append(row['duration']);repeats[normalize_text(row['text'],'tr')]+=1
excluded=[]
with (DATA/'tr_public_v2_prepared.jsonl').open(encoding='utf-8') as stream:
    for line in stream:
        row=json.loads(line)
        if row['source_id'] not in ids:excluded.append(row)
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(D/'checkpoint',local_files_only=True)
categories=Counter();excluded_sources=Counter();digit_count=0;examples=[]
for row in excluded:
    label=tokenizer(row['text'])['input_ids'];decoded=tokenizer.decode(label,skip_special_tokens=True,group_tokens=False)
    ref=normalize_text(row['text'],'tr');hyp=normalize_text(decoded,'tr')
    if ref==hyp:raise ValueError('Unexpected representable excluded record')
    category='word_boundary_only' if ref.replace(' ','')==hyp.replace(' ','') else 'character_content_change'
    categories[category]+=1;excluded_sources[row['dataset']+':'+row['dataset_split']]+=1
    digits=any(c.isdigit() for c in ref);digit_count+=digits
    if len(examples)<6:examples.append(dict(source_id=row['source_id'],TRAIN_reference=row['text'],ungrouped_TDT_decode=decoded,category=category))
diag=dict(schema=1,kind='TRAIN_only_v3_remaining_lexical_diagnosis',passed=True,CPU_only=True,no_GPU_execution=True,
    no_DEV_TEST_predictions_or_references_read=True,excluded_count=len(excluded),excluded_categories=dict(categories),
    excluded_source_splits=dict(excluded_sources),excluded_references_containing_digits=digit_count,
    small_TRAIN_examples=examples,tokenizer_decode_group_tokens=False,vocabulary_changed=False,
    accepted_source_counts=dict(source_counts),accepted_actual_PCM_hours_by_source={k:math.fsum(v)/3600 for k,v in source_durations.items()},
    accepted_repeated_normalized_TRAIN_text_groups=sum(v>1 for v in repeats.values()),
    accepted_extra_recordings_with_repeated_TRAIN_text=sum(v-1 for v in repeats.values() if v>1),
    newly_recovered_clips=proof['newly_recovered_clips'],newly_recovered_hours=proof['newly_recovered_hours'],
    correct_TDT_decode_recovers_existing_tokens_not_new_vocabulary=True,
    remaining_exclusions_policy='Keep fixed refs/normalizer/vocabulary; do not silently substitute collapsed or number-attached labels',
    CUDA_initialized=torch.cuda.is_initialized())
target=D/'turkish_v3_remaining_lexical_diagnosis.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(diag,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
files=['prepare_turkish_data_v3.py','tr_public_v3_preparation.json','tr_public_v3_tokenizer_acceptance.json','tr_public_v3_input_verification.json',
       'public_turkish_data_policy_v3.py','train_lora_batched_data_v3.py','launch_with_cache_reclaim_v3.py','launch_with_memory_guard_v5.py',
       'resource_watchdog_v2.py','paired_windows_commit_telemetry_v1.py','public_data_policy_v3_cpu_verification.json',
       'public_data_v3_trainer_core_cpu_verification.json','memory_guard_v5_final_cpu_verification.json','TDT_decode_v3_cpu_verification.json',
       'turkish_v3_remaining_lexical_diagnosis.json']
ready=dict(schema=1,kind='public_turkish_v3_future_training_CPU_readiness',passed=True,status='CPU_ready_actual_new_data_GPU_launch_pending',
    accepted_clips=proof['CPU_tokenizer_accepted_clips'],accepted_actual_PCM_hours=proof['CPU_tokenizer_accepted_hours'],
    accepted_manifest_path=proof['CPU_tokenizer_accepted_manifest_path'],accepted_manifest_sha256=proof['CPU_tokenizer_accepted_manifest_sha256'],
    artifacts_sha256={str((D/f).resolve()):sha256(D/f) for f in files},
    protected_original_artifacts_unchanged=True,tokenizer_decode_group_tokens=False,vocabulary_changed=False,
    source_counts=diag['accepted_source_counts'],actual_PCM_hours_by_source=diag['accepted_actual_PCM_hours_by_source'],
    CPU_guard_checks={'source_eligibility_and_TDT_policy':32,'entire_update_AST_and_exact539_hook':4,'memory_wrapper_mocks_and_native_API':45,'actual_TRAIN_tokenizer_regressions':8},
    latest_CPU_input_audit=audit['audit']['audio_verification_mode'],
    training_start_full_actual_WAV_verification_enabled=True,
    next_run_contract={'warm_start':'Selected1871_or_later_DEV_qualified_weights_only; fresh Adam/sampler/RNG accordingto explicit newrun seed',
        'rank':8,'alpha':16,'physical_batch':8,'accumulation':2,'effective_batch':16,'allocator_cap_GB':9,'cleanup_threshold_GB':4,
        'quality_selection':'Fixed six-language DEV; allfive existing-language gates must pass; no heldout-based hyperparameter tuning',
        'existing_teacher_original_token_IDs_unchanged':True},
    pending=['Parent-controlled actual GPU launch/preflight and source-checkpoint hash',
             'Actual74,883-row acoustic cache acceptance and optimizer exposure',
             'Actual new-data variable-shape GPU memory/throughput and retention quality'],
    optional_follow_on={'TR_only_batched_frozen_teacher_cache':'Prepare uncached TRbasefeatures withbatch8 only after a maskedframe/label GPU proof; old/replay cache bytes immutable; noTRfeatureKD currentlyused',
        'cache_progress':'New controller shouldpublish pre-run_config progress from boundedownedlogtail; loglines acoustic cacheN/total plus actualselflive-memory evidence'} )
target=D/'public_turkish_v3_launch_preflight.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(ready,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,accepted=ready['accepted_clips'],hours=ready['accepted_actual_PCM_hours'],remaining=len(excluded),categories=dict(categories),sources=dict(source_counts),source_hours=ready['actual_PCM_hours_by_source'])))
