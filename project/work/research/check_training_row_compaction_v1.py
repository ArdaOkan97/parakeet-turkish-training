"""Small stdlib object/values proof, no actual corpus/model/ML run."""
from pathlib import Path
import copy
import hashlib
import json
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(HERE))
import compact_training_row_objects_v1 as m
checks=[]
def check(ok,name):assert ok,name;checks.append(name)
def reject(callback,name):
    try:callback()
    except ValueError:checks.append(name);return
    raise AssertionError(name+' accepted')
raw={'dataset':'issai/Turkish_Speech_Corpus','dataset_revision':'d79ea01392882322e8f46657cb635236af25b6d4',
 'dataset_config':'tr','dataset_split':'train','dataset_file':'train.tsv','language':'tr','gender':'unknown',
 'training_role':'train','source':'TSC','source_id':'TSC_1','text':'Merhaba dünya','reference':'Merhaba dünya',
 'reference_normalized':'Merhaba dünya','audio_filepath':'unique/fixture1.wav','duration':1.25,
 'encoder_frames':16,'target_tokens':4,'tr_metadata_labels':[8201,8201,1000,3],'bpe512_TR_admission_complete':True}
rows=[json.loads(json.dumps({**raw,'source_id':'TSC_'+str(i),'audio_filepath':'unique/'+str(i)+'.wav'},ensure_ascii=False)) for i in range(3)]
old=copy.deepcopy(rows); row_ids=[id(row) for row in rows]; label_ids=[id(row['tr_metadata_labels']) for row in rows]
pool_id=id(rows);key_orders=[tuple(row) for row in rows];rng=random.getstate()
e=m.compact_training_rows(rows,expected_rows=3)
check(rows==old and id(rows)==pool_id and [id(row) for row in rows]==row_ids,'all_values_types_same_and_exact_original_pool_row_objects')
check([id(row['tr_metadata_labels']) for row in rows]==label_ids and [tuple(row) for row in rows]==key_orders,'original_label_lists_and_key_orders_preserved')
check(e['ordered_row_canonical_sha256_before']==e['ordered_row_canonical_sha256_after'] and e['all_rows_canonical_equal'],'streaming_ordered_per_row_hashes_exact')
check(all(next(k for k in rows[0] if k==key) is next(k for k in rows[1] if k==key) for key in rows[0]),'actual_dictionary_keys_shared_across_rows')
check(rows[0]['tr_metadata_labels'][0] is rows[0]['tr_metadata_labels'][1] is rows[1]['tr_metadata_labels'][0],'actual_immutable_8711_integer_pool_shared_in_existing_lists')
check(rows[0]['dataset'] is rows[1]['dataset'] and rows[0]['reference'] is rows[0]['text'],'bounded_metadata_and_only_equal_text_fields_share_objects')
check(random.getstate()==rng and not any(n in sys.modules for n in ('torch','nemo','transformers')),'no_rng_or_ML_execution')
check(e['estimated_duplicate_object_bytes_eliminated']>0 and e['estimated_bytes_are_measured_process_memory'] is False,'estimated_object_references_honestly_distinct_from_actual_RAM')
for bad in (True,8192,8711,-1):
    x=copy.deepcopy(old);x[0]['tr_metadata_labels'][0]=bad
    reject(lambda x=x:m.compact_training_rows(x,expected_rows=3),'unsafe_label_rejection_'+str(bad))
reject(lambda:m.compact_training_rows(old,expected_rows=253798),'partial_pool_rejected')
x=copy.deepcopy(old);x[0]['tr_metadata_labels']=tuple(x[0]['tr_metadata_labels'])
reject(lambda:m.compact_training_rows(x,expected_rows=3),'tuple_label_cannot_be_silently_retyped')
x=copy.deepcopy(old);x[0]['bpe512_TR_admission_complete']=False
reject(lambda:m.compact_training_rows(x,expected_rows=3),'partial_candidate_admission_rejected')
out=HERE/'training_row_object_compaction_stdlib_verification_v1.json'
report={'kind':'inactive_training_row_object_compaction_stdlib_verification_v1','passed':True,'assertions':len(checks),'checks':checks,
 'helper_sha256':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),'small_synthetic_rows':3,
 'real_pool_or_process_memory_measured':False,'Torch_GPU_corpus_or_checkpoints_read':False,'compaction_evidence':e}
with out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'passed':True,'assertions':len(checks),'helper_sha256':report['helper_sha256'],'report_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
