"""Preparation only: freeze a new metadata observer and exact declared raw inverse."""
import ast,difflib,hashlib,json
from pathlib import Path
F=Path(__file__).resolve().parent;FIX=F.parent/'resident_RAM_recurrent_metadata_slowdown_fix_v1'
old=FIX/'seal_source31_saved167600_recovery_v1.py';new=F/'seal_source33_saved170600_eighth_recovery_v1.py'
raw=old.read_bytes();value=raw.decode();records=[]
def edit(a,b,count=None):
 global value
 a=a.replace('\n','\r\n');b=b.replace('\n','\r\n')
 actual=value.count(a);assert actual>0 and(count is None or actual==count),(a[:120],actual,count)
 value=value.replace(a,b);records.append({'old_literal':a,'new_literal':b,'expected_count':actual})
for a,b in [('167438','170471'),('167437','170470'),('167401','170401'),('167400','170400'),
 ('167620','170620'),('167600','170600'),('338402719','344297457'),('source31','source33'),
 ('CPU11','CPU13'),('owner_000023','owner_000024'),('commits_000023','commits_000024'),
 ('resident_RAM_training_actual_24','resident_RAM_training_actual_25'),('journal23','journal24'),
 ('historical22','historical23'),('original22','original23'),('Original closed22','Original closed23'),
 ('old22','old23'),('no_old_whole_prefix22tail','no_old_whole_prefix23tail'),('ONLY37','ONLY70'),('actual37','actual70'),('new37','new70'),("'new70_count':37","'new70_count':70")]:edit(a,b)
edit('import hashlib,json,math,os,shutil,sys,time','import argparse,gzip,hashlib,io,json,math,os,shutil,sys,time',1)
edit('from resource_watchdog_v2 import sample_memory','from resource_watchdog_v2 import sample_memory\nfrom bpe_broad_production_state_v1 import digest',1)
edit("I=B/'invocations/owner_000024';A=N/'resident_RAM_training_actual_25'","I=B/'invocations/owner_000024';A=N/'resident_RAM_training_actual_25';F=Path(__file__).resolve().parent",1)
edit('def main():','def main(a):',1)
oldreview="review_path=N/'resident_RAM_recurrent_metadata_slowdown_fix_v1/ROOT_source33_fresh_signature_collect_saved170400_final_execution_review_v1.json';review=read_json(review_path)"
edit(oldreview,"review_path=Path(a.ROOT_review);review=read_json(review_path)",1)
edit("review['future_seven_runtime_bindings']","read_json(review['runtime7']['path'])['future_runtime_values']",1)
edit("review['fresh_signature_resume']","review['eighth_original_resume']")
edit("['closed_old_journal_STAT']","['journal']['original_STAT_only']",1)
edit("'long_production_validation_still_pending167420_169020':True","'human_recurrent_speed_fix_still_OPEN_not_validated_by_audio_repair':True",1)
edit("source29_saved170600_compact_recovery_observer","source33_saved170600_compact_recovery_observer",1)
edit("'ASR18_actual_source33_canonical_archive_saved170400_recovered170600_candidate_v1'","'ASR18_actual_source33_eighth_original_interval_saved170400_recovered170600_candidate_v1'",1)
archive=""" # One NEW standard170600 metadata archive, never audio or tensor data.
 archive=checkpoint/'prepared_metadata.json.gz';require(archive.stat().st_size<=8*1024**2,'Bounded new standard archive compressed extent')
 archive_raw=archive.read_bytes();archive_pin={'path':str(archive),'bytes':len(archive_raw),'sha256':hashlib.sha256(archive_raw).hexdigest()}
 with gzip.GzipFile(fileobj=io.BytesIO(archive_raw),mode='rb')as zipped:uncompressed=zipped.read(32*1024**2+1)
 require(len(uncompressed)<=32*1024**2,'Bounded new standard metadata JSON, no whole tensor/audio')
 guard();prepared=json.loads(uncompressed)
 require(len(prepared)==200 and [x['native']['update']for x in prepared]==list(range(170401,170601)),
  'All200 standard archive records retain their original ordered actual update labels')
 selected=prepared[69];metadata=selected['metadata'];native=selected['native'];row=metadata['rows'][5]
 reference=metadata['same_pass_clean_evidence'][5];certificate_pin=review['original_clip_geometry_overlay']['certificate'];certificate=read_json(certificate_pin['path'])
 require(native['update']==170470 and row['source_id']==native['source_ids'][5]==certificate['original_row']['unique_id']
  and row['full_target_ids']==reference['full_target_ids']==certificate['full_target_ids']and row['target_tokens']==99
  and row['samples']==273624 and row['encoder_frames']==214
  and row['clean_PCM_sha256']=='3a9e9240d7348caf1b2a26f53b98d362836ec6cad088a747cc31008a0cf9659b'
  and row['input_features']['shape']==[1,1711,128]and row['attention_mask']['shape']==[1,1711],
  'Actual repaired170470 row6 original full99/clean273624/hash/feature geometry reached native')
 proof=reference['before_draw_geometry_proof']
 require(reference['whole_original_clip']is True and reference['actual_geometry_and_heldout_checked']is True
  and proof['certificate']==certificate_pin and proof['source']==review['original_clip_geometry_overlay']['source']
  and proof['historical_declaration']==certificate['original_declaration']
  and proof['actual_geometry']==certificate['actual_geometry']and proof['all_original_physical_samples_used_once']is True
  and proof['original_annotations_plan_labels_and_encoder_geometry_unchanged']is True
  and proof['historical_declaration']['native_samples']==820896 and proof['historical_declaration']['clean16k_samples']==273632,
  'Actual eighth certificate/alias9/full original physical PCM and unchanged larger historical reservations')
 require(digest(metadata)==events[-1]['prepared_evidence']['metadata_sha256'],'Actual complete preparation metadata matches directnew70 commit')
 raw_native_JSON_digest=digest(native)
 routes=native['joint_work_unit_allocation_routes']
 require(type(routes)is dict and set(routes)=={str(i)for i in range(16)},'Only known original16 integer routekeys may be restored')
 integer_native=dict(native);integer_native['joint_work_unit_allocation_routes']={int(k):v for k,v in routes.items()}
 require(digest(integer_native)==events[-1]['native']['full_native_receipt_sha256'],
  'Exact full native receipt after ONLY original integer routekeys restored; raw JSON digest remains distinct')
 corrected={'global_update':170470,'archive_index':69,'consumed_row_index':6,'original_ordinal':332560,
  'row':row,'same_pass_clean_reference':reference,'full_native_receipt':native,
  'raw_native_JSON_string_routekeys_digest':raw_native_JSON_digest,
  'original_integer_routekeys_native_commit_digest':digest(integer_native),
  'metadata_digest':digest(metadata),'archive':archive_pin,'archive_uncompressed_bytes':len(uncompressed),
  'archive_uncompressed_sha256':hashlib.sha256(uncompressed).hexdigest(),'original_historical_declared_native_samples':820896,
  'original_historical_declared_clean_samples':273632,'no_freed_capacity_credit':True}
 write_json(out/'actual170470_row6_preparation_and_native.json',corrected)
"""
edit(" require('numpy'not in sys.modules and 'torch'not in sys.modules,'No payload/model/NumPy/Torch/GPU imports')",archive+" require('numpy'not in sys.modules and 'torch'not in sys.modules,'No payload/model/NumPy/Torch/GPU imports')",1)
edit("'actual70_new_snapshot':record(snapshot),'sealer_source':record(__file__)","'actual70_new_snapshot':record(snapshot),'actual170470_row6':record(out/'actual170470_row6_preparation_and_native.json'),'standard170600_preparation_archive':archive_pin,'sealer_source':record(__file__)",1)
edit("'ROOT_actual_recovery_acceptance_pending':True,","'repaired170470_row6_production_evidence':corrected,'ROOT_actual_recovery_acceptance_pending':True,",1)
edit("if __name__=='__main__':main()","if __name__=='__main__':\n p=argparse.ArgumentParser();p.add_argument('--ROOT-review',required=True);main(p.parse_args())",1)
ast.parse(value);inverse=value
for e in reversed(records):
 assert inverse.count(e['new_literal'])==e['expected_count'],e['new_literal'][:90]
 inverse=inverse.replace(e['new_literal'],e['old_literal'])
assert inverse.encode()==raw;assert not new.exists();new.write_bytes(value.encode())
def pin(p):
 b=p.read_bytes();return {'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
ledger=F/'source31_to33_natural_recovery_metadata_sealer_entire_RAW_inverse_v1.json'
assert not ledger.exists();ledger.write_bytes((json.dumps({'old_source':pin(old),'new_source':pin(new),
 'newline_normalization':'NONE; exact CRLF retained','replacements_in_forward_order':records,
 'full_raw_forward_inverse_exact':True,'new_archive_read_bound':{'compressed_bytes':8388608,'uncompressed_bytes':33554432,'number':1,
 'only':'new standard170600 metadata archive / actual170470 row6 / no audio tensor data'},
 'only_new_journal24_range':[170401,170470],'source23_original_STAT_only':True},indent=2)+'\n').encode())
diff=F/'source31_to33_natural_recovery_metadata_sealer_v1.diff';assert not diff.exists();diff.write_bytes(''.join(difflib.unified_diff(raw.decode().splitlines(True),value.splitlines(True),fromfile=str(old),tofile=str(new))).encode())
print(json.dumps({'sealer':pin(new),'inverse':pin(ledger)}),flush=True)
