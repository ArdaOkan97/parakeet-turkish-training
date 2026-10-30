import hashlib,json
from pathlib import Path
p=Path(__file__).parent
u=json.loads((p/'actual_failed_update32406_plan391_index5.json').read_text());row=u['rows'][10];r=json.loads((p/'full_member_actual_02/report.json').read_text())
def pin(path):
 path=Path(path).resolve();return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def sha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
c={'kind':'ASR18_source_bound_one_original_full_clip_geometry_certificate_v1','original_row':r['row'],'original_row_sha256':sha(r['row']),'original_declaration':r['declared_geometry'],'full_target_ids':row['tr_metadata_labels'],'full_target_ids_sha256':sha(row['tr_metadata_labels']),'raw_label_sha256':hashlib.sha256(r['row']['text'].encode()).hexdigest(),'original_TAR':r['original_TAR'],'recording_index_entry':r['recording_index_entry'],'member_sha256':r['member_sha256'],'actual_geometry':r['actual_geometry'],'decoded_frames':r['decoded_frames'],'max_abs_PTS_residual_samples':r['max_abs_PTS_residual_samples'],'native_f32_sha256':r['native_f32_sha256'],'clean_f32_sha256':r['clean_f32_sha256'],'clean_i16_sha256':r['clean_i16_sha256'],'independent_full_member_observation':pin(p/'full_member_actual_02/report.json'),'actual_failure':pin(p/'actual_failure_closure.json'),'original_annotations_indexes_and_plans_unchanged':True,'all_original_physical_samples_once':True,'sampler_order_full_pass_denominator_labels_unchanged':True,'no_generic_EOF_tolerance_or_clamp':True}
path=p/'one_original_full_clip_geometry_certificate_v1.json';assert not path.exists();path.write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(pin(path)))
