"""Inactive third exact certificate, prior delegation and truthful masked-error entry."""
import ast,difflib,hashlib,json
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research';N=Path(__file__).parent
def pin(p):
 p=Path(p).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
u=json.loads((N/'probe_actual_01/actual_failed_update_metadata.json').read_text());row=u['rows'][2];r=json.loads((N/'full_member_actual_01/report.json').read_text())
c={'kind':'ASR18_source_bound_one_original_full_clip_geometry_certificate_v1','original_row':r['row'],'original_row_sha256':digest(r['row']),'original_declaration':r['declared_geometry'],'full_target_ids':row['tr_metadata_labels'],'full_target_ids_sha256':digest(row['tr_metadata_labels']),'raw_label_sha256':hashlib.sha256(r['row']['text'].encode()).hexdigest(),'original_TAR':r['original_TAR'],'recording_index_entry':r['recording_index_entry'],'member_sha256':r['member_sha256'],'actual_geometry':r['actual_geometry'],'decoded_frames':r['decoded_frames'],'max_abs_PTS_residual_samples':r['max_abs_PTS_residual_samples'],'native_f32_sha256':r['native_f32_sha256'],'clean_f32_sha256':r['clean_f32_sha256'],'clean_i16_sha256':r['clean_i16_sha256'],'independent_full_member_observation':pin(N/'full_member_actual_01/report.json'),'actual_failure':pin(N/'actual_failure_closure.json'),'ordered_failure_localization':pin(N/'probe_actual_01/report.json'),'original_annotations_indexes_and_plans_unchanged':True,'all_original_physical_samples_once':True,'sampler_order_full_pass_denominator_labels_unchanged':True,'no_generic_EOF_tolerance_or_clamp':True,'historical_mel_frames':1497,'actual_physical_mel_frames':1497,'encoder_frames_unchanged':188}
path=N/'third_original_full_clip_geometry_certificate_v1.json';assert not path.exists();path.write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
outputs={'certificate':pin(path)}
def successor(oldpath,newpath,replacements,delta_name):
 old=oldpath.read_text();new=old
 for a,b,count in replacements:
  assert new.count(a)==count,(a,new.count(a),count);new=new.replace(a,b)
 assert not newpath.exists();ast.parse(new);newpath.write_text(new,encoding='utf-8');delta=N/delta_name;delta.write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(oldpath),tofile=str(newpath))),encoding='utf-8')
 return pin(newpath),pin(delta)
oldpath=S/'fastmss_original_clip_geometry_overlay_v2.py';old=oldpath.read_text();new=old
for a,b in [('Second exact','Third exact'),('second_original','third_original'),('overlay_v1','overlay_v2'),('a9fb5e12943343b05d333b512e1b81bd','2c21c91407c3dc1cb4111da29af1c53e'),('133836','382298'),('14.71','14.977'),('706080','718896'),('235360','239632'),('706073','718890'),('235358','239630'),('1471','1497'),('1470','1497'),('184','188'),('97','91'),('736','750'),('self.new_overlay_policy','self.third_overlay_policy'),('self.new_certificate','self.third_certificate'),('its prior sole row','its two prior certified rows')]:new=new.replace(a,b)
newpath=S/'fastmss_original_clip_geometry_overlay_v3.py';assert not newpath.exists();ast.parse(new);newpath.write_text(new,encoding='utf-8');delta=N/'overlay2_to3.diff.txt';delta.write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(oldpath),tofile=str(newpath))),encoding='utf-8');outputs.update(overlay_source=pin(newpath),overlay_delta=pin(delta))
outputs['stream_source'],outputs['stream_delta']=successor(S/'fastmss_resident_feature_stream_v3.py',S/'fastmss_resident_feature_stream_v4.py',[
 ('from fastmss_original_clip_geometry_overlay_v2 import OriginalRows','from fastmss_original_clip_geometry_overlay_v3 import OriginalRows',1)],'stream3_to4.diff.txt')
path=R/'run_ASR18_BPE9223_resident_RAM_training_v9.py';old=path.read_text();tree=ast.parse(old);lines=old.splitlines(True);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='checked_original_interval_resume');check=''.join(lines[fn.lineno-1:fn.end_lineno]);newcheck=check
for a,b in [('saved45200_tail45305','saved47600_tail47727'),('63811','39651'),('45200','47600'),('45305','47727'),('45201','47601'),('45306','47728'),('105 later','127 later')]:newcheck=newcheck.replace(a,b)
newcheck=newcheck.replace("'failed_preparation_plan','failed_update_metadata'","'actual_current_preparation_plan','unsubmitted_next_plan','failed_update_metadata'")
newcheck=newcheck.replace("require('Original interval beyond source unresolved' in failure['error']", "require('Distinct ordered plan on healthy producer' in failure['error']\n        and 'Original interval beyond source unresolved' in Path(value['original_CPU_stderr']['path']).read_text(encoding='utf-8')")
newcheck=newcheck.replace("'Actual nonresource CPU EOF failure and unchanged saved journal authority'","'Actual submit guard masked already failed ordinary CPU decoder; unchanged saved journal authority'")
outputs['driver_source'],outputs['driver_delta']=successor(path,R/'run_ASR18_BPE9223_resident_RAM_training_v10.py',[
 (check,newcheck,1),('from fastmss_resident_feature_stream_v3 import ResidentFeatureProducer','from fastmss_resident_feature_stream_v4 import ResidentFeatureProducer',1),
 ("a.invocation==8,'Actual original-interval TOOL1","a.invocation==9,'Actual original-interval TOOL1",1),
 ('Exact actual45200 saved prefix, preserved105-tail never replayed','Exact actual47600 saved prefix, preserved127-tail never replayed',1),
 ('ASR18_readonly_saved45200_prefix_after_original_interval_failure_v1','ASR18_readonly_saved47600_prefix_after_original_interval_failure_v1',1)],'driver9_to10.diff.txt')
outputs['wrapper_source'],outputs['wrapper_delta']=successor(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v9.py',R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v10.py',[
 ('run_ASR18_BPE9223_resident_RAM_training_v9','run_ASR18_BPE9223_resident_RAM_training_v10',2),('fastmss_resident_feature_stream_v3.py','fastmss_resident_feature_stream_v4.py',1),('actual abnormal45200 predecessor','actual abnormal47600 predecessor',1)],'wrapper9_to10.diff.txt')
outputs.update(failure_seal=pin(N/'actual_failure_closure.json'),independent_full_member_observation=pin(N/'full_member_actual_01/report.json'))
path=N/'third_overlay_owner10_inactive_source_handoff.json';path.write_text(json.dumps(outputs,indent=2)+'\n',encoding='utf-8');print(json.dumps({'handoff':pin(path),**outputs}),flush=True)
