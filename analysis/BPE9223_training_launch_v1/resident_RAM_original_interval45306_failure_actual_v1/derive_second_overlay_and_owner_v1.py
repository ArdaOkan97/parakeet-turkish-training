"""Inactive exact second certificate and typed preserved-prefix successor."""
import ast,difflib,hashlib,json
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research';N=Path(__file__).parent
def pin(path):
 path=Path(path).resolve();return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
u=json.loads((N/'actual_failed_update45306_plan646_index5.json').read_text());row=u['rows'][8];r=json.loads((N/'full_member_actual_01/report.json').read_text())
c={'kind':'ASR18_source_bound_one_original_full_clip_geometry_certificate_v1','original_row':r['row'],'original_row_sha256':digest(r['row']),'original_declaration':r['declared_geometry'],'full_target_ids':row['tr_metadata_labels'],'full_target_ids_sha256':digest(row['tr_metadata_labels']),'raw_label_sha256':hashlib.sha256(r['row']['text'].encode()).hexdigest(),'original_TAR':r['original_TAR'],'recording_index_entry':r['recording_index_entry'],'member_sha256':r['member_sha256'],'actual_geometry':r['actual_geometry'],'decoded_frames':r['decoded_frames'],'max_abs_PTS_residual_samples':r['max_abs_PTS_residual_samples'],'native_f32_sha256':r['native_f32_sha256'],'clean_f32_sha256':r['clean_f32_sha256'],'clean_i16_sha256':r['clean_i16_sha256'],'independent_full_member_observation':pin(N/'full_member_actual_01/report.json'),'actual_failure':pin(N/'actual_failure_closure.json'),'original_annotations_indexes_and_plans_unchanged':True,'all_original_physical_samples_once':True,'sampler_order_full_pass_denominator_labels_unchanged':True,'no_generic_EOF_tolerance_or_clamp':True,'historical_mel_frames':1471,'actual_physical_mel_frames':1470,'encoder_frames_unchanged':184}
certificate=N/'second_original_full_clip_geometry_certificate_v1.json';assert not certificate.exists();certificate.write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
oldpath=S/'fastmss_original_clip_geometry_overlay_v1.py';old=oldpath.read_text();tree=ast.parse(old);lines=old.splitlines(True)
fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='checked_certificate');check=''.join(lines[fn.lineno-1:fn.end_lineno])
replacements={'ASR18_source_bound_original_full_clip_geometry_overlay_v1':'ASR18_source_bound_second_original_full_clip_geometry_overlay_v1','15.488':'14.71','82':'97','743424':'706080','247808':'235360','1548':'1471','194':'184','743411':'706073','247804':'235358','775':'736'}
for a,b in replacements.items():check=check.replace(a,b)
# The physical clean count crosses one mel boundary; encoder T and full labels do not change.
check=check.replace('clean16k_samples=235358, mel_frames=1471','clean16k_samples=235358, mel_frames=1470')
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='OriginalRows');call=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='__call__')
method=''.join(lines[call.lineno-1:call.end_lineno]).replace('self.certificate','self.new_certificate').replace('self.overlay_policy','self.new_overlay_policy')
method=method.replace('original_annotations_plan_labels_and_frame_geometry_unchanged=True,','original_annotations_plan_labels_and_encoder_geometry_unchanged=True,\n            historical_mel_frames=declaration[\'mel_frames\'], actual_physical_mel_frames=self.new_certificate[\'actual_geometry\'][\'mel_frames\'],')
text='''"""Second exact source-bound EOF certificate, with unchanged prior/ordinary delegation.

Historical declarations reserve the larger original envelope. This row uses all
physical samples and explicitly reports mel1470 versus declared1471, with T184/U97.
"""
from copy import deepcopy
import fastmss_original_clip_geometry_overlay_v1 as prior
from fastmss_original_clip_geometry_overlay_v1 import core, BoundedOriginalReader, raw_plan, chunked_hashes, canonical_source_path, source_identity_binding
ROW_ID='alania:a9fb5e12943343b05d333b512e1b81bd_00000'
ORDINAL=133836
require,digest,same,record=core.require,core.digest,core.same,core.record
match_original=prior.match_original
decode_full_member=prior.decode_full_member

'''+check+'''

class OriginalRows(prior.OriginalRows):
    def __init__(self,validated,*,geometry_overlay,**arguments):
        self.new_overlay_policy=deepcopy(geometry_overlay)
        self.new_certificate=checked_certificate(geometry_overlay,arguments['guard'])
        require(geometry_overlay['prior_overlay']['source']==record(prior.__file__,arguments['guard']),
                'Exact original overlay remains the authority for its prior sole row')
        super().__init__(validated,geometry_overlay=geometry_overlay['prior_overlay'],**arguments)

'''+method+'\n'
newpath=S/'fastmss_original_clip_geometry_overlay_v2.py';assert not newpath.exists();ast.parse(text);newpath.write_text(text,encoding='utf-8')
outputs={'overlay_source':pin(newpath),'certificate':pin(certificate),'prior_overlay_source':pin(oldpath)}
def successor(oldpath,newpath,replacements,delta_name):
 old=oldpath.read_text();new=old
 for a,b,count in replacements:
  assert new.count(a)==count,(a,new.count(a),count);new=new.replace(a,b)
 assert not newpath.exists();ast.parse(new);newpath.write_text(new,encoding='utf-8');delta=N/delta_name;delta.write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(oldpath),tofile=str(newpath))),encoding='utf-8')
 return pin(newpath),pin(delta)
outputs['stream_source'],outputs['stream_delta']=successor(S/'fastmss_resident_feature_stream_v2.py',S/'fastmss_resident_feature_stream_v3.py',[
 ('from fastmss_original_clip_geometry_overlay_v1 import OriginalRows','from fastmss_original_clip_geometry_overlay_v2 import OriginalRows',1)],'stream2_to3.diff.txt')
path=R/'run_ASR18_BPE9223_resident_RAM_training_v8.py';old=path.read_text();tree=ast.parse(old);lines=old.splitlines(True);fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='checked_original_interval_resume');oldcheck=''.join(lines[fn.lineno-1:fn.end_lineno]);newcheck=oldcheck
for a,b in [('saved32400_tail32405','saved45200_tail45305'),('53376','63811'),('32400','45200'),('32405','45305'),('32401','45201'),('32406','45306'),('five later positive attempts','105 later positive attempts')]:newcheck=newcheck.replace(a,b)
outputs['driver_source'],outputs['driver_delta']=successor(path,R/'run_ASR18_BPE9223_resident_RAM_training_v9.py',[
 (oldcheck,newcheck,1),('from fastmss_resident_feature_stream_v2 import ResidentFeatureProducer','from fastmss_resident_feature_stream_v3 import ResidentFeatureProducer',1),
 ("a.invocation==7,'Actual original-interval TOOL1", "a.invocation==8,'Actual original-interval TOOL1",1),
 ('Exact actual32400 saved prefix, preserved five-tail never replayed','Exact actual45200 saved prefix, preserved105-tail never replayed',1),
 ('ASR18_readonly_saved32400_prefix_after_original_interval_failure_v1','ASR18_readonly_saved45200_prefix_after_original_interval_failure_v1',1)],'driver8_to9.diff.txt')
outputs['wrapper_source'],outputs['wrapper_delta']=successor(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v8.py',R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v9.py',[
 ('run_ASR18_BPE9223_resident_RAM_training_v8','run_ASR18_BPE9223_resident_RAM_training_v9',2),('fastmss_resident_feature_stream_v2.py','fastmss_resident_feature_stream_v3.py',1),('actual abnormal32400 predecessor','actual abnormal45200 predecessor',1)],'wrapper8_to9.diff.txt')
outputs['failure_seal']=pin(N/'actual_failure_closure.json');outputs['independent_full_member_observation']=pin(N/'full_member_actual_01/report.json')
handoff=N/'second_overlay_owner9_inactive_source_handoff.json';handoff.write_text(json.dumps(outputs,indent=2)+'\n',encoding='utf-8');print(json.dumps({'handoff':pin(handoff),**outputs}),flush=True)
