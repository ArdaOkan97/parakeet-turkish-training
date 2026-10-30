"""Freeze the exact seventh physical-source overlay and one inactive CPU control."""
import ast,copy,difflib,hashlib,json,re
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';F=Path(__file__).parent;N=F.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def canonical(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def write(p,v):
 p=Path(p);assert not p.exists(),p;p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
physical=read(F/'full_member_actual_01/report.json');failed=read(F/'probe_actual_01/actual_failed_update_metadata.json');row=physical['row'];item=failed['rows'][6]
require=__import__('builtins').__dict__.get('x',None)
oldcert=read(N/'resident_RAM_original_interval_post134529_failure_actual_v1/fifth_original_full_clip_geometry_certificate_v1.json')
cert=copy.deepcopy(oldcert)
for key,value in {'original_row':row,'original_row_sha256':digest(row),'original_declaration':physical['declared_geometry'],
 'full_target_ids':physical['full_target_ids'],'full_target_ids_sha256':digest(physical['full_target_ids']),
 'raw_label_sha256':hashlib.sha256(row['text_raw'].encode('utf-8')).hexdigest(),
 'original_TAR':physical['original_TAR'],'recording_index_entry':physical['recording_index_entry'],'member_sha256':physical['member_sha256'],
 'actual_geometry':{k:physical['actual_interval_geometry'][k]for k in ('native_sr','native_samples','clean16k_samples','mel_frames','encoder_frames','target_tokens')},
 'decoded_frames':physical['decoded_frames'],'max_abs_PTS_residual_samples':physical['maximum_abs_PTS_residual_samples'],
 'native_f32_sha256':physical['full_source_native_f32_sha256'],'clean_f32_sha256':physical['clean_f32_sha256'],'clean_i16_sha256':physical['clean_i16_sha256'],
 'independent_full_member_observation':pin(F/'full_member_actual_01/report.json'),'actual_failure':pin(F/'actual_failure_closure.json'),
 'ordered_failure_localization':pin(F/'probe_actual_01/report.json'),'actual_causal_CPU_closure':pin(F/'full_member_actual_01_closure.json'),
 'historical_native_reservation_retained':693264,'actual_physical_native_samples':693255}.items():cert[key]=value
cert.pop('unchanged_clean200064_mel1250_features1251_T157_U82')
cert.update(historical_clean_reservation_retained=231088,actual_full_clean_samples=231085,
 unchanged_mel1444_features1445_T181_and_all61_targets=True,actual_raw_codec_side_data=physical['packet_side_data'],
 independent_physical_ROOT_review=pin(N/'ROOT_post142894_exact_head_member_physical_CPU_actual_review_v1.json'))
assert len(cert['full_target_ids'])==61 and cert['original_declaration']['native_samples']==693264 and cert['actual_geometry']['native_samples']==693255
assert cert['original_declaration']['clean16k_samples']==231088 and cert['actual_geometry']['clean16k_samples']==231085
certpath=F/'seventh_original_full_clip_geometry_certificate_v1.json';write(certpath,cert)
base=S/'fastmss_original_clip_geometry_overlay_v5.py';old=base.read_text(encoding='utf-8');new=old.replace('Fifth','Seventh').replace('fifth','seventh')
new=new.replace('fastmss_original_clip_geometry_overlay_v4','fastmss_original_clip_geometry_overlay_v6')
new=new.replace('alania:9105d11d7d12c9c166028f98935dc107_00000',row['unique_id']).replace('four prior certified rows','six prior certified rows')
for x,y in {'417094':'210880','12.504':'14.443','600192':'693264','600190':'693255','200064':'231088','1250':'1444','157':'181','82':'61','626':'723'}.items():new=re.sub(r'(?<![\w.])'+re.escape(x)+r'(?![\w.])',y,new)
needle='native_samples=693255,\n            clean16k_samples=231088';assert needle in new;new=new.replace(needle,'native_samples=693255,\n            clean16k_samples=231085')
new=new.replace('mel1250 versus declared1250, with T157/U82','mel1444 versus declared1444, with T181/U61')
ast.parse(new);overlay=S/'fastmss_original_clip_geometry_overlay_v7.py';assert not overlay.exists();overlay.write_text(new,encoding='utf-8')
diff=F/'head_overlay5_to7_exact_row_and_prior6.diff.txt';diff.write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(base),tofile=str(overlay))),encoding='utf-8')
cpuold=S/'fastmss_resident_feature_stream_v8.py';raw=cpuold.read_bytes();needle=b'from fastmss_original_clip_geometry_overlay_v6 import OriginalRows';assert raw.count(needle)==1
cpu=S/'fastmss_resident_feature_stream_v9.py';assert not cpu.exists();cpu.write_bytes(raw.replace(needle,b'from fastmss_original_clip_geometry_overlay_v7 import OriginalRows'))
assert cpu.read_bytes().replace(b'from fastmss_original_clip_geometry_overlay_v7 import OriginalRows',needle)==raw
baseconfig=N/'resident_RAM_original_interval_post140236_failure_actual_v1/production_configuration_v1/configuration.json'
config=read(baseconfig);prior=copy.deepcopy(config);config['original_clip_geometry_overlay']={'source':pin(overlay),'certificate':pin(certpath),'ROOT_review':None,'prior_overlay':copy.deepcopy(prior['original_clip_geometry_overlay'])}
config['prior_resident_RAM_original_interval_failure']=pin(F/'actual_failure_closure.json');config['resident_RAM_resource_scheduling_policy']['CPU_protocol_source']=pin(cpu)
template=F/'configuration_candidate_template.json';write(template,config)
fixturebase=N/'resident_RAM_original_interval_post140236_failure_actual_v1/verify_sixth_terminal_interval_ingress_frontend_v1.py';t=fixturebase.read_text(encoding='utf-8')
changes=[('fastmss_resident_feature_stream_v8','fastmss_resident_feature_stream_v9'),('fastmss_original_clip_geometry_overlay_v6','fastmss_original_clip_geometry_overlay_v7'),
 ('decode_terminal_interval','decode_full_member'),('resident_RAM_periodic_quiescence_plan_representation_source_v1/configuration_candidate.json','resident_RAM_original_interval_post140236_failure_actual_v1/production_configuration_v1/configuration.json'),
 ('sixth_original_terminal_interval_geometry_certificate_v1.json','seventh_original_full_clip_geometry_certificate_v1.json'),('actual_failure_closure_v2.json','actual_failure_closure.json'),
 ('normalized[16]','normalized[14]'),("['rows'][4]","['rows'][6]"),("['contexts'][4]","['contexts'][6]"),('alania:38198deb36bb10f9e58129b2c9d100fe_00003',row['unique_id']),
 ('308496','231085'),('1928','1444'),('1929','1445'),('241','181'),('121','61'),('sixth_certificate','seventh_certificate'),('1473.712','14.442'),
 (' as sixth',' as seventh'),('sixth.prior','seventh.prior'),('policy5','policy6'),('cert5','cert6'),('len(prior_rows)==5','len(prior_rows)==6'),
 ('prior5','prior6'),('five prior','six prior'),("'row_index':4,'update_index':16","'row_index':6,'update_index':14")]
for x,y in changes:
 assert x in t,x;t=t.replace(x,y)
extra=" require(reference['before_draw_geometry_proof']['historical_declaration']['clean16k_samples']==231088 and reference['before_draw_geometry_proof']['actual_geometry']['clean16k_samples']==231085,'Explicit unchanged historical larger clean231088 versus all-physical clean231085; no padding/cropping')\n"
needle=' actual_reference=copy.deepcopy(reference);del arrays,wave';assert needle in t;t=t.replace(needle,extra+needle)
ast.parse(t);fixture=F/'verify_seventh_full_clip_ingress_frontend_v1.py';assert not fixture.exists();fixture.write_text(t,encoding='utf-8')
fdiff=F/'sixth_to_seventh_frontend_fixture.diff.txt';fdiff.write_text(''.join(difflib.unified_diff(fixturebase.read_text(encoding='utf-8').splitlines(True),t.splitlines(True),fromfile=str(fixturebase),tofile=str(fixture))),encoding='utf-8')
undo=copy.deepcopy(config)
for k in ('original_clip_geometry_overlay','prior_resident_RAM_original_interval_failure','resident_RAM_resource_scheduling_policy'):undo[k]=prior[k]
assert undo==prior and config['original_clip_geometry_overlay']['prior_overlay']==prior['original_clip_geometry_overlay']
write(F/'seventh_overlay_certificate_and_CPU_control_inactive_handoff_v1.json',{'kind':'ASR18_INACTIVE_seventh_exact_fullclip_overlay_CPU9_and_real_frontend_scope_handoff_v1',
 'scope':'ASR18_source_bound_seventh_original_full_clip_geometry_overlay_v1','overlay_source':pin(overlay),'certificate':pin(certpath),'overlay_base':pin(base),'overlay_delta':pin(diff),
 'CPU_source':pin(cpu),'prior_CPU_source':pin(cpuold),'CPU_exact_inverse_one_overlay_import':True,'configuration_template':pin(template),'proper_source_base':pin(baseconfig),
 'only3_operational_configfields_changed':True,'prior6_policy_and_all_resource_values_exact':True,'CPU_control_source':pin(fixture),'CPU_control_base':pin(fixturebase),'CPU_control_delta':pin(fdiff),
 'independent_actual_physical_report':pin(F/'full_member_actual_01/report.json'),'independent_actual_physical_closure':pin(F/'full_member_actual_01_closure.json'),
 'expected_actual_ingress_not_yet_executed':{'clean_samples':231085,'historical_clean_reservation':231088,'finite_features_shape':[1,1445,128],'mask_valid':1444,'T':181,'full_targets':61},
 'strict_negatives':8,'prior6_and_ordinary_delegation_metadata_only_method_restored':True,
 'limits':{'CPU_private_bytes':2000000000,'wall_seconds':180,'full_extra_bytes':268435456,'commit_headroom_bytes':2000000000,'physical_headroom_bytes':1500000000,'E_reserve_bytes':100000000000,'CUDA_VISIBLE_DEVICES':'-1','all_four_thread_caps':1,'priority':'BelowNormal'},
 'source_and_CPU_scope_required_before_fixture':True,'no_frontend_or_GPU_production_admission':True,'prior6_sources_unchanged':True})
print(json.dumps({'overlay':pin(overlay),'certificate':pin(certpath),'CPU9':pin(cpu),'fixture':pin(fixture),'handoff':pin(F/'seventh_overlay_certificate_and_CPU_control_inactive_handoff_v1.json')}),flush=True)
