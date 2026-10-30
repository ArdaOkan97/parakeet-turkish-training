"""Preserve metadata generator1 failure; exact import matching and CPU fixture."""
from pathlib import Path
import ast,hashlib,json
N=Path(__file__).parent
old=N/'derive_owner16_and_stream7_v1.py';text=old.read_text(encoding='utf-8')
before="assert inverse==original and not new.exists();new.write_text(text,encoding='utf-8')"
after="assert inverse==original\n    if new.exists():assert new.read_text(encoding='utf-8')==text\n    else:new.write_text(text,encoding='utf-8')"
assert text.count(before)==1;text=text.replace(before,after)
before="('fastmss_resident_feature_stream_v6','fastmss_resident_feature_stream_v7',2),"
after="('from fastmss_resident_feature_stream_v6 import checked_resource_profile','from fastmss_resident_feature_stream_v7 import checked_resource_profile',1),\n    ('from fastmss_resident_feature_stream_v6 import ResidentFeatureProducer','from fastmss_resident_feature_stream_v7 import ResidentFeatureProducer',1),"
assert text.count(before)==1;text=text.replace(before,after)
before="('run_ASR18_BPE9223_resident_RAM_training_v15','run_ASR18_BPE9223_resident_RAM_training_v16',2),"
assert text.count(before)==1;text=text.replace(before,before.replace(',2)',',3)'))
ast.parse(text);fixed=N/'derive_owner16_and_stream7_v2.py';assert not fixed.exists();fixed.write_text(text,encoding='utf-8')
old=N.parent/'resident_RAM_original_interval_post88426_failure_actual_v1/verify_fourth_terminal_row_ingress_frontend_v1.py';t=old.read_text(encoding='utf-8')
for before,after in [('fastmss_resident_feature_stream_v5','fastmss_resident_feature_stream_v7'),
    ('fastmss_original_clip_geometry_overlay_v4','fastmss_original_clip_geometry_overlay_v5'),('decode_terminal_interval','decode_full_member'),
    ('resident_RAM_original_interval_post47727_failure_actual_v1/production_configuration_v1/configuration.json','resident_RAM_single_packet_resource_source_v1/configuration_candidate.json'),
    ('fourth_original_terminal_interval_geometry_certificate_v1.json','fifth_original_full_clip_geometry_certificate_v1.json'),
    ('reader.fourth_certificate','reader.fifth_certificate'),('normalized[17]','normalized[9]'),("['rows'][13]","['rows'][3]"),
    ("['contexts'][13]","['contexts'][3]"),('alania:e4a162635e1cf1f6599a40b38b8b7a57_00004','alania:9105d11d7d12c9c166028f98935dc107_00000'),
    ("read_json(N/'actual_failure_closure.json')['actual_current_preparation_plan']","read_json(N/'actual_ordered_remaining_plan_metadata.json')['source_plan']"),
    ('clean16k_samples\':120048','clean16k_samples\':200064'),('valid_mel_frames\':750','valid_mel_frames\':1250'),
    ('physical_feature_frames\':751','physical_feature_frames\':1251'),('encoder_frames\':94','encoder_frames\':157'),
    ("==39","==82"),("==750","==1250"),('physical mel750','physical mel1250'),('53.150','12.503'),
    ("'row_index':13,'update_index':17","'row_index':3,'update_index':9")]:
    assert before in t,before;t=t.replace(before,after)
before=" base['prior_resident_RAM_original_interval_failure']=record(N/'actual_failure_closure.json')"
after=before+"\n base['resident_RAM_resource_scheduling_policy']['CPU_protocol_source']=record(S/'fastmss_resident_feature_stream_v7.py')"
assert before in t;t=t.replace(before,after)
before="original.pop('prior_resident_RAM_original_interval_failure');require(unchanged==original,'Only two operational fields changed')"
after="original.pop('prior_resident_RAM_original_interval_failure');unchanged['resident_RAM_resource_scheduling_policy']['CPU_protocol_source']=original['resident_RAM_resource_scheduling_policy']['CPU_protocol_source'];require(unchanged==original,'Only overlay/failure fields plus exact new CPU protocol source pin; resource values unchanged')"
assert before in t;t=t.replace(before,after)
before=" bad=copy.deepcopy(certificate);bad['member_sha256']='0'*64"
after=""" altered_row=copy.deepcopy(raw);altered_row['ordinal']+=1
 rejects('changed_original_row',lambda:match_original(certificate,altered_row,d,expected))
 bad_source=copy.deepcopy(certificate);bad_source['original_TAR']['recorded_sha256']='0'*64
 rejects('changed_original_source_SHA',lambda:decode_full_member(reader.broad,raw,bad_source,guard=guard,channels_bound=8,frame_samples_bound=16384))
 bad=copy.deepcopy(certificate);bad['member_sha256']='0'*64"""
assert before in t;t=t.replace(before,after)
before=" reader.close();reader=None"
after=""" import fastmss_original_clip_geometry_overlay_v5 as fifth
 original_call=fifth.prior.OriginalRows.__call__;delegated=[];sentinel=object()
 def delegation_probe(self,r,e,p):delegated.append((r,e,p));return sentinel
 try:
  fifth.prior.OriginalRows.__call__=delegation_probe
  prior_row={'ordinal':fifth.prior.ORDINAL,'source_id':fifth.prior.ROW_ID};prior_expected={}
  require(reader(prior_row,prior_expected,update['phase'])is sentinel and len(delegated)==1
      and delegated[0][0]is prior_row and delegated[0][1]is prior_expected,'Exact prior4 route delegation, metadata fixture only')
 finally:fifth.prior.OriginalRows.__call__=original_call
 require(base['original_clip_geometry_overlay']['prior_overlay']==read_json(N.parent/'resident_RAM_single_packet_resource_source_v1/configuration_candidate.json')['original_clip_geometry_overlay'],'Exact unchanged prior4 policy tree')
 previous_policy_delegation={'passed':True,'metadata_only_call_route_control':True,'no_old_row_audio_replayed':True,'parent_method_restored':True,'prior4_policy_exact':True}
 reader.close();reader=None"""
assert t.count(before)==1;t=t.replace(before,after)
t=t.replace("'negative_controls':negative,'actual_native_updates'","'negative_controls':negative,'previous_policy_delegation':previous_policy_delegation,'actual_native_updates'")
ast.parse(t);fixture=N/'verify_fifth_full_clip_ingress_frontend_v1.py';assert not fixture.exists();fixture.write_text(t,encoding='utf-8')
def pin(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
print(json.dumps({'corrected_owner_generator':pin(fixed),'CPU_fixture':pin(fixture)}),flush=True)
