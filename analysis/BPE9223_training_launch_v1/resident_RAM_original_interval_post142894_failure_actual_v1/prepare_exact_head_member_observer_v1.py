"""Freeze an independent exact full-start0 member observation; no execution."""
import ast,difflib,hashlib,json
from pathlib import Path
F=Path(__file__).parent;N=F.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
base=N/'resident_RAM_original_interval_post134529_failure_actual_v1/observe_complete_member134530_EOF_v1.py'
api=N/'resident_RAM_original_interval_post140236_failure_actual_v1/observe_exact_terminal140237_streamed_member_EOF_v2.py'
old=base.read_text(encoding='utf-8');a=api.read_text(encoding='utf-8');new=old
changes=[('134530','142895'),('rows\'][3]','rows\'][6]'),('alania:9105d11d7d12c9c166028f98935dc107_00000','alania:0ce0d55b73c946f80abea8bbe41f3b27_00000'),
 ('12.504','14.443'),("'0027'","'0014'"),('600192','693264'),('==82','==61'),('audio/0027.tar','audio/0014.tar'),('yodas3_tar_0027.sqlite','yodas3_tar_0014.sqlite'),
 ('frames==626 and count==600190','frames==723 and count==693255'),('len(clean)==200064','len(clean)==231085'),("geometry['mel_frames']==1250","geometry['mel_frames']==1444"),
 ("geometry['encoder_frames']==157","geometry['encoder_frames']==181"),("'update_index':9,'row_index':3","'update_index':14,'row_index':6"),
 ("'declared_native_minus_physical_native_samples':2","'declared_native_minus_physical_native_samples':9"),
 ('True full source retains historical clean/frontend encoder geometry','Actual full source clean count and unchanged mel/encoder geometry are distinct from historical larger declaration')]
for x,y in changes:
 assert x in new,x;new=new.replace(x,y)
new=new.replace('import hashlib,json','import ctypes,hashlib,json').replace('from collections import deque','from ctypes import wintypes\nfrom collections import deque')
api_helper=a[a.index('packet_side_read_observations=[]'):a.index('try:\n numpy_runtime(frontend=False)')]
assert '\ntry:\n    numpy_runtime(frontend=False)'in new;new=new.replace('\ntry:\n    numpy_runtime(frontend=False)','\n'+api_helper+'try:\n    numpy_runtime(frontend=False)')
api_block=a[a.index(' packet_api_provenance='):a.index(' failed_update_pin=')]
api_block='\n'.join('   '+line if line else line for line in api_block.splitlines())+'\n'
needle="    require(av.__version__=='16.0.1','Exact pinned PyAV')\n";assert needle in new
inputs={'failed_update':pin(F/'probe_actual_01/actual_failed_update_metadata.json'),'prior_probe':pin(F/'probe_actual_01/report.json')}
binding="    failed_update_pin="+repr(inputs['failed_update'])+"\n    prior_probe_pin="+repr(inputs['prior_probe'])+"\n    require(record(failed_update_pin['path'])==failed_update_pin and record(prior_probe_pin['path'])==prior_probe_pin,'Exact failed update and completed ordered probe')\n"
new=new.replace(needle,needle+api_block+binding)
needle="            for sd in packet.iter_sidedata():side.append({'packet':packets,'type':str(sd.data_type),'data_size':sd.data_size,'description':str(sd.data_desc)})"
assert needle in new
side="""            for sd in packet.iter_sidedata():
                require(len(side)<16 and sd.data_size<=4096,'Bounded exact codec side metadata')
                payload=packet_side_bytes(sd)
                observed={'packet':packets,'type':str(sd.data_type),'data_size':sd.data_size,'description':str(sd.data_desc),'raw_hex':payload.hex()}
                if str(sd.data_desc)=='Skip Samples'and len(payload)==10:
                    skip,discard,start_reason,end_reason=struct.unpack('<IIBB',payload);observed.update(skip_start_samples=skip,discard_end_samples=discard,skip_reason=start_reason,discard_reason=end_reason)
                side.append(observed)"""
new=new.replace(needle,side)
new=new.replace("duration is not None and 0<duration<=30_000_000","duration==14_461_000")
new=new.replace("        'first_packets':first_packets,'last_packets':list(last_packets),'packet_side_data':side,","        'first_packets':first_packets,'last_packets':list(last_packets),'packet_side_data':side,'packet_side_data_API_provenance':packet_api_provenance,'packet_side_data_bounded_actual_read_controls':packet_side_read_observations,")
needle="    require(report['no_model_GPU_import'],'No model/GPU');write_json"
assert needle in new;new=new.replace(needle,"    require(record(failed_update_pin['path'])==failed_update_pin and record(prior_probe_pin['path'])==prior_probe_pin,'Original labels and annotation evidence unchanged after one decode')\n"+needle)
ast.parse(new);out=F/'observe_complete_member142895_EOF_v1.py';assert not out.exists();out.write_text(new,encoding='utf-8')
diff=F/'head_member134530_to142895_scoped_observation.diff.txt';diff.write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(base),tofile=str(out))),encoding='utf-8')
oldapi=ast.parse(a);newapi=ast.parse(new)
for name in ('PacketSideDataLayout','MemoryRegion','packet_side_bytes'):
 x=next(n for n in oldapi.body if getattr(n,'name',None)==name);y=next(n for n in newapi.body if getattr(n,'name',None)==name);assert ast.dump(x,include_attributes=False)==ast.dump(y,include_attributes=False),name
handoff={'kind':'ASR18_INACTIVE_exact142895_full_head_member_independent_CPU_scope_handoff_v1','source':pin(out),'base_head_member_source':pin(base),'accepted_private_side_accessor_source':pin(api),'exact_diff':pin(diff),
 'inputs':inputs,'row':{'global_update':142895,'update_index':14,'row_index':6,'ordinal':210880,'unique_id':'alania:0ce0d55b73c946f80abea8bbe41f3b27_00000','shard':'0014','start':0,'end':14.443,'historical_native_samples':693264,'observed_decoder2_written693255_not_independent_yet':True,'targets':61},
 'exact_accessor_AST_equal_to_previously_accepted_observer2':True,'private_API_limitations':'Pinned PyAV16/CPython3.11.13Win64 owned-buffer readonly accessor; strong live object/header+publicsize/type+VirtualQuery<=4096B; no public Buffer API claim or memory write',
 'one_full_original_short_member_only':True,'container_duration_bound_microseconds':14461000,'full_native_and_clean_PCM_hashes_not_yet_known':True,
 'expected_derived_geometry_not_actual_frontend_proof':{'clean16k_samples':231085,'physical_mel_frames':1444,'physical_feature_frames':1445,'encoder_frames':181,'full_target_ids':61},
 'limits':{'private_bytes':536870912,'wall_seconds':180,'full_extra_bytes':268435456,'commit_headroom_bytes':2000000000,'physical_headroom_bytes':1500000000,'E_reserve_bytes':100000000000,'CUDA_VISIBLE_DEVICES':'-1','all_four_thread_caps':1,'priority':'BelowNormal'},
 'fresh_output':str(F/'full_member_actual_01'),'no_frontend_model_GPU_sampler_journal_or_production_source_change':True,'all6_prior_policies_untouched':True,'source_scope_required_before_ONE_execution':True}
p=F/'exact_head_member_observation_source_handoff_v1.json';assert not p.exists();p.write_text(json.dumps(handoff,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({'source':pin(out),'handoff':pin(p)}),flush=True)
