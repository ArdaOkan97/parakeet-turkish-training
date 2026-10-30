"""Preserve actual metadata1, bind genuine producer's complete four-field marker."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def pin(p):
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':h}
def write(p,d):
 assert not p.exists()
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
guard=A/'hard_effective16_configuration_guard_01/closure.json';g=read(guard)
assert g['actual_exit_code']==1 and g['pid']==6236 and g['monitor_clean'] and g['monitor_joined']
failure=A/'hard_effective16_configuration_CPU_actual_01/actual_owned_metadata_TOOL_CLOSED1_v1.json'
write(failure,{'kind':'ASR18_hard_effective16_operator_observed_metadata_TOOL_CLOSED1_v1','owner':'/root/memory_diagnosis',
 'actual_tool_chunk_id':'737110','actual_tool_exit_code':1,'actual_tool_completion_observed':True,'native_PID':6236,
 'worker_source':pin(P/'work/research/prepare_ASR18_hard_effective16_configuration_v2.py'),'guard_closure':pin(guard),
 'error':'Genuine final_geometry published_marker has mtime_ns; template provided only record3; exact actual0 rejected before bridge/audio',
 'actual_audio_frontend_GPU_or_Adam_jobs':0,'production_enabled':False})
old=S/'asr18_hard_effective16.disabled.v2.json';t=read(old)
wrapper=read(Path(t['forecast_calibration']['final_geometry_actual0']['path']))
full=wrapper['published_marker'];assert set(full)=={'path','bytes','sha256','mtime_ns'}
assert {k:full[k] for k in ('path','bytes','sha256')}==t['forecast_calibration']['final_geometry']
assert Path(full['path']).stat().st_mtime_ns==full['mtime_ns']
t['forecast_calibration']['final_geometry']=full;t['kind']='ASR18_hard_effective16_config_TEMPLATE_NOT_LAUNCHABLE_v3'
tp=S/'asr18_hard_effective16.disabled.v3.json';write(tp,t)
mp=A/'hard_effective16_full_caller_source_review_manifest_v2.json';m=read(mp);bindings={p['path']:p for p in m['bindings']}
for p in (tp,failure,Path(__file__)):bindings[str(p.resolve())]=pin(p)
for p in bindings.values():assert pin(Path(p['path']))==p
m.update(kind='ASR18_hard_effective16_CPU8_full_caller_source_review_manifest_v3',bindings=sorted(bindings.values(),key=lambda x:x['path']),
 configuration_template=pin(tp),previous_hard_source_manifest=pin(mp),preserved_actual_metadata_failure=pin(failure),
 source_delta=m['source_delta']+'; template3 restores exact genuine producer record4 final_geometry marker including unchanged mtime_ns')
out=A/'hard_effective16_full_caller_source_review_manifest_v3.json';write(out,m)
print(json.dumps({'template':pin(tp),'manifest':pin(out),'failure':pin(failure),'bindings':len(bindings)}))
