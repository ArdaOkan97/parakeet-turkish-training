"""New derivation from retained evaluation outputs; no model/job rerun."""
import hashlib,json,os,sys,time,urllib.request
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');R=Path(__file__).parent
sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from paired_windows_commit_telemetry_v1 import paired_sample
from resource_watchdog_v2 import sample_memory
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
start=time.monotonic();out=R/'WER_explanation_v1';out.mkdir(exist_ok=False)
samples=[paired_sample(),paired_sample()];write(out/'admission.json',samples)
assert all(x['global_memory_status_ex']['ullAvailPageFile_bytes']>=6028384896 and x['get_performance_info_after']['physical_available_bytes']>=3278384896 for x in samples)
source=Path(r'E:\gpu_training_scratchpad\analysis\FLEURS_full_vs_v3_20261029_v1\adapted_v3\Turkish_DEV_loader_check.json')
native=R/'native_validation_v9/Turkish_DEV.json'
write(out/'scope.json',{'inputs':[str(source),str(native)],'purpose':'Paired prediction derivation from retained outputs, not a closed training metric recapture or model rerun','read_max_each':262144,'output_max':32768,'wall':180,'private_limit':536870912})
data=[]
for i,path in enumerate([source,native]):
    memory=sample_memory();assert memory['system_memory']['commit_available_bytes']>=3509949440 and memory['system_memory']['physical_available_bytes']>=3009949440
    assert memory['worker_private_memory']['private_bytes']<536870912 and time.monotonic()-start<180
    with path.open('rb') as f:raw=f.read(262145)
    assert len(raw)<=262144
    with (out/f'retained_input_{i}_RAW.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    data.append(json.loads(raw))
a,b=data;old={x['source_id']:x for x in a['score']['details']};new={x['source_id']:x for x in b['score']['details']}
assert old.keys()==new.keys() and len(old)==50
changed=[]
for identity in old:
    x,y=old[identity],new[identity]
    assert x['reference_normalized']==y['reference_normalized']
    if x['hypothesis_normalized']!=y['hypothesis_normalized']:
        changed.append({'source_id':identity,'source_errors':x['word_errors'],'native_errors':y['word_errors'],'delta':y['word_errors']-x['word_errors'],'reference':x['reference_normalized'],'source':x['hypothesis_normalized'],'native':y['hypothesis_normalized']})
with urllib.request.urlopen('https://pypi.org/pypi/nemo-toolkit/json',timeout=10) as response:pypi_raw=response.read(2*1024**2+1)
assert len(pypi_raw)<=2*1024**2
with (out/'PyPI_metadata_RAW.json').open('xb') as f:f.write(pypi_raw);f.flush();os.fsync(f.fileno())
version=json.loads(pypi_raw)['info']['version']
result={'same_50_source_IDs_and_normalized_references':True,'source_errors':a['score']['word_errors'],'native_errors':b['score']['word_errors'],'reference_words':a['score']['reference_words'],'changed_normalized_transcripts':len(changed),'clips_worse':sum(x['delta']>0 for x in changed),'clips_better':sum(x['delta']<0 for x in changed),'clips_different_same_error_count':sum(x['delta']==0 for x in changed),'total_error_delta':sum(x['delta'] for x in changed),'latest_PyPI_nemo_toolkit':version,'cause':'Not isolated: original HF BF16/GPU runtime versus native NeMo FP32/CPU, frontend/decoding/conversion all differ; no causal ablation performed','changes':changed}
write(out/'comparison.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='changes'},ensure_ascii=False))
