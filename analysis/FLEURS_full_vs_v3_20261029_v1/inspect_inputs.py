import os, sys, json, time, hashlib, shutil
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from paired_windows_commit_telemetry_v1 import paired_sample
R=Path(__file__).parent
def save(name,value):
    with (R/name).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
samples=[paired_sample(),paired_sample()]
save('metadata_admission.json',samples)
assert all(s['global_memory_status_ex']['ullAvailPageFile_bytes']>=6028384896 and s['get_performance_info_after']['physical_available_bytes']>=3278384896 for s in samples)
assert shutil.disk_usage(R).free>=100000000000
configs={'tr':'tr_tr','en':'en_us','es':'es_419','it':'it_it','fr':'fr_fr','de':'de_de','pt':'pt_br'}
rev='70bb2e84b976b7e960aa89f1c648e09c59f894dd'
cache=P/'work/data/hf_cache/datasets--google--fleurs/snapshots'/rev/'parquet-data'
ck=Path(r'E:\gpu_training_scratchpad\checkpoints\BPE9223_resident_RAM_20261018_01_protected_DEV_v1\update_000400000\checkpoint.json')
save('metadata_scope.json',{'parquets':{l:str(cache/c/'test-00000-of-00001.parquet') for l,c in configs.items()},'checkpoint':str(ck),'maximum_wall_seconds':180,'maximum_private_bytes':536870912,'no_audio_or_tensor_payload_reads':True})
raw=ck.read_bytes();assert len(raw)<16777216
with (R/'checkpoint_metadata_RAW.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
d=json.loads(raw)
import pyarrow.parquet as pq
info={}
for l,c in configs.items():
    path=cache/c/'test-00000-of-00001.parquet';pf=pq.ParquetFile(path)
    info[l]={'path':str(path),'rows':pf.metadata.num_rows,'row_groups':pf.num_row_groups,'columns':pf.schema_arrow.names,'bytes':path.stat().st_size}
save('input_inventory.json',{'datasets':info,'revision':rev,'checkpoint_metadata_sha256':hashlib.sha256(raw).hexdigest(),'checkpoint_keys':list(d),'payload':d['payload'],'checkpoint_bytes':len(raw)})
print(json.dumps({'datasets':info,'checkpoint_keys':list(d),'payload':d['payload'],'checkpoint_sha256':hashlib.sha256(raw).hexdigest()}))
