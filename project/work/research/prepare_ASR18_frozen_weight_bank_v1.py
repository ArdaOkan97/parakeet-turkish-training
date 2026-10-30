"""CPU-only streaming BF16 shards from the existing immutable F32 base."""
import ctypes
import gc
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import struct
import sys
import threading
import time

PROJECT=Path(__file__).resolve().parents[2]
ROOT=Path(r'E:\gpu_training_scratchpad')
AREA=ROOT/'analysis/ASR18_memory_diagnosis_v1/frozen_weight_bank_v1'
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
from resource_watchdog_v2 import ResourceGuard,sample_memory

def record(path):
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':digest}

def main():
    config=json.loads((ROOT/'storage_config.json').read_text());assert config['storage_root']==str(ROOT)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    for key in ('TEMP','TMP','TORCH_EXTENSIONS_DIR'):assert Path(os.environ[key]).resolve().is_relative_to(ROOT.resolve())
    AREA.mkdir(parents=True,exist_ok=False)
    observed={'peak_private_bytes':0}
    def watch():
        guard=ResourceGuard()
        while True:
            value=sample_memory(os.getpid(),'CPU_frozen_weight_bank')
            private=value['worker_private_memory']['private_bytes'];observed['peak_private_bytes']=max(observed['peak_private_bytes'],private)
            stop,_=guard.observe(value['system_memory'],value['worker_private_memory'])
            write_json(AREA/'live.json',value)
            if stop or private>=2_000_000_000:
                write_json(AREA/'resource_stop.json',{'reason':stop or 'private2GB','sample':value});os._exit(75)
            time.sleep(.2)
    threading.Thread(target=watch,daemon=True).start()
    import torch
    assert torch.version.cuda is None
    torch.set_num_threads(1)
    base=Path((PROJECT/'work/base_model_path.txt').read_text().strip())
    source=base/'model.safetensors';assert source.resolve().is_relative_to(PROJECT.resolve())
    source_record=record(source);source_stat=(source.stat().st_size,source.stat().st_mtime_ns)
    proof=ROOT/'analysis/ASR18_memory_diagnosis_v1/mixed_load_equivalence_01/mixed_load_equivalence.json'
    policy=json.loads(proof.read_text());assert policy['passed']
    targets=policy['policy']['frozen_BF16_matrix_weights']
    outputs={}
    with source.open('rb') as stream:
        header_bytes=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(header_bytes));offset=8+header_bytes
        mapping=mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ)
        for index,name in enumerate(targets,1):
            entry=header[name];assert entry['dtype']=='F32'
            original=torch.frombuffer(mapping,dtype=torch.float32,count=math.prod(entry['shape']),offset=offset+entry['data_offsets'][0]).reshape(entry['shape'])
            converted=original.to(dtype=torch.bfloat16)
            path=AREA/(hashlib.sha256(name.encode()).hexdigest()[:20]+'.pt')
            torch.save(converted,path)
            restored=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
            assert torch.equal(converted.view(torch.uint8),restored.view(torch.uint8))
            outputs[name]={'tensor':record(path),'shape':entry['shape'],'dtype':'torch.bfloat16','source_dtype':'F32',
                'tensor_bytes':converted.numel()*converted.element_size(),
                'BF16_roundtrip_byte_exact':True}
            del original,converted,restored;gc.collect()
            if index%24==0:print(json.dumps({'matrices':index,'sampled_peak_private':observed['peak_private_bytes']}),flush=True)
        mapping.close()
    assert (source.stat().st_size,source.stat().st_mtime_ns)==source_stat
    result={'passed':True,'GPU_used':False,'ML_model_instantiated':False,'source':source_record,'source_readonly_mapping':True,
        'source_stat_unchanged':True,'policy':record(proof),'matrices':outputs,'matrix_count':len(outputs),
        'mapped_BF16_weight_bytes':sum(value['tensor_bytes'] for value in outputs.values()),
        'sampled_peak_private_bytes':observed['peak_private_bytes'],'source_code':record(Path(__file__))}
    write_json(AREA/'report.json',result);print(json.dumps({key:value for key,value in result.items() if key!='matrices'}),flush=True)

if __name__=='__main__':main()
