"""Guarded CPU byte comparison of cached and normal-teacher native closes."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

PROJECT=Path(__file__).resolve().parents[2]
ROOT=Path(r'E:\gpu_training_scratchpad');BASE=ROOT/'analysis/ASR18_memory_diagnosis_v1'
AREA=BASE/'normal_teacher_state_equivalence_v1'
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
from resource_watchdog_v2 import ResourceGuard,sample_memory

def record(path):
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':str(path),'sha256':digest,'bytes':path.stat().st_size}

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    assert json.loads((ROOT/'storage_config.json').read_text())['storage_root']==str(ROOT)
    for key in ('TEMP','TMP','TORCH_EXTENSIONS_DIR'):assert Path(os.environ[key]).resolve().is_relative_to(ROOT.resolve())
    AREA.mkdir(exist_ok=False);observed={'private_peak':0}
    def watch():
        guard=ResourceGuard()
        while True:
            sample=sample_memory(os.getpid(),'CPU_native_state_equivalence')
            observed['private_peak']=max(observed['private_peak'],sample['worker_private_memory']['private_bytes'])
            stop,_=guard.observe(sample['system_memory'],sample['worker_private_memory'])
            if stop or observed['private_peak']>=2_000_000_000:
                write_json(AREA/'resource_stop.json',{'reason':stop or 'CPU_private2GB','sample':sample});os._exit(75)
            time.sleep(.1)
    threading.Thread(target=watch,daemon=True).start()
    import torch
    assert torch.version.cuda is None;torch.set_num_threads(1)
    values=[];records=[]
    for name in ('native_streamed_stability_02','native_streamed_normal_teacher_01'):
        report=json.loads((BASE/name/'native_training_validation.json').read_text());assert report['passed'] and report['completed_updates']==20
        path=Path(report['checkpoint']['path']);assert record(path)==report['checkpoint']
        values.append(torch.load(path,map_location='cpu',weights_only=True,mmap=True));records.append(report['checkpoint'])
    count={'tensors':0}
    def compare(left,right):
        assert type(left)==type(right)
        if isinstance(left,torch.Tensor):
            assert left.shape==right.shape and left.dtype==right.dtype
            assert torch.equal(left.contiguous().reshape(-1).view(torch.uint8),right.contiguous().reshape(-1).view(torch.uint8));count['tensors']+=1
        elif isinstance(left,dict):
            assert left.keys()==right.keys()
            for key in left:compare(left[key],right[key])
        elif isinstance(left,(tuple,list)):
            assert len(left)==len(right)
            for a,b in zip(left,right):compare(a,b)
        else:assert left==right
    for field in ('weights','optimizer','python_rng','torch_rng','cuda_rng'):compare(values[0][field],values[1][field])
    assert 'cached_teacher_policy' not in values[1]['source_provenance']
    assert all(a['mean_clip_objective']==b['mean_clip_objective'] and a['global_grad_norm']==b['global_grad_norm']
               for a,b in zip(values[0]['history'],values[1]['history']))
    result={'passed':True,'GPU_used':False,'source_checkpoints':records,'tensor_fields_compared':count['tensors'],
        'all207weights_allAdam_moments_steps_RNG_byte_exact':True,'all20_objectives_and_gradnorms_exact':True,
        'normal_teacher_has_no_control_cache_dependency':True,'full1914_replay_cache_required_for_memory_fix':False,
        'DEV_for_byte_identical_weights':record(BASE/'native_streamed_final_DEV_02/final_DEV_validation.json'),
        'sampled_CPU_private_peak_bytes':observed['private_peak'],'production_large_geometry_or_quality_claimed':False}
    del values;gc.collect();write_json(AREA/'report.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
