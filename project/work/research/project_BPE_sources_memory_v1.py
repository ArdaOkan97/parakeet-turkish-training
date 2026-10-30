"""Guarded, bounded history-free source sidecars; immutable parents retained."""
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import psutil

PROJECT=Path(__file__).resolve().parents[2]
SOURCE=PROJECT/'outputs/parakeet_turkish'
ROOT=Path(r'E:\gpu_training_scratchpad')
AREA=ROOT/'analysis/ASR18_memory_diagnosis_v1/source_projection_v4'
sys.path.insert(0,str(SOURCE))
from atomic_io_retry_v1 import write_json
from resource_watchdog_v2 import sample_memory,ResourceGuard
from history_projected_checkpoint_v1 import load_without_history

def record(path):
    with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':digest}

def watchdog():
    guard=ResourceGuard()
    while True:
        value=sample_memory(os.getpid(),'CPU_source_projection')
        stop,_=guard.observe(value['system_memory'],value['worker_private_memory'])
        if stop or value['worker_private_memory']['private_bytes']>=2_000_000_000:
            write_json(AREA/'resource_stop.json',{'reason':stop or 'CPU_private_at_or_above2GB','sample':value})
            os._exit(75)
        write_json(AREA/'live.json',value)
        time.sleep(1)

def tensor_digest(torch,tensor):
    tensor=tensor.detach().cpu().contiguous().reshape(-1)
    h=hashlib.sha256()
    for piece in tensor.split(262144): h.update(ctypes.string_at(piece.data_ptr(),piece.numel()*piece.element_size()))
    return {'shape':list(tensor.shape),'dtype':str(tensor.dtype),'sha256':h.hexdigest(),'elements':tensor.numel()}

def audit(torch,value):
    result={}
    def walk(item,key):
        if isinstance(item,torch.Tensor): result[key]=tensor_digest(torch,item)
        elif isinstance(item,dict):
            for k,v in item.items(): walk(v,key+'/'+str(k))
        elif isinstance(item,(tuple,list)):
            for i,v in enumerate(item): walk(v,key+'/'+str(i))
    walk(value,'root');return result

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    for key in ('TEMP','TMP','TORCH_EXTENSIONS_DIR'): assert Path(os.environ[key]).resolve().is_relative_to(ROOT.resolve())
    AREA.mkdir(parents=True,exist_ok=False)
    threading.Thread(target=watchdog,daemon=True).start()
    import torch
    assert torch.version.cuda is None
    # Verify projection semantics before immutable source archives.
    fixture=AREA/'projection_fixture.pt'
    torch.save({'weights':{'x':torch.arange(10.)},'history':[{'step':i,'nested':[{'a':i}]*3} for i in range(1200)],
                'step':1200,'python_rng':(3,(1,2,3),None)},fixture)
    projected=load_without_history(torch,fixture)
    assert torch.equal(projected['weights']['x'],torch.arange(10.)) and projected['step']==1200
    del projected;gc.collect()
    shared={'x':3};torch.save({'history':[shared],'outside':shared},AREA/'escaped_history_fixture.pt')
    try:
        load_without_history(torch,AREA/'escaped_history_fixture.pt')
        raise AssertionError('Escaped history reference accepted')
    except ValueError: pass
    paths={'original':(PROJECT/'work/closed-sources/TSC_WorldSpeech_expanded_65948_closed_v1/best_checkpoint.pt',
                       'b0f5d5e207caeda3c030a5f11eda4ae9004e4826a7fd050a8ac8de78333e3e38'),
           'bpe':(ROOT/'runs/adaptive_20261015_TR_BPE512_full_decoder_02/latest_checkpoint.pt',
                  '71ca1897d95622c5556df6a3b568d26136004e25c4bcca515f7a242df9e9b63a')}
    reports={}
    for label,(path,expected) in paths.items():
        parent=record(path);assert parent['sha256']==expected
        print(json.dumps({'stage':'projecting_source_history','source':label,'private_bytes':psutil.Process().memory_info().private}),flush=True)
        value=load_without_history(torch,path)
        print(json.dumps({'stage':'source_projected','source':label,'private_bytes':psutil.Process().memory_info().private}),flush=True)
        before=audit(torch,value)
        value['immutable_history_parent']={'checkpoint':parent,'field':'history','retained_in_parent':True,
                                           'diagnostic_sidecar_omits_history_materialization':True}
        out=AREA/(label+'_without_history.pt')
        torch.save(value,out);step=value['step'];del value;gc.collect()
        restored=torch.load(out,map_location='cpu',weights_only=True,mmap=True)
        after=audit(torch,restored);assert before==after
        assert restored['step']==step and restored['immutable_history_parent']['checkpoint']==parent
        reports[label]={'parent':parent,'projected':record(out),'step':step,'tensor_count':len(before),
                        'all_tensors_byte_exact_after_roundtrip':True,'tensor_audit':before,
                        'private_bytes_after_verification':psutil.Process().memory_info().private}
        del restored;gc.collect()
        write_json(AREA/'progress.json',reports)
        print(json.dumps({'stage':'source_projection_verified','source':label,'step':step,'tensor_count':len(before),
                          'private_bytes':psutil.Process().memory_info().private}),flush=True)
    write_json(AREA/'report.json',{'passed':True,'GPU_used':False,'source_clocks_or_RNG_reset':False,
               'parents_unchanged':all(record(path)['sha256']==expected for path,expected in paths.values()),
               'projections':reports,'projection_loader':record(SOURCE/'history_projected_checkpoint_v1.py')})

if __name__=='__main__':main()
