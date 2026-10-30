"""CPU-only byte conversion of the actual coherent diagnostic checkpoint1."""
import argparse, gc, json, os, sys, threading, time
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(r'E:\gpu_training_scratchpad')
sys.path.insert(0, str(PROJECT/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
from bpe_streaming_checkpoint_v1 import MappedCheckpoint, write_checkpoint, record, require

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    folder=Path(a.output).resolve();require(folder.is_relative_to(ROOT.resolve()),'E-only fixture');folder.mkdir(exist_ok=False)
    numpy_path=PROJECT/'work/training-env/Lib/site-packages'
    sys.path.insert(0,str(numpy_path));import numpy;sys.path.remove(str(numpy_path))
    import torch
    require(torch.__version__.endswith('+cpu') and not torch.cuda.is_available(),'Actual CPU-only Torch required');torch.set_num_threads(1)
    from resource_watchdog_v2 import sample_memory
    config=json.loads((PROJECT/'outputs/parakeet_turkish/bpe_broad_production_v6.disabled.json').read_text())
    parent=config['coherent_diagnostic_parent']['checkpoint'];require(record(parent['path'])==parent,'Actual checkpoint1 changed')
    value=torch.load(parent['path'],map_location='cpu',weights_only=True,mmap=True)
    require(value['diagnostic_only'] is True and len(value['weights'])==207 and value['optimizer']['clocks']==[64251,2001,62251]
            and value['sampler']['broad_cursor']['draws']==14 and value['sampler']['carried_replay']['updates']==1,'Actual coherent parent state required')
    baseline=sample_memory(os.getpid(),'actual_stream_before')['worker_private_memory']['private_bytes']
    peak=baseline;count=0;error=None;done=threading.Event()
    def watch():
        nonlocal peak,count,error
        try:
            while not done.is_set():
                sample=sample_memory(os.getpid(),'actual_stream_write')['worker_private_memory'];require(sample['available'],'Available actual write sample')
                peak=max(peak,sample['private_bytes']);count+=1;done.wait(.01)
        except BaseException as exc:error=repr(exc)
    thread=threading.Thread(target=watch,daemon=True);thread.start();start=time.monotonic()
    try:saved=write_checkpoint(folder/'actual_state.json',value,torch=torch,guard=lambda:None)
    finally:done.set();thread.join(timeout=2)
    require(not thread.is_alive() and error is None and count>=3,'Actual write monitor clean and available')
    require(peak-baseline<64_000_000,'Actual CPU streaming write exceeded64MB forecast')
    elapsed=time.monotonic()-start;mapped=MappedCheckpoint(saved,torch=torch,guard=lambda:None);checks=0;tensors=0
    def equal(actual,expected):
        nonlocal checks,tensors
        checks+=1
        if isinstance(expected,torch.Tensor):
            require(actual.dtype==expected.dtype and actual.shape==expected.shape
                    and torch.equal(actual.reshape(-1).view(torch.uint8),expected.reshape(-1).view(torch.uint8)),'Actual source tensor bytes differ');tensors+=1
        elif type(expected) is dict:
            require(type(actual) is dict and list(actual)==list(expected),'Exact ordered dictionary differs')
            for k in expected:equal(actual[k],expected[k])
        elif type(expected) in (tuple,list):
            require(type(actual) is type(expected) and len(actual)==len(expected),'Exact tuple/list differs')
            for av,ev in zip(actual,expected):equal(av,ev)
        else:require(type(actual) is type(expected) and actual==expected,'Actual metadata/RNG scalar differs')
    equal(mapped.value,value);mapped.close();del value;gc.collect()
    write_json(folder/'report.json',{'passed':True,'CPU_only':True,'PID':os.getpid(),'source':record(__file__),
        'checkpoint_source':record(PROJECT/'outputs/parakeet_turkish/bpe_streaming_checkpoint_v1.py'),
        'parent_checkpoint':parent,'streamed_checkpoint':saved,'all207_weights_all3Adam_fullsampler_RNG_bytes_exact':True,
        'checks':checks,'tensors':tensors,'baseline_private_bytes':baseline,'sampled_write_peak_private_bytes':peak,
        'sampled_write_increment_bytes':peak-baseline,'available_write_samples':count,'monitor_error':error,'write_seconds':elapsed,
        'GPU_update_or_quality_claimed':False,'diagnostic_only':True})
    print(json.dumps({'passed':True,'checks':checks,'tensors':tensors,'increment':peak-baseline,'seconds':elapsed}))

if __name__=='__main__':main()
