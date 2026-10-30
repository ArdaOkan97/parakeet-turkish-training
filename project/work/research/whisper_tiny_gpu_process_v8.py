"""CPU-only codec subprocesses feed one GPU parent. All queues are bounded."""
import os,sys,pathlib,json,time,threading,argparse,concurrent.futures,multiprocessing
import whisper_tiny_gpu_batch_v1 as common
from whisper_tiny_gpu_pipeline_v5 import TinyEngine,InputGuard
from whisper_tiny_gpu_stream_v7 import FastSource,FastVAD,stream
AREA=common.AREA;ROOT=common.ROOT
_SOURCE=None

class WorkerReadGuard:
    def __init__(self):self.bytes=0;self.paused_seconds=0.;self.last=0.
    def wait(self):
        if time.monotonic()-self.last>.5:
            self.last=time.monotonic();m=common.base.memory()
            if m['commit_ratio']>=.95 or m['commit_headroom_bytes']<2_000_000_000:raise RuntimeError('Critical global memory in codec child')
    def pace(self,n):
        self.wait();self.bytes+=n
        if self.bytes>1_000_000_000:raise RuntimeError('Per-codec benchmark1GB bound')

def initialize_codec(spec):
    global _SOURCE
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    try:import av
    except ImportError:sys.path.append(r'E:\gpu_training_scratchpad\scratch\CPU_validation_v2_env\Lib\site-packages');import av
    if 'torch' in sys.modules:raise RuntimeError('Codec subprocess must not load Torch/GPU')
    _SOURCE=FastSource.__new__(FastSource);_SOURCE.guard=WorkerReadGuard();_SOURCE.local=threading.local();_SOURCE.members=spec['members'];_SOURCE.receipts=spec['receipts']

def decode_task(row):
    pcm,proof=_SOURCE.decode(row);proof.update(codec_worker_PID=os.getpid(),codec_worker_Torch_loaded=False,codec_worker_GPU_visible=False);return pcm,proof

class ProcessProxy:
    decode=staticmethod(decode_task)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--processes',default='1,2,4,8');ap.add_argument('--repeat',type=int,default=3);ap.add_argument('--group',type=int,default=64);args=ap.parse_args()
    guard=common.Guard();m=guard.check();assert m['commit_ratio']<.9 and m['commit_headroom_bytes']>=8_000_000_000
    import torch
    guard.torch=torch;torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    try:import av
    except ImportError:sys.path.append(r'E:\gpu_training_scratchpad\scratch\CPU_validation_v2_env\Lib\site-packages');import av
    leasep=ROOT/'work/gpu_training_owner.json';lease=json.loads(leasep.read_text());assert not lease.get('actual_GPU_running');lease.update(actual_GPU_running=True,status='ROOT_tiny_bounded_codec_process_benchmark_v8',actual_owned_worker_pid=os.getpid());leasep.write_text(json.dumps(lease,indent=2)+'\n')
    engine=TinyEngine(guard);vad=FastVAD()
    pin=pathlib.Path(r'E:\gpu_training_scratchpad\analysis\CPU_validation_v8\new256_tiny_primary_manifest_v8.jsonl');unique=[json.loads(x) for x in pin.read_text(encoding='utf-8').splitlines()];rows=unique*args.repeat
    old=pathlib.Path(r'E:\gpu_training_scratchpad\analysis\CPU_validation_v8\new256_tiny_primary_ledger_v8.jsonl');expected={x['segment_id']:x['decode']['PCM_f32le_sha256'] for x in map(json.loads,old.read_text(encoding='utf-8').splitlines())}
    source=FastSource(unique,InputGuard(guard));spec={'members':source.members,'receipts':source.receipts};original=concurrent.futures.ThreadPoolExecutor;results=[]
    for n in map(int,args.processes.split(',')):
        guard.check();times=[];prediction=engine.predict
        def timed_predict(waves):
            value=prediction(waves);times.append(time.perf_counter());return value
        engine.predict=timed_predict;started=time.perf_counter()
        def factory(max_workers):
            return concurrent.futures.ProcessPoolExecutor(max_workers=max_workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_codec,initargs=(spec,))
        concurrent.futures.ThreadPoolExecutor=factory
        try:
            out,metrics,held=stream(engine,vad,rows,ProcessProxy(),guard,n,args.group,AREA/f'process_v8_{n}_ledger.jsonl',expected)
        finally:
            concurrent.futures.ThreadPoolExecutor=original;engine.predict=prediction
        metrics.update(codec_processes=n,clip_operations=len(rows),unique_clips=len(unique),first_group_ready_and_inferred_s=times[0]-started,steady_clips_per_s=(len(rows)-args.group)/(times[-1]-times[0]),all_PCM_matches=len(out),codec_worker_PIDs=sorted({x['decode_proof']['codec_worker_PID'] for x in out}),CPU_children_Torch_loaded=False,CPU_children_GPU_visible=False,memory=guard.check(),includes='actual original tar decode + compiled VAD + graph tiny + durable ledger; startup included in total; repeat count explicit')
        results.append(metrics);print('PROCESS_END_TO_END '+json.dumps(metrics),flush=True);common.write('process_v8_partial.json',dict(results=results))
    common.write('process_v8_report.json',dict(at_utc=common.now(),results=results,repeat=args.repeat,source_manifest=str(pin),new_TRAIN_admitted=0,memory=guard.check()))
    engine.close();guard.stop.set();del engine;torch.cuda.empty_cache();lease.update(actual_GPU_running=False,status='ROOT_tiny_codec_process_benchmark_closed',actual_owned_worker_pid=None);leasep.write_text(json.dumps(lease,indent=2)+'\n');print('COMPLETE process_v8',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:common.write('process_v8_failure.json',dict(error=repr(e),at_utc=common.now(),pid=os.getpid()));raise
