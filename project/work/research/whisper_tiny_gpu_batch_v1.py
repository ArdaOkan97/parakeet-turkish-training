"""Bounded, language-only GPU throughput/correctness benchmark. All artifacts on E:."""
import os, sys, json, time, io, base64, hashlib, pathlib, threading, statistics, argparse, datetime
ROOT = pathlib.Path(__file__).resolve().parents[2]
AREA = pathlib.Path(r'E:\gpu_training_scratchpad\analysis\whisper_tiny_GPU_batch_v1')
sys.path.insert(0, str(ROOT/'work/research'))
import cpu_preparation_guard_v2 as base

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(name, value):
    p=AREA/name; q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');q.replace(p)

class Guard:
    def __init__(self):
        import psutil
        self.proc=psutil.Process();self.stop=threading.Event();self.reason=None;self.torch=None
        self.peak_private=0;self.peak_cuda=0;self.high=0
        threading.Thread(target=self.watch,daemon=True).start()
    def snapshot(self):
        m=base.memory();private=self.proc.memory_info().private
        cuda=self.torch.cuda.memory_reserved() if self.torch and self.torch.cuda.is_initialized() else 0
        self.peak_private=max(self.peak_private,private);self.peak_cuda=max(self.peak_cuda,cuda)
        return dict(m,private_bytes=private,cuda_reserved_bytes=cuda,peak_private_bytes=self.peak_private,peak_cuda_reserved_bytes=self.peak_cuda)
    def watch(self):
        while not self.stop.is_set():
            try:
                m=self.snapshot();self.high=self.high+1 if m['commit_ratio']>=.90 else 0
                if m['commit_ratio']>=.95 or m['commit_headroom_bytes']<2_000_000_000 or m['private_bytes']>8_000_000_000 or m['cuda_reserved_bytes']>8_000_000_000 or self.high>=3:
                    self.reason='unchanged_resource_guard';self.stop.set();write('resource_stop.json',dict(reason=self.reason,at_utc=now(),memory=m))
            except Exception as e:
                self.reason='resource_monitor_failure:'+repr(e);self.stop.set()
            self.stop.wait(5)
    def check(self):
        if self.reason:raise RuntimeError(self.reason)
        m=self.snapshot()
        if m['commit_ratio']>=.95 or m['commit_headroom_bytes']<2_000_000_000:raise RuntimeError('critical_system_memory')
        if m['private_bytes']>8_000_000_000 or m['cuda_reserved_bytes']>8_000_000_000:raise RuntimeError('own8GB_memory_cap')
        return m

def load_audio():
    import numpy as np
    try:import av
    except ImportError:
        sys.path.append(r'E:\gpu_training_scratchpad\scratch\CPU_validation_v2_env\Lib\site-packages');import av
    clips=[];seen=set();start=time.perf_counter()
    for folder in ['tiny_large_review_v1','tiny_language_followup8_v1']:
        path=AREA.parent/folder/'original_opus_review_cache.jsonl'
        for line in path.read_text(encoding='utf-8').splitlines():
            r=json.loads(line)
            if r['segment_id'] in seen:continue
            raw=base64.b64decode(r['audio_b64']);assert hashlib.sha256(raw).hexdigest()==r['proof']['sha256']
            samples=[]
            with av.open(io.BytesIO(raw)) as c:
                res=av.AudioResampler(format='fltp',layout='mono',rate=16000)
                for frame in c.decode(audio=0):
                    for f in res.resample(frame):samples.append(f.to_ndarray().reshape(-1))
                for f in res.resample(None):samples.append(f.to_ndarray().reshape(-1))
            pcm=np.concatenate(samples).astype(np.float32);clips.append((r['segment_id'],pcm));seen.add(r['segment_id'])
    return clips,time.perf_counter()-start

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batches',default='1,4,8,16,32,64,128');ap.add_argument('--mode',default='full');ap.add_argument('--out',default='baseline_full');args=ap.parse_args()
    guard=Guard();admission=guard.check();assert admission['commit_ratio']<.90 and admission['commit_headroom_bytes']>=8_000_000_000
    write(args.out+'_status.json',dict(phase='importing',at_utc=now(),pid=os.getpid(),memory=admission))
    import numpy as np,torch
    from transformers import WhisperForConditionalGeneration,WhisperFeatureExtractor
    guard.torch=torch;torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    receipt=json.loads((AREA/'HF_download_receipt.json').read_text());model_path=pathlib.Path(receipt['model_directory'])
    start=time.perf_counter();model=WhisperForConditionalGeneration.from_pretrained(model_path,local_files_only=True,dtype=torch.float16,attn_implementation='sdpa',low_cpu_mem_usage=True).eval().to('cuda')
    extractor=WhisperFeatureExtractor.from_pretrained(model_path,local_files_only=True)
    ids=torch.tensor(list(model.generation_config.lang_to_id.values()),device='cuda');languages=[k[2:-2] for k in model.generation_config.lang_to_id]
    sot=model.generation_config.decoder_start_token_id
    lease_path=ROOT/'work/gpu_training_owner.json';lease=json.loads(lease_path.read_text());assert lease['owner']=='/root'
    lease.update(status='ROOT_whisper_tiny_GPU_batch_benchmark',actual_GPU_running=True,actual_training_optimizer_running=False,actual_owned_worker_pid=os.getpid(),source=str(pathlib.Path(__file__)))
    lease_path.write_text(json.dumps(lease,indent=2)+'\n',encoding='utf-8')
    load_s=time.perf_counter()-start;clips,decode_s=load_audio()
    windows=[];manifest=[]
    for sid,pcm in clips:
        for i in range(0,len(pcm),64000):
            part=pcm[i:i+64000]
            if len(part)<4000:continue
            windows.append(part);manifest.append(dict(segment_id=sid,start_s=i/16000,duration_s=len(part)/16000))
    start=time.perf_counter();features=extractor(windows,sampling_rate=16000,return_tensors='pt').input_features
    feature_s=time.perf_counter()-start;features=features.to('cuda',dtype=torch.float16);torch.cuda.synchronize()
    def infer(x):
        decoder=torch.full((x.shape[0],1),sot,dtype=torch.long,device='cuda')
        h=model.model(input_features=x,decoder_input_ids=decoder,use_cache=False).last_hidden_state[:,0]
        # Only score language tokens: exactly the same linear rows as the full vocabulary projection.
        logits=torch.nn.functional.linear(h,model.proj_out.weight.index_select(0,ids))
        return logits.float().softmax(-1)
    results=[]
    with torch.inference_mode():
        reference=torch.cat([infer(features[i:i+1]) for i in range(len(windows))]);official=model.detect_language(input_features=features[:8]);expected=ids[reference[:8].argmax(-1)];assert torch.equal(official,expected),'Official language-ID mismatch'
        for b in [int(x) for x in args.batches.split(',')]:
            guard.check();x=features[torch.arange(b,device='cuda')%len(windows)].contiguous()
            for _ in range(3):infer(x)
            torch.cuda.synchronize();start=time.perf_counter();infer(x);torch.cuda.synchronize();single=time.perf_counter()-start
            repeats=max(3,min(80,int(1.5/max(single,.001))))
            timings=[]
            for _ in range(3):
                start=time.perf_counter()
                for j in range(repeats):out=infer(x)
                torch.cuda.synchronize();timings.append((time.perf_counter()-start)/repeats)
            ref=reference[torch.arange(b,device='cuda')%len(windows)];err=(out-ref).abs().max().item();agree=(out.argmax(-1)==ref.argmax(-1)).float().mean().item()
            result=dict(batch=b,batch_latency_s=statistics.median(timings),windows_per_second=b/statistics.median(timings),max_probability_difference_vs_batch1=err,language_agreement_vs_batch1=agree,repeats=repeats,memory=guard.check())
            results.append(result);write(args.out+'_partial.json',dict(results=results));print(json.dumps(result),flush=True)
    report=dict(kind='actual_GPU_tiny_language_only_batch_benchmark',at_utc=now(),pid=os.getpid(),model=receipt['repo'],revision=receipt['revision'],dtype='float16',attention='sdpa',projection='language_token_rows_only',encoder_frames=3000,encoder_length_cropped=False,model_load_s=load_s,cached_original_Opus_decode_s=decode_s,CPU_feature_extraction_s=feature_s,unique_clips=len(clips),unique_windows=len(windows),audio_seconds=sum(x['duration_s'] for x in manifest),results=results,memory=guard.check(),environment=dict(torch=torch.__version__,GPU=torch.cuda.get_device_name()),speech_VAD_applied=False,cutoff_x=None)
    write(args.out+'_report.json',report);write('window_manifest.json',manifest)
    predictions=[dict(**manifest[i],probabilities={lang:float(v) for lang,v in zip(languages,reference[i].cpu().tolist())}) for i in range(len(windows))];write(args.out+'_predictions.json',predictions)
    guard.stop.set();del model;torch.cuda.empty_cache();lease.update(actual_GPU_running=False,status='ROOT_whisper_tiny_GPU_batch_window_closed',actual_owned_worker_pid=None);lease_path.write_text(json.dumps(lease,indent=2)+'\n',encoding='utf-8')
    write(args.out+'_status.json',dict(phase='complete',at_utc=now(),pid=os.getpid(),report=str(AREA/(args.out+'_report.json'))))
    print(json.dumps(dict(complete=True,report=str(AREA/(args.out+'_report.json')))),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        write('latest_failure.json',dict(error=repr(e),at_utc=now(),pid=os.getpid()));raise
