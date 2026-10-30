"""GPU tiny language-share pipeline and bounded real-original-audio benchmarks.

No transcription. Non-overlapping 4s LID windows are weighted by Silero speech
timestamps. Turkish/other/uncertain are estimates, not human language labels.
"""
import pathlib,os,sys,json,time,math,hashlib,statistics,threading,concurrent.futures,importlib.util,argparse,gc
import whisper_tiny_gpu_batch_v1 as common
from whisper_tiny_gpu_fused_v3 import FusedLanguageModel
AREA=common.AREA;ROOT=common.ROOT;write=common.write

class InputGuard:
    def __init__(self,guard,budget=1_500_000_000,rate=128*1024*1024):
        self.guard=guard;self.budget=budget;self.rate=rate;self.bytes=0;self.lock=threading.Lock();self.started=time.monotonic();self.paused_seconds=0.
    def wait(self):
        if self.guard.reason:raise RuntimeError(self.guard.reason)
    def pace(self,count):
        self.wait()
        with self.lock:
            self.bytes+=count
            if self.bytes>self.budget:raise RuntimeError('Declared compressed-read budget exceeded')
            delay=self.bytes/self.rate-(time.monotonic()-self.started)
        if delay>0:time.sleep(delay)

class TinyEngine:
    def __init__(self,guard,batch=8):
        import numpy as np,torch
        from transformers import WhisperForConditionalGeneration,WhisperFeatureExtractor
        from torch.nn.attention import sdpa_kernel,SDPBackend
        self.np=np;self.torch=torch;self.guard=guard;self.batch=batch
        receipt=json.loads((AREA/'HF_download_receipt.json').read_text());folder=receipt['model_directory']
        self.model=WhisperForConditionalGeneration.from_pretrained(folder,local_files_only=True,dtype=torch.float16,attn_implementation='sdpa',low_cpu_mem_usage=True).eval().to('cuda')
        self.extractor=WhisperFeatureExtractor.from_pretrained(folder,local_files_only=True)
        self.languages=[k[2:-2] for k in self.model.generation_config.lang_to_id];self.ids=torch.tensor(list(self.model.generation_config.lang_to_id.values()),device='cuda');self.tr=self.languages.index('tr')
        self.core=FusedLanguageModel(self.model,self.ids)
        self.original_sdpa=torch.nn.functional.scaled_dot_product_attention
        original=self.original_sdpa
        def attention(q,k,v,*args,**kwargs):
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION if q.shape[-2]>1 else SDPBackend.EFFICIENT_ATTENTION):return original(q,k,v,*args,**kwargs)
        torch.nn.functional.scaled_dot_product_attention=attention
        self.filters=torch.tensor(self.extractor.mel_filters,device='cuda',dtype=torch.float32);self.hann=torch.hann_window(400,device='cuda');self.static=torch.zeros((batch,80,3000),device='cuda',dtype=torch.float16)
        with torch.inference_mode():
            side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(5):self.core(self.static)
            torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.graph_output=self.core(self.static)
    def mel(self,pcm):
        t=self.torch;spec=t.stft(pcm,400,160,window=self.hann,return_complex=True);power=(spec[...,:-1].abs()**2).contiguous();log=(self.filters.T@power).clamp_min(1e-10).log10();floor=log.amax(dim=(-2,-1),keepdim=True)-8.;log=t.maximum(log,floor);log=(log+4.)/4.
        return t.cat((log,((floor+4.)/4.).expand(len(pcm),80,3000-log.shape[-1])),dim=-1).half()
    def predict(self,windows):
        t=self.torch;results=[]
        with t.inference_mode():
            for i in range(0,len(windows),self.batch):
                self.guard.check();part=windows[i:i+self.batch];buf=t.zeros((self.batch,64400),pin_memory=True)
                for j,w in enumerate(part):
                    if not 0<len(w)<=64000:raise ValueError('Language window must be <=4s')
                    buf[j,:len(w)].copy_(t.from_numpy(w))
                self.static.copy_(self.mel(buf.to('cuda',non_blocking=True)));self.graph.replay();results.extend(self.graph_output[:len(part)].cpu().tolist())
        return results
    def close(self):self.torch.nn.functional.scaled_dot_product_attention=self.original_sdpa

class BatchedVAD:
    def __init__(self):
        import torch
        self.torch=torch;p=json.loads((AREA/'VAD_model_pin.json').read_text());modelp=pathlib.Path(p['files']['src/silero_vad/data/silero_vad.jit']['path']);assert hashlib.file_digest(modelp.open('rb'),'sha256').hexdigest()==p['files']['src/silero_vad/data/silero_vad.jit']['sha256']
        self.model=torch.jit.load(str(modelp),map_location='cpu').eval()
        util=p['files']['src/silero_vad/utils_vad.py'];path=pathlib.Path(util['path']);assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==util['sha256']
        spec=importlib.util.spec_from_file_location('pinned_silero_utils',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);self.timestamps=module.get_speech_timestamps
    def probabilities(self,clips):
        t=self.torch;length=max(len(x) for x in clips);steps=math.ceil(length/512);buf=t.zeros((len(clips),steps*512))
        for i,a in enumerate(clips):buf[i,:len(a)].copy_(t.from_numpy(a))
        self.model.reset_states();out=[]
        with t.inference_mode():
            for i in range(steps):out.append(self.model(buf[:,i*512:(i+1)*512],16000).flatten())
        scores=t.stack(out,dim=1)
        return [scores[i,:math.ceil(len(a)/512)].tolist() for i,a in enumerate(clips)]
    def __call__(self,clips):
        t=self.torch;probabilities=self.probabilities(clips);out=[]
        class Replay:
            def __init__(self,probs):self.probs=probs;self.i=0
            def reset_states(self):self.i=0
            def __call__(self,x,sr):v=t.tensor(self.probs[self.i]);self.i+=1;return v
        for a,p in zip(clips,probabilities):
            ranges=self.timestamps(t.from_numpy(a),Replay(p),sampling_rate=16000,threshold=.5,min_speech_duration_ms=250,min_silence_duration_ms=100,speech_pad_ms=0)
            out.append([(int(x['start']),int(x['end'])) for x in ranges])
        return out

def analyze(engine,vad,rows,decoded):
    t=time.perf_counter();speech=vad([a for a,_ in decoded]);vad_s=time.perf_counter()-t;windows=[];details=[];perclip=[]
    for row,(pcm,proof),ranges in zip(rows,decoded,speech):
        entries=[]
        for start in range(0,len(pcm),64000):
            end=min(start+64000,len(pcm));voiced=sum(max(0,min(end,b)-max(start,a)) for a,b in ranges)
            if voiced==0:continue
            info={'start_s':start/16000,'end_s':end/16000,'speech_s':voiced/16000,'index':len(windows)};entries.append(info);windows.append(pcm[start:end]);details.append(info)
        perclip.append((row,proof,ranges,entries,len(pcm)/16000))
    t=time.perf_counter();probs=engine.predict(windows) if windows else [];gpu_s=time.perf_counter()-t;out=[]
    for row,proof,ranges,entries,duration in perclip:
        seconds={'tr':0.,'other':0.,'uncertain':0.};records=[]
        for e in entries:
            p=probs[e['index']];top=max(range(len(p)),key=p.__getitem__);confidence=p[top];category=('tr' if top==engine.tr else 'other') if confidence>=.7 and e['speech_s']>=.25 else 'uncertain';seconds[category]+=e['speech_s'];records.append({k:v for k,v in e.items() if k!='index'}|{'top_language':engine.languages[top],'top_probability':confidence,'Turkish_probability':p[engine.tr],'category':category,'language_probabilities':p,'numerical_threshold_nearby':abs(confidence-.7)<.02})
        total=sum(seconds.values());fractions={k:v/total if total else None for k,v in seconds.items()}
        out.append({'segment_id':row['segment_id'],'recording_id':row.get('recording_id'),'shard':row.get('shard'),'start_s':row.get('start_s'),'duration_s':duration,'speech_seconds':total,'speech_ranges_samples':ranges,'estimated_confident_Turkish_speech_fraction':fractions['tr'],'confident_other_speech_fraction':fractions['other'],'uncertain_speech_fraction':fractions['uncertain'],'estimated_Turkish_share_lower':fractions['tr'],'estimated_Turkish_share_upper':(seconds['tr']+seconds['uncertain'])/total if total else None,'windows':records,'decode_proof':proof,'new_TRAIN_admitted':False})
    return out,{'VAD_s':vad_s,'GPU_pipeline_s':gpu_s,'windows':len(windows),'audio_seconds':sum(len(a)/16000 for a,_ in decoded)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',default='1,2,4,8');ap.add_argument('--limit',type=int,default=128);args=ap.parse_args()
    guard=common.Guard();mem=guard.check();assert mem['commit_ratio']<.9 and mem['commit_headroom_bytes']>=8_000_000_000
    import numpy as np,torch
    guard.torch=torch;torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    try:import av
    except ImportError:sys.path.append(r'E:\gpu_training_scratchpad\scratch\CPU_validation_v2_env\Lib\site-packages');import av
    import paced_CPU_audio_v3 as decode
    engine=TinyEngine(guard);vad=BatchedVAD()
    leasep=ROOT/'work/gpu_training_owner.json';lease=json.loads(leasep.read_text());assert not lease.get('actual_GPU_running');lease.update(status='ROOT_tiny_real_source_pipeline_v5',actual_GPU_running=True,actual_owned_worker_pid=os.getpid());leasep.write_text(json.dumps(lease,indent=2)+'\n')
    clips,_=common.load_audio()
    # Verify state isolation: batched VAD vs separately reset single-item streams.
    test=[a for _,a in clips[:8]];batched=vad.probabilities(test);individual=[vad.probabilities([a])[0] for a in test];vad_error=max(abs(a-b) for aa,bb in zip(batched,individual) for a,b in zip(aa,bb));assert vad_error<1e-4,vad_error
    silence=np.zeros(160000,np.float32);assert vad([silence])==[[]]
    # Exact duration accounting invariant; every voiced sample belongs to one bucket.
    test_rows=[{'segment_id':sid} for sid,_ in clips];test_decoded=[(a,{'source':'verified cached original Opus'}) for _,a in clips]
    check,metrics=analyze(engine,vad,test_rows,test_decoded)
    assert all(x['speech_seconds']==0 or abs(x['estimated_confident_Turkish_speech_fraction']+x['confident_other_speech_fraction']+x['uncertain_speech_fraction']-1)<1e-8 for x in check)
    write('pipeline_v5_correctness.json',{'batched_VAD_max_probability_error':vad_error,'silent_control_passed':True,'duration_accounting_passed':True,'cached_clips':len(check),'metrics':metrics})
    write('pipeline_v5_cached_predictions.json',{'language_order':engine.languages,'clips':check})
    source=pathlib.Path(r'E:\gpu_training_scratchpad\analysis\CPU_validation_v8\new256_tiny_primary_manifest_v8.jsonl');rows=[json.loads(x) for x in source.read_text(encoding='utf-8').splitlines()][:args.limit]
    expected={x['segment_id']:x['decode']['PCM_f32le_sha256'] for x in [json.loads(s) for s in pathlib.Path(r'E:\gpu_training_scratchpad\analysis\CPU_validation_v8\new256_tiny_primary_ledger_v8.jsonl').read_text(encoding='utf-8').splitlines()]}
    results=[];ledger=[];inputguard=InputGuard(guard);total_start=time.perf_counter()
    def fetch(r):
        a,proof=decode.decode_paced(r,inputguard)
        if proof['PCM_f32le_sha256']!=expected[r['segment_id']]:raise ValueError('Source PCM differs from pinned existing v8 decode')
        proof['actual_new_reader_limit_bytes_s']=inputguard.rate;proof['decoder']='unchanged bounded timestamp-exact PyAV decoder with separately admitted fast benchmark I/O limit';return a,proof
    for workers in map(int,args.workers.split(',')):
        guard.check();start=time.perf_counter();before_bytes=inputguard.bytes;phase_metrics={'VAD_s':0.,'GPU_pipeline_s':0.,'windows':0,'audio_seconds':0.};records=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            pending={};next_submit=0
            def submit():
                nonlocal next_submit
                while next_submit<len(rows) and len(pending)<32:pending[next_submit]=pool.submit(fetch,rows[next_submit]);next_submit+=1
            submit()
            for at in range(0,len(rows),16):
                chunk=rows[at:at+16];data=[]
                for i in range(at,at+len(chunk)):data.append(pending.pop(i).result());submit()
                values,metrics=analyze(engine,vad,chunk,data);records.extend(values)
                for k in phase_metrics:phase_metrics[k]+=metrics[k]
        elapsed=time.perf_counter()-start;result={'decode_workers':workers,'clips':len(records),'wall_s':elapsed,'clips_per_s':len(records)/elapsed,'audio_seconds_per_wall_second':phase_metrics['audio_seconds']/elapsed,'compressed_bytes_read':inputguard.bytes-before_bytes,'actual_PCM_hash_matches':len(records),'peak_pending_decodes':32,'includes':'original tar seek/decode + VAD + GPU mel + LID + aggregation; excludes model startup and durable ledger writes','phase_metrics':phase_metrics,'memory':guard.check()};results.append(result);ledger=records
        print('END_TO_END '+json.dumps(result),flush=True);write('pipeline_v5_partial.json',{'results':results})
    write('pipeline_v5_predictions.json',{'language_order':engine.languages,'clips':ledger})
    write('pipeline_v5_report.json',dict(at_utc=common.now(),results=results,source_manifest=str(source),source_manifest_sha256=hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),limit=args.limit,total_compressed_bytes=inputguard.bytes,all_new_storage_on_E=True,cutoff_x=None,language_confidence_threshold=.7,window_seconds=4,VAD_threshold=.5,VAD_speech_padding_ms=0,models_are_not_ground_truth=True,new_TRAIN_admitted=0,memory=guard.check()))
    engine.close();guard.stop.set();del engine;torch.cuda.empty_cache();lease.update(status='ROOT_tiny_real_source_pipeline_window_closed',actual_GPU_running=False,actual_owned_worker_pid=None);leasep.write_text(json.dumps(lease,indent=2)+'\n');print('COMPLETE pipeline_v5',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:write('pipeline_v5_failure.json',dict(error=repr(e),at_utc=common.now(),pid=os.getpid()));raise
