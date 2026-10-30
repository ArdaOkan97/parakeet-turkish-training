"""Bounded original-source language screening; one GPU, CPU-only codecs.

Fixed 4s windows, Silero speech-duration weighting, raw language probabilities.
The 90% cutoff is a candidate screen and does not admit data to training.
"""
import os,sys,pathlib,json,time,threading,queue,concurrent.futures,multiprocessing
import hashlib,sqlite3,argparse,datetime,subprocess,statistics,math,shutil
import whisper_tiny_gpu_batch_v1 as common
from whisper_tiny_gpu_pipeline_v5 import TinyEngine,analyze
from whisper_tiny_gpu_stream_v7 import FastSource,FastVAD
AREA=common.AREA;ROOT=common.ROOT
_SOURCE=None

class CodecGuard:
    def __init__(self):
        import psutil
        self.proc=psutil.Process();self.bytes=0;self.last=0.;self.start=time.monotonic();self.paused_seconds=0.
    def wait(self):
        if time.monotonic()-self.last>.5:
            self.last=time.monotonic();m=common.base.memory()
            if m['commit_ratio']>=.95 or m['commit_headroom_bytes']<2_000_000_000 or self.proc.memory_info().private>2_000_000_000:raise RuntimeError('Codec resource protection')
    def pace(self,n):
        self.wait();self.bytes+=n
        if self.bytes>45_000_000_000:raise RuntimeError('Declared per-codec45GB read budget')
        delay=self.bytes/(32*1024*1024)-(time.monotonic()-self.start)
        if delay>0:time.sleep(delay)

def initialize_codec(spec):
    global _SOURCE
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    try:import av
    except ImportError:sys.path.append(r'E:\gpu_training_scratchpad\scratch\CPU_validation_v2_env\Lib\site-packages');import av
    if 'torch' in sys.modules:raise RuntimeError('Codec child loaded Torch')
    _SOURCE=FastSource.__new__(FastSource);_SOURCE.guard=CodecGuard();_SOURCE.local=threading.local();_SOURCE.members=spec['members'];_SOURCE.receipts=spec['receipts']

def decode_task(row):
    pcm,proof=_SOURCE.decode(row)
    proof.update(codec_worker_PID=os.getpid(),codec_worker_Torch_loaded=False,codec_worker_GPU_visible=False)
    return pcm,proof

def source_spec(rows):
    e=AREA.parents[1];members={};receipts={}
    for shard in sorted({r['shard'] for r in rows}):
        p=e/f'datasets/yodas3/audio/{shard}.tar';receipt=json.loads(p.with_suffix('.tar.receipt.json').read_text())
        assert p.stat().st_size==receipt['bytes'] and receipt['revision']=='81d802f45a1c1e1f5331260484dacf1fbea81981'
        receipts[shard]=(p,receipt);wanted={r['recording_id'] for r in rows if r['shard']==shard}
        with sqlite3.connect((e/f'indexes/yodas3_tar_{shard}.sqlite').as_uri()+'?mode=ro',uri=True) as db:
            for rid,offset,size in db.execute('SELECT recording_id,offset,bytes FROM recordings'):
                if rid in wanted:members[shard,rid]=(offset,size)
        assert all((shard,rid) in members for rid in wanted)
    return {'members':members,'receipts':receipts}

class GPUUsage:
    def __init__(self):
        self.samples=[];self.stop=threading.Event();self.proc=None
        def run():
            try:
                self.proc=subprocess.Popen(['nvidia-smi','--query-gpu=utilization.gpu,power.draw,memory.used','--format=csv,noheader,nounits','-lms','1000'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,creationflags=0x08000000)
                for line in self.proc.stdout:
                    if self.stop.is_set():break
                    try:self.samples.append([time.perf_counter()]+[float(x.strip()) for x in line.split(',')])
                    except ValueError:pass
            except Exception as e:self.error=repr(e)
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start()
    def finish(self,start,end):
        self.stop.set()
        if self.proc:self.proc.terminate()
        self.thread.join(timeout=3)
        s=[x for x in self.samples if start<=x[0]<=end]
        return dict(samples=len(s),mean_GPU_utilization_percent=statistics.mean(x[1] for x in s) if s else None,max_GPU_utilization_percent=max((x[1] for x in s),default=None),mean_power_W=statistics.mean(x[2] for x in s) if s else None,max_total_GPU_memory_MiB=max((x[3] for x in s),default=None),poll_error=getattr(self,'error',None))

class AuditedEngine:
    def __init__(self,engine):self.engine=engine;self.samples=[];self.seen=0;self.boundary=0
    def __getattr__(self,k):return getattr(self.engine,k)
    def predict(self,waves):
        p=self.engine.predict(waves)
        for wave,score in zip(waves,p):
            self.seen+=1;near=abs(max(score)-.7)<.025
            if (self.seen%23==0 and len(self.samples)<128) or (near and self.boundary<64):
                self.samples.append((wave.copy(),score,near));self.boundary+=int(near)
        return p

def classify(row,cutoff):
    lo=row['estimated_Turkish_share_lower'];hi=row['estimated_Turkish_share_upper']
    row['Turkish_speech_share_cutoff']=cutoff
    row['language_screening_decision']='no_detected_speech' if lo is None else 'pass' if lo>=cutoff else 'below_cutoff' if hi<cutoff else 'uncertain_review'
    row['below_conservative_Turkish_cutoff']=lo is None or lo<cutoff
    row['measurement']='Estimated language share of VAD speech duration;4s resolution;not ground truth'
    return row

def run_stream(engine,vad,rows,spec,guard,prefix,cutoff=.9,workers=4,group=64):
    q=queue.Queue(2);stop=threading.Event();errors=[];stats={'decode_wait_s':0.,'VAD_s':0.};deadline=datetime.datetime.fromisoformat(json.loads((AREA/'timebox.json').read_text())['deadline_utc']).timestamp()-180
    def put(item):
        while not stop.is_set():
            try:q.put(item,timeout=.5);return
            except queue.Full:pass
    def produce():
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_codec,initargs=(spec,)) as pool:
                pending={};next_id=0
                def submit():
                    nonlocal next_id
                    while next_id<len(rows) and len(pending)<128 and not stop.is_set():pending[next_id]=pool.submit(decode_task,rows[next_id]);next_id+=1
                submit()
                for at in range(0,len(rows),group):
                    if stop.is_set():break
                    chunk=rows[at:at+group];decoded=[];t=time.perf_counter()
                    for i in range(at,at+len(chunk)):
                        decoded.append(pending.pop(i).result());submit()
                    stats['decode_wait_s']+=time.perf_counter()-t;t=time.perf_counter();speech=vad([a for a,_ in decoded]);stats['VAD_s']+=time.perf_counter()-t;put((chunk,decoded,speech))
                if stop.is_set():
                    for future in pending.values():future.cancel()
        except BaseException as e:errors.append(e)
        finally:put(None)
    thread=threading.Thread(target=produce,daemon=True);usage=GPUUsage();start=time.perf_counter();thread.start();counts={};clips=0;audio=0.;windows=0;inference=0.;bytes_read=0;pidset=set();speech_s=0.;last_status=0;complete=False
    ledger=AREA/(prefix+'_ledger.jsonl');decisions=AREA/(prefix+'_decisions.jsonl')
    try:
        with ledger.open('x',encoding='utf-8') as f,decisions.open('x',encoding='utf-8') as ids:
            while True:
                guard.check()
                try:item=q.get(timeout=1)
                except queue.Empty:
                    if errors:raise errors[0]
                    continue
                if item is None:complete=True;break
                chunk,decoded,ranges=item;out,m=analyze(engine,lambda _:ranges,chunk,decoded)
                for r in out:
                    classify(r,cutoff);d=r['language_screening_decision'];counts[d]=counts.get(d,0)+1;bytes_read+=r['decode_proof']['compressed_bytes_actually_read'];pidset.add(r['decode_proof']['codec_worker_PID']);speech_s+=r['speech_seconds'];f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n');ids.write(json.dumps({'segment_id':r['segment_id'],'decision':d,'estimated_Turkish_share_lower':r['estimated_Turkish_share_lower'],'estimated_Turkish_share_upper':r['estimated_Turkish_share_upper']},separators=(',',':'))+'\n')
                clips+=len(out);audio+=m['audio_seconds'];windows+=m['windows'];inference+=m['GPU_pipeline_s'];f.flush();ids.flush()
                if f.tell()>3_000_000_000 or bytes_read>160_000_000_000:raise RuntimeError('Declared ledger3GB or aggregate160GB read bound')
                if time.perf_counter()-last_status>15:
                    last_status=time.perf_counter();assert shutil.disk_usage(AREA).free>100_000_000_000
                    common.write(prefix+'_status.json',dict(at_utc=common.now(),phase='running',pid=os.getpid(),clips=clips,target=len(rows),windows=windows,counts=counts,wall_s=time.perf_counter()-start,memory=guard.check()))
                if time.time()>=deadline:break
            f.flush();os.fsync(f.fileno());ids.flush();os.fsync(ids.fileno())
        if errors:raise errors[0]
    finally:
        stop.set();thread.join(timeout=100)
        if thread.is_alive():raise RuntimeError('Codec producer did not close')
    end=time.perf_counter();gpu=usage.finish(start,end)
    metrics=dict(at_utc=common.now(),clips=clips,target=len(rows),complete=complete and clips==len(rows),wall_s=end-start,clips_per_s=clips/(end-start),audio_seconds=audio,audio_seconds_per_wall_second=audio/(end-start),speech_seconds=speech_s,windows=windows,GPU_pipeline_s=inference,counts=counts,cutoff=cutoff,compressed_bytes_read=bytes_read,codec_worker_PIDs=sorted(pidset),codec_workers=workers,CPU_children_GPU_and_Torch=False,telemetry=gpu,**stats,memory=guard.check(),new_TRAIN_admitted=0)
    common.write(prefix+'_report.json',metrics);print('REPORT '+json.dumps(metrics),flush=True);return metrics

def audit(engine,samples,guard,prefix):
    import torch,numpy as np
    from transformers import WhisperForConditionalGeneration
    wrapper=torch.nn.functional.scaled_dot_product_attention;torch.nn.functional.scaled_dot_product_attention=engine.original_sdpa
    folder=json.loads((AREA/'HF_download_receipt.json').read_text())['model_directory'];model=WhisperForConditionalGeneration.from_pretrained(folder,local_files_only=True,dtype=torch.float32,attn_implementation='sdpa',low_cpu_mem_usage=True).eval().to('cuda');reference=[]
    try:
        with torch.inference_mode():
            for at in range(0,len(samples),4):
                guard.check();part=samples[at:at+4];features=engine.extractor([x[0] for x in part],sampling_rate=16000,return_tensors='pt').input_features.to('cuda');sot=torch.full((len(part),1),model.generation_config.decoder_start_token_id,device='cuda',dtype=torch.long)
                logits=model(input_features=features,decoder_input_ids=sot,use_cache=False).logits[:,0].index_select(1,engine.ids);reference.extend(logits.softmax(-1).cpu().tolist())
        errors=[];labels=0;categories=0;near_changes=0
        def cat(p):
            k=max(range(len(p)),key=p.__getitem__)
            return ('tr' if k==engine.tr else 'other') if p[k]>=.7 else 'uncertain'
        for (_,p,near),ref in zip(samples,reference):
            errors.append(max(abs(a-b) for a,b in zip(p,ref)));labels+=int(max(range(len(p)),key=p.__getitem__)!=max(range(len(ref)),key=ref.__getitem__));changed=cat(p)!=cat(ref);categories+=changed;near_changes+=int(changed and near)
        value=dict(windows=len(samples),reference='Canonical HF full-vocabulary one-SOT-step,FP32,CPU canonical feature extractor,full3000frames',optimized='Full multilingual tiny FP16,fused,CUDAgraph,GPUmel',max_probability_difference=max(errors,default=0),top_language_changes=labels,confidence70_category_changes=categories,category_changes_in_near_threshold_set=near_changes,near_threshold_windows=sum(x[2] for x in samples),all_category_changes_explained_by_near_threshold=categories==near_changes)
        common.write(prefix+'_FP32_audit.json',value);print('FP32_AUDIT '+json.dumps(value),flush=True);return value
    finally:del model;torch.cuda.empty_cache();torch.nn.functional.scaled_dot_product_attention=wrapper

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--prefix',required=True);ap.add_argument('--pilot',action='store_true');ap.add_argument('--cutoff',type=float,default=.9);args=ap.parse_args()
    manifest=pathlib.Path(args.manifest);rows=[json.loads(x) for x in manifest.read_text(encoding='utf-8').splitlines()];assert len({x['segment_id'] for x in rows})==len(rows)
    guard=common.Guard();m=guard.check();assert m['commit_ratio']<.9 and m['commit_headroom_bytes']>=8_000_000_000 and shutil.disk_usage(AREA).free>100_000_000_000
    import torch
    guard.torch=torch;torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    leasep=ROOT/'work/gpu_training_owner.json';lease=json.loads(leasep.read_text());assert lease['owner']=='/root' and not lease.get('actual_GPU_running');lease.update(status='ROOT_tiny_original_source_v9',actual_GPU_running=True,actual_training_optimizer_running=False,actual_owned_worker_pid=os.getpid(),source=str(pathlib.Path(__file__)));leasep.write_text(json.dumps(lease,indent=2)+'\n')
    engine=TinyEngine(guard);vad=FastVAD();spec=source_spec(rows)
    common.write(args.prefix+'_admission.json',dict(at_utc=common.now(),manifest=str(manifest),manifest_sha256=hashlib.file_digest(manifest.open('rb'),'sha256').hexdigest(),source_sha256=hashlib.file_digest(pathlib.Path(__file__).open('rb'),'sha256').hexdigest(),target=len(rows),cutoff=args.cutoff,confidence_threshold=.7,window_seconds=4,full_encoder_frames=3000,raw_probability_language_order=engine.languages,memory=m,storage_E_only=True,new_TRAIN_admitted=0))
    phases=[]
    if args.pilot:
        audited=AuditedEngine(engine);phases.append(run_stream(audited,vad,rows,spec,guard,args.prefix+'_random',args.cutoff));value=audit(engine,audited.samples,guard,args.prefix)
        if value['confidence70_category_changes'] and not value['all_category_changes_explained_by_near_threshold']:raise RuntimeError('Unexplained confidence-category audit change')
    rows.sort(key=lambda r:(int(r['duration_s']/2),r['shard'],r['recording_id'],r['start_s']))
    phases.append(run_stream(engine,vad,rows,spec,guard,args.prefix+'_duration_ordered',args.cutoff))
    common.write(args.prefix+'_complete.json',dict(at_utc=common.now(),actual_native_work_complete=True,phases=phases,source_manifest=str(manifest),TRAIN_admitted=0))
    engine.close();guard.stop.set();del engine;torch.cuda.empty_cache();lease.update(status='ROOT_tiny_original_source_v9_closed',actual_GPU_running=False,actual_owned_worker_pid=None);leasep.write_text(json.dumps(lease,indent=2)+'\n');print('COMPLETE '+args.prefix,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:common.write('production_v9_failure_'+str(os.getpid())+'.json',dict(error=repr(e),at_utc=common.now(),pid=os.getpid()));raise
