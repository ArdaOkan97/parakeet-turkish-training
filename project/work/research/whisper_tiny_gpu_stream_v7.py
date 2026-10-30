"""Bounded overlapped source decode, compiled CPU VAD, and graph GPU tiny."""
import pathlib,os,sys,json,time,math,threading,queue,concurrent.futures,hashlib,argparse,sqlite3,io
import whisper_tiny_gpu_batch_v1 as common
from whisper_tiny_gpu_pipeline_v5 import TinyEngine,BatchedVAD,InputGuard,analyze
AREA=common.AREA;ROOT=common.ROOT

def speech_ranges(probs,n):
    # Pinned Silero timestamp semantics for threshold=.5,neg=.35,min250ms,
    # min_silence100ms,max_duration=infinity,padding=0. No tensor/scalar bridge.
    triggered=False;start=None;temporary=0;out=[]
    for i,p in enumerate(probs):
        at=i*512
        if p>=.5 and temporary:temporary=0
        if p>=.5 and not triggered:triggered=True;start=at;continue
        if p<.35 and triggered:
            if not temporary:temporary=at
            if at-temporary<1600:continue
            if temporary-start>4000:out.append((start,temporary))
            start=None;temporary=0;triggered=False
    if start is not None and n-start>4000:out.append((start,n))
    return out

class FastVAD(BatchedVAD):
    def probabilities(self,clips):
        t=self.torch;steps=math.ceil(max(len(x) for x in clips)/512);buf=t.zeros((len(clips),steps*512))
        for i,a in enumerate(clips):buf[i,:len(a)].copy_(t.from_numpy(a))
        with t.inference_mode():scores=self.model.audio_forward(buf,16000)
        return [scores[i,:math.ceil(len(a)/512)].tolist() for i,a in enumerate(clips)]
    def __call__(self,clips):return [speech_ranges(p,len(a)) for a,p in zip(clips,self.probabilities(clips))]

class FastSource:
    def __init__(self,rows,guard):
        self.guard=guard;self.local=threading.local();self.members={};self.receipts={}
        e=pathlib.Path(r'E:\gpu_training_scratchpad')
        for shard in set(r['shard'] for r in rows):
            archive=e/f'datasets/yodas3/audio/{shard}.tar';receipt=json.loads(archive.with_suffix('.tar.receipt.json').read_text());assert receipt['revision']=='81d802f45a1c1e1f5331260484dacf1fbea81981' and archive.stat().st_size==receipt['bytes'];self.receipts[shard]=(archive,receipt)
            with sqlite3.connect((e/f'indexes/yodas3_tar_{shard}.sqlite').as_uri()+'?mode=ro',uri=True) as db:
                for row in rows:
                    if row['shard']!=shard:continue
                    rec=db.execute('SELECT offset,bytes FROM recordings WHERE recording_id=?',(row['recording_id'],)).fetchone()
                    if not rec:raise ValueError('Missing source member')
                    self.members[(shard,row['recording_id'])]=rec
    def decode(self,row):
        import av,numpy as np
        shard=row['shard'];start=row['start_s'];duration=row['duration_s'];offset,size=self.members[(shard,row['recording_id'])];archive,receipt=self.receipts[shard]
        if size>256*1024*1024:raise ValueError('Compressed member exceeds256MiB')
        if not 2<=duration<=20 or start<0:raise ValueError('Source bounds changed')
        if not hasattr(self.local,'handles'):self.local.handles={}
        if shard not in self.local.handles:self.local.handles[shard]=archive.open('rb')
        handle=self.local.handles[shard];guard=self.guard
        class Reader(io.RawIOBase):
            def __init__(self):self.pos=0;self.bytes=0
            def readable(self):return True
            def seekable(self):return True
            def tell(self):return self.pos
            def seek(self,x,whence=0):
                at=x if whence==0 else self.pos+x if whence==1 else size+x
                if not 0<=at<=size:raise ValueError('Member seek outside bounds')
                self.pos=at;return at
            def read(self,n=-1):
                n=min(size-self.pos,65536 if n<0 else min(n,65536))
                if n<=0:return b''
                guard.pace(n);handle.seek(offset+self.pos);data=handle.read(n)
                if len(data)!=n:raise ValueError('Short original member read')
                self.pos+=n;self.bytes+=n;return data
        n=round(duration*16000);pcm=np.zeros(n,np.float32);coverage=np.zeros(n,np.bool_);started=time.monotonic()
        def retain(frame):
            if frame.pts is None:raise ValueError('Missing PTS')
            at=round((float(frame.pts*frame.time_base)-start)*16000);values=frame.to_ndarray().reshape(-1);lo=max(0,at);hi=min(n,at+len(values))
            if hi>lo:pcm[lo:hi]=values[lo-at:hi-at];coverage[lo:hi]=True
        reader=Reader()
        with av.open(reader,mode='r',format='matroska',buffer_size=32768) as container:
            stream=container.streams.audio[0];stream.codec_context.thread_count=1;stream.codec_context.thread_type='NONE';container.seek(max(0,round((start-.10)*1_000_000)),backward=True,any_frame=False);res=av.AudioResampler(format='fltp',layout='mono',rate=16000)
            for frame in container.decode(stream):
                guard.wait()
                if time.monotonic()-started>90:raise TimeoutError('Original90s decoder bound')
                if frame.pts is None:raise ValueError('Missing PTS')
                if float(frame.pts*frame.time_base)>start+duration+.15:break
                for out in res.resample(frame):retain(out)
            for out in res.resample(None):retain(out)
        if not np.isfinite(pcm).all() or coverage.mean()<.995:raise ValueError('Invalid PCM or coverage')
        return pcm,{'PCM_f32le_sha256':hashlib.sha256(pcm.tobytes()).hexdigest(),'timestamp_coverage_fraction':float(coverage.mean()),'compressed_bytes_actually_read':reader.bytes,'archive_revision':receipt['revision'],'archive_sha256':receipt['sha256'],'decoded_samples':n,'decoder_threads_per_worker':1,'no_PCM_written':True}

def stream(engine,vad,rows,source,guard,workers=2,group=64,ledger_path=None,expected=None):
    q=queue.Queue(maxsize=2);stop=threading.Event();stats={'VAD_s':0.,'decode_wait_s':0.};error=[]
    def produce():
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                pending={};next_id=0
                def submit():
                    nonlocal next_id
                    while next_id<len(rows) and len(pending)<64 and not stop.is_set():pending[next_id]=pool.submit(source.decode,rows[next_id]);next_id+=1
                submit()
                for at in range(0,len(rows),group):
                    if stop.is_set():break
                    chunk=rows[at:at+group];start=time.perf_counter();decoded=[]
                    for i in range(at,at+len(chunk)):
                        item=pending.pop(i).result()
                        if expected is not None and item[1]['PCM_f32le_sha256']!=expected[rows[i]['segment_id']]:raise ValueError('PCM differs from pinned baseline')
                        decoded.append(item);submit()
                    stats['decode_wait_s']+=time.perf_counter()-start;start=time.perf_counter();speech=vad([a for a,_ in decoded]);stats['VAD_s']+=time.perf_counter()-start
                    while not stop.is_set():
                        try:q.put((chunk,decoded,speech),timeout=.5);break
                        except queue.Full:continue
        except BaseException as e:error.append(e)
        finally:
            while not stop.is_set():
                try:q.put(None,timeout=.5);break
                except queue.Full:continue
    thread=threading.Thread(target=produce,daemon=True);started=time.perf_counter();thread.start();out=[];gpu_s=0.;audio=0.;windows=0;held=[]
    try:
        file=ledger_path.open('w',encoding='utf-8') if ledger_path else None
        while True:
            item=q.get()
            if item is None:break
            chunk,decoded,speech=item;result,m=analyze(engine,lambda _:speech,chunk,decoded);out.extend(result);gpu_s+=m['GPU_pipeline_s'];audio+=m['audio_seconds'];windows+=m['windows'];held.extend(zip(chunk,[a for a,_ in decoded]))
            if file:
                for row in result:file.write(json.dumps(row,separators=(',',':'),ensure_ascii=False)+'\n')
                file.flush()
        if error:raise error[0]
        if file:file.flush();os.fsync(file.fileno());file.close()
    finally:stop.set();thread.join(timeout=100)
    elapsed=time.perf_counter()-started
    return out,dict(clips=len(out),wall_s=elapsed,clips_per_s=len(out)/elapsed,audio_seconds=audio,audio_seconds_per_wall_second=audio/elapsed,windows=windows,GPU_pipeline_s=gpu_s,**stats),held

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=256);ap.add_argument('--workers',default='1,2,4,8');ap.add_argument('--group',type=int,default=64);ap.add_argument('--threads',type=int,default=4);args=ap.parse_args()
    guard=common.Guard();memory=guard.check();assert memory['commit_ratio']<.9 and memory['commit_headroom_bytes']>=8_000_000_000
    import numpy as np,torch
    guard.torch=torch;torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    try:import av
    except ImportError:sys.path.append(r'E:\gpu_training_scratchpad\scratch\CPU_validation_v2_env\Lib\site-packages');import av
    leasep=ROOT/'work/gpu_training_owner.json';lease=json.loads(leasep.read_text());assert not lease.get('actual_GPU_running');lease.update(status='ROOT_tiny_overlapped_source_v7',actual_GPU_running=True,actual_owned_worker_pid=os.getpid());leasep.write_text(json.dumps(lease,indent=2)+'\n')
    engine=TinyEngine(guard);vad=FastVAD();clips,_=common.load_audio();test=[a for _,a in clips];probs=vad.probabilities(test)
    class Replay:
        def __init__(self,p):self.p=p;self.i=0
        def reset_states(self):self.i=0
        def __call__(self,a,sr):v=torch.tensor(self.p[self.i]);self.i+=1;return v
    for a,p in zip(test,probs):
        ref=vad.timestamps(torch.from_numpy(a),Replay(p),sampling_rate=16000,threshold=.5,min_speech_duration_ms=250,min_silence_duration_ms=100,speech_pad_ms=0);assert speech_ranges(p,len(a))==[(x['start'],x['end']) for x in ref]
    pin=pathlib.Path(r'E:\gpu_training_scratchpad\analysis\CPU_validation_v8\new256_tiny_primary_manifest_v8.jsonl');rows=[json.loads(x) for x in pin.read_text(encoding='utf-8').splitlines()][:args.limit]
    expected={x['segment_id']:x['decode']['PCM_f32le_sha256'] for x in [json.loads(s) for s in pathlib.Path(r'E:\gpu_training_scratchpad\analysis\CPU_validation_v8\new256_tiny_primary_ledger_v8.jsonl').read_text(encoding='utf-8').splitlines()]}
    inputguard=InputGuard(guard);source=FastSource(rows,inputguard);results=[]
    for workers in map(int,args.workers.split(',')):
        before=inputguard.bytes;out,metrics,held=stream(engine,vad,rows,source,guard,workers,args.group,AREA/f'stream_v7_workers{workers}_ledger.jsonl',expected);metrics.update(workers=workers,VAD_group=args.group,CPU_threads=args.threads,compressed_bytes_read=inputguard.bytes-before,PCM_hash_matches=len(out),memory=guard.check(),includes='original archive reads, exact decode, VAD, GPU mel+tiny, duration aggregation and durable ledger fsync; OS cache not flushed')
        results.append(metrics);print('END_TO_END '+json.dumps(metrics),flush=True);common.write('stream_v7_partial.json',dict(results=results))
    common.write('stream_v7_report.json',dict(at_utc=common.now(),results=results,official_timestamp_parser_exact_matches=len(test),model_language_only=True,cutoff_x=None,new_TRAIN_admitted=0,memory=guard.check()))
    engine.close();guard.stop.set();del engine;torch.cuda.empty_cache();lease.update(actual_GPU_running=False,status='ROOT_tiny_overlapped_source_v7_closed',actual_owned_worker_pid=None);leasep.write_text(json.dumps(lease,indent=2)+'\n');print('COMPLETE stream_v7',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:common.write('stream_v7_failure.json',dict(error=repr(e),at_utc=common.now(),pid=os.getpid()));raise
