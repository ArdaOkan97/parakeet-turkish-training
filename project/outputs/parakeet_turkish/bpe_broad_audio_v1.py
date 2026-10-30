"""Opt-in original-compressed broad reader. No quality or language selection.

Bound failures are unresolved input work, never an exclusion or consumed draw.
No PCM pool, model import, installed-environment modification or data-worker gate.
"""
from __future__ import annotations
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
from fractions import Fraction

ROOT=Path(r'E:\gpu_training_scratchpad')
AREA=ROOT/'analysis/BPE_weak_then_clean_v1'
AV_TARGET=ROOT/'scratch/ASR18_audio_av_16_0_1'
MAX_AUDIO_COLUMN_UNCOMPRESSED=512*1024**2
MAX_NATIVE_PCM_BYTES=256*1024**2
MAX_COMPRESSED_CLIP_BYTES=32*1024**2

class BoundViolation(RuntimeError):
    def __init__(self,kind,identity,actual,bound):
        self.receipt={'kind':kind,'unique_id':identity,'actual':actual,'bound':bound,
                      'draw_consumed':False,'training_exclusion':False,'full_pass_complete':False}
        super().__init__(json.dumps(self.receipt,sort_keys=True))

def require(ok,message):
    if not ok:raise ValueError(message)

def scoped(path):
    path=Path(path).resolve();require(path.is_relative_to(ROOT.resolve()),'New/source broad data must be E-only')
    return path

def record(path):
    path=Path(path)
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':str(path.resolve()),'sha256':digest,'bytes':path.stat().st_size}

class Extent(io.RawIOBase):
    """A seekable read-only view; libav cannot seek outside this TAR member."""
    def __init__(self,path,offset,size,guard):
        self.stream=scoped(path).open('rb');self.offset=offset;self.size=size;self.position=0
        self.guard=guard;self.logical_bytes_read=0
        require(0<=offset and 0<size and offset+size<=Path(path).stat().st_size,'Invalid original TAR extent')
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        value=offset if whence==0 else self.position+offset if whence==1 else self.size+offset if whence==2 else -1
        require(0<=value<=self.size,'Seek outside pinned member');self.position=value;return value
    def read(self,size=-1):
        self.guard();size=self.size-self.position if size<0 else min(size,self.size-self.position)
        self.stream.seek(self.offset+self.position);result=self.stream.read(size)
        self.position+=len(result);self.logical_bytes_read+=len(result);return result
    def readinto(self,buffer):
        data=self.read(len(buffer));buffer[:len(data)]=data;return len(data)
    def close(self):
        if not self.closed:self.stream.close()
        super().close()

def resample(pcm,sr):
    import numpy as np
    from scipy.signal import resample_poly
    require(pcm.ndim==1 and len(pcm)>0 and np.isfinite(pcm).all(),'Positive finite mono input required')
    if sr==16000:return np.ascontiguousarray(pcm,dtype=np.float32)
    divisor=math.gcd(sr,16000)
    output=resample_poly(pcm,16000//divisor,sr//divisor,window=('kaiser',5.0),padtype='constant')
    require(len(output)==math.ceil(len(pcm)*16000/sr),'Resampler full-length contract differs')
    return np.ascontiguousarray(output,dtype=np.float32)

class BroadAudioReader:
    def __init__(self,*,guard=lambda:None,admit=lambda name,budget:None,heldout_PCM_hashes=None):
        self.guard=guard;self.admit=admit;self.last=None;self.heldout_PCM_hashes=heldout_PCM_hashes
        plan_path=AREA/'preparation_plan_v1.json';plan=json.loads(plan_path.read_text(encoding='utf-8-sig'))
        self.plan=record(plan_path);self.jobs={str(Path(j['destination']).resolve()):j for j in plan['download_jobs']+plan['reused_archives']}
        self.verified={};self.calls=0

    def verify_source(self,path):
        path=scoped(path);key=str(path);job=self.jobs[key]
        stat=path.stat();signature=(stat.st_size,stat.st_mtime_ns)
        if self.verified.get(key)!=signature:
            receipt_path=path.with_suffix(path.suffix+'.receipt.json')
            receipt=json.loads(receipt_path.read_text(encoding='utf-8-sig'))
            require(receipt['sha256']==job['sha256'] and receipt['revision']==job['revision']
                    and stat.st_size==job['size'],'Pinned completed source receipt differs')
            self.verified[key]=signature
        return path,job

    def kiraat(self,row):
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        import soundfile as sf
        path,job=self.verify_source(row['shard']);index=int(row['row_index']);pf=pq.ParquetFile(path)
        require(0<=index<pf.metadata.num_rows,'Original KIRAAT row index invalid')
        group=0;local=index
        for number in range(pf.metadata.num_row_groups):
            n=pf.metadata.row_group(number).num_rows
            if local<n:group=number;break
            local-=n
        metadata=pf.metadata.row_group(group)
        size=sum(metadata.column(c).total_uncompressed_size for c in range(metadata.num_columns)
                 if metadata.column(c).path_in_schema.startswith('audio.'))
        if size>MAX_AUDIO_COLUMN_UNCOMPRESSED:raise BoundViolation('Parquet_audio_column_budget',row['unique_id'],size,MAX_AUDIO_COLUMN_UNCOMPRESSED)
        self.admit('parquet_audio_'+str(self.calls),size*2+100_000_000)
        selected=None;iterator=None;batch=None
        try:
            iterator=pf.iter_batches(batch_size=1,row_groups=[group],columns=['audio','id','source_id'],use_threads=False)
            for ordinal,batch in enumerate(iterator):
                self.guard()
                if ordinal!=local:continue
                item=batch.to_pylist()[0]
                require('kiraat:'+str(item['id'])==row['unique_id'] and item['source_id']==row['source_id'],'Parquet row identity differs')
                selected=item['audio']['bytes'];break
            require(isinstance(selected,bytes) and selected.startswith(b'fLaC'),'Expected embedded original FLAC bytes')
            if len(selected)>MAX_COMPRESSED_CLIP_BYTES:raise BoundViolation('compressed_clip_bytes',row['unique_id'],len(selected),MAX_COMPRESSED_CLIP_BYTES)
        finally:
            del iterator,batch,pf
            gc.collect();pa.default_memory_pool().release_unused()
        compressed_SHA=hashlib.sha256(selected).hexdigest()
        with sf.SoundFile(io.BytesIO(selected)) as audio:
            require(audio.samplerate==24000 and audio.channels==1,'Pinned KIRAAT mono24k source differs')
            if audio.frames*4>MAX_NATIVE_PCM_BYTES:raise BoundViolation('full_clip_PCM',row['unique_id'],audio.frames*4,MAX_NATIVE_PCM_BYTES)
            self.guard();pcm=audio.read(dtype='float32',always_2d=False);sr=audio.samplerate
        # The original timestamps refer to the parent recording. NEVER trim
        # these embedded clip samples by start/end a second time.
        output=resample(pcm,sr)
        return output,{'format':'KIRAAT_embedded_FLAC','source_file_sha256':job['sha256'],
            'original_row_index':index,'audio_column_uncompressed_bytes':size,'compressed_clip_bytes':len(selected),
            'compressed_clip_sha256':compressed_SHA,'native_sample_rate':sr,'native_samples':len(pcm),
            'full_embedded_clip_used':True,'source_start_end_applied_again':False,
            'metadata_duration_delta_seconds':len(pcm)/sr-float(row['duration'])}

    def yodas(self,row,*,linear=False):
        import numpy as np
        sys.path.insert(0,str(AV_TARGET)) if str(AV_TARGET) not in sys.path else None
        import av
        require(av.__version__=='16.0.1','Pinned isolated PyAV required')
        shard=str(row['shard']);require(shard.isdigit() and len(shard)==4,'Pinned shard identity required')
        path,job=self.verify_source(ROOT/f'datasets/yodas3/audio/{shard}.tar')
        index=scoped(ROOT/f'indexes/yodas3_tar_{shard}.sqlite')
        db=sqlite3.connect(index.as_uri()+'?mode=ro',uri=True)
        try:entry=db.execute('SELECT member_name,offset,bytes FROM recordings WHERE recording_id=?',(row['source_id'],)).fetchone()
        finally:db.close()
        require(entry is not None,'Recording not yet indexed; source row stays pending')
        name,offset,size=entry;require(Path(name).stem==row['source_id'] and name.endswith('.webm'),'Pinned member identity differs')
        start=Fraction(str(row['start']));end=Fraction(str(row['end']));require(0<=start<end,'Positive original segment interval required')
        with Extent(path,offset,size,self.guard) as extent,av.open(extent,mode='r',format='webm') as container:
            require(len(container.streams.audio)==1,'Exactly one original audio stream required')
            stream=container.streams.audio[0];stream.codec_context.thread_count=1
            first=next(container.decode(stream));require(first.pts is not None,'Explicit first decoded PTS required')
            origin=first.pts*first.time_base;sr=first.sample_rate
            require(sr>0,'Native sample rate missing')
            begin=round(start*sr);finish=round(end*sr);count=finish-begin
            if count*4>MAX_NATIVE_PCM_BYTES:raise BoundViolation('full_segment_PCM',row['unique_id'],count*4,MAX_NATIVE_PCM_BYTES)
            self.admit('yodas_PCM_'+str(self.calls),count*8+50_000_000)
            pcm=np.empty(count,dtype=np.float32);written=0;decoded=0
            # Seek includes two seconds of codec preroll. Frame PTS, not packet
            # counts or a concatenation offset, selects the exact source span.
            target=origin+max(Fraction(0),start-2)
            container.seek(math.floor(target/stream.time_base),stream=stream,backward=True,any_frame=False)
            if linear:container.seek(math.floor(origin/stream.time_base),stream=stream,backward=True,any_frame=False)
            first_selected_pts=None
            for frame in container.decode(stream):
                self.guard();decoded+=1
                require(frame.pts is not None and frame.sample_rate==sr,'Stable explicit decoded PTS/rate required')
                frame_begin=round((frame.pts*frame.time_base-origin)*sr);frame_end=frame_begin+frame.samples
                if frame_end<=begin:continue
                if frame_begin>=finish:break
                raw=frame.to_ndarray()
                require(frame.format.is_planar and raw.ndim==2,'Pinned decoded planar audio required')
                if np.issubdtype(raw.dtype,np.integer):raw=raw.astype(np.float32)/float(2**(np.iinfo(raw.dtype).bits-1))
                else:raw=raw.astype(np.float32,copy=False)
                raw=raw.mean(axis=0,dtype=np.float32) if raw.shape[0]>1 else raw[0]
                low=max(begin,frame_begin);high=min(finish,frame_end)
                require(low-begin==written,'Decoded source PTS gap/overlap; never synthesize missing samples')
                pcm[low-begin:high-begin]=raw[low-frame_begin:high-frame_begin];written=high-begin
                if first_selected_pts is None:first_selected_pts=str(frame.pts*frame.time_base)
                if written==count:break
            require(written==count,'Original interval extends beyond decoded source; row stays unresolved')
            output=resample(pcm,sr)
            info={'format':'YODAS_Opus_WebM_TAR_extent','source_file_sha256':job['sha256'],'member_name':name,
                'member_offset':offset,'member_bytes':size,'member_logical_read_bytes':extent.logical_bytes_read,
                'native_sample_rate':sr,'native_samples':count,'decoded_frames':decoded,'PTS_origin':str(origin),
                'source_interval_seconds':[str(start),str(end)],'first_selected_frame_PTS':first_selected_pts,
                'linear_reference':linear,'codec_preroll_seconds':2,'temporary_member_or_PCM_written':False,
                'PyAV_version':av.__version__}
        return output,info

    def __call__(self,row,*,linear=False):
        import numpy as np
        self.guard();self.calls+=1;began=time.perf_counter()
        require(row['dataset'] in ('kiraat','alania_yodas3'),'Only declared original broad datasets')
        audio,info=self.kiraat(row) if row['dataset']=='kiraat' else self.yodas(row,linear=linear)
        require(audio.dtype==np.float32 and audio.ndim==1 and len(audio)>0 and np.isfinite(audio).all(),'Finite full16kHz mono samples required')
        canonical=np.clip(np.rint(audio*32768),-32768,32767).astype('<i2')
        digest=hashlib.sha256(canonical.tobytes()).hexdigest()
        if self.heldout_PCM_hashes is not None:require(digest not in self.heldout_PCM_hashes,'Exact heldout PCM overlap; heldout-exclusion ledger required before any draw')
        self.last={'unique_id':row['unique_id'],'source_id':row['source_id'],**info,
            'decoded_sample_rate':16000,'decoded_samples':len(audio),'duration_seconds':len(audio)/16000,
            'encoder_frames':(len(audio)//160+7)//8,'PCM_f32_sha256':hashlib.sha256(audio.tobytes()).hexdigest(),
            'PCM_i16_sha256':digest,'raw_label_sha256':hashlib.sha256(row['text'].encode()).hexdigest(),
            'original_flags_json':row['original_flags_json'],'quality_or_language_filters':[],
            'exact_heldout_PCM_check_performed':self.heldout_PCM_hashes is not None,
            'resample':'scipy.resample_poly Kaiser5.0 constant; full clip; ceil output length',
            'wall_seconds':time.perf_counter()-began,'training_admitted':False}
        return audio
