"""NEW bounded original compressed reader; frozen readers remain unchanged.

Original FLAC bytes, mono/resample rules, Opus PTS intervals and full labels
are preserved. No whole-wave quantization/tobytes hash duplication is used.
"""
import gc,hashlib,io,json,math,sqlite3,sys
from fractions import Fraction
from pathlib import Path
from bpe_broad_audio_FLAC_records_v1 import BroadAudioReader as FrozenReader
from bpe_broad_audio_v1 import ROOT,AV_TARGET,Extent as FrozenExtent,resample,record,scoped,require
from fastmss_raw_lifetime_v1 import canonical_source_path,chunked_hashes,MEDIA_PER_JOB

class BorrowedCompressedIO(io.RawIOBase):
    """Seekable libsndfile view; never duplicates the complete FLAC bytes."""
    def __init__(self,media):self.view=memoryview(media);self.position=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        target=offset if whence==0 else self.position+offset if whence==1 else len(self.view)+offset if whence==2 else -1
        require(type(target) is int and 0<=target<=len(self.view),'Unsupported compressed seek unresolved')
        self.position=target;return target
    def read(self,size=-1):
        require(type(size) is int and 0<=size<=1024**2,'Bounded compressed callback read <=1MiB; no whole-buffer duplicate')
        end=min(len(self.view),self.position+size);value=bytes(self.view[self.position:end]);self.position=end;return value
    def readinto(self,buffer):
        require(len(buffer)<=1024**2,'Bounded borrowed compressed readinto')
        end=min(len(self.view),self.position+len(buffer));count=end-self.position
        buffer[:count]=self.view[self.position:end];self.position=end;return count

class BoundedExtent(FrozenExtent):
    def read(self,size=-1):
        require(type(size) is int and 0<=size<=1024**2,'Unresolved compressed callback request: explicit <=1MiB only')
        return super().read(size)

class BoundedOriginalReader(FrozenReader):
    def __init__(self,*,ledger,**arguments):
        super().__init__(**arguments);self.ledger=ledger;self.active=None

    def read_original(self,row,*,job_key,declaration):
        import numpy as np
        require(self.active is None,'Serial original decoder only')
        self.active={'key':job_key,'declaration':declaration,'plan':self.ledger.jobs[job_key]['plan']}
        try:
            self.guard();self.calls+=1
            with self.ledger.stage(job_key,'decode_resample'):
                audio,info=self.kiraat(row) if row['dataset']=='kiraat' else self.yodas(row)
                require(audio.dtype==np.float32 and audio.ndim==1 and audio.flags.c_contiguous and len(audio)==declaration['clean16k_samples'],'Exact full declared mono16k return buffer')
                self.ledger.observe(job_key,audio,name='whole_clean_float32',maximum_bytes=4*declaration['clean16k_samples'])
            with self.ledger.stage(job_key,'canonical_hash'):
                f32,i16=chunked_hashes(audio,guard=self.guard)
            require(self.heldout_PCM_hashes is not None and i16 not in self.heldout_PCM_hashes,'Unresolved heldout PCM overlap; no draw or exclusion')
            path=canonical_source_path(row,ROOT)
            self.last={'unique_id':row['unique_id'],'source_id':row['source_id'],**info,
                'decoded_sample_rate':16000,'decoded_samples':len(audio),'duration_seconds':len(audio)/16000,
                'encoder_frames':(len(audio)//160+7)//8,'PCM_f32_sha256':f32,'PCM_i16_sha256':i16,
                'actual_original_source_path':str(path),'raw_label_sha256':hashlib.sha256(row['text'].encode()).hexdigest(),
                'original_flags_json':row['original_flags_json'],'quality_or_language_filters':[],
                'exact_heldout_PCM_check_performed':True,'resample':'scipy.resample_poly Kaiser5.0 constant; full clip; ceil output length',
                'training_admitted':False,'bounded_hash_chunk_samples':65536}
            return audio
        finally:self.active=None

    def kiraat(self,row):
        import numpy as np
        import soundfile as sf
        path,job=self.verify_source(canonical_source_path(row,ROOT));folder=ROOT/'cache/KIRAAT_original_FLAC_records_v1'/job['sha256']
        receipt_path=folder/'receipt.json';binding=record(receipt_path);receipt=json.loads(receipt_path.read_text())
        require(receipt['kind']=='ASR18_original_compressed_FLAC_records_v1' and receipt['complete'] is True and receipt['source']['source_path']==str(path) and receipt['source']['source_sha256']==job['sha256'] and receipt['source']['revision']==job['revision'] and receipt['quality_language_or_disagreement_exclusions']==[],'Exact complete original FLAC carrier')
        validation=json.loads((folder/'validation.json').read_text())
        require(validation['kind']=='ASR18_all_original_FLAC_record_bytes_validation_v1' and validation['passed'] is True and validation['carrier_receipt']==binding and validation['source_sha256']==job['sha256'] and validation['full_physical_rows']==receipt['physical_rows'] and validation['every_compressed_byte_and_physical_row_identical'] is True,'Independent all-compressed-byte/physical-row audit')
        stat=path.stat();require((stat.st_size,stat.st_mtime_ns)==(receipt['source']['source_size'],receipt['source']['source_mtime_ns']),'Original Parquet source signature')
        for pin in (receipt['records'],receipt['index']):
            target=scoped(pin['path']);st=target.stat();require(target.parent==folder and (st.st_size,st.st_mtime_ns)==(pin['bytes'],pin['mtime_ns']),'Frozen original carrier/index signature')
        if job['sha256'] not in self.carriers:
            require(record(receipt['index']['path'])=={k:receipt['index'][k] for k in ('path','bytes','sha256')},'Exact original carrier index')
            self.carriers[job['sha256']]=binding
        require(self.carriers[job['sha256']]==binding,'Carrier receipt changed')
        db=sqlite3.connect(Path(receipt['index']['path']).as_uri()+'?mode=ro&immutable=1',uri=True)
        try:entry=db.execute('SELECT clip_id,source_id,offset,bytes,sha256 FROM records WHERE row_index=?',(int(row['row_index']),)).fetchone()
        finally:db.close()
        require(entry is not None and 'kiraat:'+entry[0]==row['unique_id'] and entry[1]==row['source_id'],'Exact physical original FLAC row')
        _,_,offset,size,sha=entry
        require(0<size<=MEDIA_PER_JOB and 0<=offset and offset+size<=receipt['records']['bytes'],'Before-read compressed32MiB extent admission')
        self.guard()
        with Path(receipt['records']['path']).open('rb') as stream:stream.seek(offset);media=stream.read(size)
        require(len(media)==size and hashlib.sha256(media).hexdigest()==sha and media.startswith(b'fLaC'),'Original FLAC bytes')
        d=self.active['declaration']
        with BorrowedCompressedIO(media) as compressed, sf.SoundFile(compressed) as source:
            require(source.samplerate==d['native_sr']==24000 and source.channels==1 and source.frames==d['native_samples'],'Header proves full declared mono24k counts BEFORE PCM allocation')
            self.guard();pcm=source.read(dtype='float32',always_2d=False);sr=source.samplerate
        require(pcm.nbytes==4*d['native_samples'],'Actual native FLAC allocation matches reservation')
        output=resample(pcm,sr)
        return output,{'format':'KIRAAT_embedded_FLAC','source_file_sha256':job['sha256'],'original_row_index':int(row['row_index']),
            'compressed_clip_bytes':size,'compressed_clip_sha256':sha,'native_sample_rate':sr,'native_samples':len(pcm),
            'actual_native_channels':1,'full_embedded_clip_used':True,'source_start_end_applied_again':False,
            'metadata_duration_delta_seconds':len(pcm)/sr-float(row['duration']),'compressed_carrier_receipt':binding,
            'Parquet_materialized_in_this_reader':False,'decoded_PCM_pool_used':False}

    def yodas(self,row):
        import numpy as np
        if str(AV_TARGET) not in sys.path:sys.path.insert(0,str(AV_TARGET))
        import av
        require(av.__version__=='16.0.1','Pinned isolated PyAV')
        path,job=self.verify_source(canonical_source_path(row,ROOT));shard=row['shard']
        index=scoped(ROOT/f'indexes/yodas3_tar_{shard}.sqlite')
        db=sqlite3.connect(index.as_uri()+'?mode=ro&immutable=1',uri=True)
        try:entry=db.execute('SELECT member_name,offset,bytes FROM recordings WHERE recording_id=?',(row['source_id'],)).fetchone()
        finally:db.close()
        require(entry is not None,'Unresolved recording index, never skip')
        name,offset,size=entry;require(Path(name).stem==row['source_id'] and name.endswith('.webm'),'Original recording member identity')
        start,end=Fraction(str(row['start'])),Fraction(str(row['end']));d=self.active['declaration'];p=self.active['plan']
        require(0<=start<end and d['native_sr']==48000,'Original declared positive interval/48k')
        with BoundedExtent(path,offset,size,self.guard) as extent,av.open(extent,mode='r',format='webm') as container:
            require(len(container.streams.audio)==1,'One original stream')
            stream=container.streams.audio[0];stream.codec_context.thread_count=1
            context=stream.codec_context
            if context.sample_rate:require(context.sample_rate==d['native_sr'],'Source codec rate before first-frame decode')
            if context.layout is not None and context.layout.channels:
                require(len(context.layout.channels)<=p['channels_bound'],'Unsupported source codec channels before first-frame decode')
            # Decoder-internal frame workspace is separate from raw ndarray
            # reservations. The complete worst-case frame envelope was admitted
            # before opening libav; actual layout is checked before np.empty.
            first=next(container.decode(stream));require(first.pts is not None,'Original first PTS required')
            sr=first.sample_rate;channels=len(first.layout.channels)
            require(sr==d['native_sr'] and 1<=channels<=p['channels_bound'] and 0<first.samples<=p['frame_samples_bound'] and first.format.is_planar and first.format.bytes<=8,'Before full PCM allocation: supported actual first-frame rate/channel/layout/bytes')
            origin=first.pts*first.time_base;begin=round(start*sr);finish=round(end*sr);count=finish-begin
            require(count==d['native_samples'],'Declared original interval count before np.empty')
            self.guard();pcm=np.empty(count,dtype=np.float32);written=0;decoded=0
            target=origin+max(Fraction(0),start-2)
            container.seek(math.floor(target/stream.time_base),stream=stream,backward=True,any_frame=False)
            first_selected_pts=None
            for frame in container.decode(stream):
                self.guard();decoded+=1
                require(frame.pts is not None and frame.sample_rate==sr and len(frame.layout.channels)==channels and frame.format.is_planar and frame.format.bytes<=8 and 0<frame.samples<=p['frame_samples_bound'],'Every decoded frame within preallocated raw envelope BEFORE ndarray conversion')
                frame_begin=round((frame.pts*frame.time_base-origin)*sr);frame_end=frame_begin+frame.samples
                if frame_end<=begin:continue
                if frame_begin>=finish:break
                raw=frame.to_ndarray();require(raw.ndim==2 and raw.shape==(channels,frame.samples),'Actual planar frame geometry')
                if np.issubdtype(raw.dtype,np.integer):raw=raw.astype(np.float32)/float(2**(np.iinfo(raw.dtype).bits-1))
                else:raw=raw.astype(np.float32,copy=False)
                raw=raw.mean(axis=0,dtype=np.float32) if raw.shape[0]>1 else raw[0]
                low,high=max(begin,frame_begin),min(finish,frame_end)
                require(low-begin==written,'Original PTS gap/overlap unresolved; no synthesized samples')
                pcm[low-begin:high-begin]=raw[low-frame_begin:high-frame_begin];written=high-begin
                if first_selected_pts is None:first_selected_pts=str(frame.pts*frame.time_base)
                del raw
                if written==count:break
            require(written==count,'Original interval beyond source unresolved')
            output=resample(pcm,sr)
            info={'format':'YODAS_Opus_WebM_TAR_extent','source_file_sha256':job['sha256'],'member_name':name,
                'member_offset':offset,'member_bytes':size,'member_logical_read_bytes':extent.logical_bytes_read,
                'native_sample_rate':sr,'native_samples':count,'actual_native_channels':channels,'decoded_frames':decoded,
                'PTS_origin':str(origin),'source_interval_seconds':[str(start),str(end)],'first_selected_frame_PTS':first_selected_pts,
                'linear_reference':False,'codec_preroll_seconds':2,'temporary_member_or_PCM_written':False,'PyAV_version':av.__version__}
        return output,info
