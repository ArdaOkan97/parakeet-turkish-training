"""Read one source-identical FLAC value without live Parquet materialization."""
from bpe_broad_audio_v1 import *
from bpe_broad_audio_v1 import BroadAudioReader as OriginalReader
from bpe_broad_source_identity_v1 import FinalSourceIdentity

class BroadAudioReader(OriginalReader):
    def __init__(self,*,source_inventory_binding=None,diagnostic_only=False,**kwargs):
        super().__init__(**kwargs)
        require(source_inventory_binding is not None or diagnostic_only is True,'Production compressed carrier requires finalized source signatures')
        self.final_source_identity=FinalSourceIdentity(source_inventory_binding,self.plan,self.jobs) if source_inventory_binding else None
        self.carriers={}
    def verify_source(self,path):
        return self.final_source_identity.verify(path) if self.final_source_identity is not None else super().verify_source(path)
    def kiraat(self,row):
        import numpy as np
        import soundfile as sf
        path,job=self.verify_source(row['shard']);folder=ROOT/'cache/KIRAAT_original_FLAC_records_v1'/job['sha256']
        receipt_path=folder/'receipt.json';binding=record(receipt_path);receipt=json.loads(receipt_path.read_text())
        require(receipt['kind']=='ASR18_original_compressed_FLAC_records_v1' and receipt['complete'] is True
            and receipt['source']['source_path']==str(path) and receipt['source']['source_sha256']==job['sha256']
            and receipt['source']['revision']==job['revision'] and receipt['quality_language_or_disagreement_exclusions']==[],
            'Exact complete source FLAC carrier required; no Arrow fallback')
        validation=json.loads((folder/'validation.json').read_text())
        require(validation['kind']=='ASR18_all_original_FLAC_record_bytes_validation_v1' and validation['passed'] is True
            and validation['carrier_receipt']==binding and validation['source_sha256']==job['sha256']
            and validation['full_physical_rows']==receipt['physical_rows'] and validation['every_compressed_byte_and_physical_row_identical'] is True,
            'Independent full original FLAC-byte/physical-row identity proof required')
        stat=path.stat();require((stat.st_size,stat.st_mtime_ns)==(receipt['source']['source_size'],receipt['source']['source_mtime_ns']),'Original source changed after compressed carrier creation')
        for item in (receipt['records'],receipt['index']):
            target=scoped(item['path']);sig=target.stat()
            require(target.parent==folder and (sig.st_size,sig.st_mtime_ns)==(item['bytes'],item['mtime_ns']),'Frozen compressed carrier/index signature changed')
        if job['sha256'] not in self.carriers:
            require(record(receipt['index']['path'])=={k:receipt['index'][k] for k in ('path','bytes','sha256')},'Exact physical-row FLAC index hash changed')
            self.carriers[job['sha256']]=binding
        require(self.carriers[job['sha256']]==binding,'Carrier receipt changed during bounded reader use')
        db=sqlite3.connect(Path(receipt['index']['path']).as_uri()+'?mode=ro&immutable=1',uri=True)
        try:entry=db.execute('SELECT clip_id,source_id,offset,bytes,sha256 FROM records WHERE row_index=?',(int(row['row_index']),)).fetchone()
        finally:db.close()
        require(entry is not None and 'kiraat:'+entry[0]==row['unique_id'] and entry[1]==row['source_id'],'Original physical FLAC row identity differs')
        _,_,offset,size,sha=entry
        require(0<size<=MAX_COMPRESSED_CLIP_BYTES and 0<=offset and offset+size<=receipt['records']['bytes'],'Exact bounded original FLAC extent')
        self.guard()
        with Path(receipt['records']['path']).open('rb') as stream:stream.seek(offset);media=stream.read(size)
        require(len(media)==size and hashlib.sha256(media).hexdigest()==sha and media.startswith(b'fLaC'),'Original compressed FLAC bytes changed before consumed draw')
        with sf.SoundFile(io.BytesIO(media)) as audio:
            require(audio.samplerate==24000 and audio.channels==1,'Original pinned mono24kFLAC required')
            if audio.frames*4>MAX_NATIVE_PCM_BYTES:raise BoundViolation('full_clip_PCM',row['unique_id'],audio.frames*4,MAX_NATIVE_PCM_BYTES)
            self.guard();pcm=audio.read(dtype='float32',always_2d=False);sr=audio.samplerate
        output=resample(pcm,sr)
        return output,{'format':'KIRAAT_embedded_FLAC','source_file_sha256':job['sha256'],'original_row_index':int(row['row_index']),
            'compressed_clip_bytes':size,'compressed_clip_sha256':sha,'native_sample_rate':sr,'native_samples':len(pcm),
            'full_embedded_clip_used':True,'source_start_end_applied_again':False,'metadata_duration_delta_seconds':len(pcm)/sr-float(row['duration']),
            'compressed_carrier_receipt':binding,'Parquet_materialized_in_this_reader':False,'decoded_PCM_pool_used':False}
