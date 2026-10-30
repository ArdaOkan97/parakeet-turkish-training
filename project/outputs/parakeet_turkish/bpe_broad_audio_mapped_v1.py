"""Original full-clip decoder using read-only mapped compressed Parquet input.

Canonical decoding/resample unchanged; no whole-PCM pool or quality selection.
Final signatures are mandatory for production; incomplete-source mode is only
an explicit bounded diagnostic against pinned completed source receipts.
"""
from bpe_broad_audio_v1 import *
from bpe_broad_audio_v1 import BroadAudioReader as OriginalReader
from bpe_broad_source_identity_v1 import FinalSourceIdentity

class BroadAudioReader(OriginalReader):
    def __init__(self,*,source_inventory_binding=None,diagnostic_only=False,**kwargs):
        super().__init__(**kwargs)
        require(source_inventory_binding is not None or diagnostic_only is True,'Production mapping requires finalized source signatures')
        self.final_source_identity=FinalSourceIdentity(source_inventory_binding,self.plan,self.jobs) if source_inventory_binding else None
    def verify_source(self,path):
        return self.final_source_identity.verify(path) if self.final_source_identity is not None else super().verify_source(path)

    def kiraat(self,row):
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        import soundfile as sf
        path,job=self.verify_source(row['shard']);index=int(row['row_index']);pf=pq.ParquetFile(path,memory_map=True,pre_buffer=False,buffer_size=0)
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
        return output,{'format':'KIRAAT_embedded_FLAC','Parquet_compressed_input_ACCESS_READ_mapped':True,'Parquet_pre_buffer':False,'source_file_sha256':job['sha256'],
            'original_row_index':index,'audio_column_uncompressed_bytes':size,'compressed_clip_bytes':len(selected),
            'compressed_clip_sha256':compressed_SHA,'native_sample_rate':sr,'native_samples':len(pcm),
            'full_embedded_clip_used':True,'source_start_end_applied_again':False,
            'metadata_duration_delta_seconds':len(pcm)/sr-float(row['duration'])}
