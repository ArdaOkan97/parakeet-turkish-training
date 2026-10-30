"""One exact original nonzero terminal interval EOF, prior and ordinary paths unchanged.

Keep original annotations, larger native reservation, PTS selection/preroll and all
real interval samples. Only this certified one-sample declaration mismatch is used.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib,math,sqlite3,sys
import fastmss_original_clip_geometry_overlay_v5 as prior
from fastmss_original_clip_geometry_overlay_v5 import core,BoundedOriginalReader,raw_plan,chunked_hashes,canonical_source_path,source_identity_binding
from bpe_broad_audio_v1 import AV_TARGET,Extent,resample
ROW_ID='alania:38198deb36bb10f9e58129b2c9d100fe_00003'
ORDINAL=235125
require,digest,same,record=core.require,core.digest,core.same,core.record
match_original=prior.match_original

def checked_certificate(policy,guard):
    require(policy['source']==record(__file__,guard),'Exact physical interval overlay source')
    c=core.read_bound(policy['certificate'],guard);review=core.read_bound(policy['ROOT_review'],guard)
    require(review['accepted'] is True and review['reviewer']=='/root'
        and review['scope']=='ASR18_source_bound_sixth_original_terminal_interval_geometry_overlay_v1'
        and review['overlay_source']==policy['source'] and review['certificate']==policy['certificate'],
        'Exact ROOT reviewed nonzero terminal interval')
    require(c['kind']=='ASR18_source_bound_one_original_terminal_interval_geometry_certificate_v1'
        and c['original_row']['ordinal']==ORDINAL and c['original_row']['unique_id']==ROW_ID
        and c['original_row']['start']==1454.432 and c['original_row']['end']==1473.713
        and digest(c['original_row'])==c['original_row_sha256']
        and digest(c['full_target_ids'])==c['full_target_ids_sha256'] and len(c['full_target_ids'])==121,
        'Exact original nonzero annotation and full121 labels')
    require(c['original_declaration']==dict(ordinal=ORDINAL,source_component=0,native_sr=48000,
        native_samples=925488,clean16k_samples=308496,mel_frames=1928,encoder_frames=241,target_tokens=121,status='declared')
        and c['actual_geometry']==dict(native_sr=48000,native_samples=925487,clean16k_samples=308496,
        mel_frames=1928,encoder_frames=241,target_tokens=121) and c['decoded_frames']==1185
        and c['original_PTS_interval_begin']==69812736 and c['original_PTS_interval_declared_finish']==70738224
        and c['actual_PTS_interval_finish']==70738223 and c['selected_physical_source_range']==[69812712,70738199]
        and c['full_source_native_samples']==70738199 and c['independent_full_member_decoded_frames']==73686
        and c['max_abs_PTS_residual_samples']=='24','Exact source-bound EOF and unchanged clean/T/U')
    observed=core.read_bound(c['independent_full_member_observation'],guard)
    require(observed['full_original_frames_observed_once'] is True
        and observed['selected_interval_original_samples_used_once'] is True
        and observed['full_source_range_equals_original_PTS_selected_PCM'] is True
        and observed['synthesized_samples']==0 and observed['selected_interval_skipped_samples']==0
        and {k:observed['actual_interval_geometry'][k]for k in c['actual_geometry']}==c['actual_geometry']
        and observed['actual_interval_geometry']['physical_feature_frames']==1929
        and observed['full_target_ids']==c['full_target_ids']
        and observed['actual_EOF_PTS_minus_cumulative_samples']==c['actual_EOF_PTS_minus_cumulative_samples']==24
        and observed['selected_PTS_begin_minus_physical_begin']==c['selected_PTS_begin_minus_physical_begin']==24
        and observed['packet_side_data']==c['packet_side_data']==[dict(packet=73686,type='skip_samples',data_size=10,description='Skip Samples',raw_hex='00000000310000000000',skip_start_samples=0,discard_end_samples=49,skip_reason=0,discard_reason=0)]
        and observed['native_interval_f32_sha256']==c['native_f32_sha256']
        and all(observed[k]==c[k]for k in ('member_sha256','clean_f32_sha256','clean_i16_sha256',
            'recording_index_entry','original_TAR','original_PTS_interval_begin',
            'original_PTS_interval_declared_finish','actual_PTS_interval_finish',
            'selected_physical_source_range','full_source_native_samples','full_source_native_f32_sha256')),
        'Independent full-member and original PTS-selected interval PCM evidence')
    return c

def decode_terminal_interval(reader,row,certificate,*,guard,channels_bound,frame_samples_bound):
    """Original two-second preroll/PTS coordinates; exact certified EOF and PCM only."""
    import numpy as np
    if str(AV_TARGET)not in sys.path:sys.path.insert(0,str(AV_TARGET))
    import av
    require(av.__version__=='16.0.1','Pinned original PyAV')
    path,job=reader.verify_source(canonical_source_path(row,core.ROOT));original=certificate['original_TAR'];stat=path.stat()
    require(str(path)==original['path'] and stat.st_size==original['bytes']
        and stat.st_mtime_ns==original['mtime_ns'] and job['sha256']==original['recorded_sha256'],
        'Exact original source signature')
    index=core.ROOT/('indexes/yodas3_tar_'+row['shard']+'.sqlite');db=sqlite3.connect(index.as_uri()+'?mode=ro&immutable=1',uri=True)
    try:entry=db.execute('SELECT member_name,offset,bytes FROM recordings WHERE recording_id=?',(row['source_id'],)).fetchone()
    finally:db.close()
    require(entry is not None and dict(zip(('member_name','offset','bytes'),entry))==certificate['recording_index_entry'],
        'Exact original member, no replacement')
    name,offset,size=entry;sha=hashlib.sha256()
    with path.open('rb')as source:
        source.seek(offset);left=size
        while left:
            guard();block=source.read(min(1024**2,left));require(block,'Full original member');sha.update(block);left-=len(block)
    require(sha.hexdigest()==certificate['member_sha256'],'Exact original compressed bytes')
    start,end=Fraction(str(row['start'])),Fraction(str(row['end']));begin=round(start*48000);finish=round(end*48000)
    count=finish-begin;require(0<start<end and count==certificate['original_declaration']['native_samples']
        and begin==certificate['original_PTS_interval_begin'] and finish==certificate['original_PTS_interval_declared_finish'],
        'Unchanged original nonzero interval and larger raw allocation')
    with Extent(path,offset,size,guard)as extent,av.open(extent,mode='r',format='webm')as container:
        require(len(container.streams.audio)==1,'One original stream');stream=container.streams.audio[0];stream.codec_context.thread_count=1
        require(stream.codec_context.name=='opus' and stream.time_base==Fraction(1,1000),'Exact certified Opus/ms clock')
        first=next(container.decode(stream));channels=len(first.layout.channels);origin=first.pts*first.time_base
        require(first.pts is not None and origin==0 and first.sample_rate==48000 and 1<=channels<=channels_bound
            and first.format.is_planar and first.format.bytes<=8 and 0<first.samples<=frame_samples_bound,
            'Original first frame allocation envelope')
        guard();pcm=np.empty(count,dtype=np.float32);written=decoded=0;first_selected_pts=None
        target=origin+max(Fraction(0),start-2);container.seek(math.floor(target/stream.time_base),stream=stream,backward=True,any_frame=False)
        for frame in container.decode(stream):
            guard();decoded+=1
            require(frame.pts is not None and frame.sample_rate==48000 and len(frame.layout.channels)==channels
                and frame.format.is_planar and frame.format.bytes<=8 and 0<frame.samples<=frame_samples_bound,
                'Every original frame before ndarray allocation')
            frame_begin=round((frame.pts*frame.time_base-origin)*48000);frame_end=frame_begin+frame.samples
            if frame_end<=begin:continue
            require(frame_begin<finish,'No original frames beyond exact certified terminal interval')
            raw=frame.to_ndarray();require(raw.shape==(channels,frame.samples),'Original planar shape')
            if np.issubdtype(raw.dtype,np.integer):raw=raw.astype(np.float32)/float(2**(np.iinfo(raw.dtype).bits-1))
            else:raw=raw.astype(np.float32,copy=False)
            raw=raw.mean(axis=0,dtype=np.float32)if channels>1 else raw[0]
            low,high=max(begin,frame_begin),min(finish,frame_end)
            require(low-begin==written,'Original exact PTS gap/overlap guard, no filler')
            pcm[low-begin:high-begin]=raw[low-frame_begin:high-frame_begin];written=high-begin
            if first_selected_pts is None:first_selected_pts=str(frame.pts*frame.time_base)
            del raw
        require(decoded==certificate['decoded_frames'] and written==certificate['actual_geometry']['native_samples']
            and frame_end==certificate['actual_PTS_interval_finish'] and frame.samples==911,
            'Exact certified actual EOF, no generic clamp or tolerance')
        logical=extent.logical_bytes_read
    actual=pcm[:written]
    require(hashlib.sha256(memoryview(actual).cast('B')).hexdigest()==certificate['native_f32_sha256'],
        'Every actual original PTS-selected sample exact')
    output=resample(actual,48000);f32,i16=chunked_hashes(output,guard=guard)
    require(len(output)==certificate['actual_geometry']['clean16k_samples'] and f32==certificate['clean_f32_sha256']
        and i16==certificate['clean_i16_sha256'],'Exact full original interval resampling and PCM hashes')
    return output,dict(format='YODAS_Opus_WebM_TAR_extent',source_file_sha256=job['sha256'],member_name=name,
        member_offset=offset,member_bytes=size,member_logical_read_bytes=logical,native_sample_rate=48000,
        native_samples=written,actual_native_channels=channels,decoded_frames=decoded,PTS_origin=str(origin),
        source_interval_seconds=[str(start),str(end)],first_selected_frame_PTS=first_selected_pts,
        full_physical_recording_used=False,full_physical_selected_interval_used=True,
        original_annotation_end_preserved=True,original_derived_native_samples=count,linear_reference=False,
        codec_preroll_seconds=2,initial_decoded_frame_retained=False,temporary_member_or_PCM_written=False,
        PyAV_version=av.__version__)

class OriginalRows(prior.OriginalRows):
    def __init__(self,validated,*,geometry_overlay,**arguments):
        self.sixth_overlay_policy=deepcopy(geometry_overlay)
        self.sixth_certificate=checked_certificate(geometry_overlay,arguments['guard'])
        require(geometry_overlay['prior_overlay']['source']==record(prior.__file__,arguments['guard']),
                'Exact original overlay remains the authority for its five prior certified rows')
        super().__init__(validated,geometry_overlay=geometry_overlay['prior_overlay'],**arguments)

    def __call__(self, row, expected, phase):
        if row.get('ordinal') != ORDINAL and row.get('source_id') != ROW_ID:
            return super().__call__(row, expected, phase)
        import numpy as np
        raw, declaration = self.resolve(row, expected, phase)
        match_original(self.sixth_certificate, raw, declaration, expected)
        self.original_source_files = []; self.geometry_proof = None; self.replay_proof = None
        require(self.active_job is None, 'Prior raw buffers actually released')
        # Charge the unchanged larger historical envelope, never claim freed credit.
        p = raw_plan(native_samples=declaration['native_samples'], native_rate=declaration['native_sr'],
            clean_samples=declaration['clean16k_samples'], phase=phase, role=expected['role'],
            channels_bound=self.raw_contract['maximum_runtime_channels'],
            frame_samples_bound=self.raw_contract['maximum_decoder_frame_samples'])
        work = p['workspace_incremental_envelopes']; review = self.raw_contract
        require(work['decoder'] <= review['decoder_workspace_allowance_bytes'], 'Same decoder workspace')
        if self.frontend_enabled:
            require(work['weak_transform_FFT'] <= review['transform_workspace_allowance_bytes']
                    and work['original_frontend_spectral'] <= review['frontend_workspace_allowance_bytes'],
                    'Same original transform/frontend workspace')
        workspace = max(work.values()) if self.frontend_enabled else work['decoder']
        prospective = p['raw_peak_bytes'] + p['compressed_media_cap_bytes'] + workspace + review['numerical_runtime_reserve_bytes']
        observation = self.guard()
        require(observation['worker_private_memory']['private_bytes'] + prospective <= self.private_limit,
                'Unchanged current-private plus complete original raw/media/workspace before allocation')
        core.admit(prospective, self.guard)
        key = digest(expected['context']); self.ledger.begin(key, p); self.active_job = key
        if self.broad is None:
            self.broad = BoundedOriginalReader(ledger=self.ledger,
                source_inventory_binding=self.validated['bindings']['source_identity_inventory'],
                guard=self.guard, heldout_PCM_hashes=self.heldout)
        with self.ledger.stage(key, 'decode_resample'):
            wave, info = decode_terminal_interval(self.broad, raw, self.sixth_certificate, guard=self.guard,
                channels_bound=p['channels_bound'], frame_samples_bound=p['frame_samples_bound'])
            self.ledger.observe(key, wave, name='whole_clean_float32', maximum_bytes=4*declaration['clean16k_samples'])
        with self.ledger.stage(key, 'canonical_hash'): f32, h = chunked_hashes(wave, guard=self.guard)
        require(h not in self.heldout, 'Exact heldout PCM overlap remains unresolved')
        c = self.sixth_certificate['actual_geometry']['clean16k_samples']
        require(wave.dtype == np.float32 and wave.ndim == 1 and wave.flags.c_contiguous
                and len(wave) == c and core.geometry(c)['encoder_frames'] == row['encoder_frames'],
                'Entire actual waveform and unchanged encoder T')
        actual = dict(info, unique_id=raw['unique_id'], source_id=raw['source_id'],
            actual_original_source_path=str(canonical_source_path(raw, core.ROOT)),
            decoded_sample_rate=16000, decoded_samples=c, encoder_frames=row['encoder_frames'],
            raw_label_sha256=self.sixth_certificate['raw_label_sha256'], waveform_sha256=f32, PCM_i16_sha256=h)
        item = self.broad.final_source_identity.entries[actual['actual_original_source_path']]
        self.original_source_files = [source_identity_binding(raw, item, actual, core.ROOT),
            item['index'], item['receipt'], self.sixth_overlay_policy['certificate']]
        self.geometry_proof = dict(kind='ASR18_exact_original_terminal_interval_geometry_reconciliation_v1',
            certificate=self.sixth_overlay_policy['certificate'], source=self.sixth_overlay_policy['source'],
            historical_declaration=deepcopy(declaration), actual_geometry=deepcopy(self.sixth_certificate['actual_geometry']),
            original_annotations_plan_labels_and_encoder_geometry_unchanged=True,
            historical_mel_frames=declaration['mel_frames'], actual_physical_mel_frames=self.sixth_certificate['actual_geometry']['mel_frames'],
            all_original_selected_interval_physical_samples_used_once=True, sampler_or_Adam_frontier_advanced=False)
        self.last = dict(actual, **core.geometry(c)); self.stable(); return wave

