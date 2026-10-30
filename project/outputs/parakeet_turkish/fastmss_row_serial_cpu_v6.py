"""Inactive original-audio row-serial CPU preparation primitives.

No numerical runtime imports at module scope. No native training, sampler,
teacher rewrite, PCM pool, DATA publication or resource relaxation lives here.
"""
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import zipfile

import fastmss_phase_aware_plan_bridge_v5 as bridge
import fastmss_phase_aware_feature_window_v4 as sink
from fastmss_owned_process_closure_v2 import identity
from fastmss_raw_lifetime_v1 import RawLifetimeLedger,raw_plan,canonical_source_path,source_identity_binding,chunked_hashes,CAPACITY as RAW_CAPACITY
from asr18_physical_feature_geometry_v2 import from_samples

ROOT, PROJECT = bridge.ROOT, bridge.PROJECT
TRAINING_SITE = PROJECT/'work/training-env/Lib/site-packages'
CPU_ENV = ROOT/'scratch/BPE_broad_CPU_preflight_env_v2'
CONTRACT_KIND = 'FastMSS_row_serial_CPU_preparation_contract_v6'
REQUEST_KIND = 'FastMSS_row_serial_CPU_update_request_v6'
require, same, digest, read_bound = bridge.require, bridge.same, bridge.digest, bridge.read_bound
record, write_new, scoped = sink.record, sink.write_new, bridge.scoped

def resource_guard():
    from resource_watchdog_v2 import sample_memory
    value=sample_memory(os.getpid(),'row_serial_CPU_preparation')
    w,s=value['worker_private_memory'],value['system_memory']
    require(value['paired_sampling_within_one_second'] is True and w['available'] is True and s['available'] is True and w['private_bytes']<2_000_000_000,'Fresh paired CPU/private2GB telemetry required')
    require(s['commit_used_bytes']<.90*s['commit_limit_bytes'] and s['commit_available_bytes']>=2_000_000_000 and shutil.disk_usage(ROOT).free>=100_000_000_000,'Unchanged global90/2GB/E100GB guards')
    return value

def admit(budget,guard=resource_guard):
    require(type(budget) is int and 0<budget<2_000_000_000,'Explicit reviewed CPU incremental forecast required')
    value=guard();s=value['system_memory']
    require(s['commit_used_bytes']+budget<.90*s['commit_limit_bytes'] and s['commit_available_bytes']-budget>=2_000_000_000,'Fresh prospective CPU admission denied; no exclusion/retry')
    return {'sample':value,'forecast_incremental_bytes':budget,'passed':True}

def environment():
    config=json.loads((ROOT/'storage_config.json').read_text())
    require(config['storage_root']==str(ROOT) and config['allow_fallback_outside_root'] is False,'Required E storage policy')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1','CUDA masked CPU producer only')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): require(os.environ.get(key)=='1','Four numerical-library threadcaps required')
    for key in ('TEMP','TMP','HF_HOME','TORCH_EXTENSIONS_DIR'): require(Path(os.environ[key]).resolve().is_relative_to(ROOT.resolve()),'Process-local E environment required')

def verify_contract(pin,session,*,guard=resource_guard):
    contract=read_bound(pin,guard)
    require(contract['kind']==CONTRACT_KIND and contract['production_enabled'] is False and contract['purpose']==bridge.PURPOSE and same(session['CPU_preparation_contract'],pin),'Explicit unarmed session-bound CPU source/resource contract')
    require(contract['maximum_window_updates']==20 and contract['maximum_session_windows']==2 and contract['maximum_session_native_updates']==40 and contract['feature_capacity_bytes']==bridge.CAPACITY and contract['raw_PCM_capacity_bytes']==bridge.PCM_CAPACITY and contract['raw_PCM_maximum_jobs']==2 and contract['raw_PCM_cache_written'] is False,'Frozen finite/serial cache limits')
    require(same(contract['DATA_evidence_sha256'],session['DATA_evidence_sha256']) and same(contract['augmentation_identity'],session['augmentation_identity']),'Exact DATA/policy/runtime identity, no producer rewrite')
    paths={'common':Path(__file__),'bridge':Path(bridge.__file__),'sink':Path(sink.__file__),
        'process_helper':Path(__file__).with_name('fastmss_owned_process_closure_v2.py'),
        'stdlib_feature_validator':Path(__file__).with_name('fastmss_stdlib_feature_validation_v1.py'),
        'reference_worker':PROJECT/'work/research/produce_FastMSS_independent_rows_CPU_v6.py',
        'feature_worker':PROJECT/'work/research/produce_FastMSS_serial_features_CPU_v6.py',
        'controller':PROJECT/'work/research/prepare_FastMSS_phase_window_CPU_v6.py',
        'guard':PROJECT/'work/research/ASR18_bounded_CPU_phase_v3.py',
        'broad_reader':Path(__file__).with_name('bpe_broad_audio_FLAC_records_v1.py'),
        'original_reader':Path(__file__).with_name('bpe_broad_audio_v1.py'),
        'source_identity':Path(__file__).with_name('bpe_broad_source_identity_v1.py'),
        'replay_labels':Path(__file__).with_name('bpe_replay_cache_labels_v1.py'),
        'geometry':Path(__file__).with_name('bpe_runtime_declared_geometry_v1.py'),
        'cleaner':Path(__file__).with_name('bpe_cleaner_data_v1.py'),
        'clean_phase_reader':Path(__file__).with_name('fastmss_clean_phase_reader_v1.py'),
        'watchdog':Path(__file__).with_name('resource_watchdog_v2.py'),
        'metadata_locks':Path(__file__).with_name('bpe_windows_final_metadata_lock_v1.py'),
        'declared_geometry_index':Path(__file__).with_name('bpe_declared_geometry_index_v1.py'),
        'label_catalog':Path(__file__).with_name('bpe_training_label_catalog_v1.py'),
        'expanded_rows':Path(__file__).with_name('expanded_training_rows_v1.py'),
        'scalar_metadata':Path(__file__).with_name('bpe_checkpoint_scalar_metadata_v1.py')}
    paths.update(raw_lifetime=Path(__file__).with_name('fastmss_raw_lifetime_v1.py'),
        bounded_decoder=Path(__file__).with_name('fastmss_bounded_original_audio_v1.py'),
        physical_geometry=Path(__file__).with_name('asr18_physical_feature_geometry_v2.py'),
        family_observer=Path(__file__).with_name('asr18_owned_family_observation_v4.py'))
    paths.update(scipy_signal_backend=TRAINING_SITE/'scipy/signal/_signaltools.py',
        scipy_polyphase_backend=TRAINING_SITE/'scipy/signal/_upfirdn.py',
        scipy_FIR_backend=TRAINING_SITE/'scipy/signal/_fir_filter_design.py',
        soundfile_backend=TRAINING_SITE/'soundfile.py')
    require(set(contract['sources'])==set(paths),'Entire explicit controller dependency identity required')
    for key,path in paths.items(): require(same(record(path,guard),contract['sources'][key]),'CPU preparation source mismatch: '+key)
    require(same(record(contract['outer_parent_source']['path'],guard),contract['outer_parent_source']),'Exact external native-parent source binding')
    require(Path(contract['python']['path']).resolve()==CPU_ENV/'Scripts/python.exe','Existing E CPU interpreter only')
    require(same(record(contract['python']['path'],guard),contract['python']),'Pinned CPU interpreter changed')
    require(contract['resource_review']['passed'] is True and contract['resource_review']['scope']=='CPU_preparation_only' and contract['resource_review']['native_GPU_fit_claimed'] is False,'Separate explicit CPU-stage scope, never circular GPU admission')
    for key in ('reference_incremental_bytes','frontend_incremental_bytes'): bridge.integer(contract['resource_review'][key],1,1_999_999_999)
    for key in ('reference_maximum_private_bytes','frontend_maximum_private_bytes'): bridge.integer(contract['resource_review'][key],1,2_000_000_000)
    require(contract['resource_review']['maximum_simultaneous_children']==1,'One serial CPU child, whole-private allowance for resident native admission')
    review=contract['resource_review']
    for key in ('controller_private_reserve_bytes','shim_private_reserve_bytes','whole_owned_private_allowance_bytes'):bridge.integer(review[key],1)
    require(review['whole_owned_private_allowance_bytes']>=review['controller_private_reserve_bytes']+review['shim_private_reserve_bytes']+max(review['reference_maximum_private_bytes'],review['frontend_maximum_private_bytes']),'Whole CPU controller+one leaf+shim allowance, not leaf peak alone')
    bridge.integer(contract['maximum_child_seconds'],1,3600)
    sink._frontend(contract['frontend'],guard)
    verify_raw_review(contract['raw_lifetime_review'])
    return contract

def verify_raw_review(review):
    require(review['kind']=='FastMSS_preallocation_raw_lifetime_review_v1' and review['accepted'] is True and review['raw_capacity_bytes']==RAW_CAPACITY and review['maximum_jobs']==2 and review['compressed_media_per_job_bytes']==32*1024**2,'Explicit unchanged raw128MiB/two-job + separate compressed32MiB domains')
    for key in ('maximum_runtime_channels','maximum_decoder_frame_samples','decoder_workspace_allowance_bytes','transform_workspace_allowance_bytes','frontend_workspace_allowance_bytes','numerical_runtime_reserve_bytes'):bridge.integer(review[key],1)
    require(review['maximum_runtime_channels']<=8 and review['maximum_decoder_frame_samples']<=16384,'Supported explicit first-frame bounds, not full corpus channel validation')
    return review

def normalize_declared_TR(raw,declaration,ids):
    return {**raw,'broad_original':deepcopy(raw),'recording_source_id':raw['source_id'],'source_id':raw['unique_id'],
        'language':'tr','augmentation_role':'student_tr','encoder_frames':declaration['encoder_frames'],
        'target_tokens':len(ids),'tr_metadata_labels':list(ids),'training_input_type':'TR_frame_target_metadata_v1',
        'declared_geometry':deepcopy(declaration),'runtime_full_audio_geometry_and_heldout_check_required':True,
        'declared_frontend_geometry':from_samples(declaration['clean16k_samples'])}

def stage_guard(contract,stage):
    limit=contract['resource_review'][stage+'_maximum_private_bytes']
    def check():
        v=resource_guard();require(v['worker_private_memory']['private_bytes']<=limit,'Reviewed whole CPU-child private allowance exceeded')
        return v
    return check

def load_request(pin,guard=resource_guard):
    request=read_bound(pin,guard)
    require(request['kind']==REQUEST_KIND and request['production_enabled'] is False,'Exact unarmed CPU update request')
    planned=read_bound(request['bridge'],guard);session=read_bound(planned['session'],guard)
    require(planned['kind']==bridge.KIND and planned['bridge_source_sha256']==record(bridge.__file__,guard)['sha256'] and planned['production_enabled'] is False,'Source-bound bridge4, no diagnostic-kind spoof')
    contract=verify_contract(request['contract'],session,guard=guard)
    validated=bridge.validate_DATA(planned['DATA_evidence'],guard)
    index=bridge.integer(request['update_index'],0,len(planned['updates'])-1)
    require(planned['maximum_feature_window_bytes']==bridge.CAPACITY and planned['raw_PCM_capacity_bytes']==bridge.PCM_CAPACITY and len(planned['updates'])<=20,'Fixed bounded window')
    return request,planned,planned['updates'][index],contract,validated

def numpy_runtime(*,frontend=False):
    # Preserve the independently measured CPU import order. Never install or
    # import the shared training environment's CUDA Torch in this child.
    sys.path.insert(0,str(TRAINING_SITE))
    import numpy as np
    sys.path.remove(str(TRAINING_SITE))
    torch=None
    if frontend:
        import torch
        require(torch.__version__=='2.7.1+cpu' and torch.version.cuda is None and Path(torch.__file__).resolve().is_relative_to(CPU_ENV),'Exact E CPU-only Torch required')
        torch.set_num_threads(4)
    sys.path.append(str(TRAINING_SITE))
    return np,torch

def geometry(samples):
    require(type(samples) is int and samples>=160,'Positive full waveform geometry')
    valid=samples//160
    return {'clean16k_samples':samples,'valid_mel_frames':valid,'physical_feature_frames':valid+1,'encoder_frames':(valid+7)//8}

def clean_header_geometry(samples,sample_rate,channels,expected_encoder_frames):
    """Fail before raw_plan/admission/allocation for a mismatched clean WAV."""
    require(type(samples) is int and samples>=160 and type(sample_rate) is int and sample_rate==16000
            and type(channels) is int and channels==1 and type(expected_encoder_frames) is int
            and expected_encoder_frames>0,'Exact original clean mono16k header and declared T required')
    actual=geometry(samples)
    require(actual['encoder_frames']==expected_encoder_frames,'Clean WAV header T differs before any PCM allocation or raw admission')
    return actual

def check_frontend(arrays,samples,np):
    g=geometry(samples);features,mask=arrays['input_features'],arrays['attention_mask']
    require(features.dtype==np.float32 and features.shape==(1,g['physical_feature_frames'],128) and mask.dtype==np.bool_ and mask.shape==(1,g['physical_feature_frames']),'Original physical STFT N//160+1 frames; never crop trailing frame')
    require(int(mask.sum())==g['valid_mel_frames'] and np.array_equal(mask,np.arange(g['physical_feature_frames'])[None,:]<g['valid_mel_frames']) and np.isfinite(features).all() and not features[~mask].any(),'Exact valid N//160 mask/zero-pad differs')
    require(g['physical_feature_frames']<=8*g['encoder_frames']+1,'Conservative physical8T+1 role bound')
    return g

def targets_sha(ids): return hashlib.sha256(b''.join(struct.pack('<I',v) for v in ids)).hexdigest()

class OriginalRows:
    """Lifetime read-locked final scalar metadata; serial decode, no PCM cache."""
    def __init__(self,validated,*,raw_contract,private_limit_bytes,frontend_enabled,guard=resource_guard):
        self.validated,self.guard=validated,guard;self.connections=[];self.locks=None;self.broad=None;self.cleaner=None
        self.raw_contract=verify_raw_review(raw_contract);self.private_limit=bridge.integer(private_limit_bytes,1)
        require(type(frontend_enabled) is bool,'Explicit reference/frontend raw scope')
        self.frontend_enabled=frontend_enabled;self.ledger=RawLifetimeLedger(guard=guard);self.active_job=None
        from bpe_declared_geometry_index_v1 import open_final,signature
        from bpe_windows_final_metadata_lock_v1 import FinalMetadataReadLocks
        self.signature=signature
        b=validated['bindings'];self.pins=[b[k] for k in ('final_catalog','final_labels','declared_geometry_index')]
        audits=validated['declared_geometry']['scalar_audit']['source_signatures']
        require(same([signature(p['path']) for p in self.pins],audits),'Final metadata signatures must match independent whole-DATA audit')
        self.signatures=deepcopy(audits)
        self.locks=FinalMetadataReadLocks([p['path'] for p in self.pins])
        try:
            for pin in self.pins:self.connections.append(open_final(pin['path']))
            ref=read_bound(b['heldout_PCM'],guard)
            require(ref['complete'] is True and ref['reference_clips']==12273 and len(ref['PCM_i16_sha256'])==12273,'Whole pinned heldout PCM reference set')
            self.heldout=set(ref['PCM_i16_sha256'])
        except BaseException:self.close();raise
        self.last=None;self.original_source_files=[];self.geometry_proof=None;self.replay_proof=None

    def stable(self):
        self.guard();require(same([self.signature(p['path']) for p in self.pins],self.signatures),'Immutable final metadata changed during preparation')

    def resolve(self,row,expected,phase):
        self.stable();base=deepcopy(row)
        base.pop('audio_draw_context',None)
        require(base['source_id']==expected['source_id'] and base['language']==expected['language'],'Exact ordered row identity')
        if expected['role']=='student_tr' and phase==bridge.WEAK:
            from bpe_runtime_declared_geometry_v1 import validate_declared_row
            ordinal=base['ordinal'];raw=self.connections[0].execute('SELECT * FROM clips WHERE ordinal=?',(ordinal,)).fetchone()
            labels=self.connections[1].execute('SELECT target_ids_u32,target_tokens,text_sha256 FROM labels WHERE ordinal=?',(ordinal,)).fetchone()
            declaration=self.connections[2].execute('SELECT * FROM geometry WHERE ordinal=?',(ordinal,)).fetchone()
            require(raw is not None and labels is not None and declaration is not None,'No unresolved retained metadata may be skipped')
            raw=dict(raw);blob,u,text_sha=labels;ids=[v[0] for v in struct.iter_unpack('<I',blob)]
            require(len(blob)==4*u and ids==expected['full_target_ids'] and text_sha==expected['raw_text_sha256'] and same(base['broad_original'],raw),'Exact complete raw source/full BPE text/labels')
            d=validate_declared_row(dict(declaration),raw,full_target_tokens=u)
            normalized=normalize_declared_TR(raw,d,ids)
            require(same(base,normalized),'Final original normalized source row differs')
            return raw,d
        if expected['role']=='student_tr':
            from bpe_cleaner_data_v1 import CleanerRows
            if self.cleaner is None:self.cleaner=CleanerRows()
            original=self.cleaner.row(base['cleaner_original_ordinal'])
            require(same(base,original) and base['tr_metadata_labels']==expected['full_target_ids'],'Exact complete original cleaner source/full targets')
        else:
            from bpe_replay_cache_labels_v1 import full_replay_labels
            pin=base['full_replay_label_binding']['cache']
            ids,proof=full_replay_labels(base,pin,guard=self.guard)
            require(ids==expected['full_target_ids'] and same(proof,base['full_replay_label_binding']),'Genuine original full replay cache targets')
        return None,None

    def __call__(self,row,expected,phase):
        import numpy as np
        raw,declaration=self.resolve(row,expected,phase);self.original_source_files=[];self.geometry_proof=None;self.replay_proof=None
        require(self.active_job is None,'Prior row raw buffers must be actually released before new decode')
        header=None
        if raw is None:
            import soundfile as sf
            with sf.SoundFile(scoped(row['audio_filepath'],source=True)) as stream:
                require(stream.samplerate==16000 and stream.channels==1 and stream.frames>=160,'Before-allocation original clean mono16k header')
                header={'samples':int(stream.frames),'sample_rate':stream.samplerate,'channels':stream.channels}
                clean_header_geometry(header['samples'],header['sample_rate'],header['channels'],row['encoder_frames'])
            n=header['samples'];sr=16000;c=n
        else:n,sr,c=declaration['native_samples'],declaration['native_sr'],declaration['clean16k_samples']
        plan=raw_plan(native_samples=n,native_rate=sr,clean_samples=c,phase=phase,role=expected['role'],
            channels_bound=1 if raw is None or raw['dataset']=='kiraat' else self.raw_contract['maximum_runtime_channels'],
            frame_samples_bound=self.raw_contract['maximum_decoder_frame_samples'])
        work=plan['workspace_incremental_envelopes'];review=self.raw_contract
        require(work['decoder']<=review['decoder_workspace_allowance_bytes'],'Unresolved decoder workspace envelope')
        if self.frontend_enabled:
            require(work['weak_transform_FFT']<=review['transform_workspace_allowance_bytes'] and work['original_frontend_spectral']<=review['frontend_workspace_allowance_bytes'],'Unresolved transform/frontend workspace envelope')
        workspace=max(work.values()) if self.frontend_enabled else work['decoder']
        prospective=plan['raw_peak_bytes']+plan['compressed_media_cap_bytes']+workspace+review['numerical_runtime_reserve_bytes']
        observation=self.guard();require(observation['worker_private_memory']['private_bytes']+prospective<=self.private_limit,'Unresolved whole private + raw/media/workspace BEFORE allocation')
        admit(prospective,self.guard)
        key=digest(expected['context']);self.ledger.begin(key,plan);self.active_job=key
        if raw is not None:
            from fastmss_bounded_original_audio_v1 import BoundedOriginalReader
            from bpe_runtime_declared_geometry_v1 import verify_actual_row
            if self.broad is None:self.broad=BoundedOriginalReader(ledger=self.ledger,source_inventory_binding=self.validated['bindings']['source_identity_inventory'],guard=self.guard,heldout_PCM_hashes=self.heldout)
            wave=self.broad.read_original(raw,job_key=key,declaration=declaration);actual=deepcopy(self.broad.last)
            actual.pop('wall_seconds',None) # telemetry is not immutable PCM identity
            self.geometry_proof=verify_actual_row(declaration,raw,actual,full_target_tokens=len(expected['full_target_ids']))
            actual_path=canonical_source_path(raw,ROOT)
            item=self.broad.final_source_identity.entries[str(actual_path)]
            source=source_identity_binding(raw,item,actual,ROOT);self.original_source_files=[source]
            if 'compressed_carrier_receipt' in actual:
                receipt=read_bound(actual['compressed_carrier_receipt'],self.guard)
                self.original_source_files += [receipt['records'],receipt['index'],actual['compressed_carrier_receipt']]
            else:self.original_source_files += [p for p in (item['index'],item['receipt']) if p is not None]
        else:
            import soundfile as sf
            path=scoped(row['audio_filepath'],source=True);pin=record(path,self.guard)
            with self.ledger.stage(key,'decode_resample'):
                with sf.SoundFile(path) as stream:
                    require(stream.samplerate==header['sample_rate'] and stream.channels==header['channels'] and stream.frames==header['samples'],'Clean source header unchanged BEFORE full read')
                    wave=stream.read(dtype='float32',always_2d=False)
                self.ledger.observe(key,wave,name='whole_original_clean',maximum_bytes=4*c)
            require(same(pin,record(path,self.guard)),'Original clean source changed during decode')
            self.original_source_files=[pin];actual={'native_sample_rate':16000,'native_samples':len(wave),'decoded_sample_rate':16000,'decoded_samples':len(wave),'encoder_frames':geometry(len(wave))['encoder_frames']}
            self.geometry_proof={'kind':'FastMSS_whole_original_clean_wave_geometry_v1','actual':deepcopy(actual),'whole_original_clip':True,'no_resample_or_trim':True}
        require(wave.dtype==np.float32 and wave.ndim==1 and wave.flags.c_contiguous and len(wave)==c and geometry(len(wave))['encoder_frames']==row['encoder_frames'],'Whole original waveform/declared T')
        with self.ledger.stage(key,'canonical_hash'):f32,h=chunked_hashes(wave,guard=self.guard)
        require(h not in self.heldout,'Exact heldout PCM hit is unresolved; no filtering/exclusion or draw consumption')
        self.last={**actual,'waveform_sha256':f32,'PCM_i16_sha256':h,**geometry(len(wave))}
        if expected['role']=='replay':
            self.replay_proof=genuine_replay_feature(row,expected,self.last['waveform_sha256'],self.guard)
            self.original_source_files.append(row['full_replay_label_binding']['cache'])
        self.stable();return wave

    def transform(self,operation,wave,row,context):
        require(self.active_job is not None,'Admitted raw row required before weak transform')
        with self.ledger.stage(self.active_job,'weak_transform'):
            prepared,receipt=operation(wave,row,context)
            self.ledger.observe(self.active_job,prepared,name='prepared_or_clean_alias',maximum_bytes=4*len(wave))
            return prepared,receipt

    def frontend(self,operation,wave):
        require(self.active_job is not None,'Admitted raw row required before original frontend')
        with self.ledger.stage(self.active_job,'original_frontend_waveform'):return operation(wave)

    def finish_raw_row(self):
        require(self.active_job is not None,'Exactly one admitted raw row')
        self.ledger.finish(self.active_job);self.active_job=None

    def raw_receipt(self):return self.ledger.receipt()

    def close(self):
        for c in self.connections:c.close()
        self.connections=[]
        if self.cleaner is not None:self.cleaner.close()
        if self.locks is not None:self.locks.close();self.locks=None

def genuine_replay_feature(row,expected,clean_sha,guard):
    from bpe_replay_cache_labels_v1 import ReplayMetadataUnpickler
    from bpe_checkpoint_scalar_metadata_v1 import TensorDescriptor
    pin=row['full_replay_label_binding']['cache'];path=scoped(pin['path'],source=True)
    require(same(record(path,guard),pin),'Exact genuine original replay cache SHA')
    before=(path.stat().st_size,path.stat().st_mtime_ns)
    with zipfile.ZipFile(path) as archive:
        entries=[i for i in archive.infolist() if i.filename.endswith('/data.pkl')]
        require(len(entries)==1 and entries[0].file_size<=1024**2,'Single bounded inert original replay metadata')
        prefix=entries[0].filename[:-8]
        require(archive.read(prefix+'byteorder')==b'little','Original little-endian replay storage')
        with archive.open(entries[0]) as stream:
            saved=ReplayMetadataUnpickler(stream).load();require(stream.read(1)==b'','Single full replay metadata')
        f=saved['features'];require(saved['source_id']==row['source_id'] and type(f) is TensorDescriptor and f.shape==(row['encoder_frames'],1024) and f.stride==(1024,1),'Full original contiguous teacher feature view')
        widths={'FloatStorage':4,'HalfStorage':2,'BFloat16Storage':2};width=widths[f.storage.dtype];count=math.prod(f.shape)
        require(f.offset+count<=f.storage.elements,'Full feature view bounds')
        entry=archive.getinfo(prefix+'data/'+f.storage.key)
        require(entry.file_size==f.storage.elements*width<=128*1024**2,'Bounded original teacher storage')
        sha=hashlib.sha256()
        with archive.open(entry) as stream:
            stream.seek(f.offset*width);remaining=count*width
            while remaining:
                guard();block=stream.read(min(1024**2,remaining));require(block,'Full genuine feature storage required');sha.update(block);remaining-=len(block)
    require(before==(path.stat().st_size,path.stat().st_mtime_ns),'Genuine replay cache changed during source proof')
    return {'source_id':row['source_id'],'cache':pin,'target_ids_u32_sha256':expected['target_ids_u32_sha256'],'clean_waveform_sha256':clean_sha,'genuine_feature_sha256':sha.hexdigest(),'genuine_feature_sha256_basis':'original stored dtype contiguous storage bytes','feature_dtype':f.storage.dtype,'feature_shape':list(f.shape),'feature_offset':f.offset,'teacher_features_rewritten':False}

def reference_entry(row,expected,reader):
    return {'original_row_sha256':expected['original_row_sha256'],'context':deepcopy(expected['context']),
        'full_target_ids':deepcopy(expected['full_target_ids']),'target_ids_u32_sha256':expected['target_ids_u32_sha256'],
        'expected_audio':{'sample_rate':16000,'samples':reader.last['clean16k_samples'],'waveform_sha256':reader.last['waveform_sha256']},
        'whole_original_clip':True,'augmentation_applied':False,'actual_geometry_and_heldout_checked':True,
        'actual_geometry':deepcopy(reader.last),'before_draw_geometry_proof':deepcopy(reader.geometry_proof),
        'original_source_files':deepcopy(reader.original_source_files),
        **({'teacher_or_replay_waveform_changed':False,'genuine_teacher_feature_cache':deepcopy(reader.replay_proof)} if expected['role']=='replay' else {})}
