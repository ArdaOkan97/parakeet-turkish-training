"""NEW reviewed opt-in TRAIN/TR preparation; no producer/draw/commit ownership.

Recorded cafe/fan/traffic and the frozen 16 simulated impulses replace the
Gaussian/two-position pilot only in a separately admitted benchmark namespace.
Training keeps exactly original N samples. Listening full tails are separate.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import fastmss_real_noise_v1 as mixer
import fastmss_real_noise_train_adapter_v1 as decoder
import fastmss_room_bank_preview_v1 as rooms

RATE=16000
WEAK='weak_label_continued_training'
KIND='FastMSS_recorded_noise_16room_TRAIN_policy_v1'
IDENTITY='FastMSS_recorded_noise_16room_ordered_draw_identity_v1'
RAW_EXTRA=24*1024**2


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()


def pin(path,guard=lambda:None):
    path=Path(path).resolve();project=Path(__file__).resolve().parents[2]
    mixer.require(path.is_relative_to(mixer.ROOT.resolve()) or
        path.suffix=='.py' and path.is_relative_to(project),'E assets/metadata or existing project source only')
    before=path.stat();mixer.require(0<before.st_size<=32*1024**2,'Bounded exact source/asset/metadata pin')
    hashed=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1024**2):guard();hashed.update(block)
    after=path.stat();mixer.require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Pinned file changed')
    return {'path':str(path),'bytes':before.st_size,'sha256':hashed.hexdigest()}


def policy_identity(policy_pin,guard=lambda:None):
    mixer.require(pin(policy_pin['path'],guard)==policy_pin,'Exact new training policy pin')
    policy=json.loads(Path(policy_pin['path']).read_text(encoding='utf-8-sig'))
    mixer.require(policy['kind']==KIND and policy['benchmark_enabled'] is True
        and policy['production_enabled'] is False and policy['execution_requires_ROOT_review'] is True
        and policy['sample_rate']==RATE and policy['max_waveform_seconds']==120
        and policy['apply_probability']==.5 and policy['noise_probability']==.5 and policy['rir_probability']==.5
        and policy['snr_db_choices']==[20,10,5] and policy['noise_excerpt_seconds']==30
        and policy['RIR_length_policy']=='strongest_tap_unit_L2_full_convolution_keep_original_N_RMS_restore'
        and policy['peak_ceiling']==.98 and policy['noise_mode']=='verified_recorded_assets_only',
        'Exact explicit original-N training recipe; no silent pilot or preview policy')
    mixer.require(type(policy['seed']) is int and 0<=policy['seed']<2**64,'Exact local seed')
    for key in ('noise_registry','room_registry'):
        mixer.require(pin(policy[key]['path'],guard)==policy[key],'Frozen licensed/simulated registry differs')
    return policy,{'kind':IDENTITY,'policy_file':policy_pin,'policy_sha256':hashlib.sha256(canonical(policy)).hexdigest(),
        'adapter':pin(__file__,guard),'mixer':pin(mixer.__file__,guard),'decoder':pin(decoder.__file__,guard),
        'room_provenance_helper':pin(rooms.__file__,guard),'noise_registry':policy['noise_registry'],
        'room_registry':policy['room_registry'],'draw_owner':'immutable original sampler source/epoch/draw_index',
        'commit_owner':'native caller after actual positive optimizer update',
        'production_enabled':False,'new_draw_counter_or_global_RNG_advance':False}


class RecordedRoomTrainingAdapter:
    def __init__(self,policy_pin,*,reader,guard,ROOT_review,configuration):
        self.guard=guard;self.reader=reader
        self.policy,self.identity=policy_identity(policy_pin,guard)
        review=json.loads(Path(ROOT_review['path']).read_text(encoding='utf-8-sig'))
        mixer.require(pin(ROOT_review['path'],guard)==ROOT_review and review['accepted'] is True
            and review['reviewer']=='/root' and review['configuration']==configuration
            and review['scope']=='recorded_noise_16room_aggregate_CPU_first4_benchmark_v1',
            'Exact external ROOT configuration execution review, no implicit activation')
        verified=mixer.validate_registry(self.policy['noise_registry']['path'],guard=guard)
        mixer.require(verified['binding']==self.policy['noise_registry']
            and len(verified['registry']['assets'])==3
            and all(a['license']['audio_specific'] is True for a in verified['registry']['assets']),
            'Exactly the three verified audio-specific CC0 real recordings')
        self.noises=verified['registry']['assets']
        self.rirs=rooms.load_room_bank(self.policy['room_registry']['path'],guard=guard)
        mixer.require(len(self.rirs)==16 and len({a['id'] for a in self.rirs})==16,
                      'Exactly sixteen frozen simulated room/position conditions')
        for asset in self.rirs:
            mixer.require(asset['sample_rate']==RATE and type(asset['samples']) is int
                and 0<asset['samples']<=7*RATE,'Bounded actual impulse geometry')

    def transform(self,wave,row,context):
        """Upgrade active raw reservation BEFORE any new augmentation allocation."""
        n=len(wave); key=self.reader.active_job;ledger=self.reader.ledger
        mixer.require(key is not None and key in ledger.jobs and not ledger.jobs[key]['stages'],
                      'Exactly one admitted clean row, outside other raw stages')
        eligible=context['split']=='train' and context['phase']==WEAK and row['language']=='tr' and n<=120*RATE
        if eligible:
            plan=deepcopy(ledger.jobs[key]['plan']);new_raw=max(plan['raw_peak_bytes'],40*n+RAW_EXTRA)
            fft=1<<(max(1,n+7*RATE-1)-1).bit_length()
            workspace=max(plan['workspace_incremental_envelopes']['weak_transform_FFT'],96*fft)
            review=self.reader.raw_contract
            mixer.require(workspace<=review['transform_workspace_allowance_bytes'],
                          'Expanded actual RIR FFT workspace must fit unchanged transform allowance')
            from fastmss_raw_lifetime_v1 import CAPACITY
            other=sum(v['plan']['raw_peak_bytes'] for k,v in ledger.jobs.items() if k!=key)
            mixer.require(other+new_raw<=CAPACITY and len(ledger.jobs)<=2,'No raw128MiB/two-job relaxation')
            whole_workspace=max(workspace,*plan['workspace_incremental_envelopes'].values())
            extra=new_raw+plan['compressed_media_cap_bytes']+whole_workspace+review['numerical_runtime_reserve_bytes']
            observation=self.guard()
            mixer.require(observation['worker_private_memory']['private_bytes']+extra<=self.reader.private_limit,
                          'New raw/media/noise-resample/RIR workspace BEFORE allocation exceeds leaf private cap')
            import fastmss_row_serial_cpu_v8 as core
            core.admit(extra,self.guard)
            plan['raw_peak_bytes']=new_raw;plan['stage_raw_peaks']['weak_transform']=new_raw
            plan['workspace_incremental_envelopes']['weak_transform_FFT']=workspace
            plan['recorded_room_extension']={'source':self.identity['adapter'],'raw_formula':'max(original,40*N+24MiB)',
                'raw_bytes':new_raw,'FFT_workspace_bytes':workspace,'whole_stage_workspace_bytes':whole_workspace,'noise_native_maximum_frames':30*48000,
                'maximum_native_channels':2,'no_decoded_asset_cache':True}
            ledger.jobs[key]['plan']=plan;ledger.peak=max(ledger.peak,other+new_raw)
            ledger.events.append({'event':'recorded_room_extension_admitted_before_allocation','job':key,
                'raw_reserved_bytes':new_raw,'FFT_workspace_bytes':workspace,'whole_stage_workspace_bytes':whole_workspace,
                'prospective_incremental_bytes':extra,'adapter':self.identity['adapter']})
        return self.reader.transform(self.apply,wave,row,context)

    def frontend(self,operation,wave):
        # Transform libraries/caches may have grown since the original decode
        # admission. Re-admit the complete current raw/media/MAX(all stages)
        # domain immediately before the unchanged original frontend allocates.
        key=self.reader.active_job
        mixer.require(key is not None and not self.reader.ledger.jobs[key]['stages'],
                      'Fresh quiescent raw stage before original frontend')
        plan=self.reader.ledger.jobs[key]['plan'];review=self.reader.raw_contract
        workspace=max(plan['workspace_incremental_envelopes'].values())
        extra=plan['raw_peak_bytes']+plan['compressed_media_cap_bytes']+workspace+review['numerical_runtime_reserve_bytes']
        value=self.guard()
        mixer.require(value['worker_private_memory']['private_bytes']+extra<=self.reader.private_limit,
                      'Post-transform current private plus complete frontend/raw/media/workspace domain exceeds leaf cap')
        import fastmss_row_serial_cpu_v8 as core
        admitted=core.admit(extra,self.guard)
        self.reader.ledger.events.append({'event':'recorded_room_fresh_full_domain_before_frontend','job':key,
            'raw_reserved_bytes':plan['raw_peak_bytes'],'whole_stage_workspace_bytes':workspace,
            'prospective_incremental_bytes':extra,'adapter':self.identity['adapter']})
        return self.reader.frontend(operation,wave)

    def _observe(self,array,name):
        plan=self.reader.ledger.jobs[self.reader.active_job]['plan']
        self.reader.ledger.observe(self.reader.active_job,array,name=name,maximum_bytes=plan['stage_raw_peaks']['weak_transform'])
        return array

    def apply(self,wave,row,context):
        import numpy as np
        p=self.policy;n=len(wave)
        base={'policy_sha256':self.identity['policy_sha256'],'source_id':row['source_id'],'applied':False,
            'label_policy':'unchanged_single_target','epoch':context['epoch'],'draw_index':context['draw_index'],
            'ordered_context_sha256':hashlib.sha256(canonical(context)).hexdigest()}
        mixer.require(type(context['epoch']) is int and context['epoch']>=0
            and type(context['draw_index']) is int and context['draw_index']>=0
            and context['source_id']==row['source_id'] and context['language']==row['language'],
            'Exact actual original source/language/epoch/draw context')
        if context['split']!='train' or context['phase']!=WEAK or row['language']!='tr':
            return wave,{**base,'reason':'protected_replay_teacher_cleaner_or_evaluation'}
        if n>120*RATE:return wave,{**base,'reason':'complete_long_source_retained_clean_above120s'}
        mixer._wave(wave,guard=self.guard)
        stats=mixer._stats(wave,guard=self.guard)
        if stats['RMS']<=1e-12:return wave,{**base,'reason':'silent_complete_source_retained'}
        seed=int.from_bytes(hashlib.sha256(canonical({'seed':p['seed'],'policy':self.identity['policy_sha256'],
            'source_id':row['source_id'],'epoch':context['epoch'],'draw_index':context['draw_index']})).digest()[:16],'little')
        rng=np.random.Generator(np.random.PCG64(seed));base['augmentation_seed']=str(seed)
        if rng.random()>=p['apply_probability']:return wave,{**base,'reason':'deterministic_clean_control'}
        output=wave;operations=[]
        if rng.random()<p['rir_probability']:
            import soundfile as sf
            from scipy.signal import fftconvolve
            asset=self.rirs[int(rng.integers(len(self.rirs)))]
            mixer.require(rooms.pin(asset['audio']['path'],guard=self.guard)==asset['audio'],'Selected impulse SHA')
            with sf.SoundFile(asset['audio']['path']) as source:
                mixer.require(source.samplerate==RATE and source.channels==1 and len(source)==asset['samples']
                    and len(source)<=7*RATE,'Selected impulse header before read')
                rir=self._observe(source.read(dtype='float32'),'recorded_room_RIR')
            strongest=int(np.argmax(np.abs(rir)));aligned=rir[strongest:]
            norm=math.sqrt(float(np.dot(aligned.astype(np.float64),aligned.astype(np.float64))))
            mixer.require(norm>0,'Nonzero finite impulse energy')
            aligned=self._observe((aligned/np.float32(norm)).astype(np.float32),'normalized_RIR')
            full=self._observe(fftconvolve(wave,aligned,mode='full').astype(np.float32),'full_convolution')
            output=self._observe(full[:n].copy(),'original_N_reverberant')
            wet=mixer._stats(output,guard=self.guard)
            mixer.require(wet['RMS']>1e-12,'RIR cannot erase original speech')
            output*=np.float32(stats['RMS']/wet['RMS'])
            operations.append({'kind':'rir','asset_id':asset['id'],'room_id':asset['room_id'],'position':asset['position'],
                'asset_audio':asset['audio'],'registry':p['room_registry'],'strongest_tap_offset_samples':strongest,
                'physical_direct_arrival_claimed':False,'normalization':'unit_L2','RMS_restored_over_original_N':True,
                'input_samples':n,'output_samples':n,'source_speech_cropped':False,
                'convolution_tail_policy':'discard_output_after_original_N_not_full_tail_preview'})
            del rir,aligned,full,wet
        if rng.random()<p['noise_probability']:
            asset=self.noises[int(rng.integers(len(self.noises)))];snr=p['snr_db_choices'][int(rng.integers(3))]
            decode_seed=int(rng.integers(0,2**64,dtype=np.uint64));segment_seed=int(rng.integers(0,2**64,dtype=np.uint64))
            noise,decoded=decoder.decode_noise_excerpt(asset,seed=decode_seed,maximum_seconds=30,guard=self.guard)
            self._observe(noise,'recorded_noise_excerpt')
            segment,selection=mixer.prepare_noise_segment(noise,n,segment_seed,guard=self.guard)
            self._observe(segment,'recorded_noise_segment')
            prepared=mixer.prepare_mix(output,segment,snr,guard=self.guard)
            for key in ('scaled_noise','unscaled_mixture'):self._observe(prepared[key],key)
            operations.append({'kind':'noise','asset_id':asset['id'],'asset_audio':decoded['asset_audio'],
                'license_SPDX':asset['license']['spdx'],'registry':p['noise_registry'],'decoder':decoded,
                'segment':selection,'mix':prepared['receipt'],'Gaussian_used':False})
            output=prepared['unscaled_mixture']
            del prepared,segment,noise
        if not operations:return wave,{**base,'reason':'deterministic_operation_clean_control'}
        peak=mixer._stats(output,guard=self.guard)['peak'];gain=min(1.,p['peak_ceiling']/peak) if peak else 1.
        result=self._observe((output*np.float32(gain)).astype(np.float32),'final_recorded_room_original_N')
        mixer.require(result.shape==wave.shape and result.dtype==np.float32 and result.flags.c_contiguous
            and mixer._stats(result,guard=self.guard)['peak']<=1.,'Exact original-N finite no-clipping output')
        return result,{**base,'applied':True,'operations':operations,'peak_scale':gain,'samples_unchanged':n,
            'sample_rate':RATE,'speed_or_speaker_mixing':False,'prepared_not_committed':True,
            'sampler_journal_frontier_advanced':False,'adapter':self.identity['adapter']}
