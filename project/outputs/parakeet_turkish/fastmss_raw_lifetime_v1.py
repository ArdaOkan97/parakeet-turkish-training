"""Preallocation reservations for owned serial raw-waveform lifetimes.

The raw128MiB/two-job ledger is separate from compressed bytes and numerical
backend/feature workspace. Unsupported plans fail before allocation/draw;
there is no filtering, clipping of source length, fallback or implicit grace.
"""
from contextlib import contextmanager
from copy import deepcopy
import gc,hashlib,math,weakref

CAPACITY=128*1024**2
MAX_JOBS=2
MEDIA_PER_JOB=32*1024**2
CHUNK=65536

def require(value,message):
    if not value:raise ValueError(message)

def integer(value,minimum=0):
    require(type(value) is int and value>=minimum,'Exact nonboolean declared integer');return value

def canonical_source_path(row,root):
    from pathlib import Path
    root=Path(root).resolve()
    if row['dataset']=='kiraat':
        path=Path(row['shard']).resolve();require(path.is_relative_to(root) and path.suffix=='.parquet','Canonical original KIRAAT source path')
    else:
        require(row['dataset']=='alania_yodas3' and type(row['shard']) is str and len(row['shard'])==4 and row['shard'].isdigit(),'Exact YODAS four-digit shard')
        path=(root/'datasets/yodas3/audio'/str(row['shard']+'.tar')).resolve()
    return path

def source_identity_binding(row,item,actual,root):
    path=canonical_source_path(row,root)
    require(item['path']==str(path) and actual['actual_original_source_path']==str(path) and actual['source_file_sha256']==item['sha256'],'Canonical original source path and decoder SHA must match final inventory')
    for key in ('bytes','mtime_ns'):integer(item[key],1)
    require(item['receipt'] is not None and (row['dataset']!='alania_yodas3' or item['index'] is not None),'Preserve every source receipt and YODAS TAR index binding')
    return {k:item[k] for k in ('path','bytes','sha256','mtime_ns')}

def raw_plan(*,native_samples,native_rate,clean_samples,phase,role,channels_bound,frame_samples_bound):
    require(phase in ('weak_label_continued_training','cleaner_data_fine_tuning') and role in ('student_tr','replay','teacher'),'Explicit protected phase/role; no implicit weak routing')
    n=integer(native_samples,1);sr=integer(native_rate,1);c=integer(clean_samples,160)
    channels=integer(channels_bound,1);frame=integer(frame_samples_bound,1)
    require(sr in (16000,24000,48000) and c==(n*16000+sr-1)//sr,'Whole declared native/resampled geometry')
    require(channels<=8 and frame<=16384,'Explicit supported runtime channel/frame ceiling; unknown remains unresolved')
    hold=4*c
    # Native mono, a maximum decoder frame plus retained previous/first frame,
    # ndarray conversion and integer normalization/mono scratch, resample output
    # including finite FIR padding. Original polyphase output stays float32.
    frame_raw=channels*frame*8*6+frame*4*2
    decode=4*n+frame_raw+4*(c+128)+256*1024
    quant=CHUNK*8 # f32 scaled scratch + i16 + finite mask and bounded views
    weak=phase=='weak_label_continued_training' and role=='student_tr' and c<=120*16000
    # Frozen adapter RMS paths make full f64 copies/squares and a prepared f32
    # buffer/noise. FFT frequency-domain workspace is accounted separately.
    transform=32*c+1024**2 if weak else 0
    frontend_waveform=24*c+64*1024 # original tensor copies/pad/preemphasis masks
    retained=hold+(hold if weak else 0)
    peaks={'decode_resample':hold+decode,'canonical_hash':hold+quant,
        'weak_transform':retained+transform,'original_frontend_waveform':retained+frontend_waveform}
    raw_peak=max(peaks.values())
    fft_length=1<<(max(1,c+24000-1)-1).bit_length()
    workspaces={'decoder':8*1024**2,'weak_transform_FFT':6*16*fft_length if weak else 0,
        'original_frontend_spectral':(c//160+1)*257*8*10+8*1024**2}
    return {'kind':'FastMSS_preallocation_raw_lifetime_plan_v1','native_samples':n,'native_rate':sr,
        'clean16k_samples':c,'channels_bound':channels,'frame_samples_bound':frame,
        'stage_raw_peaks':peaks,'raw_peak_bytes':raw_peak,'clean_hold_bytes':hold,
        'prepared_hold_bytes':hold if weak else 0,'workspace_incremental_envelopes':workspaces,
        'compressed_media_cap_bytes':MEDIA_PER_JOB,'maximum_live_raw_jobs':MAX_JOBS,
        'weak_transform_possible':weak,'source_excluded_or_cropped':False,'draw_consumed':False}

class RawLifetimeLedger:
    def __init__(self,*,guard=lambda:None):
        self.guard=guard;self.jobs={};self.events=[];self.peak=0;self.job_peak=0;self.failed=False
    def begin(self,key,plan):
        self.guard();require(not self.failed and key not in self.jobs and len(self.jobs)<MAX_JOBS,'Owned raw job reused/failed/two-job limit')
        required=integer(plan['raw_peak_bytes'],1)
        require(sum(v['plan']['raw_peak_bytes'] for v in self.jobs.values())+required<=CAPACITY,'UNRESOLVED before allocation: simultaneous raw128MiB envelope does not fit')
        self.jobs[key]={'plan':deepcopy(plan),'arrays':[],'stages':[]}
        self.peak=max(self.peak,sum(v['plan']['raw_peak_bytes'] for v in self.jobs.values()));self.job_peak=max(self.job_peak,len(self.jobs))
        self.events.append({'event':'admitted_before_allocation','job':key,'raw_reserved_bytes':required})
    @contextmanager
    def stage(self,key,name):
        self.guard();job=self.jobs[key]
        require(not self.failed and name in job['plan']['stage_raw_peaks'] and not job['stages'],'Only one declared raw stage, no hidden overlapping operation')
        job['stages'].append(name)
        self.events.append({'event':'stage_preallocation_envelope_entered','job':key,'stage':name,'raw_stage_peak_bytes':job['plan']['stage_raw_peaks'][name]})
        try:yield
        except BaseException:self.failed=True;raise
        finally:
            job['stages'].pop();self.events.append({'event':'stage_returned','job':key,'stage':name});self.guard()
    def observe(self,key,array,*,name,maximum_bytes):
        # A post-allocation assertion supplements, never substitutes for begin
        # and stage admission. Weakrefs prove raw return buffers are gone.
        self.guard();job=self.jobs[key]
        require(not self.failed and integer(array.nbytes)<=integer(maximum_bytes) and array.flags.c_contiguous,'Actual raw allocation exceeds declared source/phase envelope')
        job['arrays'].append((name,weakref.ref(array),int(array.nbytes)))
        self.events.append({'event':'actual_return_buffer_verified','job':key,'name':name,'actual_bytes':int(array.nbytes),'declared_maximum_bytes':maximum_bytes})
    def finish(self,key):
        self.guard();gc.collect();job=self.jobs[key]
        require(not job['stages'] and all(ref() is None for _,ref,_ in job['arrays']),'Raw reservation cannot retire while a clean/prepared/decode return buffer is live')
        del self.jobs[key];self.events.append({'event':'raw_buffers_observed_released','job':key})
    def receipt(self):
        require(not self.jobs and not self.failed,'Only a fully released successful raw ledger can seal')
        return {'kind':'FastMSS_actual_raw_lifetime_ledger_v1','passed':True,'capacity_bytes':CAPACITY,
            'maximum_jobs':MAX_JOBS,'peak_reserved_raw_bytes':self.peak,'peak_simultaneous_jobs':self.job_peak,
            'events':deepcopy(self.events),'actual_raw_allocations_preceded_by_reservation':True,
            'compressed_and_spectral_workspace_are_separate_domains':True,'draw_or_sampler_advanced':False}

def chunked_hashes(wave,*,guard=lambda:None):
    """Exact original f32 multiply/rint/clip/i16 bytes, <=65536 samples scratch."""
    import numpy as np
    require(wave.dtype==np.float32 and wave.ndim==1 and wave.flags.c_contiguous,'Original contiguous mono float32 only')
    f32,i16=hashlib.sha256(),hashlib.sha256()
    for start in range(0,len(wave),CHUNK):
        guard();part=wave[start:start+CHUNK];require(np.isfinite(part).all(),'Nonfinite original source stays unresolved')
        f32.update(memoryview(part).cast('B'))
        scaled=part*32768;require(scaled.dtype==np.float32,'Pinned original canonical f32 promotion')
        np.rint(scaled,out=scaled);np.clip(scaled,-32768,32767,out=scaled)
        quantized=scaled.astype('<i2');i16.update(memoryview(quantized).cast('B'))
        del scaled,quantized,part
    return f32.hexdigest(),i16.hexdigest()
