"""Read-only GPU pacing gates and rate-limited bounded compressed-member input."""
import datetime
import hashlib
import io
import json
import os
import pathlib
import sqlite3
import time

import cpu_preparation_guard_v2 as base
from cpu_preparation_guard_v2 import PROJECT,ROOT,append_json,now,scoped,write_json

AREA=ROOT/'analysis/CPU_validation_v3'
base.AREA=AREA
PROOF=ROOT/'analysis/three_epoch_entry_v1/telemetry_stop_75250_v1/actual_recovery_native_positive_v1.json'
PROOF_SHA='12bfb6244061353884c27f137903cd59dac0b05d9dc9fa1f66a66a81f0d72a9e'

class PacedGuard(base.Guard):
    def __init__(self):
        super().__init__('paced_audio_QC_v3')
        self.proof_verified=False;self.last_poll=0.;self.last_snapshot=None;self.last_reason=None
        self.paused_seconds=0.;self.file_bytes_read=0;self.io_rate=1024*1024
        self.status(read_rate_limit_bytes_s=self.io_rate,read_block_max_bytes=65536,maximum_audio_workers=1,
            dynamic_pause_steps_mod250=[0,230,240],max_own_RSS_bytes=2_000_000_000)

    def gate_snapshot(self):
        current=time.monotonic()
        if self.last_snapshot is not None and current-self.last_poll<1:return self.last_snapshot
        self.last_poll=current
        snapshot={'allow':False,'reason':'pacing_state_unavailable'}
        try:
            if not self.proof_verified:
                if not PROOF.is_file():snapshot['reason']='ROOT_native_positive_proof_missing';return snapshot
                if hashlib.sha256(PROOF.read_bytes()).hexdigest()!=PROOF_SHA:raise ValueError('ROOT proof hash mismatch')
                self.proof_verified=True
            queue=json.loads((ROOT/'experiment_queue_state.json').read_text(encoding='utf-8-sig'))
            owner=json.loads((PROJECT/'work/gpu_training_owner.json').read_text(encoding='utf-8-sig'))
            loop_path=scoped(owner['actual_controller_loop'])
            if loop_path.suffix.lower()!='.json':loop_path=loop_path/'loop_status.json'
            loop=json.loads(loop_path.read_text(encoding='utf-8-sig'))
            step=int(loop['published_step']);resume=int(loop['resume_step'])
            sample=loop.get('actual_worker_self_memory',{}).get('sample',{})
            observed=sample.get('observed_at_utc') or sample.get('system_memory',{}).get('observed_at_utc')
            utc=datetime.datetime.now(datetime.timezone.utc)
            age=(utc-datetime.datetime.fromisoformat(observed)).total_seconds() if observed else 1e9
            loop_age=(utc-datetime.datetime.fromisoformat(loop['updated_at_utc'])).total_seconds()
            snapshot.update(step=step,resume_step=resume,loop_status_path=str(loop_path),worker_self_age_seconds=age,
                controller_age_seconds=loop_age,queue_status=queue.get('status'),loop_status=loop.get('status'))
            if queue.get('status')!='running' or loop.get('status')!='running':snapshot['reason']='GPU_queue_or_loop_not_running'
            elif queue.get('run_id')!=loop.get('run_id'):snapshot['reason']='GPU_current_run_identity_differs'
            elif step<=resume:snapshot['reason']='current_GPU_window_native_positive_update_pending'
            elif step%250 in {0,230,240}:snapshot['reason']='GPU_DEV_checkpoint_sensitive_steps'
            elif not 0<=age<=15 or not 0<=loop_age<=15:snapshot['reason']='GPU_self_or_controller_telemetry_stale'
            elif not loop.get('memory_observation',{}).get('sample_fresh',False):snapshot['reason']='GPU_memory_telemetry_not_fresh'
            else:snapshot.update(allow=True,reason='running_positive_fresh_outside_DEV_checkpoint')
        except (OSError,KeyError,TypeError,ValueError) as e:snapshot.update(allow=False,reason='pacing_read_unavailable',error=repr(e))
        finally:self.last_snapshot=snapshot
        return snapshot

    def wait(self,reserve_bytes=0):
        import psutil
        while True:
            super().wait(reserve_bytes)
            if psutil.Process().memory_info().rss>2_000_000_000:raise RuntimeError('Own paced CPU2GB RSS guard exceeded')
            snapshot=self.gate_snapshot();reason=snapshot['reason']
            if reason!=self.last_reason:
                event={'at_utc':now(),'pid':os.getpid(),**snapshot,'file_bytes_read':self.file_bytes_read}
                append_json(AREA/'pacing_gate_events_v3.jsonl',event);self.last_reason=reason
                self.status(phase='CPU_work_allowed' if snapshot['allow'] else 'CPU_work_paused_for_GPU',pacing_gate=snapshot,
                    file_bytes_read=self.file_bytes_read)
            if snapshot['allow']:return snapshot
            before=time.monotonic();time.sleep(1);self.paused_seconds+=time.monotonic()-before

    def pace(self,count):
        self.wait();until=time.monotonic()+count/self.io_rate
        while time.monotonic()<until:
            self.wait();time.sleep(min(.05,max(0,until-time.monotonic())))
        self.file_bytes_read+=count

    def read_file(self,path,expected_bytes=None):
        path=pathlib.Path(path).resolve()
        if not (path.is_relative_to(ROOT) or path.is_relative_to(PROJECT)):raise ValueError('Read path outside authorized storage/project')
        data=bytearray()
        with path.open('rb') as f:
            while True:
                self.wait();block=f.read(65536)
                if not block:break
                self.pace(len(block));data.extend(block)
                if expected_bytes is not None and len(data)>expected_bytes:raise ValueError('Pinned read length exceeded')
                if len(data)%(4*1024*1024)==0:self.status(phase='paced_existing_file_read',active_file=str(path),active_file_bytes_read=len(data),file_bytes_read=self.file_bytes_read)
        if expected_bytes is not None and len(data)!=expected_bytes:raise ValueError('Pinned read length mismatch')
        return bytes(data)

class MemberReader(io.RawIOBase):
    def __init__(self,path,offset,size,guard):
        self.stream=pathlib.Path(path).open('rb');self.offset=offset;self.size=size;self.position=0;self.guard=guard;self.bytes_read=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        if whence==0:position=offset
        elif whence==1:position=self.position+offset
        elif whence==2:position=self.size+offset
        else:raise ValueError('Unsupported bounded member seek')
        if not 0<=position<=self.size:raise ValueError('Seek outside pinned member extent')
        self.position=position;return position
    def read(self,count=-1):
        left=self.size-self.position
        count=min(left,65536 if count<0 else min(count,65536))
        if count<=0:return b''
        self.guard.pace(count);self.stream.seek(self.offset+self.position);data=self.stream.read(count)
        if len(data)!=count:raise ValueError('Truncated pinned compressed source')
        self.position+=len(data);self.bytes_read+=len(data);return data
    def close(self):
        if not self.closed:self.stream.close()
        super().close()

def decode_paced(row,guard):
    import av
    import numpy as np
    shard=row['shard'];rid=row['recording_id'];start=row['start_s'];duration=row['duration_s']
    if not 2<=duration<=20 or start<0:raise ValueError('Out of bounds requested clip')
    db=sqlite3.connect('file:'+(ROOT/f'indexes/yodas3_tar_{shard}.sqlite').as_posix()+'?mode=ro',uri=True)
    record=db.execute('SELECT offset,bytes FROM recordings WHERE recording_id=?',(rid,)).fetchone();db.close()
    if not record:raise ValueError('Missing pinned compressed member')
    offset,size=record
    if size>256*1024*1024:raise ValueError('Compressed recording exceeds unchanged256MiB bound')
    archive=ROOT/f'datasets/yodas3/audio/{shard}.tar';receipt=json.loads(archive.with_suffix('.tar.receipt.json').read_text(encoding='utf-8'))
    if receipt['revision']!='81d802f45a1c1e1f5331260484dacf1fbea81981' or archive.stat().st_size!=receipt['bytes']:raise ValueError('Pinned archive binding differs')
    guard.wait();before=time.monotonic();pause_start=guard.paused_seconds;n=round(duration*16000)
    pcm=np.zeros(n,np.float32);coverage=np.zeros(n,np.bool_)
    def retain(frame):
        if frame.pts is None:raise ValueError('Audio PTS missing')
        at=round((float(frame.pts*frame.time_base)-start)*16000);a=frame.to_ndarray().reshape(-1)
        lo=max(0,at);hi=min(n,at+len(a))
        if hi>lo:pcm[lo:hi]=a[lo-at:hi-at];coverage[lo:hi]=True
    with MemberReader(archive,offset,size,guard) as reader:
        with av.open(reader,mode='r',format='matroska',buffer_size=32768) as container:
            stream=container.streams.audio[0];stream.codec_context.thread_count=1;stream.codec_context.thread_type='NONE'
            container.seek(max(0,round((start-.10)*1_000_000)),backward=True,any_frame=False)
            rs=av.AudioResampler(format='fltp',layout='mono',rate=16000)
            for frame in container.decode(stream):
                guard.wait()
                if time.monotonic()-before-(guard.paused_seconds-pause_start)>90:raise TimeoutError('Bounded active decoder90s exceeded')
                if frame.pts is None:raise ValueError('Source PTS missing')
                if float(frame.pts*frame.time_base)>start+duration+.15:break
                for out in rs.resample(frame):retain(out)
            for out in rs.resample(None):retain(out)
        read_bytes=reader.bytes_read
    if not np.isfinite(pcm).all() or float(coverage.mean())<.995:raise ValueError('Invalid timestamp-bound PCM coverage')
    return pcm,{'compressed_member_bytes':size,'compressed_member_offset':offset,'compressed_bytes_actually_read':read_bytes,
        'archive_sha256_bound_to_existing_receipt':receipt['sha256'],'PCM_f32le_sha256':hashlib.sha256(pcm.tobytes()).hexdigest(),
        'timestamp_coverage_fraction':float(coverage.mean()),'decode_elapsed_wall_s':time.monotonic()-before,
        'GPU_sensitive_pause_s':guard.paused_seconds-pause_start,'sample_rate':16000,'decoded_samples':n,
        'decoder':'PyAV direct bounded seekable original tar member; no extraction; threads1; <=1MiB/s',
        'permanent_or_temporary_PCM_WAV_or_compressed_member_written':False,'whole_compressed_member_hash_recomputed':False}
