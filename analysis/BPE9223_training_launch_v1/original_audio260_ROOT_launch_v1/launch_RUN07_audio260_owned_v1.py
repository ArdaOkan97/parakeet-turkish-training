"""Saved260 exact abnormal-audio continuation with unchanged full-run admissions."""
import argparse, ctypes, json, os, runpy, shutil, subprocess, sys, time
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path
import psutil

P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
sys.path[:0]=[str(P/'outputs/parakeet_turkish'),str(P/'work/research')]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
import run_ASR18_BPE9223_full_run_owner_v19 as service

def now(): return datetime.now(timezone.utc).isoformat()
def identity():
    p=psutil.Process();return dict(pid=p.pid,ppid=p.ppid(),create_time=p.create_time())
def dead(v):
    try: return abs(psutil.Process(v['pid']).create_time()-v['create_time'])>.001
    except psutil.NoSuchProcess:return True

def main():
    a=argparse.ArgumentParser();a.add_argument('--handoff',required=True);a.add_argument('--folder',required=True)
    a.add_argument('--prelaunch',required=True);args=a.parse_args()
    require(sys.flags.utf8_mode==1 and os.environ.get('PYTHONUTF8')=='1','UTF8 before metadata/process entry')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and all(os.environ.get(k)=='1' for k in
        ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'CPU-masked single-thread parent environment')
    service.storage()
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    kernel.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    kernel.SetPriorityClass.restype=wintypes.BOOL
    require(kernel.SetPriorityClass(kernel.GetCurrentProcess(),0x4000),'Actual BelowNormal parent class')
    require(psutil.Process().nice()==psutil.BELOW_NORMAL_PRIORITY_CLASS,'Validate actual BelowNormal parent')
    h=read_json(args.handoff)
    for k in ('configuration','execution_review','entry','source_manifest','resume_request'):
        require(record(h[k]['path'])==h[k],'Exact launch handoff '+k)
    c=read_json(h['configuration']['path']);review=read_json(h['execution_review']['path'])
    require(review['accepted'] is True and review['reviewer']=='/root' and review['configuration']==h['configuration']
        and review['invocation']=='original_audio260_resume' and review['resume_request']==h['resume_request'],
        'Genuine exact ROOT saved260 original-audio continuation execution')
    require(review['administrative_maximum_new_updates'] is None and h['administrative_update_stop'] is None
        and h['full_run_updates']==401072,'Full declared training with no discretionary update stop')
    require(c['full_owner_controller_source']==h['entry'] and c['parent_driver_source']==record(service.__file__),
        'Direct entry and frozen parent services source identities')
    service.check_bindings(c)
    from asr18_BPE9223_full_run_controller_v23 import checked_failed_audio260_resume_request
    def normal_guard():
        sample=service.sample_memory(os.getpid(),'original_audio260_resume_metadata_admission')
        require(sample['paired_sampling_within_one_second'] and sample['worker_private_memory']['private_bytes']<=128000000,
            'Bounded owned original-audio260 metadata parent')
        system=sample['system_memory'];require(system['commit_fraction']<.95 and system['commit_available_bytes']>2000000000,
            'Existing system critical/headroom while verifying saved260 state')
        return sample
    snapshot=checked_failed_audio260_resume_request(h['configuration'],review,h['resume_request'],
        parent_source=c['parent_driver_source'],guard=normal_guard)
    lease_path=P/'work/gpu_training_owner.json';dispatch_path=N/'current_training_dispatch.json'
    lease=read_json(lease_path)
    failure_pin=c['prior_committed260_original_audio_failure'];failed=read_json(failure_pin['path'])
    pointer=read_json(snapshot['current_capture']['pointer']['path']);updates=pointer['committed_updates']
    require(updates==260 and snapshot['source_failure']==failure_pin
        and snapshot['current_capture']==failed['current_checkpoint_capture']
        and failed['actual_tool_exit_code']==1 and failed['actual_native_normal_exit0_claimed'] is False
        and failed['all_captured_identities_dead'] is True and len(failed['all_captured_identities'])==7
        and all(dead(x) for x in failed['all_captured_identities']),
        'Exact aborted original writer/dead captured family and preserved260; no normal0 relabel')
    require(lease['owner']=='/root/memory_diagnosis' and lease['actual_GPU_running'] is False
        and lease['actual_owned_worker_pid'] is None
        and lease.get('actual_parent') in (None,failed['producer']),'Exclusive quiescent saved260 lease')
    prior=[failure_pin]
    require(not psutil.Process().children(recursive=True),'No existing owned child before new launch')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.free,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    require(len(rows)==1,'Exact single GPU inventory')
    total,free,used=[int(x.strip())*1024**2 for x in rows[0].split(',')]
    policy=c['CUDA_capacity_policy']
    require(total>=policy['minimum_physical_bytes'] and free>=policy['allocator_cap_bytes']+policy['hard_free_physical_reserve_bytes'],
        'Reviewed physical GPU capacity/free desktop reserve admission')
    free_E=shutil.disk_usage(r'E:\gpu_training_scratchpad').free
    require(free_E>=100000000000+602189856768,'Reviewed E reserve plus complete declared storage horizon')
    samples=[]
    for _ in range(3):
        row={'merged_CPU':service.fresh_admit(1995108864),
            'native':service.fresh_admit(c['entire_worker_forecast']['entire_worker_private_forecast_bytes']+384000000+64000000),
            'fresh_restore':service.fresh_admit(3350000000+384000000+64000000)}
        require(all(v['passed'] for v in row.values()),'Fresh unchanged paired system/whole-domain budgets')
        samples.append(row);time.sleep(.2)
    folder=Path(args.folder).resolve();pre=Path(args.prelaunch).resolve()
    require(folder.is_relative_to(N) and pre.is_relative_to(N) and not folder.exists() and not pre.exists(),
        'Fresh owned E launch output and prelaunch receipt')
    me=identity()
    write_json(pre,dict(kind='ASR18_actual_exact_original_audio260_RUN07_continuation_prelaunch_v1',passed=True,producer=me,
        configuration=h['configuration'],execution_review=h['execution_review'],actual_entry_controller_source=h['entry'],
        parent_service_authority=record(service.__file__),actual_GPU_inventory=dict(total_bytes=total,free_bytes=free,used_bytes=used),
        E_free_bytes=free_E,samples=samples,prior_closed_failures_preserved=prior,utf8_mode=sys.flags.utf8_mode,
        PYTHONUTF8=os.environ['PYTHONUTF8'],CPU_mask_before_launch=True,BelowNormal_verified=True,
        exact_saved260_same_genesis_namespace=c['run_namespace'],resume_request=h['resume_request'],
        preserved_optimizer_updates=updates,planned_native_updates=401072,administrative_update_stop=None))
    lease.update(status='production_RUN07_saved260_continuation_starting',purpose='Human-authorized original-source full401072 training',
        running=True,actual_GPU_running=False,actual_training_optimizer_running=False,actual_owned_worker_pid=None,
        actual_owned_launcher_pid=None,actual_parent=me,actual_owned_parent_pid=me['pid'],actual_owned_session=None,
        current_owned_session=None,current_run_folder=str(folder),run_namespace=c['run_namespace'],
        current_configuration=h['configuration'],current_execution_review=h['execution_review'],
        committed_updates=updates,production_enabled=True,mode='training',updated_at_utc=now())
    write_json(lease_path,lease)
    dispatch=read_json(dispatch_path)
    dispatch.update(status='production_RUN07_saved260_continuation_starting',mode='training',owner=lease['owner'],running=True,
        actual_GPU_running=False,actual_training_optimizer_running=False,actual_parent=me,
        actual_owned_session=None,current_owned_session=None,current_run_folder=str(folder),run_namespace=c['run_namespace'],
        configuration=h['configuration'],execution_review=h['execution_review'],committed_updates=updates,
        planned_native_updates=401072,administrative_update_stop=None,updated_at_utc=now())
    write_json(dispatch_path,dispatch)
    print(json.dumps(dict(actual_parent=me,actual_prelaunch=record(pre),configuration=h['configuration'],folder=str(folder))),flush=True)
    sys.argv=[h['entry']['path'],'--config',h['configuration']['path'],'--review',h['execution_review']['path'],'--folder',str(folder),'--resume-request',h['resume_request']['path']]
    runpy.run_path(h['entry']['path'],run_name='__main__')

if __name__=='__main__':main()
