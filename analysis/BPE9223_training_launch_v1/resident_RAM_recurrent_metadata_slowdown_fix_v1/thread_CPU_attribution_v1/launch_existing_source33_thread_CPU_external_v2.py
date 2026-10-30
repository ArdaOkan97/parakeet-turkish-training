"""One separately reviewed existing-source33 nonblocking observation; external guard remains active through all parsing/writing/cleanup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import psutil

P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).parent
FIXTURE=F/'observe_existing_source33_thread_CPU_and_stacks_v2.py'
PYTHON=Path('E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe')
RESULT=F/'existing_source33_thread_CPU_external_result_v1.json'
OUTPUT=F/'existing_source33_thread_CPU_actual_02'
LIMIT=536870912;PREFETCH=268435456
PROTECTED={3652,31268,30732,26892}


def require(value,message):
    if not value:raise ValueError(message)


def pin(path):
    p=Path(path).resolve();data=p.read_bytes();require(len(data)<=8*1024**2,'Compact/source pin only')
    return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def identity(proc):return {'pid':proc.pid,'create_time':proc.create_time(),'ppid':proc.ppid()}


def same_alive(who):
    try:return abs(psutil.Process(who['pid']).create_time()-who['create_time'])<.001
    except psutil.NoSuchProcess:return False


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ROOT-review',required=True);args=parser.parse_args()
    me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    review=json.loads(Path(args.ROOT_review).read_text(encoding='utf-8'))
    require(review['accepted']is True and review['reviewer']=='/root'
        and review['scope']=='ASR18_existing_source33_GetThreadTimes_nonblocking60s_attribution_v1'
        and review['live_observer_source']==pin(FIXTURE)and review['external_live_launcher_source']==pin(__file__)
        and Path(review['output_path']).resolve()==OUTPUT.resolve(),'Exact ROOT source/diagnostic binding')
    require(all(type(w['pid'])is int for w in review['protected_production_identities']),'Exact ROOT-bound production cleanup exclusions')
    PROTECTED.update(w['pid']for w in review['protected_production_identities'])
    require(not RESULT.exists()and not OUTPUT.exists(),'ONE fresh actual observation, preserve every prior outcome')
    for value in review['input_pins']:require(pin(value['path'])==value,'All compact/source inputs pinned before launch')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and os.environ.get('PYTHONUTF8')=='1'
        and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
        and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),
        'Original E process-local CPU-only limits')
    began=time.monotonic();popen=None;popen_identity=None;observed={};cleanup=[];guard_error=None;peak=0;initial=None;last=None
    exact_external_interpreter_launcher=identity(me.parent())
    def remember(proc):
        who=identity(proc);observed[(who['pid'],who['create_time'])]=who;return who
    remember(me)
    def observe(admission=False):
        nonlocal peak,last
        require(same_alive(exact_external_interpreter_launcher),'Exact live external interpreter launcher included conservatively in family')
        family=[me,psutil.Process(exact_external_interpreter_launcher['pid'])]
        if popen is not None and popen_identity is not None and same_alive(popen_identity):
            try:
                root=psutil.Process(popen_identity['pid']);family+=[root,*root.children(recursive=True)]
            except psutil.NoSuchProcess:pass
        total=0
        for proc in family:
            try:
                if proc.pid!=exact_external_interpreter_launcher['pid']:remember(proc)
                sample=sample_memory(proc.pid,'existing_source33_thread_CPU_external_guard')
                require(sample['paired_sampling_within_one_second'],'Original paired memory sample')
                total+=sample['worker_private_memory']['private_bytes'];last=sample
            except psutil.NoSuchProcess:continue
        peak=max(peak,total);system=last['system_memory']
        prospective=LIMIT if admission else 0
        require(total<=LIMIT and time.monotonic()-began<120,'Sampled family512MiB/external120s')
        require(system['commit_available_bytes']>=(4518435456+prospective if admission else 2000000000+PREFETCH)
            and system['physical_available_bytes']>=1500000000+PREFETCH+prospective
            and shutil.disk_usage('E:/').free>=100000000000,'Admission FULL225+prefetch+prospective512MiB; runtime originalfloors+FULLprefetch; E100GB')
        return sample
    stdout=F/'existing_source33_thread_CPU_v1_stdout.log';stderr=F/'existing_source33_thread_CPU_v1_stderr.log'
    result={'kind':'ASR18_existing_source33_thread_CPU_nonblocking_external_result_v1','passed':False,
        'external_identity':identity(me),'launcher_identity':identity(me.parent()),'ROOT_review':pin(args.ROOT_review),
        'fixture_source':pin(FIXTURE),'external_source':pin(__file__),'input_pins_before':review['input_pins'],
        'Popen_returncode':None,'external_guard_error':None,'required_external_cleanup':cleanup}
    try:
        initial=observe(admission=True)
        with stdout.open('xb')as out,stderr.open('xb')as err:
            command=[str(PYTHON),'-Xutf8','-B',str(FIXTURE),'--ROOT-review',str(Path(args.ROOT_review).resolve())]
            popen=subprocess.Popen(command,cwd=P,stdout=out,stderr=err,
                creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            popen_identity=remember(psutil.Process(popen.pid))
            result['original_Popen_identity']=popen_identity
            result['exact_arguments']=command
            while popen.poll()is None:
                observe();time.sleep(.05)
            result['Popen_returncode']=popen.wait(timeout=10)
            observe()
    except BaseException as error:
        guard_error=repr(error);result['external_exception_traceback']=traceback.format_exc()
    finally:
        if popen is not None and popen.poll()is None:
            try:
                require(popen_identity is not None and same_alive(popen_identity),'Cleanup exact initial Popen identity, never PID alone')
                require(popen_identity['pid'] not in PROTECTED,'Never native, feature CPU or reader cleanup')
                root=psutil.Process(popen_identity['pid'])
                owned=[]
                for proc in root.children(recursive=True):
                    who=remember(proc);depth=len(proc.parents());owned.append((depth,who))
                owned.sort(key=lambda item:item[0],reverse=True)
                owned=[who for _,who in owned]+[popen_identity]
                for who in owned:
                    if not same_alive(who):continue
                    require(who['pid'] not in PROTECTED,'Protected production identities are never cleanup targets')
                    require(same_alive(popen_identity),'Original owned launcher remains exact during ancestry check')
                    proc=psutil.Process(who['pid'])
                    require(who==popen_identity or any(p.pid==popen_identity['pid']and abs(p.create_time()-popen_identity['create_time'])<.001 for p in proc.parents()),'Fresh exact owned ancestry before cleanup')
                    require(same_alive(who),'Recheck exact descendant create-time immediately before kill')
                    proc.kill();cleanup.append(who)
                popen.wait(timeout=10)
            except BaseException as error:result['cleanup_error']=repr(error)
        for who in list(observed.values()):
            if who['pid']==me.pid:continue
            try:
                proc=psutil.Process(who['pid'])
                if abs(proc.create_time()-who['create_time'])<.001:proc.wait(timeout=10)
            except psutil.NoSuchProcess:pass
            except psutil.TimeoutExpired:result.setdefault('not_closed_exact_identities',[]).append(who)
        result['external_guard_error']=guard_error
        result['recorded_identities']=list(observed.values())
        result['all_observed_owned_descendants_exact_dead']=[{'identity':who,'exact_dead':not same_alive(who)}
            for who in observed.values()if who['pid']!=me.pid]
        result['sampled_family_private_peak_bytes']=peak;result['initial_resource_sample']=initial;result['last_resource_sample']=last
        result['external_wall_seconds']=time.monotonic()-began
        result['input_pins_after']=[pin(v['path'])for v in review['input_pins']]
        result['inputs_before_after_exact']=result['input_pins_after']==review['input_pins']
        report=OUTPUT/'report.json'
        if report.exists():
            result['report']=pin(report);result['actual_report_passed']=json.loads(report.read_text(encoding='utf-8'))['passed']
        result['stdout']=pin(stdout)if stdout.exists()else None;result['stderr']=pin(stderr)if stderr.exists()else None
        result['passed']=bool(result['Popen_returncode']==0 and guard_error is None and not cleanup
            and not result.get('not_closed_exact_identities')and result.get('actual_report_passed')is True
            and result['inputs_before_after_exact']and all(v['exact_dead']for v in result['all_observed_owned_descendants_exact_dead']))
        with RESULT.open('x',encoding='utf-8')as file:json.dump(result,file,indent=2);file.write('\n')
    print(json.dumps({'external_result':pin(RESULT),'passed':result['passed'],'Popen_returncode':result['Popen_returncode'],
        'guard_error':guard_error,'cleanup':cleanup,'sampled_family_peak':peak}),flush=True)
    require(result['passed'],'Preserve actual failed diagnostic; no automatic rerun')


if __name__=='__main__':main()
