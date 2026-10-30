"""One reviewed CPU metadata observation with external original resource guards."""
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
FIXTURE=F/'verify_fresh_signature_and_collect_observation_CPU_v1.py'
PYTHON=Path('E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe')
RESULT=F/'fresh_signature_collect_observation_external_result_v1.json'
OUTPUT=F/'fresh_signature_collect_observation_actual_CPU_01'
LIMIT=536870912;PREFETCH=268435456


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
        and review['scope']=='ASR18_fresh_single_open_signature_and_collect_observation_CPU_v1'
        and review['fixture_source']==pin(FIXTURE)and review['launcher_source']==pin(__file__)
        and Path(review['output']).resolve()==OUTPUT.resolve(),'Exact ROOT source/diagnostic binding')
    require(not RESULT.exists()and not OUTPUT.exists(),'ONE fresh actual observation, preserve every prior outcome')
    for value in review['input_pins']:require(pin(value['path'])==value,'All compact/source inputs pinned before launch')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and os.environ.get('PYTHONUTF8')=='1'
        and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
        and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),
        'Original E process-local CPU-only limits')
    began=time.monotonic();popen=None;observed={};cleanup=[];guard_error=None;peak=0;initial=None;last=None
    def remember(proc):
        who=identity(proc);observed[(who['pid'],who['create_time'])]=who;return who
    remember(me)
    def observe(admission=False):
        nonlocal peak,last
        family=[me]
        if popen is not None:
            try:
                root=psutil.Process(popen.pid);family+=[root,*root.children(recursive=True)]
            except psutil.NoSuchProcess:pass
        total=0
        for proc in family:
            try:
                remember(proc);sample=sample_memory(proc.pid,'fresh_signature_collect_external_CPU')
                require(sample['paired_sampling_within_one_second'],'Original paired memory sample')
                total+=sample['worker_private_memory']['private_bytes'];last=sample
            except psutil.NoSuchProcess:continue
        peak=max(peak,total);system=last['system_memory']
        prospective=LIMIT if admission else 0
        require(total<=LIMIT and time.monotonic()-began<240,'Sampled family512MiB/external240s')
        require(system['commit_available_bytes']>=2000000000+PREFETCH+prospective
            and system['physical_available_bytes']>=1500000000+PREFETCH+prospective
            and shutil.disk_usage('E:/').free>=100000000000,'Original floors plus FULLprefetch/prospective512MiB/E100GB')
        return sample
    stdout=F/'fresh_signature_collect_stdout.log';stderr=F/'fresh_signature_collect_stderr.log'
    result={'kind':'ASR18_fresh_signature_collect_observation_external_result_v1','passed':False,
        'external_identity':identity(me),'launcher_identity':identity(me.parent()),'ROOT_review':pin(args.ROOT_review),
        'fixture_source':pin(FIXTURE),'external_source':pin(__file__),'input_pins_before':review['input_pins'],
        'Popen_returncode':None,'external_guard_error':None,'required_external_cleanup':cleanup}
    try:
        initial=observe(admission=True)
        with stdout.open('xb')as out,stderr.open('xb')as err:
            command=[str(PYTHON),'-Xutf8','-B',str(FIXTURE),'--ROOT-review',str(Path(args.ROOT_review).resolve())]
            popen=subprocess.Popen(command,cwd=P,stdout=out,stderr=err,
                creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            remember(psutil.Process(popen.pid))
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
                root=psutil.Process(popen.pid);owned=[*root.children(recursive=True),root]
                for proc in owned:
                    who=remember(proc);proc.kill();cleanup.append(who)
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
