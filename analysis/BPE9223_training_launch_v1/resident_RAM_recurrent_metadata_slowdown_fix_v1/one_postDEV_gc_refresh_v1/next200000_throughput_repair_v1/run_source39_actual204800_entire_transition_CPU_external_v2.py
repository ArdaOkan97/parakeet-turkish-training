"""INACTIVE finite metadata CPU launcher; separate ROOT admission required."""
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

P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
W=Path(__file__).resolve().parent
CAP=536870912
EXTRA=72*1024**2


def require(value,message):
    if not value:raise ValueError(message)


def identity(proc):return dict(pid=proc.pid,create_time=proc.create_time(),ppid=proc.ppid())


def alive(who):
    try:
        proc=psutil.Process(who['pid'])
        return proc.create_time()==who['create_time']and proc.ppid()==who['ppid']and proc.is_running()
    except psutil.NoSuchProcess:return False


def pin(path):
    path=Path(path);require(path.stat().st_size<=8388608,'Bounded compact/source pin')
    data=path.read_bytes();return dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def write(path,value):
    with Path(path).open('x',encoding='utf8',newline='\n')as stream:
        json.dump(value,stream,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return pin(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ROOT-review',required=True);args=parser.parse_args()
    me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    review=json.loads(Path(args.ROOT_review).read_text(encoding='utf-8-sig'))
    require(review['accepted']is True and review['reviewer']=='/root'
        and review['scope']=='ASR18_source39_actual204800_entire_transition_main_path_metadata_CPU_v1'
        and review['external_source']==pin(__file__),'Separate exact finite CPU scope')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and os.environ.get('PYTHONUTF8')=='1'
        and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
        and Path(os.environ['TORCH_EXTENSIONS_DIR'])==Path('E:/gpu_training_scratchpad/cache/torch_extensions'),
        'Dot-sourced E-only environment/caps1/hidden CPU')
    output=Path(review['external_output']);fixture_output=Path(review['fixture_output'])
    require(output.drive.lower()=='e:'and not output.exists()and not fixture_output.exists(),'Fresh exact E namespaces')
    output.mkdir();began=time.monotonic();popen=None;root_identity=None;owned={};cleanup=[];races=[];peak=0;last=None
    parent=identity(me.parent());protected={i['pid']for i in review['protected_runtime_identities']}
    result=dict(kind='ASR18_source38_entire_transition_main_path_metadata_external_v1',passed=False,
        external=identity(me),external_launcher=parent,ROOT_review=pin(args.ROOT_review),input_pins_before=review['input_pins'],
        Popen_returncode=None,guard_error=None,cleanup=cleanup,races=races,TORCH_EXTENSIONS_DIR=os.environ['TORCH_EXTENSIONS_DIR'])
    def remember(proc):
        who=identity(proc);owned[(who['pid'],who['create_time'])]=who;return who
    def observe(admission=False):
        nonlocal peak,last
        require(alive(parent),'Exact external launcher remains sampled/live, never owned cleanup')
        family=[(me,identity(me)),(psutil.Process(parent['pid']),parent)]
        if root_identity is not None and alive(root_identity):
            root=psutil.Process(root_identity['pid'])
            require(identity(root)==root_identity,'Exact initial owned Popen root')
            family.append((root,remember(root)))
            for child in root.children(recursive=True):
                try:who=remember(child)
                except psutil.NoSuchProcess:
                    require(len(races)<32,'Bounded exited-child race records');races.append(dict(pid=child.pid,ctime='UNKNOWN_UNRECORDED',phase='owned_descendant_identity'));continue
                family.append((child,who))
        total=0;members=[];seen=set()
        for proc,who in family:
            if who['pid']in seen:continue
            seen.add(who['pid'])
            try:private=proc.memory_info().private
            except psutil.NoSuchProcess:
                require(proc.pid not in (me.pid,parent['pid'])and(who['pid'],who['create_time'])in owned,'Only already captured owned-child exit race')
                require(len(races)<32,'Bounded exited-child race records');races.append(dict(identity=who,phase='owned_private_memory'));continue
            total+=private;members.append(dict(identity=who,private_bytes=private))
        last=sample_memory(me.pid,'source38_external_full_family')
        peak=max(peak,total);sample=dict(sample=last,family=members,family_private=total,E_free=shutil.disk_usage('E:/').free)
        if admission:result['initial_numeric']=write(output/'initial_numeric_before_predicate.json',sample)
        require(last['paired_sampling_within_one_second']is True and total<=CAP and time.monotonic()-began<120,'Paired sample/fullfamily512MiB/120s')
        require(last['system_memory']['commit_available_bytes']>=(5130803840 if admission else 2343932928)
            and last['system_memory']['physical_available_bytes']>=(2380803840 if admission else 1843932928)
            and sample['E_free']>=100000000000,'FULL225/FULLprefetch/72 additional/prospective512/original runtime floors/E100GB')
        return sample
    try:
        result['initial']=observe(True)
        for item in review['input_pins']:require(pin(item['path'])==item,'Frozen exact metadata/source inputs')
        for who in review['protected_runtime_identities']:require(alive(who)is who['expected_alive'],'Exact protected identities without rebind')
        stdout=output/'stdout.log';stderr=output/'stderr.log'
        with stdout.open('xb')as out,stderr.open('xb')as err:
            command=[str(Path('E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe')),'-X','utf8','-B',review['fixture_source']['path'],'--ROOT-review',str(Path(args.ROOT_review))]
            popen=subprocess.Popen(command,cwd=P,stdout=out,stderr=err,creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            root_identity=remember(psutil.Process(popen.pid));result['Popen_identity']=root_identity;result['command']=command
            while popen.poll()is None:observe();time.sleep(.05)
            result['Popen_returncode']=popen.wait(timeout=10);result['last']=observe()
    except BaseException as error:result['guard_error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        if popen is not None and popen.poll()is None:
            try:
                require(root_identity is not None and alive(root_identity),'Exact original owned root before cleanup')
                root=psutil.Process(root_identity['pid']);targets=[]
                for proc in root.children(recursive=True):targets.append((len(proc.parents()),remember(proc)))
                targets.sort(key=lambda pair:pair[0],reverse=True)
                for who in [w for _,w in targets]+[root_identity]:
                    if not alive(who):continue
                    require(who['pid']not in protected and alive(root_identity),'Never protected PID cleanup; initial root exact')
                    proc=psutil.Process(who['pid'])
                    require(who==root_identity or any(p.pid==root_identity['pid']and p.create_time()==root_identity['create_time']for p in proc.parents()),'Fresh exact owned ancestry')
                    require(alive(who),'Recheck exact identity immediately before owned kill');proc.kill();cleanup.append(who)
                popen.wait(timeout=10)
            except BaseException as error:result['cleanup_error']=repr(error)
        result['owned_identities']=list(owned.values());result['owned_exact_deaths']=[]
        for who in owned.values():
            if alive(who):
                try:psutil.Process(who['pid']).wait(timeout=2)
                except psutil.NoSuchProcess:pass
                except psutil.TimeoutExpired:pass
            result['owned_exact_deaths'].append(dict(identity=who,exact_dead=not alive(who)))
        result['wall_seconds']=time.monotonic()-began;result['family_private_peak']=peak;result['last_sample']=last
        result['input_pins_after']=[pin(item['path'])for item in review['input_pins']]
        for name in ('stdout','stderr'):
            path=output/(name+'.log');result[name]=pin(path)if path.exists()else None
        report=fixture_output/'report.json'
        result['report']=pin(report)if report.exists()else None
        report_passed=json.loads(report.read_text())['passed']if report.exists()else False
        result['passed']=bool(result['Popen_returncode']==0 and result['guard_error']is None and not cleanup and not result.get('cleanup_error')
            and all(i['exact_dead']for i in result['owned_exact_deaths'])and result['input_pins_before']==result['input_pins_after']and report_passed)
        final=write(output/'external_result.json',result)
        print(json.dumps(dict(result=final,passed=result['passed'],Popen_returncode=result['Popen_returncode'],guard_error=result['guard_error'])),flush=True)
    require(result['passed'],'Preserve actual failure, no automatic retry')


if __name__=='__main__':main()
