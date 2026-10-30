"""Fresh remaining family allowance; actual current family is already global.

No native memory credit, cap reduction or process/tool closure is implied.
The pure validator is also used by explicitly synthetic metadata controls.
"""
import time
from pathlib import Path
from atomic_io_retry_v1 import read_json
from bpe_broad_production_state_v1 import require,digest
from resource_watchdog_v2 import sample_memory

def ident(value):
    require(type(value) is dict and set(value)=={'pid','ppid','create_time'} and type(value['pid']) is int
        and value['pid']>0 and type(value['ppid']) is int and value['ppid']>0 and type(value['create_time']) is float
        and value['create_time']>0,'Exact positive PID/ctime identity, no bool')
    return (value['pid'],value['create_time'])

def validate_context(value,*,source,configuration,nonce,native,now,live,descendants):
    require(value['kind']=='ASR18_actual_owned_GPU_family_context_v1' and value['source']==source
        and value['configuration']==configuration and value['nonce']==nonce and type(nonce) is str and len(nonce)==32,
        'Exact live source/config/request nonce')
    require(type(value['observed_monotonic']) in (int,float) and 0<=now-value['observed_monotonic']<=2,'Fresh family context<=2s')
    observer=value['observer'];launcher=value['launcher'];saved=value['native'];shims=value['all_live_shims']
    require(ident(saved)==ident(native) and digest(saved)==digest(native),'Exact expected owned native identity')
    require(ident(observer)!=ident(saved) and ident(launcher)!=ident(saved) and launcher['ppid']==observer['pid']
        and native['ppid'] in {observer['pid'],launcher['pid']},'Actual observer/launcher/native lineage')
    require(type(shims) is list and shims and len({ident(v) for v in shims})==len(shims)
        and ident(launcher) in {ident(v) for v in shims} and ident(native) not in {ident(v) for v in shims},'Every shim exactly once, never native')
    require(all(live(v) for v in [observer,native,*shims]),'Every exact source-owned identity still live')
    require({ident(v) for v in descendants(observer)}=={ident(v) for v in [native,*shims]},'No omitted/new/unowned descendant')
    return observer,shims

def remaining(parent_private,shim_privates):
    require(type(parent_private) is int and 0<parent_private<=128_000_000 and type(shim_privates) is list
        and shim_privates and all(type(v) is int and v>0 for v in shim_privates) and sum(shim_privates)<=64_000_000,
        'Unchanged128MB parent/64MB ALL-shim hard caps; exact available private')
    return (128_000_000-parent_private)+(64_000_000-sum(shim_privates))

def admit_remaining(folder,*,configuration,source,nonce,guard,identity,alive,worker_increment):
    import psutil
    require(type(worker_increment) is int and worker_increment>=0,'Nonnegative full remaining worker increment')
    path=Path(folder)/'owned_family_context.json';checks=[];native=identity()
    def descendants(observer):
        values=[]
        for p in psutil.Process(observer['pid']).children(recursive=True):
            try:values.append(identity(p.pid))
            except psutil.NoSuchProcess:pass
        return values
    for _ in range(3):
        guard();started=time.monotonic();context=read_json(path)
        observer,shims=validate_context(context,source=source,configuration=configuration,nonce=nonce,native=native,
            now=started,live=alive,descendants=descendants)
        samples=[]
        for v in [observer,*shims]:
            require(alive(v),'Exact family identity before sample');sample=sample_memory(v['pid'],'actual_remaining_owned_family')
            require(sample['paired_sampling_within_one_second'] and sample['worker_private_memory']['available']
                and sample['worker_private_memory']['pid']==v['pid'] and alive(v),'Available fresh PID/ctime-bound private')
            samples.append({'identity':v,'sample':sample})
        parent=samples[0]['sample']['worker_private_memory']['private_bytes'];shim=[v['sample']['worker_private_memory']['private_bytes'] for v in samples[1:]]
        unused=remaining(parent,shim);own=sample_memory(native['pid'],'native_global_after_family_sample');s=own['system_memory'];w=own['worker_private_memory']
        require(time.monotonic()-started<=1 and own['paired_sampling_within_one_second'] and s['available'] and w['available'] and w['pid']==native['pid'],'Fresh paired global/current native after family observations')
        validate_context(context,source=source,configuration=configuration,nonce=nonce,native=native,now=time.monotonic(),live=alive,descendants=descendants)
        addition=worker_increment+unused
        passed=s['commit_used_bytes']+addition<.90*s['commit_limit_bytes'] and s['commit_available_bytes']-addition>2_000_000_000
        checks.append({'context':context,'family_samples':samples,'actual_current_parent_private_bytes':parent,'actual_current_ALL_shim_private_bytes':sum(shim),
            'remaining_family_allowance_bytes':unused,'worker_increment_bytes':worker_increment,'global_increment_bytes':addition,'sample':own,'passed':passed})
        time.sleep(.05)
    return {'passed':all(v['passed'] for v in checks),'samples':checks,'parent_cap_bytes':128_000_000,'ALL_shim_cap_bytes':64_000_000,
        'current_family_ALREADY_in_global_not_subtracted_from_system':True,'native_counted_only_in_worker_remaining_increment':True}
