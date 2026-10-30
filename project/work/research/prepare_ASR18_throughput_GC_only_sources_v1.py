"""Small NEW observation/GC-only arm over the closed baseline harness."""
import ast,hashlib,json,copy
from pathlib import Path
P=Path(__file__).resolve().parents[2];R=P/'work/research';A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def read(p):return json.loads(Path(p).read_text())
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
    assert not p.exists();p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n');return pin(p)
source=R/'run_ASR18_native_throughput_component_v1.py';target=R/'run_ASR18_native_throughput_component_v2.py'
t=source.read_text();t=t.replace('import argparse,copy,ctypes,gc,importlib.util,json,os,shutil,subprocess,sys,time',
    'import argparse,copy,ctypes,gc,importlib.util,json,os,shutil,subprocess,sys,time\nfrom contextlib import contextmanager')
helpers='''class NoOpProfiler:
    """Matched final speed observation: no phase hooks/events/gradient readbacks."""
    def __init__(self,model,optimizer,*,torch,folder,guard):
        self.folder=Path(folder);self.guard=guard;self.updates=[]
    def install(self):return {'enabled':False,'phase_hooks_or_CUDA_events_or_gradient_readbacks':False}
    def checked_guard(self):return self.guard()
    def feature(self,function,index):return function(index)
    def stage(self,original,name,**extra):return original(name,**extra)
    def run_update(self,operation,*,update):
        require(1<=update<=4 and len(self.updates)<4,'Same exact bounded four-update speed check')
        began=time.monotonic();result=operation()
        self.updates.append({'update':update,'native_wall_seconds':time.monotonic()-began});return result
    def close(self):
        write_json(self.folder/'profile_summary.json',{'kind':'ASR18_matched_profiler_off_native_wall_observation_v1',
            'source':record(__file__),'update_wall_seconds':self.updates,'profile_enabled':False,'production_enabled':False})
        return record(self.folder/'profile_summary.json')


@contextmanager
def resident_gc_scope(folder,*,enabled,optimizer,torch,guard):
    if not enabled:
        yield;return
    from asr18_live_state_fingerprint_v1 import live_state_fingerprint,require_exact_live_restore
    require(gc.get_freeze_count()==0,'Fresh worker with no previously frozen generation')
    before=live_state_fingerprint(optimizer,torch=torch,guard=guard)
    pointers=[(id(p),p.data_ptr(),p._version,p.grad) for p in optimizer.parameters.values()]
    began=time.monotonic();collected=gc.collect();gc.freeze();frozen=True
    try:
        count=gc.get_freeze_count();require(count>0,'Actual quiescent resident graph frozen once')
        require_exact_live_restore(before,live_state_fingerprint(optimizer,torch=torch,guard=guard))
        require(pointers==[(id(p),p.data_ptr(),p._version,p.grad) for p in optimizer.parameters.values()],
            'Freeze retains all207 original Parameter identities/pointers/versions/no gradients')
        write_json(folder/'actual_resident_GC_freeze.json',{'source':record(__file__),'frozen_objects':count,
            'initial_full_collect_objects':collected,'setup_wall_seconds':time.monotonic()-began,
            'all207_threeAdam_all4RNG_before_after_freeze_exact':True,
            'per_row_full_gc_collect_synchronize_empty_cache_and_invariants_unchanged':True,'host_credit_bytes':0})
        yield
    finally:
        if frozen:
            before_unfreeze=None
            try:
                if not optimizer.poisoned:
                    before_unfreeze=live_state_fingerprint(optimizer,torch=torch,guard=guard)
                    pointers=[(id(p),p.data_ptr(),p._version,p.grad) for p in optimizer.parameters.values()]
            finally:
                # Also unfreeze after a failed/poisoned native operation. Such
                # a path never claims scientific-state or overall success.
                began=time.monotonic();gc.unfreeze();objects=gc.collect()
            require(gc.get_freeze_count()==0,'Actual resident generation unfreezes before model/mmap teardown')
            if before_unfreeze is not None:
                require_exact_live_restore(before_unfreeze,live_state_fingerprint(optimizer,torch=torch,guard=guard))
                require(pointers==[(id(p),p.data_ptr(),p._version,p.grad) for p in optimizer.parameters.values()],
                    'Unfreeze/fullcollect keeps actual current207 Parameter identities/pointers/versions/gradients')
            write_json(folder/'actual_resident_GC_unfreeze.json',{'source':record(__file__),
                'full_collect_objects':objects,'teardown_wall_seconds':time.monotonic()-began,
                'all207_threeAdam_all4RNG_before_after_unfreeze_exact':before_unfreeze is not None,'final_frozen_objects':0})


'''
t=t.replace('def checked(config,review,action):',helpers+'def checked(config,review,action):')
t=t.replace("'Exact original first4 finite profile, no long pipeline rebuild')", "'Exact original first4 finite profile, no long pipeline rebuild')\n    require(c['optimization'] in ('baseline','gc_freeze_only','frozen_cache') and type(c['profile_enabled']) is bool,\n        'One explicit causal variant and matched observation mode, no implicit combination')")
start=t.index('        from asr18_frozen_encoder_cuda_cache_v2 import install,assert_idle')
end=t.index("    write_json(folder/'initial_state.json',initial_value)",start)
block=t[start:end];t=t[:start]+"        if c['optimization']=='frozen_cache':\n"+''.join('    '+line if line.strip() else line for line in block.splitlines(True))+t[end:]
t=t.replace("if arm=='candidate':assert_idle(model,torch=torch,guard=guard)","if arm=='candidate' and c['optimization']=='frozen_cache':assert_idle(model,torch=torch,guard=guard)")
t=t.replace('profiler=NativeProfiler(model,optimizer,torch=torch,folder=folder,guard=guard);profile_policy=profiler.install()',
    "Profiler=NativeProfiler if c['profile_enabled'] else NoOpProfiler\n    profiler=Profiler(model,optimizer,torch=torch,folder=folder,guard=guard);profile_policy=profiler.install()")
start=t.index("    Profiler=NativeProfiler if c['profile_enabled'] else NoOpProfiler")
end=t.index("    require(coordinator.index==optimizer.updates==4",start)
block=t[start:end]
t=t[:start]+"    with resident_gc_scope(folder,enabled=c['optimization']=='gc_freeze_only',optimizer=optimizer,torch=torch,guard=guard):\n"+''.join('    '+line if line.strip() else line for line in block.splitlines(True))+t[end:]
t=t.replace("'initial_state':record(folder/'initial_state.json'),'profile':profile", "'optimization':c['optimization'],'profile_enabled':c['profile_enabled'],\n        'initial_state':record(folder/'initial_state.json'),'profile':profile")
ast.parse(t);assert not target.exists();target.write_text(t,encoding='utf-8',newline='\n')
old_m=read(A/'native_throughput_first4_source_manifest_v1.json');m=copy.deepcopy(old_m)
items={x['path']:x for x in m['bindings']};items[str(target)]=pin(target)
m['bindings']=sorted(items.values(),key=lambda v:v['path']);m['component_sources']['harness']=pin(target)
m['resource_policy'].update(candidate_added_entire_persistent_bank_charged_without_host_credit_bytes=0,
    candidate_worker_private_forecast_bytes=6780646272,candidate_full_family_bytes=6972646272,
    candidate_optimization='gc_freeze_only',host_or_CUDA_savings_assumed=False)
mp=write(A/'native_throughput_GC_only_source_manifest_v1.json',m)
baseline=A/'native_throughput_baseline_actual_01';c=read(A/'native_throughput_first4_configuration_01/configuration_pending_execution_review.json')
c.update(source_manifest=mp,harness_source=pin(target),resource_policy=m['resource_policy'],
    candidate_baseline_initial=pin(baseline/'train/initial_state.json'),candidate_shared_features=pin(baseline/'shared_features.json'),
    optimization='gc_freeze_only',profile_enabled=True,original_baseline_overall=pin(baseline/'overall_result.json'),
    actual_baseline_summary=pin(baseline/'actual_baseline_profile_summary_v1.json'))
out=A/'native_throughput_GC_only_configuration_01';out.mkdir(exist_ok=False)
cp=write(out/'configuration_pending_execution_review.json',c)
proposal={'kind':'ASR18_paired_resident_GC_freeze_only_component_source_delta_v1','old_harness':pin(source),'new_harness':pin(target),
    'source_manifest':mp,'configuration':cp,'baseline_summary':c['actual_baseline_summary'],'baseline_overall_result':c['original_baseline_overall'],
    'candidate_baseline_initial':c['candidate_baseline_initial'],'candidate_shared_features':c['candidate_shared_features'],
    'actual_baseline_TOOL0_chunk':'497adc','resource_policy':m['resource_policy'],'new_CPU_preparation_allowed':False,
    'actual_output':str(A/'native_throughput_GC_only_actual_01'),'source_only':True,'execution_admitted':False,
    'one_change':'Quiescent fullcollect+gc.freeze once before first forward/profiler; keep all original perrow GC/sync/emptycache/RNG/pointer checks; exact finally unfreeze/fullcollect before teardown',
    'profile_off_noop_adapter_inactive_for_this_arm':True,'freeze_setup_teardown_fingerprint_checks_separate_from_native_time':True,
    'candidate_cache_not_enabled_in_this_arm':True,'no_host_credit_or_guard_relaxation':True,'production_enabled':False}
pp=write(A/'native_throughput_GC_only_source_delta_v1.json',proposal)
print(json.dumps({'harness':pin(target),'source_manifest':mp,'configuration':cp,'delta':pp}))
