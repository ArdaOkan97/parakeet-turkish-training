"""Metadata only: one already-created profile, no attachment or target operation."""
import hashlib,json,math,os,shutil,sys,time
from collections import defaultdict
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).parent;O=F/'py_spy_existing_native40012_actual_01'
NATIVE='asr18_bpe9223_grouped_ram_native_v1.py';PREFETCH='asr18_bpe9223_private_scalar_prefetch_v1.py'
began=time.monotonic()
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
    with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def guard(admission=False):
    v=sample_memory(os.getpid(),'py_spy_existing_profile_metadata_aggregate');m=v['system_memory']
    assert v['paired_sampling_within_one_second'] and v['worker_private_memory']['private_bytes']<=536870912
    reserve=536870912 if admission else 0
    assert m['commit_available_bytes']>=2000000000+268435456+reserve
    assert m['physical_available_bytes']>=1500000000+268435456+reserve
    assert shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-began<120
    return v
def dead(w):
    try:return abs(psutil.Process(w['pid']).create_time()-w['create_time'])>=.001
    except psutil.NoSuchProcess:return True
assert os.environ['CUDA_VISIBLE_DEVICES']=='-1' and os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
initial=guard(True)
external=json.loads((O/'external_result.json').read_text(encoding='utf-8'))
tool=json.loads((O/'actual_TOOL.json').read_text(encoding='utf-8'))
assert external['passed']is True and external['Popen_returncode']==0 and tool['exit_code']==0
assert external['external_guard_error']is None and not external['required_external_cleanup'] and 'session_id'not in tool
observed={}
for w in external['recorded_identities']+[external['external_identity'],external['launcher_identity'],external['sampler_identity']]:
    assert w['pid']!=40012;observed[(w['pid'],w['create_time'])]=w
closure=[{'identity':w,'exact_dead':dead(w)}for w in observed.values()]
assert all(v['exact_dead']for v in closure)
profile=O/'profile.speedscope.json';assert profile.stat().st_size<=67108864
raw=profile.read_bytes();actual={'path':str(profile.resolve()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
assert actual==external['outputs']['profile.speedscope.json']
data=json.loads(raw);del raw
frames=data['shared']['frames']
def key(frame):return (frame.get('file'),frame.get('name'),frame.get('line'))
def rows(aggregate,total):
    return [{'file':k[0],'function':k[1],'line':k[2],'sample_count':v[0],
        'weight_seconds':v[1],'percent_of_THIS_thread_sample_weight':100*v[1]/total if total else None}
        for k,v in sorted(aggregate.items(),key=lambda kv:kv[1][1],reverse=True)]
def norm(frame):return str(frame.get('file','')).replace('\\','/').lower()
def category(stack):
    fs=[frames[i]for i in stack]
    if any('/torch/cuda/'in norm(f)and f.get('name')=='synchronize'for f in fs):return 'torch_cuda_synchronize_Python_wrapper'
    if any('/torch/cuda/'in norm(f)and f.get('name')=='empty_cache'for f in fs):return 'torch_cuda_empty_cache_Python_wrapper'
    if any(f.get('name')=='assert_model'for f in fs):return 'assert_model_Python_path'
    if any(norm(f).endswith(('resource_watchdog_v2.py','paired_windows_commit_telemetry_v1.py'))for f in fs):return 'resource_sample_Python_path'
    if any(f.get('name')in('get_rng_state','get_rng_state_all','getstate')for f in fs):return 'RNG_state_Python_wrapper'
    if any(norm(f).endswith(NATIVE)and f.get('line')==214 for f in fs):return 'native_line214_collect_sync_cache_snapshot_chain_UNRESOLVED'
    if any(norm(f).endswith(NATIVE)and f.get('line')==233 for f in fs):return 'native_line233_optimizer_zeroGC_sync_cache_chain_UNRESOLVED'
    if any(norm(f).endswith(NATIVE)and type(f.get('line'))is int and 212<=f['line']<=226 for f in fs):return 'other_group_cleanup_region_Python_path'
    if any(norm(f).endswith(PREFETCH)for f in fs):return 'scalar_prefetch_Python_path'
    return 'other_Python_path'
threads=[];main=[]
for p in data['profiles']:
    assert p['type']=='sampled' and p['unit']=='seconds' and len(p['samples'])==len(p['weights'])
    leaf=defaultdict(lambda:[0,0.]);inclusive=defaultdict(lambda:[0,0.]);groups=defaultdict(lambda:[0,0.]);total=0.
    for index,(stack,w)in enumerate(zip(p['samples'],p['weights'])):
        assert type(w)in(int,float)and math.isfinite(w)and w>=0
        assert all(type(i)is int and 0<=i<len(frames)for i in stack)
        total+=w
        if stack:
            k=key(frames[stack[-1]]);leaf[k][0]+=1;leaf[k][1]+=w
        for k in {key(frames[i])for i in stack}:inclusive[k][0]+=1;inclusive[k][1]+=w
        k=category(stack);groups[k][0]+=1;groups[k][1]+=w
        if index%1000==0:guard()
    r={'name':p['name'],'samples':len(p['samples']),'sample_weight_seconds':total,
        'all_leaf_sites':rows(leaf,total),'all_inclusive_sites':rows(inclusive,total),
        'exclusive_stack_categories':[{'category':k,'sample_count':v[0],'weight_seconds':v[1],
            'percent_of_THIS_thread_sample_weight':100*v[1]/total if total else None}
            for k,v in sorted(groups.items(),key=lambda kv:kv[1][1],reverse=True)],
        'scalar_prefetch_inclusive_sites':rows({k:v for k,v in inclusive.items()if str(k[0]).replace('\\','/').lower().endswith(PREFETCH)},total)}
    threads.append(r)
    if '"MainThread"'in p['name']:main.append(r)
assert len(main)==1,'Exact one named MainThread, no guessed thread substitution'
report={'kind':'ASR18_one_actual_nonblocking60s_Python_profile_evidence_v1','passed':True,
 'external_result':pin(O/'external_result.json'),'actual_TOOL':pin(O/'actual_TOOL.json'),'profile':actual,
 'ROOT_scope':external['ROOT_review'],'main_thread':main[0],'all_thread_profiles':threads,
 'health_before':external['health_before'],'health_after':external['health_after'],
 'sampler_stdout':external['outputs']['sampler.stdout.log'],'sampler_stderr':external['outputs']['sampler.stderr.log'],
 'method':'Each leaf gets its sample weight once; inclusive each distinct(file,function,line)once per stack, recursive duplicates deduplicated. Categories are priority-exclusive observed Python paths. ALL frames/sites kept; no selection of favorable interval.',
 'limitations':['Thread wall sample weights are NOT CPU times, additive across concurrent threads, or exact component timers.',
  'Inclusive sites overlap and must NOT be added. Nonblocking non-atomic reads may omit/inaccurately read frames; preserve raw stderr/stdout sampling errors.',
  'Native1 line214 chains builtin gc.collect, CUDA synchronize, emptycache, snapshot. Without a visible Python wrapper, that line stays unresolved; no pureGC or GPU-kernel claim.',
  'Native1 line233 chains optimizer/zero_grad/gc/sync/emptycache; likewise unresolved unless a deeper Python wrapper is present.',
  'Sampling observation does not select a production remedy or prove future speed/fit.'],
 'initial_resource_sample':initial,'last_resource_sample':guard(),'aggregation_source':pin(__file__)}
write(O/'actual_profile_attribution.json',report)
completion={'kind':'ASR18_py_spy_existing_native_actual_metadata_completion_v1','passed':True,
 'actual_outer_TOOL_exit':tool['exit_code'],'actual_outer_TOOL_chunk':tool['chunk_id'],'actual_sampler_Popen_returncode':external['Popen_returncode'],
 'actual_TOOL':pin(O/'actual_TOOL.json'),'external_result':pin(O/'external_result.json'),'attribution':pin(O/'actual_profile_attribution.json'),
 'profile':actual,'ROOT_scope':external['ROOT_review'],'all_exact_metadata_identities_dead':True,
 'recorded_owned_identity_closure':closure,'input_pins_before_after_exact':external['input_pins_before_after_exact'],
 'external_guard_error':None,'required_external_cleanup':[],
 'target_was_never_stopped_suspended_injected_or_restarted':external['no_existing_native_stop_suspend_resume_injection_or_restart_called'],
 'target_health_before':external['health_before']['progress'],'target_health_after':external['health_after']['progress'],
 'completion_source':pin(__file__)}
write(O/'actual_metadata_completion.json',completion)
print(json.dumps({'completion':pin(O/'actual_metadata_completion.json'),'attribution':completion['attribution'],
 'profile':actual,'main_samples':main[0]['samples'],'main_sample_weight_seconds':main[0]['sample_weight_seconds'],
 'main_exclusive_categories':main[0]['exclusive_stack_categories'],'identity_closure':closure}))
