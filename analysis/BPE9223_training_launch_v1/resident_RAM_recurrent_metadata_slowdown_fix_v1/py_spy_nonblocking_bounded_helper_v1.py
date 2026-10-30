"""Review-gated nonblocking Python stacks only; never mutates a live target.

No suspend/resume, injection, native unwinding, admin fallback, target signaling,
source mutation, GPU operation or environment mutation of another process.
"""
import hashlib, json, math, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
import psutil

P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
S=P/'outputs/parakeet_turkish'
sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory

F=Path(__file__).parent.resolve()
EXE=F/'py_spy_0_4_2_win_amd64_v1/py-spy.exe'
PYTHON=P/'work/nemo-native-env-v2/Scripts/python.exe'
TOY=F/'py_spy_toy_target_v1.py'
B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01')
I=B/'invocations/owner_000022'
LIMIT=536870912
PREFETCH=268435456
TARGET={'pid':40012,'create_time':1792743161.6231525,'ppid':20284}
DRIVER_SHA='eb148001e6851993072f10610e77af16566987b8c0f4ed19d33aee3b1724b3e2'
RUNTIME_SHA='5f3bc165a47810c134d649a10c73726f42d18b9f11d42a79a44e5d7cf3f499a4'
EXE_SHA='6e83ab095f40453742d01a2719a178f308534015da519ec0659d2584139dc995'
RESOURCES={'family_private_cap_bytes':LIMIT,'full_prospective_family_reserve_bytes':LIMIT,
 'full_prefetch_reservation_bytes':PREFETCH,'commit_floor_bytes':2000000000,'physical_floor_bytes':1500000000,
 'E_free_floor_bytes':100000000000,'external_wall_seconds':120,'poll_seconds':.1,
 'profile_byte_cap':67108864,'log_byte_cap_each':4194304,'priority':'BelowNormal'}

def require(v,message):
    if not v:raise ValueError(message)

def pin(p,cap=8*1024**2):
    p=Path(p).resolve();require(p.stat().st_size<=cap,'Bounded source/metadata pin')
    data=p.read_bytes();return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}

def read(p,cap=8*1024**2):
    p=Path(p);require(p.stat().st_size<=cap,'Bounded compact JSON')
    return json.loads(p.read_text(encoding='utf-8'))

def write(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')

def identity(proc):return {'pid':proc.pid,'create_time':proc.create_time(),'ppid':proc.ppid()}

def exact_alive(who):
    try:return abs(psutil.Process(who['pid']).create_time()-who['create_time'])<.001
    except psutil.NoSuchProcess:return False

def command(pid,duration,profile):
    return [str(EXE),'record','--pid',str(pid),'--duration',str(duration),'--rate','100',
        '--nonblocking','--idle','--threads','--full-filenames','--format','speedscope','--output',str(profile)]

def live_health():
    require(exact_alive(TARGET),'Exact admitted native40012 create-time remains alive')
    boot=read(I/'boot.json');progress=read(B/'progress/latest.json')
    require(boot['producer']==TARGET and boot['source']['sha256']==DRIVER_SHA
        and boot['configuration']['sha256']==RUNTIME_SHA,'Current22 exact source/runtime/target')
    require(pin(boot['source']['path'])==boot['source'] and pin(I/'configuration.json')==boot['configuration'],
        'Admitted source30 and runtime configuration immutable')
    require(progress['producer']==TARGET and progress['planned_native_updates']==401072
        and progress['administrative_update_stop'] is None and not progress['optimizer_poisoned']
        and not progress['schedule_poisoned'] and not (I/'actual_failure.json').exists(),
        'Current22 healthy original full-goal frontier')
    return {'target':TARGET,'progress':progress,'boot':pin(I/'boot.json'),'runtime':pin(I/'configuration.json')}

def parse_profile(profile,mode):
    value=read(profile,RESOURCES['profile_byte_cap'])
    require(isinstance(value,dict) and isinstance(value.get('shared',{}).get('frames'),list)
        and isinstance(value.get('profiles'),list) and value['profiles'],'Actual speedscope structure')
    frames=value['shared']['frames'];summaries=[];total=0;toy_seen=False
    for p in value['profiles']:
        require(p.get('type')=='sampled' and p.get('unit')=='seconds','Sampled seconds profile')
        samples=p.get('samples');weights=p.get('weights')
        require(isinstance(samples,list) and isinstance(weights,list) and len(samples)==len(weights),'Exact sample/weight shape')
        selected={};weight_total=0.
        for stack,w in zip(samples,weights):
            require(isinstance(stack,list) and all(type(j)is int and 0<=j<len(frames) for j in stack),'Bounded frame indices')
            require(type(w)in(int,float) and math.isfinite(w) and w>=0,'Finite nonnegative sample weight')
            total+=1;weight_total+=w
            for j in stack:
                fr=frames[j];name=fr.get('name','');file=fr.get('file','')
                if str(file).replace('\\','/').lower()==str(TOY).replace('\\','/').lower() and name.startswith('toy_'):
                    toy_seen=True
            if stack:
                leaf=frames[stack[-1]]
                key=(leaf.get('file'),leaf.get('name'),leaf.get('line'))
                selected[key]=selected.get(key,0.)+w
        summaries.append({'name':p.get('name'),'samples':len(samples),'sample_weight_seconds':weight_total,
            'top_leaf_sites':[{'file':k[0],'function':k[1],'line':k[2],'weight_seconds':v}
                for k,v in sorted(selected.items(),key=lambda kv:kv[1],reverse=True)[:40]]})
    require(total>0,'At least one actual stack sample')
    if mode=='toy':require(toy_seen,'Same-interpreter toy Python frame actually sampled')
    return {'frames':len(frames),'profiles':summaries,'total_samples_all_threads':total,'toy_frame_seen':toy_seen,
        'weights_are_sampled_wall_time_not_CPU_or_exact_component_timers':True,
        'nonblocking_partial_or_inaccurate_reads_possible':True,
        'C_internals_and_gc_vs_chained_native_line_not_fully_resolved':True}

def run(mode,review_path,launcher_source):
    require(mode in('toy','live'),'Known exact mode')
    me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    review=read(review_path);scope='ASR18_py_spy_same_interpreter_toy_compatibility_CPU_v1' if mode=='toy' else 'ASR18_existing_source30_nonblocking_stack_sample_v1'
    out=F/('py_spy_same_interpreter_toy_actual_01' if mode=='toy' else 'py_spy_existing_native40012_actual_01')
    require(review['accepted'] is True and review['reviewer']=='/root' and review['scope']==scope
        and review['mode']==mode and review['helper_source']==pin(__file__)
        and review['launcher_source']==pin(launcher_source) and review['resources']==RESOURCES
        and Path(review['output']).resolve()==out,'Exact separate ROOT scope/source/resources')
    if mode=='toy':require(review['toy_source']==pin(TOY),'Exact toy source pin')
    else:
        require(review['target']==TARGET,'Exact existing target; no generic PID selection')
        toy_accept=read(review['completed_toy_ROOT_review']['path'])
        require(pin(review['completed_toy_ROOT_review']['path'])==review['completed_toy_ROOT_review']
            and toy_accept['accepted'] is True and toy_accept['reviewer']=='/root'
            and toy_accept['scope']=='ASR18_py_spy_same_interpreter_toy_compatibility_actual_CPU_v1'
            and toy_accept['executable']==pin(EXE) and toy_accept['helper_source']==pin(__file__),
            'Genuine completed toy acceptance before any live attach')
    for v in review['input_pins']:require(pin(v['path'])==v,'All exact inputs before any process')
    require(pin(EXE)['sha256']==EXE_SHA,'Official-hash-verified extracted standalone executable')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('PYTHONUTF8')=='1'
        and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
        and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path(r'E:\gpu_training_scratchpad')),
        'Only owned metadata/toy env restricted, admitted parent unchanged')
    require(not out.exists(),'ONE fresh actual output; no automatic retry')
    if mode=='live':health_before=live_health()
    else:health_before=None
    out.mkdir();began=time.monotonic();observed={};popen=None;toy=None;cleanup=[];peak=0;sample_first=None;sample_last=None
    report={'kind':scope,'passed':False,'mode':mode,'ROOT_review':pin(review_path),'external_identity':identity(me),
        'launcher_identity':identity(me.parent()),'input_pins_before':review['input_pins'],'health_before':health_before,
        'resources':RESOURCES,'required_external_cleanup':cleanup,'Popen_returncode':None,'toy_returncode':None,
        'no_existing_native_stop_suspend_resume_injection_or_restart_called':True,'external_guard_error':None}
    profile=out/'profile.speedscope.json';stdout=out/'sampler.stdout.log';stderr=out/'sampler.stderr.log'
    def remember(proc):
        who=identity(proc);observed[(who['pid'],who['create_time'])]=who;return who
    remember(me)
    def observe(admission=False):
        nonlocal peak,sample_first,sample_last
        family={me.pid:me}
        for owned in (toy,popen):
            if owned is None:continue
            try:
                root=psutil.Process(owned.pid)
                for proc in [root,*root.children(recursive=True)]:family[proc.pid]=proc
            except psutil.NoSuchProcess:pass
        total=0
        for proc in family.values():
            try:
                remember(proc);value=sample_memory(proc.pid,'py_spy_owned_metadata_only')
                require(value['paired_sampling_within_one_second'] and value['worker_private_memory']['available'],'Original paired private/system sampling')
                total+=value['worker_private_memory']['private_bytes'];sample_last=value
            except psutil.NoSuchProcess:continue
        require(sample_last is not None,'Resource sample available')
        if sample_first is None:sample_first=sample_last
        peak=max(peak,total);system=sample_last['system_memory'];prospective=LIMIT if admission else 0
        require(total<=LIMIT and time.monotonic()-began<120,'Owned family512MiB/wall120s')
        require(system['commit_available_bytes']>=2000000000+PREFETCH+prospective
            and system['physical_available_bytes']>=1500000000+PREFETCH+prospective
            and shutil.disk_usage('E:/').free>=100000000000,'Original floors plus FULLprefetch/prospective512MiB/E100GB')
        for path,cap in ((profile,RESOURCES['profile_byte_cap']),(stdout,RESOURCES['log_byte_cap_each']),(stderr,RESOURCES['log_byte_cap_each'])):
            require(not path.exists() or path.stat().st_size<=cap,'Bounded sampling output/log')
        if mode=='live':require(exact_alive(TARGET),'Stop only profiler if exact live target exits')
    try:
        observe(admission=True)
        if mode=='toy':
            env=os.environ.copy();report['toy_arguments']=[str(PYTHON),'-Xutf8','-B',str(TOY),'--output',str(out)]
            with (out/'toy.stdout.log').open('xb') as o,(out/'toy.stderr.log').open('xb') as e:
                toy=subprocess.Popen(report['toy_arguments'],cwd=out,env=env,stdout=o,stderr=e,creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            report['toy_identity']=remember(psutil.Process(toy.pid))
            wait_start=time.monotonic()
            while not (out/'toy_ready.json').exists():
                require(toy.poll() is None and time.monotonic()-wait_start<15,'Bounded owned toy readiness')
                observe();time.sleep(.1)
            ready=read(out/'toy_ready.json');require(ready['pid']==toy.pid and Path(ready['interpreter']).resolve()==PYTHON.resolve(),'Genuine same-interpreter ready')
            report['toy_ready']=ready;pid=toy.pid;duration=8
        else:pid=TARGET['pid'];duration=60
        cmd=command(pid,duration,profile)
        if mode=='live':require(cmd==review['exact_sampler_arguments'],'Immutable exact live nonblocking command')
        report['exact_sampler_arguments']=cmd
        with stdout.open('xb') as o,stderr.open('xb') as e:
            popen=subprocess.Popen(cmd,cwd=out,stdout=o,stderr=e,creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            report['sampler_identity']=remember(psutil.Process(popen.pid))
            while popen.poll() is None:
                if toy is not None:require(toy.poll() is None,'Owned toy remains alive while sampled')
                observe();time.sleep(.1)
            report['Popen_returncode']=popen.wait(timeout=5)
        require(report['Popen_returncode']==0,'Actual sampler must exit0; preserve any error without retry')
        if toy is not None:
            write(out/'toy_stop.json',{'reason':'owned finite toy sample completed'})
            toy_began=time.monotonic()
            while toy.poll() is None:
                require(time.monotonic()-toy_began<10,'Bounded cooperative toy closure')
                observe();time.sleep(.1)
            report['toy_returncode']=toy.wait(timeout=5)
            complete=read(out/'toy_completion.json');require(report['toy_returncode']==0 and complete['passed'] is True,'Toy clean thread/module closure')
            report['toy_completion']=pin(out/'toy_completion.json')
        observe()
        report['profile_summary']=parse_profile(profile,mode)
        if mode=='live':report['health_after']=live_health()
    except BaseException as error:
        report['external_guard_error']={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
    finally:
        # Only owned profiler is killed on a live error. The existing native is NEVER in this list.
        for label,owned in (('profiler',popen),('owned_toy',toy)):
            if owned is None:continue
            if owned.poll() is None:
                if label=='owned_toy' and not (out/'toy_stop.json').exists():write(out/'toy_stop.json',{'reason':'owned fixture exception cleanup'})
                try:owned.wait(timeout=3 if label=='owned_toy' else .05)
                except subprocess.TimeoutExpired:
                    try:
                        expected=report['sampler_identity'] if label=='profiler' else report['toy_identity']
                        require(exact_alive(expected),'Cleanup may act only on the exact original owned process')
                        root=psutil.Process(owned.pid)
                        for proc in [*root.children(recursive=True),root]:
                            who=remember(proc);require(who['pid']!=TARGET['pid'],'Existing native can never be cleanup target')
                            proc.kill();cleanup.append({'role':label,'identity':who})
                        owned.wait(timeout=10)
                    except BaseException as error:report.setdefault('cleanup_errors',[]).append(repr(error))
            if label=='profiler':report['Popen_returncode']=owned.returncode
            else:report['toy_returncode']=owned.returncode
        dead=[]
        for who in observed.values():
            if who['pid']==me.pid:continue
            try:
                proc=psutil.Process(who['pid'])
                if abs(proc.create_time()-who['create_time'])<.001:proc.wait(timeout=5)
            except psutil.NoSuchProcess:pass
            except psutil.TimeoutExpired:pass
            dead.append({'identity':who,'exact_dead':not exact_alive(who)})
        report['recorded_identities']=list(observed.values());report['owned_descendant_closure']=dead
        report['family_private_peak_bytes']=peak;report['initial_resource_sample']=sample_first;report['last_resource_sample']=sample_last
        report['external_wall_seconds']=time.monotonic()-began
        try:
            report['input_pins_after']=[pin(v['path']) for v in review['input_pins']]
            report['input_pins_before_after_exact']=report['input_pins_after']==review['input_pins']
        except BaseException as error:
            report['input_pins_after_error']=repr(error);report['input_pins_before_after_exact']=False
        report['outputs']={}
        for p in out.iterdir():
            if p.is_file():
                report['outputs'][p.name]=pin(p,RESOURCES['profile_byte_cap']) if p.stat().st_size<=RESOURCES['profile_byte_cap'] else {
                    'path':str(p),'bytes':p.stat().st_size,'sha256':None,'hash_omitted_due_to_output_byte_bound':True}
        report['passed']=bool(report['Popen_returncode']==0 and report['external_guard_error'] is None
            and not cleanup and not report.get('cleanup_errors') and all(v['exact_dead'] for v in dead)
            and report['input_pins_before_after_exact'] and report.get('profile_summary'))
        write(out/'external_result.json',report)
    print(json.dumps({'result':pin(out/'external_result.json'),'passed':report['passed'],
        'Popen_returncode':report['Popen_returncode'],'error':report['external_guard_error'],'cleanup':cleanup}),flush=True)
    require(report['passed'],'Preserve actual failure; no automatic rerun/fallback')
