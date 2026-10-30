import hashlib, json, os, pathlib, shutil, subprocess, sys, time, traceback
import psutil
W=pathlib.Path(__file__).resolve().parent
SCOPE=W/'ROOT_current_stale_Trackio5_exact_ONE_same_source_reader_restart_scope_v1.json'
scope=json.loads(SCOPE.read_text(encoding='utf-8'))
OUT=pathlib.Path(scope['output_folder']);OUT.mkdir(exist_ok=False)
T=pathlib.Path('E:/gpu_training_scratchpad/analysis/BPE9223_Trackio_live_v1')
def require(value,message):
    if not value:raise ValueError(message)
def identity(p):return dict(pid=p.pid,create_time=p.create_time(),ppid=p.ppid())
def pin(path):
    p=pathlib.Path(path);raw=p.read_bytes()
    return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n')as f:
        json.dump(value,f,sort_keys=True,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    return pin(path)
def same_alive(who):
    try:return abs(psutil.Process(who['pid']).create_time()-who['create_time'])<.001
    except psutil.NoSuchProcess:return False
def exact_current(who):
    p=psutil.Process(who['pid'])
    require(abs(p.create_time()-who['create_time'])<.001 and p.ppid()==who['ppid'],'Frozen exactctime/ancestry; no retarget')
    return p
def compact(name):
    path=T/name;raw=path.read_bytes();require(len(raw)<=1048576,'Bounded current compact only')
    data=json.loads(raw)
    keys=('pid','create_time','source_sha256','status','updated_unix','mirrored_update','journal_offset','native_producer',
        'training_worker_alive','last_verified_checkpoint_update','journal_path','error','offset','last_update')
    return dict(pin=dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),
        mtime_ns=path.stat().st_mtime_ns,value={k:data[k]for k in keys if k in data})
me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
entry=dict(identity=identity(me),launcher=identity(me.parent()),production_session_not_touched=76082)
write(OUT/'entry.json',entry)
began=time.monotonic();child=None;result=dict(kind='ASR18_current_stale_Trackio5_one_same_source_restart_actual_v1',
    ROOT_review=pin(SCOPE),entry=entry,old_actions=[],old_exact_deaths=[],new_identities=[],error=None,
    second_launch=False,manual_DB_STATE_journal_edits=False,production_change=False)
try:
    require(result['ROOT_review']['sha256']=='9518a8d2963bfb941093479ecff5a9866bcdcb71940ef7a11af859b80b37a31d','Exact ROOT restart scope')
    sys.path.insert(0,str(pathlib.Path(scope['cwd'])/'outputs/parakeet_turkish'))
    from resource_watchdog_v2 import sample_memory
    def guard(phase,initial=False):
        sample=sample_memory(me.pid,phase)
        value=dict(sample=sample,private_bytes=me.memory_info().private,E_free=shutil.disk_usage('E:/gpu_training_scratchpad').free)
        if initial:result['initial_numeric']=write(OUT/'initial_numeric_before_predicate.json',value)
        require(value['private_bytes']<=536870912 and time.monotonic()-began<180,'Original outerself512MiB/180s')
        require(sample['system_memory']['commit_available_bytes']>=2000000000 and sample['system_memory']['physical_available_bytes']>=1500000000,'Original reader2GBcommit1.5GBphysical floors')
        require(value['E_free']>=100000000000,'Original E100GB')
        return value
    result['initial']=guard('Trackio5_restart_actual_initial',True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'CPUonly/caps1 processlocal')
    require(os.environ.get('TORCH_EXTENSIONS_DIR')=='E:\\gpu_training_scratchpad\\cache\\torch_extensions','Exact dot-sourced storage environment')
    source=pathlib.Path(scope['source']['path']);actual_source=pin(source)
    require(actual_source['sha256']==scope['source']['sha256']and actual_source['bytes']==scope['source']['bytes'],'Same exact v5 source')
    database=pathlib.Path(scope['database']['path']);state=T/'mirror_state.json'
    require(database.is_file()and state.is_file(),'Existing database and STATE required; never bootstrap backup/reseed')
    result['database_STAT_before']=dict(path=str(database),bytes=database.stat().st_size,mtime_ns=database.stat().st_mtime_ns)
    result['before_state']=compact('mirror_state.json');result['before_status']=compact('current_status.json');result['before_owner']=compact('owner.json')
    result['old_identities']=[dict(identity=identity(exact_current(who)),command_line=exact_current(who).cmdline())for who in scope['old']]
    for who in scope['old']:
        if same_alive(who):
            p=exact_current(who)
            result['old_actions'].append(dict(identity=identity(p),action='terminate exact stale reader'if who['role']=='reader'else'terminate exact own launcher only if still alive'))
            p.terminate()
            try:
                observed=p.wait(timeout=10)
                result['old_actions'][-1]['actual_wait_return']=observed
            except psutil.TimeoutExpired:raise ValueError('Exact old process did not exit within bounded wait; no forcekill/retry')
        require(not same_alive(who),'Exact old identity dead after bounded supported terminate')
        result['old_exact_deaths'].append(dict(identity=who,exact_dead=True))
    listeners=[dict(pid=c.pid,laddr=list(c.laddr),status=c.status)for c in psutil.net_connections(kind='tcp')if c.laddr and c.laddr.port==7860 and c.status==psutil.CONN_LISTEN]
    result['listeners_after_old_death']=listeners;require(not listeners,'Port7860 released; no duplicate server or other-process targeting')
    guard('Trackio5_restart_actual_prelaunch')
    require(database.is_file()and state.is_file()and pin(source)==actual_source,'Fresh source/DB/STATE unchanged contract immediately before launch')
    require(compact('mirror_state.json')['pin']==result['before_state']['pin'],'No manual STATE mutation; existing exact checkpoint')
    stdout=OUT/'reader_stdout.log';stderr=OUT/'reader_stderr.log'
    with stdout.open('xb')as out,stderr.open('xb')as err:
        child=subprocess.Popen(scope['argv'],cwd=scope['cwd'],stdin=subprocess.DEVNULL,stdout=out,stderr=err,
            env=dict(os.environ),creationflags=subprocess.CREATE_NO_WINDOW)
    result['new_launcher']=identity(psutil.Process(child.pid));result['new_identities'].append(result['new_launcher'])
    result['argv']=scope['argv'];result['cwd']=scope['cwd']
    print(json.dumps(dict(stage='single_new_reader_launch',new_launcher=result['new_launcher']),sort_keys=True),flush=True)
    previous_body=None
    while time.monotonic()-began<165:
        guard('Trackio5_restart_wait_first64_state_advance')
        require(child.poll()is None,'New reader launcher exited; preserve actual failure/no retry')
        owner=json.loads((T/'owner.json').read_text(encoding='utf-8'))
        if owner.get('pid')not in [who['pid']for who in scope['old']]:
            body=psutil.Process(owner['pid'])
            require(abs(body.create_time()-owner['create_time'])<.001 and body.ppid()==child.pid,'New exact sourceowner belongs to single owned launcher')
            require(owner['source_sha256']==actual_source['sha256'],'New reader same pinned source')
            current=identity(body)
            if previous_body is None:result['new_identities'].append(current);previous_body=current
            require(current==previous_body,'Newbody identity stable, no rebind')
            require(body.memory_info().private<=1500000000,'Original reader selfprivate1.5GB')
            result['new_owner']=compact('owner.json')
        observed_state=compact('mirror_state.json')
        value=observed_state['value']
        if value.get('journal_path')==str(pathlib.Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01/journals/commits_000027.jsonl'))and value.get('offset',0)>0:
            require(previous_body is not None,'Genuine exact newreader owner before state advance')
            result['first_observed_successful_batch_state']=observed_state
            result['status_after_state_advance']=compact('current_status.json')
            result['passed']=True;break
        time.sleep(.5)
    require(result.get('passed')is True,'No observed first successful J27batch by bounded deadline; no second launch')
    result['DB_backup_or_history_reseed_performed']=False
except BaseException as exc:
    result['passed']=False;result['error']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
finally:
    result['wall_seconds']=time.monotonic()-began
    result['Popen_returncode']=None if child is None else child.poll()
    for name in ('reader_stdout.log','reader_stderr.log'):
        path=OUT/name
        if path.exists():result[name]=pin(path)
    result['known_old_exact_dead_at_write']=[dict(identity=who,exact_dead=not same_alive(who))for who in scope['old']]
    result['new_processes_intentionally_left_running_no_automatic_cleanup_or_retry']=child is not None and child.poll()is None
    result_pin=write(OUT/'result.json',result)
    print(json.dumps(dict(result=result_pin,passed=result.get('passed'),new_identities=result['new_identities'],error=result['error']),sort_keys=True),flush=True)
require(result.get('passed'),'Preserve actual reader restart result; no automatic second launch')
