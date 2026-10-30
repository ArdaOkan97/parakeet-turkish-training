"""One fixed60-second read-only process metadata sample for current source39 wait attribution."""
import os,sys,json,time,pathlib,hashlib,shutil
import psutil
ROOT=pathlib.Path('E:/gpu_training_scratchpad')
P=pathlib.Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
W=ROOT/'analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1'
B=ROOT/'runs/BPE9223_resident_RAM_20261018_01'
I=B/'invocations/owner_000027'
def write_new(p,v):
 b=json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False).encode()
 with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def require(v,m):
 if not v:raise ValueError(m)
def main():
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and sys.flags.utf8_mode==1,'CPUhidden UTF8')
 require(all(os.environ.get(k)=='1'for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']),'Threadcaps1')
 require(bool(os.environ.get('TORCH_EXTENSIONS_DIR')),'Eenv TORCH_EXTENSIONS_DIR')
 require(shutil.disk_usage(ROOT).free>=100000000000+1048576,'E100GB fullreserve')
 scopepath=pathlib.Path(sys.argv[1]);scopebytes=scopepath.read_bytes();scope=json.loads(scopebytes)
 require(scope['kind']=='ASR18_ROOT_source39_current_waits_fixed60_process_metadata_exact_ONE_execution_scope_v1','Exactscopekind')
 require(scope['source']['path']==str(pathlib.Path(__file__).resolve()),'Exact sourcepath')
 raw=pathlib.Path(__file__).read_bytes()
 require(len(raw)==scope['source']['bytes']and hashlib.sha256(raw).hexdigest()==scope['source']['sha256'],'Exactsourcepin')
 out=W/'source39_current_waits_fixed60_after_metrics_unmounted_actual_v1';out.mkdir(exist_ok=False)
 sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
 from resource_watchdog_v2 import sample_memory
 initial=sample_memory(os.getpid(),'source39_fixed60_metadata_initial')
 initialpin=write_new(out/'initial_numeric_before_predicate.json',initial)
 identities=scope['processes'];samples=[];error=None;progress_before=None;progress_after=None
 selfp=psutil.Process();started=time.monotonic()
 try:
  m=initial['system_memory']
  require(initial['paired_sampling_within_one_second'],'Original pairedsample')
  require(m['commit_available_bytes']>=5130803840 and m['physical_available_bytes']>=2380803840,'Full originalnative+72+metadata512 prospective')
  require(initial['worker_private_memory']['private_bytes']<536870912,'Self512MiB')
  require(not(I/'actual_failure.json').exists(),'Current no productionfailure')
  progress_before=json.loads((B/'progress/latest.json').read_bytes())
  require(progress_before['producer']['pid']==34788 and progress_before['source']['sha256']=='430f67f5a513fc1751e6c36b2e5faca7f9c0828ea3d0c91b5f5d46309e284143','Exact currentproducer source')
  require(progress_before['actual_optimizer_updates']>=206000 and progress_before['last_saved_updates']>=206000,'Changed natural206000 savedstate')
  require(progress_before['planned_native_updates']==401072 and progress_before['administrative_update_stop']is None,'Full target/adminNone')
  def point(index):
   elapsed=time.monotonic()-started
   require(elapsed<180 and selfp.memory_info().private<536870912,'Observer wall/self bounds')
   mem=sample_memory(os.getpid(),'source39_fixed60_metadata_runtime')
   sm=mem['system_memory'];require(sm['commit_available_bytes']>=2612368384 and sm['physical_available_bytes']>=2112368384,'Original runtimefloors+72+fullmetadata512')
   roles={}
   for role,v in identities.items():
    p=psutil.Process(v['pid']);require(abs(p.create_time()-v['create_time'])<.001 and p.ppid()==v['ppid'],'Exact roleidentity '+role)
    ct=p.cpu_times();io=p.io_counters()
    roles[role]={'pid':p.pid,'creation':p.create_time(),'ppid':p.ppid(),'CPU_user_seconds':ct.user,'CPU_system_seconds':ct.system,'IO':io._asdict(),'private_bytes':p.memory_info().private,'threads':p.num_threads()}
   return {'sample_index':index,'monotonic_seconds':time.monotonic(),'elapsed_seconds':time.monotonic()-started,'roles':roles,'system_memory':sm}
  t0=time.monotonic()
  for index in range(61):
   delay=t0+index-time.monotonic()
   if delay>0:time.sleep(delay)
   samples.append(point(index))
  progress_after=json.loads((B/'progress/latest.json').read_bytes())
 except BaseException as e:error={'type':type(e).__name__,'message':str(e)}
 record={'kind':'ASR18_source39_current_waits_fixed60_process_metadata_actual_v1','passed':error is None,'scope':{'path':str(scopepath),'bytes':len(scopebytes),'sha256':hashlib.sha256(scopebytes).hexdigest()},'entry':{'pid':os.getpid(),'create_time':selfp.create_time(),'ppid':selfp.ppid()},'initial':initialpin,'samples':samples,'progress_before':progress_before,'progress_after':progress_after,'error':error,'wall_seconds':time.monotonic()-started,'IO_is_process_operation_bytes_not_physical_disk_evidence':True,'CPU_is_process_total_not_exclusive_function_or_threadCPU':True,'no_attach_stack_frames_model_Torch_audio_GPU_or_control':True,'no_journal_or_DB_read':True,'full401072_and_originalguards_preserved':True}
 pin=write_new(out/'result.json',record);print(json.dumps({'pin':pin,'passed':record['passed'],'entry':record['entry'],'samples':len(samples),'error':error}),flush=True)
 return 0 if error is None else 1
if __name__=='__main__':raise SystemExit(main())
