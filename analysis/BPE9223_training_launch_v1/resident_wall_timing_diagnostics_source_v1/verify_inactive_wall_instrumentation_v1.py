"""Finite CPU-only AST and real timing-helper semantics proof; never imports training."""
import ast,copy,hashlib,json,os,random,shutil,sys,time
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
N=Path(__file__).parent;OUT=N/'CPU_instrumentation_actual_01'
sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
from asr18_resident_wall_timings_v1 import WallTimings,wall_time,wall_call,NAMES
def require(value,message):
 if not value:raise ValueError(message)
def pin(path):
 path=Path(path);data=path.read_bytes();return {'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def write(path,value):
 with Path(path).open('x',encoding='utf-8',newline='')as f:json.dump(value,f,indent=2);f.write('\n')
require(not OUT.exists(),'Fresh owned E CPU proof');OUT.mkdir()
require(sys.flags.utf8_mode==1 and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1','Upfront UTF8 and CPU mask')
for key in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):require(os.environ.get(key)=='1','Four thread caps1')
self=psutil.Process();self.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
who={'pid':self.pid,'create_time':self.create_time(),'ppid':self.ppid()};began=time.monotonic();peak=0;samples=[]
def guard(admission=0):
 global peak
 sample=sample_memory(os.getpid(),'inactive_wall_timer_CPU_proof');m=sample['system_memory'];w=sample['worker_private_memory']
 require(sample['paired_sampling_within_one_second'] and m['available'] and w['available'],'Actual paired resource telemetry')
 peak=max(peak,w['private_bytes']);require(peak<=512*1024**2 and time.monotonic()-began<180,'512MiB/180s finite CPU proof')
 require(m['commit_available_bytes']-admission>=2_000_000_000 and m['physical_available_bytes']-admission>=1_500_000_000,'Full remaining CPU admission and unchanged real headroom')
 require(shutil.disk_usage('E:/').free>=100_000_000_000+1024**2,'E100GB plus bounded proof output')
 samples.append(sample);return sample
try:
 guard(512*1024**2)
 # Compile only the reviewable AST eraser definitions, never the source generator.
 generator=ast.parse((N/'derive_inactive_timed_successor_v2.py').read_text())
 definitions=[copy.deepcopy(n)for n in generator.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in('code','Erase')]
 scope={'ast':ast};exec(compile(ast.fix_missing_locations(ast.Module(body=definitions,type_ignores=[])),str(N/'derive_inactive_timed_successor_v2.py'),'exec'),scope)
 source_pairs=[('native',S/'asr18_BPE9223_grouped_RAM_native_v1.py',S/'asr18_BPE9223_grouped_RAM_native_v2.py'),('driver',R/'run_ASR18_BPE9223_resident_RAM_training_v12.py',R/'run_ASR18_BPE9223_resident_RAM_training_v14.py')]
 ast_results={};calls={}
 for label,old,new in source_pairs:
  before=ast.parse(old.read_text());after=ast.parse(new.read_text());erased=scope['Erase']().visit(copy.deepcopy(after));ast.fix_missing_locations(erased)
  require(ast.dump(before,include_attributes=False)==ast.dump(erased,include_attributes=False),'Exact full original executable AST after declared timing erasure')
  ast_results[label+'_erased_AST_equals_original']=True
  ast_results[label+'_original_AST_sha256']=hashlib.sha256(ast.dump(before,include_attributes=False).encode()).hexdigest()
  def selected_calls(tree):
   return [ast.unparse(n.func)for n in ast.walk(tree)if isinstance(n,ast.Call)and (ast.unparse(n.func).startswith(('torch.','optimizer.','np.random.','random.'))or ast.unparse(n.func)in('gc.collect','journal.append'))]
  require(sorted(selected_calls(before))==sorted(selected_calls(after)),'Existing CUDA/cache/optimizer/RNG/journal call counts unchanged; exact erased AST establishes operation order')
  calls[label]=selected_calls(before)
  guard()
 # Execute the real helper with nested CPU operations, each possible exceptional boundary,
 # return-object identity, and diagnostic-clock failures. No model/audio/GPU imports.
 tests=[];rng=random.getstate()
 class Sentinel(BaseException):pass
 result_object=object()
 def run(instrumented,fail_at=None,clock=None):
  log=[];timings=WallTimings()if clock is None else WallTimings(clock=clock);error=Sentinel('original operation')
  def operation(name):
   log.append(name)
   if name==fail_at:raise error
   return result_object
  def invoke(name,channel):
   if instrumented:return wall_call(timings,channel,operation,name)
   return operation(name)
  try:
   try:
    invoke('packet','packet_wait')
    if instrumented:
     with wall_time(timings,'native_update_total'):
      invoke('forward','native_encoder_and_stem');invoke('backward','native_backward_enqueue')
      invoke('optimizer','native_optimizer_step')
    else:
     invoke('forward','native_encoder_and_stem');invoke('backward','native_backward_enqueue');invoke('optimizer','native_optimizer_step')
    result=invoke('journal','transaction_journal_append_total')
   finally:invoke('cleanup','native_final_gc')
  except BaseException as caught:return log,caught is error,None,timings.snapshot()
  return log,None,result is result_object,timings.snapshot()
 for failure in(None,'packet','forward','backward','optimizer','journal','cleanup'):
  old=run(False,failure);new=run(True,failure);require(old[:3]==new[:3],'Original CPU operation order/result/exception identity')
  tests.append({'failure_at':failure,'same_call_order_and_result_or_exception_identity':True,'observed_calls':new[0]})
 class BrokenClock:
  def __call__(self):raise RuntimeError('injected diagnostic clock failure')
 for failure in(None,'forward','journal','cleanup'):
  old=run(False,failure);new=run(True,failure,BrokenClock());require(old[:3]==new[:3] and new[3]['diagnostic_errors']>0,'Diagnostic failure cannot replace operation result or original exception')
 tests.append({'injected_clock_failures_keep_original_behavior':True})
 vals=iter((3.,2.));t=WallTimings(clock=lambda:next(vals));require(wall_call(t,'packet_wait',lambda:result_object)is result_object and t.errors==1,'Negative monotonic interval recorded without changing result')
 t=WallTimings();require(wall_call(t,'unknown_channel',lambda:result_object)is result_object and t.errors==1,'Unknown diagnostic label does not affect operation')
 require(wall_call(None,'packet_wait',lambda:result_object)is result_object,'Optional disabled timing preserves result')
 require(random.getstate()==rng and 'torch'not in sys.modules and 'numpy'not in sys.modules,'No RNG advance or model/array runtime import')
 guard()
 # Real owned E output operations in original write/flush/fsync order, same bytes.
 payload=b'bounded CPU timing operation order proof\n';file_reports=[]
 for timed in(False,True):
  path=OUT/('timed.bin'if timed else'original.bin');order=[];t=WallTimings()
  with path.open('xb')as f:
   for name,call in [('write',lambda:f.write(payload)),('flush',f.flush),('fsync',lambda:os.fsync(f.fileno()))]:
    order.append(name)
    if timed:wall_call(t,'transaction_journal_append_total',call)
    else:call()
  require(f.closed and order==['write','flush','fsync'],'Real file handle closed and unchanged operation order')
  file_reports.append({'file':pin(path),'call_order':order,'handle_closed':f.closed})
 require(file_reports[0]['file']['sha256']==file_reports[1]['file']['sha256'],'Real output bytes exact')
 record_bytes=len(json.dumps(WallTimings().snapshot(),separators=(',',':')).encode())
 require(len(NAMES)==33 and len(set(NAMES))==33 and record_bytes<65536,'Fixed bounded scalar window record')
 guard()
 report={'kind':'ASR18_actual_CPU_existing_operation_wall_instrumentation_proof_v1','passed':True,'source':pin(__file__),'producer':who,**ast_results,
  'helper_source':pin(S/'asr18_resident_wall_timings_v1.py'),'RAM_native_source':pin(source_pairs[0][2]),'driver_source':pin(source_pairs[1][2]),
  'wrapper_source':pin(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v14.py'),'generator_source':pin(N/'derive_inactive_timed_successor_v2.py'),
  'same_original_selected_call_order':calls,'CPU_semantics_controls':tests,'real_file_operation_controls':file_reports,
  'journal_source_untouched':pin(S/'bpe_bounded_production_journal_v4.py'),'fixed_channels':33,'empty_window_scalar_record_bytes':record_bytes,
  'GPU_barriers_added':0,'no_GPU_audio_model_or_training_call':True,'actual_native_updates':0,'draws_consumed':0,
  'diagnostic_claim_limit':'Wall time around existing operations includes existing waits; CUDA enqueue channels are not GPU kernel duration. Whole journal append leaves serialization/validation/write/fsync attribution unresolved.',
  'activation_authorized':False,'current_admitted_process_or_environment_modified':False,'peak_private_bytes':peak,'elapsed_seconds':time.monotonic()-began,'resource_samples':samples}
 write(N/'actual_CPU_instrumentation_proof_v1.json',report)
 print(json.dumps({'report':pin(N/'actual_CPU_instrumentation_proof_v1.json'),'passed':True,'peak_private_bytes':peak,'elapsed_seconds':time.monotonic()-began}),flush=True)
finally:
 write(OUT/'CPU_completion.json',{'producer':who,'elapsed_seconds':time.monotonic()-began,'peak_private_bytes':peak,'actual_exit_unobserved_inside_child':True})
