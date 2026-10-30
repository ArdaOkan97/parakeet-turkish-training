import os,sys,json,time,ctypes,hashlib,ast,traceback
from pathlib import Path
import psutil
X=Path(__file__).parent
SCOPE=X/'ROOT_reader8_Windows_publication_INACTIVE_SOURCE_preparation_scope_v1.json'
INITIAL=X/'reader8_Windows_publication_SOURCE_input_read_initial_BEFORE_predicate_v2.json'
CAPTURE=X/'reader8_Windows_publication_SOURCE_inputs_capture_v2.json'
RESULT=X/'reader8_Windows_publication_SOURCE_input_read_result_v2.json'
EXIT=X/'reader8_Windows_publication_SOURCE_input_read_numeric_exit_v2.json'
class PI(ctypes.Structure):
 _fields_=[('cb',ctypes.c_ulong),('CommitTotal',ctypes.c_size_t),('CommitLimit',ctypes.c_size_t),('CommitPeak',ctypes.c_size_t),('PhysicalTotal',ctypes.c_size_t),('PhysicalAvailable',ctypes.c_size_t),('SystemCache',ctypes.c_size_t),('KernelTotal',ctypes.c_size_t),('KernelPaged',ctypes.c_size_t),('KernelNonpaged',ctypes.c_size_t),('PageSize',ctypes.c_size_t),('HandleCount',ctypes.c_ulong),('ProcessCount',ctypes.c_ulong),('ThreadCount',ctypes.c_ulong)]
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def pin(p):
 b=p.read_bytes();return {'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def sample():
 z=PI();z.cb=ctypes.sizeof(z);assert ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(z),z.cb)
 return {'commit_available_bytes':(z.CommitLimit-z.CommitTotal)*z.PageSize,'physical_available_bytes':z.PhysicalAvailable*z.PageSize,'self_private_bytes':psutil.Process().memory_info().private,'E_free_bytes':psutil.disk_usage('E:/').free,'self':who(psutil.Process()),'launcher':who(psutil.Process().parent()),'unix':time.time(),'CUDA_VISIBLE_DEVICES':os.getenv('CUDA_VISIBLE_DEVICES'),'TORCH_EXTENSIONS_DIR':os.getenv('TORCH_EXTENSIONS_DIR')}
start=time.monotonic()
def guard():
 v=sample();assert time.monotonic()-start<=180;assert v['commit_available_bytes']>=3509949440;assert v['physical_available_bytes']>=3009949440;assert v['self_private_bytes']<=536870912;assert v['E_free_bytes']>=100000000000
 return v
CONDITIONAL=X/'reader8_Windows_publication_SOURCE_input_read_v2_conditional_execution_scope.json'
DENIAL=json.loads("{\"initial\":{\"path\":\"E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1/next212000_failure_repair_v1/reader8_Windows_publication_SOURCE_input_read_initial_BEFORE_predicate_v1.json\",\"bytes\":483,\"sha256\":\"af7735f27b92e8816fd2f6e60b48c54970226ea4b86564f1c88a617834f52f91\"},\"result\":{\"path\":\"E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1/next212000_failure_repair_v1/reader8_Windows_publication_SOURCE_input_read_result_v1.json\",\"bytes\":1650,\"sha256\":\"b504d4da0a276c54269daf3ed5bf0c0dd5a9040833463819cac1f2309e98210e\"},\"numeric\":{\"path\":\"E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1/next212000_failure_repair_v1/reader8_Windows_publication_SOURCE_input_read_numeric_exit_v1.json\",\"bytes\":237,\"sha256\":\"3adc56b158d1fd8f9d8b69fe915cc66f4801821fee3989ca8fadefcc56e7e599\"}}")
rc=1
try:
 for p in [INITIAL,CAPTURE,RESULT,EXIT,CONDITIONAL]:assert not p.exists(),str(p)
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 initial=sample();write(INITIAL,initial)
 assert initial['commit_available_bytes']>=6028384896,initial
 assert initial['physical_available_bytes']>=3278384896,initial
 assert initial['self_private_bytes']<=536870912 and initial['E_free_bytes']>=100000000000
 assert os.getenv('CUDA_VISIBLE_DEVICES')=='-1'
 assert os.getenv('TORCH_EXTENSIONS_DIR')==r'E:\gpu_training_scratchpad\cache\torch_extensions'
 scope_bytes=SCOPE.read_bytes();scope=json.loads(scope_bytes)
 assert scope['kind']=='ROOT_reader8_Windows_atomic_status_STATE_publication_INACTIVE_SOURCE_preparation_scope_v1'
 assert scope['accepted']is True and scope['reviewer']=='/root'and scope['owner']=='/root/memory_diagnosis'
 assert scope['failure_ROOT']['sha256']=='570b97558ad2186d2fa109631b11169957b817f09ef4bc6148015332dd496ced'
 assert scope['base_reader']['sha256']=='d7bb6494d197cfe4b4f7bc8c5af1f4b720be199e2fb8d3077cb8af45e450da22'
 original_scope_pin={'path':str(SCOPE.resolve()),'bytes':len(scope_bytes),'sha256':hashlib.sha256(scope_bytes).hexdigest()}
 write(CONDITIONAL,{'kind':'ASR18_reader8_SOURCE_input_read_v2_conditional_execution_binding','owner':'/root/memory_diagnosis',
  'authorization':'Direct parent message separately authorizes ONE fresh v2 SOURCE-input read after later genuinely changed ordinary headroom meets unchanged full bounds; oldv1 CLOSED no source reads.',
  'original_ROOT_SOURCE_scope':original_scope_pin,'v1_denial_pins_provenance_only_no_reopen':DENIAL,
  'initial_BEFORE_full_predicate':pin(INITIAL),'actual_initial':initial,
  'resources':{'prospective_commit':6028384896,'prospective_physical':3278384896,'runtime_commit':3509949440,'runtime_physical':3009949440,'self_private':536870912,'wall':180,'E_free':100000000000},
  'SOURCE_inputs':[scope['base_reader']]+scope['allowed_SOURCE_inputs'],'no_candidate_import_execution_or_actual_logs':True})
 inputs=[scope['base_reader']]+scope['allowed_SOURCE_inputs']
 captured=[]
 for expected in inputs:
  guard();p=Path(expected['path']);b=p.read_bytes();actual={'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
  assert actual['bytes']==expected['bytes'] and actual['sha256']==expected['sha256']
  captured.append({'pin':actual,'text':b.decode('utf-8-sig')})
 write(CAPTURE,{'kind':'NEW_allowed_SOURCE_input_capture_only_v1','inputs':captured,'source_or_ROOT_receipt_only':True,'actual_quality_state_or_data_read':False})
 write(RESULT,{'passed':True,'initial':pin(INITIAL),'capture':pin(CAPTURE),'identities':[initial['self'],initial['launcher']],'final_sample':guard(),'candidate_imported_or_executed':False,'conditional_execution_binding':pin(CONDITIONAL),'original_ROOT_scope':original_scope_pin})
 rc=0
except BaseException as e:
 if not RESULT.exists():write(RESULT,{'passed':False,'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(limit=8),'identities':[who(psutil.Process()),who(psutil.Process().parent())]})
finally:
 write(EXIT,{'numeric_os_exit':rc,'identities':[who(psutil.Process()),who(psutil.Process().parent())]})
 print(json.dumps({'numeric':rc,'initial':pin(INITIAL) if INITIAL.exists() else None,'result':pin(RESULT),'exit':pin(EXIT),'capture':pin(CAPTURE) if CAPTURE.exists() else None,'conditional_scope':pin(CONDITIONAL) if CONDITIONAL.exists() else None,'actual_initial':initial if 'initial'in globals()else None}))
sys.exit(rc)

