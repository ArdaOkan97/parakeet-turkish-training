import os,sys,json,time,ctypes,hashlib,ast,traceback
from pathlib import Path
import psutil
X=Path(__file__).parent
SCOPE=X/'ROOT_source41_original212000_quality_natural212200_resumed_metadata_INACTIVE_SOURCE_preparation_scope_v1.json'
INITIAL=X/'original212000_source41_SCHEMA_SOURCE_read_initial_BEFORE_predicate_v1.json'
CAPTURE=X/'original212000_source41_SCHEMA_SOURCE_capture_v1.json'
RESULT=X/'original212000_source41_SCHEMA_SOURCE_read_result_v1.json'
EXIT=X/'original212000_source41_SCHEMA_SOURCE_read_numeric_exit_v1.json'
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
rc=1
try:
 for p in [INITIAL,CAPTURE,RESULT,EXIT]:assert not p.exists(),str(p)
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 initial=sample();write(INITIAL,initial)
 assert initial['commit_available_bytes']>=6028384896,initial
 assert initial['physical_available_bytes']>=3278384896,initial
 assert initial['self_private_bytes']<=536870912 and initial['E_free_bytes']>=100000000000
 assert os.getenv('CUDA_VISIBLE_DEVICES')=='-1'
 assert os.getenv('TORCH_EXTENSIONS_DIR')==r'E:\gpu_training_scratchpad\cache\torch_extensions'
 scope=json.loads(SCOPE.read_text(encoding='utf-8'))
 inputs=[
 {'path':str(X/'run_ASR18_BPE9223_resident_RAM_training_v41_dual_config_v3.py'),'bytes':142995,'sha256':'bd1017bb044c21961a850861f7dc362c5fbcd83463c523428cd1ef0a185f4774'},
 {'path':str(X/'asr18_current_plan_throughput_checkpoint_transition_v6_dual_config_v3.py'),'bytes':18467,'sha256':'a0a1f7e3760d677be1de1183cb05d486d9834f4394cba36f22e1042a1691cdc2'},
 {'path':str(X/'source41_saved212000_final_configuration_v1/configuration.json'),'bytes':235793,'sha256':'6d72936bd6d79530051e10a270493f454ec09b69ff23d60107669950288fc2b9'}]
 captured=[]
 for expected in inputs:
  guard();p=Path(expected['path']);b=p.read_bytes();actual={'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
  assert actual['bytes']==expected['bytes'] and actual['sha256']==expected['sha256']
  captured.append({'pin':actual,'text':b.decode('utf-8-sig')})
 write(CAPTURE,{'kind':'NEW_allowed_SOURCE_input_capture_only_v1','inputs':captured,'source_or_ROOT_receipt_only':True,'actual_quality_state_or_data_read':False})
 write(RESULT,{'passed':True,'initial':pin(INITIAL),'capture':pin(CAPTURE),'identities':[initial['self'],initial['launcher']],'final_sample':guard(),'candidate_imported_or_executed':False})
 rc=0
except BaseException as e:
 if not RESULT.exists():write(RESULT,{'passed':False,'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(limit=8),'identities':[who(psutil.Process()),who(psutil.Process().parent())]})
finally:
 write(EXIT,{'numeric_os_exit':rc,'identities':[who(psutil.Process()),who(psutil.Process().parent())]})
 print(json.dumps({'numeric':rc,'initial':pin(INITIAL) if INITIAL.exists() else None,'result':pin(RESULT),'exit':pin(EXIT),'capture':pin(CAPTURE) if CAPTURE.exists() else None}))
sys.exit(rc)

