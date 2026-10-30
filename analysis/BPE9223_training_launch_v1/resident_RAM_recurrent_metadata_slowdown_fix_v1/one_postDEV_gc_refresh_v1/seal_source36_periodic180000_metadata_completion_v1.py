"""Routine actual observer outcome and exact metadata identity closure only."""
import argparse,hashlib,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent;A=F.parent.parent/'resident_RAM_training_actual_26';BEGIN=time.monotonic()
def require(v,m):
 if not v:raise ValueError(m)
def pin(p):
 p=Path(p).resolve();require(p.stat().st_size<=8*1024**2,'Bounded compact source/metadata only');b=p.read_bytes()
 return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):
 require(Path(p).stat().st_size<=8*1024**2,'Compact JSON only');return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def alive(v):
 try:return abs(psutil.Process(v['pid']).create_time()-v['create_time'])<.001
 except psutil.NoSuchProcess:return False
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def guard(admission=False):
 s=sample_memory(os.getpid(),'source36_periodic180000_metadata_outcome_closure');m=s['system_memory']
 require(s['paired_sampling_within_one_second'] and s['worker_private_memory']['private_bytes']<536870912 and time.monotonic()-BEGIN<180 and m['commit_available_bytes']>=(5055306368 if admission else 2268435456) and m['physical_available_bytes']>=(2305306368 if admission else 1768435456) and shutil.disk_usage('E:/').free>=100000000000,'Original bounded metadata fullprospective512/full225/FULLprefetch/floors/E100GB')
 return s
def main(a):
 me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('PYTHONUTF8')=='1' and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')) and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'Process-local E CPU-only metadata environment')
 require(a.actual_TOOL_code==0,'Seal genuine successful TOOL outcome, no invented numeric native/CPU exits')
 initial=guard(True);status=A/'actual_periodic180000_and_resumed_training_status.json';r=read(status)
 require(r['passed'] is True and r['periodic_optimizer_updates']==180000 and r['actual_optimizer_updates']>=180220 and r['owned_session']==42997 and r['full_run_target']==401072 and r['administrative_update_stop'] is None,'Actual admitted scheduled phase and post180200 training')
 dead=[]
 for role,k in [('periodic_metadata_observer','sealer_identity'),('periodic_metadata_interpreter_launcher','sealer_launcher')]:
  v=r[k];require(not alive(v),'Actual metadata observer exact identity still alive');dead.append({'role':role,**v,'exact_alive':False})
 boot=read(Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01/invocations/owner_000025/boot.json'))
 child=read(A/'current_child.json');native=r['native'];CPU=r['current_CPU']
 require(native==boot['producer'] and alive(native) and alive(CPU),'Same admitted native and current postDEV CPU exactalive')
 family=[{'role':'native',**native},{'role':'native_interpreter_launcher',**child['launcher']},{'role':'wrapper',**who(psutil.Process(child['launcher']['ppid']))},{'role':'CPU13',**CPU},{'role':'CPU_interpreter_launcher',**who(psutil.Process(CPU['ppid']))}]
 require(all(alive(v) for v in family),'Five exact current runtime identities alive')
 tool=A/'actual_periodic180000_metadata_TOOL_result_v1.json'
 write(tool,{'kind':'actual_source36_periodic180000_metadata_TOOL_result_v1','tool':'functions.exec/tools.exec_command','chunk_id':a.actual_TOOL_chunk,'exit_code':a.actual_TOOL_code,'wall_time_seconds':a.wall_seconds,'actual_status':pin(status),'direct_command_not_Popen_fixture':True,'owned_production_session_consumed':False,'production_session':42997})
 result={'kind':'actual_source36_periodic180000_metadata_completion_v1','passed':True,'actual_TOOL_result':pin(tool),'actual_status':pin(status),'observer_source':r['metadata_sealer_source'],'completion_source':pin(__file__),'recorded_observer_exact_deaths':dead,'completion_identity':who(me),'completion_interpreter_launcher_identity':who(me.parent()),'completion_pair_alive_at_write_require_independent_post_exit_check':True,'current5_exactalive':family,'initial_guard':initial,'final_guard':guard(),'no_journal_prediction_plan_archive_tensor_audio_candidate_GC_or_production_operation':True,'long178820_181020_and_human_speed_fix_still_pending':True}
 path=A/'actual_periodic180000_metadata_completion_v1.json';write(path,result)
 print(json.dumps({'completion':pin(path),'actual_TOOL':pin(tool),'observer_exact_deaths':dead,'current5':family,'completion_identity':result['completion_identity'],'completion_launcher_identity':result['completion_interpreter_launcher_identity']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--actual-TOOL-chunk',dest='actual_TOOL_chunk',required=True);p.add_argument('--actual-TOOL-code',dest='actual_TOOL_code',type=int,required=True);p.add_argument('--wall-seconds',type=float,required=True);main(p.parse_args())
