"""Outcome and exact metadata identity closure for the one bound long observer."""
import argparse,hashlib,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent;OUT=F/'actual_source36_dynamic_long_validation_01';BEGIN=time.monotonic()
def require(v,m):
 if not v:raise ValueError(m)
def pin(p):
 p=Path(p).resolve();require(p.stat().st_size<=8*1024**2,'Bounded source/metadata only');b=p.read_bytes()
 return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):
 require(Path(p).stat().st_size<=8*1024**2,'Compact report only');return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def alive(v):
 try:return abs(psutil.Process(v['pid']).create_time()-v['create_time'])<.001
 except psutil.NoSuchProcess:return False
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def guard(admission=False):
 s=sample_memory(os.getpid(),'source36_long181020_observer_outcome_closure');m=s['system_memory']
 require(s['paired_sampling_within_one_second'] and s['worker_private_memory']['private_bytes']<536870912 and time.monotonic()-BEGIN<180 and m['commit_available_bytes']>=(5055306368 if admission else 2268435456) and m['physical_available_bytes']>=(2305306368 if admission else 1768435456) and shutil.disk_usage('E:/').free>=100000000000,'Original bounded metadata fullprospective512/full225/FULLprefetch/floors/E100GB')
 return s
def main(a):
 me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('PYTHONUTF8')=='1' and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')) and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'Process-local E CPU-only metadata environment')
 require(a.actual_TOOL_code==0,'Actual completed observation TOOL outcome only')
 initial=guard(True);report=OUT/'report.json';r=read(report);v=r['predeclared_validation']
 require(r['measurement_completed'] is True and v['start_updates']==178820 and v['end_updates']==181020 and r['current_progress']['actual_optimizer_updates']>=181020,'Same predeclared2200 actual-job interval completed')
 dead=[]
 for role,k in [('long_metadata_observer','observer_identity'),('long_metadata_interpreter_launcher','observer_launcher_identity')]:
  x=r[k];require(not alive(x),'Observed metadata exact identity remains alive');dead.append({'role':role,**x,'exact_alive':False})
 require(all(alive(x) for x in r['current_five_family']),'Same five production runtime identities alive;42997 untouched')
 tool=OUT/'actual_TOOL_result_v1.json'
 write(tool,{'kind':'actual_source36_long181020_observer_TOOL_result_v1','tool':'functions.exec/tools.exec_command','chunk_id':a.actual_TOOL_chunk,'exit_code':a.actual_TOOL_code,'wall_time_seconds':a.wall_seconds,'actual_report':pin(report),'direct_command_not_Popen_fixture':True,'production_session':42997,'owned_production_session_consumed':False,'measurement_completion_does_not_itself_claim_durable_fix':True})
 result={'kind':'actual_source36_long181020_metadata_completion_v1','measurement_completed':True,'actual_TOOL_result':pin(tool),'actual_report':pin(report),'observer_source':pin(F/'observe_source36_dynamic_durable_validation_v2.py'),'completion_source':pin(__file__),'recorded_observer_exact_deaths':dead,'completion_identity':who(me),'completion_interpreter_launcher_identity':who(me.parent()),'completion_pair_alive_at_write_require_independent_post_exit_check':True,'same5_production_exactalive':r['current_five_family'],'comparison_criteria_from_actual_observer':r['comparison_criteria'],'initial_guard':initial,'final_guard':guard(),'no_original_timing_points_journal_prediction_plan_archive_tensor_audio_candidate_GC_or_production_operation':True,'independent_ROOT_durable_result_assessment_pending':True}
 path=OUT/'actual_metadata_completion_v1.json';write(path,result)
 print(json.dumps({'completion':pin(path),'actual_TOOL':pin(tool),'observer_exact_deaths':dead,'completion_identity':result['completion_identity'],'completion_launcher_identity':result['completion_interpreter_launcher_identity']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--actual-TOOL-chunk',dest='actual_TOOL_chunk',required=True);p.add_argument('--actual-TOOL-code',dest='actual_TOOL_code',type=int,required=True);p.add_argument('--wall-seconds',type=float,required=True);main(p.parse_args())
