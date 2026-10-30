"""Routine metadata outcome/death closure only; never reads journal or payload."""
import hashlib,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent
A=F.parent.parent/'resident_RAM_training_actual_26'
OUT=A/'actual_dynamic_natural_recovery_01'
def require(v,m):
 if not v:raise ValueError(m)
def pin(p):
 p=Path(p).resolve();require(p.stat().st_size<=8*1024**2,'Compact metadata/source only');b=p.read_bytes()
 return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):
 require(Path(p).stat().st_size<=8*1024**2,'Compact JSON only');return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def alive(v):
 try:return abs(psutil.Process(v['pid']).create_time()-v['create_time'])<.001
 except psutil.NoSuchProcess:return False
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('PYTHONUTF8')=='1' and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')) and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'Process-local E CPU-only metadata environment')
sample=sample_memory(me.pid,'source36_recovery_outcome_closure');m=sample['system_memory']
require(sample['paired_sampling_within_one_second'] and sample['worker_private_memory']['private_bytes']<536870912 and m['commit_available_bytes']>=5055306368 and m['physical_available_bytes']>=2305306368 and shutil.disk_usage('E:/').free>=100000000000,'Original full prospective512/full225/prefetch/floors/E100GB metadata admission')
status=OUT/'actual_recovery_status.json';r=read(status)
require(r['passed'] is True and r['natural_saved_updates']==179000 and r['new_positive_range']==[178801,178821] and r['new_positive_segment']['sha256']=='44296a91ff3c4fcb9222dfdc7794086e789295aa79dac07801f692c97839eb23','Actual successful one observation and exact same new21 snapshot metadata')
dead=[]
for role,k in [('recovery_observer','sealer_identity'),('recovery_interpreter_launcher','sealer_launcher_identity')]:
 v=r[k];require(not alive(v),'Actual observer exact identity remains live');dead.append({'role':role,**v,'exact_alive':False})
require(all(alive(v) for v in r['all5_exact_alive']),'Same5 production identities remain alive; do not touch owned42997')
tool=OUT/'actual_recovery_TOOL_result_v1.json'
write(tool,{'kind':'actual_source36_natural179000_recovery_TOOL_result_v1','tool':'functions.exec/tools.exec_command','chunk_id':'142151','exit_code':0,'wall_time_seconds':0.5519419,'owned_production_session_consumed':False,'production_session':42997,'direct_command_not_Popen_fixture':True,'actual_status':pin(status),'actual_new21_segment':r['new_positive_segment']})
result={'kind':'actual_source36_natural179000_recovery_metadata_completion_v1','passed':True,'actual_TOOL_result':pin(tool),'actual_status':pin(status),'observer_source':pin(F/'seal_source36_dynamic_natural_recovery_v1.py'),'completion_source':pin(__file__),'recorded_observer_exact_deaths':dead,'completion_identity':who(me),'completion_interpreter_launcher_identity':who(me.parent()),'completion_pair_alive_at_write_require_independent_post_exit_check':True,'same5_production_exactalive':r['all5_exact_alive'],'sample':sample,'no_journal_snapshot_prefix_whole_tensor_audio_archive_candidate_GC_or_production_operation':True,'human_speed_fix_open':True}
path=OUT/'actual_recovery_metadata_completion_v1.json';write(path,result)
print(json.dumps({'completion':pin(path),'actual_TOOL':pin(tool),'observer_exact_deaths':dead,'completion_identity':result['completion_identity'],'completion_launcher_identity':result['completion_interpreter_launcher_identity']}),flush=True)
