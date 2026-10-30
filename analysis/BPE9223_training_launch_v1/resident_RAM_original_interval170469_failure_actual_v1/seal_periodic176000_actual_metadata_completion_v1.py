"""Metadata-only completion after actual172k seal; no production operation."""
import hashlib,json,os,time
from pathlib import Path
import psutil
F=Path(__file__).resolve().parent;A=F.parent/'resident_RAM_training_actual_25'
assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
def read(p):return json.loads(Path(p).read_bytes())
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();assert len(b)<=2*1024**2
 return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def identity(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
statuspath=A/'actual_periodic176000_and_resumed_training_status.json'
toolpath=A/'actual_periodic176000_sealer_v1_TOOL.json'
s=read(statuspath);t=read(toolpath)
assert s['passed']is True and t['actual_TOOL_exit_code']==0 and s['actual_optimizer_updates']>176200
actual=read(A/'actual_periodic176000_metadata_Popen_result_v1.json')
assert actual['Popen_wait_code']==0 and actual['actual_exception']is None
known=[s['sealer_identity'],s['sealer_launcher'],*actual['observed_exact_job_identities'],actual['executor_identity'],actual['executor_launcher_identity']]
unique={(who['pid'],who['create_time']):who for who in known}
closed=[]
for who in unique.values():
 current=None
 try:current=identity(psutil.Process(who['pid']))
 except psutil.NoSuchProcess:pass
 assert current is None or abs(current['create_time']-who['create_time'])>=.001
 closed.append({'identity':who,'exact_dead':True,'current_PID_identity_if_reused':current})
value={'kind':'ASR18_actual_periodic176000_metadata_sealer_completion_v1','created_unix':time.time(),
 'actual_status':pin(statuspath),'actual_TOOL':pin(toolpath),'actual_TOOL_exit_code':0,
 'actual_Popen_result':pin(A/'actual_periodic176000_metadata_Popen_result_v1.json'),'actual_Popen_wait_code':0,
 'all_exact_original_metadata_identities_dead':True,'exact_original_metadata_identity_closure_records':closed,
 'all_six_core_evidence_and_seven_prediction_pins_in_status':True,
 'no_science_or_guard_change_or_extraDEV':True,'production_owned_session65067_continues':'UNCHANGED',
 'human_recurring_slowdown_fix_remains_open':True,'no_warm_speed_success_claim':True,
 'completion_source':pin(__file__),'completion_identity':identity(psutil.Process()),
 'completion_launcher_identity':identity(psutil.Process().parent())}
path=A/'actual_periodic176000_metadata_sealer_completion.json';assert not path.exists()
with path.open('x',encoding='utf-8',newline='\n')as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({'completion':pin(path),'exact_original_metadata_deaths':closed,
 'completion_identity':value['completion_identity'],'completion_launcher_identity':value['completion_launcher_identity']}),flush=True)
