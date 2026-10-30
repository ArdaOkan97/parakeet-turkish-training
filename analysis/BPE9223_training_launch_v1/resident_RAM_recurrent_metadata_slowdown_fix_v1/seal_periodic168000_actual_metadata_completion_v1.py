import hashlib,json,os,time
from pathlib import Path
import psutil
F=Path(__file__).parent;A=F.parent/'resident_RAM_training_actual_24'
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
statuspath=A/'actual_periodic168000_and_resumed_training_status.json';toolpath=A/'actual_periodic168000_sealer_v2_TOOL.json';s=read(statuspath);t=read(toolpath)
assert s['passed']is True and t['actual_TOOL_exit_code']==0 and s['actual_optimizer_updates']>168200
closed=[]
for w in (s['sealer_identity'],s['sealer_launcher']):
 current=None
 try:
  p=psutil.Process(w['pid']);current={'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
 except psutil.NoSuchProcess:pass
 assert current is None or abs(current['create_time']-w['create_time'])>=.001
 closed.append({'identity':w,'exact_dead':True,'current_PID_identity_if_reused':current})
value={'kind':'ASR18_actual_periodic168000_metadata_sealer_completion_v1','created_unix':time.time(),'actual_status':pin(statuspath),'actual_TOOL':pin(toolpath),'actual_TOOL_exit_code':0,
 'all_exact_metadata_identities_dead':True,'exact_metadata_identity_closure_records':closed,
 'failed_v1_TOOL_preserved_historical_PID_ctime_UNRECORDED':pin(A/'actual_periodic168000_metadata_sealer_failed_v1_TOOL.json'),
 'corrected_v2_ROOT_scope':pin(F/'ROOT_source31_periodic168000_metadata_observer_v2_repair_scope_v1.json'),
 'closed_ROOT_partial_scoring_phase':pin(F/'ROOT_periodic168000_fixed350_and_CPU11_max2_phase_partial_verification_v1.json'),
 'closed_ROOT_direct20_no_journal_reread':pin(F/'ROOT_periodic168000_direct20_sameplan31_resumed_positive_verification_v1.json'),
 'all_six_core_evidence_and_seven_prediction_pins_in_status':True,'no_science_or_guard_change_or_extraDEV':True,
 'production_owned_session2901_continues':'UNCHANGED','full_frozen_long_validation_and_human_speed_fix_still_pending':True}
p=A/'actual_periodic168000_metadata_sealer_completion.json'
with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({'completion':pin(p),'TOOL':pin(toolpath),'exact_metadata_deaths':closed}))
