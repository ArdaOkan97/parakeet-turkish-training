import hashlib,json,os,time
from pathlib import Path
import psutil
F=Path(__file__).parent;O=F.parent/'resident_RAM_training_actual_24/actual_saved167600_recovery_01'
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
r=read(O/'actual_recovery_status.json');t=read(O/'actual_recovery_sealer_TOOL.json')
assert r['passed']is True and t['actual_TOOL_exit_code']==0
seg=r['new37_segment'];assert seg['start_offset']==338402719 and seg['end_offset']==338475435 and seg['bytes']==72716 and seg['sha256']=='ff6e0cfd4a48e893ec2d1126ec5924e0d601cf6487194089b604407255c13194'
assert r['new37_count']==37 and r['new37_range']==[167401,167437]
closed=[]
for w in (r['sealer_identity'],r['sealer_launcher']):
 current=None
 try:
  p=psutil.Process(w['pid']);current={'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
 except psutil.NoSuchProcess:pass
 dead=current is None or abs(current['create_time']-w['create_time'])>=.001
 assert dead;closed.append({'identity':w,'exact_dead':True,'current_PID_identity_if_reused':current})
value={'kind':'ASR18_actual_source31_natural167600_recovery_metadata_completion_v1','created_unix':time.time(),
 'actual_status':pin(O/'actual_recovery_status.json'),'actual_TOOL':pin(O/'actual_recovery_sealer_TOOL.json'),'actual_TOOL_exit_code':0,
 'all_exact_metadata_identities_dead':True,'exact_metadata_identity_closure_records':closed,
 'closed_ROOT_boot_review':pin(F/'ROOT_source31_actual_boot_runtime7_saved167400_restore_verification_v1.json'),
 'closed_ROOT_direct37_review':pin(F/'ROOT_source31_saved167400_direct37_redone_positive_verification_v1.json'),
 'actual_new37_snapshot':r['actual37_snapshot'],'same_closed_ROOT_new37_bytes_and_hash':True,
 'no_original_journal_reread_by_this_completion':True,'actual_native_and_owner_continued_unchanged':True,'speed_fix_not_accepted_from_recovery':True}
p=O/'actual_recovery_metadata_completion.json'
with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({'completion':pin(p),'TOOL':pin(O/'actual_recovery_sealer_TOOL.json'),'exact_metadata_deaths':len(closed)}))
