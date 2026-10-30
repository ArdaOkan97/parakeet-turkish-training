"""Compact actual launch metadata only; no process or model operation."""
import copy,hashlib,json,os,time
from pathlib import Path
import psutil
F=Path(__file__).parent;A=F.parent/'resident_RAM_training_actual_24';B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000023'
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def who(pid):
 p=psutil.Process(pid);return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
reviewpath=F/'ROOT_source31_fresh_signature_collect_saved167400_final_execution_review_v1.json';review=read(reviewpath)
boot=read(I/'boot.json');child=read(A/'current_child.json');c=read(I/'configuration.json');base=read(review['source_configuration']['path'])
assert boot['source']==review['driver_source']and boot['source_checkpoint']==review['initial_checkpoint']
assert all(c[k]==v for k,v in review['future_seven_runtime_bindings'].items())
inverse=copy.deepcopy(c)
for k in review['original_seven_runtime_binding_kinds']:
 if k in base:inverse[k]=base[k]
 else:del inverse[k]
assert inverse==base
native=who(boot['producer']['pid']);launch=who(child['launcher']['pid']);wrapper=who(launch['ppid'])
assert native==boot['producer']and launch==child['launcher']and native['ppid']==launch['pid']
value={'kind':'ASR18_actual_owned_source31_boot_session_v1','created_unix':time.time(),'owned_session':2901,'owner':'/root/memory_diagnosis','initial_TOOL_chunk':'3887ed',
 'actual_initial_TOOL':pin(A/'actual_source31_owned_launch_initial_chunk.json'),'execution_review':pin(reviewpath),'boot':pin(I/'boot.json'),'configuration':pin(I/'configuration.json'),
 'current_child':pin(A/'current_child.json'),'source_configuration':review['source_configuration'],'all7_runtime_bindings_exact':True,'all_other_source_configuration_values_exact':True,
 'actual_native_identity':native,'actual_native_launcher_identity':launch,'actual_wrapper_identity':wrapper,'all_three_boot_identities_exact_alive':True,
 'restore_receipt_present_at_snapshot':(I/'actual_parent_restore.json').exists(),'branch_receipt_present_at_snapshot':(I/'actual_journal_branch.json').exists(),
 'CPU_ready_present_at_snapshot':(I/'CPU/actual_CPU_ready.json').exists(),'actual_failure_present_at_snapshot':(I/'actual_failure.json').exists(),
 'positive_recovery_or_speed_not_claimed_from_boot':True,'full_target_updates':401072,'administrative_update_stop':None}
p=A/'actual_owned_source31_boot_session.json'
with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({'boot_session':pin(p),'native':native,'launcher':launch,'wrapper':wrapper,'restore_present':value['restore_receipt_present_at_snapshot'],'branch_present':value['branch_receipt_present_at_snapshot']}))
