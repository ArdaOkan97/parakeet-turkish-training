import hashlib,json,os
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1')
F=N/'resident_RAM_two_logical_packet_lookahead_v1';A=N/'resident_RAM_training_actual_22';B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000021'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def who(pid):
 p=psutil.Process(pid);return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
boot=read(I/'boot.json');child=read(A/'current_child.json');runtime=read(I/'configuration.json');review=read(F/'ROOT_source29_approved_two_logical_lookahead_saved158600_final_execution_review_v1.json')
assert child['invocation']==21 and boot['source']==review['driver_source'] and boot['source_checkpoint']==review['initial_checkpoint']
assert all(runtime[k]==v for k,v in review['future_seven_runtime_bindings'].items())
ids={'native':boot['producer'],'native_interpreter_launcher':child['launcher'],'wrapper':who(child['launcher']['ppid'])}
assert all(abs(who(x['pid'])['create_time']-x['create_time'])<.001 for x in ids.values())
receipt={'kind':'ASR18_actual_source29_owned_boot_runtime_bindings_v1','owned_session':87302,'actual_initial_chunk_id':'69c0d2',
 'admitted_ROOT_review':pin(F/'ROOT_source29_approved_two_logical_lookahead_saved158600_final_execution_review_v1.json'),
 'boot':pin(I/'boot.json'),'current_child':pin(A/'current_child.json'),'effective_configuration':pin(I/'configuration.json'),
 'actual_runtime_identities':ids,'seven_runtime_bindings_exact':True,'restore_and_positive_training_still_pending':not(I/'actual_parent_restore.json').exists(),
 'CPU_identity_pending_until_actual_ready':True,'no_extra_GPU_probe_or_second_launch':True}
with (A/'actual_owned_source29_boot_session.json').open('x',encoding='utf-8')as f:json.dump(receipt,f,indent=2);f.write('\n')
# Sole-owner metadata pointer only; do not touch admitted source/environment.
lease_path=P/'work/gpu_training_owner.json';lease=read(lease_path)
assert lease['owner']=='/root/memory_diagnosis'and lease['actual_owned_worker_pid']==ids['native']['pid']and lease['actual_GPU_running']is True
old_session=lease.get('current_owned_session');lease['current_owned_session']=87302
history=lease.setdefault('owner_session_pointer_history',[])
history.append({'prior_owned_session':old_session,'current_owned_session':87302,'current_native':ids['native'],
 'actual_boot_receipt':pin(A/'actual_owned_source29_boot_session.json')})
import sys
sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
write_json(lease_path,lease)
print(json.dumps({'boot_receipt':pin(A/'actual_owned_source29_boot_session.json'),'ids':ids,'owned_session':87302,
 'session_pointer_only_reconciled_from':old_session,'restore_present':(I/'actual_parent_restore.json').exists(),
 'actual_failure_present':(I/'actual_failure.json').exists()}))
