import hashlib,json
from pathlib import Path
import psutil
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1');A=N/'resident_RAM_training_actual_22';F=N/'resident_RAM_two_logical_packet_lookahead_v1'
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000021'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def who(pid):
 p=psutil.Process(pid);return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
ready=read(I/'CPU/actual_CPU_ready.json');progress=read(B/'progress/latest.json');child=read(A/'current_child.json');boot=read(I/'boot.json')
assert progress['producer']==boot['producer']and progress['actual_optimizer_updates']>=158621 and not progress['optimizer_poisoned']and not progress['schedule_poisoned']
assert ready['packet_credit']['logical_packet_maximum']==2 and ready['source']['protocol']['sha256']=='d4353132b93abefd5fb72d0417b408264ac0fe8a95215db2cba00914e17ad05f'
cpu=ready['producer_identity'];ids={'native':boot['producer'],'native_interpreter_launcher':child['launcher'],
 'wrapper':who(child['launcher']['ppid']),'CPU':cpu,'CPU_interpreter_launcher':who(cpu['ppid'])}
assert ids['CPU_interpreter_launcher']['ppid']==boot['producer']['pid']
assert all(abs(who(w['pid'])['create_time']-w['create_time'])<.001 for w in ids.values())
report={'kind':'ASR18_actual_source29_firstpositive_CPU11_current5_identity_v1','owned_session':87302,
 'current_progress':progress,'actual_runtime_identities':ids,'all_five_exact_alive':True,
 'actual_CPU_ready':pin(I/'CPU/actual_CPU_ready.json'),'actual_CPU_protocol':ready['source']['protocol'],
 'actual_packet_credit':ready['packet_credit'],'genuine_positive_beyond_old158620_frontier':True,
 'full158800_recovery_and_fixed400_speed_acceptance_pending':True,
 'ROOT_closed_boot_restore':pin(F/'ROOT_source29_actual_boot_runtime7_saved158600_restore_verification_v1.json')}
p=A/'actual_source29_firstpositive_runtime_family.json'
with p.open('x',encoding='utf-8')as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'runtime_receipt':pin(p),'CPU_ready':report['actual_CPU_ready'],'all5':ids,
 'actual_updates':progress['actual_optimizer_updates'],'last_saved':progress['last_saved_updates']}))
