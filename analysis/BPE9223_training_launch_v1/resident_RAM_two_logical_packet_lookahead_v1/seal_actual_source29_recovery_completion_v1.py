import hashlib,json
from pathlib import Path
import psutil
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1');A=N/'resident_RAM_training_actual_22';F=N/'resident_RAM_two_logical_packet_lookahead_v1';O=A/'actual_saved158800_recovery_01'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dead(w):
 try:return abs(psutil.Process(w['pid']).create_time()-w['create_time'])>=.001
 except psutil.NoSuchProcess:return True
status=read(O/'actual_recovery_status.json');people=[status['sealer_identity'],status['sealer_launcher']]
assert all(dead(x)for x in people)
assert status['new21_count']==21 and status['new21_segment']=={'path':str(Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01/journals/commits_000021.jsonl')),
 'start_offset':321119606,'end_offset':321160591,'bytes':40985,'sha256':'d45c4c2545054d0a53d2f4962e56eced7903fb1ad44f718449b84a3b2424519d'}
boot=F/'ROOT_source29_actual_boot_runtime7_saved158600_restore_verification_v1.json';direct=F/'ROOT_source29_saved158600_direct21_redone_positive_verification_v1.json'
assert pin(boot)['sha256']=='4380727ac2e4ad34c6b76252a9fb0f90993beec43d40c99c9b8a6b49d4835b7f'
assert pin(direct)['sha256']=='66da6c78d7e8cde0017a0526ff24e843326baa16b531b9cd6f1051cae3e0081c'
value={'kind':'ASR18_actual_source29_natural158800_recovery_metadata_sealer_completion_v1','actual_TOOL_exit_code':0,
 'actual_TOOL_chunk_id':'2b38d1','actual_metadata_sealer_status':pin(O/'actual_recovery_status.json'),
 'sealer_exact_identities_dead':[{'identity':x,'exact_dead':dead(x)}for x in people],
 'ROOT_closed_actual_boot_runtime7_restore':pin(boot),'ROOT_closed_direct21_positive_verification':pin(direct),
 'new21_segment_exactly_matches_closed_ROOT40985_d45c4c25':True,
 'no_ROOT_original_journal_reread_requested':True,'ROOT_actual_recovery_acceptance_pending':True,
 'full401072_training_continues_owned87302_unchanged':True,'fixed400_speed_result_still_pending':True}
path=O/'actual_metadata_sealer_completion.json'
with path.open('x',encoding='utf-8')as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({'status':pin(O/'actual_recovery_status.json'),'completion':pin(path),
 'progress':status['actual_progress']['actual_optimizer_updates'],'all5':status['actual_all5_exact_alive']}))
