"""Bounded metadata/new-event observation; no old-whole-journal or tensor reads."""
import datetime,hashlib,json
from pathlib import Path
import psutil
N=Path(__file__).parent;B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01')
A=N/'resident_RAM_training_actual_12';I=B/'invocations/owner_000011';C=B/'checkpoints/update_000133200'
def require(ok,msg):
    if not ok:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def pin(p):
    p=Path(p);raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(p,v):require(not p.exists(),'Fresh actual receipt');p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
progress=read(B/'progress/latest.json');boot=read(I/'boot.json');restore=read(I/'actual_parent_restore.json')
branch=read(I/'actual_journal_branch.json');CPU=read(I/'CPU/actual_CPU_ready.json');persistence=read(C/'persistence_verified.json');pointer=read(C/'pointer.json')
require(progress['producer']==boot['producer']and progress['actual_optimizer_updates']>=133220
    and progress['last_saved_updates']>=133200 and progress['optimizer_poisoned']is False and progress['schedule_poisoned']is False
    and progress['source']==boot['source']and progress['planned_native_updates']==401072 and progress['administrative_update_stop']is None,
    'Actual same new owner, passed prior failure and post133200 positive work')
require(restore['passed']and restore['pointer']==boot['source_checkpoint']
    and restore['all207_threeAdam_all4RNG_match_actual_pre_save']and restore['next_draw_packet_match']
    and restore['restore_occurred_only_at_startup_before_native']and restore['branch']['original_tail_modified']is False,
    'Fresh exact saved133000 restore before native')
require(branch['saved_prefix']['offset']==270832675 and branch['new_active_journal']['offset']==270832675
    and branch['new_active_journal']['sha256']==branch['saved_prefix']['sha256']
    and branch['historical_unsaved_tail_updates']==list(range(133001,133138))
    and branch['native_updates_or_draws_in_copy']==0,'Only saved prefix copied,137 historical attempts retained')
require(pointer['committed_updates']==persistence['optimizer_updates']==133200 and persistence['passed']
    and persistence['named_tensors_verified']==854 and persistence['weights_verified']==207 and persistence['Adam_states_verified']==215
    and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256']
    and pointer['saved_semantic_state_verified']['passed'],'Actual full streamed133200 persistence, independent PRE/SAVED equality')
newjournal=Path(branch['new_active_journal']['path']);require(newjournal==B/'journals/commits_000011.jsonl','Exact active new lineage')
raw=[];events=[]
with newjournal.open('rb')as f:
    f.seek(270832675)
    for update in range(133001,133139):
        line=f.readline(131073);require(line.endswith(b'\n')and len(line)<=131072,'Bounded complete new event')
        value=json.loads(line)
        require(value['optimizer_updates']==value['native']['update']==value['prepared_evidence']['actual_positive_native_update']==update,
            'New ordered actual positive update, never copied historical tail')
        require(value['prepared_evidence']['producer_source_sha256']==CPU['source']['protocol']['sha256'],'Exact new CPU6 preparation provenance')
        raw.append(line);events.append(value)
    end=f.tell()
for who in (boot['producer'],CPU['producer_identity']):
    require(abs(psutil.Process(who['pid']).create_time()-who['create_time'])<.001,'Same actual native/CPU identity remains alive')
snapshot=A/'actual_post133200_progress.json';write(snapshot,progress)
segment=A/'actual_new_positive_events_133001_133138.jsonl';require(not segment.exists(),'Fresh bounded new segment');segment.write_bytes(b''.join(raw))
report={'kind':'ASR18_actual_single_live_packet_saved133000_recovery_and133200_persistence_v1','passed':True,
    'actual_owned_session':60255,'invocation':11,'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'ROOT_execution_review':pin(N/'ROOT_single_live_packet_saved133000_execution_review_v1.json'),
    'boot':pin(I/'boot.json'),'actual_fresh_restore':pin(I/'actual_parent_restore.json'),'actual_journal_branch':pin(I/'actual_journal_branch.json'),
    'actual_CPU_ready':pin(I/'CPU/actual_CPU_ready.json'),'source_configuration':pin(N/'resident_RAM_single_packet_resource_source_v1/configuration_candidate.json'),
    'new_native_identity':boot['producer'],'new_CPU_identity':CPU['producer_identity'],
    'same_native_and_CPU_actually_alive':True,'postpositive_progress':pin(snapshot),'actual_optimizer_updates':progress['actual_optimizer_updates'],
    'saved_checkpoint':pin(C/'pointer.json'),'saved_persistence':pin(C/'persistence_verified.json'),
    'new138_ordered_positive_event_segment':pin(segment),'new_segment_source_offset':270832675,'new_segment_end_offset':end,
    'first_new_positive_update':events[0]['optimizer_updates'],'last_bounded_positive_update':events[-1]['optimizer_updates'],
    'all137_historical_unsaved_attempts_genuinely_redone':True,'prior_failure_frontier_passed':True,
    'no_old_whole_journal_tensor_or_archive_read_by_this_observer':True,
    'full401072_adminNone':True,'no_guard_relaxation_or_measured_memory_saving_claimed':True,
    'inactive_timing14_and_interval_preflight_not_activated':True}
out=A/'actual_saved133200_recovery_status.json';write(out,report);print(json.dumps(pin(out)),flush=True)
