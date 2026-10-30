"""Bounded metadata recovery seal after source22 naturally saves156400.
Only new101 journal events, first10 existing timing records and compact metadata.
No model/audio/tensor-payload/oldwhole-journal read, GPU/reader probe or state change.
"""
import hashlib,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
A=N/'resident_RAM_training_actual_18';F=N/'resident_RAM_system_headroom156300_failure_actual_v1'
B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000017'
began=time.monotonic();psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
def require(v,m):
 if not v:raise ValueError(m)
def pin(p):
 p=Path(p);raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def guard():
 s=sample_memory(os.getpid(),'source22_saved156400_actual_recovery_metadata');m=s['system_memory']
 require(s['paired_sampling_within_one_second']and s['worker_private_memory']['private_bytes']<512*1024**2 and m['commit_available_bytes']>=2000000000+268435456 and m['physical_available_bytes']>=1500000000+268435456 and shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-began<180,'Original metadata/full256MiB/headroom/E guards')
 return s
initial=guard();require(initial['system_memory']['commit_available_bytes']>=2000000000+268435456+512*1024**2 and initial['system_memory']['physical_available_bytes']>=1500000000+268435456+512*1024**2,'Full512MiB metadata admission')
status_path=A/'actual_native_pressure_saved156400_recovery_status.json';require(not status_path.exists(),'One actual recovery status')
failure=read(F/'actual_failure_closure.json');execution_path=N/'ROOT_native_preupdate_commit_pressure_saved156200_execution_review_v1.json'
require(pin(execution_path)['sha256']=='d793e48a0b999a8aa993d5934951ca29cddfa4364659b72f81b73d4a83790300','Exact admitted ROOT execution')
boot=read(I/'boot.json');restore=read(I/'actual_parent_restore.json');branch=read(I/'actual_journal_branch.json')
require(boot['source']==pin(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v22.py')
 and restore['passed']is True and restore['all207_threeAdam_all4RNG_match_actual_pre_save']is True
 and restore['next_draw_packet_match']is True and restore['restore_occurred_only_at_startup_before_native']is True
 and restore['pointer']==failure['inputs']['pointer'] and restore['branch']['original_tail_modified']is False,'Actual saved156200 complete restoration/nextdraw before native')
saved=failure['saved_prefix_verified_once'];oldwhole=failure['whole_journal_verified_once']
require(branch['predecessor']==pin(F/'actual_failure_closure.json') and branch['saved_prefix']==saved
 and branch['preserved_original']==oldwhole and branch['historical_unsaved_tail_updates']==list(range(156201,156301))
 and branch['native_updates_or_draws_in_copy']==0 and branch['new_active_journal']=={'path':str(B/'journals/commits_000017.jsonl'),'offset':saved['offset'],'sha256':saved['sha256']}
 and Path(oldwhole['path']).stat().st_size==oldwhole['bytes'],'Actual AFTER-restore saved-prefix copy/0native0draw; oldwhole16+100tail historical, no new full old read')
raw=(B/'progress/latest.json').read_bytes();progress=json.loads(raw)
require(progress['producer']==boot['producer'] and progress['producer']['pid']==12328
 and progress['actual_optimizer_updates']>156400 and progress['native_updates_this_process']==progress['actual_optimizer_updates']-156200
 and not progress['optimizer_poisoned']and not progress['schedule_poisoned']
 and progress['planned_native_updates']==401072 and progress['administrative_update_stop']is None,'Same actual healthy source22 positive post156400, fullgoal/no finitecap')
oldprogress=read(failure['inputs']['last_emitted_progress']['path'])
require(len(progress['clocks'])==3 and all(new-old==progress['actual_optimizer_updates']-156300 for new,old in zip(progress['clocks'],oldprogress['clocks'])),'All3 Adam clocks continue exact saved lineage')
CK=B/'checkpoints/update_000156400';pointer=read(CK/'pointer.json');persistence=read(CK/'persistence_verified.json')
require(pointer['committed_updates']==156400 and persistence['passed']is True and persistence['optimizer_updates']==156400
 and persistence['named_tensors_verified']==854 and persistence['weights_verified']==207 and persistence['Adam_states_verified']==215
 and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256']
 and persistence['journal_prefix']==pointer['journal_prefix']
 and persistence['saved_semantic_state_verified']['passed']is True
 and persistence['saved_semantic_state_verified']['optimizer_updates']==156400
 and persistence['saved_semantic_state_verified']['saved_identity_frontier_sampler_and_journal_verified_before_publication']is True,
 'Standard156400 all854/207/215 PRE-SAVED and saved semantic publication')
config=read(I/'configuration.json');source_config_path=Path(config['source_configuration']['path'])
require(pin(source_config_path)==config['source_configuration'] and config['checkpoint_interval_updates']==200
 and read(source_config_path)['native_update_commit_pressure_policy']==config['native_update_commit_pressure_policy']
 and config['native_update_commit_pressure_policy']['parameters']['additional_native_commit_pressure_buffer_bytes']==2500000000,
 'Exact admitted source/runtime pressure policy and original200 checkpoint cadence')
newjournal=B/'journals/commits_000017.jsonl';segment_path=A/'actual_new101_positive_events_156201_156301.jsonl';require(not segment_path.exists(),'One bounded new segment')
segment_hash=hashlib.sha256()
with newjournal.open('rb')as source,segment_path.open('xb')as target:
 source.seek(saved['offset']);start=source.tell()
 for expected in range(156201,156302):
  line=source.readline(16385);require(line.endswith(b'\n')and len(line)<=16384,'One complete bounded new positive event')
  event=json.loads(line)
  require(event['optimizer_updates']==event['native']['update']==event['prepared_evidence']['actual_positive_native_update']==expected
   and event['prepared_evidence']['producer_source_sha256']=='e91ce45fc500df7a61a43e79437f78a6c068894e613fcd9014d52f894d5851d8',
   'Exactly ordered new positive156201..156301 with unchanged actualCPU9 source')
  target.write(line);segment_hash.update(line);guard()
 end=source.tell()
segment=pin(segment_path);require(segment['bytes']==end-start and segment['sha256']==segment_hash.hexdigest(),'Exact bounded101 segment offsets/hash')
timing_path=A/'actual_first200_native_pressure_timings.jsonl';require(not timing_path.exists(),'One bounded timing snapshot')
timings=[]
with (I/'private_metadata_window_timings.jsonl').open('rb')as source,timing_path.open('xb')as target:
 for i in range(10):
  line=source.readline(16385);require(line.endswith(b'\n')and len(line)<=16384,'Bounded completed timing record')
  entry=json.loads(line);require(entry['initial_updates']==156200+20*i and entry['committed_updates']==156220+20*i,'Only natural first200 production windows')
  require(entry['native_commit_pressure_wait_seconds']>=0 and 0<=entry['native_commit_pressure_waited_updates']<=20,'Actual admitted buffer wait telemetry')
  timings.append(entry);target.write(line)
require(timings[-1]['checkpoint_included']is True,'First200 timing includes natural156400 save')
ready_path=I/'CPU/actual_CPU_ready.json';ready=read(ready_path);CPU=ready['producer_identity'];child=read(A/'current_child.json')
family=[]
for role,who in [('native',boot['producer']),('native_interpreter_launcher',child['launcher']),('wrapper',{'pid':child['launcher']['ppid']}),('CPU',CPU),('CPU_interpreter_launcher',{'pid':CPU['ppid']})]:
 actual=psutil.Process(who['pid']);ctime=actual.create_time()
 require('create_time'not in who or abs(ctime-who['create_time'])<.001,'Exact current owned identity '+role)
 family.append({'role':role,'pid':actual.pid,'create_time':ctime,'ppid':actual.ppid(),'actual_exact_identity_alive':True})
family_path=A/'actual_source22_live_runtime_family_after156400.json';require(not family_path.exists(),'One current family capture');family_path.write_text(json.dumps(family,indent=2)+'\n',encoding='utf-8')
snapshot=A/'actual_source22_post156400_positive_progress.json';require(not snapshot.exists(),'One positive progress snapshot');snapshot.write_bytes(raw)
value={'kind':'ASR18_actual_source22_native_pressure_saved156400_recovery_status_v1','passed':True,'owned_session':35083,
 'actual_optimizer_updates':progress['actual_optimizer_updates'],'native_updates_this_process':progress['native_updates_this_process'],
 'native':progress['producer'],'current_CPU':CPU,'all3_clocks_advanced_exactly':True,'optimizer_poisoned':False,'schedule_poisoned':False,
 'evidence':{'ROOT_execution':pin(execution_path),'boot':pin(I/'boot.json'),'actual_restore':pin(I/'actual_parent_restore.json'),'after_restore_savedprefix_branch':pin(I/'actual_journal_branch.json'),
 'pointer156400':pin(CK/'pointer.json'),'persistence156400':pin(CK/'persistence_verified.json'),'positive_post156400_progress':pin(snapshot),'current_CPU_ready':pin(ready_path),'current_family':pin(family_path),
 'new101_positive_events':segment,'first200_existing_production_timings':pin(timing_path),'actual_effective_configuration':pin(I/'configuration.json'),'source_configuration':pin(source_config_path)},
 'new101_segment':{'path':str(newjournal),'offset_start':start,'offset_end':end,'bytes':end-start,'sha256':segment['sha256'],'updates':list(range(156201,156302))},
 'full156200_restore_before_native':True,'saved_prefix_copy_only_after_restore_with0native0draw':True,
 'old100_positive_tail_attempts_historical_and_genuinely_redone':True,'oldwhole16_verification_reused_from_closed_failure_and_actual_afterrestore_branch':oldwhole,
 'no_oldwhole_journal_or_payload_reread_by_this_observer':True,'saved156400_full854_207_215_PRE_SAVED_and_semantic_verification':True,
 'actual_changed_policy_buffer_wait_observation':{'completed_production_windows':10,'extra_buffer_bytes':2500000000,'wait_seconds':sum(x['native_commit_pressure_wait_seconds']for x in timings),
 'waited_updates':sum(x['native_commit_pressure_waited_updates']for x in timings),'no_memory_saving_required_allocation_forecast_throughput_or_prevention_claim':True},
 'native_update_commit_pressure_policy':config['native_update_commit_pressure_policy'],
 'Trackio_reader_verification_owned_by_ROOT':{'reader_PID_from_ROOT':26892,'last_ROOT_report':'healthy/advancing correctjournal17 catchup; preserve actual old current_status until caughtup','no_reader_probe_or_restart_by_this_observer':True,'fresh_ROOT_verification_required_before_acceptance':True},
 'full_run_target':401072,'administrative_update_stop':None,'training_continues':True,'actual_recovery_acceptance_and_Backlog_still_pending_ROOT':True,
 'guard_initial':initial,'guard_final':guard()}
status_path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':pin(status_path),'current_updates':progress['actual_optimizer_updates'],'native':progress['producer'],'CPU':CPU,'segment':value['new101_segment'],'pointer156400':pin(CK/'pointer.json'),'persistence156400':pin(CK/'persistence_verified.json')}),flush=True)
