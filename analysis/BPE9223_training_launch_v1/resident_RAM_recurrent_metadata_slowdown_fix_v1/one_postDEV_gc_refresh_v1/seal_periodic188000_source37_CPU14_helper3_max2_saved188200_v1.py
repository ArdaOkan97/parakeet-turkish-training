"""Compact actual188k full350, source37 CPU14/helper3/max2 normal closure and resumed training."""
import argparse,datetime,hashlib,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000026';A=Path(__file__).parent.parent.parent/'resident_RAM_training_actual_27'
began=time.monotonic();psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
def require(v,m):
 if not v:raise ValueError(m)
def pin(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def exact_alive(who):
 try:return abs(psutil.Process(who['pid']).create_time()-who['create_time'])<.001
 except psutil.NoSuchProcess:return False

parser=argparse.ArgumentParser()
parser.add_argument('--ROOT-review',required=True)
args=parser.parse_args()
def guard():
 v=sample_memory(os.getpid(),'periodic188000_compact_actual_metadata');m=v['system_memory']
 require(v['paired_sampling_within_one_second']and v['worker_private_memory']['private_bytes']<512*1024**2 and m['commit_available_bytes']>=2000000000+268435456 and m['physical_available_bytes']>=1500000000+268435456 and shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-began<180,'Original bounded metadata/headroom/E/full256MiB guards');return v
initial=guard()
initial_path=A/'actual_periodic188000_initial_full_resource_sample_v1.json'
with initial_path.open('x',encoding='utf-8',newline='\n')as initial_file:
 json.dump({'kind':'ASR18_periodic188000_original_initial_full_metadata_resource_sample_v1',
  'sample':initial,'original_runtime_guard_returned':True,
  'required_commit_bytes':4518435456+512*1024**2,
  'required_physical_bytes':1500000000+268435456+512*1024**2,
  'full_prospective_predicate_passed':initial['system_memory']['commit_available_bytes']>=4518435456+512*1024**2 and initial['system_memory']['physical_available_bytes']>=1500000000+268435456+512*1024**2,
  'captured_before_original_full_prospective_require_and_DEV_metadata_reads':True,
  'same_original_sample_no_extra_resource_query':True,
  'sealer_identity':{'pid':os.getpid(),'create_time':psutil.Process().create_time(),'ppid':os.getppid()},
  'sealer_launcher_identity':{'pid':os.getppid(),'create_time':psutil.Process(os.getppid()).create_time(),'ppid':psutil.Process(os.getppid()).ppid()},
  'metadata_source':pin(Path(__file__))},initial_file,indent=2)
 initial_file.write('\n');initial_file.flush();os.fsync(initial_file.fileno())
require(initial['system_memory']['commit_available_bytes']>=4518435456+512*1024**2 and initial['system_memory']['physical_available_bytes']>=1500000000+268435456+512*1024**2,'Full approved225 plus prospective512MiB metadata reserve')
# This new execution gate remains absent until ROOT accepts the genuine fullpass
# and binds its actual successor cohort. No identity or preparation count is guessed.
execution_review=read(args.ROOT_review)
require(execution_review['kind']=='ROOT_periodic188000_source37_CPU14_metadata_only_exact_execution_scope_v1'
 and execution_review['accepted']is True and execution_review['reviewer']=='/root'
 and execution_review['metadata_source']==pin(Path(__file__)),
 'Separate exact ROOT execution binding for this immutable source')
fullpass_binding=execution_review['actual_fullpass_interposition']
require(fullpass_binding['boundary_optimizer_updates']==187937
 and fullpass_binding['cohort_start_optimizer_updates']==187937,
 'Actual first fullpass187937 followed by the independently bound CPU cohort')
fullpass_ROOT_pin=fullpass_binding['ROOT_actual_fullpass_review']
require(pin(fullpass_ROOT_pin['path'])==fullpass_ROOT_pin,'Exact genuine fullpass ROOT gate')
fullpass_ROOT=read(fullpass_ROOT_pin['path'])
require(fullpass_ROOT['accepted']is True and fullpass_ROOT['reviewer']=='/root',
 'ROOT accepted actual fullpass interposition, never a preparation promise')
pre_periodic_ready_path=I/'CPU_after_DEV_1/actual_CPU_ready.json'
require(pin(pre_periodic_ready_path)==fullpass_binding['pre_periodic_CPU_ready'],
 'Exact actual post-fullpass ready pin supplied by the future execution gate')
pre_periodic_ready=read(pre_periodic_ready_path)
require(pre_periodic_ready['producer_identity']==fullpass_binding['pre_periodic_CPU']
 and pre_periodic_ready['source']['protocol']==pin(S/'fastmss_resident_feature_stream_v14.py')
 and pre_periodic_ready['packet_credit']['logical_packet_maximum']==2,
 'Actual post-fullpass CPU14/max2 cohort, not historical20436')
require(fullpass_binding['pre_periodic_CPU']['ppid']==fullpass_binding['pre_periodic_CPU_launcher']['pid'],
 'Exact native CPU ancestry in the future ROOT-bound cohort')
configuration_pin=pin(I/'configuration.json')
require(configuration_pin=={'path':str(I/'configuration.json'),'bytes':226990,
 'sha256':'39dc6a3aa4b7139d6209751de769d5aef63a507e7122f56a6720afd85bdb966d'},
 'Unchanged admitted source37 actual runtime configuration')
boot_ROOT_pin=execution_review['source37_actual_boot_ROOT']
require(pin(boot_ROOT_pin['path'])==boot_ROOT_pin
 and boot_ROOT_pin['sha256']=='d20639fe1176cb0b3894d3415b14646ecd720f1ac1f11cad0ba25921d45ed5fc',
 'Hash binding to closed source37 boot acceptance; no inherited-event re-audit')
inherited_pin=pin(I/'actual_inherited_consumed180000_refresh_opportunity.json')
require(inherited_pin==execution_review['inherited_consumed180000_receipt'],
 'Original180k event inherited consumed; no second event or historical fingerprint read')
state=read(B/'periodic_DEV.json');entries=[v for v in state['completed_evaluations']if v['optimizer_updates']==188000];require(len(entries)==1,'One real188k record');entry=entries[0]
previous=[v for v in state['completed_evaluations']if v['optimizer_updates']==184000];require(len(previous)==1,'One previous184k quality record');previous_errors=previous[0]['score']['word_errors']['tr']
report=read(entry['DEV']['path']);preservation=read(entry['frontier_preservation']['path']);CK=B/'checkpoints/update_000188000'
require(pin(entry['DEV']['path'])==entry['DEV']and pin(entry['frontier_preservation']['path'])==entry['frontier_preservation']and pin(CK/'pointer.json')==entry['checkpoint_pointer'],'Exact periodic record/source pointers')
require(report['passed']and report['optimizer_updates']==188000 and report['actual_native_updates_in_DEV']==0 and report['all207_threeAdam_all4RNG_unchanged']and not report['TEST_used']and not report['retention_early_stop'],'Actual unchanged full350/no-native/TESTfalse/earlystopfalse')
for k in ('all207_threeAdam_all4RNG_unchanged','both_cursors_journal_and_nextdraw_unchanged','fullpass_and_winner_state_unchanged_here'):require(preservation[k]is True,'Exact actual preservation '+k)
require(preservation['actual_native_updates_in_DEV']==0 and preservation['actual_sampler_or_replay_advance']is False and preservation['optimizer_updates']==188000,'Actual unchanged frontier')
require(entry['periodic_record_advances_full_pass_or_winner']is False and entry['shared_with_full_pass_DEV']is False,'Periodic only; no invented pass boundary')
predictions={};counts={}
for lang,p in report['predictions'].items():
 require(pin(p['path'])==p,'Exact genuine prediction file '+lang)
 rows=[json.loads(line)for line in Path(p['path']).read_text(encoding='utf-8').splitlines()if line];require(len(rows)==50,'Exactly50 genuine '+lang)
 predictions[lang]=p;counts[lang]=len(rows);del rows;guard()
require(set(predictions)==set(('tr','en','es','it','fr','de','pt'))and entry['score']['full350_DEV_complete'],'Exact7x50 DEV')
persistence=read(CK/'persistence_verified.json');require(persistence['passed']and persistence['optimizer_updates']==188000 and persistence['named_tensors_verified']==854 and persistence['weights_verified']==207 and persistence['Adam_states_verified']==215 and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256'],'Full854/207/215 PRE-SAVED persistence')
qpath=I/'CPU_periodic_000188000_quiescence.json';q=read(qpath);closed=q['CPU_actual0']
drained=q['private_packets_discarded']
require(type(drained)is int and 0<drained<=20,'Actual bounded unconsumed immediate successor drain')
expected_native=188000-fullpass_binding['cohort_start_optimizer_updates']
expected_completed=expected_native+drained
require(expected_native==fullpass_binding['expected_native_updates_to_periodic']
 and drained==fullpass_binding['expected_private_packets_drained']
 and expected_completed==fullpass_binding['expected_total_preparations'],
 'Counts derived from genuine cohort frontier and actual drain, independently bound by ROOT')
require(q['optimizer_updates']==188000 and q['private_packets_discarded']==drained and q['actual_native_updates']==0 and q['authoritative_sampler_replay_or_journal_advance']is False and q['private_plan_regenerated_or_reordered']is False and q['full_prefetch_and_DEV_reservations_unchanged']is True and q['only_original_two_carried_RNG_representation_paths_normalized']is True,'Real original periodic quiescence/unchanged authoritative state')
require(closed['actual_Popen_wait_code']==0 and closed['transport_threads_joined']is True and closed['pipe_handles_closed']is True and closed['completed_updates']==expected_completed,'Genuine source-derived CPU cohort plus bounded drain/wait0/threads+handles closed')
require(q['unconsumed_private_plan']==preservation['pending_private_plan_retained']and pin(q['unconsumed_private_plan']['path'])==q['unconsumed_private_plan'],'Same exact original pending plan retained')
credit_child=closed['packet_credit'];credit_parent=closed['packet_credit_owner']
require(credit_child['all_owned_kernel_handles_closed']is True and credit_parent['all_owned_kernel_handles_closed']is True and credit_child['remaining_owned_handle_count']==credit_parent['remaining_owned_handle_count']==0,'Actual child and parent named-credit handles closed after wait and transport joins')
require(credit_child['successful_preparation_acquires']==credit_parent['successful_release_posts']==closed['completed_updates']==expected_completed,'Exactly one acquire and release for every real original CPU prepared packet including actual bounded drain')
require(credit_parent['descriptor']==credit_child['descriptor']and credit_parent['descriptor']['logical_packet_maximum']==2 and closed['maximum_live_receiver_packets']==1 and closed['peak_live_receiver_packets']==1,'Actual max2 logical permits/oneLIVEreceiver current production phase')
require(credit_parent['last_release_previous_count']==1 and sum(credit_parent['release_previous_count_histogram'].values())==expected_completed and not credit_parent['cancelled'],'Final actual return1to2 with no missing or cancellation posts')
require(closed['queue_capacity_bytes']==268435456 and closed['source']['protocol']==pin(S/'fastmss_resident_feature_stream_v14.py'),'Original fullqueue and exact CPU14 source')
who_old=closed['producer_identity']
require(who_old==fullpass_binding['pre_periodic_CPU'],
 'Periodic closure belongs to the exact genuinely admitted post-fullpass cohort')
require(not exact_alive(fullpass_binding['pre_periodic_CPU_launcher']),
 'Exact post-fullpass CPU launcher dead after original normal wait0')
try:old_now=psutil.Process(who_old['pid']).create_time();old_dead=abs(old_now-who_old['create_time'])>=.001
except psutil.NoSuchProcess:old_dead=True
require(old_dead,'Actual old CPU exact identity dead after wait0')
newready_path=I/'CPU_after_periodic_DEV_000188000/actual_CPU_ready.json';newready=read(newready_path);CPU=newready['producer_identity']
require(abs(psutil.Process(CPU['pid']).create_time()-CPU['create_time'])<.001,'Actual postDEV newCPU alive')
raw=(B/'progress/latest.json').read_bytes();progress=json.loads(raw);native=progress['producer'];require(native==read(I/'boot.json')['producer']and progress['actual_optimizer_updates']>188000 and not progress['optimizer_poisoned']and not progress['schedule_poisoned']and abs(psutil.Process(native['pid']).create_time()-native['create_time'])<.001,'Same actual healthy native and resumedpositive')
require(progress['actual_optimizer_updates']>188200,'Actual post188200 positive training')
nextCK=B/'checkpoints/update_000188200';nextpersistence=read(nextCK/'persistence_verified.json')
require(nextpersistence['passed']and nextpersistence['optimizer_updates']==188200 and nextpersistence['named_tensors_verified']==854 and nextpersistence['weights_verified']==207 and nextpersistence['Adam_states_verified']==215 and nextpersistence['independent_pre_save_state_sha256']==nextpersistence['saved_state_sha256'],'Next standard188200 full854/207/215 PRE-SAVED persistence')
require(persistence['saved_semantic_state_verified']['passed']is True and nextpersistence['saved_semantic_state_verified']['passed']is True,'Both natural checkpoints genuine semantic preservation')
require(progress['producer']==read(I/'boot.json')['producer']and native['pid']==19320 and abs(native['create_time']-1792803517.7236164)<.001 and progress['source']==pin(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v37.py')and progress['planned_native_updates']==401072 and progress['administrative_update_stop']is None and not(I/'actual_failure.json').exists(),'Same healthy admitted source37 full401072')
require(newready['packet_credit']['logical_packet_maximum']==2,'PostDEV newCPU really uses max2 named credit')
observation=A/'actual_periodic188000_postpositive_progress.json';require(not observation.exists(),'One positive snapshot');observation.write_bytes(raw)
status={'kind':'ASR18_actual_periodic188000_full350_quiescence_and_resumed_training_v1','passed':True,'owned_session':26493,'source':progress['source'],'native':native,'current_CPU':CPU,'actual_optimizer_updates':progress['actual_optimizer_updates'],'native_updates_this_process':progress['native_updates_this_process'],'periodic_optimizer_updates':188000,
 'evidence':{'DEV':entry['DEV'],'preservation':entry['frontier_preservation'],'pointer':entry['checkpoint_pointer'],'persistence':pin(CK/'persistence_verified.json'),'pending_plan':q['unconsumed_private_plan'],'post_positive_progress':pin(observation)},'standard_next188200_checkpoint':{'pointer':pin(nextCK/'pointer.json'),'persistence':pin(nextCK/'persistence_verified.json'),'optimizer_updates':188200,'full854_207_215_PRE_SAVED_passed':True},'prediction_files':predictions,'prediction_counts':counts,'score':entry['score'],'retention_cap_misses':{k:entry['score']['word_errors'][k]-limit for k,limit in entry['score']['retention_limits'].items()if entry['score']['word_errors'][k]>limit},
 'TR_change_vs184000':{'previous_errors':previous_errors,'current_errors':entry['score']['word_errors']['tr'],'reference_words':825,'error_change':entry['score']['word_errors']['tr']-previous_errors,'percentage_point_change':(entry['score']['word_errors']['tr']-previous_errors)/825*100},'all207_threeAdam_fourRNG_bothcursors_journal_nextdraw_and_pendingplan_preserved':True,'actual_DEV_native_updates':0,'fullpass_or_winner_advance':False,'TEST_used':False,'retention_early_stop':False,
 'actual_normal_periodic_helper3_CPU14_max2_phase':{'receipt':pin(qpath),'private_packets_drained_and_released':drained,'actual_cohort_native_updates':expected_native,'actual_cohort_start_optimizer_updates':fullpass_binding['cohort_start_optimizer_updates'],'CPU_actual_Popen_wait_code':0,'CPU_completed_updates':closed['completed_updates'],'transport_threads_joined':True,'pipe_handles_closed':True,'original_CPU':who_old,'original_CPU_observed_exact_dead':True,'original_CPU_launcher':fullpass_binding['pre_periodic_CPU_launcher'],'original_CPU_launcher_observed_exact_dead':True,'same_pending_plan_retained':True,'only_two_original_RNG_representation_paths_normalized':True,'authoritative_sampler_replay_journal_advance':False,'private_plan_regenerated_or_reordered':False,'all_reservations_unchanged':True,'current_new_CPU_ready':pin(newready_path),'current_new_CPU':CPU,'actual_child_credit_receipt':credit_child,'actual_parent_credit_receipt':credit_parent,'maximum_logical_packets':2,'maximum_live_receiver_packets':1,'actual_derived_acquires_and_returns_including_bounded_drain':True,'expected_completed_from_actual_cohort_and_drain':expected_completed,'final_return_previous1_to_full2':True},
 'actual_fullpass_interposition':{'ROOT_review':fullpass_ROOT_pin,'boundary_optimizer_updates':187937,'pre_periodic_CPU_ready':pin(pre_periodic_ready_path),'cohort_binding':fullpass_binding},'inherited_event_consumed_no_second_GC':{'closed_boot_ROOT':boot_ROOT_pin,'inherited_receipt':inherited_pin,'no_historical_fingerprint_or_event_reaudit':True,'no_new_GC_capture_sync_or_worker_refresh':True},'actual_runtime_configuration':configuration_pin,'ROOT_execution_review':pin(args.ROOT_review),'DEV_wall_seconds':preservation['wall_seconds'],'DEV_private_peak_bytes_resident_scope':preservation['actual_phase_private']['private_peak_bytes'],'DEV_incremental_allowance_bytes':preservation['incremental_private_allowance_bytes'],'actual_DEV_admission':preservation['admission'],'full_run_target':401072,'administrative_update_stop':None,'training_continues':True,'no_extra_GPU_job_DEV_inference_export_or_study':True,'initial_metadata_full_resource_sample':pin(initial_path),'last_metadata_resource_sample':guard(),'sealer_identity':{'pid':os.getpid(),'create_time':psutil.Process().create_time(),'ppid':os.getppid()},'sealer_launcher':{'pid':os.getppid(),'create_time':psutil.Process(os.getppid()).create_time(),'ppid':psutil.Process(os.getppid()).ppid()},'metadata_sealer_source':pin(Path(__file__))}
path=A/'actual_periodic188000_and_resumed_training_status.json';require(not path.exists(),'One actual quality status');path.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({'status':pin(path),'actual_updates':progress['actual_optimizer_updates'],'current_CPU':CPU,'CPU_quiescence':pin(qpath),'score':entry['score']}),flush=True)