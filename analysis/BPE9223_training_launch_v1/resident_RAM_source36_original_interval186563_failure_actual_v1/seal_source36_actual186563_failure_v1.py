"""One bounded metadata-only failure/tail preservation; no decode, restore or launch."""
import datetime,hashlib,json,math,os,shutil,sys,time,traceback
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent;N=F.parent
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000025';C=I/'CPU_after_periodic_DEV_000184000';CK=B/'checkpoints/update_000186400'
BEGIN=time.monotonic();ME=psutil.Process();ME.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
RESULT=F/'actual_source36_failure186563_saved186400_closure_v1.json'

def require(v,m):
 if not v:raise ValueError(m)

def read(p):
 require(p.stat().st_size<=8*1024**2,'Bounded current metadata/source only');return json.loads(p.read_text(encoding='utf-8-sig'))

def pin(p):
 require(p.stat().st_size<=8*1024**2,'Compact/source/evidence pins only');raw=p.read_bytes();return {'path':str(p.resolve()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def write(p,value):
 with p.open('x',encoding='utf-8',newline='\n')as out:json.dump(value,out,indent=2,ensure_ascii=False);out.write('\n');out.flush();os.fsync(out.fileno())

def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}

def exact(v):
 try:return abs(psutil.Process(v['pid']).create_time()-v['create_time'])<.001
 except psutil.NoSuchProcess:return False

def guard():
 samples=[sample_memory(p.pid,'source36_actual186563_failure_metadata_seal')for p in(ME,ME.parent())]
 m=samples[-1]['system_memory'];private=sum(v['worker_private_memory']['private_bytes']for v in samples)
 require(all(s['paired_sampling_within_one_second']for s in samples)and private<=536870912 and m['commit_available_bytes']>=5055306368 and m['physical_available_bytes']>=2305306368 and shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-BEGIN<180,'Original512MiB family/fullprospective512/full225/FULLprefetch/floors/E100GB/180s metadata bounds')
 return {'samples':samples,'sampled_pair_private_bytes':private}

def main():
 initial=guard();require(not RESULT.exists(),'Fresh immutable actual failure closure')
 failure=read(I/'actual_failure.json');cpu_failure=read(C/'actual_CPU_failure.json');monitor=read(I/'monitor_completion.json');worker=read(I/'private_metadata_worker_completion.json');progress=read(B/'progress/latest.json');latest=read(B/'latest_checkpoint.json');pointer=read(CK/'pointer.json')
 require(failure['actual_optimizer_updates']==186563 and failure['last_checkpoint']==186400 and failure['error']=="ValueError('Never release credit from errored transport')",'Exactactual parent releasefailure/frontiers')
 require('line 1149' in failure['traceback']and 'stream.release(token)'in failure['traceback'],'Original aftercommit/archive release traceback')
 require(cpu_failure['kind']=='error'and cpu_failure['completed_updates']==2563 and cpu_failure['message']=="ValueError('Original interval beyond source unresolved')",'Actual current CPU primary preparationerror persisted separately')
 require(monitor['joined']is True and monitor['clean']is True and monitor['error']is None,'Genuine original Monitor joinedclean; no resourcecause invention')
 require(latest['committed_updates']==186400 and latest['full_state_verified']is True and pin(CK/'pointer.json')==latest['pointer']and pin(CK/'persistence_verified.json')==latest['persistence'],'Currentlatest fullsaved186400 pointers exact; closed ROOTsemantic audit reused')
 saved=pointer['journal_prefix'];journal=Path(saved['path']);require(saved['offset']==375720854 and saved['sha256']=='c658c5c1711f0e43b3b764e0a8875e90cdd5137f7d27469dd28936802fd02e80','Published savedprefix binding only, NO prefixread/hash')
 identities=[{'role':'native','pid':21912,'create_time':1792780981.1092248,'ppid':37108},{'role':'native_launcher','pid':37108,'create_time':1792780981.1024935,'ppid':29600},{'role':'wrapper','pid':29600,'create_time':1792780979.698954,'ppid':40472},{'role':'CPU13','pid':39856,'create_time':1792790240.3032615,'ppid':35440},{'role':'CPU_launcher','pid':35440,'create_time':1792790240.2869232,'ppid':21912}]
 deaths=[{'identity':v,'exact_dead':not exact(v)}for v in identities];require(all(v['exact_dead']for v in deaths),'Allfive originalowned production exactdead')
 before=journal.stat();count=before.st_size-saved['offset'];require(0<count<=2*1024**2,'ONLY fresh unsaved suffix <=2MiB, never wholejournal')
 tail_path=F/'preserved_new186401_186563_unsaved_journal_tail.jsonl';require(not tail_path.exists(),'One suffix snapshot only')
 with journal.open('rb')as src:
  src.seek(saved['offset']);raw=src.read(count);require(len(raw)==count,'Exactly declared NEWsuffix bytes')
 with tail_path.open('xb')as out:out.write(raw);out.flush();os.fsync(out.fileno())
 after=journal.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Closed originaljournal unchanged during single suffixread')
 require(raw.endswith(b'\n'),'Completepublished suffix, no fabricated/truncated event')
 events=[json.loads(line)for line in raw.splitlines()];updates=[e['optimizer_updates']for e in events];require(updates==list(range(186401,186564)),'Exactly163 newlypublished historical positives')
 tx=[e['transaction_sha256']for e in events];tokens=[e['prepared_evidence']['token']for e in events]
 require(len(set(tx))==len(set(tokens))==163,'163 distinct genuine transaction/token records')
 for e in events:
  u=e['optimizer_updates'];native=e['native'];require(native['update']==u and math.isfinite(native['mean_objective'])and native['mean_objective']>0 and math.isfinite(native['global_grad_norm']),'Recorded positive native/full finite objective+gradient')
  require(e['prepared_evidence']['producer_source_sha256']=='abcf2552fcd3bfdd7624ab16dc6a70eaa2fce3beab3b54fb2b642213e373b263'and e['prepared_evidence']['token'].startswith('39856:'),'CurrentactualCPU13/source+token')
  require(len(native['source_ids'])==len(set(native['source_ids']))==16 and native['languages'][:14]==['tr']*14 and len(native['languages'])==16 and native['languages'][-1]=='pt','Full16IDs/14TR2replay actual records')
  require(e['broad_draws']==14*u and e['replay_updates']==u and e['cleaner_draws']==0,'Absolute original bothsampler clocks')
 published=updates[-1];gap=failure['actual_optimizer_updates']-published;require(gap==0,'This actualreleaseaftercommit trace and suffix establish gap0')
 reader_id={'pid':26892,'create_time':1792652904.3359847,'ppid':30128};reader_launcher={'pid':30128,'create_time':1792652904.283631,'ppid':10644};require(exact(reader_id)and exact(reader_launcher),'Originalreader pair healthy unchanged')
 reader_path=Path('E:/gpu_training_scratchpad/analysis/BPE9223_Trackio_live_v1/current_status.json');reader=read(reader_path)
 lease_path=P/'work/gpu_training_owner.json';lease=read(lease_path)
 standard_ROOT=N/'resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/ROOT_source36_standard186400_metadata_and_PRE_SAVED_equality_verification_v1.json'
 require(pin(standard_ROOT)['sha256']=='c1358e5d7f6acc52ae1cb7d3564046da15984855e179887615d13caceef19801','Reuse closedROOT186400standard receipt only, no payloadaudit rerun')
 inputs=[I/'actual_failure.json',C/'actual_CPU_failure.json',C/'actual_CPU_ready.json',C/'CPU_stderr.log',I/'monitor_completion.json',I/'private_metadata_worker_completion.json',B/'progress/latest.json',B/'latest_checkpoint.json',B/'active_journal.json',CK/'pointer.json',CK/'persistence_verified.json',standard_ROOT,N/'resident_RAM_training_actual_26/current_child.json',F/'actual_owned42997_terminal_TOOL1_v1.json',Path(__file__)]
 payload=Path(pointer['payload']['path']);payload_stat=payload.stat();require(payload_stat.st_size==pointer['payload']['bytes'],'PayloadSTAT only size preserved; NO payloadread/hash/GPUrestore')
 value={'kind':'ASR18_source36_actual186563_saved186400_original_interval_abnormal_closure_v1','passed':True,'observed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':pin(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v36.py'),'CPU_source':pin(S/'fastmss_resident_feature_stream_v13.py'),'actual_parent_failure':pin(I/'actual_failure.json'),'actual_CPU_primary_failure':pin(C/'actual_CPU_failure.json'),'recorded_primary_preparation_error':cpu_failure['message'],'actual_parent_propagated_release_error':failure['error'],'transport_primary_error_source':'CPU13_receive528-536 persists errorframe then _fail453-459; release586 sees error. Parent abort621 overwrites transient self.error; persistent CPUfailure/stderr retain original.', 'cleanup_masking_error':'No separate actual abort-error file exists; parent failure is original release guard. Underlying CPUcause is separately recorded, not guessed fromrelease.','actual_optimizer_updates':186563,'last_checkpoint_updates':186400,'last_emitted_progress_updates':progress['actual_optimizer_updates'],'published_journal_frontier':published,'actual_optimizer_minus_published_gap':gap,'all_live_unsaved_updates_require_coherent_full_restore_and_genuine_redo':True,'historical_unsaved_positive_count':163,'saved_pointer':pin(CK/'pointer.json'),'saved_fullstate_ROOT_review':pin(standard_ROOT),'saved_prefix_PUBLICATION_PIN_NOT_REREAD':saved,'original_journal_STAT_ONLY':{'path':str(journal),'bytes':before.st_size,'mtime_ns':before.st_mtime_ns},'preserved_unsaved_tail':pin(tail_path),'tail_range':{'start_offset':saved['offset'],'end_offset':before.st_size,'bytes':count,'first_optimizer_update':updates[0],'last_optimizer_update':published,'count':len(events),'source_original_read_count':1,'all163_native_positive_finite_currentCPU13_tokens_and_absolute_cursors':True},'source_bound_candidate':'PENDING_SEPARATE_METADATA_LOCALIZATION','exact_failed_row':'UNKNOWN','physical_interval_cause':'UNKNOWN','separate_numeric_OS_exits':{'native':'UNRECORDED_UNKNOWN','CPU':'UNRECORDED_UNKNOWN','wrapper':'UNRECORDED_UNKNOWN'},'owned_TOOL_result':pin(F/'actual_owned42997_terminal_TOOL1_v1.json'),'all5_exact_deaths':deaths,'normal_CPU0_or_creditdrain_claimed':False,'actual_transport_join_pipe_kernelhandle_count_receipts':'UNRECORDED; abort code has join/close checks and no CPU_abort_error file, but no new normal/abnormal return receipt was persisted. Not inferred from death.','CPU_abort_error_file_absent':not(I/'CPU_abort_error.json').exists(),'original_Monitor':monitor,'scalar_worker_completion':worker,'last_recorded_CPU_resource_telemetry':cpu_failure['actual_resource_telemetry'],'no_recorded_allocation_wait_or_resource_denial_in_CPU_error':cpu_failure['actual_resource_telemetry']['allocation_wait_count']==0,'current_lease_compact':lease,'reader_identity':reader_id,'reader_launcher_identity':reader_launcher,'reader_current_status_capture':reader,'reader_expected_workerfalse_no_restart':True,'payload_STAT_ONLY':{'path':str(payload),'bytes':payload_stat.st_size,'mtime_ns':payload_stat.st_mtime_ns,'read_or_hash_or_GPUrestore':False},'input_pins':{p.name:pin(p)for p in inputs},'initial_metadata_guard':initial,'final_metadata_guard':guard(),'metadata_identity':who(ME),'metadata_launcher_identity':who(ME.parent()),'metadata_pair_alive_at_write_requires_exactdeath_after_actualTOOL':True,'no_decoder_baseline_fullmember_frontend_model_GPU_state_GC_LR_source_or_guard_change':True,'future_recovery_NOT_admitted':True,'timing538fda_proposal_HELD':True}
 write(RESULT,value)
 print(json.dumps({'closure':pin(RESULT),'tail':value['preserved_unsaved_tail'],'tail_range':value['tail_range'],'actual_optimizer_updates':186563,'published':published,'gap':gap,'metadata_identity':value['metadata_identity'],'metadata_launcher_identity':value['metadata_launcher_identity']}),flush=True)

if __name__=='__main__':
 try:main()
 except BaseException as e:
  failure={'kind':'ASR18_source36_actual186563_metadata_sealer_failed_v1','passed':False,'error':{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()},'metadata_identity':who(ME),'metadata_launcher_identity':who(ME.parent()),'source_original_suffix_may_already_have_been_preserved_no_automatic_reread_or_rerun':True}
  out=F/'actual_metadata_sealer_failure_v1.json'
  if not out.exists():write(out,failure)
  print(json.dumps({'failure':pin(out),'error':failure['error']}),flush=True);raise
