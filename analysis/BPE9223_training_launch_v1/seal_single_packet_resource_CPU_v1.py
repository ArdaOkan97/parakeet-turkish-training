"""Metadata-only finite CPU closure and inactive exact continuation handoff."""
import ast,copy,datetime,hashlib,json,sys
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
N=Path(__file__).parent;D=N/'resident_RAM_single_packet_resource_source_v1';A=N/'resident_RAM_single_packet_resource_CPU_actual_v2'
sys.path[:0]=[str(S),str(R)]
def pin(p):
    p=Path(p).resolve();raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def require(ok,msg):
    if not ok:raise ValueError(msg)
def write(p,v):require(not Path(p).exists(),'Fresh metadata receipt');Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def dead(who):
    try:now=psutil.Process(who['pid']).create_time()
    except psutil.NoSuchProcess:now=None
    gone=now is None or abs(now-who['create_time'])>=.001
    require(gone,'Original fixture identity must actually be dead')
    return {'original_identity':who,'observed_current_create_time':now,'original_identity_dead':gone,'observed_pid_absent':now is None}
handoff=read(D/'source_handoff_v1.json');report=read(A/'report.json');completion=read(A/'actual_fixture_completion.json')
require(report['passed']and report['actual_optimizer_updates']==report['actual_draws']==0
    and report['audio_or_model_loaded']is False and report['GPU_used']is False
    and completion['observer_joined']and completion['observer_error']is None and completion['abnormal_cleanup']==[],
    'Real finite CPU0 controls and joined observer, no science work')
deaths=[dead(completion['fixture_identity'])];identities={}
for item in A.glob('*/actual_child_identity.json'):
    data=read(item)
    for key in ('launcher_identity','actual_worker_identity'):
        who=data[key];identities[(who['pid'],who['create_time'])]=who
for who in report['family_at_peak']:identities[(who['pid'],who['create_time'])]={k:who[k]for k in ('pid','create_time')}
deaths.extend(dead(w)for w in identities.values());write(A/'actual_closed_CPU_identity_observation.json',{'identities':deaths,
    'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exact_recorded_identities_only':True})
import run_ASR18_BPE9223_resident_RAM_training_v15 as driver
config=read(handoff['configuration']['path']);failure=read(config['prior_resident_RAM_headroom_failure']['path'])
pointer=failure['source_evidence']['saved_pointer'];branch=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01')
checked=driver.checked_headroom_resume(config['prior_resident_RAM_headroom_failure'],base=config,folder=branch,pointer_pin=pointer)
require(checked['saved_optimizer_updates']==133000 and len(checked['preserved_tail_positive_updates'])==137,'Serialized exact typed abnormal gate')
prior=read(handoff['prior_configuration']['path']);inverse=copy.deepcopy(config);inverse.pop('resident_RAM_resource_scheduling_policy')
inverse['prior_resident_RAM_headroom_failure']=prior['prior_resident_RAM_headroom_failure'];require(inverse==prior,'Only two operational fields')
base_sources={
    'RAM_native':S/'asr18_BPE9223_grouped_RAM_native_v1.py',
    'DEV_helper':S/'asr18_resident_serial350_DEV_v2.py',
    'prefetch':S/'asr18_BPE9223_private_scalar_prefetch_v1.py'}
native_pin=pin(base_sources['RAM_native']);require(native_pin['sha256']=='e4be25460f8958e918430837764347273c9288778d06edb016c44e7acb606085','Original numerical native bytes unchanged')
cpu_receipt={'kind':'ASR18_single_live_packet_resource_actual_CPU_closure_receipt_v1','actual_exec_TOOL_exit_code':0,
    'actual_exec_TOOL_chunk_id':'b44789','source':pin(N/'verify_single_live_packet_resource_CPU_v2.py'),
    'report':pin(A/'report.json'),'completion':pin(A/'actual_fixture_completion.json'),
    'actual_dead_identities':pin(A/'actual_closed_CPU_identity_observation.json'),
    'prior_fixture1_actual_exec_TOOL_exit_code':1,'prior_fixture1_actual_exec_TOOL_chunk_id':'460e18',
    'prior_fixture1_failure':'ModuleNotFoundError: numpy before protocol use; fixed only fixture production CPU NumPy initialization',
    'prior_fixture1_source':pin(N/'verify_single_live_packet_resource_CPU_v1.py'),
    'prior_fixture1_completion':pin(N/'resident_RAM_single_packet_resource_CPU_actual_v1/actual_fixture_completion.json'),
    'elapsed_seconds':report['elapsed_seconds'],'family_peak_private_bytes':report['family_peak_private_bytes'],
    'full_original_256MiB_packet_accepted':True,'single_live_including_delivered_unreleased':True,
    'order_release_backpressure_and_exception_cleanup_passed':True,'production_audio_or_GPU_validation_claimed':False,
    'historical_queue_cause_or_measured_saving_claimed':False,'production_admitted':False}
write(A/'actual_CPU0_closure_receipt.json',cpu_receipt)
output=N/'resident_RAM_training_actual_12';inv=branch/'invocations/owner_000011';newjournal=branch/'journals/commits_000011.jsonl'
require(not output.exists()and not inv.exists()and not newjournal.exists(),'No new production invocation/journal exists')
review_expected={
    'accepted':True,'reviewer':'/root','driver_source':handoff['sources']['driver'],
    'wrapper_source':handoff['sources']['wrapper'],'RAM_native_source':native_pin,
    'CPU_source':handoff['sources']['CPU'],'DEV_helper_source':pin(base_sources['DEV_helper']),
    'source_configuration':handoff['configuration'],'initial_checkpoint':pointer,
    'latest_failure':failure['source_evidence']['actual_failure'],
    'private_scalar_prefetch_source':pin(base_sources['prefetch']),
    'private_scalar_prefetch_actual_CPU_review':config['private_scalar_prefetch_policy']['actual_CPU_control_review'],
    'additional_prefetch_host_reservation_bytes':268435456,'bounded_RAM_queue_bytes':268435456,
    'periodic_DEV_interval_updates':4000,'initial_owner_invocation':11,
    'headroom_resume':config['prior_resident_RAM_headroom_failure'],
    'resource_scheduling_policy':config['resident_RAM_resource_scheduling_policy']}
arguments=['--review','<ROOT final exact execution review required>',
    '--configuration',handoff['configuration']['path'],'--pointer',pointer['path'],
    '--latest-failure',failure['source_evidence']['actual_failure']['path'],'--branch',str(branch),
    '--output',str(output),'--queue-bytes','268435456','--start-invocation','11',
    '--abnormal-resume',config['prior_resident_RAM_headroom_failure']['path']]
final={'kind':'ASR18_INACTIVE_single_live_packet_saved133000_final_review_handoff_v1','production_admitted':False,
    'source_handoff':pin(D/'source_handoff_v1.json'),'sources':handoff['sources'],'configuration':handoff['configuration'],
    'actual_CPU0_closure':pin(A/'actual_CPU0_closure_receipt.json'),'actual_CPU_report':pin(A/'report.json'),
    'actual_failure':config['prior_resident_RAM_headroom_failure'],'saved_checkpoint':pointer,
    'failure_ROOT_evidence_review':handoff['ROOT_failure_evidence_review'],'unchanged_sources':{k:pin(v)for k,v in base_sources.items()},
    'serialized_typed_resume_gate_passed':True,'saved_prefix':checked['saved_journal_prefix'],
    'historical137_positive_tail':checked['preserved_unsaved_tail'],'old_whole_journal_not_reread_by_seal':True,
    'future_invocation':11,'future_new_journal':str(newjournal),'future_output':str(output),
    'future_only_saved_prefix_copy_AFTER_genuine_restore':True,'no_tail_replay':True,
    'ROOT_execution_review_required_fields':review_expected,'exact_future_wrapper_arguments':arguments,
    'full401072_adminNone_and_all_guards_reservations_unchanged':True,
    'new_resource_profile_is_changed_workload_not_proved_historical_cause_or_measured_saving':True,
    'timing14_and_interval_preflight_remain_inactive':True}
write(D/'final_execution_handoff_v1.json',final)
print(json.dumps({'CPU_closure':pin(A/'actual_CPU0_closure_receipt.json'),
    'final_handoff':pin(D/'final_execution_handoff_v1.json')}),flush=True)
