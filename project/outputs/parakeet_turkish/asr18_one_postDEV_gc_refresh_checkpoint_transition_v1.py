"""INACTIVE typed intentional source33 checkpoint transition for one first-periodic-DEV GC refresh.
No model/audio/NumPy imports. Real closure and final ROOT execution binding remain
required; this source alone never authorizes a stop, restore or training launch.
"""
import copy,json
from pathlib import Path
import psutil
from atomic_io_retry_v1 import read_json
from bpe_broad_audio_v1 import record,require
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
S=P/'outputs/parakeet_turkish'
ROOT=Path('E:/gpu_training_scratchpad')
F=ROOT/'analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1'
APPROVAL=F/'ROOT_direct_human_then_fix_authorization_v1.json'
SOURCE33_BASE=F.parent/'resident_RAM_original_interval170469_failure_actual_v1/production_configuration_source32_v1/configuration.json'
EVENT=F/'one_postDEV_gc_refresh_v1'
CODEC=F/'asr18_canonical_prepared_archive_v1.py'
ARCHIVE_SCOPE='ASR18_canonical_bytes_prepared_archive_policy_v1'


def exact_dead(who):
    try:return abs(psutil.Process(who['pid']).create_time()-who['create_time'])>=.001
    except psutil.NoSuchProcess:return True



def validate_intentional_optimizer_journal_gap(failure,published_journal_frontier):
    """Keep the real optimizer counter distinct from successfully published events."""
    actual=failure['actual_optimizer_updates']
    require(type(actual)is int and type(published_journal_frontier)is int
        and actual-published_journal_frontier in (0,1)
        and 'Actual owner cooperative stop'in failure['error'],
        'Only actual cooperative stop with zero or one unpublished optimizer counter advance')
    gap=actual-published_journal_frontier;trace=failure['traceback'];site='no_unpublished_optimizer_counter'
    if gap:
        journal_guard=('in commit_RAM_transaction'in trace and 'in append'in trace
            and 'self.namespace.check(); self.binding()'in trace and 'self.guard()'in trace)
        require(journal_guard,
            'One unpublished counter requires exact original commit-to-journal pre-append cooperative guard trace')
        site='journal_preappend_original_guard'
    return {'actual_optimizer_frontier':actual,'published_journal_frontier':published_journal_frontier,
        'unpublished_optimizer_counter_advance':gap,'source_backed_cooperative_stop_site':site,
        'unjournaled_objective_gradient_full_native_receipt':'UNKNOWN_UNPERSISTED'if gap else'NOT_APPLICABLE',
        'missing_journal_event_or_completed_native_receipt_fabricated':False,
        'full_checkpoint_restore_and_genuine_redo_required':True}


def checked_one_postDEV_refresh_checkpoint_resume(pin,*,base,folder,pointer_pin):
    """Only the reviewed real intentional TOOL1, exact saved state and retained tail."""
    folder=Path(folder)
    require(pin==base['prior_resident_RAM_one_postDEV_gc_refresh_checkpoint_transition']and record(pin['path'])==pin,
        'Exact actual intentional checkpoint transition closure; never a normal0 alias')
    value=read_json(pin['path'])
    require(value['kind']=='ASR18_actual_source33_checkpoint_one_postDEV_gc_refresh_transition_v1'
        and value['owned_session']==65067 and value['old_owner_invocation']==24 and value['new_owner_invocation']==25
        and value['actual_outer_TOOL_exit_code']==1 and value['normal0_claimed']is False
        and value['intentional_cooperative_stop_observed']is True,
        'Actual supported cooperative stop, outer1 and no invented normal0')
    approval=record(APPROVAL)
    require(approval['sha256']=='e02734964bd14b3e1bc6fc25d9146995233d738b80a96d6e2d0719c74d071ca6'
        and value['human_approval']==approval
        and read_json(APPROVAL)['human_message']=='then fix?',
        'Exact direct authorization to fix recurrent slowdown through a reviewed checkpoint transition')
    checked_one_postDEV_refresh_policy(base)
    previous=read_json(SOURCE33_BASE);restored=legacy_source33_configuration(base)
    require(restored==previous,'Only new GC lifecycle policy and actual intentional transition differ from proper source33 base')
    e=value['evidence']
    for name,item in e.items():require(record(item['path'])==item,'Actual compact transition evidence '+name)
    require(e['pointer']==pointer_pin and e['old_driver']==record(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v33.py')
        and e['old_wrapper']==record(P/'work/research/run_ASR18_BPE9223_resident_RAM_autocontinue_v33.py')
        and e['old_CPU']==record(S/'fastmss_resident_feature_stream_v13.py')
        and e['old_configuration']==record(folder/'invocations/owner_000024/configuration.json')
        and e['actual_failure']==record(folder/'invocations/owner_000024/actual_failure.json')
        and e['stop']==record(folder/'invocations/owner_000024/stop.json'),
        'Exact actual source33/CPU13 invocation24 and supported stop file, no historical source alias')
    stop=read_json(e['stop']['path']);intent=read_json(e['transition_intent']['path']);review=read_json(e['ROOT_pre_stop_review']['path'])
    require(review['accepted']is True and review['reviewer']=='/root'
        and review['scope']=='ASR18_one_postDEV_refresh_checkpoint_stop_v1'
        and review['transition_intent']==e['transition_intent']
        and review['human_approval']==approval and review['old_driver_source']==e['old_driver']
        and stop['kind']=='ASR18_reviewed_one_postDEV_refresh_checkpoint_cooperative_stop_v1'
        and stop['transition_intent']==e['transition_intent']and stop['ROOT_pre_stop_review']==e['ROOT_pre_stop_review']
        and intent['owned_session']==65067 and intent['old_owner_invocation']==24
        and intent['human_approval']==approval,
        'Separate exact reviewed pre-stop intent, never a metadata-created production stop')
    who=value['all_five_runtime_identities']
    require([x['role']for x in who]==['native','native_interpreter_launcher','wrapper','CPU','CPU_interpreter_launcher']
        and who==intent['expected_runtime_identities']and all(exact_dead(x)for x in who),
        'All five exact recorded runtime identities actually dead before new lease or GPU writes')
    require(who[0]['pid']==3652 and abs(who[0]['create_time']-1792764340.2396317)<.001
        and read_json(e['old_boot']['path'])['source']==e['old_driver']
        and read_json(e['old_CPU_ready']['path'])['source']['protocol']==e['old_CPU'],
        'Same admitted live owner and honest old source33/CPU13 provenance')
    failure=read_json(e['actual_failure']['path']);tool=read_json(e['actual_TOOL']['path'])
    monitor=read_json(e['monitor_completion']['path'])
    require(monitor['joined']is True and monitor['clean']is True and monitor['error']is None,
        'Only clean original Monitor cooperative stop, never concurrent abnormal guard failure')
    saved=value['last_fully_saved_optimizer_updates'];actual=value['actual_optimizer_updates']
    published=value['published_journal_optimizer_updates']
    gap=validate_intentional_optimizer_journal_gap(failure,published)
    require(gap==value['optimizer_vs_journal_frontier'],'Exact actual counter/journal distinction; no missing event fabricated')
    require(type(saved)is int and saved>=174000 and saved%200==0 and saved%4000!=0
        and type(actual)is int and saved<=published<=actual<saved+200
        and failure['last_checkpoint']==saved and failure['actual_optimizer_updates']==actual
        and 'Actual owner cooperative stop'in failure['error']
        and tool['owned_session']==65067 and tool['actual_TOOL_exit_code']==1
        and tool['normal0_claimed']is False,
        'Natural full200 checkpoint outside DEV; real stop exception and honestly preserved later attempts')
    require(stop['observed_saved_optimizer_updates']<=saved
        and stop['observed_native_updates']<=stop['observed_saved_optimizer_updates']+20
        and stop['observed_saved_optimizer_updates']%200==0,
        'Observer acted early after a natural verified publication, without claiming atomic zero-tail')
    pointer=read_json(pointer_pin['path']);persistence=read_json(e['persistence']['path'])
    require(pointer['committed_updates']==saved and read_json(folder/'latest_checkpoint.json')['pointer']==pointer_pin
        and persistence['passed']is True and persistence['optimizer_updates']==saved
        and persistence['named_tensors_verified']==854 and persistence['weights_verified']==207
        and persistence['Adam_states_verified']==215
        and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256']
        and persistence['saved_semantic_state_verified']['passed']is True,
        'Genuine full854/207/215 PRE-SAVED semantic publication, no tensor payload read here')
    prefix=pointer['journal_prefix'];stat=value['closed_old_journal_STAT'];tail=value['preserved_unsaved_positive_tail']
    require(prefix==value['saved_journal_prefix']and persistence['journal_prefix']==prefix
        and Path(prefix['path'])==folder/'journals/commits_000024.jsonl'
        and stat['path']==prefix['path']and stat['bytes']==prefix['offset']+tail['bytes']
        and record(tail['path'])==tail and Path(tail['path']).resolve().is_relative_to(F)
        and tail['bytes']<=4*1024**2,
        'Only compact preserved new tail and original24 STAT, never historical journal/prefix/whole')
    current=Path(stat['path']).stat()
    require(current.st_size==stat['bytes']and current.st_mtime_ns==stat['mtime_ns'],
        'Closed source33 whole24 remains untouched')
    events=[json.loads(line)for line in Path(tail['path']).read_bytes().splitlines()]
    require(len(events)==published-saved and all(x['optimizer_updates']==x['native']['update']==
        x['prepared_evidence']['actual_positive_native_update']==u
        for u,x in zip(range(saved+1,published+1),events)),
        'Exact ordered committed attempts remain historical and must genuinely redo')
    active=read_json(folder/'active_journal.json')
    require(Path(active['journal_prefix']['path'])==Path(prefix['path']),
        'Old24 remains authoritative until after exact restore and new prefix-only branch')
    return {**value,'saved_journal_prefix':prefix,'unsaved_tail_bytes':tail['bytes'],
        'whole_journal_including_preserved_tail':stat,'preserved_unsaved_tail':tail,
        'preserved_tail_positive_updates':list(range(saved+1,published+1))}


def legacy_source33_configuration(base):
    previous=read_json(SOURCE33_BASE);restored=copy.deepcopy(base)
    require('one_postDEV_gc_refresh_policy'in restored and
        'prior_resident_RAM_one_postDEV_gc_refresh_checkpoint_transition'in restored,
        'Both explicit new operational fields, no historical policy weakening')
    del restored['one_postDEV_gc_refresh_policy']
    del restored['prior_resident_RAM_one_postDEV_gc_refresh_checkpoint_transition']
    require(restored==previous,'Every proper source33 science/resource/CPU13/MAX2/225 value is exact')
    return restored


def checked_one_postDEV_refresh_policy(base):
    legacy_source33_configuration(base)
    policy=base['one_postDEV_gc_refresh_policy']
    require(policy=={
        'kind':'ASR18_one_first_periodic_DEV_quiescent_gc_refresh_policy_v1',
        'helper_source':record(EVENT/'asr18_one_postDEV_gc_refresh_v1.py'),
        'capture_source':record(EVENT/'asr18_postDEV_refresh_state_capture_v1.py'),
        'original_fingerprint_source':record(S/'asr18_live_state_fingerprint_v1.py'),
        'original_gc_scope_source':record(P/'work/research/run_ASR18_native_throughput_component_v15.py'),
        'original_native_source':record(S/'asr18_BPE9223_grouped_RAM_native_v1.py'),
        'original_scalar_prefetch_source':record(S/'asr18_BPE9223_private_scalar_prefetch_v1.py'),
        'preparation_scope':record(F/'ROOT_one_postDEV_gc_refresh_inactive_preparation_scope_v1.json'),
        'semantic_capture_amendment':record(F/'ROOT_one_postDEV_gc_refresh_inactive_preparation_semantic_verification_amendment_v1.json'),
        'frequency':'at_most_one_first_genuine_scheduled_periodic_DEV_per_native_process',
        'operations':['unfreeze','captured_actual_original_full_collect','freeze'],
        'scalar_join_drop_before_refresh_recreate_original_after':True,
        'exact_queued_plan_object_and_pin_retained_no_draw':True,
        'additional_bookkeeping_reserve_bytes':16777216,
        'full_event_commit_threshold_bytes':4535212672,
        'full_event_physical_threshold_bytes':1785212672,
        'maximum_original_fingerprint_transfer_slice_bytes':4194304,
        'two_added_idle_fingerprint_explicit_synchronizations':True,
        'all_original_native_perupdate_GC_sync_cache_calls_and_final_GC_teardown_unchanged':True,
        'opaque_framework_cycle_absence_proved':False,
        'completed_CPU_review':policy['completed_CPU_review']},'Exact one-event changed GC lifecycle hypothesis')
    proof=policy['completed_CPU_review']
    require(type(proof)is dict and record(proof['path'])==proof,'Genuine completed finite CPU ROOT review required')
    review=read_json(proof['path'])
    require(review['accepted']is True and review['reviewer']=='/root'and review['scope']==policy['kind']
        and review['helper_source']==policy['helper_source']and review['capture_source']==policy['capture_source']
        and record(review['actual_CPU_control_report']['path'])==review['actual_CPU_control_report']
        and review['all_exact_metadata_identities_dead']is True,
        'Real CPython/lifecycle proof and exact metadata closure; scientific GPU portion remains source/METHOD_STUB')
    import sys
    if str(EVENT)not in sys.path:sys.path.insert(0,str(EVENT))
    from asr18_one_postDEV_gc_refresh_v1 import OnePostDEVRefresh
    from asr18_postDEV_refresh_state_capture_v1 import capture_idle_state,compare_idle_state
    return OnePostDEVRefresh,capture_idle_state,compare_idle_state


def checked_archive_representation_policy(base):
    policy=base['prepared_metadata_archive_representation_policy']
    require(policy=={
        'kind':ARCHIVE_SCOPE,'helper_source':record(CODEC),
        'record_encoding':'original_canonical_UTF8_bytes_after_positive_commit',
        'checkpoint_encoding':'stream_original_canonical_JSON_array_into_original_exclusive_gzip',
        'historical_record_mutation_forbidden':True,'checkpoint_interval_updates':200,
        'native_GC_guard_and_science_unchanged':True,
        'completed_CPU_review':policy['completed_CPU_review']},'Exact immutable original-canonical archive representation policy')
    proof=policy['completed_CPU_review'];require(type(proof)is dict and record(proof['path'])==proof,
        'Actual completed ROOT codec proof required before admitted production')
    review=read_json(proof['path'])
    require(review['accepted']is True and review['reviewer']=='/root'and review['scope']==ARCHIVE_SCOPE
        and review['helper_source']==policy['helper_source']
        and record(review['actual_CPU_control_report']['path'])==review['actual_CPU_control_report']
        and review['all_exact_metadata_identities_dead']is True,'Typed completed actual codec CPU review')
    import importlib.util
    spec=importlib.util.spec_from_file_location('asr18_canonical_prepared_archive_v1',CODEC)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
