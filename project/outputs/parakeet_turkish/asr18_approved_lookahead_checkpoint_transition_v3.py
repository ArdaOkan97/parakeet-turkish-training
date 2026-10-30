"""INACTIVE typed intentional source27 checkpoint transition for approved lookahead.
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
F=ROOT/'analysis/BPE9223_training_launch_v1/resident_RAM_two_logical_packet_lookahead_v1'
APPROVAL=F.parent/'resident_RAM_source27_speed_review_v1/ROOT_direct_human_lookahead_and_checkpoint_restart_approval_v1.json'
SOURCE27_BASE=F.parent/'resident_RAM_source26_postrestore_selector_failure_actual_v1/inactive_selector_fix_v2/production_configuration_v1/configuration.json'


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


def checked_lookahead_checkpoint_resume(pin,*,base,folder,pointer_pin):
    """Only the reviewed real intentional TOOL1, exact saved state and retained tail."""
    folder=Path(folder)
    require(pin==base['prior_resident_RAM_approved_lookahead_checkpoint_transition']and record(pin['path'])==pin,
        'Exact actual intentional checkpoint transition closure; never a normal0 alias')
    value=read_json(pin['path'])
    require(value['kind']=='ASR18_actual_approved_source27_checkpoint_lookahead_transition_v1'
        and value['owned_session']==56620 and value['old_owner_invocation']==20 and value['new_owner_invocation']==21
        and value['actual_outer_TOOL_exit_code']==1 and value['normal0_claimed']is False
        and value['intentional_cooperative_stop_observed']is True,
        'Actual supported cooperative stop, outer1 and no invented normal0')
    approval=record(APPROVAL)
    require(approval['sha256']=='f92036ea0b494897d85cf15d69c8eba07fb1abfd97febcf0a18099a0d8caf9fb'
        and value['human_approval']==approval
        and read_json(APPROVAL)['answer']=='Approve lookahead and checkpoint restart',
        'Exact direct approval of logical lookahead exception and checkpoint transition')
    policy=base['resident_RAM_packet_credit_policy']
    require(policy['maximum_logical_packets']==2 and policy['human_approval']==approval
        and policy['helper_source']==record(F/'asr18_owned_packet_credit_v2.py'),
        'Only approved logical max2; byte limits,225 cushion and receiverLIVE1 retained')
    # Preserve every old field, numerical identity, byte/floor/reserve and historical lane.
    previous=read_json(SOURCE27_BASE);restored=copy.deepcopy(base)
    del restored['prior_resident_RAM_approved_lookahead_checkpoint_transition']
    restored['resident_RAM_packet_credit_policy']=previous['resident_RAM_packet_credit_policy']
    restored['resident_RAM_resource_scheduling_policy']['CPU_protocol_source']=previous['resident_RAM_resource_scheduling_policy']['CPU_protocol_source']
    require(restored==previous,'Only new credit policy/CPU source and actual transition provenance differ from source27 base')
    e=value['evidence']
    for name,item in e.items():require(record(item['path'])==item,'Actual compact transition evidence '+name)
    require(e['pointer']==pointer_pin and e['old_driver']==record(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v27.py')
        and e['old_wrapper']==record(P/'work/research/run_ASR18_BPE9223_resident_RAM_autocontinue_v27.py')
        and e['old_CPU']==record(S/'fastmss_resident_feature_stream_v10.py')
        and e['old_configuration']==record(folder/'invocations/owner_000020/configuration.json')
        and e['actual_failure']==record(folder/'invocations/owner_000020/actual_failure.json')
        and e['stop']==record(folder/'invocations/owner_000020/stop.json'),
        'Exact actual source27/CPU10 invocation20 and supported stop file, no historical source alias')
    stop=read_json(e['stop']['path']);intent=read_json(e['transition_intent']['path']);review=read_json(e['ROOT_pre_stop_review']['path'])
    require(review['accepted']is True and review['reviewer']=='/root'
        and review['scope']=='ASR18_approved_two_logical_packet_checkpoint_stop_v1'
        and review['transition_intent']==e['transition_intent']
        and review['human_approval']==approval and review['old_driver_source']==e['old_driver']
        and stop['kind']=='ASR18_reviewed_approved_lookahead_checkpoint_cooperative_stop_v1'
        and stop['transition_intent']==e['transition_intent']and stop['ROOT_pre_stop_review']==e['ROOT_pre_stop_review']
        and intent['owned_session']==56620 and intent['old_owner_invocation']==20
        and intent['human_approval']==approval,
        'Separate exact reviewed pre-stop intent, never a metadata-created production stop')
    who=value['all_five_runtime_identities']
    require([x['role']for x in who]==['native','native_interpreter_launcher','wrapper','CPU','CPU_interpreter_launcher']
        and who==intent['expected_runtime_identities']and all(exact_dead(x)for x in who),
        'All five exact recorded runtime identities actually dead before new lease or GPU writes')
    require(who[0]['pid']==9704 and abs(who[0]['create_time']-1792727133.7017913)<.001
        and read_json(e['old_boot']['path'])['source']==e['old_driver']
        and read_json(e['old_CPU_ready']['path'])['source']['protocol']==e['old_CPU'],
        'Same admitted live owner and honest old source27/CPU10 provenance')
    failure=read_json(e['actual_failure']['path']);tool=read_json(e['actual_TOOL']['path'])
    saved=value['last_fully_saved_optimizer_updates'];actual=value['actual_optimizer_updates']
    published=value['published_journal_optimizer_updates']
    gap=validate_intentional_optimizer_journal_gap(failure,published)
    require(gap==value['optimizer_vs_journal_frontier'],'Exact actual counter/journal distinction; no missing event fabricated')
    require(type(saved)is int and saved>=157000 and saved%200==0 and saved%4000!=0
        and type(actual)is int and saved<=published<=actual<saved+200
        and failure['last_checkpoint']==saved and failure['actual_optimizer_updates']==actual
        and 'Actual owner cooperative stop'in failure['error']
        and tool['owned_session']==56620 and tool['actual_TOOL_exit_code']==1
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
        and Path(prefix['path'])==folder/'journals/commits_000020.jsonl'
        and stat['path']==prefix['path']and stat['bytes']==prefix['offset']+tail['bytes']
        and record(tail['path'])==tail and Path(tail['path']).resolve().is_relative_to(F)
        and tail['bytes']<=4*1024**2,
        'Only compact preserved new tail and original20 STAT, never historical18/19 or old whole journal')
    current=Path(stat['path']).stat()
    require(current.st_size==stat['bytes']and current.st_mtime_ns==stat['mtime_ns'],
        'Closed source27 whole20 remains untouched')
    events=[json.loads(line)for line in Path(tail['path']).read_bytes().splitlines()]
    require(len(events)==published-saved and all(x['optimizer_updates']==x['native']['update']==
        x['prepared_evidence']['actual_positive_native_update']==u
        for u,x in zip(range(saved+1,published+1),events)),
        'Exact ordered committed attempts remain historical and must genuinely redo')
    active=read_json(folder/'active_journal.json')
    require(Path(active['journal_prefix']['path'])==Path(prefix['path']),
        'Old20 remains authoritative until after exact restore and new prefix-only branch')
    return {**value,'saved_journal_prefix':prefix,'unsaved_tail_bytes':tail['bytes'],
        'whole_journal_including_preserved_tail':stat,'preserved_unsaved_tail':tail,
        'preserved_tail_positive_updates':list(range(saved+1,published+1))}
