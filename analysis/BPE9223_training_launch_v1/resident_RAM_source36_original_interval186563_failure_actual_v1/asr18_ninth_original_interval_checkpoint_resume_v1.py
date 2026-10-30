"""INACTIVE exact source36 interval failure / ninth-file / consumed180k transition.

Only source and compact accepted ROOT metadata are read. Closed journal suffix,
event, physical/ingress reports, tensor/audio payloads and old studies are not
re-audited. Actual full-state restore and genuine redo remain production duties.
"""
import copy
from pathlib import Path
import sys
import threading

import psutil
from atomic_io_retry_v1 import read_json
from bpe_broad_audio_v1 import record, require

P = Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
S = P / 'outputs/parakeet_turkish'
ROOT = Path('E:/gpu_training_scratchpad')
BRANCH = ROOT / 'runs/BPE9223_resident_RAM_20261018_01'
F = Path(__file__).resolve().parent
BASE = BRANCH / 'invocations/owner_000025/configuration.json'
BASE_SHA = 'f866afb3e9cd7e8db15c341e7df5ecbc8c6cc7d6714153783c389fafc54498a7'
CLOSURE = F / 'actual_source36_failure186563_saved186400_closure_v1.json'
ROOT_FAILURE = F / 'ROOT_source36_failure186563_saved186400_actual_independent_verification_v1.json'
ROOT_CPU = F / 'ROOT_ninth_CPU14_actual01_original_ingress_frontend_independent_verification_v1.json'
ROOT_TRANSPORT = F / 'ROOT_CPU14_one_overlay_import_only_transport_compatibility_successor_v1.json'
CERTIFICATE = F / 'ninth_original_full_clip_geometry_certificate_v1.json'
EVENT_FOLDER = F.parent / 'resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1'
ROOT_STANDARD = EVENT_FOLDER / 'ROOT_source36_standard186400_metadata_and_PRE_SAVED_equality_verification_v1.json'
ROOT_EVENT = EVENT_FOLDER / 'ROOT_source36_actual_first_periodic180000_idle_refresh_preservation_verification_v1.json'
ROOT_QUALITY = F.parent / 'ROOT_periodic180000_full350_source36_idle_refresh_CPU13_max2_normal_saved180200_and_resumed_training_verification_v1.json'
FAILURE_FIELD = 'prior_resident_RAM_ninth_original_interval_failure'
INHERITED_FIELD = 'inherited_one_postDEV_refresh_opportunity_policy'
INHERITED_KIND = 'ASR18_inherited_consumed180000_refresh_opportunity_after_ninth_original_interval_v1'
REPLACED_FIELDS = ('original_clip_geometry_overlay', 'resident_RAM_resource_scheduling_policy', 'resident_RAM_packet_credit_policy')
RUNTIME_FIELDS = ('parent_driver_source', 'worker_driver_source', 'resident_RAM_stream_configuration',
                  'resident_branch_namespace', 'checkpoint_interval_updates', 'source_checkpoint', 'source_configuration')


def exact_dead(who):
    try:
        return abs(psutil.Process(who['pid']).create_time() - who['create_time']) >= .001
    except psutil.NoSuchProcess:
        return True


def pinned_root(path, sha):
    require(record(path)['sha256'] == sha, 'Exact immutable closed ROOT receipt')
    value = read_json(path)
    require(value['accepted'] is True and value['reviewer'] == '/root', 'Genuine independent completed ROOT review')
    return value


def legacy_source36_runtime_configuration(base):
    """Prove the only five operational fields against actual I25/f866, not an old source config."""
    require(record(BASE)['sha256'] == BASE_SHA, 'Proper actual current source36 runtime base')
    old = read_json(BASE)
    restored = copy.deepcopy(base)
    require(FAILURE_FIELD in restored and INHERITED_FIELD in restored, 'Both explicit ninth failure/inherited fields')
    del restored[FAILURE_FIELD]
    del restored[INHERITED_FIELD]
    for key in REPLACED_FIELDS:
        restored[key] = copy.deepcopy(old[key])
    require(restored == old, 'Only three ninth-file operational replacements plus failure and consumed provenance additions')
    return old


def legacy_source36_policy_configuration(base):
    """Existing legacy validators see the unchanged proper36 source, after exact actual-runtime reconstruction.

    This adapter never supplies old runtime values to future runtime bindings.
    Both future stream/source config pins are derived from the final new file by
    the unchanged driver/wrapper command and c.update path.
    """
    old = legacy_source36_runtime_configuration(base)
    proper_pin = old['source_configuration']
    require(old['resident_RAM_stream_configuration'] == proper_pin and record(proper_pin['path']) == proper_pin,
            'Both actual I25 source/stream pins are the same proper36 source configuration')
    proper = read_json(proper_pin['path'])
    reconstructed = copy.deepcopy(proper)
    reconstructed.update(parent_driver_source=record(P / 'work/research/run_ASR18_BPE9223_resident_RAM_training_v36.py'),
                         worker_driver_source=record(P / 'work/research/run_ASR18_BPE9223_resident_RAM_training_v36.py'),
                         resident_RAM_stream_configuration=proper_pin, resident_branch_namespace=str(BRANCH),
                         checkpoint_interval_updates=200, source_checkpoint=old['source_checkpoint'],
                         source_configuration=proper_pin)
    require(reconstructed == old and len(RUNTIME_FIELDS) == 7, 'Original complete seven actual runtime values reconstructed exactly')
    return proper


def checked_inherited_refresh_policy(base):
    policy = base[INHERITED_FIELD]
    event = pinned_root(ROOT_EVENT, '7caeec0f471f63027b0566e8869a131920a79a732421994b6bd6a6a6ca708e9a')
    quality = pinned_root(ROOT_QUALITY, '3fb24d985e3ddbd29477b712e8ffdaa535a997ce363baa48b24fa349133097e5')
    require(quality['closed_actual_first_periodic_idle_event_preservation_review'] == record(ROOT_EVENT)
            and quality['all207_threeAdam_fourRNG_cursors_journal_nextdraw_preserved'] is True
            and event['additional16MiB_above_full225_and_FULLprefetch_no_credit'] is True
            and event['additional_operations'] == ['unfreeze', 'captured_original_full_collect', 'freeze'],
            'Closed genuine first180k event and full-state preservation accepted; no old event re-audit')
    require(policy == {'kind': INHERITED_KIND, 'historical_event_optimizer_updates': 180000,
                      'historical_event_ROOT_review': record(ROOT_EVENT), 'historical_quality_ROOT_review': record(ROOT_QUALITY),
                      'historical_actual_event': event['event'],
                      'historical_full_state_fingerprint': event['source_bound_real_original_full854_fingerprint_and_ID_storage_versions_gradNone_frontiers_fourRNG_queued_before_after_capture_exact'],
                      'consumed_before_native_loop': True, 'new_owner_completed': False,
                      'new_owner_GC_capture_sync_or_ineligible_event_allowed': False,
                      'additional_bookkeeping_reserve_bytes': 16777216},
            'Exact inherited consumed provenance only; no second eligible/ineligible refresh and no freed credit')
    require(base['one_postDEV_gc_refresh_policy']['additional_bookkeeping_reserve_bytes'] == 16777216,
            'Original full16MiB event bookkeeping reserve retained')
    return policy


def checked_ninth_original_policy(base):
    old = legacy_source36_runtime_configuration(base)
    proof = record(ROOT_CPU)
    require(proof['sha256'] == '46392765789ac059a6a00569ab8d2b6527b19a6a7a45021638c5e54254f4777f', 'Actual completed ninth CPU14 ROOT gate')
    overlay = base['original_clip_geometry_overlay']
    require(overlay == {'prior_overlay': old['original_clip_geometry_overlay'],
                        'source': record(S / 'fastmss_original_clip_geometry_overlay_v10.py'),
                        'certificate': record(CERTIFICATE), 'ROOT_review': proof}, 'Exact ninth file and entire prior8 policy')
    review = read_json(ROOT_CPU)
    require(review['accepted'] is True and review['reviewer'] == '/root'
            and review['scope'] == 'ASR18_source_bound_ninth_original_full_clip_geometry_overlay_v1'
            and review['overlay_source'] == overlay['source'] and review['certificate'] == overlay['certificate']
            and review['CPU_source'] == record(S / 'fastmss_resident_feature_stream_v14.py')
            and review['actual_TOOL']['TOOL_exit_code'] == 0
            and review['actual_TOOL']['actual_external_stdout_terminal']['Popen_returncode'] == 0
            and len(review['all8_job_manual_sealer_exact_dead']) == 8
            and all(x['exact_dead'] is True for x in review['all8_job_manual_sealer_exact_dead'])
            and review['finite_actual_frontend_and_valid_mask_from_reviewed_source_passed'] is True,
            'Real original ingress/frontend/finite/mask/full66 and all8 exact metadata deaths; no raw proof reread')
    resource = copy.deepcopy(old['resident_RAM_resource_scheduling_policy'])
    resource['CPU_protocol_source'] = record(S / 'fastmss_resident_feature_stream_v14.py')
    credit = copy.deepcopy(old['resident_RAM_packet_credit_policy'])
    credit['actual_CPU_control_review'] = record(ROOT_TRANSPORT)
    require(base['resident_RAM_resource_scheduling_policy'] == resource and base['resident_RAM_packet_credit_policy'] == credit
            and record(ROOT_TRANSPORT)['sha256'] == 'b794ac22769d6eba2b78ddee87e40af8e0f89dbe23a32f96d8d580241aeabe41',
            'Only CPU14 provenance/derived import-only transport successor; original guards/math/MAX2/225 exact')
    require(base[FAILURE_FIELD] == pinned_root(ROOT_FAILURE,
            'e43279049b8f5fc26e9a8d85c36ebb1983e667b9c1bb02a8197d4cc35a3c8b84')['owner_closure'],
            'Exact accepted source36 abnormal failure, never old eighth or intentional transition')
    checked_inherited_refresh_policy(base)
    from fastmss_resident_feature_stream_v14 import checked_resource_profile, checked_packet_credit_policy
    checked_resource_profile(base)
    checked_packet_credit_policy(base)
    return overlay


def checked_ninth_original_resume(pin, *, base, folder, pointer_pin):
    checked_ninth_original_policy(base)
    require(Path(folder).resolve() == BRANCH.resolve(), 'Exact current resident branch')
    failure = pinned_root(ROOT_FAILURE, 'e43279049b8f5fc26e9a8d85c36ebb1983e667b9c1bb02a8197d4cc35a3c8b84')
    standard = pinned_root(ROOT_STANDARD, 'c1358e5d7f6acc52ae1cb7d3564046da15984855e179887615d13caceef19801')
    require(pin == base[FAILURE_FIELD] == failure['owner_closure'] and record(pin['path']) == pin,
            'Exact sealed actual186563 failure closure; metadata pin only, no old suffix/event read')
    require(failure['actual_optimizer'] == failure['published'] == 186563 and failure['saved'] == 186400
            and failure['gap0_trace_after_native_commit'] is True
            and failure['primary_error'] == "ValueError('Original interval beyond source unresolved')"
            and failure['propagated_error'] == "ValueError('Never release credit from errored transport')"
            and failure['actual42997_ownerterminal']['owned_exec_session'] == 42997
            and failure['actual42997_ownerterminal']['actual_TOOL_exit_code'] == 1
            and failure['actual42997_ownerterminal']['normal0_claimed'] is False
            and all(v == 'UNRECORDED_UNKNOWN' for v in failure['separate_numeric_OSexits'].values()),
            'Actual primary/cleanup failure separated, gap0, outer1 and unknown individual OS exits')
    require(failure['Monitor']['joined'] is True and failure['Monitor']['clean'] is True and failure['Monitor']['error'] is None
            and failure['scalar_completion']['joined'] is True
            and failure['scalar_completion']['worker_connections_closed_on_worker'] is True,
            'Closed original Monitor/scalar receipts, not a fabricated normal transport closure')
    expected = [(21912,1792780981.1092248),(37108,1792780981.1024935),(29600,1792780979.698954),
                (39856,1792790240.3032615),(35440,1792790240.2869232)]
    deaths = failure['known_exact_deaths'][:5]
    require([(x['identity']['pid'],x['identity']['create_time']) for x in deaths] == expected
            and all(x['exact_dead'] is True and exact_dead(x['identity']) for x in deaths), 'Exact old five dead before new authority')
    require(pointer_pin == standard['pointer'] and pointer_pin['sha256'] == '21ac6818f9c761367e0a5d24e75ee60afd4a818c15a1fe0ce2554f0fdd08e2ed'
            and failure['saved_fullstate_ROOT_review'] == record(ROOT_STANDARD)
            and standard['named_tensors854_geometry_alignment_all_referenced'] is True
            and standard['weights207_Adam215_three_partitions_fourRNG_clocks'] is True
            and standard['original_scalar_semantics_verifier_result']['passed'] is True
            and standard['original_scalar_semantics_verifier_result']['optimizer_updates'] == 186400,
            'Accepted full854/207/215/3Adam4RNG semantic PRE-SAVED186400; no payload read or restore claim')
    saved = standard['original_scalar_semantics_verifier_result']['journal_prefix']
    stat = failure['original_journal_STAT_ONLY']
    tail = failure['tail163']
    require(saved['offset'] == 375720854 and saved['sha256'] == 'c658c5c1711f0e43b3b764e0a8875e90cdd5137f7d27469dd28936802fd02e80'
            and Path(saved['path']) == BRANCH / 'journals/commits_000025.jsonl'
            and stat['path'] == saved['path'] and stat['bytes'] == saved['offset'] + tail['bytes'] == 376041847
            and failure['tail_range']['count'] == 163
            and failure['tail_range']['all163_native_positive_finite_currentCPU13_tokens_and_absolute_cursors'] is True,
            'Accepted saved prefix and historical163 COPY; no journal/prefix/tail reopen or re-audit')
    current = Path(stat['path']).stat()
    require(current.st_size == stat['bytes'] and current.st_mtime_ns == stat['mtime_ns'], 'Original25 STAT only and untouched')
    require(Path(read_json(BRANCH / 'active_journal.json')['journal_prefix']['path']) == Path(saved['path'])
            and read_json(BRANCH / 'latest_checkpoint.json')['pointer'] == pointer_pin,
            'Old25/saved186400 remain current before full restore and saved-prefix-only new26 branch')
    return {'kind': 'ASR18_accepted_source36_original_interval186563_saved186400_ninth_transition_v1',
            'actual_failure_closure': pin, 'actual_failure_ROOT': record(ROOT_FAILURE),
            'saved_optimizer_updates': 186400, 'actual_optimizer_updates': 186563,
            'published_journal_optimizer_updates': 186563, 'optimizer_journal_gap': 0,
            'saved_journal_prefix': saved, 'whole_journal_including_preserved_tail': stat,
            'preserved_unsaved_tail': tail, 'unsaved_tail_bytes': tail['bytes'],
            'preserved_tail_positive_updates': list(range(186401,186564)),
            'all163_require_genuine_redo_then_repaired186564': True,
            'closed_ROOT_receipts_reused_no_tail_or_event_read': True}


def inherit_consumed_postDEV_opportunity(owner, *, base, restored_updates, guard, emit):
    """Metadata-only consumed inheritance; zero eligible/ineligible/capture/GC calls.

    New owner's attempted/completed stay false. The historical successful event
    belongs to I25, not this fresh owner, and no event receipt is synthesized.
    """
    require(threading.current_thread() is threading.main_thread(), 'MainThread inheritance authority')
    module = sys.modules.get('asr18_one_postDEV_gc_refresh_v2')
    require(module is not None and type(owner) is module.OnePostDEVRefresh, 'Actual unchanged original helper2 owner')
    policy = checked_inherited_refresh_policy(base)
    require(type(restored_updates) is int and restored_updates >= 186400
            and owner.expected_update == (restored_updates // 4000 + 1) * 4000
            and owner.opportunity_consumed is False and owner.attempted is False and owner.completed is False
            and owner.operations_completed == [] and owner.failed_phase is None and owner.ineligible_reason is None,
            'Fresh owner, genuine restored frontier, no attempt or duplicate inheritance')
    require(callable(guard) and callable(emit), 'Original guard and actual bounded provenance writer')
    guard()
    owner.opportunity_consumed = True
    receipt = {'kind': 'ASR18_actual_inherited_consumed180000_refresh_opportunity_v1',
               'passed': True, 'historical_event_optimizer_updates': 180000,
               'restored_optimizer_updates': restored_updates, 'policy': copy.deepcopy(policy),
               'historical_event_performed_by_prior_invocation': 25,
               'opportunity_consumed_before_native_loop': True, 'new_owner_attempted': owner.attempted,
               'new_owner_completed': owner.completed, 'new_owner_operations_completed': list(owner.operations_completed),
               'new_eligible_or_ineligible_event_capture_GC_sync_calls': 0,
               'host_credit_bytes': 0, 'additional16MiB_bookkeeping_reserve_retained': True}
    emit(receipt)
    guard()
    return receipt
