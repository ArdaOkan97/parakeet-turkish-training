"""INACTIVE exact source37 post204000-DEV cooperative checkpoint transition.
No stop/control, restore, draw, tensor, decoder or numerical operation runs here.
Original old policy validators receive the exact proper source37 configuration;
future runtime uses the NEW actual configuration and its original seven fields.
"""
import copy
from pathlib import Path

import psutil
from atomic_io_retry_v1 import read_json
from bpe_broad_audio_v1 import record, require

P = Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
S = P / 'outputs/parakeet_turkish'
ROOT = Path('E:/gpu_training_scratchpad')
BRANCH = ROOT / 'runs/BPE9223_resident_RAM_20261018_01'
W = Path(__file__).resolve().parent
BASE_RUNTIME = BRANCH / 'invocations/owner_000026/configuration.json'
BASE_RUNTIME_SHA = '39dc6a3aa4b7139d6209751de769d5aef63a507e7122f56a6720afd85bdb966d'
POLICY = 'authoritative_current_plan_identity_and_main_path_attribution_policy'
PRIOR = 'prior_resident_RAM_source37_postDEV204000_throughput_transition'
EXTRA_HOST_BYTES = 72 * 1024**2


def exact_dead(who):
    try:return abs(psutil.Process(who['pid']).create_time() - who['create_time']) >= .001
    except psutil.NoSuchProcess:return True


def legacy_source37_configuration(base):
    require(record(BASE_RUNTIME)['sha256'] == BASE_RUNTIME_SHA, 'Exact accepted original source37 runtime dictionary')
    runtime = read_json(BASE_RUNTIME)
    proper_pin = runtime['source_configuration']
    require(runtime['resident_RAM_stream_configuration'] == proper_pin and record(proper_pin['path']) == proper_pin,
            'Both original source37 stream/source pins are the same proper source configuration')
    proper = read_json(proper_pin['path'])
    candidate = copy.deepcopy(base)
    require(POLICY in candidate and PRIOR in candidate, 'Explicit throughput policy and actual intentional-stop lane')
    del candidate[POLICY];del candidate[PRIOR]
    require(candidate == proper, 'ONLY two new operational fields; all original source37 science/guards/9files/LR exact')
    restored = copy.deepcopy(proper)
    restored.update(parent_driver_source=record(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v37.py'),
        worker_driver_source=record(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v37.py'),
        resident_RAM_stream_configuration=proper_pin, resident_branch_namespace=str(BRANCH),
        checkpoint_interval_updates=200, source_checkpoint=runtime['source_checkpoint'], source_configuration=proper_pin)
    require(restored == runtime, 'All seven accepted current runtime bindings reconstructed, not reused as future values')
    return proper


def checked_current_plan_throughput_policy(base):
    legacy_source37_configuration(base)
    policy = base[POLICY]
    require(policy['kind'] == 'ASR18_authoritative_current_plan_identity_and_main_path_attribution_v1'
        and policy['identity_helper'] == record(W/'current_plan_identity_view_v3.py')
        and policy['phase_helper'] == record(W/'main_path_scalar_attribution_v1.py')
        and policy['additional_native_pressure_gate_source'] == record(W/'asr18_native_update_commit_pressure_bookkeeping_v4.py')
        and policy['additional_host_reservation_bytes'] == EXTRA_HOST_BYTES
        and policy['identity_reservation_bytes'] == 8*1024**2
        and policy['conservative_phase_bookkeeping_reservation_bytes'] == 64*1024**2
        and policy['original_three_fresh_canonical_plan_digests_unchanged'] is True
        and policy['same_original_row_guard_two_fresh_signatures_poison_and_deepcopy'] is True
        and policy['original225_MAX2_FULLprefetch_guards_and_consumed180k_unchanged'] is True,
        'Exact helper sources and separately charged conservative72MiB; no binary/hash-cache/guard/science change')
    proof_pin = policy['actual_CPU_control_review']
    require(type(proof_pin) is dict and record(proof_pin['path']) == proof_pin, 'Genuine completed positive CPU proof required')
    proof = read_json(proof_pin['path'])
    require(proof['accepted'] is True and proof['reviewer'] == '/root'
        and proof['kind'] == 'ASR18_authoritative_current_plan_identity_view_actual_CPU_v1'
        and proof['helper_source'] == policy['identity_helper']
        and proof['all_exact_metadata_identities_dead'] is True
        and proof['CPU_performance_supports_activation'] is True,
        'Original row/propose/replay semantic and both fixed positive performance conditions; no source-only gate')
    from current_plan_identity_view_v3 import OriginalSourceAuthority,current_plan_identity_view
    from main_path_scalar_attribution_v1 import MainPathScalarAttribution
    from asr18_native_update_commit_pressure_bookkeeping_v4 import wait_for_native_update_commit_pressure
    return OriginalSourceAuthority,current_plan_identity_view,MainPathScalarAttribution,wait_for_native_update_commit_pressure


def checked_current_plan_throughput_checkpoint_resume(pin, *, base, folder, pointer_pin):
    require(Path(folder).resolve() == BRANCH.resolve() and pin == base[PRIOR] and record(pin['path']) == pin,
        'Exact actual source37 postDEV204000 intentional closure, never an old abnormal/normal lane')
    checked_current_plan_throughput_policy(base)
    value = read_json(pin['path'])
    require(value['kind'] == 'ASR18_actual_source37_postDEV204000_throughput_transition_v1'
        and value['owned_session'] == 26493 and value['old_owner_invocation'] == 26 and value['new_owner_invocation'] == 27
        and value['actual_outer_TOOL_exit_code'] == 1 and value['normal0_claimed'] is False
        and value['intentional_cooperative_stop_observed'] is True,
        'Supported real cooperative exception and actual outer1, no invented OS exits or normal0')
    e = value['evidence']
    for key, item in e.items():
        if key != 'original_journal_stat':require(record(item['path']) == item, 'Exact new compact transition evidence '+key)
    require(e['old_driver'] == record(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v37.py')
        and e['old_wrapper'] == record(P/'work/research/run_ASR18_BPE9223_resident_RAM_autocontinue_v37.py')
        and e['old_CPU'] == record(S/'fastmss_resident_feature_stream_v14.py')
        and e['old_configuration'] == record(BASE_RUNTIME) and e['pointer'] == pointer_pin
        and e['actual_failure'] == record(BRANCH/'invocations/owner_000026/actual_failure.json')
        and e['stop'] == record(BRANCH/'invocations/owner_000026/stop.json'), 'Exact current source37/I26/CPU14 and supported stop')
    monitor = read_json(e['monitor_completion']['path'])
    require(monitor['joined'] is True and monitor['clean'] is True and monitor['error'] is None,
        'Original Monitor joined clean, no concurrent resource failure relabelled intentional')
    failure = read_json(e['actual_failure']['path'])
    from asr18_one_postDEV_gc_refresh_checkpoint_transition_v2 import validate_intentional_optimizer_journal_gap
    gap = validate_intentional_optimizer_journal_gap(failure,value['published_journal_optimizer_updates'])
    require(gap == value['optimizer_vs_journal_frontier'], 'Honest original optimizer/published0-or1 distinction')
    saved, actual, published = value['last_fully_saved_optimizer_updates'],value['actual_optimizer_updates'],value['published_journal_optimizer_updates']
    require(saved == 204000 and saved <= published <= actual < saved+200
        and failure['actual_optimizer_updates'] == actual and failure['last_checkpoint'] == saved,
        'Original204000 full checkpoint after real original DEV; all possible later tail genuinely redone')
    stop = read_json(e['stop']['path'])
    require(stop['kind'] == 'ASR18_human_authorized_postDEV204000_cooperative_checkpoint_stop_v1'
        and stop['optimizer_updates_observed'] == 204000 and stop['normal_exit_or_no_tail_race_claimed'] is False,
        'Exact reviewed watcher actual204000 startup opportunity, no atomic/normal0 assertion')
    identities = value['all_recorded_runtime_identities']
    require(3 <= len(identities) <= 16 and len({(i['pid'],i['create_time']) for i in identities}) == len(identities)
        and all(exact_dead(i) for i in identities)
        and any(i['pid'] == 19320 and abs(i['create_time']-1792803517.7236164)<.001 for i in identities),
        'Every original parent and captured bounded startup descendant exact dead, not launcher-only inference')
    original_identities = (
        {'pid':19320,'create_time':1792803517.7236164,'ppid':39856},
        {'pid':39856,'create_time':1792803517.7165897,'ppid':21928},
        {'pid':21928,'create_time':1792803516.8916032,'ppid':41376},
        {'pid':42068,'create_time':1792830275.0700862,'ppid':40536},
        {'pid':40536,'create_time':1792830275.0545292,'ppid':19320})
    require(all(any(all(i.get(k)==v for k,v in original.items()) for i in identities) for original in original_identities)
        and all(i['pid'] not in (26892,30128) for i in identities),
        'All exact original native/launcher/wrapper AND actual old200k CPU pair, readers excluded and alive')
    watcher = read_json(e['watcher_result']['path'])
    stage_pin = watcher['prestop_evidence']
    require(watcher['passed'] is True and watcher['stop_published'] is True
        and record(stage_pin['path']) == stage_pin and stop['evidence'] == stage_pin,
        'Actual admitted watcher published once after complete persisted stage evidence')
    stage = read_json(stage_pin['path'])
    require(stage['decision'] == 'READY' and stage['actual'] == 204000 and stage['ready_absent'] is True,
        'Genuine original204000 postDEV pre-ready opportunity')
    captured = stage['current_native_descendants']['descendants']
    require(all(any(i['pid'] == d['identity']['pid'] and i['create_time'] == d['identity']['create_time'] for i in identities) for d in captured),
        'ALL watcher-captured startup children included; no method-close or inner OS exit inference from death')
    pointer = read_json(pointer_pin['path']);persistence = read_json(e['persistence']['path'])
    require(pointer['committed_updates'] == saved and pointer['diagnostic_only'] is False and pointer['all_window_updates_committed'] is True
        and persistence['passed'] is True and persistence['optimizer_updates'] == saved
        and read_json(BRANCH/'latest_checkpoint.json')['pointer'] == pointer_pin, 'Latest full854 semantic PRE-SAVED checkpoint')
    stat = e['original_journal_stat'];path = Path(stat['path']);s = path.stat()
    require(stat['kind'] == 'STAT_ONLY_original_closed_journal_v1' and s.st_size == stat['bytes'] and s.st_mtime_ns == stat['mtime_ns'],
        'Original whole journal STAT ONLY; never hash/read its closed prefix or historical tail')
    require(pointer['journal_prefix'] == value['saved_journal_prefix']
        and value['preserved_tail_positive_updates'] == published-saved
        and value['unsaved_tail_bytes'] == stat['bytes']-pointer['journal_prefix']['offset']
        and value['original_tail_modified'] is False and value['tail_is_historical_only_requires_genuine_redo'] is True,
        'Saved-prefix-only branch authority and honest preserved historical tail')
    return {'saved_journal_prefix':value['saved_journal_prefix'], 'whole_journal_including_preserved_tail':stat,
        'preserved_tail_positive_updates':value['preserved_tail_positive_updates'], 'unsaved_tail_bytes':value['unsaved_tail_bytes'],
        'actual_transition':pin, 'full_checkpoint_restore_and_genuine_redo_required':True}
