"""INACTIVE coherent source40 adapter-gate saved212000 recovery; distinct CPU configuration roles.
Original science and CPU14 are unchanged; no restore/control occurs here.
"""
import copy
from pathlib import Path
import psutil
from atomic_io_retry_v1 import read_json
from bpe_broad_audio_v1 import record,require

P=Path(r"C:/Users/ardao/Documents/Codex/2026-10-10/i-x20")
S=P/"outputs/parakeet_turkish"
ROOT=Path(r"E:/gpu_training_scratchpad")
BRANCH=ROOT/"runs/BPE9223_resident_RAM_20261018_01"
W=Path(r"E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1")
X=Path(__file__).resolve().parent
POLICY="authoritative_current_plan_identity_and_main_path_attribution_policy"
PRIOR="prior_resident_RAM_source37_postDEV204000_throughput_transition"
REPAIR="original_CPU14_startup_reader7_recovery_policy"
FAILURE="prior_resident_RAM_source40_adapter212000_failure"
EXTRA_HOST_BYTES=(72+8)*1024**2

def exact_dead(who):
    try:return abs(psutil.Process(who["pid"]).create_time()-who["create_time"])>=.001
    except psutil.NoSuchProcess:return True

def legacy_source37_configuration(base):
    require(record(BASE_SOURCE["path"])==BASE_SOURCE,"Exact accepted source39 SOURCE dictionary")
    baseline=read_json(BASE_SOURCE["path"])
    candidate=copy.deepcopy(base)
    require(REPAIR in candidate and FAILURE in candidate,"Explicit new startup/reader policy and actual212000 failure")
    del candidate[REPAIR];del candidate[FAILURE]
    require(candidate==baseline,"ONLY two new operational fields; all original source39 science/LR/GC/225/FULL exact")
    proper=copy.deepcopy(baseline)
    del proper[POLICY];del proper[PRIOR]
    return proper

def checked_startup_reader_repair_policy(base):
    p=base[REPAIR];cpu=p["CPU_startup_policy"]
    require(p["kind"]=="ASR18_source42_original_CPU14_Reader8_normal259800_recovery_v1"
        and cpu["kind"]=="ASR18_original_CPU14_startup_receipts_launch_adapter_v1"
        and cpu["adapter_source"]==record(X/"CPU14_startup_launch_adapter_v15.py")
        and cpu["runner_source"]==record(X/"CPU14_owned_startup_only_receipt_runner_v2.py")
        and cpu["original_CPU14_source"]==record(S/"fastmss_resident_feature_stream_v14.py")
        and cpu["additional_host_reservation_bytes"]==8*1024**2
        and cpu["maximum_receipt_bytes"]==1024**2 and cpu["four_coarse_hooks_once"]is True
        and cpu["no_per_update_observer"]is True
        and p["additional_host_reservation_bytes"]==8*1024**2
        and p["total_additional_native_host_bytes"]==EXTRA_HOST_BYTES,
        "Exact startup-only adapter/runner and separately charged8MiB ABOVE original72; no privatecap or science change")
    require(p["reader_template"]==READER_TEMPLATE and record(READER_TEMPLATE["path"])==READER_TEMPLATE
        and p["projection_source"]==PROJECTION and record(PROJECTION["path"])==PROJECTION
        and p["API_helper_source"]==API_HELPER and record(API_HELPER["path"])==API_HELPER
        and Path(p["API_both_shapes_ROOT"]["path"]).resolve()==API_ROOT_PATH.resolve()
        and record(p["API_both_shapes_ROOT"]["path"])==p["API_both_shapes_ROOT"],
        "Exact tested Reader8/helper4/projector2 plus NEW immutable ROOT both-shapes authority")
    api=read_json(p["API_both_shapes_ROOT"]["path"])
    require(api["kind"]=="ROOT_reader8_both_original500000_shapes_and_lifetime_actual_acceptance_v1"
        and api["accepted"]is True and api["reviewer"]=="/root"
        and api["candidate_reader"]==READER_TEMPLATE and api["projection_source"]==PROJECTION
        and api["API_helper_source"]==API_HELPER,"NEW Reader8 BOTH actual500000 shapes and full lifetime ROOT required")
    for key,kind,accepted_key,shape_key,shape_name in (
        ("metadata_live_order_ROOT","ROOT_reader8_first_original500000_shape_and_lifetime_actual_acceptance_v1","accepted_first_shape","completed_first_shape","metadata_live_order"),
        ("adverse_reverse_interleaved_ROOT","ROOT_reader8_second_original500000_adverse_shape_and_lifetime_actual_acceptance_v1","accepted_adverse_shape","completed_second_shape","adverse_reverse_interleaved")):
        shape_pin=api[key]
        require(record(shape_pin["path"])==shape_pin,"Exact NEW shape Root receipt")
        shape=read_json(shape_pin["path"])
        require(shape["kind"]==kind and shape[accepted_key]is True and shape["reviewer"]=="/root"
            and shape["candidate_reader"]==READER_TEMPLATE and shape["projection_source"]==PROJECTION
            and shape[shape_key]["shape"]==shape_name and shape[shape_key]["original_history_rows"]==500000
            and shape["actual_public_Python_exit"]==shape["actual_external_Popen_wait"]==shape["actual_server_Popen_wait"]==0
            and shape["normal_port_released"]is True
            and shape["completed_full_lifetime"]["cancellation_keeps_slot_until_actual_completion"]is True
            and shape["completed_full_lifetime"]["original_exception_identity"]is True,
            "Actual new candidate shape/full lifetime/normal Popen-method closure, not historical API authority")
    proof_pin=cpu["actual_new_adapter_CPU_control_review"]
    require(type(proof_pin)is dict and record(proof_pin["path"])==proof_pin,
        "Genuine completed NEW adapter CPU control review required; SOURCE-only gate cannot launch")
    proof=read_json(proof_pin["path"])
    require(proof["kind"]=="ASR18_original_CPU14_startup_adapter_actual_CPU_v1"
        and proof["accepted"]is True and proof["reviewer"]=="/root"
        and proof["adapter_source"]==cpu["adapter_source"]and proof["runner_source"]==cpu["runner_source"]
        and proof["CPU14_source"]==cpu["original_CPU14_source"]
        and proof["full_first_packet_passed"]is True
        and proof["dual_distinct_config_roles_and_actual_driver_call_AST_passed"]is True
        and proof["actual_worker_wire_configuration_equals_incoming_stream_pin"]is True
        and proof["driver_source"]==record(X/"r8_resume_driver_v42.py")
        and proof["original_CPU14_worker_and_transport_unchanged"]is True
        and proof["all_exact_known_identities_dead"]is True,
        "New launch-path FULL original firstpacket/lifetime/closure actual proof, not old CPU03 coverage waiver")
    reader_pin=p["actual_reader_replacement_ROOT"]
    require(type(reader_pin)is dict and record(reader_pin["path"])==reader_pin,
        "Genuine separately admitted actual reader replacement ROOT required")
    reader=read_json(reader_pin["path"])
    require(reader["kind"]=="ROOT_reader8_actual_replacement_acceptance_v1"
        and reader["accepted"]is True and reader["reviewer"]=="/root"
        and reader["template_source"]==READER_TEMPLATE
        and reader["projection_source"]==PROJECTION and reader["API_helper_source"]==API_HELPER
        and reader["old_exact_pair_dead"]is True and reader["mirror_thread_alive"]is True
        and reader["HTTP_ready"]is True and reader["reader_ready"]is True and reader["error"]is None,
        "Actual new reader ready/thread-safe/current-journal acceptance before training")
    for identity in (reader["new_reader_identity"],reader["new_reader_launcher_identity"]):
        observed=psutil.Process(identity["pid"])
        require(abs(observed.create_time()-identity["ctime"])<.001 and observed.ppid()==identity["ppid"],
            "Exact accepted new reader pair remains alive")
    status=read_json(reader["status_path"])
    require(status["pid"]==reader["new_reader_identity"]["pid"]
        and status["create_time"]==reader["new_reader_identity"]["ctime"]
        and status["mirror_thread_alive"]is True and status["HTTP_ready"]is True
        and status["reader_ready"]is True and status["error"]is None,
        "Current local mirror/thread/HTTP health, processalive alone is insufficient")
    return p

BASE_SOURCE={'path': 'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\one_postDEV_gc_refresh_v1\\next200000_throughput_repair_v1\\source39_actual204800_configuration_v1.json', 'bytes': 229536, 'sha256': 'a7317c860e5548cbfd6fad0ce292b224247a5fa370eed004ff3eebbd5559d11f'}
READER_TEMPLATE={"path":"E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\one_postDEV_gc_refresh_v1\\next200000_throughput_repair_v1\\next212000_failure_repair_v1\\serve_BPE9223_Trackio_live_v8_INACTIVE_v1.py","bytes":28141,"sha256":"efd774dc960f54191260e2f883bdb95ec970985af12935d850fc9ced5b30d724"}
PROJECTION={'path': 'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\one_postDEV_gc_refresh_v1\\next200000_throughput_repair_v1\\next212000_failure_repair_v1\\incremental_scalar_dashboard_projection_v2.py', 'bytes': 10819, 'sha256': '0960e7a2942fa639d3b18f87afb097decbad832fdd7d028146d7e03630dde9ac'}
API_HELPER={'path': 'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\one_postDEV_gc_refresh_v1\\next200000_throughput_repair_v1\\next212000_failure_repair_v1\\Trackio_owned_bounded_sync_API_dispatch_v4.py', 'bytes': 6960, 'sha256': '9634c43887f831f66d8f59dfc143ded6779418646dab2763d1bf25343cf16ada'}
API_ROOT_PATH=X/"ROOT_reader8_both500k_actual_acceptance_0110_v1"/"acceptance.json"
FAILURE_ROOT={'path': 'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\one_postDEV_gc_refresh_v1\\next200000_throughput_repair_v1\\next212000_failure_repair_v1\\ROOT_source40_I28_failure_independent_seal_and_next_SOURCE_scope_v1\\acceptance.json', 'bytes': 13844, 'sha256': 'c1c1c82069e31cbd9871581dd9f0c11a05ad3459f025d589a1c3c7f34364e580'}
SAVED_ROOT={'path': 'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\one_postDEV_gc_refresh_v1\\next200000_throughput_repair_v1\\ROOT_NEW212000_saved_full854_metadata_independent_acceptance_v1\\acceptance.json', 'bytes': 2995, 'sha256': 'ddce6024bea880f6cb107015778d10635d11f69dd588d2221da452a634763861'}
SOURCE_DERIVED_POINTER={'path': 'E:\\gpu_training_scratchpad\\runs\\BPE9223_resident_RAM_20261018_01\\checkpoints\\update_000212000\\pointer.json', 'bytes': 16120, 'sha256': '9bef17e4fd0ed37f34f056595ca09ec91d2a69d80786165d6ca866b270bcbe07'}
REQUIRED_OLD_IDENTITIES=[{'pid': 34788, 'create_time': 1792842518.865024, 'ppid': 32332}, {'pid': 32332, 'create_time': 1792842518.85846, 'ppid': 4964}, {'pid': 4964, 'create_time': 1792842518.480302, 'ppid': 6664}, {'pid': 6664, 'create_time': 1792842518.4723651, 'ppid': 37480}, {'pid': 23496, 'create_time': 1792848390.67016, 'ppid': 39100}, {'pid': 39100, 'create_time': 1792848390.6293392, 'ppid': 34788}, {'pid': 33668, 'create_time': 1792855655.0849347, 'ppid': 34788}]

def checked_current_plan_throughput_policy(base):
    legacy_source37_configuration(base)
    policy = base[POLICY]
    require(policy['kind'] == 'ASR18_authoritative_current_plan_identity_and_main_path_attribution_v1'
        and policy['identity_helper'] == record(W/'current_plan_identity_view_v3.py')
        and policy['phase_helper'] == record(W/'main_path_scalar_attribution_v1.py')
        and policy['additional_native_pressure_gate_source'] == record(W/'asr18_native_update_commit_pressure_bookkeeping_v4.py')
        and policy['additional_host_reservation_bytes'] == 72*1024**2
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
    checked_startup_reader_repair_policy(base)
    from current_plan_identity_view_v3 import OriginalSourceAuthority,current_plan_identity_view
    from main_path_scalar_attribution_v1 import MainPathScalarAttribution
    from asr18_native_update_commit_pressure_bookkeeping_v4 import wait_for_native_update_commit_pressure
    return OriginalSourceAuthority,current_plan_identity_view,MainPathScalarAttribution,wait_for_native_update_commit_pressure


def checked_current_plan_throughput_checkpoint_resume(pin,*,base,folder,pointer_pin):
    raise AssertionError("Reader8 successor admits genuine normal259800 continuation only; historical abnormal transitions stay closed")
    require(Path(folder).resolve()==BRANCH.resolve()and pin==base[FAILURE]==FAILURE_ROOT
        and record(pin["path"])==pin,"Exact accepted source40 I28 adapter-gate failure authority")
    checked_current_plan_throughput_policy(base)
    value=read_json(pin["path"])
    require(value["kind"]=="ROOT_source40_I28_actual_adapter_gate_failure_independent_seal_v1"
        and value["accepted"]is True and value["reviewer"]=="/root"
        and value["owner50280_consumed_ONCE_only_by_owner"]is True
        and value["owned_TOOL_reported_exit"]==1 and value["actual_direct_wrapper_numeric_OS_exit"]==1
        and value["separate_native_numeric_OS_exit"]=="UNRECORDED_UNKNOWN"
        and value["actual_primary_error"]=="AssertionError()"
        and value["compound_assert_false_operands"]=="UNRECORDED_UNKNOWN"
        and value["actual_optimizer_updates"]==value["published_journal_update"]==value["saved_update"]==212000
        and value["gap"]==0 and value["new_positive_or_unsaved_tail_updates"]==0,
        "Honest NEW adapter failure/direct wrapper1/native exit and operand UNKNOWN; coherent212000 zero-gap")
    deaths=value["all4known_native_and_sealer_exactdead"]
    require(len(deaths)==4 and all(d["exact_dead"]is True and exact_dead(d["identity"])for d in deaths)
        and all(any(d["identity"]==i for d in deaths)for i in REQUIRED_FAILURE_IDENTITIES),
        "All recorded NEW exact native/sealer identities dead, no invented wrapper identity")
    wrapper=value["wrapper_PID_only"]
    require(wrapper["pid"]==26912 and wrapper["observed_absent"]is True
        and wrapper["ctime"]=="UNRECORDED_UNKNOWN",
        "Historical wrapper ctime/exact death remains UNKNOWN; accepted PIDabsence is not exactdeath")
    restore=value["actual_full_restore"];branch=value["actual_journal_branch"]
    require(restore["passed"]is True and restore["all207_threeAdam_all4RNG_match_actual_pre_save"]is True
        and restore["next_draw_packet_match"]is True and restore["restore_occurred_only_at_startup_before_native"]is True
        and branch["historical_unsaved_tail_updates"]==0 and branch["native_updates_or_draws_in_copy"]==0,
        "Accepted full restore-before-savedprefix-copy/zero draw, not a successful native update")
    require(record(SAVED_ROOT["path"])==SAVED_ROOT,"Exact closed ROOT full854/PRE-SAVED authority")
    saved=read_json(SAVED_ROOT["path"])
    require(saved["accepted"]is True and saved["reviewer"]=="/root"and saved["updates"]==212000
        and saved["weights"]==207 and saved["Adam_states"]==215 and saved["Adam_partitions"]==[17,6,192]
        and saved["all4RNG"]is True and saved["named_tensor_references"]==854
        and saved["integer_Adam_JSON_keys_only_normalized"]is True
        and saved["semantic_identity_sampler_journal_nextdraw_equal"]is True,
        "All original full-state geometry/scalars/samplers/nextdraw, no new payloadhash or GPUrestore claim")
    require(pointer_pin==SOURCE_DERIVED_POINTER and record(pointer_pin["path"])==pointer_pin,
        "Exact saved212000 pointer bytes/hash required before future restore")
    pointer=read_json(pointer_pin["path"])
    require(pointer["committed_updates"]==212000 and pointer["diagnostic_only"]is False
        and pointer["all_window_updates_committed"]is True,"Full coherent212000 pointer")
    prefix=pointer["journal_prefix"]
    require(prefix==branch["saved_prefix"]and prefix["offset"]==426026870
        and prefix["sha256"]=="6621855eb4191790c3a741e3e1c19014f00db87b9e471d3a4c70802cbddaaf0a"
        and Path(prefix["path"]).resolve().is_relative_to(BRANCH.resolve()),"Original published J27 savedprefix only")
    expected=base[REPAIR]["closed_original_journal_STAT"]
    require(type(expected)is dict and expected["kind"]=="STAT_ONLY_original_closed_journal_v1"
        and expected["path"]==prefix["path"]and expected["bytes"]==prefix["offset"],"Original J27 STAT binding")
    actual=Path(expected["path"]).stat()
    require(actual.st_size==expected["bytes"]and actual.st_mtime_ns==expected["mtime_ns"],"Original J27 STAT only")
    failed=base[REPAIR]["closed_failed_I28_journal_STAT"]
    require(failed==value["NEW_J28_STAT_ONLY"]and failed["body_read"]is False
        and failed["path"]==branch["new_active_journal"]["path"]and failed["bytes"]==prefix["offset"],
        "Separately preserved failed J28 copiedprefix STAT, no body/hash/tail promotion")
    failed_actual=Path(failed["path"]).stat()
    require(failed_actual.st_size==failed["bytes"]and failed_actual.st_mtime_ns==failed["mtime_ns"],"Failed J28 STAT unchanged")
    require(base[REPAIR]["actual_reader_current_continuation_ROOT"]==FAILURE_ROOT,
        "Current same-reader J28 continuation authority must be separately exact ROOT-bound")
    historical_reader=read_json(base[REPAIR]["actual_reader_replacement_ROOT"]["path"])
    require(historical_reader["verified_updates"]>=212000
        and historical_reader["journal_path"]==prefix["path"]and historical_reader["journal_offset"]==prefix["offset"],
        "Genuine original reader ROOT verified saved212000 J27 lineage; separate from fresh health")
    require(branch["new_active_journal"]["offset"]==prefix["offset"]
        and branch["new_active_journal"]["sha256"]==prefix["sha256"],
        "Accepted J28 is exact copied J27 prefix with no new draws; no invented current verification")
    status=read_json(historical_reader["status_path"])
    reader=value["current_reader_health"]
    require(status["pid"]==reader["pid"]and status["create_time"]==reader["create_time"]
        and status["journal_path"]==reader["journal_path"]==failed["path"]
        and status["journal_offset"]==prefix["offset"]
        and ("last_verified_checkpoint_update"not in status or
            (type(status["last_verified_checkpoint_update"])is int and status["last_verified_checkpoint_update"]>=212000))
        and status["mirror_thread_alive"]is True and status["HTTP_ready"]is True
        and status["reader_ready"]is True and status["error"]is None,
        "Healthy existing same reader on failed J28 prefix; no restart or stale J27-current claim")
    return dict(saved_journal_prefix=prefix,whole_journal_including_preserved_tail=expected,
        preserved_closed_failed_journal_STAT=failed,preserved_tail_positive_updates=0,unsaved_tail_bytes=0,
        actual_transition=pin,full_checkpoint_restore_and_genuine_redo_required=True,
        honest_original_optimizer_vs_journal_gap=0,no_unsaved_positive_updates_after212000=True)

REQUIRED_FAILURE_IDENTITIES=[{'pid': 25176, 'create_time': 1792873894.854458, 'ppid': 32972}, {'pid': 32972, 'create_time': 1792873894.8463151, 'ppid': 26912}, {'pid': 8144, 'create_time': 1792874294.539028, 'ppid': 824}, {'pid': 824, 'create_time': 1792874294.5242395, 'ppid': 17644}]
