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
    import urllib.request,time
    p=base[REPAIR];cpu=p["CPU_startup_policy"]
    require(p["kind"]=="ASR18_source43_CPU14_Reader9_abnormal276000_recovery_v1"
        and cpu["adapter_source"]==record(X/"CPU14_startup_launch_adapter_v15.py")
        and cpu["runner_source"]==record(X/"CPU14_owned_startup_only_receipt_runner_v2.py")
        and cpu["original_CPU14_source"]==record(S/"fastmss_resident_feature_stream_v14.py")
        and cpu["additional_host_reservation_bytes"]==8388608
        and p["total_additional_native_host_bytes"]==EXTRA_HOST_BYTES,"Unchanged CPU transport/science/reservations")
    proof_pin=cpu["actual_new_adapter_CPU_control_review"];require(record(proof_pin["path"])==proof_pin,"Exact CPU09 proof")
    proof=read_json(proof_pin["path"])
    require(proof["accepted"]is True and proof["reviewer"]=="/root"
        and proof["driver_source"]==record(X/"r9_recovery_driver_v43.py")
        and proof["adapter_source"]==cpu["adapter_source"]and proof["runner_source"]==cpu["runner_source"]
        and proof["CPU14_source"]==cpu["original_CPU14_source"]
        and proof["full_first_packet_passed"]is True and proof["all_exact_known_identities_dead"]is True,"Actual current full20x16 transport and normal closure")
    reader_pin=p["actual_reader_replacement_ROOT"];require(record(reader_pin["path"])==reader_pin,"Current Reader9 acceptance")
    reader=read_json(reader_pin["path"])
    require(reader["accepted"]is True and reader["HTTP_200"]is True
        and reader["template_source"]==record(p["reader_template"]["path"])
        and reader["API_helper_source"]==record(p["API_helper_source"]["path"]),"Actual Reader9 HTTP and source")
    for who in(reader["new_reader_identity"],reader["new_reader_launcher_identity"]):
        observed=psutil.Process(who["pid"]);require(abs(observed.create_time()-who["ctime"])<.001 and observed.ppid()==who["ppid"],"Exact current Reader9 identity")
    status=read_json(reader["status_path"])
    require(status["pid"]==reader["new_reader_identity"]["pid"]and status["mirror_thread_alive"]is True
        and status["reader_ready"]is True and status["error"]is None and time.time()-status["updated_unix"]<30,"Fresh Reader9 status")
    with urllib.request.urlopen("http://127.0.0.1:7860/version",timeout=2)as response:require(response.status==200,"Actual HTTP gate")
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
    require(Path(folder).resolve()==BRANCH.resolve(),"Same branch")
    require(pin==base[REPAIR]["abnormal276000_recovery"]and record(pin["path"])==pin,"Exact NEW recovery authority")
    value=read_json(pin["path"])
    require(value["accepted"]is True and value["reviewer"]=="/root"
        and value["kind"]=="ROOT_actual_I31_unknown_exit_276180_saved276000_recovery_v1"
        and value["numeric_OS_exit"]is None and value["normal_completion_claimed"]is False
        and value["checkpoint_pointer"]==pointer_pin and value["last_positive"]==276180
        and value["symbolic_fullstate_PRE_SAVED_equal"]is True
        and value["preserved_tail_positive_updates"]==list(range(276001,276181)),"Honest unknown exit, full-state checkpoint, retained180 positives")
    for who in value["exact_dead_identities"]:require(exact_dead(who),"All recorded identities dead")
    for pid in value["absent_PID_only"]:require(not psutil.pid_exists(pid),"Recorded PID absence; past ctime unknown")
    pointer=read_json(pointer_pin["path"])
    require(pointer["committed_updates"]==276000 and pointer["all_window_updates_committed"]is True
        and pointer["diagnostic_only"]is False and pointer["journal_prefix"]==value["saved_journal_prefix"],"Exact saved276000")
    whole=value["whole_journal_including_preserved_tail"];observed=Path(whole["path"]).stat()
    require(observed.st_size==whole["bytes"]and observed.st_mtime_ns==whole["mtime_ns"],"Original failed journal unchanged")
    return dict(saved_journal_prefix=pointer["journal_prefix"],whole_journal_including_preserved_tail=whole,
        preserved_tail_positive_updates=value["preserved_tail_positive_updates"],unsaved_tail_bytes=353826,
        actual_transition=pin,full_checkpoint_restore_and_genuine_redo_required=True)
