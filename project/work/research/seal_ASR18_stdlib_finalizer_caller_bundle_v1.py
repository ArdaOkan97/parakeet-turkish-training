"""Seal genuine discard/metadata controls and inactive full caller10 closure."""
import ast,datetime,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from fastmss_row_serial_cpu_v6 import environment,resource_guard
from asr18_reprepared_array_equivalence_v1 import validate_retirement
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');F=A.parent/'FastMSS_integration_v1'


def main():
    environment();sample=resource_guard();m=sample['system_memory'];budget=64_000_000
    require(m['commit_used_bytes']+budget<.9*m['commit_limit_bytes'] and m['commit_available_bytes']-budget>2_000_000_000,'Fresh64MB metadata sealing forecast')
    run=A/'full_DATA_two_window_native_02';retirement=run/'discard_partial_window_02';guard_path=A/'failed_CPU_finalization_window_discard2_guard_01/closure.json'
    report=read_json(retirement/'report.json');closed=read_json(guard_path)
    require(report['passed'] is True and report['shared_feature_slot_empty'] is True and report['original_TOOL1_unchanged'] is True
        and closed['actual_exit_code']==0 and closed['monitor_clean'] is True and closed['monitor_joined'] is True
        and closed['observed']['monitor_error'] is None and closed['observed']['samples']>=3,'Actual completed cleanup/guard0 only')
    actual=retirement/'actual_owned_CLOSED0.json';require(not actual.exists(),'Immutable new real cleanup0 proof')
    write_json(actual,{'kind':'ASR18_actual_exact_owned_failed_window_discard_TOOL0_v2','owner':'/root/memory_diagnosis',
        'actual_exit_code':0,'actual_tool_chunk_id':'ad7d93','native_PID':closed['pid'],
        'report':record(retirement/'report.json'),'guard':record(guard_path),'source':report['source'],
        'original_failed_trial':report['actual_failed_trial'],'original_TOOL1_preserved':True})
    controls=A/'array_repreparation_metadata_CPU_actual_01';cg=A/'array_repreparation_metadata_CPU_guard_01/closure.json'
    cr=read_json(controls/'report.json');cc=read_json(cg)
    require(cr['passed'] is True and cr['check_count']==43 and cc['actual_exit_code']==0
        and cc['monitor_clean'] is True and cc['monitor_joined'] is True and cc['observed']['samples']>=3,
        'Actual43 metadata controls0')
    control0=controls/'actual_owned_CLOSED0.json';require(not control0.exists(),'Immutable real metadata control0 proof')
    write_json(control0,{'kind':'ASR18_array_repreparation_metadata_controls_actual_TOOL0_v1','owner':'/root/memory_diagnosis',
        'actual_exit_code':0,'actual_tool_chunk_id':'a9f68b','native_PID':cc['pid'],
        'report':record(controls/'report.json'),'guard':record(cg),'source':cr['source'],
        'actual_NPY_or_audio_or_GPU_read':False})
    prior_failure=record(run/'actual_owned_CLOSED1_v1.json')
    failure,ret,archive=validate_retirement(record(actual),prior_failure=prior_failure,guard=resource_guard)
    require(len(archive['metadata'])==41 and len(archive['closures'])==20,'All old owner/update/active metadata preserved')
    prerequisite=retirement/'actual_prior_retirement_gate_verification_v1.json'
    write_json(prerequisite,{'passed':True,'source':record(S/'asr18_reprepared_array_equivalence_v1.py'),
        'actual_cleanup0':record(actual),'failed_trial1':prior_failure,'durable_archive':ret['preserved_metadata'],
        'old_metadata_files':len(archive['metadata']),'old_normal_frontend_closures':20,
        'cache_arrays_or_audio_read':False,'new_repreparation_or_native_approval_claimed':False})
    previous=A/'concrete_multiwindow_caller_source_review_manifest_v9.json';old=read_json(previous)
    pins={x['path']:x for x in old['bindings']}
    for pin in pins.values():require(record(pin['path'])==pin,'Preserved predecessor source/history signature')
    cpu_path=F/'FastMSS_stdlib_finalization_CPU6_source_review_manifest_v1.json';cpu=read_json(cpu_path)
    direct=[R/'run_ASR18_full_DATA_multiwindow_v9.py',S/'asr18_full_DATA_native_window_v4.py',
        S/'asr18_reprepared_array_equivalence_v1.py',R/'prepare_ASR18_full_DATA_multiwindow_configuration_v7.py',
        S/'asr18_full_DATA_multiwindow.disabled.v9.json',*[Path(x['path']) for x in cpu['source']]]
    queue=[x for x in direct if x.suffix=='.py'];seen=set();edges={}
    while queue:
        path=queue.pop().resolve()
        if str(path) in seen:continue
        seen.add(str(path));pins[str(path)]=record(path);imports=[]
        for node in ast.walk(ast.parse(path.read_bytes(),filename=str(path))):
            names=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                for parent in (path.parent,S,R):
                    candidate=parent/(name.split('.')[0]+'.py')
                    if candidate.is_file() and candidate.resolve().is_relative_to(P):imports.append(str(candidate.resolve()));queue.append(candidate);break
        edges[str(path)]=sorted(set(imports))
    extras=[Path(__file__),previous,cpu_path,Path(cpu['template']['path']),Path(cpu['interface']['path']),
        Path(cpu['actual_synthetic_controls']['path']),Path(cpu['supported_actual_failure']['path']),
        A/'ROOT_CPU6_stdlib_finalization_source_read_review_v1.json',A/'ROOT_failed_CPU_finalization_window_discard2_read_review_v1.json',
        R/'discard_ASR18_failed_original_CPU_window_v2.py',R/'validate_ASR18_array_repreparation_metadata_v1.py',
        actual,control0,prerequisite,Path(ret['preserved_metadata']['path']),retirement/'report.json',guard_path,
        controls/'report.json',cg,Path(failure['retained_feature_inventory']['path'])]
    for path in direct+extras:pins[str(path.resolve())]=record(path)
    for pin in pins.values():require(record(pin['path'])==pin,'Exact final source/evidence binding changed')
    manifest=A/'concrete_multiwindow_caller_source_review_manifest_v10.json';require(not manifest.exists(),'New immutable full source closure')
    value={'kind':'ASR18_concrete_resident_first_full_DATA_multiwindow_caller_source_review_manifest_v10',
        'owner':'/root/memory_diagnosis','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'requested_source_scope':'source_read_review_for_future_bounded_full_DATA_two_window_diagnostic',
        'later_execution_review_scope_required':'finite_full_DATA_two_window_diagnostic_v1',
        'exact_execution_configuration_binding_required':True,'production_enabled':False,
        'direct_sources':[record(x) for x in direct],'bindings':sorted(pins.values(),key=lambda x:x['path']),
        'owned_resolvable_AST_import_edges':edges,'preserved_predecessor_source_and_history_manifest':record(previous),
        'CPU_owner_bundle':record(cpu_path),'CPU_contract_template':cpu['template'],
        'configuration_template':record(S/'asr18_full_DATA_multiwindow.disabled.v9.json'),
        'configuration_assembler':record(R/'prepare_ASR18_full_DATA_multiwindow_configuration_v7.py'),
        'configuration_assembler_executed':False,'actual_new_metadata_controls':record(control0),
        'actual_prior_failed_window_owned_retirement0':record(actual),'actual_retirement_prerequisite_gate':record(prerequisite),
        'actual_failed_controller1_never_reclassified':prior_failure,
        'supported_change':{'finalization_and_cleanup':'NEW sink4 stdlib bounded validation-only path; native NumPy save/arrays/take_row unchanged',
            'CPU_runtime':'Intentionally no-NumPy controller/64MiB reserve preserved; no installs or runtime changes',
            'new_source_owner':'CPU6 new owner/source/contract kinds, never oldowner3 relabeled',
            'before_GPU':'New successful CPUwindow0 must match all640 original fullNPY hashes,320 ordered row/context/fulltarget/PCM identities and20preparedpayloads',
            'old_data':'Failure1, all40normal child0 and all640hashes+full original small metadata durably retained',
            'resident_persistence_recovery':'Unchanged reviewed phase separation; every captured checkpoint actual restore0 before overall pass'},
        'unchanged_budgets':{'resident_native':5560813440,'write_and_persistence_increment':64000000,
            'resident_entire':5624813440,'fresh_final_restore_entire':3350000000,'restore_increment':350000000,
            'native_variation_margin':750000000,'CPU_whole_family':1934217728,'maximum_windows':2,'maximum_new_updates':40},
        'new_original_audio_frontend_or_GPU_trial_performed':False,
        'old640array_byte_equivalence_on_changed_producer_claimed':False,
        'full_source_resource_config_and_fresh_admission_still_required':True}
    write_json(manifest,value)
    print({'manifest':record(manifest),'bindings':len(pins),'owned_AST_sources':len(seen),
        'actual_retirement0':record(actual),'actual_metadata43_0':record(control0),'real_positive_cleanup_gate':record(prerequisite)},flush=True)


if __name__=='__main__':main()
