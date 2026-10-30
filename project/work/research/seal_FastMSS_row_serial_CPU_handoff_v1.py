"""Seal source/actual-protocol evidence only; no DATA/audio/cache execution."""
import ast,json,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
import fastmss_row_serial_cpu_v1 as c

def main():
    c.environment();guard=c.resource_guard
    area=c.ROOT/'analysis/FastMSS_integration_v1';s=PROJECT/'outputs/parakeet_turkish';r=PROJECT/'work/research'
    names={'common':s/'fastmss_row_serial_cpu_v1.py','bridge':s/'fastmss_phase_aware_plan_bridge_v4.py',
        'sink':s/'fastmss_phase_aware_feature_window_v2.py','process_helper':s/'fastmss_owned_process_closure_v1.py',
        'reference_worker':r/'produce_FastMSS_independent_rows_CPU_v1.py','feature_worker':r/'produce_FastMSS_serial_features_CPU_v1.py',
        'controller':r/'prepare_FastMSS_phase_window_CPU_v1.py','guard':r/'ASR18_bounded_CPU_phase_v3.py',
        'broad_reader':s/'bpe_broad_audio_FLAC_records_v1.py','original_reader':s/'bpe_broad_audio_v1.py',
        'source_identity':s/'bpe_broad_source_identity_v1.py','replay_labels':s/'bpe_replay_cache_labels_v1.py',
        'geometry':s/'bpe_runtime_declared_geometry_v1.py','cleaner':s/'bpe_cleaner_data_v1.py',
        'clean_phase_reader':s/'fastmss_clean_phase_reader_v1.py','watchdog':s/'resource_watchdog_v2.py',
        'metadata_locks':s/'bpe_windows_final_metadata_lock_v1.py','declared_geometry_index':s/'bpe_declared_geometry_index_v1.py',
        'label_catalog':s/'bpe_training_label_catalog_v1.py','expanded_rows':s/'expanded_training_rows_v1.py',
        'scalar_metadata':s/'bpe_checkpoint_scalar_metadata_v1.py'}
    sources={k:c.record(p,guard) for k,p in names.items()}
    report=c.record(area/'row_serial_protocol_actual_05/report.json',guard)
    raw=c.read_bound(report,guard);monitor=c.record(area/'row_serial_protocol_guard_05/closure.json',guard)
    g=c.read_bound(monitor,guard)
    assert raw['passed'] is True and raw['check_count']==52 and g['actual_exit_code']==0 and g['monitor_clean'] and g['monitor_joined']
    assert all(c.same(pin,c.record(pin['path'],guard)) for pin in raw['sources'])
    receipt={'kind':'ASR18_guarded_CPU_tool_completion_v1','owner':'/root/fastmss_integration',
        'actual_native_exit_code':0,'actual_exec_tool_exit_code':0,'actual_tool_chunk_id':'9739b7',
        'native_PID':g['pid'],'report':report,'guard_closure':monitor,'checks':52,
        'peak_private_bytes':g['observed']['peak_private_bytes'],'actual_shared_CACHE_unchanged':True,
        'actual_DATA_or_frontend_or_native_training_proven':False}
    receipt_path=area/'FastMSS_row_serial_CPU_protocol_actual_CLOSED0_v1.json';c.write_new(receipt_path,receipt)
    archived=c.read_bound(c.record(area/'actual_cleaner_feature_window_durable_metadata_after_native21_v1.json',guard),guard)
    owner=archived['metadata']['owner.json']
    template={'kind':c.CONTRACT_KIND,'purpose':c.bridge.PURPOSE,'production_enabled':False,
        'TEMPLATE_NOT_LAUNCHABLE':True,'DATA_evidence_sha256':None,'outer_parent_source':None,
        'sources':sources,'python':c.record(c.CPU_ENV/'Scripts/python.exe',guard),
        'augmentation_identity':owner['augmentation_identity'],'frontend':owner['frontend'],
        'maximum_child_seconds':300,'maximum_window_updates':20,'maximum_session_windows':2,
        'maximum_session_native_updates':40,'feature_capacity_bytes':c.bridge.CAPACITY,
        'raw_PCM_capacity_bytes':c.bridge.PCM_CAPACITY,'raw_PCM_maximum_jobs':2,'raw_PCM_cache_written':False,
        'resource_review':{'passed':False,'scope':'CPU_preparation_only','native_GPU_fit_claimed':False,
            'reference_incremental_bytes':None,'frontend_incremental_bytes':None,
            'reference_maximum_private_bytes':None,'frontend_maximum_private_bytes':None,
            'controller_private_reserve_bytes':None,'shim_private_reserve_bytes':None,
            'whole_owned_private_allowance_bytes':None,'maximum_simultaneous_children':1}}
    template_path=area/'FastMSS_row_serial_CPU_contract_TEMPLATE_NOT_LAUNCHABLE_v1.json';c.write_new(template_path,template)
    interface={'kind':'FastMSS_row_serial_CPU_controller_source_interface_v1','source_only':True,'production_enabled':False,
        'controller':sources['controller'],'session_field':'CPU_preparation_contract: exact bound complete reviewed contract',
        'CLI':'CPU_python -B ASR18_bounded_CPU_phase_v3.py --folder E_guard --seconds bounded --script prepare_FastMSS_phase_window_CPU_v1.py -- --bridge <record JSON or record-JSON file> --output <new E folder> [--prior-boundary-verification <dynamic bound record JSON>]',
        'API':'main(bridge_binding, output, contract_override=None, prior_boundary_verification=None)',
        'dynamic_prior_boundary':{'required_only_window_index':1,'kind':'FastMSS_native_owner_prior_window_CPU_handoff_v1',
            'fields':['owner=/root/memory_diagnosis','passed=true','bridge exact requested binding','session exact session binding','parent_source exact contract.outer_parent_source','boundary_sha256=canonical digest of actual predecessor boundary','actual_all_state_prefix_checkpoint_verified=true']},
        'output':{'file':'report.json','fields':['owner','manifest','closures','independent_reference_proofs','frontend_proofs','bridge','contract','producer','worker_source','parent_source','actual_optimizer_updates=0','sampler_journal_frontier_advanced=false','all_children_actually_closed0=true']},
        'ordered_runtime':['validate final DATA/geometry and immutable session/contract sources before creating CACHE',
            'independent original-audio reference child: serial whole clip, actual native/rate/full16k/T/heldout + full labels + genuine replay cache bytes; metadata only',
            'observe real reference Popen0 and clean joined guard; create exact input binding',
            'serial frontend child decodes again and independently reproduces source/geometry/PCM/fulltarget evidence; no saved PCM pool',
            'weak/TR only exact frozen policy; cleaner/replay direct original clean; original CPU frontend arrays/mask retained in owned feature files',
            'observe real frontend Popen0 and clean guard, finalize sink manifest; all arrays retained until native final coherent/discard lifecycle'],
        'physical_geometry':{'physical_feature_frames':'N//160+1','valid_mask_frames':'N//160','encoder_T':'ceil(valid/8)',
            'role_physical_upper_bound':'8*T+1','trailing_masked_frame_cropped':False},
        'runtime_policy':{'CPU_Torch':'2.7.1+cpu','Torch_threads':4,'OMP_MKL_OpenBLAS_NumExpr_environment':1,
            'NumPy_SciPy':'exact frozen augmentation identity including versions; mismatch rejects',
            'reference_controller_Torch_frontend_model_imports':False,'new_environment_changes':False},
        'resource_contract':'Whole allowance >= controller reserve + shim reserve + max(reference leaf, frontend leaf); native owner adds entire allowance to resident GPU parent forecast. Actual controller polls exact owned PID family. Fresh global90/headroom2GB/E100GB and individual2GB guards unchanged.',
        'resident_idle':{'kind':'ASR18_resident_native_coherent_idle_boundary_v1','not_process_CLOSED0':True,
            'live_witness_path':'bounded E path, atomically replaced by exact native parent; immutable proof does not expire by original timestamp',
            'witness_kind':'ASR18_resident_native_live_guard_witness_v1',
            'witness_fields':['idle_proof exact binding','parent_source','producer/observer PID+PPID+create_time','request_nonce','command_state=coherent_idle_CPU_preparation','resume_command_issued=false','parent_monitor_live=true','observed_monotonic','parent_guard_observation','native_guard_observation'],
            'freshness':'witness and both PID-bound paired samples <=2 seconds; current identities live; native callback verifies exact current nonce/checkpoint/state at cleanup',
            'intermediate_checkpoint_cleanup':False,'final_whole_window_coherent_cleanup_with_idle_callback':True},
        'unchanged_final_DATA_actual_tool_requirement':True,'template':c.record(template_path,guard),
        'remaining':['genuinely final DATA/raw aggregate/whole-DATA actual0/declared coverage','reviewed concrete CPU resource allowances + outer native parent source','actual row-serial original-source/frontend positive preparation','new native caller integration/combined resident resource proof','bounded two-window checkpoint/cleanup/resume proof'],
        'quality_claim':False,'full_population_decode_validation_claim':False,'full_recipe_or_production_approval':False}
    interface_path=area/'FastMSS_row_serial_CPU_controller_source_interface_v1.json';c.write_new(interface_path,interface)
    def delta(a,b):
        def defs(p):return {n.name:ast.dump(n) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        before,after=defs(a),defs(b)
        return {'changed':[k for k in before if k in after and before[k]!=after[k]],'added':[k for k in after if k not in before],'removed':[k for k in before if k not in after]}
    manifest={'kind':'FastMSS_row_serial_CPU_controller_source_review_manifest_v1','production_enabled':False,
        'sources':sources,'fixture_source':c.record(r/'validate_FastMSS_serial_CPU_protocol_v4.py',guard),
        'actual_protocol':c.record(receipt_path,guard),'interface':c.record(interface_path,guard),
        'frozen_predecessors_unchanged':{'bridge3':c.record(s/'fastmss_phase_aware_plan_bridge_v3.py',guard),
            'sink1':c.record(s/'fastmss_phase_aware_feature_window_v1.py',guard)},
        'bridge_AST_delta':delta(s/'fastmss_phase_aware_plan_bridge_v3.py',names['bridge']),
        'sink_AST_delta':delta(s/'fastmss_phase_aware_feature_window_v1.py',names['sink']),
        'actual_whole_final_DATA_or_frontend_positive_or_GPU_tested':False,
        'shared_CACHE_mutated':False,'native_owner_source_review_required_before_actual_preparation':True}
    manifest_path=area/'FastMSS_row_serial_CPU_controller_source_review_manifest_v1.json';c.write_new(manifest_path,manifest)
    print(json.dumps({'manifest':c.record(manifest_path,guard),'interface':c.record(interface_path,guard),'actual':c.record(receipt_path,guard),'sources':{k:sources[k] for k in ('bridge','sink','process_helper','common','controller','reference_worker','feature_worker')}}))

if __name__=='__main__':main()
