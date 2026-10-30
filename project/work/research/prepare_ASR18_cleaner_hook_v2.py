"""Freeze clean1/forecast/coordinator bundle for ROOT review, never launch GPU."""
import json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=Path(__file__).resolve().parent;sys.path[:0]=[str(S),str(R)]
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import write_json
from asr18_owned_source_closure_v1 import owned_closure
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');F=A.parent/'FastMSS_integration_v1'
def main():
    old=R/'run_ASR18_feature_window_native_v1.py';driver=R/'run_ASR18_cleaner_hook_v2.py'
    text=old.read_text().replace('from asr18_feature_window_native_probe_v1 import prepare_probe','from asr18_cleaner_precomputed_probe_v2 import prepare_probe')
    text=text.replace("progress['completed_updates']>0","progress['native_updates_this_child']>0")
    text=text.replace('asr18_feature_window_control_v1.disabled.json','asr18_cleaner_hook_v2.disabled.json')
    require(not driver.exists() and text!=old.read_text(),'New immutable clean diagnostic driver');driver.write_text(text,encoding='utf-8',newline='\n')
    hp=F/'FastMSS_actual_clean_feature_CPU_handoff_v1.json';h=json.loads(hp.read_text())
    require(h['passed'] and h['actual_owned_CPU_exit']['actual_exit_code']==0 and h['actual_owned_CPU_exit']['actual_tool_chunk']=='0a580c'
        and h['actual_original_frontend_exit']['actual_exit_code']==0,'Actual closed CPU/frontend source proof')
    require(all(record(v['path'])==v for v in h['source_bindings']),'Clean CPU source changed')
    plan=json.loads(Path(h['actual_native_owner_diagnostic_plan']['path']).read_text())
    calibration={'summary_binding':record(A/'feature_window_native20_01/actual_sameprocess20_summary_v1.json'),
        'archive_binding':record(F/'weak_fixed14_feature_window20_durable_metadata_after_native20_v1.json'),
        'retirement_binding':record(F/'weak_fixed14_feature_window20_actual_retirement_v1.json'),
        'native_source_binding':record(S/'asr18_precomputed_native_v2.py')}
    config={'kind':'ASR18_actual_cleaner_hook_after20_disabled_diagnostic_v1','production_enabled':False,'diagnostic_only':True,
        'maximum_native_updates':1,'maximum_wall_seconds':900,'feature_window_manifest':h['manifest'],'feature_window_owner':h['owner'],
        'actual_CPU_handoff':record(hp),'parent_checkpoint':plan['source_parent_checkpoint'],'forecast_calibration':calibration,
        'phase':'cleaner_data_fine_tuning','physical_batch':1,'TR_per_update':2,'replay_per_update':2,'full_cleaner_mean_U':11972886/352768,
        'source_original_clocks':[64250,2000,62250],'initial_checkpoint_clocks':[64270,2020,62270],'expected_final_clocks':[64271,2021,62271],
        'parent_is_unqualified_diagnostic_fork_not_production_seed':True,'production_phase_transition_claimed':False,'full_broad_corpus_passes':0,
        'parent_frontend_augmenter_transport_permitted':False,'feature_cache_cleanup_permitted_by_GPU_child':False,
        'resource_guards':{'CUDA_allocator_cap_bytes':8000000000,'full_joint_bytes':1700000000,'native_endcache_bytes':3000000000,
            'system_yield':.9,'three_fresh_observations':3,'critical':.95,'minimum_headroom_bytes':2000000000}}
    cp=S/'asr18_cleaner_hook_v2.disabled.json';require(not cp.exists(),'Fresh pinned config');write_json(cp,config)
    entries=[driver,S/'asr18_cleaner_precomputed_probe_v2.py',S/'asr18_precomputed_native_v2.py',S/'asr18_calibrated_geometry_forecast_v2.py',
        S/'asr18_metadata_checkpoint_binding_v1.py',S/'asr18_cleaner_fixture_cursor_v1.py',S/'bpe_replay_bucket_no_runtime_v1.py',
        S/'bpe_production_window_coordinator_v2.py',S/'bpe_broad_final_population_v5.py',R/'validate_ASR18_geometry_forecast_v2.py',
        R/'validate_ASR18_production_window_v2.py',R/'prepare_ASR18_post20_replay_metadata_v3.py',R/'plan_ASR18_cleaner_after20_v1.py',R/'asr18_owned_source_closure_v1.py']
    entries.extend(Path(v['path']) for v in h['source_bindings']);sources=owned_closure(entries)
    other=[cp,hp,Path(h['manifest']['path']),Path(h['owner']['path']),Path(h['update']['path']),Path(h['report']['path']),
        Path(h['actual_native_owner_diagnostic_plan']['path']),F/'FastMSS_actual_clean_feature_source_review_manifest_v1.json',
        A/'native_frontier_CPU_actual_01/report.json',A/'native_frontier_CPU_guard_01/closure.json',A/'geometry_forecast_CPU_actual_02/report.json',A/'geometry_forecast_CPU_guard_02/closure.json',
        A/'geometry_forecast_CPU_guard_01/closure.json',A/'production_window_CPU_actual_02/report.json',A/'production_window_CPU_guard_02/closure.json',
        A/'post20_replay_metadata_actual_03/report.json',A/'post20_replay_metadata_guard_03/closure.json',
        A/'cleaner_after20_plan_guard_01/closure.json',A/'feature_window_native20_01/actual_owned_CLOSED0.json',
        A/'feature_window_native20_01/actual_sameprocess20_summary_v1.json',A/'feature_window_native20_01/forecast_audit_final_v1.json',
        A/'feature_window_native20_DEV_01/actual_DEV_summary_v1.json',A/'feature_window_native20_DEV_01/actual_owned_CLOSED0.json',
        A/'precomputed_native_phase_tail_derivation_v1.json',A/'ROOT_canonical2D_source_read_review_v1.json',
        A/'ROOT_feature_window_native_conditional20_read_review_v2.json',A/'ROOT_closed_native20_serialDEV_source_read_review_v1.json']
    other.extend(Path(v['path']) for v in calibration.values() if 'path'in v)
    # Explicit producer closure/guard and source policy dependencies are also
    # pinned although the native child will not launch any CPU producer.
    for section in ('actual_owned_CPU_exit','actual_original_frontend_exit'):other.append(Path(h[section]['guard']['path']))
    sources=sorted(set(sources));other=sorted(set(other));bindings=[record(p) for p in sorted(set(sources+other))]
    manifest={'kind':'ASR18_frozen_clean1_prospective_forecast_coordinator_source_request_v2','scope':'bounded_production_caller_preparation',
        'accepted':False,'production_approval_claimed':False,'native_updates_this_child_bound':1,'initial_global_diagnostic_updates':20,
        'expected_global_updates_after':21,'sources':[record(p) for p in sources],'bindings':bindings,'owned_transitive_python_sources':len(sources),
        'config':record(cp),'CPU_feature_handoff':record(hp),'source_GPU_parent':plan['source_parent_checkpoint'],
        'new_native_math_policy_introduced':False,'new_caller_scope':'Explicit cleaner2TR+actualpost20FR/PT/fullmeans/tail; checkpoint21/exactrestore and prefix/cleanpayload resume',
        'new_forecast_basis':'Actualnative20peak4.68281344GB+750MBwarmup+64MBteacherfeatureextension; runninghighwater250MB+64MB; envelope of sharedstudentencoder1115T/419U, replayU<=75/fixed8x8teachergrid',
        'prior_role_specific_forecast_preflight_rejected_FR_T206_over190':True,'new_forecast_absolute_private_bytes':5496813440,
        'new_forecast_is_empirical_finite_admission_basis_not_every_shape_or_production_proof':True,
        'closed_CPU_controls':{'geometry_forecast':8,'window_coordinator':26,'actual_clean_lifecycle':20},
        'CPU_state_controls_are_not_native_production_or_fullpopulation_evidence':True,
        'pending_final_population_successor':'v5remainsunarmed; newdeclared-geometry/runtime-matchschemaafterDATAfinalinterface, nofullPCMscanrequired',
        'separate_future_coordinator_not_exercised_by_this_GPU_probe':record(S/'bpe_production_window_coordinator_v2.py'),
        'full_broad_corpus_passes':0,'production_cleaner_transition_claimed':False,'guards_unchanged':True,'long_production_training_started':False}
    path=A/'production_cleaner_forecast_coordinator_review_manifest_v2.json';require(not path.exists(),'Fresh immutable review bundle');write_json(path,manifest)
    print(json.dumps({'manifest':record(path),'config':record(cp),'owned_python_sources':len(sources),'total_bindings':len(bindings),'GPU_launched':False}))
if __name__=='__main__':main()
