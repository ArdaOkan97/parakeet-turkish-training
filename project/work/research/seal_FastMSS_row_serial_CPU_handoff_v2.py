"""Freeze NEW bounded-raw CPU successor and actual focused-control evidence."""
import json,sys
from copy import deepcopy
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
import fastmss_row_serial_cpu_v2 as c

def main():
    c.environment();guard=c.resource_guard;F=c.ROOT/'analysis/FastMSS_integration_v1';R=P/'work/research'
    old=c.read_bound(c.record(F/'FastMSS_row_serial_CPU_contract_TEMPLATE_NOT_LAUNCHABLE_v1.json',guard),guard)
    paths={key:Path(pin['path']) for key,pin in old['sources'].items()}
    paths.update(common=S/'fastmss_row_serial_cpu_v2.py',reference_worker=R/'produce_FastMSS_independent_rows_CPU_v2.py',
        feature_worker=R/'produce_FastMSS_serial_features_CPU_v2.py',controller=R/'prepare_FastMSS_phase_window_CPU_v2.py',
        raw_lifetime=S/'fastmss_raw_lifetime_v1.py',bounded_decoder=S/'fastmss_bounded_original_audio_v1.py',
        physical_geometry=S/'asr18_physical_feature_geometry_v2.py',family_observer=S/'asr18_owned_family_observation_v2.py',
        scipy_signal_backend=c.TRAINING_SITE/'scipy/signal/_signaltools.py',scipy_polyphase_backend=c.TRAINING_SITE/'scipy/signal/_upfirdn.py',
        scipy_FIR_backend=c.TRAINING_SITE/'scipy/signal/_fir_filter_design.py',soundfile_backend=c.TRAINING_SITE/'soundfile.py')
    sources={key:c.record(path,guard) for key,path in paths.items()}
    report=c.record(F/'raw_lifetime_controls_actual_03/report.json',guard);raw=c.read_bound(report,guard)
    closure=c.record(F/'raw_lifetime_controls_guard_03/closure.json',guard);monitor=c.read_bound(closure,guard)
    assert raw['passed'] is True and raw['check_count']==52 and monitor['actual_exit_code']==0 and monitor['monitor_clean'] is True and monitor['monitor_joined'] is True
    assert all(c.same(pin,c.record(pin['path'],guard)) for pin in raw['sources'])
    receipt={'kind':'ASR18_guarded_CPU_tool_completion_v1','owner':'/root/fastmss_integration',
        'actual_native_exit_code':0,'actual_exec_tool_exit_code':0,'actual_tool_chunk_id':'285f00','native_PID':monitor['pid'],
        'report':report,'guard_closure':closure,'check_count':52,'coordinator_peak_private_bytes':monitor['observed']['peak_private_bytes'],
        'valid_samples':monitor['observed']['samples'],'controlled_owned_CPU_family_peak_bytes':raw['actual_owned_orphan_family_peak_bytes'],
        'raw_peak_reserved_bytes':raw['raw_receipt']['peak_reserved_raw_bytes'],'raw_peak_jobs':raw['raw_receipt']['peak_simultaneous_jobs'],
        'normal_orphaned_leaf_exit0_claimed':False,'real_DATA_audio_frontend_model_GPU_or_CACHE_proven':False}
    actual_path=F/'FastMSS_bounded_raw_CPU_controls_actual_CLOSED0_v1.json';c.write_new(actual_path,receipt)
    template=deepcopy(old);template.update(kind=c.CONTRACT_KIND,TEMPLATE_NOT_LAUNCHABLE=True,sources=sources,
        DATA_evidence_sha256=None,outer_parent_source=None)
    template['raw_lifetime_review']={'kind':'FastMSS_preallocation_raw_lifetime_review_v1','accepted':False,
        'raw_capacity_bytes':134217728,'maximum_jobs':2,'compressed_media_per_job_bytes':33554432,
        'maximum_runtime_channels':None,'maximum_decoder_frame_samples':None,'decoder_workspace_allowance_bytes':None,
        'transform_workspace_allowance_bytes':None,'frontend_workspace_allowance_bytes':None,'numerical_runtime_reserve_bytes':None}
    template['resource_review']['outer_controller_launcher_reserve_bytes']=None
    template_path=F/'FastMSS_row_serial_CPU_contract_TEMPLATE_NOT_LAUNCHABLE_v2.json';c.write_new(template_path,template)
    interface={'kind':'FastMSS_bounded_raw_row_serial_CPU_controller_source_interface_v2','production_enabled':False,
        'sources':{key:sources[key] for key in ('common','reference_worker','feature_worker','controller','raw_lifetime','bounded_decoder')},
        'contract_kind':c.CONTRACT_KIND,'request_kind':c.REQUEST_KIND,
        'CLI':'CPU_python -B ASR18_bounded_CPU_phase_v3.py --folder E_guard --seconds bounded --script prepare_FastMSS_phase_window_CPU_v2.py -- --bridge <bound-record JSON or record-JSON file> --output <new E folder> [--prior-boundary-verification <dynamic bound record JSON>]',
        'unchanged_outer_interface':'report.json owner/manifest/closures/independent_reference_proofs/frontend_proofs; actual source/context/fulltargets/provisional-state semantics unchanged; bridge4/sink2 remain frozen',
        'ROOT_resource_review':'Exact source-bound ROOT CPU resource-review file must contain raw_lifetime_review; native assembler copies it verbatim into contract/config and retains entire review pin. No arbitrary CLI object or implicit default.',
        'raw_lifetime_review_fields':template['raw_lifetime_review'],
        'allocation_admission':['Exact source-declared native/count/rate/ceil16k geometry yields complete per-row raw peak BEFORE decoder allocation; clean WAV header yields exact full sample count before read.',
            'Ledger admits at most128MiB summed raw envelopes and two jobs. Current controller runs one serial CPU leaf and observes actual prior exit, so there is no reference/frontend cross-stage raw overlap.',
            'KIRAAT actual mono24k/full FLAC frames checked before read. YODAS codec metadata/first decoded frame rate/channel/layout/format/sample cap checked before np.empty, every later frame before ndarray conversion.',
            'Native mono + retained/current decoder frames + resample output/filter padding + clean/prepared + RMS/noise scratch + original frontend waveform copies are declared raw envelopes; full returned clean/prepared buffers cannot release reservation until weakrefs are actually dead.',
            'Compressed input extent remains <=32MiB; borrowed seekable FLAC view avoids whole BytesIO copy, callback read/readinto<=1MiB. Source I/O/codec/FFT/spectral frontend workspace and numerical-library overhead are separate explicit private-budget domains.',
            'Per-row raw+media+workspace+runtime projection must fit current leaf-private ceiling AND fresh system90/headroom2GB before allocation; all existing individual2GB/E100GB/threadcaps remain.',
            'Unsupported counts/rates/channel/frame/envelope are unresolved BEFORE any draw; never crop, skip, filter, silently relax, or claim full-corpus fit.'],
        'row_and_source_repairs':{'row':'normalize_declared_TR reproduces complete planner4 row, including exact declared_frontend_geometry/from_samples/source SHA/finite-validation flag; equality remains strict, no arbitrary field dropping.',
            'source':'canonical dataset path is original KIRAAT Parquet path or E datasets/yodas3/audio/{four-digit shard}.tar; actual decoder source SHA/path equals final inventory; source/index/receipt or original FLAC carrier records/index/receipt retained.'},
        'canonical_hash':'<=65536 sample chunks use exact original float32 multiply32768/rint/clip/i16 bytes; independent F32 SHA uses memoryviews. Both inherited whole-wave hash and common whole-wave scratch paths are replaced.',
        'raw_CPU_results':'Each successful reference/frontend report includes raw_lifetime_receipt with preallocation/stage/buffer-release events, reserved peaks, no frontier commit.',
        'inner_failure_cleanup':'Controller2 captures exact Popen ancestry using frozen owned-family helper2; failure cleanup acts even after launcher exit and observes all captured PID/create-time identities dead; truthful failure metadata never claims normal leaf0.',
        'policy_scope':'Existing exact frozen weak/TR-only Gaussian+twoRIR pilot, >120s clean bypass; cleaner/replay/teacher/evaluation protections unchanged. No new assets or policy/quality claim.',
        'unchanged_original_math':'Original whole FLAC/Opus PTS selection + scipy.resample_poly Kaiser5 constant/full ceil length + original CPU frontend physical N//160+1/valid N//160 arrays; no new source trim/normalization.',
        'actual_controls':c.record(actual_path,guard),'template':c.record(template_path,guard),
        'remaining':['ROOT complete NEW source review','genuine final DATA/raw aggregate/actual tool0/declared coverage','ROOT explicit raw/workspace/whole-family resource review and exact configuration','actual original-source bounded decoder/frontend positive proof','native multiwindow lifecycle and forecast proof'],
        'genuine_final_DATA_tool0_requirement_relaxed':False,'actual_full_corpus_raw_fit_claimed':False,'actual_original_audio_frontend_GPU_or_training_tested':False}
    interface_path=F/'FastMSS_bounded_raw_row_serial_CPU_source_interface_v2.json';c.write_new(interface_path,interface)
    preserved={key:c.record(path,guard) for key,path in {'common1':S/'fastmss_row_serial_cpu_v1.py',
        'controller1':R/'prepare_FastMSS_phase_window_CPU_v1.py','reference1':R/'produce_FastMSS_independent_rows_CPU_v1.py',
        'frontend1':R/'produce_FastMSS_serial_features_CPU_v1.py','bridge4':S/'fastmss_phase_aware_plan_bridge_v4.py',
        'sink2':S/'fastmss_phase_aware_feature_window_v2.py','caller4':R/'run_ASR18_full_DATA_multiwindow_v4.py'}.items()}
    manifest={'kind':'FastMSS_bounded_raw_row_serial_CPU_source_review_manifest_v2','production_enabled':False,
        'sources':sources,'fixture_source':c.record(R/'validate_FastMSS_raw_lifetime_CPU_v3.py',guard),
        'actual_controls':c.record(actual_path,guard),'interface':c.record(interface_path,guard),'template':c.record(template_path,guard),
        'frozen_predecessors_preserved':preserved,'whole_DATA_actual_tool_proof_unchanged':True,
        'no_original_audio_frontend_CACHE_GPU_preparation':True,'full_corpus_supported_or_quality_gain_or_production_approval_claimed':False}
    manifest_path=F/'FastMSS_bounded_raw_row_serial_CPU_source_review_manifest_v2.json';c.write_new(manifest_path,manifest)
    print(json.dumps({'manifest':c.record(manifest_path,guard),'interface':c.record(interface_path,guard),'template':c.record(template_path,guard),'actual':c.record(actual_path,guard)}))

if __name__=='__main__':main()
