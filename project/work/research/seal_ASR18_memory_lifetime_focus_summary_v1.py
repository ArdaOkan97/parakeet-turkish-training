"""Consolidate completed memory evidence and one Backlog milestone; no ML."""
import hashlib,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
E=Path(r'E:\gpu_training_scratchpad');A=E/'analysis/ASR18_memory_diagnosis_v1'
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_production_state_v1 import require
def pin(path):
    path=Path(path).resolve();require(path.is_relative_to(P) or path.is_relative_to(E),'Authorized storage only');h=hashlib.sha256();st=path.stat()
    with path.open('rb') as f:
        while b:=f.read(4*1024**2):h.update(b)
    require((st.st_size,st.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns),'Immutable proof/source')
    return {'path':str(path),'bytes':st.st_size,'sha256':h.hexdigest()}
def main():
    require(os.environ['CUDA_VISIBLE_DEVICES']=='-1' and read_json(E/'storage_config.json')['storage_root']==str(E),'E-only metadata, no GPU')
    leasepath=P/'work/gpu_training_owner.json';lease=read_json(leasepath)
    require(lease['owner']=='/root/memory_diagnosis' and lease['actual_GPU_running'] is False and lease['actual_owned_worker_pid'] is None,'Exclusive owner actually idle')
    paths={
        'exact_conv_only_CNN_component':A/'frozen_CNN_lifetime_conv_only_actual_01/actual_conv_only_summary_v1.json',
        'one_short_real207_update_and_exact_fresh_restore':A/'single_TR207_actual_01/actual_short_TR207_summary_v1.json',
        'first_fixed20_failed_native_admission':A/'fixed_TR_replay207_actual_01/train/native_update_1_admission.json',
        'corrected_family_fixed20_actual1':A/'fixed_TR_replay207_actual_02/actual_owned_CLOSED1_summary_v1.json',
        'corrected_family_execution_review':A/'ROOT_fixed_TR_replay207_configuration02_family_accounting_source_resource_execution_review_v1.json',
        'official_TDT_cold':A/'official_TDT_cold_host_actual_01/actual_cold_host_summary_v1.json',
        'six_kernel_cache_no_host_saving':A/'official_TDT_cache_hit_actual_03/actual_cache_causal_summary_v1.json',
        'heaptrim_no_host_remedy':A/'official_TDT_cache_hit_actual_05/official_TDT/worker_report.json',
        'Adam_foreachFalse_numerical_failure':A/'original207_Adam_pair_actual_01/actual_optimizer_pair_summary_v1.json',
        'clip_foreachFalse_numerical_failure':A/'original207_clip_single_actual_02/canonical_raw_fingerprint_audit_v1.json',
        'teacher_parameter_storage_only':A/'teacher_lifetime_storage_metadata_v1.json'}
    sources={name:pin(path) for name,path in {
        'conv_only_CNN':S/'asr18_temporal_tiled_subsampling_v2.py',
        'original207_optimizer':S/'bpe_broad_full207_optimizer_v1.py',
        'unchanged_physical1_native':S/'asr18_precomputed_native_v2.py',
        'preservation_objective':S/'asr18_narrow_objective_v1.py',
        'streamed_source_bootstrap':S/'bpe_broad_streamed_bootstrap_v2.py',
        'fixed20_harness':P/'work/research/run_ASR18_fixed_TR_replay207_memory_v2.py',
        'fixed20_operations':P/'work/research/asr18_fixed_TR_replay207_operations_v2.py',
        'exact_unused_family_accounting':S/'asr18_exact_remaining_family_v1.py',
        'optimizer_pair_rejected_candidate':P/'work/research/run_ASR18_original207_Adam_pair_v1.py',
        'clip_only_rejected_candidate':P/'work/research/run_ASR18_original207_clip_single_v2.py',
        'canonical_full_state_fingerprint':S/'asr18_live_state_fingerprint_v1.py',
        'canonical_JSON_integer_key_verifier':S/'asr18_streamed_persistence_fingerprint_v1.py',
        'this_metadata_sealer':Path(__file__)}.items()}
    evidence={k:pin(v) for k,v in paths.items()};stamp=datetime.now(timezone.utc).isoformat()
    report={'kind':'ASR18_memory_lifetime_focus_actual_partial_result_v1','owner':'/root/memory_diagnosis','observed_at_utc':stamp,
        'focus_deadline_utc':'2026-10-18T00:16:16Z','deadline_complete':False,'production_enabled':False,'full_memory_fix_or_launch_ready':False,
        'actual_GPU_running':False,'exclusive_GPU_owner_preserved':True,'source_inventory':sources,'actual_evidence':evidence,
        'supported_remedy':{'kind':'Instance-only frozen CNN conv tiling with original full-shape streamed Linear once',
            'all10_component_full_BF16_outputs_byte_exact':True,'all4RNG_input_backing_exact':True,
            'measured8914_frame_forward_private_delta_reduction_bytes':245_850_112,
            'full15925_frame_forward_private_delta_reduction_bytes':358_268_928,
            'CUDA_only_credit_used':False,'scope':'Ten deterministic component inputs; not full native trajectory byte-equivalence'},
        'actual_short_native_boundary':{'passed':True,'owned_session':2427,'actual_tool_chunk':'080c03','native_PID':36712,'recovery_PID':18180,
            'actual_training_updates':1,'source_clocks_after':[64251,2001,62251],'all207_3Adam_all4RNG_exact_fresh_recovery':True,
            'native_sampled_private_peak_bytes':3_787_587_584,'native_OS_lifetime_through_Adam_bytes':3_765_723_136,
            'native_CUDA_peak_bytes':423_855_616,'scope':'Original1s cleanTR/T13/U6, effective1, no teacher/replay; unqualified diagnostic only',
            'writer_postwrite_OS_peak_observed':False,'fixed20_successor_adds_explicit_postwrite_OS_stage':True},
        'representative20_status':{'actual_updates':0,'actual_forward_calls':0,'teacher_present_before_ready':True,
            'first_attempt':{'owned_session':29626,'actual_tool_exit_code':1,'actual_tool_chunk':'3d5347','old_full192MB_family_overcount_preserved':True},
            'corrected_attempt':{'owned_session':17852,'actual_tool_exit_code':1,'actual_tool_chunk':'b8f652','exact_unused_family_allowance_bytes':[172_441_600,172_507_136],
                'all3_global_forecasts_denied':True,'projected_global90_excess_bytes_range':[319_340_627,390_705_235]},
            'unchanged_worker_cap_bytes':4_900_000_000,'unchanged_full_family_cap_bytes':5_092_000_000,'unchanged_variation_bytes':750_000_000,
            'source_sampler_draws_consumed':0,'all_owned_families_dead':True,'scope':'Fixed14TR+2replay, two selectedTR and all6 first replay refs; no representative20 actual update or checkpoint recovery proof'},
        'causal_negatives':{'six_TDT_cache_hits':'All6 exact cached kernels hit and strict loss/gradient/RNG exact; no host saving, only compile time benefit',
            'own_CRT_heap_trim':'About1.06MB currentprivate reduction only',
            'optional_import_filter':'About16.23MB only; no adopted training runtime change',
            'checkpoint_lazy_import':'About44MB CPU-only first-call cost; not missing600MB',
            'Adam_foreachFalse':{'both_actual_native0':True,'outer_actual1':True,'conditional_mature_Adam_private_reduction_bytes':11_182_080,
                'changed_final_weights':['decoder.embedding.weight','joint.head.weight'],'all_Adam_moments_exact':True,'adopted':False},
            'clip_foreachFalse':{'actual_native0':True,'outer_actual1':True,'norm_and_clipped_gradients_not_exact':True,
                'changed_weights':186,'changed_Adam_tensor_hashes':388,'conditional_clip_private_reduction_bytes':29_061_120,'adopted':False},
            'teacher_precompute':'Only72,514,072B parameter storage versus12,592,128B fixed6probability grids; actual host reclamation unproved, insufficient to cover deficit'},
        'remaining_requirement':'Sustained original207 teacher/all6replay updates with checkpoint1/10/20 and actual fresh full-state recoveries, then broader caller geometry/lifecycle validation. No full production or retention/quality claim.',
        'next_condition':'No unchanged retry or automatic application closure. Only human-reported AND actually verified external workload change may support fresh strict admission on exact reviewed fixed20config02; otherwise external headroom remains required.',
        'preservation':'Qualified original source weights/Adam/RNG, all failed receipts and guards retained; no source checkpoint or user application changed.'}
    target=A/'memory_lifetime_focus_actual_partial_summary_v1.json';require(not target.exists(),'Do not rewrite consolidated evidence');write_json(target,report)
    marker='ASR18-memory-focus-20261017-v1'
    note=(f'{stamp} [{marker}]: Measured frozen-CNN conv-only lifetime remedy preserved all10 fullBF16 component outputs/RNG/input/backing exactly and reduced measured8914/full15925 forward-private increments by245.85/358.27MB. '
        'Actual original-source all207 short control OWN2427/tool080c03 CLOSED0: one1s cleanTR update, mature clocks64251/2001/62251, coherent checkpoint and fresh complete207/3Adam/all4RNG restore0; sampled nativepeak3.788GB, OS-throughAdam3.766GB. '
        'This is effective1/no replay/teacher, not representative stability, quality or production. Two representative fixed20 attempts29626 and17852 actually stopped1 BEFORE anyforward/Adam; exact unused128MB-parent/64MB-ALLshim accounting corrected duplicate reservation but all3 fresh global forecasts still missed90% by319-391MB after background growth. '
        'CUDA/private guards and750MB variation stayed unchanged. All6cachedTDT kernels preserved exact outputs but saved compilation time only; heaptrim~1MB/importfilter~16MB/teacherlogicalnet~60MB do not solve deficit. '
        'Optimizer-only actual default/AdamFalse both0 then strict pair1: only~11MB conditional saving and2weight mismatches; clipFalse actual0 then strict pair1 with norm/gradient/state mismatch and~29MB saving; neither adopted. '
        'All processes closed/sole leaseidle. Full memoryfix/representative20/40/readiness remain unproved; no diagnostic promotion. Only human-reported+verified external workload change can support fresh exact admission; no automatic app closure, guard relaxation or unchanged retry. '
        f'Consolidated exact receipts/source inventory: {target}.')
    binary=S/'tools/backlog.exe';view=subprocess.run([str(binary),'task','view','ASR-18','--json'],cwd=S,check=True,capture_output=True,text=True,encoding='utf-8')
    already=marker in json.loads(view.stdout)['task'].get('implementationNotes','')
    if not already:subprocess.run([str(binary),'task','edit','ASR-18','--append-notes',note,'--add-ref',str(target)],cwd=S,check=True,capture_output=True,text=True,encoding='utf-8')
    refresh=subprocess.run([str(P/'work/nemo-native-env-v2/Scripts/python.exe'),'-X','utf8','-B',str(S/'refresh_backlog.py')],cwd=S,check=True,capture_output=True,text=True,encoding='utf-8')
    indices=[int(v.stem.rsplit('v',1)[1]) for v in A.glob('launch_readiness_v*.json') if v.stem.rsplit('v',1)[1].isdigit()];status=A/f'launch_readiness_v{max(indices)+1}.json'
    write_json(status,{'kind':'ASR18_current_memory_focus_idle_status_v1','owner':lease['owner'],'updated_at_utc':stamp,'actual_GPU_running':False,
        'actual_owned_GPU_PID':None,'production_enabled':False,'full_memory_fix_or_launch_ready':False,'deadline_complete':False,
        'consolidated_actual_summary':pin(target),'current_stage':'GPU_idle_memory_remedy_partial_short207_pass_representative20_blocked_external_headroom_required'})
    pointerpath=A/'current_launch_readiness.json';pointer=read_json(pointerpath);pointer.update(updated_at_utc=stamp,current_readiness=pin(status),
        current_stage='GPU_idle_memory_remedy_partial_short207_pass_representative20_blocked_external_headroom_required',actual_GPU_running=False,actual_owned_GPU_PID=None,
        consolidated_memory_focus_summary=pin(target));write_json(pointerpath,pointer)
    lease.update(updated_at_utc=stamp,status=pointer['current_stage'],diagnostic_phase=pointer['current_stage']);write_json(leasepath,lease)
    print(json.dumps({'summary':pin(target),'current_status':pin(status),'BacklogCLI_appended':not already,'Backlog_marker':marker,'refresh_actual_exit_code':refresh.returncode,
        'refresh_stdout':refresh.stdout.strip(),'no_ML_GPU_model_audio_payload_job':True}))
if __name__=='__main__':main()
