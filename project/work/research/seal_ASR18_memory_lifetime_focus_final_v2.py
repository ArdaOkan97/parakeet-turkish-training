"""Seal completed bounded memory evidence and update the existing milestone."""
import argparse,copy,hashlib,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
E=Path(r'E:\gpu_training_scratchpad');A=E/'analysis/ASR18_memory_diagnosis_v1';R=P/'work/research'
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_production_state_v1 import require
from asr18_streamed_persistence_fingerprint_v1 import normalize_fingerprint
import importlib.util
spec=importlib.util.spec_from_file_location('guard',R/'run_ASR18_frozen_CNN_lifetime_pair_v8.py');H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
pin=H.record

def main():
    p=argparse.ArgumentParser();p.add_argument('--hard-actual-tool-chunk',required=True);p.add_argument('--hard-owned-session',type=int,required=True);args=p.parse_args()
    require(args.hard_actual_tool_chunk=='719ae2' and args.hard_owned_session==30150,'Own actual consumed hard TOOL0, not file inference')
    require(os.environ['CUDA_VISIBLE_DEVICES']=='-1' and read_json(E/'storage_config.json')['storage_root']==str(E),'E CPU metadata only')
    leasepath=P/'work/gpu_training_owner.json';lease=read_json(leasepath)
    require(lease['owner']=='/root/memory_diagnosis' and lease['actual_GPU_running'] is False and lease['actual_owned_worker_pid'] is None,'Exact exclusive owner actually idle')
    run=A/'single_full_hard_TR207_actual_01';overall=read_json(run/'overall_result.json');train=read_json(run/'train/training_result.json');recovery=read_json(run/'recover/recovery_result.json')
    require(overall['passed'] and overall['actual_updates']==1 and train['passed'] and train['actual_optimizer_updates']==1 and recovery['passed'],'Actual whole hard update/persistence/fresh recovery pass')
    require(recovery['all207_threeAdam_all4RNG_match_actual_PRE_save'] and recovery['sampler_snapshot_exact_unconsumed'],'Actual complete fresh hard state recovery')
    observations={};stage_summary={}
    for mode in ('prepare','train','recover'):
        process=read_json(run/mode/'actual_process0.json');guard=read_json(run/mode/'guard_closure.json')
        require(process['actual_Popen_wait_code']==0 and process['all_captured_identities_dead'] and not any(H.alive(v) for v in process['captured_identities']),'Every owned hard phase genuine0/dead')
        require(pin(process['report']['path'])==process['report'] and pin(process['guard']['path'])==process['guard'],'Exact complete report/guard bindings')
        require(guard['pid']==process['native']['pid'] and guard['actual_exit_code']==0 and guard['monitor_clean'] and guard['monitor_joined'] and guard['error'] is None,'Exact hard clean joined guard0')
        observations[mode]={'actual_process0':pin(run/mode/'actual_process0.json'),'native':process['native'],'guard0':pin(run/mode/'guard_closure.json'),
            'sampled_private_peak_bytes':guard['peak_private_bytes'],'system_commit_peak_fraction':guard['system_commit_peak_fraction'],'samples':guard['samples']}
        if mode!='prepare':
            stages=[json.loads(v) for v in (run/mode/'stages.jsonl').read_text().splitlines()]
            observations[mode].update(OS_lifetime_private_peak_bytes=max(v['private_high_water']['lifetime_private_peak_bytes'] for v in stages),CUDA_peak_allocated_bytes=max(v['cuda']['peak_allocated'] for v in stages))
            stage_summary[mode]=[{'phase':v['phase'],'current_private_bytes':v['private_high_water']['current_private_bytes'],'OS_lifetime_private_peak_bytes':v['private_high_water']['lifetime_private_peak_bytes'],'CUDA_peak_allocated_bytes':v['cuda']['peak_allocated']} for v in stages]
    require(pin(train['checkpoint']['path'])==train['checkpoint'],'Exact hard metadata signature');doc=read_json(train['checkpoint']['path'])
    require(pin(doc['payload']['path'])==doc['payload'],'Exact full hard payload signature')
    pre=normalize_fingerprint(read_json(train['PRE_save_fingerprint']['path']));require(pre['state']['updates']==1 and pre['state']['clocks']==[64251,2001,62251],'Exact hard canonical original mature frontier')
    frontier=read_json(run/'train/positive_native_frontier_immediately_after_step.json');require(frontier['actual_optimizer_updates']==1 and frontier['actual_clocks']==[64251,2001,62251] and frontier['actual_step_completed'],'Durable positive frontier before late guards')
    prepared=read_json(run/'prepare/prepared.json');require(prepared['row']['encoder_frames']==1991 and len(prepared['full_target_ids'])==574 and prepared['features']['input_features']['shape']==[1,15925,128],'Full original hard shape/targets')
    outer=read_json(run/'guard_closure.json');require(outer['actual_exit_code']==0 and outer['monitor_clean'] and outer['monitor_joined'],'Actual outer guard0')
    stamp=datetime.now(timezone.utc).isoformat()
    hard={'kind':'ASR18_single_full_hard_TR207_actual_CLOSED0_summary_v1','owner':'/root/memory_diagnosis','owned_session':args.hard_owned_session,'actual_tool_exit_code':0,
        'actual_tool_chunk_id':args.hard_actual_tool_chunk,'observed_at_utc':stamp,'overall':pin(run/'overall_result.json'),'actual_updates':1,'sampler_draws_consumed':0,
        'source_original_clocks_before':[64250,2000,62250],'source_original_clocks_after':[64251,2001,62251],'whole_audio_samples':2547840,'duration_seconds':159.24,
        'full_encoder_frames':1991,'full_target_tokens':574,'physical_frontend_frames':15925,'valid_mask_frames':15924,'source_id':'kiraat:src01792-00990',
        'heldout_current12273_reference_bank_membership_reconciled':True,'CPU_original_preparation':pin(run/'prepare/prepared.json'),'positive_frontier':pin(run/'train/positive_native_frontier_immediately_after_step.json'),
        'checkpoint':train['checkpoint'],'payload':doc['payload'],'PRE_save':train['PRE_save_fingerprint'],'fresh_recovery':pin(run/'recover/recovery_result.json'),
        'all207_threeAdam_all4RNG_and_source_snapshot_exact':True,'unchanged3GB_endcache_passed':True,'observations':observations,'stage_observations':stage_summary,
        'teacher_or_replay':False,'original_source_not_diagnostic_seed':True,'new_complete_input_CNN_whole_trajectory_byte_equivalence_claimed':False,'production_enabled':False}
    hardpath=run/'actual_full_hard_TR207_summary_v1.json';require(not hardpath.exists(),'Immutable hard summary');write_json(hardpath,hard)
    oldpath=A/'memory_lifetime_focus_actual_partial_summary_v1.json';report=copy.deepcopy(read_json(oldpath));report.update(kind='ASR18_memory_lifetime_focus_bounded_validation_final_result_v2',observed_at_utc=stamp,
        deadline_complete=False,authorized_experiments_complete=True,no_further_experiments=True,actual_GPU_running=False,
        memory_remedy_validated_in_explicit_finite_scope=True,full_memory_fix_or_launch_ready=False,production_enabled=False)
    conv=read_json(report['actual_evidence']['exact_conv_only_CNN_component']['path']);cases={v['case']['physical_frames']:v for v in conv['cases']}
    a=cases[8914]['measured_component_private_forward_delta_reduction_bytes'];b=cases[15925]['measured_component_private_forward_delta_reduction_bytes']
    require((a,b)==(245846016,358273024),'Derive corrected exact reductions from actual bound case records')
    report['supported_remedy'].update(measured8914_frame_forward_private_delta_reduction_bytes=a,full15925_frame_forward_private_delta_reduction_bytes=b,
        prior_partial_v1_two_4KiB_transcription_errors_corrected_without_rewriting_original=True,
        actual_native_integration='Applied instance-only conv tiling with original full-M Linear once to the scoped short, fixed replay20 continuation, and full hard1 diagnostics; whole-trajectory equality to old pipeline not claimed.')
    cpu=read_json(A/'CPU6_first_window_only_actual_01/report.json');equiv=read_json(cpu['exact_equivalence']['path'])
    require(cpu['passed'] and cpu['actual_GPU_or_Adam_or_authoritative_draws']==0 and equiv['passed'] and equiv['byte_exact_full_NPY_files']==640 and equiv['exact_full_rows']==320 and equiv['exact_update_payloads']==20,'Actual corrected CPU6 stdlib finalization and exact reprepare proof')
    cpu_family=read_json(A/'CPU6_first_window_only_actual_01/CPU_window_0/owned_CPU_family_completion.json')
    fixedpath=A/'fixed_TR_replay207_resume19_actual_01/actual_fixed20_continuation_summary_v1.json';fixed=read_json(fixedpath)
    require(fixed['actual_tool_exit_code']==0 and fixed['actual_total_contiguous_updates']==20 and fixed['actual_new_updates']==19 and fixed['final_original_source_clocks']==[64270,2020,62270],'Exact source1 plus new19/fresh three capture proof')
    report['representative20_status']={'passed':True,'actual_total_contiguous_updates':20,'old_partial_actual_updates':1,'old_partial_actual_TOOL_exit_code':1,'new_same_process_updates':19,
        'twenty_same_process_updates_claimed':False,'source_clocks_after':[64270,2020,62270],'selected_original_TR_rows':2,'selected_genuine_replay_languages':['de','en','es','fr','it','pt'],
        'physical_batch':1,'effective14TR_plus2replay':16,'source_sampler_draws_consumed':0,'captured_checkpoints':[1,10,20],'all3_fresh_recoveries_exact':True,
        'native_OS_private_peak_bytes':fixed['observations']['train']['OS_lifetime_private_peak_bytes'],'native_CUDA_peak_allocated_bytes':fixed['observations']['train']['CUDA_peak_allocated_bytes'],
        'fresh_recovery_OS_private_peak_bytes':fixed['observations']['recover']['OS_lifetime_private_peak_bytes'],'scope':'Fixed8 source rows only; not corpus passes, replay maxima, full quality/retention or production.'}
    report['CPU6_corrected_finalization']={'owned_session':99618,'actual_tool_exit_code':0,'actual_tool_chunk_id':'74a124','all40_reference_frontend_children_actual0':True,
        'exact_full_NPY_files':640,'exact_ordered_rows':320,'exact_update_payloads':20,'measured_CPU_child_family_private_peak_bytes':cpu_family['measured_family_peak_private_bytes'],
        'prepared_window_retained_uncommitted':True,'actual_GPU_Adam_authoritative_draws':0}
    report['full_hard_original207_control']={'passed':True,'summary':pin(hardpath),'full_samples':2547840,'full_T':1991,'full_U':574,'effective_rows':1,
        'native_OS_private_peak_bytes':observations['train']['OS_lifetime_private_peak_bytes'],'native_CUDA_peak_allocated_bytes':observations['train']['CUDA_peak_allocated_bytes'],
        'fresh_all207_threeAdam_all4RNG_source_snapshot_recovery_exact':True,'teacher_or_replay':False}
    externalpath=A/'external_workload_change_fixed20_readmission_01.json';report['actual_external_workload_change']={'evidence':pin(externalpath),
        'human_reported_intent':True,'actual_material_change_verified':True,'conservative_prior_idle_commit_bytes':30400000000,'verified_maximum_idle_commit_bytes':25218633728,
        'verified_drop_bytes':5181366272,'three_fulljob_fresh_admissions_passed':True,'automatic_user_application_closure':False,'specific_application_cause_inferred':False}
    report['actual_evidence'].update(prior_partial_summary_preserved=pin(oldpath),CPU6_actual_report=pin(A/'CPU6_first_window_only_actual_01/report.json'),
        CPU6_exact640_320_20=pin(cpu['exact_equivalence']['path']),fixed20_continuation=pin(fixedpath),full_hard1=pin(hardpath),verified_external_workload_change=pin(externalpath))
    for name,path in {'resume19_harness':R/'run_ASR18_fixed_TR_replay207_resume19_v2.py','resume19_operations':R/'asr18_fixed_TR_replay207_resume19_operations_v2.py',
        'hard_harness':R/'run_ASR18_single_full_hard_TR207_memory_v3.py','CPU6_only_wrapper':R/'run_ASR18_CPU6_first_window_only_v1.py','final_sealer':Path(__file__)}.items():report['source_inventory'][name]=pin(path)
    report['remaining_requirement']='Integrate validated conv-only lifetime path into full population caller; genuine resident multiwindow CPU/native/checkpoint lifecycle, full phase/tail callbacks and optimizer-free serial350DEV/winner/durable resume remain to prove. Earlier DE/PT retention failures remain; no quality/retention gain or production admission from these diagnostics.'
    report['next_condition']='No further experiments in this focus session. Future production or changed full-caller jobs require exact source/config/resource review and fresh unchanged system/private/CUDA guards; no diagnostic checkpoint promotion.'
    report['interpretation']={'measured_causes':'Large activation overlap was reduced by conv-only tiling; independent probes measured persistent cold CUDA/BLAS/TDT host commitment. No claim of identifying every Windows commit contributor.',
        'empirical_native_results_are_not_additive_component_forecast_credit':True,'external_headroom_was_material_to_admission':True,'all_failed_forecasts_and_numerical_candidate_rejections_preserved':True,
        'no_foreachFalse_noLt_or_cuDNNv7_adopted':True,'unchanged750MB_variation_global90_95_headroom2GB_E100GB_CUDA8GB_guards':True}
    target=A/'memory_lifetime_focus_bounded_validation_final_summary_v2.json';require(not target.exists(),'Immutable final successor');write_json(target,report)
    binary=S/'tools/backlog.exe';view=subprocess.run([str(binary),'task','view','ASR-18','--json'],cwd=S,check=True,capture_output=True,text=True,encoding='utf-8');notes=json.loads(view.stdout)['task'].get('implementationNotes','')
    require(notes.count('ASR18-memory-focus-20261017-v1')==1,'Update the ONE existing milestone; never append a duplicate marker')
    note=(f'{stamp}: Final bounded-validation update to the existing memory-focus milestone above. Correct exact CNN8914/15925 component private-increment reductions are245,846,016/358,273,024B (prior JSON had two4KiB transcription errors; rounded note was correct). '
        'ActualCPU-only OWN99618/tool74a124 CLOSED0 proved CPU6/sink4 stdlib finalization, all40 source/frontend children0 and exact640NPY/320orderedrows/20payload equality; retained uncommitted/noGPU/Adam/draw. '
        'Human-directed external workload change was actually verified: idlecommit<=25.219GB versus conservative30.4GB; no automatic app closure or specificapp attribution. Original fixed4.9GB attempt OWN86866/tool7d0aae performed1update/checkpoint1 then correctly stopped1 beforeupdate2 because measuredpeak+unchanged750MB exceeded that forecast. '
        'NEW exact continuation OWN59114/tooldc905b restored PRE1/all207/3Adam/all4RNG/sourceprefix, then19sameprocess updates2..20; total20 across2trainingprocesses, finalmatureclocks64270/2020/62270. Old1/new10/20checkpoint lineage and all3 freshrestores0 exact. NativeOSpeak4.356GB/CUDA0.566GB. '
        'Full159.24s/T1991/U574 TRAIN control OWN30150/tool719ae2 CLOSED0: one original-source all207/threeAdam update, coherent checkpoint/fresh complete restore0; no truncation/targets lost, no replay. '
        f'NativeOSpeak{observations["train"]["OS_lifetime_private_peak_bytes"]/1e9:.3f}GB/CUDA{observations["train"]["CUDA_peak_allocated_bytes"]/1e9:.3f}GB. '
        'All750MB/global/private/CUDA/storage guards preserved; rejected foreach/noLt/v7 and old failures remain unadopted. This validates the stated finite memory remedy; fullcaller conv integration, real multiwindow/phase lifecycle and serialDEV/quality/retention/production readiness remain unproved. No diagnostic promotion or further experiments. '
        f'Final source-bound successor: {target}.')
    update=subprocess.run([str(binary),'task','edit','ASR-18','--append-notes',note,'--add-ref',str(target)],cwd=S,check=True,capture_output=True,text=True,encoding='utf-8')
    refresh=subprocess.run([str(P/'work/nemo-native-env-v2/Scripts/python.exe'),'-X','utf8','-B',str(S/'refresh_backlog.py')],cwd=S,check=True,capture_output=True,text=True,encoding='utf-8')
    indices=[int(v.stem.rsplit('v',1)[1]) for v in A.glob('launch_readiness_v*.json') if v.stem.rsplit('v',1)[1].isdigit()];status=A/f'launch_readiness_v{max(indices)+1}.json'
    write_json(status,{'kind':'ASR18_memory_focus_bounded_validation_idle_status_v2','owner':lease['owner'],'updated_at_utc':stamp,'actual_GPU_running':False,'actual_owned_GPU_PID':None,
        'production_enabled':False,'full_memory_fix_or_launch_ready':False,'no_further_experiments':True,'deadline_complete':False,'consolidated_final_summary':pin(target),
        'current_stage':'GPU_idle_bounded_CNN_CPU6_fixed20_recovery_fullhard1_proofs_passed_production_integration_pending'})
    pointerpath=A/'current_launch_readiness.json';pointer=read_json(pointerpath);pointer.update(updated_at_utc=stamp,current_readiness=pin(status),current_stage=read_json(status)['current_stage'],
        actual_owned_session=None,actual_GPU_running=False,actual_owned_GPU_PID=None,current_run_folder=str(run),consolidated_memory_focus_summary=pin(target),hard1_actual_summary=pin(hardpath))
    write_json(pointerpath,pointer)
    lease.update(updated_at_utc=stamp,status=pointer['current_stage'],diagnostic_phase=pointer['current_stage'],actual_training_optimizer_running=False,
        actual_GPU_running=False,actual_owned_worker_pid=None,current_evidence=pin(target),production_enabled=False)
    write_json(leasepath,lease)
    receipt=A/'memory_lifetime_focus_final_BacklogCLI_refresh_actual_v2.json';write_json(receipt,{'summary':pin(target),'hard_summary':pin(hardpath),'status':pin(status),
        'BacklogCLI_actual_exit_code':update.returncode,'refresh_actual_exit_code':refresh.returncode,'refresh_stdout':refresh.stdout.strip(),'new_milestone_created':False,
        'existing_milestone_marker':'ASR18-memory-focus-20261017-v1','same_milestone_final_update_appended_once':True,'history_preserved':True,'source':pin(__file__)})
    print(json.dumps({'summary':pin(target),'hard_summary':pin(hardpath),'status':pin(status),'BacklogCLI_refresh_receipt':pin(receipt),'BacklogCLI_exit':update.returncode,'refresh_exit':refresh.returncode}))

if __name__=='__main__':main()
