"""Real owned native wait0 and exact checkpoint continuation; no hotpath supervisor."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import psutil
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path[:0]=[str(S),str(P/'work/research')]
NINTH_TRANSITION_FOLDER=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_source36_original_interval186563_failure_actual_v1')
sys.path.insert(0,str(NINTH_TRANSITION_FOLDER))
THROUGHPUT_FOLDER=Path('E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\one_postDEV_gc_refresh_v1\\next200000_throughput_repair_v1')
sys.path.insert(0,str(THROUGHPUT_FOLDER))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from asr18_memory_diagnosis_v2 import storage


def main(a):
    storage();require(sys.flags.utf8_mode==1 and os.environ.get('PYTHONUTF8')=='1','UTF8 before owner entry')
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    review=read_json(a.review);driver=P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v38.py'
    require(review['accepted'] is True and review['reviewer']=='/root'
        and review['driver_source']==record(driver)
        and review['RAM_native_source']==record(S/'asr18_BPE9223_grouped_RAM_native_v1.py')
        and review['CPU_source']==record(S/'fastmss_resident_feature_stream_v14.py')
        and review['DEV_helper_source']==record(S/'asr18_resident_serial350_DEV_v2.py')
        and review['source_configuration']==record(a.configuration)
        and review['initial_checkpoint']==record(a.pointer) and review['latest_failure']==record(a.latest_failure),
        'Exact one ROOT source/science/real CPU output execution binding')
    require(a.queue_bytes==review['bounded_RAM_queue_bytes']==268435456 and review['periodic_DEV_interval_updates']==4000,
        'Unchanged bounded queue and periodic cadence')
    candidate_configuration=read_json(a.configuration)
    from asr18_current_plan_throughput_checkpoint_transition_v3 import (checked_current_plan_throughput_policy,
        checked_current_plan_throughput_checkpoint_resume,legacy_source37_configuration)
    throughput_active='authoritative_current_plan_identity_and_main_path_attribution_policy'in candidate_configuration
    if throughput_active:checked_current_plan_throughput_policy(candidate_configuration)
    configuration=legacy_source37_configuration(candidate_configuration)if throughput_active else candidate_configuration
    policy=configuration['private_scalar_prefetch_policy']
    from asr18_ninth_original_interval_checkpoint_resume_v1 import (checked_ninth_original_policy,checked_ninth_original_resume,legacy_source36_policy_configuration)
    ninth_policy_active='prior_resident_RAM_ninth_original_interval_failure'in configuration
    if ninth_policy_active:checked_ninth_original_policy(configuration)
    original_gc_policy_configuration=legacy_source36_policy_configuration(configuration)if ninth_policy_active else configuration
    from asr18_one_postDEV_gc_refresh_checkpoint_transition_v2 import (checked_one_postDEV_refresh_policy,
        checked_one_postDEV_refresh_checkpoint_resume,legacy_source33_configuration)
    checked_one_postDEV_refresh_policy(original_gc_policy_configuration)
    legacy_configuration=legacy_source33_configuration(original_gc_policy_configuration)
    require(review['one_postDEV_gc_refresh_policy']==configuration['one_postDEV_gc_refresh_policy'],
        'Exact one-event GC lifecycle and original byte/science/guard policy')
    require(review['private_scalar_prefetch_source']==policy['source']==record(S/'asr18_BPE9223_private_scalar_prefetch_v1.py')
        and review['private_scalar_prefetch_actual_CPU_review']==policy['actual_CPU_control_review']
        and record(policy['actual_CPU_control_review']['path'])==policy['actual_CPU_control_review']
        and review['additional_prefetch_host_reservation_bytes']==policy['additional_host_reservation_bytes']==268435456,
        'Exact integrated helper, finite equality review and full added host reservation')
    require(review['periodic_DEV_CPU_quiescence_policy']==configuration['periodic_DEV_CPU_quiescence_policy']
        and configuration['periodic_DEV_CPU_quiescence_policy']['source']==record(S/'asr18_resident_periodic_CPU_quiescence_v3.py'),
        'Exact reviewed periodic CPU phase scheduling; unchanged DEV and full reservations')
    from asr18_fresh_signature_collect_checkpoint_transition_v1 import checked_fresh_signature_collect_policy,checked_fresh_signature_collect_checkpoint_resume
    checked_fresh_signature_collect_policy(configuration)
    require(review['fresh_metadata_signature_and_collect_observation_policy']==configuration['fresh_metadata_signature_and_collect_observation_policy'],
        'Exact reviewed fresh signatures and observational original collect policy')
    from asr18_canonical_archive_checkpoint_transition_v1 import checked_archive_representation_policy,checked_canonical_archive_checkpoint_resume
    checked_archive_representation_policy(configuration)
    require(review['prepared_metadata_archive_representation_policy']==configuration['prepared_metadata_archive_representation_policy'],
        'Exact reviewed canonical archive representation; all old protections retained')
    require(a.postrestore_selector_resume is None or (a.cushion225_resume is not None and a.prior_process is None),
        'Exact startup repair cannot enter a normal or unrelated predecessor lane')
    if a.prior_process is not None:
        require(a.throughput_resume is None and a.ninth_original_resume is None and a.gc_refresh_resume is None and a.eighth_original_resume is None and a.fresh_signature_resume is None and a.canonical_archive_resume is None and a.lookahead_resume is None and a.cushion225_resume is None and a.packet_credit_resume is None and a.native_headroom_resume is None and a.abnormal_resume is None and a.original_interval_resume is None and a.source_transition_resume is None and a.DEV_frontend_resume is None and a.DEV_admission_resume is None and a.periodic_quiescence_resume is None,'New normal-boundary entry requires genuine prior wait0, not historical failure relabel')
    elif a.throughput_resume is not None:
        require(throughput_active and all(x is None for x in (a.ninth_original_resume,a.gc_refresh_resume,a.eighth_original_resume,
            a.fresh_signature_resume,a.canonical_archive_resume,a.lookahead_resume,a.cushion225_resume,a.postrestore_selector_resume,
            a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume,a.original_interval_resume,a.source_transition_resume,
            a.DEV_frontend_resume,a.DEV_admission_resume,a.periodic_quiescence_resume))
            and review['throughput_resume']==record(a.throughput_resume)
            and review['throughput_transition_validator']==record(THROUGHPUT_FOLDER/'asr18_current_plan_throughput_checkpoint_transition_v3.py')
            and review['authoritative_current_plan_identity_and_main_path_attribution_policy']==candidate_configuration['authoritative_current_plan_identity_and_main_path_attribution_policy']
            and a.start_invocation==review['initial_owner_invocation']==27,
            'Only reviewed current source37 post204000DEV cooperative checkpoint transition, no normal0 alias')
    elif a.ninth_original_resume is not None:
        require(all(x is None for x in (a.gc_refresh_resume,a.eighth_original_resume,a.fresh_signature_resume,
            a.canonical_archive_resume,a.lookahead_resume,a.cushion225_resume,a.postrestore_selector_resume,
            a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume,a.original_interval_resume,
            a.source_transition_resume,a.DEV_frontend_resume,a.DEV_admission_resume,a.periodic_quiescence_resume))
            and review['ninth_original_resume']==record(a.ninth_original_resume)
            and review['ninth_original_transition_validator']==record(NINTH_TRANSITION_FOLDER/'asr18_ninth_original_interval_checkpoint_resume_v1.py')
            and review['original_clip_geometry_overlay']==configuration['original_clip_geometry_overlay']
            and review['packet_credit_policy']==configuration['resident_RAM_packet_credit_policy']
            and review['inherited_one_postDEV_refresh_opportunity_policy']==configuration['inherited_one_postDEV_refresh_opportunity_policy']
            and a.start_invocation==review['initial_owner_invocation']==26,
            'Only exact source36 interval186563 saved186400 ninth repair and consumed180k provenance')
    elif a.gc_refresh_resume is not None:
        require(all(x is None for x in (a.eighth_original_resume,a.fresh_signature_resume,a.canonical_archive_resume,a.lookahead_resume,
            a.cushion225_resume,a.postrestore_selector_resume,a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume,
            a.original_interval_resume,a.source_transition_resume,a.DEV_frontend_resume,a.DEV_admission_resume,a.periodic_quiescence_resume))
            and review['gc_refresh_resume']==record(a.gc_refresh_resume)
            and review['gc_refresh_transition_validator']==record(S/'asr18_one_postDEV_gc_refresh_checkpoint_transition_v2.py')
            and a.start_invocation==review['initial_owner_invocation']==25,
            'Only exact reviewed actual source33 intentional checkpoint stop, never normal0/historical failure')
    elif a.eighth_original_resume is not None:
        require(all(x is None for x in (a.fresh_signature_resume,a.canonical_archive_resume,a.lookahead_resume,a.cushion225_resume,
            a.postrestore_selector_resume,a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume,a.original_interval_resume,
            a.source_transition_resume,a.DEV_frontend_resume,a.DEV_admission_resume,a.periodic_quiescence_resume))
            and review['eighth_original_resume']==record(a.eighth_original_resume)
            and review['eighth_original_transition_validator']==record(Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_original_interval170469_failure_actual_v1/asr18_eighth_original_interval_checkpoint_resume_v1.py'))
            and review['original_clip_geometry_overlay']==configuration['original_clip_geometry_overlay']
            and review['packet_credit_policy']==configuration['resident_RAM_packet_credit_policy']
            and a.start_invocation==review['initial_owner_invocation']==24,
            'Only exact reviewed actual source31 interval failure, CPU13 and completed eighth ingress proof')
    elif a.fresh_signature_resume is not None:
        require(all(x is None for x in (a.canonical_archive_resume,a.lookahead_resume,a.cushion225_resume,a.postrestore_selector_resume,
            a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume,a.original_interval_resume,a.source_transition_resume,
            a.DEV_frontend_resume,a.DEV_admission_resume,a.periodic_quiescence_resume))
            and review['fresh_signature_resume']==record(a.fresh_signature_resume)
            and review['fresh_signature_transition_validator']==record(S/'asr18_fresh_signature_collect_checkpoint_transition_v1.py')
            and review['human_approval']==record(Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/ROOT_direct_human_then_fix_authorization_v1.json'))
            and a.start_invocation==review['initial_owner_invocation']==23,
            'Only exact reviewed actual source30 signature/collect transition, never normal0')
    elif a.canonical_archive_resume is not None:
        require(all(x is None for x in (a.lookahead_resume,a.cushion225_resume,a.postrestore_selector_resume,a.packet_credit_resume,
            a.native_headroom_resume,a.abnormal_resume,a.original_interval_resume,a.source_transition_resume,
            a.DEV_frontend_resume,a.DEV_admission_resume,a.periodic_quiescence_resume))
            and review['canonical_archive_resume']==record(a.canonical_archive_resume)
            and review['canonical_archive_transition_validator']==record(S/'asr18_canonical_archive_checkpoint_transition_v1.py')
            and review['human_approval']==record(Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/ROOT_direct_human_then_fix_authorization_v1.json'))
            and a.start_invocation==review['initial_owner_invocation']==22,
            'Only exact reviewed actual source29 archive transition; never normal0')
    elif a.lookahead_resume is not None:
        require(all(x is None for x in (a.cushion225_resume,a.postrestore_selector_resume,a.packet_credit_resume,
            a.native_headroom_resume,a.abnormal_resume,a.original_interval_resume,a.source_transition_resume,
            a.DEV_frontend_resume,a.DEV_admission_resume,a.periodic_quiescence_resume))
            and review['lookahead_resume']==record(a.lookahead_resume)
            and review['lookahead_transition_validator']==record(S/'asr18_approved_lookahead_checkpoint_transition_v3.py')
            and review['human_approval']==configuration['resident_RAM_packet_credit_policy']['human_approval']
            and review['packet_credit_policy']==configuration['resident_RAM_packet_credit_policy']
            and review['native_update_commit_pressure_policy']==configuration['native_update_commit_pressure_policy']
            and a.start_invocation==review['initial_owner_invocation']==21,
            'Only exact reviewed approved intentional source27 checkpoint transition, never normal0')
    elif a.cushion225_resume is not None:
        require(a.packet_credit_resume is None and a.native_headroom_resume is None and a.abnormal_resume is None
            and a.original_interval_resume is None and a.source_transition_resume is None and a.DEV_frontend_resume is None
            and a.DEV_admission_resume is None and a.periodic_quiescence_resume is None
            and review['cushion225_resume']==record(a.cushion225_resume)
            and review['native_update_commit_pressure_policy']==configuration['native_update_commit_pressure_policy']
            and review['packet_credit_policy']==configuration['resident_RAM_packet_credit_policy']
            and a.start_invocation==review['initial_owner_invocation']==(20 if a.postrestore_selector_resume is not None else 19)
            and (a.postrestore_selector_resume is None or review['postrestore_selector_resume']==record(a.postrestore_selector_resume)),
            'Exact source24 TOOL1 saved156600 plus human-approved225 cushion, never normal0')
    elif a.packet_credit_resume is not None:
        require(a.native_headroom_resume is None and a.abnormal_resume is None and a.original_interval_resume is None
            and a.source_transition_resume is None and a.DEV_frontend_resume is None and a.DEV_admission_resume is None
            and a.periodic_quiescence_resume is None and review['packet_credit_resume']==record(a.packet_credit_resume)
            and review['packet_credit_policy']==configuration['resident_RAM_packet_credit_policy']
            and review['native_update_commit_pressure_policy']==configuration['native_update_commit_pressure_policy']
            and a.start_invocation==review['initial_owner_invocation']==18,
            'Exact source22 added-buffer TOOL1 saved156400 and reviewed end-to-end credit, never normal0')
    elif a.native_headroom_resume is not None:
        require(a.abnormal_resume is None and a.original_interval_resume is None and a.source_transition_resume is None
            and a.DEV_frontend_resume is None and a.DEV_admission_resume is None and a.periodic_quiescence_resume is None
            and review['native_headroom_resume']==record(a.native_headroom_resume)
            and review['native_update_commit_pressure_policy']==configuration['native_update_commit_pressure_policy']
            and a.start_invocation==review['initial_owner_invocation']==17,
            'Exact source20 TOOL1 saved156200 plus changed conservative admission, never normal0')
    elif a.periodic_quiescence_resume is not None:
        require(a.abnormal_resume is None and a.original_interval_resume is None and a.source_transition_resume is None and a.DEV_frontend_resume is None and a.DEV_admission_resume is None
            and review['periodic_quiescence_resume']==record(a.periodic_quiescence_resume)
            and a.start_invocation==review['initial_owner_invocation']==14,
            'Exact saved140000 helper1 failure is not normal0')
    elif a.DEV_admission_resume is not None:
        require(a.abnormal_resume is None and a.original_interval_resume is None and a.source_transition_resume is None and a.DEV_frontend_resume is None
            and review['DEV_admission_resume']==record(a.DEV_admission_resume)
            and a.start_invocation==review['initial_owner_invocation']==13,
            'Exact saved136000 prospective DEV failure is not normal0')
    elif a.abnormal_resume is not None:
        require(a.original_interval_resume is None and a.source_transition_resume is None and a.DEV_frontend_resume is None
            and review['headroom_resume']==record(a.abnormal_resume)
            and review['resource_scheduling_policy']==configuration['resident_RAM_resource_scheduling_policy']
            and a.start_invocation==review['initial_owner_invocation']==11,
            'Exact reviewed changed-workload abnormal133137 saved133000 entry, never normal0')
    elif a.original_interval_resume is not None:
        require(a.source_transition_resume is None and a.DEV_frontend_resume is None
            and review['original_interval_resume']==record(a.original_interval_resume)
            and review['original_clip_geometry_overlay']==configuration['original_clip_geometry_overlay']
            and a.start_invocation==review['initial_owner_invocation']==16,
            'Exact reviewed source-bound physical geometry correction and actual abnormal142800 predecessor')
    elif a.source_transition_resume is not None:
        require(a.DEV_frontend_resume is None and review['source_transition_resume']==record(a.source_transition_resume)
            and review['source_transition_validator']==record(S/'asr18_resident_source_transition_v2.py'),
            'Exact reviewed actual intentional TOOL1 transition, no normal0 fiction')
    else:
        require(a.DEV_frontend_resume is not None and review['DEV_frontend_resume']==record(a.DEV_frontend_resume),
            'Only the already-supported typed4000 failure lane when no genuine prior process0')
    branch=Path(a.branch).resolve();control=Path(a.output).resolve()
    from run_ASR18_BPE9223_resident_RAM_training_v38 import checked_native_commit_pressure_policy
    checked_native_commit_pressure_policy(configuration)
    if str(Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_original_interval170469_failure_actual_v1'))not in sys.path:sys.path.insert(0,str(Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_original_interval170469_failure_actual_v1')))
    from asr18_eighth_original_interval_checkpoint_resume_v1 import checked_eighth_original_policy,checked_eighth_original_resume
    checked_eighth_original_policy(legacy_configuration)
    if a.throughput_resume is not None:
        checked_current_plan_throughput_checkpoint_resume(record(a.throughput_resume),base=candidate_configuration,folder=branch,pointer_pin=record(a.pointer))
    if a.ninth_original_resume is not None:
        checked_ninth_original_resume(record(a.ninth_original_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))
    if a.gc_refresh_resume is not None:
        checked_one_postDEV_refresh_checkpoint_resume(record(a.gc_refresh_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))
    if a.eighth_original_resume is not None:
        checked_eighth_original_resume(record(a.eighth_original_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))
    require(review['native_update_commit_pressure_policy']==configuration['native_update_commit_pressure_policy'],'Exact ROOT changed native pressure scheduling policy')
    if a.fresh_signature_resume is not None:
        checked_fresh_signature_collect_checkpoint_resume(record(a.fresh_signature_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))
    if a.canonical_archive_resume is not None:
        checked_canonical_archive_checkpoint_resume(record(a.canonical_archive_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))
    if a.lookahead_resume is not None:
        from asr18_approved_lookahead_checkpoint_transition_v3 import checked_lookahead_checkpoint_resume
        checked_lookahead_checkpoint_resume(record(a.lookahead_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))
    if a.cushion225_resume is not None:
        from run_ASR18_BPE9223_resident_RAM_training_v38 import checked_cushion225_resume
        checked_cushion225_resume(record(a.cushion225_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))
    if a.postrestore_selector_resume is not None:
        from run_ASR18_BPE9223_resident_RAM_training_v38 import checked_postrestore_selector_resume
        checked_postrestore_selector_resume(record(a.postrestore_selector_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))
    if a.packet_credit_resume is not None:
        from run_ASR18_BPE9223_resident_RAM_training_v38 import checked_packet_credit_resume
        checked_packet_credit_resume(record(a.packet_credit_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))
    from fastmss_resident_feature_stream_v14 import checked_packet_credit_policy
    checked_packet_credit_policy(configuration)
    if a.native_headroom_resume is not None:
        from run_ASR18_BPE9223_resident_RAM_training_v38 import checked_current_native_headroom_resume
        checked_current_native_headroom_resume(record(a.native_headroom_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))
    if a.abnormal_resume is not None:
        from run_ASR18_BPE9223_resident_RAM_training_v38 import checked_headroom_resume
        checked_headroom_resume(record(a.abnormal_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))
    if a.original_interval_resume is not None:
        from run_ASR18_BPE9223_resident_RAM_training_v38 import checked_original_interval_resume
        checked_original_interval_resume(record(a.original_interval_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))
    if a.source_transition_resume is not None:
        from asr18_resident_source_transition_v2 import checked_source_transition_resume
        checked_source_transition_resume(record(a.source_transition_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))
    if a.DEV_admission_resume is not None:
        from run_ASR18_BPE9223_resident_RAM_training_v38 import checked_DEV_admission_resume
        checked_DEV_admission_resume(record(a.DEV_admission_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))
    if a.periodic_quiescence_resume is not None:
        from run_ASR18_BPE9223_resident_RAM_training_v38 import checked_periodic_quiescence_resume
        checked_periodic_quiescence_resume(record(a.periodic_quiescence_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))
    require(control.is_relative_to(Path(r'E:\gpu_training_scratchpad')) and not control.exists(),'Fresh E loop records');control.mkdir()
    pointer=Path(a.pointer);prior=None if a.prior_process is None else Path(a.prior_process);invocation=a.start_invocation;root_pid=os.getpid()
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';env['PYTHONUTF8']='1'
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    while True:
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
        require(len(rows)==1,'Exactly one admitted training GPU')
        total,free=[int(v.strip())*1024**2 for v in rows[0].split(',')]
        require(total>=17000000000 and free>=14000000000,'Physical GPU capacity plus unchanged12GB allocator/2GBfree reserve')
        command=[str(P/'work/nemo-native-env-v2/Scripts/python.exe'),'-X','utf8','-B',str(driver),
            '--configuration',a.configuration,'--stream-configuration',a.configuration,'--pointer',str(pointer),
            '--latest-failure',a.latest_failure,'--folder',str(branch),'--invocation',str(invocation),
            '--queue-bytes',str(a.queue_bytes),'--maximum-wall-seconds','86400']
        if prior is not None:command+=['--prior-process',str(prior)]
        elif invocation==a.start_invocation:
            if a.postrestore_selector_resume is not None:command+=['--postrestore-selector-resume',a.postrestore_selector_resume]
            if a.throughput_resume is not None:command+=['--throughput-resume',a.throughput_resume]
            elif a.ninth_original_resume is not None:command+=['--ninth-original-resume',a.ninth_original_resume]
            elif a.gc_refresh_resume is not None:command+=['--gc-refresh-resume',a.gc_refresh_resume]
            elif a.eighth_original_resume is not None:command+=['--eighth-original-resume',a.eighth_original_resume]
            elif a.fresh_signature_resume is not None:command+=['--fresh-signature-resume',a.fresh_signature_resume]
            elif a.canonical_archive_resume is not None:command+=['--canonical-archive-resume',a.canonical_archive_resume]
            elif a.lookahead_resume is not None:command+=['--lookahead-resume',a.lookahead_resume]
            elif a.cushion225_resume is not None:command+=['--cushion225-resume',a.cushion225_resume]
            elif a.packet_credit_resume is not None:command+=['--packet-credit-resume',a.packet_credit_resume]
            elif a.native_headroom_resume is not None:command+=['--native-headroom-resume',a.native_headroom_resume]
            elif a.periodic_quiescence_resume is not None:command+=['--periodic-quiescence-resume',a.periodic_quiescence_resume]
            elif a.DEV_admission_resume is not None:command+=['--DEV-admission-resume',a.DEV_admission_resume]
            elif a.abnormal_resume is not None:command+=['--abnormal-resume',a.abnormal_resume]
            elif a.original_interval_resume is not None:command+=['--original-interval-resume',a.original_interval_resume]
            elif a.source_transition_resume is not None:command+=['--source-transition-resume',a.source_transition_resume]
            else:command+=['--DEV-frontend-resume',a.DEV_frontend_resume]
        if invocation==1:command+=['--validation-stop','460']
        log=control/('native_'+str(invocation).zfill(6)+'.log')
        with log.open('xb')as stream:
            child=subprocess.Popen(command,cwd=P,env=env,stdout=stream,stderr=subprocess.STDOUT)
            launcher={'pid':child.pid,'create_time':psutil.Process(child.pid).create_time(),'ppid':root_pid}
            write_json(control/'current_child.json',{'invocation':invocation,'launcher':launcher,'command':command,
                'log':str(log),'branch':str(branch),'source_checkpoint':record(pointer)})
            print(json.dumps({'actual_native_launch':launcher,'invocation':invocation,'log':str(log),'saved_source_updates':read_json(pointer)['committed_updates']}),flush=True)
            code=child.wait()
        inv=branch/'invocations'/('owner_'+str(invocation).zfill(6))
        require(code==0,'Actual native process failed; preserve all state/logs, no unchanged retry')
        report=read_json(inv/'normal_completion.json');who=report['producer']
        try:alive=abs(psutil.Process(who['pid']).create_time()-who['create_time'])<.001
        except psutil.NoSuchProcess:alive=False
        require(not alive and report['passed'] is True and report['all_live_updates_saved'] is True,'Real normal0 and all live updates coherently saved')
        closure={'kind':'ASR18_actual_resident_RAM_GPU_process_wait0_v1','actual_Popen_wait_code':code,'launcher':launcher,
            'producer':who,'producer_observed_dead':True,'worker_report':record(inv/'normal_completion.json'),
            'source':record(__file__),'worker_source':record(driver),'log':record(log)}
        prior=control/('native_'+str(invocation).zfill(6)+'_actual0.json');write_json(prior,closure)
        pointer=Path(report['latest_checkpoint']['pointer']['path'])
        print(json.dumps({'actual_normal0':record(prior),'saved_updates':report['actual_optimizer_updates'],'automatic_continuation':report['completed_boundaries']<3}),flush=True)
        if report['completed_boundaries']==3:break
        invocation+=1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--review',required=True);p.add_argument('--configuration',required=True)
    p.add_argument('--pointer',required=True);p.add_argument('--latest-failure',required=True);p.add_argument('--branch',required=True)
    p.add_argument('--output',required=True);p.add_argument('--queue-bytes',type=int,default=268435456);p.add_argument('--start-invocation',type=int,default=1);p.add_argument('--prior-process');p.add_argument('--throughput-resume');p.add_argument('--ninth-original-resume');p.add_argument('--gc-refresh-resume');p.add_argument('--eighth-original-resume');p.add_argument('--fresh-signature-resume');p.add_argument('--canonical-archive-resume');p.add_argument('--lookahead-resume');p.add_argument('--postrestore-selector-resume');p.add_argument('--cushion225-resume');p.add_argument('--packet-credit-resume');p.add_argument('--native-headroom-resume');p.add_argument('--periodic-quiescence-resume');p.add_argument('--DEV-admission-resume');p.add_argument('--original-interval-resume');p.add_argument('--abnormal-resume');p.add_argument('--source-transition-resume');p.add_argument('--DEV-frontend-resume');main(p.parse_args())
