"""Fixed diagnostic row operations; no source sampler or production frontier."""
import copy,gc,hashlib,json,os,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';E=Path(r'E:\gpu_training_scratchpad')
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_production_state_v1 import require,digest
from resource_watchdog_v2 import sample_memory


def execute_gpu(c,folder,watch,mode,rootfolder,*,H,record,current_lifetime,fresh):
    guard=watch.check
    import numpy as np
    from preload_nemo_linker_v1 import preload
    coherent=preload();import torch;torch.set_num_threads(4)
    from launch_with_memory_guard_TSC_resume_v3 import validate_capacity
    capacity=validate_capacity(torch.cuda.get_device_properties(0).total_memory,8.);torch.cuda.set_per_process_memory_fraction(capacity['per_process_allocator_fraction'],device=0)
    cap=3_000_000_000 if mode=='train' else 3_350_000_000
    def stage(name,**extra):
        watch.phase=name;guard();torch.cuda.synchronize();v=sample_memory(os.getpid(),name);v.update(extra);v['private_high_water']=current_lifetime(cap)
        v['cuda']={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}
        require(v['cuda']['peak_allocated']<=2_500_000_000,'Strict fixed-row component CUDA peak; allocator8GB unchanged')
        with (folder/'stages.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(v)+'\n')
        return v
    def admit(name,increment):
        from asr18_exact_remaining_family_v1 import admit_remaining
        a=admit_remaining(folder,configuration=c['_configuration'],source=c['source_bindings']['harness'],nonce=c['_family_nonce'],guard=guard,identity=H.pid_identity,alive=H.alive,worker_increment=increment)
        write_json(folder/(name+'_admission.json'),a);require(a['passed'],'Fresh complete phase plus exact UNUSED parent/ALL-shim allowance')
    from native_tdt_minimal_import_v1 import prepare as minimal_import
    minimal=minimal_import();stage('runtime_ready',coherent=coherent)
    from stream_frozen_encoder_matrices_v1 import load
    from streamed_generation_device_v1 import apply as logical_device
    from asr18_token_only_carrier_v1 import load as token_carrier
    base=Path((P/'work/base_model_path.txt').read_text().strip());model,loader=load(base);logical=logical_device(model);processor=token_carrier(base)
    teacher={name:copy.deepcopy(getattr(model,name)).eval().requires_grad_(False) for name in ('encoder_projector','decoder','joint')} if mode=='train' else {}
    from bpe_broad_streamed_bootstrap_v2 import bootstrap
    optimizer,provenance,replay=bootstrap(model,processor,torch=torch,stage=stage,admit=admit)
    from tdt_joint_canonical_2d_fp32_v2 import apply as canonical_joint
    from asr18_temporal_tiled_subsampling_v2 import apply as conv_only
    provenance.update(canonical_joint=canonical_joint(model.joint),conv_only_CNN=conv_only(model,torch=torch,guard=guard),
        minimal_import=minimal,logical_device=logical,frozen_loader=loader,teacher_modules_created=mode=='train')
    from bpe_broad_production_v3 import assert_model
    from asr18_live_state_fingerprint_v1 import live_state_fingerprint,require_exact_live_restore
    from asr18_streamed_persistence_fingerprint_v1 import normalize_fingerprint
    from bpe_streaming_training_state_v1 import checkpoint,restore
    from bpe_broad_production_state_v1 import CommitJournal
    from asr18_metadata_checkpoint_binding_v1 import MetadataCheckpointBinding
    from asr18_precomputed_native_v2 import native_update
    from asr18_phase_private_monitor_v1 import PhasePrivateMonitor
    assert_model(model,optimizer,torch);require(sum(p.numel()*p.element_size() for p in optimizer.parameters.values())==81_459_760,'Exact original207 state')
    binding=MetadataCheckpointBinding(c['augmentation_identity'],guard=guard);identity={'kind':'ASR18_fixed20_TR_replay_unqualified_memory_control_v1','config':c['continuation']['source_configuration'],'production_enabled':False,'no_sampler_commit':True}
    require(record(c['continuation']['prepared']['path'])==c['continuation']['prepared'],'Immutable prior actual0 full original frontend preparation')
    prepared=read_json(c['continuation']['prepared']['path']);require(prepared['passed'] and len(prepared['prepared_rows'])==8,'Actual closed eight-source CPU preparation')
    journal=CommitJournal(folder/'unconsumed_source_journal.jsonl');sampler={'kind':'ASR18_unconsumed_original_source_replay_snapshot_v1','source_snapshot':copy.deepcopy(replay),'sampler_draws_consumed':0};prefix=journal.binding()
    if mode=='train':
        ready=stage('actual_original207_teacher_all_setup_ready');require(ready['worker_private_memory']['private_bytes']<=2_100_000_000,'Actual teacher-inclusive ready<=2.1GB')
        from asr18_live_state_fingerprint_v1 import require_exact_live_restore
        origin=c['continuation'];failure=read_json(origin['failed_actual_tool1']['path'])
        require(record(origin['failed_actual_tool1']['path'])==origin['failed_actual_tool1'] and failure['actual_tool_exit_code']==1 and failure['actual_tool_chunk_id']=='7d0aae' and failure['actual_optimizer_updates']==1,'Genuine observed failure afterone, no relabel')
        require(not any(H.alive(v) for v in failure['owned_identities']),'All original native/parent/shim identities actually dead')
        for key in ('checkpoint1','PRE_save1','native_progress','source_configuration','source_guard','source_stages','source_parent_guard','prepared_actual_process0'):
            require(record(origin[key]['path'])==origin[key],'Exact source continuation pin:'+key)
        source_progress=read_json(origin['native_progress']['path']);require(source_progress['actual_updates']==1 and len(source_progress['history'])==1,'Exactly prior positiveupdate1 history')
        initial_pre=normalize_fingerprint(read_json(origin['PRE_save1']['path']))
        before=stage('before_actual_checkpoint1_ancestor_restore');require(before['worker_private_memory']['private_bytes']<=3_000_000_000 and before['worker_private_memory']['private_bytes']+350_000_000<=3_350_000_000,'Entire initial ancestor recovery scope')
        cap=3_350_000_000;watch.cap=cap;admit('initial_exact_checkpoint1_restore',350_000_000)
        restored,ancestor_journal,ancestor_branch=restore(origin['checkpoint1'],torch=torch,model=model,optimizer=optimizer,identity=identity,augmentation=binding,new_journal_path=folder/'restored_source_prefix_from_checkpoint1.jsonl',diagnostic_only=True,guard=guard)
        require_exact_live_restore(initial_pre,live_state_fingerprint(optimizer,torch=torch,guard=guard))
        require(optimizer.updates==1 and digest(restored)==digest(sampler) and ancestor_journal.binding()['sha256']==origin['source_journal_prefix']['sha256'] and ancestor_journal.binding()['offset']==origin['source_journal_prefix']['offset'] and ancestor_branch['original_checkpoint_prefix']==origin['source_journal_prefix'],'Exact update1 sampler/journal ancestor without commit')
        journal=ancestor_journal;prefix=journal.binding();source_snapshot=digest(replay);first_target=c['resource_policy']['training_private_hard_cap_bytes']
        point=stage('after_exact_checkpoint1_full207_threeAdam_all4RNG_ancestor_restore')
        high=max(point['private_high_water']['lifetime_private_peak_bytes'],origin['postwrite_source_OS_lifetime_peak_bytes']);history=copy.deepcopy(source_progress['history'])
        captures=[{'update':1,'checkpoint':origin['checkpoint1'],'PRE_save_fingerprint':origin['PRE_save1'],'identity':identity,'sampler':sampler,'journal_prefix':origin['source_journal_prefix'],'source_actual_tool1':origin['failed_actual_tool1']}]
        write_json(folder/'initial_ancestor_recovery.json',{'passed':True,'actual_update':1,'all207_threeAdam_all4RNG_PRE1_exact':True,'sampler_and_journal_ancestor_exact':True,'branch':ancestor_branch,'source_actual_tool1':origin['failed_actual_tool1']})
        for index,pin in enumerate(prepared['updates']):
            if index==0:continue
            require(record(pin['path'])==pin,'Immutable actual prepared update');update=read_json(pin['path']);items=[prepared['prepared_rows'][i] for i in update['row_indices']]
            target=max(first_target,high+750_000_000+64_000_000);require(target<=c['resource_policy']['training_private_hard_cap_bytes'],'Actual highwater+unchanged750M+64Mwriter must fit declared5.624813440GBforecast, original8GBguard unchanged')
            current=sample_memory(os.getpid(),'before_native_update')['worker_private_memory'];require(current['available'] and current['private_bytes']<target,'Fresh native baseline')
            admit(f'native_update_{index+1}',target-current['private_bytes']);cap=target;watch.cap=cap
            def features(local):
                out={}
                for key,v in items[local]['features'].items():
                    guard();require(record(v['file']['path'])==v['file'],'Exact full original prepared NPY');a=np.load(v['file']['path'],allow_pickle=False)
                    require(hashlib.sha256(memoryview(a).cast('B')).hexdigest()==v['array_sha256'],'Exact full feature/mask array');out[key]=a
                return out
            lease=read_json(P/'work/gpu_training_owner.json');lease['actual_training_optimizer_running']=True;write_json(P/'work/gpu_training_owner.json',lease)
            monitor=PhasePrivateMonitor();monitor.start()
            try:result=native_update(model,teacher,optimizer,[i['row'] for i in items],update['draws'],feature_for_row=features,update_binding=pin,torch=torch,
                population_mean_U=c['population_mean_U'],update=index+1,stage=stage,stop_check=guard,phase='cleaner_data_fine_tuning',feature_update_index=index)
            finally:phase=monitor.finish()
            lease['actual_training_optimizer_running']=False;write_json(P/'work/gpu_training_owner.json',lease)
            point=stage(f'after_actual_update_{index+1}');high=max(high,phase['private_peak_bytes'],point['private_high_water']['lifetime_private_peak_bytes'])
            require(optimizer.updates==index+1 and digest(replay)==source_snapshot and journal.binding()==prefix,'Exact resumed update without source sampler/journal consumption')
            history.append({'update':index+1,'native':result,'phase':phase,'actual_private_high_water':high});write_json(folder/'native_progress.json',{'actual_updates':index+1,'history':history,'checkpoints':captures,'sampler_draws_consumed':0,'production_enabled':False})
            if index+1 in (1,10,20):
                pre=live_state_fingerprint(optimizer,torch=torch,guard=guard);prepath=folder/f'PRE_save_{index+1}.json';write_json(prepath,pre)
                now=sample_memory(os.getpid(),'before_write')['worker_private_memory'];require(now['available'] and now['private_bytes']+64_000_000<=cap,'Writer64MB remains in wholeworker cap; no private-drop assumption')
                admit(f'checkpoint_write_{index+1}',64_000_000);stage(f'before_checkpoint_write_{index+1}')
                saved=checkpoint(folder/f'after_{index+1}.state.json',torch=torch,optimizer=optimizer,state=sampler,provenance=provenance,identity=identity,
                    augmentation=binding,journal=journal,diagnostic_only=True,guard=guard)
                require_exact_live_restore(pre,live_state_fingerprint(optimizer,torch=torch,guard=guard));postwrite=stage(f'after_checkpoint_write_and_persistence_{index+1}');high=max(high,postwrite['private_high_water']['lifetime_private_peak_bytes'])
                captures.append({'update':index+1,'checkpoint':saved,'PRE_save_fingerprint':record(prepath),'identity':identity,'sampler':sampler,'journal_prefix':prefix})
        write_json(folder/'training_result.json',{'passed':True,'source':c['source_bindings']['harness'],'producer':H.pid_identity(),'captures':captures,
            'actual_optimizer_updates':20,'actual_new_optimizer_updates':19,'initial_checkpoint1_exact_recovery':record(folder/'initial_ancestor_recovery.json'),'sampler_draws_consumed':0,'history':history,'full_recipe_or_quality_proof':False,'production_enabled':False})
    else:
        actual=read_json(rootfolder/'train/actual_process0.json');require(actual['actual_Popen_wait_code']==0 and actual['all_captured_identities_dead']
            and actual['configuration']==c['_configuration'] and not any(H.alive(v) for v in actual['captured_identities']),'Genuine prior native0/dead before fresh recovery')
        require(record(actual['report']['path'])==actual['report'] and record(actual['guard']['path'])==actual['guard'],'Exact native closed result')
        saved=read_json(actual['report']['path']);require(saved['actual_optimizer_updates']==20 and [v['update'] for v in saved['captures']]==[1,10,20],'Every genuine checkpoint, no omitted capture')
        recovered=[]
        for part in saved['captures']:
            require(record(part['PRE_save_fingerprint']['path'])==part['PRE_save_fingerprint'],'Actual independent PREsave pin')
            pre=normalize_fingerprint(read_json(part['PRE_save_fingerprint']['path']));before=stage(f'before_restore_{part["update"]}')
            require(before['worker_private_memory']['private_bytes']<=3_000_000_000 and before['worker_private_memory']['private_bytes']+350_000_000<=3_350_000_000,'Fresh restore startup/entireworker envelope')
            admit(f'exact_recovery_{part["update"]}',350_000_000)
            state,branch_journal,branch=restore(part['checkpoint'],torch=torch,model=model,optimizer=optimizer,identity=identity,augmentation=binding,
                new_journal_path=folder/f'exact_prefix_{part["update"]}.jsonl',diagnostic_only=True,guard=guard)
            require_exact_live_restore(pre,live_state_fingerprint(optimizer,torch=torch,guard=guard));require(optimizer.updates==part['update'] and digest(state)==digest(part['sampler']),'Exact native count and unconsumed snapshot')
            stage(f'after_exact_restore_{part["update"]}');recovered.append({'update':part['update'],'all207_threeAdam_all4RNG_PREsave_exact':True,'branch':branch})
        write_json(folder/'recovery_result.json',{'passed':True,'source':c['source_bindings']['harness'],'producer':H.pid_identity(),'recovered':recovered,'actual_native_updates':0,'production_enabled':False})
