"""ONE complete159.24s T1991/U574 TR diagnostic, original207/threeAdam, no sampler commit.

CPU original-frontend preparation, training and exact recovery are distinct
owned children. No launch without exact ROOT source/config/resource review.
"""
import argparse,copy,ctypes,gc,hashlib,importlib.util,json,os,secrets,shutil,subprocess,sys,time
from ctypes import wintypes
from functools import partial
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
E=Path(r'E:\gpu_training_scratchpad');A=E/'analysis/ASR18_memory_diagnosis_v1';LEASE=P/'work/gpu_training_owner.json';OWNER='/root/memory_diagnosis'
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_production_state_v1 import require,digest
from resource_watchdog_v2 import sample_memory,PROCESS_MEMORY_COUNTERS_EX
spec=importlib.util.spec_from_file_location('component_guard',P/'work/research/run_ASR18_frozen_CNN_lifetime_pair_v8.py')
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
record=H.record

def checked(config,review):
    H.storage();c=read_json(config);r=read_json(review)
    require(c['kind']=='ASR18_single_full_hard_TR207_memory_diagnostic_v1' and c['production_enabled'] is False
        and c['native_updates']==1 and c['sampler_draws']==0 and c['teacher_or_replay_forward'] is False,'Exact one-row unqualified diagnostic')
    require(r['accepted'] is True and r['reviewer']=='/root' and r['configuration']==record(config)
        and r['source_bindings']==c['source_bindings'] and r['resource_policy']==c['resource_policy'],'Exact ROOT component approval')
    for pin in c['source_bindings'].values():require(record(pin['path'])==pin,'Frozen source changed')
    require(c['source_bindings']['harness']==record(__file__),'Exact driver source')
    for pin in c['evidence'].values():require(record(pin['path'])==pin,'Exact small prior evidence')
    require(record(c['CPU_python']['path'])==c['CPU_python'] and record(c['GPU_python']['path'])==c['GPU_python'],'Exact existing interpreters')
    return c

def current_lifetime(cap):
    k=ctypes.WinDLL('kernel32',use_last_error=True);p=ctypes.WinDLL('psapi',use_last_error=True);k.GetCurrentProcess.restype=wintypes.HANDLE
    p.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),wintypes.DWORD]
    p.GetProcessMemoryInfo.restype=wintypes.BOOL;v=PROCESS_MEMORY_COUNTERS_EX();v.cb=ctypes.sizeof(v)
    require(p.GetProcessMemoryInfo(k.GetCurrentProcess(),ctypes.byref(v),v.cb),'Available owned lifetime telemetry')
    require(v.PrivateUsage<=cap and v.PeakPagefileUsage<=cap,'Actual current AND lifetime private cap, including short spikes')
    return {'current_private_bytes':int(v.PrivateUsage),'lifetime_private_peak_bytes':int(v.PeakPagefileUsage)}

def fresh(increment,guard):
    samples=[]
    for _ in range(3):
        guard();v=sample_memory(os.getpid(),'single_TR207_prospective');s=v['system_memory'];w=v['worker_private_memory']
        fits=v['paired_sampling_within_one_second'] and s['available'] and w['available']
        fits=fits and s['commit_used_bytes']+increment<.90*s['commit_limit_bytes'] and s['commit_available_bytes']-increment>2_000_000_000
        samples.append({'sample':v,'incremental_forecast_bytes':increment,'passed':fits});time.sleep(.05)
    return {'passed':all(v['passed'] for v in samples),'samples':samples}

def prepare(c,folder,guard):
    require(os.environ['CUDA_VISIBLE_DEVICES']=='-1','CPU original frontend, retained independently decoded PCM only')
    admission=fresh(1_500_000_000,guard);write_json(folder/'fresh1500MB.json',admission);require(admission['passed'],'Fresh bounded long-row CPU frontend allowance')
    site=P/'work/training-env/Lib/site-packages';sys.path.insert(0,str(site));import numpy as np;sys.path.remove(str(site))
    import torch
    require(torch.__version__=='2.7.1+cpu' and torch.version.cuda is None and Path(torch.__file__).resolve().is_relative_to(E),'Existing E CPU Torch only')
    sys.path.append(str(site));torch.set_num_threads(4)
    from transformers import AutoFeatureExtractor
    import inspect
    proof=read_json(c['evidence']['hard_PCM_reference']['path']);item=proof['rows'][0]
    require(proof['passed'] and proof['heldout_PCM_check_performed'] and item['row']==c['selected_row'],'Independent original heldout/complete-row proof')
    bankpin=c['evidence']['full_heldout_PCM_bank'];require(record(bankpin['path'])==bankpin,'Exact current complete heldout PCM bank')
    bank=read_json(bankpin['path']);raw=read_json(c['evidence']['current_final_DATA_raw']['path'])
    require(record(c['evidence']['current_final_DATA_raw']['path'])==c['evidence']['current_final_DATA_raw'] and bankpin==raw['bindings']['heldout_PCM'],'Bank bound by current final DATA aggregate')
    require(bank['passed'] is True and bank['complete'] is True and bank['reference_clips']==len(bank['PCM_i16_sha256'])==12273 and bank['metadata_binding']==raw['bindings']['heldout_metadata'],'Complete12273 original heldout references and exact metadata')
    require(item['reader_evidence']['PCM_i16_sha256']=='ebeb085f96a560b05d92b8eded2e356e9b495ffb32b8594adc25c973ca40fecc' and item['reader_evidence']['PCM_i16_sha256'] not in bank['PCM_i16_sha256'],'Existing immutable whole-row PCM_i16 absent from current complete bank, no redecode/rescan')
    require(record(item['PCM_cache']['path'])==item['PCM_cache'],'Exact retained whole PCM NPY before load')
    wave=np.load(item['PCM_cache']['path'],allow_pickle=False)
    require(wave.dtype==np.float32 and wave.shape==(2_547_840,) and hashlib.sha256(memoryview(wave).cast('B')).hexdigest()==c['PCM_array_sha256'],'Exact whole159.24s original PCM, no crop/augmentation')
    extractor=AutoFeatureExtractor.from_pretrained(c['frontend']['base_path'],local_files_only=True)
    require(record(inspect.getfile(type(extractor)))==c['frontend']['source'],'Exact original frontend')
    inputs=dict(extractor([wave],sampling_rate=16000,return_tensors='pt',padding='longest',return_attention_mask=True,subsampling_factor=8))
    mask=inputs['attention_mask'].bool();features=inputs['input_features'].masked_fill(~mask.unsqueeze(-1),0).detach().cpu().contiguous().numpy();mask=mask.cpu().contiguous().numpy()
    require(features.shape==(1,15925,128) and mask.shape==(1,15925) and int(mask.sum())==15924 and (int(mask.sum())+7)//8==1991,'Complete physical15925/valid15924/encoder1991')
    files={}
    for name,array in {'input_features':features,'attention_mask':mask}.items():
        path=folder/(name+'.npy');np.save(path,array,allow_pickle=False)
        files[name]={'file':record(path),'array_sha256':hashlib.sha256(memoryview(array).cast('B')).hexdigest(),'shape':list(array.shape),'dtype':str(array.dtype)}
    write_json(folder/'prepared.json',{'kind':'ASR18_single_full_hard_TR_original_feature_preparation_v1','passed':True,'producer':H.pid_identity(),'source':record(__file__),
        'row':item['row'],'full_target_ids':item['row']['tr_metadata_labels'],'features':files,'independent_PCM_equal':True,'original_frontend_source_exact':True,
        'saved_frontend_array_equivalence_claimed':False,'sampler_or_optimizer_advanced':False,'waveform_cropped_or_augmented':False,'production_enabled':False})

def gpu(c,folder,watch,mode,rootfolder):
    guard=watch.check
    import numpy as np
    from preload_nemo_linker_v1 import preload
    coherent=preload();import torch;torch.set_num_threads(4)
    from launch_with_memory_guard_TSC_resume_v3 import validate_capacity
    capacity=validate_capacity(torch.cuda.get_device_properties(0).total_memory,8.)
    torch.cuda.set_per_process_memory_fraction(capacity['per_process_allocator_fraction'],device=0)
    cap=3_000_000_000 if mode=='train' else 3_350_000_000
    def stage(name,**extra):
        watch.phase=name;guard();torch.cuda.synchronize();v=sample_memory(os.getpid(),name);v.update(extra)
        require(torch.cuda.max_memory_allocated()<=7_500_000_000,'Full hard-shape CUDA ceiling below unchanged8GB allocator cap')
        v['private_high_water']=current_lifetime(cap);v['cuda']={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
            'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}
        with (folder/'stages.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(v)+'\n')
        return v
    def admit(name,increment):
        from asr18_exact_remaining_family_v1 import admit_remaining
        v=admit_remaining(folder,configuration=c['_configuration'],source=c['source_bindings']['harness'],nonce=c['_family_nonce'],guard=guard,identity=H.pid_identity,alive=H.alive,worker_increment=increment)
        write_json(folder/(name+'_admission.json'),v);require(v['passed'],'Fresh exact full hard worker plus unused remaining family: '+name)
    from native_tdt_minimal_import_v1 import prepare as minimal_import
    minimal=minimal_import();stage('runtime_ready',coherent=coherent,capacity=capacity)
    from stream_frozen_encoder_matrices_v1 import load
    from streamed_generation_device_v1 import apply as device_policy
    from asr18_token_only_carrier_v1 import load as token_carrier
    base=Path((P/'work/base_model_path.txt').read_text().strip());model,loader=load(base);logical=device_policy(model);processor=token_carrier(base)
    from bpe_broad_streamed_bootstrap_v2 import bootstrap
    optimizer,provenance,replay=bootstrap(model,processor,torch=torch,stage=stage,admit=admit)
    from tdt_joint_canonical_2d_fp32_v2 import apply as canonical_joint
    from asr18_temporal_tiled_subsampling_v2 import apply as conv_only
    provenance.update(canonical_joint=canonical_joint(model.joint),conv_only_CNN=conv_only(model,torch=torch,guard=guard),
        minimal_import=minimal,logical_device=logical,frozen_loader=loader,teacher_modules_created=False)
    from bpe_broad_production_v3 import assert_model
    assert_model(model,optimizer,torch);require(sum(p.numel()*p.element_size() for p in optimizer.parameters.values())==81_459_760,'Exact actual207 complete FP32 weight/gradient descriptor budget')
    ready=stage('original207_threeAdam_bootstrap_complete')
    require(ready['worker_private_memory']['private_bytes']<=2_100_000_000,'Actual full207 ready prerequisite<=2.1GB; no assumed private credit')
    from asr18_live_state_fingerprint_v1 import live_state_fingerprint,require_exact_live_restore
    from asr18_streamed_persistence_fingerprint_v1 import normalize_fingerprint
    from bpe_streaming_training_state_v1 import checkpoint,restore
    from bpe_broad_production_state_v1 import CommitJournal
    from asr18_metadata_checkpoint_binding_v1 import MetadataCheckpointBinding
    binding=MetadataCheckpointBinding(c['augmentation_identity'],guard=guard)
    identity={'kind':'ASR18_single_full_hard_TR_unqualified_native207_memory_control_v1','config':c['_configuration'],
        'row':c['selected_row'],'production_enabled':False,'no_sampler_commit':True}
    if mode=='train':
        prepared=read_json(rootfolder/'prepare/prepared.json');require(prepared['row']==c['selected_row'] and prepared['original_frontend_source_exact'],'Exact owned original CPU preparation')
        arrays={}
        for name,v in prepared['features'].items():
            require(record(v['file']['path'])==v['file'],'Exact feature file')
            array=np.load(v['file']['path'],allow_pickle=False);require(hashlib.sha256(memoryview(array).cast('B')).hexdigest()==v['array_sha256'],'Exact full feature array')
            arrays[name]=array
        from asr18_precomputed_collator_v1 import collate_precomputed_row
        from asr18_narrow_objective_v1 import pt_focused_batch_objective
        from tdt_tiled_full_native_v7 import bounded_tdt_loss
        from bpe_broad_production_state_v1 import weight_objective
        ready=stage('actual_original207_threeAdam_all_setup_ready')
        require(ready['worker_private_memory']['private_bytes']<=2_100_000_000,'Actual all-setup ready<=2.1GB before first forward')
        proposed=c['resource_policy']['training_private_hard_cap_bytes']
        extra=proposed-ready['worker_private_memory']['private_bytes']
        require(proposed<=c['resource_policy']['training_private_hard_cap_bytes'],'Full hard phase within exact6.540646272GB reviewed worker forecast')
        admit('native_plus_write_after_actual_ready',extra)
        cap=proposed;watch.cap=cap;replay_before=digest(replay)
        lease=read_json(LEASE);require(lease['actual_owned_worker_pid']==os.getpid(),'Exact native owner before update')
        lease['actual_training_optimizer_running']=True;write_json(LEASE,lease)
        model.train();model.encoder.eval();optimizer.zero_grad();before=stage('before_first_native_forward')
        batch=collate_precomputed_row(prepared['row'],arrays,prepared['full_target_ids'],device='cuda');del arrays
        result=pt_focused_batch_objective(model,{},batch,old_size=8193,replay_kl=40.,duration_kl=2.,feature_distillation=20.,
            portuguese_token_kl_multiplier=1.,chunk_size=16,tdt_loss_fn=partial(bounded_tdt_loss,guard=guard),tdt_host_lengths=True)
        result=weight_objective(result,[prepared['row']],c['population_mean_U'],torch_module=torch)
        require(bool(torch.isfinite(result.total)),'Finite original complete-target objective');loss=float(result.total.detach());stage('after_first_native_forward')
        result.total.backward();stage('after_first_native_backward')
        require(len(optimizer.parameters)==207 and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in optimizer.parameters.values()),'All207 actual finite gradients')
        del result,batch;norm=float(optimizer.step())
        actual_clocks=optimizer.state_dict()['clocks']
        write_json(folder/'positive_native_frontier_immediately_after_step.json',{'kind':'ASR18_single_full_hard_TR207_actual_positive_frontier_v1','source':record(__file__),'configuration':c['_configuration'],'producer':H.pid_identity(),
            'actual_optimizer_updates':optimizer.updates,'actual_clocks':actual_clocks,'expected_actual_clocks':[64251,2001,62251],'actual_global_grad_norm':norm,
            'source_sampler_snapshot_unchanged':digest(replay)==replay_before,'sampler_draws_consumed':0,'actual_step_completed':True,
            'end_cache_postwrite_recovery_pending':True,'overall_passed':False,'production_enabled':False})
        require(optimizer.updates==1 and actual_clocks==[64251,2001,62251] and digest(replay)==replay_before,'Exact one positive mature-state step without source sampler consumption')
        optimizer.zero_grad();gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.synchronize()
        require(torch.cuda.memory_reserved()<=3_000_000_000,'Unchanged native3GB end-cache bound after real graph release/GC/sync/emptycache')
        stage('after_one_actual_threeAdam_update')
        lease=read_json(LEASE);lease['actual_training_optimizer_running']=False;write_json(LEASE,lease)
        require(digest(replay)==replay_before,'Original carried sampler/RNG snapshot never consumed')
        require(optimizer.updates==1 and optimizer.state_dict()['clocks']==[64251,2001,62251],'One original mature all207/threeAdam update')
        journal=CommitJournal(folder/'unconsumed_source_journal.jsonl');sampler={'kind':'ASR18_unconsumed_original_source_replay_snapshot_v1','source_snapshot':copy.deepcopy(replay),'sampler_draws_consumed':0}
        prefix=journal.binding();pre=live_state_fingerprint(optimizer,torch=torch,guard=guard);write_json(folder/'PRE_save_live_fingerprint.json',pre)
        now=sample_memory(os.getpid(),'before_checkpoint_write')['worker_private_memory'];require(now['available'] and now['private_bytes']+64_000_000<=cap,'64MB writer included in hard wholeworker cap')
        admit('checkpoint_write',64_000_000)
        stage('before_checkpoint_write')
        saved=checkpoint(folder/'after_one.state.json',torch=torch,optimizer=optimizer,state=sampler,provenance=provenance,identity=identity,
            augmentation=binding,journal=journal,diagnostic_only=True,guard=guard)
        require_exact_live_restore(pre,live_state_fingerprint(optimizer,torch=torch,guard=guard));require(journal.binding()==prefix,'No source sampler/journal commit')
        stage('after_checkpoint_write_and_persistence')
        write_json(folder/'training_result.json',{'passed':True,'source':record(__file__),'producer':H.pid_identity(),'checkpoint':saved,'identity':identity,
            'PRE_save_fingerprint':record(folder/'PRE_save_live_fingerprint.json'),'sampler':sampler,'journal_prefix':prefix,'loss':loss,'global_grad_norm':norm,
            'actual_optimizer_updates':1,'sampler_draws_consumed':0,'full_recipe_or_quality_proof':False,'production_enabled':False})
    else:
        actual=read_json(rootfolder/'train/actual_process0.json');require(actual['actual_Popen_wait_code']==0 and actual['all_captured_identities_dead'] is True
            and actual['source']==record(__file__) and actual['configuration']==c['_configuration']
            and not any(H.alive(v) for v in actual['captured_identities']),'Genuine previous native process0 before fresh recovery')
        require(record(actual['report']['path'])==actual['report'] and record(actual['guard']['path'])==actual['guard'],'Exact native result/guard pins')
        original=read_json(rootfolder/'train/training_result.json');pre=normalize_fingerprint(read_json(original['PRE_save_fingerprint']['path']))
        ready=stage('actual_original207_threeAdam_recovery_setup_ready')
        require(ready['worker_private_memory']['private_bytes']<=2_100_000_000,'Actual recovery all-setup ready<=2.1GB')
        admit('exact_checkpoint_recovery',350_000_000)
        state,journal,branch=restore(original['checkpoint'],torch=torch,model=model,optimizer=optimizer,identity=identity,augmentation=binding,
            new_journal_path=folder/'exact_restored_unconsumed_source_prefix.jsonl',diagnostic_only=True,guard=guard)
        require_exact_live_restore(pre,live_state_fingerprint(optimizer,torch=torch,guard=guard))
        require(digest(state)==digest(original['sampler']) and optimizer.updates==1 and digest(state['source_snapshot'])==digest(replay),'Exact source snapshots remain unconsumed')
        stage('after_all207_threeAdam_all4RNG_exact_recovery')
        write_json(folder/'recovery_result.json',{'passed':True,'source':record(__file__),'producer':H.pid_identity(),
            'all207_threeAdam_all4RNG_match_actual_PRE_save':True,'sampler_snapshot_exact_unconsumed':True,'branch':branch,'actual_native_updates':0,'production_enabled':False})

def leaf(config,review,folder,mode,rootfolder,family_nonce):
    c=checked(config,review);c['_configuration']=record(config);c['_family_nonce']=family_nonce;producer=H.pid_identity();write_json(folder/'boot.json',{'producer':producer,'source':record(__file__),'family_nonce':family_nonce})
    cap=1_800_000_000 if mode=='prepare' else 3_000_000_000 if mode=='train' else 3_350_000_000
    watch=H.Watch(folder,cap,worker=True);code=1
    try:
        if mode!='prepare':
            deadline=time.monotonic()+15
            while time.monotonic()<deadline:
                lease=read_json(LEASE)
                if lease['owner']==OWNER and lease['actual_GPU_running'] is True and lease['actual_owned_worker_pid']==os.getpid():break
                watch.check();time.sleep(.05)
            else:raise ValueError('Actual exclusive GPU handshake')
        if mode=='prepare':prepare(c,folder,watch.check)
        else:gpu(c,folder,watch,mode,rootfolder)
        code=0
    finally:watch.close(code)

def parent(config,review,folder):
    c=checked(config,review);require(folder.resolve().is_relative_to(E.resolve()) and not folder.exists(),'Fresh E-only run')
    folder.mkdir();watch=H.Watch(folder,128_000_000);prior=read_json(LEASE);require(prior['owner']==OWNER and prior['actual_GPU_running'] is False,'Idle exclusive GPU owner')
    code=1
    try:
        for mode,forecast in [('prepare',1_992_000_000),('train',6_732_646_272),('recover',3_542_000_000)]:
            admission=fresh(forecast,watch.check);write_json(folder/(mode+'_prelaunch.json'),admission);require(admission['passed'],'Fresh phase-owned family admission, no unchanged retry')
            path=folder/mode;path.mkdir();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='-1' if mode=='prepare' else '0';env['TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT']='16'
            require(not env.get('DISABLE_ADDMM_CUDA_LT') and not env.get('TORCH_CUDNN_V8_API_DISABLED'),'No failed backend switches')
            family_nonce=secrets.token_hex(16);observer=H.pid_identity()
            python=c['CPU_python']['path'] if mode=='prepare' else c['GPU_python']['path'];known={};native=None;child=None
            try:
                with (path/'worker.log').open('xb') as log:
                    child=subprocess.Popen([python,'-X','utf8','-B',str(Path(__file__).resolve()),'--config',config,'--ROOT-review',review,'--folder',str(path),
                        '--leaf',mode,'--root-folder',str(folder),'--family-nonce',family_nonce],cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    launcher=H.pid_identity(child.pid);began=time.monotonic()
                    while child.poll() is None:
                        watch.check();require(time.monotonic()-began<c['maximum_stage_seconds'][mode],'Finite phase wall budget')
                        try:
                            import psutil
                            for process in [psutil.Process(child.pid),*psutil.Process(child.pid).children(recursive=True)]:
                                try:v=H.pid_identity(process.pid);known[(v['pid'],v['create_time'])]=v
                                except psutil.NoSuchProcess:pass
                        except psutil.NoSuchProcess:pass
                        if native is None and (path/'boot.json').exists():
                            boot=read_json(path/'boot.json');native=boot['producer'];require(boot['source']==record(__file__) and boot['family_nonce']==family_nonce and H.alive(native) and (native['pid']==child.pid or native['ppid']==child.pid) and (native['pid'],native['create_time']) in known,'Exact captured live owned child')
                            if mode!='prepare':
                                lease=copy.deepcopy(prior);lease.update(actual_GPU_running=True,actual_owned_worker_pid=native['pid'],actual_training_optimizer_running=False);write_json(LEASE,lease)
                        total=sample_memory(os.getpid(),'single_TR_owned_family')['worker_private_memory']['private_bytes'];shim=0
                        for v in known.values():
                            if not H.alive(v):continue
                            w=sample_memory(v['pid'],'single_TR_known_child')['worker_private_memory']
                            if not w['available'] and not H.alive(v):continue
                            require(w['available'],'Available exact owned family');total+=w['private_bytes']
                            if native is None or v['pid']!=native['pid']:shim+=w['private_bytes']
                        if native is not None and H.alive(native):
                            write_json(path/'owned_family_context.json',{'kind':'ASR18_actual_owned_GPU_family_context_v1','source':record(__file__),'configuration':record(config),'nonce':family_nonce,
                                'observer':observer,'launcher':launcher,'native':native,'all_live_shims':[v for v in known.values() if H.alive(v) and v['pid']!=native['pid']],
                                'observed_monotonic':time.monotonic()})
                        require(total<=(1_992_000_000 if mode=='prepare' else 6_732_646_272 if mode=='train' else 3_542_000_000) and shim<=64_000_000,'Exact family and launcher ceilings')
                        time.sleep(.05)
                    actual=child.wait();deadline=time.monotonic()+2
                    while any(H.alive(v) for v in known.values()) and time.monotonic()<deadline:watch.check();time.sleep(.02)
                    require(actual==0 and native is not None and not any(H.alive(v) for v in known.values()),'Real handle0 and all exact identities dead')
                    g=read_json(path/'guard_closure.json');require(g['actual_exit_code']==0 and g['pid']==native['pid'] and g['monitor_clean'] and g['monitor_joined'] and g['samples']>=3,'Exact clean native guard0')
                    resultpath=path/('prepared.json' if mode=='prepare' else 'training_result.json' if mode=='train' else 'recovery_result.json')
                    result=read_json(resultpath);require(result['passed'] is True and result['producer']==native and result['source']==record(__file__),'Exact closed source/result/producer')
                    write_json(path/'actual_process0.json',{'kind':'ASR18_single_TR207_owned_phase_process0_v1','actual_Popen_wait_code':0,'all_captured_identities_dead':True,
                        'captured_identities':list(known.values()),'native':native,'source':record(__file__),'configuration':record(config),'report':record(resultpath),'guard':record(path/'guard_closure.json')})
            finally:
                import psutil
                for v in reversed(list(known.values())):
                    if H.alive(v):psutil.Process(v['pid']).terminate()
                deadline=time.monotonic()+10
                while any(H.alive(v) for v in known.values()) and time.monotonic()<deadline:watch.check();time.sleep(.05)
                require(not any(H.alive(v) for v in known.values()),'Every owned native dead before lease release')
                if child is not None:
                    if child.poll() is None:child.terminate()
                    child.wait(timeout=5)
                write_json(LEASE,prior)
        write_json(folder/'overall_result.json',{'passed':True,'actual_updates':1,'actual_training_process0':record(folder/'train/actual_process0.json'),
            'actual_fresh_recovery_process0':record(folder/'recover/actual_process0.json'),'sampler_draws_consumed':0,'qualified':False,'production_enabled':False});code=0
    finally:watch.close(code)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--ROOT-review',required=True);p.add_argument('--folder',required=True)
    p.add_argument('--leaf',choices=('prepare','train','recover'));p.add_argument('--root-folder');p.add_argument('--family-nonce');a=p.parse_args()
    if a.leaf:leaf(a.config,a.ROOT_review,Path(a.folder),a.leaf,Path(a.root_folder),a.family_nonce)
    else:parent(a.config,a.ROOT_review,Path(a.folder))
