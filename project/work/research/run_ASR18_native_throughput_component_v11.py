"""NEW coarse residual attribution only; verified blockedFP32 first4 math unchanged.

Baseline and separately admitted candidate borrow the exact same newly owned
immutable feature files. Every arm is original-source bootstrap, own scientific
commits/checkpoint, actual native exit, then a separate exact fresh restoration.
"""
import argparse,copy,ctypes,gc,importlib.util,json,os,shutil,subprocess,sys,time
from contextlib import contextmanager
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
E=Path(r'E:\gpu_training_scratchpad');A=E/'analysis/ASR18_memory_diagnosis_v1'
LEASE=P/'work/gpu_training_owner.json';OWNER='/root/memory_diagnosis'
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from resource_watchdog_v2 import sample_memory,ResourceGuard
from asr18_memory_diagnosis_v2 import storage,now
from run_ASR18_full_run_owner_v15 import private_lifetime_limit,fresh_admit
spec=importlib.util.spec_from_file_location('throughput_component_guard',P/'work/research/run_ASR18_frozen_CNN_lifetime_pair_v8.py')
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)


class NoOpProfiler:
    """Matched final speed observation: no phase hooks/events/gradient readbacks."""
    def __init__(self,model,optimizer,*,torch,folder,guard):
        self.folder=Path(folder);self.guard=guard;self.updates=[]
    def install(self):return {'enabled':False,'phase_hooks_or_CUDA_events_or_gradient_readbacks':False}
    def observe_row(self,*args):return None
    def checked_guard(self):return self.guard()
    def feature(self,function,index):return function(index)
    def stage(self,original,name,**extra):return original(name,**extra)
    def run_update(self,operation,*,update):
        require(1<=update<=4 and len(self.updates)<4,'Same exact bounded four-update speed check')
        began=time.monotonic();result=operation()
        self.updates.append({'update':update,'native_wall_seconds':time.monotonic()-began});return result
    def close(self):
        write_json(self.folder/'profile_summary.json',{'kind':'ASR18_matched_profiler_off_native_wall_observation_v1',
            'source':record(__file__),'update_wall_seconds':self.updates,'profile_enabled':False,'production_enabled':False})
        return record(self.folder/'profile_summary.json')


class GroupOracleProfiler(NoOpProfiler):
    """Full group-reference row/gradient hashes, without per-module phase hooks."""
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.torch=kw['torch'];self.optimizer=a[1];self.gradients=[];self.rows=[];self.active=False;self.update=0
    def install(self):
        self.original_clip=self.torch.nn.utils.clip_grad_norm_
        from asr18_live_state_fingerprint_v1 import tensor_digest
        def clipping(*a,**kw):
            if not self.active:return self.original_clip(*a,**kw)
            before={name:tensor_digest(p.grad,torch=self.torch,guard=self.guard) for name,p in sorted(self.optimizer.parameters.items())}
            norm=self.original_clip(*a,**kw)
            after={name:tensor_digest(p.grad,torch=self.torch,guard=self.guard) for name,p in sorted(self.optimizer.parameters.items())}
            require(len(before)==len(after)==207,'Every accumulated and clipped actual grouped gradient')
            self.gradients.append({'update':self.update,'all207_before_original_clip':before,'all207_after_original_clip':after,
                'norm':tensor_digest(norm,torch=self.torch,guard=self.guard)})
            return norm
        self.clipping=clipping;self.torch.nn.utils.clip_grad_norm_=clipping
        return {'grouped_exact_oracle_observers':True,'per_module_phase_hooks_or_CUDA_events':False,'reference_oracle_hash_time_included_in_update_wall':True}
    def observe_row(self,index,features,result,mask):
        from asr18_live_state_fingerprint_v1 import tensor_digest
        require(self.active and index==len(self.rows),'Every original row exactly once in original order')
        self.rows.append({'label':'row_'+str(index+1),'encoder_valid_features':tensor_digest(features,torch=self.torch,guard=self.guard),
            'encoder_full_group_mask':tensor_digest(mask,torch=self.torch,guard=self.guard),
            'all_perclip_loss_terms':{k:tensor_digest(v,torch=self.torch,guard=self.guard) for k,v in sorted(result.per_clip.items())},
            'total':tensor_digest(result.total,torch=self.torch,guard=self.guard),'target_tokens':result.target_tokens})
    def run_update(self,operation,*,update):
        self.active=True;self.update=update;self.rows=[]
        try:
            result=super().run_update(operation,update=update)
            require(len(self.rows)==16,'All original14TR+2replay row outputs observed')
            write_json(self.folder/('profile_update_'+str(update)+'.json'),{'full_row_output_fingerprints':self.rows})
            return result
        finally:self.active=False
    def close(self):
        current=self.torch.nn.utils.clip_grad_norm_;self.torch.nn.utils.clip_grad_norm_=self.original_clip
        require(current is self.clipping,'Restore exact original clip after bounded grouped observation')
        write_json(self.folder/'profile_summary.json',{'kind':'ASR18_grouped_exact_full_row_gradient_oracle_observation_v1','source':record(__file__),
            'update_wall_seconds':self.updates,'all207_gradient_fingerprints':self.gradients,'profile_enabled':False,
            'oracle_hashes_observation_overhead_included':True,'production_enabled':False})
        return record(self.folder/'profile_summary.json')


@contextmanager
def cache_backward_calls(model,*,torch,enabled):
    """Candidate only: scope the unchanged Tensor.backward to its actual root GraphTask."""
    if not enabled:
        yield;return
    from asr18_frozen_encoder_cuda_cache_v3 import backward_scope
    original=torch.Tensor.backward
    def scoped(tensor,*args,**kwargs):
        with backward_scope(model,tensor,torch=torch):
            return original(tensor,*args,**kwargs)
    torch.Tensor.backward=scoped;healthy=False
    try:
        yield;healthy=True
    finally:
        current=torch.Tensor.backward;torch.Tensor.backward=original
        if healthy:require(current is scoped,'Profiler returned the exact candidate backward wrapper before original restoration')


@contextmanager
def resident_gc_scope(folder,*,enabled,optimizer,torch,guard):
    if not enabled:
        yield;return
    from asr18_live_state_fingerprint_v1 import live_state_fingerprint,require_exact_live_restore
    require(gc.get_freeze_count()==0,'Fresh worker with no previously frozen generation')
    before=live_state_fingerprint(optimizer,torch=torch,guard=guard)
    pointers=[(id(p),p.data_ptr(),p._version,p.grad) for p in optimizer.parameters.values()]
    began=time.monotonic();collected=gc.collect();gc.freeze();frozen=True
    try:
        count=gc.get_freeze_count();require(count>0,'Actual quiescent resident graph frozen once')
        require_exact_live_restore(before,live_state_fingerprint(optimizer,torch=torch,guard=guard))
        require(pointers==[(id(p),p.data_ptr(),p._version,p.grad) for p in optimizer.parameters.values()],
            'Freeze retains all207 original Parameter identities/pointers/versions/no gradients')
        write_json(folder/'actual_resident_GC_freeze.json',{'source':record(__file__),'frozen_objects':count,
            'initial_full_collect_objects':collected,'setup_wall_seconds':time.monotonic()-began,
            'all207_threeAdam_all4RNG_before_after_freeze_exact':True,
            'per_row_full_gc_collect_synchronize_empty_cache_and_invariants_unchanged':True,'host_credit_bytes':0})
        yield
    finally:
        if frozen:
            before_unfreeze=None
            try:
                if not optimizer.poisoned:
                    before_unfreeze=live_state_fingerprint(optimizer,torch=torch,guard=guard)
                    pointers=[(id(p),p.data_ptr(),p._version,p.grad) for p in optimizer.parameters.values()]
            finally:
                # Also unfreeze after a failed/poisoned native operation. Such
                # a path never claims scientific-state or overall success.
                began=time.monotonic();gc.unfreeze();objects=gc.collect()
            require(gc.get_freeze_count()==0,'Actual resident generation unfreezes before model/mmap teardown')
            if before_unfreeze is not None:
                require_exact_live_restore(before_unfreeze,live_state_fingerprint(optimizer,torch=torch,guard=guard))
                require(pointers==[(id(p),p.data_ptr(),p._version,p.grad) for p in optimizer.parameters.values()],
                    'Unfreeze/fullcollect keeps actual current207 Parameter identities/pointers/versions/gradients')
            write_json(folder/'actual_resident_GC_unfreeze.json',{'source':record(__file__),
                'full_collect_objects':objects,'teardown_wall_seconds':time.monotonic()-began,
                'all207_threeAdam_all4RNG_before_after_unfreeze_exact':before_unfreeze is not None,'final_frozen_objects':0})


def checked(config,review,action):
    storage();c=read_json(config);r=read_json(review)
    require(c['kind']=='ASR18_original_first4_native_throughput_component_configuration_v1'
        and c['production_enabled'] is False and c['native_updates'] in (1,4) and c['window_count']==1,
        'Exact original first4 plan prefix, finite grouped diagnostic only')
    require(c['grouped_Transformer'] is True and type(c['grouped_original_head_reference']) is bool and c['profile_enabled'] is True,
        'Explicit grouped topology and independently declared original-head reference, no serial byte claim')
    require(c['grouped_oracle_observers_enabled'] is False,'Coarse phases only, no row/gradient readback observers')
    import ast
    policy_tree=ast.parse((S/'tdt_blocked_fp32_joint_view_v1.py').read_text(encoding='utf-8'))
    blocked_policy=ast.literal_eval(next(n.value for n in policy_tree.body if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='POLICY' for v in n.targets)))
    require(c['declared_loss_joint_policy']==blocked_policy and c['loss_complete_union_cap_bytes']==1_700_000_000
        and c['loss_packed_work_unit_cap_bytes']==128_000_000,'Exact larger-T numerical and complete allocation policy')
    require(c['optimization'] in ('baseline','gc_freeze_only','frozen_cache','frozen_cache_gc_freeze') and type(c['profile_enabled']) is bool,
        'One explicit causal variant and matched observation mode, no implicit combination')
    require(r['accepted'] is True and r['reviewer']=='/root' and r['configuration']==record(config)
        and r['scope']=='finite_original_first4_native_throughput_component_v1' and action in r['allowed_actions']
        and r['resource_policy']==c['resource_policy'],'Exact separately reviewed action and unchanged resource policy')
    manifest=read_json(c['source_manifest']['path']);require(record(c['source_manifest']['path'])==c['source_manifest'],'Exact source closure')
    for value in manifest['bindings']:require(record(value['path'])==value,'Frozen dependency changed')
    require(record(__file__)==c['harness_source'],'Actual parent/worker entrypoint identity')
    c['_configuration']=record(config);return c,r


class ParentWatch(H.Watch):
    def check(self):
        value=super().check();value['owned_current_and_lifetime']=private_lifetime_limit(128_000_000);return value


def fresh(increment,guard):
    values=[]
    for _ in range(3):
        guard();v=fresh_admit(increment);values.append(v);time.sleep(.05)
    require(all(v['passed'] for v in values),'Three fresh full prospective admissions; no unchanged retry')
    return {'passed':True,'incremental_forecast_bytes':increment,'samples':values}


def hot_guard(folder):
    # Same actual native hot guard as runtime6: one source-owned stop stat.
    require(not (folder/'stop.json').exists(),'Actual owned stop request; preserve partial failure')


def _tuple(value):return tuple(_tuple(v) for v in value) if isinstance(value,list) else value


def restore_rng(value,*,torch,np,random):
    random.setstate(_tuple(value['python']))
    np.random.set_state((value['numpy'][0],np.array(value['numpy'][1],dtype=np.uint32),*value['numpy'][2:]))
    torch.set_rng_state(torch.tensor(value['torch'],dtype=torch.uint8))
    torch.cuda.set_rng_state_all([torch.tensor(v,dtype=torch.uint8) for v in value['cuda']])
    require(digest(H.capture_rng(torch,np,random))==digest(value),'Exact measured baseline all4 initial RNG, never arbitrary seed')


def gpu(c,folder,watch,mode,arm,shared,baseline):
    import random
    from preload_nemo_linker_v1 import preload
    began=time.monotonic();coherent=preload();import torch,numpy as np
    from fastmss_owned_process_closure_v2 import verify as verify_process0
    closed_CPU=read_json(shared['actual_CPU_process0']['path'])
    require(record(shared['actual_CPU_process0']['path'])==shared['actual_CPU_process0'], 'Exact newly prepared CPU closure')
    verify_process0(closed_CPU,kind='ASR18_owned_CPU_process_completion_v1',owner=OWNER,guard=lambda:hot_guard(folder))
    CPU_report=read_json(shared['CPU_report']['path'])
    require(record(shared['CPU_report']['path'])==shared['CPU_report'] and CPU_report['owner']==shared['owner']
        and CPU_report['manifest']==shared['manifest'],'Actual closed CPU report and same immutable owner/manifest')
    torch.set_num_threads(4)
    physical=int(torch.cuda.get_device_properties(0).total_memory)
    require(physical>=17_000_000_000,'12GB allocator +3GB desktop/driver +2GB physical reserve')
    require(c['CUDA_capacity_policy']=={'allocator_cap_bytes':12_000_000_000,'minimum_physical_bytes':17_000_000_000,
        'driver_desktop_reserve_bytes':3_000_000_000,'hard_free_physical_reserve_bytes':2_000_000_000},'Exact newly authorized GPU capacity')
    capacity={'physical_cuda_device_bytes':physical,'cap_bytes':12_000_000_000,
        'per_process_allocator_fraction':12_000_000_000/physical,'authority_source':record(__file__)}
    torch.cuda.set_per_process_memory_fraction(capacity['per_process_allocator_fraction'],device=0)
    cap=c['resource_policy'][arm+'_worker_private_forecast_bytes'] if mode=='train' else 3_350_000_000
    watch.cap=cap;guard=lambda:hot_guard(folder)
    def stage(name,**extra):
        watch.phase=name;watch.check();torch.cuda.synchronize()
        v=sample_memory(os.getpid(),name);v['owned_current_and_lifetime']=private_lifetime_limit(cap)
        v.update(extra);v['monotonic']=time.monotonic();v['utc']=now()
        v['cuda']={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
            'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}
        free,total=torch.cuda.mem_get_info();prospective=int(extra.get('prospective_joint_GPU_bytes',0))
        require(total==physical and free-prospective>=2_000_000_000,'Actual2GB physical reserve including prospective complete joint union')
        v['global_GPU_free_bytes']=free;v['physical_GPU_bytes']=total
        require(v['cuda']['peak_allocated']<=12_000_000_000,'New human-authorized12GB allocator cap')
        with (folder/'stages.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(v)+'\n')
        return v
    def admit(name,increment):
        # Conservative unchanged parent128+ALL-shim64 future allowance is
        # additional here; no negative remainder or borrowed/private credit.
        value=fresh(increment+192_000_000,watch.check);write_json(folder/(name+'_admission.json'),value)
    from native_tdt_minimal_import_v1 import prepare
    minimal=prepare();stage('runtime_ready',coherent=coherent,capacity=capacity)
    admit('original_readonly_base_and_teacher',900_000_000)
    from stream_frozen_encoder_matrices_v1 import load
    from streamed_generation_device_v1 import apply as logical_device
    from asr18_token_only_carrier_v1 import load as token_carrier
    base=Path((P/'work/base_model_path.txt').read_text().strip());model,loader=load(base);logical=logical_device(model);processor=token_carrier(base)
    teacher={name:copy.deepcopy(getattr(model,name)).eval().requires_grad_(False) for name in ('encoder_projector','decoder','joint')} if mode=='train' else None
    from bpe_broad_streamed_bootstrap_v2 import bootstrap
    optimizer,provenance,snapshot=bootstrap(model,processor,torch=torch,stage=stage,admit=admit)
    from tdt_joint_canonical_2d_fp32_v2 import apply as canonical_joint
    from asr18_temporal_tiled_subsampling_v2 import apply as conv_only
    provenance.update(canonical_joint=canonical_joint(model.joint),conv_only=conv_only(model,torch=torch,guard=guard),
        minimal_import=minimal,logical_device=logical,frozen_loader=loader)
    from asr18_full_run_source_context_v1 import FullRunSourceContext,full_run_next_draw_packet
    from asr18_live_state_fingerprint_v1 import live_state_fingerprint,require_exact_live_restore,tensor_digest
    from asr18_streamed_persistence_fingerprint_v1 import normalize_fingerprint,verify_persistence
    from asr18_metadata_checkpoint_binding_v1 import MetadataCheckpointBinding
    from bpe_streaming_training_state_v1 import checkpoint,restore
    from bpe_broad_production_state_v1 import CommitJournal
    context=FullRunSourceContext(c['planning_arguments'],guard=guard);schedule=context.new_schedule()
    require(digest(snapshot)==digest(context.snapshot),'Original source snapshot unchanged')
    binding=MetadataCheckpointBinding(c['augmentation_identity'],guard=guard)
    identity={'kind':'ASR18_original_first4_throughput_unqualified_checkpoint_identity_v1','config':c['_configuration'],
        'arm':arm,'shared_feature_owner':shared['owner'],'production_enabled':False,'production_source_eligible':False}
    if mode=='recover':
        actual=read_json(baseline/'actual_process0.json')
        require(actual['actual_Popen_wait_code']==0 and actual['all_captured_identities_dead']
            and not any(H.alive(v) for v in actual['captured_identities']),'Actual native0 and every exact captured identity dead before fresh restore')
        saved=read_json(actual['report']['path']);require(record(actual['report']['path'])==actual['report'] and saved['arm']==arm,'Exact native report/arm')
        pre=normalize_fingerprint(read_json(saved['PRE_save_fingerprint']['path']))
        ready=stage('actual_fresh_restore_ready')
        require(ready['worker_private_memory']['private_bytes']<=3_000_000_000
            and ready['worker_private_memory']['private_bytes']+350_000_000<=3_350_000_000,'Actual fresh restore startup bound')
        admit('current_checkpoint_restore',350_000_000)
        state,journal,branch=restore(saved['checkpoint'],torch=torch,model=model,optimizer=optimizer,identity=identity,
            augmentation=binding,new_journal_path=folder/'restored_journal_prefix.jsonl',diagnostic_only=True,guard=guard)
        schedule.load_state_dict(state);require_exact_live_restore(pre,live_state_fingerprint(optimizer,torch=torch,guard=guard))
        require(digest(state)==saved['sampler_sha256'] and optimizer.updates==c['native_updates']
            and digest(full_run_next_draw_packet(schedule,context,optimizer_updates=c['native_updates'],guard=guard))==saved['next_draw_sha256'],
            'Full scientific schedule/nextdraw and all4 RNG exact before any further draw')
        stage('actual_all207_threeAdam_all4RNG_current_restored');context.close()
        write_json(folder/'recovery_result.json',{'passed':True,'source':record(__file__),'producer':H.pid_identity(),'arm':arm,
            'all207_threeAdam_all4RNG_sampler_nextdraw_exact':True,'actual_native_updates':0,'branch':branch,'production_enabled':False})
        return
    if arm=='candidate':
        base_initial=read_json(baseline/'initial_state.json');require(record(baseline/'initial_state.json')==c['candidate_baseline_initial'],
            'Separately pinned actual baseline initial state')
        restore_rng(base_initial['all4_rng'],torch=torch,np=np,random=random)
    def teacher_digest():
        return {name:{key:tensor_digest(value,torch=torch,guard=guard) for key,value in sorted(module.state_dict().items())}
            for name,module in sorted(teacher.items())}
    initial=live_state_fingerprint(optimizer,torch=torch,guard=guard);teacher_initial=teacher_digest()
    initial_value={'full_state':initial,'all4_rng':H.capture_rng(torch,np,random),'teacher':teacher_initial,
        'original_schedule_sha256':digest(schedule.state_dict()),'active207_pointers_versions':
        {name:{'ptr':p.data_ptr(),'version':p._version} for name,p in optimizer.parameters.items()}}
    if arm=='candidate':
        require_exact_live_restore(normalize_fingerprint(base_initial['full_state']),initial)
        require(base_initial['teacher']==teacher_initial and base_initial['original_schedule_sha256']==initial_value['original_schedule_sha256'],
            'Paired actual teacher/full source sampler initial state')
        if c['optimization'] in ('frozen_cache','frozen_cache_gc_freeze'):
            from asr18_frozen_encoder_cuda_cache_v3 import install,assert_idle
            def state_digest():return digest({'state':live_state_fingerprint(optimizer,torch=torch,guard=guard),'teacher':teacher_digest()})
            def admit_cache(request):
                require(request['logical_new_persistent_cuda_bytes']==1_217_066_496
                    and request['host_private_savings_credited_bytes']==0,'Exact additional bank, never host credit')
                admit('persistent294_cache_upload',1_217_066_496+4*1024**2)
                require(torch.cuda.memory_allocated()+1_217_066_496+16*1024**2<3_000_000_000,
                    'Candidate persistent bank plus explicit observed outputs inside unchanged3GB idle/endcache gate')
                return True
            before=time.monotonic();policy=install(model,torch=torch,bank_binding=loader['bank'],guard=guard,
                admit_cache=admit_cache,owner_state_digest=state_digest)
            write_json(folder/'actual_cache_install.json',{'policy':policy,'wall_seconds':time.monotonic()-before,'idle':assert_idle(model,torch=torch,guard=guard)})
    write_json(folder/'initial_state.json',initial_value)
    from fastmss_phase_aware_feature_window_v6 import PhaseAwareFeatureWindowReader
    from bpe_production_window_coordinator_v4 import WindowCoordinator
    from functools import partial
    from asr18_grouped_precomputed_native_v4 import native_update as grouped_update
    native_update=partial(grouped_update,reference_original_head=c['grouped_original_head_reference'],
        observe=lambda *args:profiler.observe_row(*args))
    from asr18_native_throughput_profiler_v2 import NativeProfiler
    plan=read_json(c['plan']['path']);require(record(c['plan']['path'])==c['plan'] and len(plan['updates'])==4
        and digest(plan)==digest(context.validation_plan(schedule,completed_boundaries=0,maximum_updates=4)),
        'Exact immutable original four-update plan, never independent reshuffle')
    reader=PhaseAwareFeatureWindowReader(shared['owner'],shared['manifest'],guard=guard)
    journal=CommitJournal(folder/'actual_commits.jsonl');coordinator=WindowCoordinator(schedule,optimizer,journal,plan,resolve_identity=context.resolve_identity)
    original_step=optimizer.step
    def positive_step(*a,**kw):
        out=original_step(*a,**kw)
        write_json(folder/'actual_positive_frontier.json',{'actual_updates':optimizer.updates,'clocks':[64250+optimizer.updates,2000+optimizer.updates,62250+optimizer.updates],
            'producer':H.pid_identity(),'configuration':c['_configuration'],'source':record(__file__),'scientific_sampler_commit_pending':True,'production_enabled':False})
        return out
    optimizer.step=positive_step
    with cache_backward_calls(model,torch=torch,enabled=arm=='candidate' and c['optimization'] in ('frozen_cache','frozen_cache_gc_freeze')), resident_gc_scope(folder,enabled=c['optimization'] in ('gc_freeze_only','frozen_cache_gc_freeze'),optimizer=optimizer,torch=torch,guard=guard):
        from asr18_grouped_coarse_phase_profiler_v2 import CoarseProfiler
        Profiler=CoarseProfiler
        profiler=Profiler(model,optimizer,torch=torch,folder=folder,guard=guard);profile_policy=profiler.install()
        ready=stage('actual_all_setup_before_first_forward')
        require(ready['worker_private_memory']['private_bytes']<=(4_217_066_496 if c['optimization'] in ('frozen_cache','frozen_cache_gc_freeze') else 3_000_000_000),
            'Source startup bound plus entire additional candidate bank, no host credit')
        require(ready['worker_private_memory']['private_bytes']+750_000_000+64_000_000<cap,
            'Actual teacher/cache-inclusive ready leaves unchanged variation and write reserves')
        # No cold-state reset: first update includes its actual cold CUDA modules.
        history=[];native_began=time.monotonic()
        try:
            for index,entry in enumerate(plan['updates'][:c['native_updates']]):
                def verify(entry,evidence):
                    actual=read_json(evidence['path']);require(record(evidence['path'])==evidence and all(actual[k]==entry[k]
                        for k in ('native_transaction_sha256','before_sampler_sha256','after_sampler_sha256','batch_id')),'Actual closed prepared transaction')
                    return True
                def execute(entry,tx):
                    rows=context.resolve_rows(tx.phase,tx.TR_draws,tx.replay_rows);require(digest(rows)==digest(entry['rows']),'Exact full original source/targets/replay order')
                    stage('before_native_'+str(index+1));lease=read_json(LEASE);lease['actual_training_optimizer_running']=True;write_json(LEASE,lease)
                    def features(i):return reader.take_row(index,i,context=entry['contexts'][i],native_transaction_sha256=tx.identity_digest,before_sampler_sha256=digest(tx.before))
                    try:
                        out=profiler.run_update(lambda:native_update(model,teacher,optimizer,rows,entry['draws'],
                            feature_for_row=lambda i:profiler.feature(features,i),update_binding=reader.update_record(index),torch=torch,
                            population_mean_U=schedule.means[tx.phase],update=index+1,stage=lambda name,**kw:profiler.stage(stage,name,**kw),
                            stop_check=profiler.checked_guard,phase=tx.phase,feature_update_index=index),update=index+1)
                        require(reader.prepared_payload(index)==out['augmentation'],'Exact unchanged feature payload consumption')
                        return out
                    finally:
                        lease=read_json(LEASE);lease['actual_training_optimizer_running']=False;write_json(LEASE,lease)
                result=coordinator.consume(prepare_evidence=lambda entry:reader.update_record(index),native_call=execute,verify_preparation=verify)
                history.append(result['native']);write_json(folder/'native_progress.json',{'actual_updates':optimizer.updates,'committed':coordinator.index,'arm':arm})
                if arm=='candidate' and c['optimization'] in ('frozen_cache','frozen_cache_gc_freeze'):assert_idle(model,torch=torch,guard=guard)
        finally:profile=profiler.close();optimizer.step=original_step
        native_wall=time.monotonic()-native_began
    require(coordinator.index==optimizer.updates==c['native_updates'] and digest(schedule.state_dict())==plan['updates'][c['native_updates']-1]['after_sampler_sha256'],
        'All four actual optimizer and scientific sampler/journal commits, no dropped row')
    pre=live_state_fingerprint(optimizer,torch=torch,guard=guard);prepath=folder/'PRE_save_fingerprint.json';write_json(prepath,pre)
    state=copy.deepcopy(schedule.state_dict());next_packet=full_run_next_draw_packet(schedule,context,optimizer_updates=c['native_updates'],guard=guard)
    nextpath=folder/'next_draw.json';write_json(nextpath,next_packet)
    before_write=stage('before_actual_checkpoint_write')
    require(before_write['worker_private_memory']['private_bytes']+64_000_000<=cap,'Entire write increment remains in whole-worker cap')
    admit('actual_checkpoint_write',64_000_000);before=time.monotonic()
    saved=checkpoint(folder/('after'+str(c['native_updates'])+'.state.json'),torch=torch,optimizer=optimizer,state=state,provenance=provenance,identity=identity,
        augmentation=binding,journal=journal,diagnostic_only=True,guard=guard)
    proof=verify_persistence(saved,expected_pre_save=pre,identity=identity,sampler_state=state,journal_prefix=journal.binding(),diagnostic_only=True,guard=guard)
    require_exact_live_restore(pre,live_state_fingerprint(optimizer,torch=torch,guard=guard));write_json(folder/'persistence_verified.json',proof)
    checkpoint_wall=time.monotonic()-before
    require(teacher_initial==teacher_digest(),'Teacher unchanged through actual full objectives/update/checkpoint')
    point=stage('actual_checkpoint_persisted');context.close()
    report={'passed':True,'source':record(__file__),'producer':H.pid_identity(),'configuration':c['_configuration'],'arm':arm,
        'optimization':c['optimization'],'profile_enabled':c['profile_enabled'],
        'grouped_oracle_observers_enabled':c['grouped_oracle_observers_enabled'],
        'initial_state':record(folder/'initial_state.json'),'profile':profile,'profile_policy':profile_policy,'actual_optimizer_updates':c['native_updates'],
        'history':history,'checkpoint':saved,'PRE_save_fingerprint':record(prepath),'persistence':record(folder/'persistence_verified.json'),
        'sampler_sha256':digest(state),'next_draw':record(nextpath),'next_draw_sha256':digest(next_packet),'journal_prefix':journal.binding(),
        'native_window_wall_seconds_including_profile':native_wall,'checkpoint_persistence_wall_seconds':checkpoint_wall,
        'startup_before_native_seconds':native_began-began,'final_stage':point,'production_source_eligible':False,'production_enabled':False}
    write_json(folder/'training_result.json',report)


def leaf(config,review,folder,mode,arm,shared,baseline):
    c,_=checked(config,review,arm);folder=Path(folder);producer=H.pid_identity()
    write_json(folder/'boot.json',{'producer':producer,'source':record(__file__),'configuration':record(config),'mode':mode,'arm':arm})
    cap=c['resource_policy'][arm+'_worker_private_forecast_bytes'] if mode=='train' else 3_350_000_000
    watch=H.Watch(folder,cap,worker=True);code=1
    try:
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            lease=read_json(LEASE)
            if lease['owner']==OWNER and lease['actual_GPU_running'] is True and lease['actual_owned_worker_pid']==producer['pid']:break
            watch.check();time.sleep(.05)
        else:raise ValueError('Exact sole-owned GPU handshake required')
        gpu(c,folder,watch,mode,arm,read_json(shared),Path(baseline) if baseline else None);code=0
    finally:watch.close(code)


def run_leaf(config,review,folder,mode,arm,shared,baseline,parent_watch):
    c=read_json(config);cap=c['resource_policy'][arm+'_worker_private_forecast_bytes'] if mode=='train' else 3_350_000_000
    write_json(folder.parent/(folder.name+'_prelaunch.json'),fresh(cap+192_000_000,parent_watch.check))
    prior=read_json(LEASE);require(prior['owner']==OWNER and prior['actual_GPU_running'] is False,'Actual idle sole GPU lease')
    folder.mkdir();known={};native=None;child=None;code=1;timeline=None
    env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='0';env['TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT']='16'
    require(not env.get('DISABLE_ADDMM_CUDA_LT') and not env.get('TORCH_CUDNN_V8_API_DISABLED'),'No failed backend flags inherited')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    began=time.monotonic()
    try:
        with (folder/'worker.log').open('xb') as log:
            args=[c['GPU_python']['path'],'-X','utf8','-B',str(Path(__file__).resolve()),'--config',config,'--ROOT-review',review,
                '--folder',str(folder),'--leaf',mode,'--action',arm,'--shared',str(shared)]
            if baseline:args+=['--baseline-train',str(baseline)]
            child=subprocess.Popen(args,env=env,cwd=P,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            launcher=H.pid_identity(child.pid);known[(launcher['pid'],launcher['create_time'])]=launcher
            while child.poll() is None:
                parent_watch.check();require(time.monotonic()-began<c['maximum_GPU_phase_seconds'],'Finite profiled native/recovery wall')
                import psutil
                try:
                    for process in [psutil.Process(child.pid),*psutil.Process(child.pid).children(recursive=True)]:
                        try:value=H.pid_identity(process.pid);known[(value['pid'],value['create_time'])]=value
                        except psutil.NoSuchProcess:pass
                except psutil.NoSuchProcess:pass
                if native is None and (folder/'boot.json').exists():
                    boot=read_json(folder/'boot.json');native=boot['producer']
                    require(boot['source']==record(__file__) and boot['configuration']==record(config) and H.alive(native)
                        and (native['pid'],native['create_time']) in known,'Actual early native PID/source/config before heavy imports')
                    lease=copy.deepcopy(prior);lease.update(actual_GPU_running=True,actual_owned_worker_pid=native['pid'],
                        actual_owned_launcher_pid=child.pid,actual_training_optimizer_running=False,status='native_throughput_profile',updated_at_utc=now());write_json(LEASE,lease)
                    print(json.dumps({'actual_native_PID':native['pid'],'actual_launcher_PID':child.pid,'mode':mode,'arm':arm}),flush=True)
                # Timeline helper is independently source-reviewed and stays in
                # the SAME unchanged all-shim64MB domain as every launcher.
                if mode=='train' and native is not None and timeline is None:
                    from asr18_gpu_utilization_timeline_v1 import GPUUtilizationTimeline
                    timeline_folder=folder/'gpu_timeline';timeline_folder.mkdir()
                    timeline=GPUUtilizationTimeline(timeline_folder,guard=parent_watch.check)
                    observed=timeline.start();item=H.pid_identity(observed['pid'])
                    require(item['ppid']==os.getpid() and abs(item['create_time']-observed['create_time'])<.001,
                        'Same owned monitor Popen identity/FILETIME and psutil observation')
                    known[(item['pid'],item['create_time'])]=item
                own=parent_watch.check()['worker_private_memory']['private_bytes'];total=own;shim=0
                for item in known.values():
                    if not H.alive(item):continue
                    mem=sample_memory(item['pid'],'throughput_exact_owned_family')['worker_private_memory']
                    if not mem['available'] and not H.alive(item):continue
                    require(mem['available'],'Available exact captured private telemetry, never guessed0')
                    total+=mem['private_bytes']
                    if native is not None and item['pid']!=native['pid']:shim+=mem['private_bytes']
                require(total<=cap+192_000_000 and (native is None or shim<=64_000_000),
                    'Unchanged whole current-family/parent128/all-shim64 caps; unresolved startup is not misclassified')
                if timeline is not None:timeline.poll()
                time.sleep(.1)
            actual=child.wait()
            if timeline is not None:timeline_result=timeline.stop();timeline=None
            deadline=time.monotonic()+10
            while any(H.alive(v) for v in known.values()) and time.monotonic()<deadline:parent_watch.check();time.sleep(.05)
            require(actual==0 and native is not None and not any(H.alive(v) for v in known.values()),'Actual handle0 and ALL captured identities dead')
            closure=read_json(folder/'guard_closure.json');require(closure['actual_exit_code']==0 and closure['pid']==native['pid']
                and closure['monitor_clean'] and closure['monitor_joined'] and closure['samples']>=3,'Exact clean joined native guard0')
            result=folder/('training_result.json' if mode=='train' else 'recovery_result.json');report=read_json(result)
            require(report['passed'] is True and report['producer']==native and report['source']==record(__file__),'Exact actual result/source/producer')
            proof={'kind':'ASR18_throughput_component_actual_owned_process0_v1','actual_Popen_wait_code':0,
                'all_captured_identities_dead':True,'captured_identities':list(known.values()),'native':native,'source':record(__file__),
                'configuration':record(config),'report':record(result),'guard':record(folder/'guard_closure.json'),
                'wall_seconds':time.monotonic()-began,'GPU_timeline':timeline_result if mode=='train' else None}
            write_json(folder/'actual_process0.json',proof);code=0;return record(folder/'actual_process0.json')
    finally:
        if timeline is not None:timeline.stop(check_guard=False)
        # Reconcile a verified early native even if its outer shim already exited.
        if (folder/'boot.json').exists():
            boot=read_json(folder/'boot.json');item=boot['producer']
            require(boot['source']==record(__file__) and boot['configuration']==record(config)
                and (item['pid']==launcher['pid'] or item['ppid']==launcher['pid']), 'Exact early entrypoint on exit/failure')
            known[(item['pid'],item['create_time'])]=item
            if H.alive(item):
                import psutil
                for proc in psutil.Process(item['pid']).children(recursive=True):
                    try:value=H.pid_identity(proc.pid);known[(value['pid'],value['create_time'])]=value
                    except psutil.NoSuchProcess:pass
        from asr18_owned_family_observation_v4 import terminate_known_family,wait_captured_dead
        if child is not None:
            if code:terminate_known_family(child,launcher,known)
            wait_captured_dead(known,timeout=10)
            if child.poll() is None:child.terminate()
            child.wait(timeout=5)
        require(not any(H.alive(v) for v in known.values()),'No surviving native/monitor/shim before lease release')
        write_json(LEASE,prior)


# run_CPU is copied without numerical/protocol changes from reviewed parent15;
# only its actual __file__ producer identity is now this exact component parent.
def run_CPU(config, script, arguments, folder, *, guard, report_relative='report.json'):
    """One actual bounded CPU controller/cleanup child, source-bound Popen proof."""
    from fastmss_owned_process_closure_v2 import identity, observe
    folder.mkdir(exist_ok=False); monitor = folder/'guard'; log = folder/'worker.log'; phase_began=time.monotonic(); known={}; family_peak=0; family_samples=0; outer_seen=False
    env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = '-1'
    require(record(config['CPU_python']['path']) == config['CPU_python']
            and record(config['CPU_guard_source']['path']) == config['CPU_guard_source'], 'Pinned CPU interpreter/guard before launch')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): env[key] = '1'
    admission = fresh_admit(config['CPU_owned_allowance_bytes']); write_json(folder/'admission.json', admission)
    require(admission['passed'], 'Fresh CPU allowance does not fit; preserve, no DATA hold')
    with log.open('xb') as stream:
        child = subprocess.Popen([config['CPU_python']['path'], '-X', 'utf8', '-B', config['CPU_guard_source']['path'],
            '--folder', str(monitor), '--seconds', str(config['maximum_CPU_phase_seconds']), '--script', str(script), '--', *arguments],
            env=env, stdout=stream, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        launcher = identity(child.pid); began = time.monotonic()
        known[(launcher['pid'],launcher['create_time'])] = launcher
        exit_transition = None
        try:
            from asr18_owned_family_observation_v4 import owned_family_sample
            from asr18_outer_CPU_exit_transition_v1 import wait_actual_CPU_family_exit
            while child.poll() is None:
                guard(); require(time.monotonic()-began < config['maximum_CPU_phase_seconds']+10, 'Finite CPU controller timeout')
                try:family=owned_family_sample(child,launcher,known=known,guard=guard)
                except ProcessLookupError:
                    # A launcher syscall can race normal exit before poll sees0.
                    # Drain captured children under their own unchanged cap;
                    # the outer native parent retains its separate ParentGuard.
                    exit_transition = wait_actual_CPU_family_exit(child,launcher,known,guard=guard,
                        CPU_family_private_allowance_bytes=config['CPU_owned_allowance_bytes'])
                    break
                family_samples+=1;family_peak=max(family_peak,family['private_bytes_sum'])
                outer_seen=outer_seen or family['outer_launcher_included']
                write_json(folder/'owned_CPU_family_live.json',family)
                require(family['private_bytes_sum']<=config['CPU_owned_allowance_bytes'],'Whole CPU family private allowance exceeded, including outer launcher')
                time.sleep(.2)
            require(child.wait() == 0, 'Actual CPU phase failed; artifacts preserved, no unchanged retry')
            if exit_transition is None:
                exit_transition = wait_actual_CPU_family_exit(child,launcher,known,guard=guard,
                    CPU_family_private_allowance_bytes=config['CPU_owned_allowance_bytes'])
            family_peak=max(family_peak,exit_transition['captured_CPU_family_private_peak_bytes'])
            write_json(folder/'owned_CPU_family_exit_transition.json',exit_transition)
        except BaseException:
            from asr18_owned_family_observation_v4 import terminate_known_family
            monitor.mkdir(exist_ok=True); write_json(monitor/'stop.json', {'reason': 'owned_parent_failed'})
            if child.poll() is None:
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
            # Even an exited outer launcher can leave captured descendants.
            # Observe their termination before considering this family retired.
            terminate_known_family(child,launcher,known)
            raise
    require(family_samples>=1 and outer_seen,'Actual outer launcher+CPU-family telemetry required; no guessed zero overhead')
    write_json(folder/'owned_CPU_family_completion.json',{'passed':True,'measured_family_peak_private_bytes':family_peak,
        'family_samples':family_samples,'reviewed_entire_CPU_allowance_bytes':config['CPU_owned_allowance_bytes'],
        'inner_controller_family_allowance_bytes':config['CPU_inner_owned_allowance_bytes'],
        'additional_outer_launcher_reserve_bytes':config['CPU_outer_controller_launcher_reserve_bytes'],
        'outer_launcher_included':True,'actual_parent_phase_wall_seconds':time.monotonic()-phase_began,
        'launcher_disappearance_or_normal_exit_drain':exit_transition,
        'only_actual_owned_ancestry_observed':True})
    report = record(folder/report_relative)
    return observe(child, kind='ASR18_owned_CPU_process_completion_v1', owner=OWNER, report=report,
        worker_source=record(script), parent_source=record(__file__), guard_source=config['CPU_guard_source'],
        guard_closure=record(monitor/'closure.json'), launcher_identity=launcher,
        parent_guard_observation=sample_memory(os.getpid(), 'CPU_actual_wait_observer'), output=folder/'actual_owned_process0.json', guard=guard)


def parent(config,review,folder,arm,baseline_folder=None):
    c,r=checked(config,review,arm);folder=Path(folder).resolve()
    require(folder.is_relative_to(A) and not folder.exists(),'One fresh owned component output')
    folder.mkdir();watch=ParentWatch(folder,128_000_000);code=1
    try:
        if arm=='baseline':
            write_json(folder/'CPU_prelaunch.json',fresh(c['CPU_owned_allowance_bytes']+128_000_000,watch.check))
            CPU=run_CPU(c,Path(c['CPU_controller_source']['path']),['--bridge',json.dumps(c['bridge']),'--output',str(folder/'CPU_prepared')],
                folder/'CPU_controller',guard=watch.check,report_relative='../CPU_prepared/report.json')
            closed=read_json(CPU['path']);prepared=read_json(closed['report']['path']);shared_path=folder/'shared_features.json'
            require(prepared['passed'] and prepared['all_children_actually_closed0'] and prepared['actual_optimizer_updates']==0,
                'Genuine newly prepared immutable CPU8 window only')
            shared={'kind':'ASR18_new_owned_immutable_first4_paired_features_v1','owner':prepared['owner'],'manifest':prepared['manifest'],
                'actual_CPU_process0':CPU,'CPU_report':closed['report'],'configuration':record(config),'source':record(__file__),
                'reference_and_frontend_children_all0':True,'no_old_namespace_adopted':True,'retain_for_separately_reviewed_matched_candidate':True}
            write_json(shared_path,shared);baseline_train=None
        else:
            require(baseline_folder is not None and r['baseline_overall_result']==record(Path(baseline_folder)/'overall_result.json'),
                'Candidate only after separately reviewed genuine baseline native/fresh recovery/outer0')
            b=read_json(r['baseline_overall_result']['path']);require(b['passed'] and b['arm']=='baseline','Actual baseline pass')
            shared_path=Path(baseline_folder)/'shared_features.json';baseline_train=Path(baseline_folder)/'train'
            require(record(baseline_train/'initial_state.json')==c['candidate_baseline_initial'],'Actual exact initial baseline state bound before candidate')
        native=run_leaf(config,review,folder/'train','train',arm,shared_path,baseline_train,watch)
        recovery=run_leaf(config,review,folder/'recover','recover',arm,shared_path,folder/'train',watch)
        write_json(folder/'overall_result.json',{'passed':True,'arm':arm,'actual_native_updates':c['native_updates'],'actual_native_process0':native,
            'actual_fresh_recovery_process0':recovery,'shared_features':record(shared_path),'production_enabled':False,
            'source':record(__file__),'configuration':record(config)});code=0
    finally:watch.close(code)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--ROOT-review',required=True);p.add_argument('--folder',required=True)
    p.add_argument('--action',choices=('baseline','candidate'),required=True);p.add_argument('--baseline-folder')
    p.add_argument('--leaf',choices=('train','recover'));p.add_argument('--shared');p.add_argument('--baseline-train');a=p.parse_args()
    if a.leaf:leaf(a.config,a.ROOT_review,Path(a.folder),a.leaf,a.action,a.shared,a.baseline_train)
    else:parent(a.config,a.ROOT_review,Path(a.folder),a.action,a.baseline_folder)
