"""NEW dense-FP32 same-policy oracle, original autograd. No ASR/Adam/draw/update."""
import argparse,copy,gc,json,os,random,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish'
sys.path.insert(0,str(S))
import run_ASR18_native_throughput_component_v7 as H7
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from asr18_memory_diagnosis_v2 import storage,now
from resource_watchdog_v2 import sample_memory
from run_ASR18_full_run_owner_v15 import private_lifetime_limit
H=H7.H;OWNER=H7.OWNER;LEASE=H7.LEASE;A=H7.A


def checked(config,review):
    storage();c=read_json(config);r=read_json(review)
    require(c['kind']=='ASR18_dense_FP32_trained_joint_byte_oracle_configuration_v1'
        and c['source']==record(__file__) and c['optimizer_updates']==0 and c['production_enabled'] is False,
        'Finite source-bound trained joint only')
    require(c['worker_private_forecast_bytes']==6_500_000_000 and c['parent_private_bytes']==128_000_000
        and c['all_shim_private_bytes']==64_000_000 and c['maximum_wall_seconds']==240,
        'Exact unchanged-domain finite resource proposal')
    require(r['accepted'] is True and r['reviewer']=='/root' and r['configuration']==record(config)
        and r['scope']=='trained_joint_dense_FP32_full_clip_compactDP_oracle_v1','Exact ROOT execution review')
    manifest=read_json(c['source_manifest']['path'])
    require(record(c['source_manifest']['path'])==c['source_manifest'],'Frozen source manifest')
    for pin in manifest['bindings']:require(record(pin['path'])==pin,'Frozen source changed')
    for pin in c['fixtures']:require(record(pin['path'])==pin,'Actual trained fixture changed')
    return c


def validate(c,folder,watch):
    from preload_nemo_linker_v1 import preload
    coherent=preload();require(not coherent['cuda_initialized'] and coherent['actual_binding_linker']['version']==[12,9],
        'Same coherent CUDA12.9 preload')
    import torch,numpy as np
    torch.set_num_threads(4)
    physical=int(torch.cuda.get_device_properties(0).total_memory)
    require(physical>=17_000_000_000,'12GB allocator +3GB driver/desktop +2GB physical reserve')
    cap={'physical_cuda_device_bytes':physical,'cap_bytes':12_000_000_000,
        'per_process_allocator_fraction':12_000_000_000/physical,'cuda_device_index':0,
        'new_capacity_authority_source':record(__file__),'driver_desktop_reserve_bytes':3_000_000_000,
        'hard_physical_reserve_bytes':2_000_000_000}
    torch.cuda.set_per_process_memory_fraction(cap['per_process_allocator_fraction'],device=0)
    torch.backends.cuda.matmul.allow_tf32=False
    from native_tdt_minimal_import_v1 import prepare
    prepare()
    from tdt_joint_canonical_2d_fp32_v2 import apply as canonical_joint
    from tdt_dense_fp32_joint_view_v1 import DenseFP32JointView,POLICY
    from native_tdt_checkpointed_v4 import chunked_tdt_loss as reference_loss
    from tdt_dense_full_native_v1 import bounded_tdt_loss as candidate_loss
    from tdt_dense_full_native_allocation_budget_v1 import allocation_budget
    guard=lambda:H7.hot_guard(folder)
    def boundary(name,prospective_GPU_bytes=0):
        guard();watch.phase=name;watch.check();private_lifetime_limit(c['worker_private_forecast_bytes'])
        torch.cuda.synchronize();free,total=torch.cuda.mem_get_info()
        require(total==physical and free-prospective_GPU_bytes>=2_000_000_000,'Actual global-card2GB physical reserve and full prospective dense union')
        require(torch.cuda.max_memory_allocated()<=12_000_000_000,'NEW explicit12GB allocator ceiling')
        return {'global_free_bytes':free,'physical_bytes':total,'prospective_GPU_union_bytes':prospective_GPU_bytes}
    def hashes(values):return [H.tensor_hash(v,torch) for v in values]
    def exact(left,right):
        checks=[a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)) for a,b in zip(left,right)]
        return checks
    class Joint(torch.nn.Module):
        def __init__(self,width):super().__init__();self.head=torch.nn.Linear(width,8716)
    initial=H.capture_rng(torch,np,random);observations=[]
    for fixture in c['fixtures']:
        boundary('before_actual_trained_fixture')
        saved=torch.load(fixture['path'],map_location='cpu',weights_only=True,mmap=True)
        before=H.capture_rng(torch,np,random)
        original=Joint(saved['ep'].shape[-1]).to('cuda');original.load_state_dict(saved['joint']);canonical_joint(original)
        H7.restore_rng(before,torch=torch,np=np,random=random)
        joint=DenseFP32JointView(original);head=tuple(joint.parameters())
        require(tuple(id(p) for p in head)==tuple(id(p) for p in original.parameters()),'Exact original two trained Parameters, no copies')
        head_before=hashes(head);versions=[(p.data_ptr(),p._version,p.grad) for p in head]
        e=saved['ep'][:,:15].to('cuda');d=saved['dp'].to('cuda');labels=saved['labels'].to('cuda')
        budget=allocation_budget(15,int(labels.shape[1]),projected_elements=e.numel()+d.numel(),head_elements=sum(p.numel() for p in head),hidden_width=e.shape[-1])
        for scale in (.125,-.375,1.):
            values=[];gradients=[];walls=[];admissions=[]
            for name,function in (('official_full_vocabulary_reference',reference_loss),('complete_clip_compactDP_candidate',candidate_loss)):
                admissions.append(boundary(name,budget['total_budget_bytes']))
                before=H.capture_rng(torch,np,random);began=time.perf_counter()
                ep=e.detach().requires_grad_();dp=d.detach().requires_grad_()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    value=function(joint,ep,dp,labels,8711,8192,(0,1,2,3,4),reduction='none',recompute=True,max_joint_bytes=8_000_000_000,
                        **({'guard':guard,'tile_joint_bytes':1_000_000_000} if function is candidate_loss else {}))
                gradients.append(torch.autograd.grad(value,(ep,dp,*head),grad_outputs=torch.full_like(value,scale)))
                values.append(value.detach());torch.cuda.synchronize();walls.append(time.perf_counter()-began)
                require(digest(before)==digest(H.capture_rng(torch,np,random)),'Each full dense arm all4 RNG unchanged')
                del value,ep,dp;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();boundary(name+'_released')
            checks=exact((values[0],),(values[1],))+exact(gradients[0],gradients[1])
            item={'fixture':fixture,'T':15,'U':int(labels.shape[1]),'upstream_scale':scale,'allocation_budget':budget,'admissions':admissions,
                'checks_loss_ep_dp_weight_bias_strict_bytes':checks,'reference_loss_and_gradient_hashes':hashes((values[0],*gradients[0])),
                'candidate_loss_and_gradient_hashes':hashes((values[1],*gradients[1])),'original_and_candidate_wall_seconds':walls,'passed':all(checks)}
            observations.append(item);write_json(folder/'progress.json',{'observations':observations,'passed_so_far':all(v['passed'] for v in observations)})
            require(item['passed'],'Dense same-policy independent official-DP byte oracle failed; preserve/no tolerance widening')
            del values,gradients;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
        require(head_before==hashes(head) and versions==[(p.data_ptr(),p._version,p.grad) for p in head],
            'Actual head bytes/pointers/versions/no-gradient unchanged')
        del saved,joint,original,head,e,d,labels;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();boundary('trained_fixture_released')
    require(digest(initial)==digest(H.capture_rng(torch,np,random)),'Complete fixture all4 RNG unchanged')
    require(torch.cuda.memory_reserved()<=3_000_000_000,'Unchanged final3GB endcache')
    report={'kind':'ASR18_dense_FP32_trained_joint_byte_oracle_v1','passed':True,'producer':H.pid_identity(),'source':record(__file__),
        'observations':observations,'policy':POLICY,'all4_RNG_unchanged':True,'optimizer_updates':0,'full_ASR_model_loaded':False,
        'canonical_T1_or_BF16_trajectory_equivalence_claimed':False,'native_training_speed_or_hard_shape_fit_claimed':False,'production_enabled':False,
        'final_private':private_lifetime_limit(c['worker_private_forecast_bytes']),'physical_GPU_bytes':physical,
        'CUDA':{'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'end_reserved':torch.cuda.memory_reserved()}}
    write_json(folder/'result.json',report)


def worker(c,config,folder):
    producer=H.pid_identity();write_json(folder/'boot.json',{'producer':producer,'source':record(__file__),'configuration':record(config)})
    watch=H.Watch(folder,c['worker_private_forecast_bytes'],worker=True);code=1
    try:
        until=time.monotonic()+15
        while time.monotonic()<until:
            lease=read_json(LEASE)
            if lease['owner']==OWNER and lease['actual_GPU_running'] is True and lease['actual_owned_worker_pid']==producer['pid']:break
            watch.check();time.sleep(.05)
        else:raise ValueError('Exact actual native lease handshake')
        validate(c,folder,watch);code=0
    finally:watch.close(code)


def parent(c,config,review,folder):
    require(folder.is_relative_to(A) and not folder.exists(),'One new E actual output');folder.mkdir()
    parent_guard=folder/'parent_guard';parent_guard.mkdir()
    watch=H7.ParentWatch(parent_guard,128_000_000);code=1;child=None;native=None;known={};launcher=None
    prior=read_json(LEASE);require(prior['owner']==OWNER and prior['actual_GPU_running'] is False,'Actual sole idle GPU owner')
    try:
        write_json(folder/'prelaunch.json',H7.fresh(c['worker_private_forecast_bytes']+192_000_000,watch.check))
        env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='0';env['TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT']='16'
        require(not env.get('DISABLE_ADDMM_CUDA_LT') and not env.get('TORCH_CUDNN_V8_API_DISABLED'),'No rejected backend flags')
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
        with (folder/'worker.log').open('xb') as log:
            child=subprocess.Popen([c['GPU_python']['path'],'-X','utf8','-B',str(Path(__file__).resolve()),'--config',config,
                '--ROOT-review',review,'--folder',str(folder),'--worker'],cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            launcher=H.pid_identity(child.pid);known[(launcher['pid'],launcher['create_time'])]=launcher;began=time.monotonic()
            import psutil
            while child.poll() is None:
                watch.check();require(time.monotonic()-began<c['maximum_wall_seconds'],'Finite trained-joint oracle wall')
                try:
                    for proc in [psutil.Process(child.pid),*psutil.Process(child.pid).children(recursive=True)]:
                        try:item=H.pid_identity(proc.pid);known[(item['pid'],item['create_time'])]=item
                        except psutil.NoSuchProcess:pass
                except psutil.NoSuchProcess:pass
                if native is None and (folder/'boot.json').exists():
                    boot=read_json(folder/'boot.json');native=boot['producer']
                    require(boot['source']==record(__file__) and boot['configuration']==record(config)
                        and (native['pid'],native['create_time']) in known and H.alive(native),'Actual early native identity')
                    current=copy.deepcopy(prior);current.update(actual_GPU_running=True,actual_owned_worker_pid=native['pid'],
                        actual_owned_launcher_pid=child.pid,actual_training_optimizer_running=False,status='trained_joint_dense_FP32_oracle',updated_at_utc=now())
                    write_json(LEASE,current);print(json.dumps({'actual_native_PID':native['pid'],'launcher_PID':child.pid,'folder':str(folder)}),flush=True)
                total=watch.check()['worker_private_memory']['private_bytes'];shims=0
                for item in known.values():
                    if not H.alive(item):continue
                    value=sample_memory(item['pid'],'trained_joint_exact_family')['worker_private_memory']
                    if not value['available'] and not H.alive(item):continue
                    require(value['available'],'No missing private telemetry guessed0');total+=value['private_bytes']
                    if native is not None and item['pid']!=native['pid']:shims+=value['private_bytes']
                require(total<=c['worker_private_forecast_bytes']+192_000_000 and (native is None or shims<=64_000_000),
                    'Same whole-family and exact all-shim domain')
                time.sleep(.1)
            actual=child.wait();until=time.monotonic()+10
            while any(H.alive(v) for v in known.values()) and time.monotonic()<until:watch.check();time.sleep(.05)
            require(actual==0 and native is not None and not any(H.alive(v) for v in known.values()),'Actual wait0 and all captured identities dead')
            closure=read_json(folder/'guard_closure.json')
            require(closure['pid']==native['pid'] and closure['actual_exit_code']==0 and closure['monitor_clean']
                and closure['monitor_joined'] and closure['samples']>=3,'Actual clean joined worker guard0')
            result=read_json(folder/'result.json');require(result['passed'] is True and result['producer']==native
                and result['source']==record(__file__),'Exact trained fixture actual result')
            write_json(folder/'actual_process0.json',{'passed':True,'actual_Popen_wait_code':0,'all_captured_identities_dead':True,
                'captured_identities':list(known.values()),'native':native,'report':record(folder/'result.json'),
                'worker_guard':record(folder/'guard_closure.json'),'source':record(__file__),'configuration':record(config),
                'wall_seconds':time.monotonic()-began});code=0
    finally:
        from asr18_owned_family_observation_v4 import terminate_known_family,wait_captured_dead
        if (folder/'boot.json').exists():
            boot=read_json(folder/'boot.json');item=boot['producer']
            require(boot['source']==record(__file__) and boot['configuration']==record(config)
                and (item['pid']==launcher['pid'] or item['ppid']==launcher['pid']),'Exact early identity on failure')
            known[(item['pid'],item['create_time'])]=item
        if child is not None:
            if code:terminate_known_family(child,launcher,known)
            wait_captured_dead(known,timeout=10)
            if child.poll() is None:child.terminate()
            child.wait(timeout=5)
        require(not any(H.alive(v) for v in known.values()),'No surviving owned GPU before lease release')
        write_json(LEASE,prior);watch.close(code)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--ROOT-review',required=True)
    parser.add_argument('--folder',required=True);parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    config=checked(args.config,args.ROOT_review);folder=Path(args.folder).resolve()
    worker(config,args.config,folder) if args.worker else parent(config,args.config,args.ROOT_review,folder)
