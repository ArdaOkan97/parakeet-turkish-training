"""Small repeatable stdlib fixture. All ML/CUDA/native/closed-state inputs are mocks."""
from pathlib import Path
import argparse,ast,builtins,hashlib,json,math,sys,tempfile,time,types
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE/'outputs/parakeet_turkish'))
import run_TSC_headlr_fork_v1 as c

def main(output):
    checks=[]
    def check(ok,label):
        if not ok:raise AssertionError(label)
        checks.append(label)
    def reject(fn,label):
        try:fn()
        except (ValueError,KeyError):checks.append(label);return
        raise AssertionError('Accepted '+label)
    pins=c.source_pins();check(len(pins)>=33,'Actual shared/source hashes match')
    check(c.caller_line(c.HERE/c.WORKER)==581,'Actual worker single reviewed optimizer sync581')
    import tsc_optimization_transition_v1 as transition
    check(c.canonical([{'test':'ş','step':1500}])==transition.canonical([{'test':'ş','step':1500}]),'Compact history canonical hash matches producer')
    tree=ast.parse((c.HERE/c.WORKER).read_text(encoding='utf-8'))
    funcs=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('parser','validate_arguments')]
    scope={'argparse':argparse,'Path':Path,'DATA':BASE/'work/data','OUTPUT':c.HERE,'math':math,
        'native_loss_identity':lambda args:None,'closed_source_identity':lambda args:None}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(c.HERE/c.WORKER),'exec'),scope)
    with tempfile.TemporaryDirectory(prefix='TSC_headlr_fixture_',dir=BASE/'work') as temporary:
        folder=Path(temporary);asset=folder/'opaque_closed_input.json';asset.write_text('{}')
        rid='mock_TSC_headlr_new';paths={key:str(folder/(key+'.json')) for key in ('native','identity','memory','cache','io','stop','log')}
        argv=['--run-id',rid,'--head-lr','1e-05','--batch-size','8','--accumulation','2','--steps','44958',
            '--eval-every-updates','250','--replay-seed','20261012','--resume-optimization',str(asset),
            '--optimization-source-receipt',str(asset),'--native-backend-provenance',paths['native']]
        for flag in ('--tsc-input-proof','--tsc-training-approval','--source-closure','--tr-manifest',
            '--replay-manifest','--tr-frame-metadata','--tr-frame-metadata-proof','--data-v3-proof'):
            argv.extend((flag,str(asset)))
        args=scope['validate_arguments'](scope['parser']().parse_args(argv))
        check(args.resume_optimization==asset and args.resume_exact is None,'Actual parser admits initial whole-state optimization mode')
        check(args.head_lr==1e-5 and args.lora_lr==3e-4 and args.replay_kl==10,'Actual parser preserves only head LR change')
        check(args.steps==44958 and args.eval_every_updates==250 and args.batch_size*args.accumulation==16,'Actual parser preserves44958/250/effective16')
        bad=dict(vars(args));bad['head_lr']=float('nan');reject(lambda:scope['validate_arguments'](types.SimpleNamespace(**bad)),'Nonfinite scientific option rejected')
        exactargv=c.remove_option(argv,'--resume-optimization')+['--resume-exact',str(asset)]
        exactargs=scope['validate_arguments'](scope['parser']().parse_args(exactargv))
        check(exactargs.resume_exact==asset and exactargs.resume_optimization is None,'Actual parser admits separate same-family exact mode')
        plan={'run_id':rid,'source_sha256':pins,'worker_arguments':argv,'worker_arguments_sha256':c.domain.guards.argument_sha(argv),
            'paths':paths,'source_closure':c.record(asset),'source_checkpoint':c.record(asset)}
        c.validate_plan(plan);checks.append('Actual dispatcher plan validation accepts fresh synthetic assets')
        badplan={**plan,'worker_arguments':c.set_option(argv,'--head-lr','5e-05')}
        reject(lambda:c.validate_plan(badplan),'Dispatcher rejects undeclared head LR')
        badplan={**plan,'worker_arguments':argv+['--resume-exact',str(asset)]}
        reject(lambda:c.validate_plan(badplan),'Dispatcher rejects mixed restore modes')
        events=[];cuda=types.SimpleNamespace()
        cuda.synchronize=lambda:events.append('synchronize') or 42
        reserved=[5_000_000_000]
        cuda.memory_allocated=lambda:2_800_000_000;cuda.memory_reserved=lambda:reserved[0];cuda.max_memory_allocated=lambda:6_000_000_000
        def clear():events.append('empty_cache');reserved[0]=3_000_000_000
        cuda.empty_cache=clear;cuda.get_device_properties=lambda index:types.SimpleNamespace(total_memory=16_000_000_000)
        cuda.set_per_process_memory_fraction=lambda fraction,device:events.append(('cap',fraction,device))
        fake_torch=types.SimpleNamespace(cuda=cuda)
        worker=types.ModuleType(Path(c.WORKER).stem);worker.parser=scope['parser'];worker.validate_arguments=scope['validate_arguments']
        original_write=lambda path,value:Path(path).write_text(json.dumps(value))
        worker.write_json=original_write
        def serializer(path,value):events.append('serializer_once');Path(path).write_bytes(value)
        worker.save_training_state=serializer
        # Real frame/filename/line resolves through the inherited synchronize wrapper.
        run_scope={'torch':fake_torch,'worker':worker,'events':events,'fixture':folder,'Path':Path}
        text='\n'*579+'def run(args):\n    value=torch.cuda.synchronize()\n    events.append(("dispatch",args.head_lr,args.steps,value))\n    worker.save_training_state(fixture/"state.pt",b"exact-original-serialized-bytes")\n    worker.write_json(fixture/"worker.json",{"step":1501})\n'
        exec(compile(text,str(c.HERE/c.WORKER),'exec'),run_scope);worker.run=run_scope['run']
        benchmark=types.ModuleType('benchmark');benchmark.write_json=original_write
        preload=types.ModuleType('preload_nemo_linker_v1')
        preload.preload=lambda:events.append('preload_before_torch') or {'kind':'coherent_native_cuda12_9_preload_v1','actual_binding_linker':{'version':[12,9]},'cuda_initialized':False}
        sampler=lambda *a,**k:types.SimpleNamespace(start=lambda:None,stop_and_join=lambda:None,evidence=lambda:{'thread_join_completed':True})
        sample=lambda pid,phase:{'observed_monotonic':time.monotonic(),'observed_at_utc':c.now(),'phase':phase,'system_memory':{},'worker_private_memory':{},'calling_process_commit_memory':{}}
        originals={name:sys.modules.get(name) for name in ('benchmark','preload_nemo_linker_v1',worker.__name__)}
        original_import=builtins.__import__;original_creation=c.shared_memory.creation_identity;original_sample=c.shared_memory.safe_sample
        original_sampler=c.shared_memory.LiveTelemetrySampler;original_telemetry=c.shared_memory.UpdateTelemetry
        def importer(name,*values,**kwargs):
            if name=='torch':events.append('mock_torch_import');return fake_torch
            return original_import(name,*values,**kwargs)
        try:
            sys.modules.update({'benchmark':benchmark,'preload_nemo_linker_v1':preload,worker.__name__:worker})
            builtins.__import__=importer;c.shared_memory.creation_identity=lambda pid:{'available':True,'pid':pid}
            c.shared_memory.safe_sample=sample;c.shared_memory.LiveTelemetrySampler=sampler
            c.shared_memory.UpdateTelemetry=lambda pid,sink:original_telemetry(pid,sink=sink,sample=sample)
            c.run_worker(plan)
        finally:
            builtins.__import__=original_import;c.shared_memory.creation_identity=original_creation;c.shared_memory.safe_sample=original_sample
            c.shared_memory.LiveTelemetrySampler=original_sampler;c.shared_memory.UpdateTelemetry=original_telemetry
            for name,value in originals.items():
                if value is None:sys.modules.pop(name,None)
                else:sys.modules[name]=value
        check(events.index('preload_before_torch')<events.index('mock_torch_import'),'Actual dispatch preloads before mocked Torch import')
        check(('cap',.5,0) in events,'Actual dispatcher sets8GB process allocator cap')
        check(('dispatch',1e-5,44958,42) in events,'Actual nested parser/validate/run reached; original synchronize return retained')
        check(events.count('empty_cache')==1,'Actual source-bound run581 cache hook fires once')
        check(events.count('serializer_once')==1 and (folder/'state.pt').read_bytes()==b'exact-original-serialized-bytes','Atomic checkpoint hook invokes original serializer once with exact bytes')
        check(worker.write_json is original_write and worker.save_training_state is serializer and cuda.synchronize() ==42,'Dispatcher restores original IO/serializer/CUDA methods')
        cache=c.read(paths['cache']);memory=c.read(paths['memory'])
        check(cache['cache_reclamation']['eligible_update_calls']==1 and cache['cache_reclamation']['caller_line']==581,'Actual suffix cache evidence uses581')
        check(memory['owned_update_memory_telemetry']['eligible_update_calls']==1 and '581' in memory['owned_update_memory_telemetry']['callback_scope'],'Actual telemetry counts suffix and declares actual caller')
        check(c.read(Path(paths['native']).with_suffix('.completion.json'))['status']=='complete','Actual mocked nested completion published')
        check(c.read(paths['io'])['sources_unchanged'] and c.read(paths['memory'])['sources_unchanged'],'All frozen shared sources match after mocked dispatch')
    check('torch' not in sys.modules and not any(name.startswith('nemo') for name in sys.modules),'No real Torch/NeMo imports, GPU/model/data/state deserialization')
    p=Path(output).resolve();c.require(not p.exists(),'Fixture output must be fresh')
    c.write(p,{'schema':1,'kind':'TSC_headlr_onefile_mocked_dispatch_CPU_verification_v1','passed':True,'checks':len(checks),
        'check_names':checks,'controller':c.record(c.__file__),'worker':c.record(c.HERE/c.WORKER),'transition':c.record(transition.__file__),
        'source_sha256':pins,'mocks':['Torch/CUDA','coherent DLL preload','Win32 owned telemetry','ML worker.run except actual source-backed caller frame',
            'external closed source admission','opaque checkpoint serializer'],'actual_GPU_or_full_state_restore_verified':False})
    print(json.dumps({'passed':True,'checks':len(checks),'output':str(p)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
