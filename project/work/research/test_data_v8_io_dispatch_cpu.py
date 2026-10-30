"""Actual nested dispatcher, real trainer parser and original CPU serializer.

The only training substitute is a tiny CPU run function; no GPU/model training.
All launchers and the trainer parser/checkpoint serializer are real new sources.
"""
from pathlib import Path
import sys,json,tempfile,types,ast
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/parakeet_turkish';sys.path.insert(0,str(D))
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
import train_lora_batched_data_v8 as worker
import launch_with_io_retry_v6 as io
import launch_with_memory_guard_v12 as memory
import launch_with_cache_reclaim_v9 as cache
import native_tdt_checkpointed_v2 as native
from cuda_cache_reclaim_v1 import file_sha
checks=[];received=[];calls=[]
for cap in (6.,7.):
    assert memory.validate_capacity(16_000_000_000,cap)['cap_bytes']==int(cap*1e9)
    checks.append('accepted_allocator_cap_'+str(cap))
for invalid in (5.,8.,9.,float('nan')):
    try: memory.validate_capacity(16_000_000_000,invalid)
    except ValueError: checks.append('unreviewed_allocator_cap_rejected_'+str(invalid))
    else: raise AssertionError('Unreviewed cap accepted')
def check(name,value):
    if not value:raise AssertionError(name)
    checks.append(name)
model=torch.nn.Module();model.encoder=torch.nn.Module();model.encoder.layers=torch.nn.ModuleList([])
model.register_parameter('fixture',torch.nn.Parameter(torch.ones(2)))
stateful=types.SimpleNamespace(state_dict=lambda:{'CPUfixture':True})
original_save=worker.save_training_state
fake_cuda=types.SimpleNamespace(synchronize=lambda:calls.append('sync'),memory_allocated=lambda:2e9,memory_reserved=lambda:5e9,
    max_memory_allocated=lambda:3e9,empty_cache=lambda:calls.append('empty'),get_rng_state_all=lambda:[],
    get_device_properties=lambda *_:types.SimpleNamespace(total_memory=16_000_000_000,name='CPUfixture'),
    set_per_process_memory_fraction=lambda fraction,device=0:calls.append(('cap',fraction)),is_initialized=lambda:False)
before_rng=torch.get_rng_state().clone()
with tempfile.TemporaryDirectory(dir=ROOT/'work/research',prefix='actual_v4_dispatch_CPU_') as temporary:
    temporary=Path(temporary)
    def marker(args):
        received.append(args)
        check('actual_trainer_parser_eval250',args.eval_every_updates==250)
        check('actual_trainer_parser_B8x2',args.batch_size==8 and args.accumulation==2)
        check('actual_trainer_parser_data_v3',args.tr_manifest.name=='tr_public_v3_train.jsonl')
        check('actual_trainer_parser_nemo',args.tdt_implementation=='nemo')
        check('actual_trainer_parser_native_options',args.native_max_joint_bytes==1700000000 and args.native_backend_provenance.name=='CPU_native_fixture.json')
    def checkpoint(args):
        worker.save_training_state(temporary/(args.run_id+'.pt'),model,stateful,stateful,run_id=args.run_id,step=1,
            config_identity={'fixture':True},history=[{'step':1}],best_score=.5,selected_step=1,patience=0,retention_failures=0,
            next_eval_epoch=.5,train_wall_seconds=1.,audio_seconds=2.,language_audio_seconds={'tr':2.},evaluation_wall_seconds=0.,initial_gate={'fixture':True})
    code='def run(args):\n    marker(args)\n'+'\n'*536+'    torch.cuda.synchronize()\n    checkpoint(args)\n'
    tree=ast.parse(code);call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.cuda.synchronize')
    check('stub_exact_original_sourceframe539',call.lineno==539)
    scope=dict(torch=torch,marker=marker,checkpoint=checkpoint)
    exec(compile(code,str(Path(worker.__file__).resolve()),'exec'),scope)
    with patch.object(torch,'cuda',fake_cuda),patch.object(worker,'run',scope['run']),patch.object(worker,'native_loss_identity',return_value={'CPU_fixture_only':True}):
        args=types.SimpleNamespace(worker='train_lora_batched_data_v8.py',provenance=temporary/'direct_io.json',
            expected_worker_sha=file_sha(worker.__file__),retry_seconds=1.,arguments=['--','--run-id','direct_cpu','--eval-every-updates','250','--batch-size','8','--accumulation','2','--tdt-implementation','nemo','--native-backend-provenance','CPU_native_fixture.json','--native-max-joint-bytes','1700000000'])
        io.main(args)
        direct=json.loads(args.provenance.read_text())
        check('real_IO_v2_dispatch_complete',direct['status']=='complete' and direct['kind']=='atomic_io_retry_v6')
        saved=torch.load(temporary/'direct_cpu.pt',map_location='cpu',weights_only=True)
        check('unchanged_serializer_saved_complete_fixture_state',saved['step']==1 and saved['sampler']=={'CPUfixture':True} and torch.equal(saved['weights']['fixture'],torch.ones(2)))
        check('unchanged_serializer_schema_RNG_fields',saved['schema']==2 and 'torch_rng' in saved and 'python_rng' in saved and saved['cuda_rng']==[])
        check('atomic_private_temps_committed',not any('.atomic-v1-' in p.name for p in temporary.iterdir()))
        # Restore the original before a second real import/adapter invocation.
        worker.save_training_state=original_save
        args=types.SimpleNamespace(provenance=temporary/'nested_io.json',cache_provenance=temporary/'nested_cache.json',
            memory_provenance=temporary/'nested_memory.json',worker_identity=temporary/'nested_identity.json',
            cap_gb=6.,threshold_gb=4.,max_events=128,retry_seconds=1.,validate_only=False,
            arguments=['--','--run-id','nested_cpu','--eval-every-updates','250','--batch-size','8','--accumulation','2','--tdt-implementation','nemo','--native-backend-provenance','CPU_native_fixture.json','--native-max-joint-bytes','1700000000'])
        memory.main(args)
        nested_io=json.loads(args.provenance.read_text());nested_cache=json.loads(args.cache_provenance.read_text());nested_memory=json.loads(args.memory_provenance.read_text())
        check('actual_whole_nested_dispatch_reached_worker_run',len(received)==2 and nested_io['status']=='complete')
        check('actual_nested_sources_bound',nested_io['launcher_sha256']==file_sha(io.__file__) and nested_cache['launcher_sha256']==file_sha(cache.__file__) and nested_memory['memory_launcher_sha256']==file_sha(memory.__file__))
        check('actual_nested_cache_scope539_observed',nested_cache['cache_reclamation']['eligible_update_calls']==1 and nested_cache['cache_reclamation']['reclamations']==1)
        check('actual_nested_memory_policy_cap6cache4',nested_memory['cap_bytes']==6e9 and nested_memory['cleanup_threshold_bytes']==4e9)
        check('actual_native_cleanup_readonly_callback_bound',nested_memory['native_cleanup_evidence']['adapter_sha256']==file_sha(native.__file__) and nested_memory['native_cleanup_adapter_sha256']==file_sha(native.__file__))
        check('actual_nested_stdlib_sampler_joined',nested_memory['all_phase_owned_memory_telemetry']['thread_join_completed'])
        check('exact_original_serializer_adapter_still_preserved',worker.save_training_state.__wrapped__ is original_save)
    worker.save_training_state=original_save
check('CPU_Torch_RNG_unchanged',torch.equal(torch.get_rng_state(),before_rng))
check('CUDA_not_initialized',not torch.cuda.is_initialized())
target=D/'data_v8_actual_io_dispatch_cpu_verification.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(dict(schema=1,passed=True,CPU_only=True,check_count=len(checks),checks=checks,
    io_launcher_sha256=file_sha(io.__file__),cache_launcher_sha256=file_sha(cache.__file__),memory_launcher_sha256=file_sha(memory.__file__),trainer_sha256=file_sha(worker.__file__),
    scope='Actual NEWnesteddispatch/trainerparser/originalCPUserializer/cachecall539; tinyCPUrun/nativebackendidentity substitutes only; noCUDAinit, NeMoenvironment or GPUloss proof'),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,checks=len(checks))))
