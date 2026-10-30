"""Tiny actual launcher dispatch with stdlib AST parser and synthetic CUDA only."""
import argparse
import ast
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import tempfile
import types

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/parakeet_turkish';sys.path.insert(0,str(OUT))
checks=[]
def check(name,value):
    if not value:raise AssertionError(name)
    checks.append(name)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tree(name):return ast.parse((OUT/name).read_text(encoding='utf-8'))
def function(module,name):return next(node for node in module.body if isinstance(node,ast.FunctionDef) and node.name==name)

oldtree=tree('train_lora_batched_data_v10.py');newtree=tree('train_lora_batched_data_v11.py')
def cleanrun(tree):
    return ast.Module(body=[n for n in function(tree,'run').body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='source_names' for t in n.targets))],type_ignores=[])
check('run_math_sampler_optimizer_eval_save_AST_identical',ast.dump(cleanrun(oldtree))==ast.dump(cleanrun(newtree)))
for name in ('save_training_state','parser','validate_arguments','BalancedBucketSampler','native_loss_identity','native_loss_function'):
    selector=lambda tree:next(n for n in tree.body if getattr(n,'name',None)==name)
    check(name+'_AST_identical',ast.dump(selector(oldtree))==ast.dump(selector(newtree)))
# Execute only exact native exporter/parser/validation definitions, not imports.
native_tree=tree('native_tdt_checkpointed_v3.py')
native=types.ModuleType('native_tdt_checkpointed_v3');native.__file__=str(OUT/'native_tdt_checkpointed_v3.py')
for name in ('SOURCE_FILES','DEFAULT_MAX_JOINT_BYTES','CACHE_POLICY'):
    assignment=next(node for node in native_tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets))
    setattr(native,name,ast.literal_eval(assignment.value))
exec(compile(ast.Module(body=[function(native_tree,'configuration_identity')],type_ignores=[]),native.__file__,'exec'),native.__dict__)
native.cache_cleanup_evidence=lambda:{'kind':'CPU_stub_readonly_cleanup','adapter_sha256':sha(native.__file__)}
sys.modules[native.__name__]=native
worker=types.ModuleType('train_lora_batched_data_v11');worker.__file__=str(OUT/'train_lora_batched_data_v11.py')
worker.__dict__.update(argparse=argparse,Path=Path,math=math,hashlib=hashlib,json=json,copy=copy,ROOT=ROOT,DATA=ROOT/'work/data',OUTPUT=OUT,file_sha=sha)
names=('parser','validate_arguments','native_loss_identity','canonical_identity_sha','exact_resume_identity_matches')
exec(compile(ast.Module(body=[function(newtree,name) for name in names],type_ignores=[]),worker.__file__,'exec'),worker.__dict__)
sys.modules[worker.__name__]=worker
benchmark=types.ModuleType('benchmark');benchmark.write_json=lambda *_:None;sys.modules['benchmark']=benchmark
received=[];saved=[];cuda_calls=[]
def original_serializer(path,*values,**keywords):
    saved.append((values,keywords));Path(path).write_bytes(b'exact_original_serializer_passthrough_fixture')
worker.save_training_state=original_serializer
def marker(args):received.append(args)
torch=types.ModuleType('torch')
cuda=types.SimpleNamespace(synchronize=lambda *args:cuda_calls.append('sync'),memory_allocated=lambda *args:2_800_000_000,memory_reserved=lambda *args:5_000_000_000,max_memory_allocated=lambda *args:3_000_000_000,empty_cache=lambda:cuda_calls.append('empty'),get_device_properties=lambda *_:types.SimpleNamespace(total_memory=16_000_000_000,name='SYNTHETIC CPU fixture'),set_per_process_memory_fraction=lambda f,device=0:cuda_calls.append(('cap',f)),is_initialized=lambda:False)
torch.cuda=cuda;sys.modules['torch']=torch
import launch_with_io_retry_v9 as io
import launch_with_cache_reclaim_v12 as cache
import launch_with_memory_guard_v15 as memory
import launch_native_training_v5 as outer
for cap in (7.,8.,9.):check('cap_allowed_'+str(cap),memory.validate_capacity(16e9,cap)['cap_bytes']==cap*1e9)
for cap in (6.,10.,float('nan')):
    try:memory.validate_capacity(16e9,cap)
    except ValueError:checks.append('cap_rejected_'+str(cap))
    else:raise AssertionError('Undeclared allocator cap accepted')

with tempfile.TemporaryDirectory(dir=ROOT/'work/research',prefix='native_split_dispatch_stdlib_') as temporary:
    temporary=Path(temporary)
    ready=temporary/'native_ready.json'
    run_id='CPU_native_split_dispatch'
    ready_value={'kind':'native_tdt_training_backend_v3','status':'ready','run_id':run_id,'tdt_implementation':'nemo','trainer_sha256':sha(worker.__file__),'adapter_sha256':sha(native.__file__),'preload':{'kind':'coherent_native_cuda12_9_preload_v1','module_sha256':sha(OUT/'preload_nemo_linker_v1.py'),'successful_launcher_sha256':sha(OUT/'launch_nemo_linker_v4.py'),'actual_binding_linker':{'version':[12,9]},'cuda_initialized':False}}
    ready.write_text(json.dumps(ready_value),encoding='utf-8')
    arguments=['--','--run-id',run_id,'--eval-every-updates','250','--batch-size','8','--accumulation','2','--tdt-implementation','nemo','--native-backend-provenance',str(ready),'--native-max-joint-bytes','1700000000']
    code='def run(args):\n    marker(args)\n'+'\n'*541+'    torch.cuda.synchronize()\n    save_checkpoint(args)\n'
    check('compiled_mock_scope_source_line544',next(n for n in ast.walk(ast.parse(code)) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.cuda.synchronize').lineno==544)
    scope={'torch':torch,'marker':marker,'save_checkpoint':lambda args:worker.save_training_state(temporary/'checkpoint.pt','fixture_positional',complete_optimizer_sampler_RNG=True)}
    exec(compile(code,worker.__file__,'exec'),scope);worker.run=scope['run']
    args=types.SimpleNamespace(provenance=temporary/'io.json',cache_provenance=temporary/'cache.json',memory_provenance=temporary/'memory.json',worker_identity=temporary/'identity.json',cap_gb=8.,threshold_gb=4.,max_events=128,retry_seconds=1.,validate_only=False,arguments=arguments)
    memory.main(args)
    ie=json.loads(args.provenance.read_text());ce=json.loads(args.cache_provenance.read_text());me=json.loads(args.memory_provenance.read_text())
    check('actual_nested_dispatch_reaches_real_trainer_parser_validation_and_mock_run',len(received)==1 and received[0].eval_every_updates==250 and received[0].batch_size==8 and received[0].accumulation==2 and received[0].native_max_joint_bytes==1700000000)
    check('IO8_completed_and_source_hash_bound',ie['status']=='complete' and ie['launcher_sha256']==sha(io.__file__))
    check('cache11_dynamic_run544_hook_called',ce['cache_reclamation']['eligible_update_calls']==1 and ce['cache_reclamation']['reclamations']==1)
    check('memory14_cap7_end4_bound',me['cap_bytes']==8e9 and me['cleanup_threshold_bytes']==4e9 and me['allocator_cap_applied'])
    check('original_serializer_arguments_and_bytes_preserved',saved==[(('fixture_positional',),{'complete_optimizer_sampler_RNG':True})] and (temporary/'checkpoint.pt').read_bytes()==b'exact_original_serializer_passthrough_fixture')
    check('all_phase_sampler_stopped_joined',me['all_phase_owned_memory_telemetry']['thread_join_completed'])
    check('native3_evidence_source_bound',me['native_cleanup_evidence']['adapter_sha256']==sha(native.__file__))
    # Use the actual closed862 source identity with a tiny synthetic state file.
    receipt=json.loads((OUT/'runs/adaptive_20261011_hf_return_window_01/runtime_identity_continuation_v1.json').read_text(encoding='utf-8'))
    parent=json.loads(Path(receipt['source_run_config']['path']).read_text(encoding='utf-8'))
    old=parent['config_identity'];fork=temporary/'fork_latest.pt';fork.write_bytes(b'CPU_full_state_identity_fixture')
    receipt_path=temporary/'runtime_identity.json';new_ready=temporary/'migration_ready.json'
    policy={'cap_gb':8,'end_cache_gb':4,'native_inactive_threshold_bytes':750000000}
    ready_value['declared_allocator_policy']=policy;new_ready.write_text(json.dumps(ready_value),encoding='utf-8')
    receipt['target_run_id']=run_id
    receipt['fork_latest']={'path':str(fork),'sha256':sha(fork),'step':862}
    receipt['native_backend_transition']['new_path']=str(new_ready)
    receipt_path.write_text(json.dumps(receipt),encoding='utf-8')
    current=copy.deepcopy(old)
    for oldname,item in receipt['approved_source_changes'].items():
        current['source_sha256'].pop(oldname);current['source_sha256'][item['new_filename']]=item['new_sha256']
    added=receipt['added_identity_validator'];current['source_sha256'][added['filename']]=added['sha256']
    current['input_sha256'].pop(str(Path(receipt['replaced_previous_migration']['path']).resolve()))
    current['input_sha256'].pop(str(Path(parent['native_backend_provenance']).resolve()))
    current['input_sha256'][str(new_ready.resolve())]=sha(new_ready)
    current['input_sha256'][str(receipt_path.resolve())]=sha(receipt_path)
    current['native_loss']['native_backend_provenance_sha256']=sha(new_ready)
    args=types.SimpleNamespace(run_id=run_id,tdt_implementation='nemo',resume_exact=fork,resume_runtime_migration=receipt_path,native_backend_provenance=new_ready,native_max_joint_bytes=1700000000)
    state={'step':862,'selected_step':0,'config_identity':old}
    check('actual_checker_accepts_only_identity_paths_and_routing_changes',worker.exact_resume_identity_matches(state,args,current))
    for key,value in [('head_lr',.0001),('replay_fraction',.25),('rank',16),('schedule','cosine')]:
        bad=copy.deepcopy(current);bad['training_arguments'][key]=value
        try:worker.exact_resume_identity_matches(state,args,bad)
        except ValueError:checks.append('exact_migration_rejects_changed_'+key)
        else:raise AssertionError('Scientific change silently accepted')
    bad=copy.deepcopy(current);bad['native_loss']['sigma']=.02
    try:worker.exact_resume_identity_matches(state,args,bad)
    except ValueError:checks.append('exact_migration_rejects_changed_native_sigma')
    else:raise AssertionError('Native loss changed')
    outer_ready=temporary/'unwritten_outer_ready.json'
    outer_worker=arguments[1:];outer_worker[outer_worker.index('--native-backend-provenance')+1]=str(outer_ready)
    oa=types.SimpleNamespace(native_provenance=outer_ready,memory_launcher=Path(memory.__file__),memory_launcher_sha256=sha(memory.__file__),trainer_sha256=sha(worker.__file__),memory_arguments=['--cap-gb','8','--threshold-gb','4','--',*outer_worker],validate_only=True)
    check('outer4_actual_validate_cap8_routes_trainer10_without_preload',outer.validate(oa)[4]==run_id)

report={'schema':1,'passed':True,'check_count':len(checks),'checks':checks,'actual_Torch_or_NeMo_imported':False,'GPU_execution':False,'source_sha256':{name:sha(OUT/name) for name in ('train_lora_batched_data_v11.py','launch_with_io_retry_v9.py','launch_with_cache_reclaim_v12.py','launch_with_memory_guard_v15.py','launch_native_training_v5.py','native_tdt_checkpointed_v3.py')},'scope':'Actual NEW launcher/real AST-extracted parser/native identity/exact migration checks; mocked Torch/CUDA and worker.run; original serializer passthrough plus static serializer AST equality. No actual tensor/optimizer restore or hardware efficacy claim.'}
target=OUT/'HF_862_chain_stdlib_dispatch_cpu_verification_v1.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'checks':len(checks),'actual_Torch_or_NeMo_imported':False}))
