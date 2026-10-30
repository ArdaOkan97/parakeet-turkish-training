"""Focused actual release AST/cache API check; all runtime objects are opaque mocks."""
from pathlib import Path
import ast
import copy
import hashlib
import importlib
import json
import random
import sys
import weakref
from types import ModuleType, SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(HERE))
worker=HERE/'train_lora_batched_tsc_bpe_row_schedule_memory_v3.py'
helper=importlib.import_module('tsc_bpe_row_schedule_memory_v3')
old=ast.parse((HERE/'train_lora_batched_tsc_bpe_row_schedule_memory_v2.py').read_text(encoding='utf-8'))
new=ast.parse(worker.read_text(encoding='utf-8'))
old_f={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)}
new_f={n.name:n for n in new.body if isinstance(n,ast.FunctionDef)}
run=new_f['run']; checks=[]
def check(ok,name):
    assert ok,name;checks.append(name)
start=next(i for i,n in enumerate(run.body) if ast.unparse(n)=='save(latest_path, start_step)')+1
end=next(i for i in range(start,len(run.body)) if ast.unparse(run.body[i])=='model.train()')
release=run.body[start:end]
check(any(ast.unparse(n)=='del expanded, context' for n in release),'only_proven_unused_admission_objects_deleted')
check(any('save(best_path, start_step' in ast.unparse(n) for n in run.body[:start])
      and any('vocabulary_audit.json' in ast.unparse(n) for n in run.body[:start])
      and any('equality = initial_source_equivalence' in ast.unparse(n) for n in run.body[:start]),
      'actual_source300_audit_and_both_initial_state_commits_before_release')
check(not any(isinstance(n,ast.Name) and n.id in ('expanded','context') and isinstance(n.ctx,ast.Load)
              for statement in run.body[end:] for n in ast.walk(statement)), 'no_post_release_context_or_expanded_reads')
normalized=copy.deepcopy(new)
for function in normalized.body:
    if isinstance(function,ast.FunctionDef) and function.name=='run':del function.body[start:end]
text=ast.unparse(normalized)
for a,b in sorted((('tsc_bpe_row_schedule_memory_v3.py','tsc_bpe_row_schedule_memory_v2.py'),
                  ('train_lora_batched_tsc_bpe_row_schedule_memory_v3.py','train_lora_batched_tsc_bpe_row_schedule_memory_v2.py'),
                  ('run_TSC_bpe_row_schedule_memory_v3.py','run_TSC_bpe_row_schedule_memory_v2.py')),key=lambda x:len(x[0]),reverse=True):text=text.replace(a,b)
nf={n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
for name in old_f:
    if name!='native_loss_identity':check(ast.dump(old_f[name],include_attributes=False)==ast.dump(nf[name],include_attributes=False),'unchanged_function_AST_'+name)

class Runtime(dict):pass
class Cache:
    def __init__(self,value):self.value=value
    def cache_info(self):return SimpleNamespace(currsize=int(self.value is not None))
    def cache_clear(self):self.value=None
history=[{'step':1}]; config={'phase':'OPAQUE_ACTUAL_CONFIG_MOCK'}
sampler=object(); initial_gate={'gate':'retained'}; language_audio={'tr':1.}
runtime=Runtime(history=history,unused_source_sampler_template=Runtime(),config_template=Runtime())
runtime_ref=weakref.ref(runtime)
expanded={'inherited_runtime':runtime,'acquisition':'OPAQUE_SAME_OPTIMIZER','audit':{}}
del runtime
context=Runtime(unused_admission_template=Runtime());context_ref=weakref.ref(context)
new_cache=Cache(context);original_cache=Cache(Runtime())
original=importlib.import_module('tsc_bpe_acquisition_transition_v1')
old_new_cache,old_original_cache=helper._context,original._context
old_memory_module=sys.modules.get('launch_with_memory_guard_TSC_resume_v3')
fake=ModuleType('launch_with_memory_guard_TSC_resume_v3')
samples=[]
fake.safe_sample=lambda pid,phase:(samples.append(phase) or {'phase':phase,'actual_host_measurement':False,'explicit_mock':True})
saved=[]
acquisition=SimpleNamespace(optimizer_updates_completed=0,source_acquisition_updates=500,cumulative_step=7000)
env={'expanded':expanded,'context':context,'history':history,'config':config,'sampler':sampler,
     'initial_gate':initial_gate,'language_audio':language_audio,'acquisition':acquisition,
     'args':SimpleNamespace(run_id='inactive_release_fixture_no_run'), 'PHASE':helper.PHASE,
     'run_dir':Path('DISPOSABLE_NO_WRITES'),'start_step':1,'file_sha':lambda p:'OPAQUE_PROOF_SHA',
     'proof_path':lambda args:Path('NO_REAL_PROOF'),'write_json':lambda p,v:saved.append(v),
     'release_admission_caches':helper.release_admission_caches}
before_rng=random.getstate(); before_active=copy.deepcopy(acquisition.__dict__)
try:
    helper._context=new_cache;original._context=original_cache
    sys.modules[fake.__name__]=fake
    del context,expanded
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(release),type_ignores=[])),str(worker),'exec'),env)
    check(runtime_ref() is None and context_ref() is None,'actual_release_AST_drops_only_unused_runtime_and_cached_context_aliases')
    check(env['history'] is history and env['config'] is config and env['sampler'] is sampler
          and env['initial_gate'] is initial_gate and env['language_audio'] is language_audio,
          'authoritative_history_config_sampler_gate_language_clocks_keep_same_objects')
    check(random.getstate()==before_rng and acquisition.__dict__==before_active,'actual_release_AST_does_not_draw_rng_or_change_optimizer_frontier')
    check(samples==['before_startup_admission_release','after_startup_admission_release'],'host_only_samples_bracket_release')
    check(saved[0]['expanded_inherited_runtime_alias_removed'] is True
          and saved[0]['cache_release']['scheduled_admission_entries_after']==0
          and saved[0]['cache_release']['original_acquisition_admission_entries_after']==0,
          'actual_cache_clear_API_reports_local_admissions_released')
finally:
    helper._context=old_new_cache;original._context=old_original_cache
    if old_memory_module is None:sys.modules.pop(fake.__name__,None)
    else:sys.modules[fake.__name__]=old_memory_module
check(not any(name in sys.modules for name in ('torch','transformers','nemo')),'no_ML_import_or_GPU_execution')
out=HERE/'TSC_BPE_row_schedule_startup_alias_release_CPU_verification_v3.json'
value={'kind':'inactive_scheduled_row_startup_alias_release_CPU_verification_v3','passed':True,'checks':checks,'assertions':len(checks),
 'worker_sha256':hashlib.sha256(worker.read_bytes()).hexdigest(),'helper_sha256':hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest(),
 'actual_worker_release_AST_and_helper_cache_clear_executed':True,'runtime_state_PIDs_and_host_samples_explicitly_mocked':True,
 'actual_source_model_checkpoint_or_live_data_read':False,'actual_memory_relief_or_hardware_measured':False,
 'direct_original_source_deletion_already_in_predecessor':True,'activation_ready_claimed':False}
with out.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({'passed':True,'assertions':len(checks),'report':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
