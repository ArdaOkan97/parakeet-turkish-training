"""CPU AST/cadence/source-dispatch guards; no Torch or GPU import."""
from pathlib import Path
import ast,copy,json,sys
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(D))
from cuda_cache_reclaim_v1 import file_sha
old=ast.parse((D/'train_lora_batched_data_v3.py').read_text(encoding='utf-8'))
new=ast.parse((D/'train_lora_batched_data_v4.py').read_text(encoding='utf-8'))
checks=[]
def check(n,x):
    if not x:raise AssertionError(n)
    checks.append(n)
def loop(tree):
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    return copy.deepcopy(next(n for n in run.body if isinstance(n,ast.For) and getattr(n.target,'id',None)=='step'))
prior,changed=loop(old),loop(new)
new_if=next(n for n in ast.walk(changed) if isinstance(n,ast.If) and 'args.eval_every_updates' in ast.unparse(n.test))
old_if=next(n for n in ast.walk(prior) if isinstance(n,ast.If) and 'sampler.turkish_epoch' in ast.unparse(n.test))
new_if.test=copy.deepcopy(old_if.test)
check('entire_update_loop_identical_except_explicit_DEV_condition',ast.dump(prior,include_attributes=False)==ast.dump(changed,include_attributes=False))
sync=[n for n in ast.walk(new) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.cuda.synchronize']
check('same_exact_optimizer_end_run539',len(sync)==1 and sync[0].lineno==539)
condition=next(n.test for n in ast.walk(loop(new)) if isinstance(n,ast.If) and 'args.eval_every_updates' in ast.unparse(n.test))
code=compile(ast.Expression(condition),'<cadencefixture>','eval')
from types import SimpleNamespace
def due(step,interval=250,epoch=0.,next_eval=1.,cap=2000):return eval(code,{'args':SimpleNamespace(eval_every_updates=interval),'step':step,'sampler':SimpleNamespace(turkish_epoch=epoch),'next_eval':next_eval,'configured_steps':cap})
for s in (250,500,750,1000,1250):check('every250-'+str(s),due(s))
for s in (1,249,251,499):check('no_unrequested_update_eval-'+str(s),not due(s))
check('earlier_epoch_threshold_retained',due(117,next_eval=.01,epoch=.012))
check('final_configured_step_evaluates',due(137,cap=137))
check('None_preserves_prior_epoch_schedule',not due(250,interval=None) and due(250,interval=None,epoch=1))
validate=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='validate_arguments')
check('positive_interval_guard_present','args.eval_every_updates <= 0' in ast.unparse(validate))
check('cadence_part_of_exact_resume_config_identity','"eval_every_updates"' in (D/'train_lora_batched_data_v4.py').read_text())
new_cache=(D/'launch_with_cache_reclaim_v4.py').read_text();old_cache=(D/'launch_with_cache_reclaim_v3.py').read_text()
check('cache_dispatch_only_expected_worker_pin_kind_changes',new_cache.replace('train_lora_batched_data_v4.py','train_lora_batched_data_v3.py').replace(file_sha(D/'train_lora_batched_data_v4.py'),file_sha(D/'train_lora_batched_data_v3.py')).replace('cache_reclamation_training_launcher_v4','cache_reclamation_training_launcher_v3')==old_cache)
new_wrap=(D/'launch_with_memory_guard_v7.py').read_text();old_wrap=(D/'launch_with_memory_guard_v6.py').read_text()
check('memory_dispatch_only_expected_worker_cache_pin_kind_changes',new_wrap.replace('train_lora_batched_data_v4.py','train_lora_batched_data_v3.py').replace('launch_with_cache_reclaim_v4','launch_with_cache_reclaim_v3').replace(file_sha(D/'train_lora_batched_data_v4.py'),file_sha(D/'train_lora_batched_data_v3.py')).replace(file_sha(D/'launch_with_cache_reclaim_v4.py'),file_sha(D/'launch_with_cache_reclaim_v3.py')).replace('memory_guard_training_launcher_v7','memory_guard_training_launcher_v6')==old_wrap)
check('Torch_not_imported','torch' not in sys.modules)
target=D/'fast_cadence_v4_cpu_verification.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(dict(schema=1,passed=True,CPU_only=True,check_count=len(checks),checks=checks,trainer_sha256=file_sha(D/'train_lora_batched_data_v4.py'),cache_launcher_sha256=file_sha(D/'launch_with_cache_reclaim_v4.py'),memory_launcher_sha256=file_sha(D/'launch_with_memory_guard_v7.py'),reused_wrapper6_guards_path=str((D/'memory_guard_v6_final_cpu_verification.json').resolve()),reused_wrapper6_guards_sha256=file_sha(D/'memory_guard_v6_final_cpu_verification.json')),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,checks=len(checks))))
