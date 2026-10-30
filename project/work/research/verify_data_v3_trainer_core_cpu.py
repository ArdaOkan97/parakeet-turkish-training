"""AST equality of the entire training-update loop and CPU real-frame hook."""
from pathlib import Path
import sys
import ast
import json

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'outputs/parakeet_turkish'))
from cuda_cache_reclaim_v1 import CudaCacheReclaimer,file_sha
directory=ROOT/'outputs/parakeet_turkish'
old=directory/'train_lora_batched.py';new=directory/'train_lora_batched_data_v3.py'
checks=[]
def check(name,value):
    if not value:raise AssertionError(name)
    checks.append(name)
def update_loop(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    return next(n for n in run.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step')
check('entire-training-update-loop-AST-identical',ast.dump(update_loop(old),include_attributes=False)==ast.dump(update_loop(new),include_attributes=False))
class CUDA:
    synchronize_calls=0;empty_calls=0
    def synchronize(self):self.synchronize_calls+=1;return 123
    def memory_allocated(self):return 2_000_000_000
    def memory_reserved(self):return 5_000_000_000
    def max_memory_allocated(self):return 3_000_000_000
    def empty_cache(self):self.empty_calls+=1
cuda=CUDA();hook=CudaCacheReclaimer(cuda,new,threshold_gb=4,expected_sha=file_sha(new))
namespace={'cuda':cuda}
code='def run():\n'+'\n'*537+'    return cuda.synchronize()\n'
exec(compile(code,str(new.resolve()),'exec'),namespace)
hook.install()
try:
    check('new-worker-exact539-frame-hook',namespace['run']()==123 and hook.eligible_calls==1 and hook.reclamations==1 and cuda.empty_calls==1)
    check('other-caller-not-reclaimed',cuda.synchronize()==123 and hook.eligible_calls==1 and hook.ignored_calls==1 and cuda.empty_calls==1)
finally:hook.restore()
check('hook-original-restored',cuda.synchronize()==123 and hook.eligible_calls==1)
target=directory/'public_data_v3_trainer_core_cpu_verification.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(dict(schema=1,passed=True,CPU_only=True,check_count=len(checks),checks=checks,
    original_trainer_sha256=file_sha(old),new_trainer_sha256=file_sha(new),
    scope='Source/real-frame CPU fixture proof, not actual new-data GPU execution or numerical-equivalence waiver'),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,checks=len(checks))))
