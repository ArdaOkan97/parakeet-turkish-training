"""Check the actual name-resolution repair and preserved training AST before relaunch."""
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import json
import symtable
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/parakeet_turkish'
old=ROOT/'work/closed-sources/uniform_full_PT02_dispatch_v1/train_lora_batched_tsc_uniform_full_v1.py'
new=OUT/'train_lora_batched_tsc_uniform_full_v1.py'
raw=new.read_text(encoding='utf-8'); tree=ast.parse(raw); prior=ast.parse(old.read_text(encoding='utf-8'))
scope=next(c for c in symtable.symtable(raw,str(new),'exec').get_children() if c.get_name()=='run')
assert scope.lookup('file_record').is_global() and not scope.lookup('file_record').is_local()
assert scope.lookup('record').is_local()
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='record' for n in ast.walk(tree))
class UndoAlias(ast.NodeTransformer):
    def visit_Name(self,node):
        if node.id=='file_record':node.id='record'
        return node
    def visit_alias(self,node):
        if node.name=='record' and node.asname=='file_record':node.asname=None
        return node
assert ast.dump(UndoAlias().visit(ast.parse(raw)),include_attributes=False)==ast.dump(prior,include_attributes=False)
args=SimpleNamespace(
    portuguese_preparation_proof=OUT/'Portuguese_FLEURS_retention_preparation_v1.json',
    portuguese_baseline_proof=OUT/'Portuguese_retention_baselines_v1.json',
    portuguese_replay_manifest=ROOT/'work/data/portuguese_retention_v1/replay_train_pt.jsonl',
    portuguese_dev_manifest=ROOT/'work/data/portuguese_retention_v1/validation_pt.jsonl')
def file_record(path):
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':str(path),'sha256':digest}
run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
calls=[n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='file_record' and isinstance(n.args[0],ast.Attribute) and isinstance(n.args[0].value,ast.Name) and n.args[0].value.id=='args']
assert len(calls)==5
for call in calls:
    result=eval(compile(ast.Expression(call),str(new),'eval'),{'file_record':file_record,'args':args})
    assert Path(result['path']).is_file() and len(result['sha256'])==64
controller=OUT/'run_TSC_uniform_full_v1.py'; ct=ast.parse(controller.read_text(encoding='utf-8'))
pins={n.targets[0].id:ast.literal_eval(n.value) for n in ct.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'WORKER_SHA','EXPECTED_SYNC_LINE'}}
digest=hashlib.sha256(new.read_bytes()).hexdigest()
assert pins['WORKER_SHA']==digest and pins['EXPECTED_SYNC_LINE']==667
assert 'torch.cuda.synchronize()' in raw.splitlines()[666]
receipt={'passed':True,'kind':'ROOT_uniform_worker_local_shadow_repair_v1','worker_sha256':digest,
    'controller_sha256':hashlib.sha256(controller.read_bytes()).hexdigest(),
    'sole_change_file_record_alias':True,'scientific_training_AST_identical_after_alias_reversal':True,
    'startup_file_record_expressions_executed':5,'actual_gpu_trial':False,
    'actual_state_roundtrip_claimed':False,'optimizer_observer_line_preserved':667}
dest=OUT/'TSC_uniform_full_record_shadow_ROOT_verification_v1.json'
assert not dest.exists()
dest.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
