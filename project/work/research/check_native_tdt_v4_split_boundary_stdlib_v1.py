"""Source/one-call stub only. No Torch, native kernel, model, tensor or CUDA."""
import ast
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
import weakref

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'outputs/parakeet_turkish'
old=HERE/'native_tdt_checkpointed_v3.py';new=HERE/'native_tdt_checkpointed_v4.py'
before=ast.parse(old.read_text(encoding='utf-8'));after=ast.parse(new.read_text(encoding='utf-8'))
functions=lambda tree:{n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
a,b=functions(before),functions(after)
checks=[]
def require(ok,label):
    if not ok:raise AssertionError(label)
    checks.append(label)

require(set(a)==set(b),'same callable API')
for name in set(a)-{'_clip_loss','configuration_identity','cache_cleanup_evidence'}:
    require(ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False),
        'unchanged actual source function '+name)
trim=copy.deepcopy(b['_clip_loss'])
added=[node for node in trim.body if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
    and isinstance(node.value.func,ast.Name) and node.value.func.id=='reclaim_inactive_cuda_cache']
require(len(added)==1 and ast.unparse(added[0].value)=="reclaim_inactive_cuda_cache(ep.device, 'after_split_before_native_gradient')",
    'exact one requested cleanup call')
trim.body.remove(added[0])
require(ast.dump(trim,include_attributes=False)==ast.dump(a['_clip_loss'],include_attributes=False),
    'clip math and RNG path exactly original AST after removing cleanup')
assignments=lambda tree:{n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
    if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('SOURCE_FILES','CACHE_POLICY')}
constants_old,constants_new=assignments(before),assignments(after)
require(constants_old['SOURCE_FILES']==constants_new['SOURCE_FILES'],'official native pins unchanged')
policy=constants_new['CACHE_POLICY'].copy()
require(policy.pop('after_packed_split_before_native_gradient') is True and policy==constants_old['CACHE_POLICY'],
    'only added boundary policy;750MB and cleanup mechanics unchanged')

events=[];packed_reference=None
class Act:
    dtype='float32'
    def float(self):events.append('float');return self
    def contiguous(self):events.append('contiguous');return self
class Packed:
    shape=(1,2,3,8716)
    def __getitem__(self,index):events.append('split');return Act()
    def __del__(self):events.append('packed_deleted')
class Projected:
    shape=(1,2,4)
    device=SimpleNamespace(type='cuda')
    def __getitem__(self,index):return self
class DecoderProjected(Projected):shape=(1,3,4)
class Value:shape=(1,)
def joint(*args):
    global packed_reference
    value=Packed();packed_reference=weakref.ref(value);events.append('joint');return value
def cleanup(device,phase):
    gc.collect()
    require(packed_reference() is None,'mock packed owner deleted before cleanup')
    events.append(phase)
def loss(*args):
    events.append('mock_native_loss_entry_and_gradient_allocation')
    require(len(args)==12 and args[5:]==(8192,[0,1,2,3,4],'none',0.,0.,0.,0.),'backend operands/settings unchanged')
    random.uniform(0,1);return Value()
backend=SimpleNamespace(loss=loss,blank=8192,durations=[0,1,2,3,4],reduction='none',fastemit_lambda=0.,clamp=0.,sigma=0.,omega=0.)
env={'_require':lambda ok,message:None if ok else (_ for _ in ()).throw(ValueError(message)),
    'torch':SimpleNamespace(float32='float32',nn=SimpleNamespace(functional=SimpleNamespace(log_softmax=lambda x,dim:x))),
    'random':random,'reclaim_inactive_cuda_cache':cleanup}
exec(compile(ast.fix_missing_locations(ast.Module(body=[b['_clip_loss']],type_ignores=[])),str(new),'exec'),env)
state=random.getstate()
value=env['_clip_loss'](joint,Projected(),DecoderProjected(),object(),object(),object(),backend,8711,5)
require(value.shape==(1,) and random.getstate()==state,'raw return schema and real Python RNG restored around mock native call')
require(events.index('packed_deleted')<events.index('after_split_before_native_gradient')
    <events.index('mock_native_loss_entry_and_gradient_allocation'),'split/del cleanup occurs before backend gradient allocation entry')

payload={'kind':'inactive_native_tdt_v4_split_boundary_stdlib_verification_v1','passed':True,'checks':len(checks),
    'check_labels':checks,'adapter':{'path':str(new),'sha256':hashlib.sha256(new.read_bytes()).hexdigest()},
    'immutable_parent':{'path':str(old),'sha256':hashlib.sha256(old.read_bytes()).hexdigest()},
    'fixture_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'stub_route_events':events,
    'Torch_or_NeMo_imported':False,'actual_tensor_native_kernel_or_CUDA':False,
    'actual_inactive_packed_storage_or_memory_reclaimed_proven':False,'actual_GPU_numerical_or_memory_proof_pending':True}
target=HERE/'native_tdt_v4_split_boundary_stdlib_verification_v1.json'
with target.open('x',encoding='utf-8') as stream:json.dump(payload,stream,indent=2);stream.write('\n')
print(json.dumps({'passed':True,'checks':len(checks),'adapter_sha256':payload['adapter']['sha256'],'report':str(target)}))
