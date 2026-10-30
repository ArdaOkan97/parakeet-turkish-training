"""NEW exact record3-driver path fix; AST/static caller scan, no target imports."""
from pathlib import Path
import ast,hashlib,json
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def write_source(old,new,changes):
    assert not new.exists();text=old.read_text(encoding='utf-8-sig')
    for before,after,count in changes:
        assert text.count(before)==count; text=text.replace(before,after)
    ast.parse(text);compile(text,str(new),'exec');new.write_text(text,encoding='utf-8',newline='\n')
    raw=new.read_bytes();print({'path':str(new),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'AST_only':True})
write_source(S/'asr18_full_run_native_owner_v4.py',S/'asr18_full_run_native_owner_v5.py',[
 ("record(self.config['worker_driver_source'])","record(self.config['worker_driver_source']['path'])",2)])
write_source(R/'asr18_full_run_owned_GPU_runtime_v5.py',R/'asr18_full_run_owned_GPU_runtime_v6.py',[
 ('from asr18_full_run_native_owner_v4 import FullRunNativeOwner','from asr18_full_run_native_owner_v5 import FullRunNativeOwner',1)])
for path in (S/'asr18_full_run_native_owner_v5.py',R/'asr18_full_run_owned_GPU_runtime_v6.py',
             S/'asr18_full_run_parent_controller_v7.py',R/'run_ASR18_full_run_owner_v9.py'):
    text=path.read_text(encoding='utf-8-sig');tree=ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='record' and node.args:
            raw=ast.unparse(node.args[0])
            assert raw not in ("config['worker_driver_source']","self.config['worker_driver_source']"),raw
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    path=Path(path).resolve();raw=path.read_bytes();return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(path,value):
    assert not path.exists();path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n');return pin(path)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
source=read(A/'full_owner_source_contract_v3.json')
mapping={str(S/'asr18_full_run_native_owner_v4.py'):pin(S/'asr18_full_run_native_owner_v5.py'),
         str(R/'asr18_full_run_owned_GPU_runtime_v5.py'):pin(R/'asr18_full_run_owned_GPU_runtime_v6.py')}
source['bindings']=[mapping.get(item['path'],item) for item in source['bindings']]
for old,new in mapping.items():
    source['direct_sources'].pop(Path(old).name,None);source['direct_sources'][Path(new['path']).name]=new
source_pin=write(A/'full_owner_source_contract_v4.json',source)
identity=read(A/'full_owner_validation_identity_v3.json');identity['full_run_source_contract']=source_pin
worker=pin(R/'asr18_full_run_owned_GPU_runtime_v6.py')
identity['persistent_prefix_sources'].update(native_worker=worker,recovery_worker=worker)
identity_pin=write(A/'full_owner_validation_identity_v4.json',identity)
inputs=read(A/'full_owner_exact_ROOT_review_inputs_v4.json')
inputs.update(source_manifest=source_pin,identity=identity,identity_binding=identity_pin,run_identity_sha256=digest(identity),worker_source=worker)
inputs['DEV_native_boundary_sources'].update(native_worker=worker,recovery_worker=worker)
inputs['DEV_source_bindings'].pop('asr18_full_run_native_owner_v4.py',None)
inputs['DEV_source_bindings']['asr18_full_run_native_owner_v5.py']=pin(S/'asr18_full_run_native_owner_v5.py')
input_pin=write(A/'full_owner_exact_ROOT_review_inputs_v5.json',inputs)
print(json.dumps({'source_contract':source_pin,'identity':identity_pin,'ROOT_inputs':input_pin,'bindings':len(source['bindings'])}))
