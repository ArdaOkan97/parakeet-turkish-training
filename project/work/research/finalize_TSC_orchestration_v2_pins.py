"""Fill NEW draft pins from the frozen actual v2 route receipt; no runtime launch."""
import ast,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def replace(path,name,value):
    text=path.read_text(encoding='utf-8')
    text,n=re.subn(r'^'+re.escape(name)+r'\s*=.*$',name+' = '+repr(value),text,flags=re.M)
    assert n==1,(path,name,n)
    ast.parse(text)
    path.write_text(text,encoding='utf-8',newline='\n')
receipt=HERE/'TSC_runtime_chain_pins_v2.json';expected='7f50e684b62bd250d4c3fe51248d44380a27aeeb8e6c627bceb47de6d2fde37b'
assert sha(receipt)==expected
chain=json.loads(receipt.read_text(encoding='utf-8'));assert chain['kind']=='isolated_TSC_runtime_chain_pins_v2' and chain['passed_CPU_routing_checks'] and len(chain['source_sha256'])==23
for name,digest in chain['source_sha256'].items():assert sha(HERE/name)==digest,name
memory='launch_with_memory_guard_TSC_resume_v2.py';memory_sha='67a99f29575fbc1d765167b754463d34738879d060d41b1b5486ae3acaa37db5'
assert sha(HERE/memory)==memory_sha
first=HERE/'run_hf_TSC_arm_v2.py';replace(first,'CHAIN_SHA',expected)
outer=HERE/'launch_TSC_same_science_resume_v2.py';replace(outer,'PINS',chain['source_sha256']);replace(outer,'RUNTIME_MEMORY_SHA',memory_sha)
exact=HERE/'resume_TSC_same_science_v2.py';replace(exact,'TSC_CONTROLLER_SHA',sha(first))
supervisor=HERE/'supervise_TSC_same_science_v2.py'
for name,path in (('FIRST_SHA',first),('HELPER_SHA',exact),('PRELOADER_SHA',outer)):replace(supervisor,name,sha(path))
print(json.dumps({'source_sha256':{p.name:sha(p) for p in (first,outer,exact,supervisor)},'route_receipt_sha256':expected,'actual_runtime_source_count':23}))
