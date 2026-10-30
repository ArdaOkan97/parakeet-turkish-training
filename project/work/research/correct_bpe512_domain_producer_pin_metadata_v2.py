"""Preserve frozen routes/receipt; correct one truthful producer-kind declaration."""
import ast,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/"outputs/parakeet_turkish"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
source=HERE/"HF_BPE512_domain_runtime_chain_pins_v2.json"
value=json.loads(source.read_text(encoding="utf-8"));source_hash=sha(source)
assert all(sha(HERE/name)==digest for name,digest in value["source_sha256"].items())
def kind(name,prefix):
    tree=ast.parse((HERE/name).read_text(encoding="utf-8"));values=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Dict):continue
        for key,item in zip(node.keys,node.values):
            if isinstance(key,ast.Constant) and key.value=="kind" and isinstance(item,ast.Constant) and isinstance(item.value,str) and item.value.startswith(prefix):values.append(item.value)
    assert len(values)==1,(name,values)
    return values[0]
initial=kind("launch_with_memory_guard_bpe512_v2.py","memory_guard_training_launcher_")
exact=kind("launch_with_memory_guard_BPE512_resume_v2.py","memory_guard_training_launcher_")
assert initial==exact=="memory_guard_training_launcher_bpe512_v1"
value["initial_memory_kind"]=initial;value["exact_memory_kind"]=exact
value["actual_runtime_producer_kinds"]={"io":kind("launch_with_io_retry_bpe512_v2.py","atomic_io_retry_"),
    "cache":kind("launch_with_cache_reclaim_bpe512_v2.py","cache_reclamation_training_launcher_"),"initial_memory":initial,"exact_memory":exact}
value["declaration_correction"]={"created_at_utc":datetime.now(timezone.utc).isoformat(),"superseded_receipt":{"path":str(source),"sha256":source_hash},
    "reason":"Frozen newv2 wrappers retain inherited v1 evidence-kind literals. Prior exact_memory_kind label was inaccurate; actual producer kinds are now source-AST verified. Runtime sources and their scientific/admission semantics are unchanged.",
    "required_runtime_distinction":"Validate exact memory source0252f27a.../initial1506f562... and matching argv/admission; the shared inherited evidence kind alone does not distinguish initial from exact.",
    "runtime_sources_changed":False,"CPU_dispatch_receipt_rewritten":False}
path=HERE/"HF_BPE512_domain_runtime_chain_pins_actual_producer_v2.json"
with path.open("x",encoding="utf-8") as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
assert sha(source)==source_hash
print(json.dumps({"path":str(path),"sha256":sha(path),"actual_runtime_producer_kinds":value["actual_runtime_producer_kinds"]}))
