"""Freeze the NEW prepared domain-BPE routes and actual mocked routing receipt."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/"outputs/parakeet_turkish"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
parent=HERE/"HF_BPE512_runtime_chain_pins_v2.json"
old=json.loads(parent.read_text(encoding="utf-8"))
for name,value in old["source_sha256"].items():assert sha(HERE/name)==value
names=[]
for name in old["source_sha256"]:
    for stem in ("cuda_cache_caller_bpe512","launch_with_io_retry_bpe512","launch_with_cache_reclaim_bpe512",
            "launch_with_memory_guard_bpe512","launch_native_training_bpe512","train_lora_batched_bpe512","hf_bpe512_training"):
        if name==stem+"_v1.py":name=stem+"_v2.py"
    names.append(name)
names.extend(("launch_with_memory_guard_BPE512_resume_v2.py","turkish_domain_sampler_v2.py","turkish_domain_sampler_v1.py","train_lora_batched_data_v11.py"))
source_sha={n:sha(HERE/n) for n in names}
proof_path=HERE/"HF_BPE512_domain_runtime_CPU_dispatch_verification_v2.json"
proof=json.loads(proof_path.read_text(encoding="utf-8"))
assert proof["passed"] and proof["check_count"]==32
assert all(source_sha[name]==value for name,value in proof["source_sha256"].items())
path=HERE/"HF_BPE512_domain_runtime_chain_pins_v2.json"
value={"schema":1,"kind":"isolated_HF_BPE512_domain_runtime_chain_pins_v2","status":"prepared_only",
    "passed_CPU_routing_checks":True,"created_at_utc":datetime.now(timezone.utc).isoformat(),
    "source_sha256":source_sha,"source_paths":{n:str(HERE/n) for n in names},
    "worker_sha256":source_sha["train_lora_batched_bpe512_v2.py"],"outer_launcher":"launch_native_training_bpe512_v2.py",
    "memory_launcher":"launch_with_memory_guard_bpe512_v2.py","exact_memory_launcher":"launch_with_memory_guard_BPE512_resume_v2.py",
    "exact_memory_kind":"memory_guard_training_launcher_BPE512_resume_v2",
    "caller_target":{"function":"run","line":618,"worker_sha256":source_sha["train_lora_batched_bpe512_v2.py"]},
    "policy":{**old["policy"],"Turkish_sampling_mode":"domain30","FLEURS_fraction":.3,"other_domain":"CV (noTSC)",
        "replay_rng_independent_of_Turkish":True,"source_replay_seed":"must preserve latest CLOSED domain recipe",
        "minimum_domain_horizon_original_pool":13009,"fresh_weights_only_optimizer":"initial arm only",
        "exact_same_family_full_state_continuation":True,"exact_resume_requires_original_actual_pre_Adam_300_equivalence":True},
    "CPU_dispatch_proof_path":str(proof_path),"CPU_dispatch_proof_sha256":sha(proof_path),
    "repeatable_fixture_path":str(ROOT/"work/research/check_bpe512_domain_runtime_dispatch_v2.py"),
    "repeatable_fixture_sha256":sha(ROOT/"work/research/check_bpe512_domain_runtime_dispatch_v2.py"),
    "prior_original_BPE_route_receipt":{"path":str(parent),"sha256":sha(parent),"unchanged":True},
    "actual_GPU_training_or_model_execution":False,"actual_GPU_six_language_equivalence":"required inside frozen trainer before Adam; pending",
    "closed_source_selected_weights":"latest actual CLOSED qualifying domain BEST required; no source selected or conversion activated here",
    "old_sources_or_active_processes_changed":False,
    "fixture_scope":"Actual five-layer route/parser/validator/atomic serializer/caller with ML,Win32 sampling,preload,and external source identities mocked. Exact memory admission is actual stdlib; no checkpoint parsing/actual exact restoration claimed."}
with path.open("x",encoding="utf-8") as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
print(json.dumps({"path":str(path),"sha256":sha(path),"proof_sha256":sha(proof_path),"sources":len(names)}))
