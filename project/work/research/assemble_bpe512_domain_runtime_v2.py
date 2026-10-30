"""Narrow NEW BPE domain route assembly; no old runtime/core changes or ML imports."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/"outputs/parakeet_turkish"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
old=json.loads((HERE/"HF_BPE512_runtime_chain_pins_v2.json").read_text(encoding="utf-8"))["source_sha256"]
for name,value in old.items():assert sha(HERE/name)==value
WORKER="train_lora_batched_bpe512_v2.py";WS="3722684dded5026ddfabf9e30f5ea568b757fce31b4e1854d8b7b5aae04ec0a3"
HELPER="hf_bpe512_training_v2.py";HS="7b7692251e7e66667da31d5b3515026da15fded4a0e9d4fb4cfe0185727c316f"
assert sha(HERE/WORKER)==WS and sha(HERE/HELPER)==HS
def convert(text):
    for stem in ("cuda_cache_caller_bpe512","launch_with_io_retry_bpe512","launch_with_cache_reclaim_bpe512",
            "launch_with_memory_guard_bpe512","launch_native_training_bpe512","train_lora_batched_bpe512","hf_bpe512_training"):
        text=text.replace(stem+"_v1",stem+"_v2")
    text=text.replace(old["train_lora_batched_bpe512_v1.py"],WS).replace(old["hf_bpe512_training_v1.py"],HS)
    text=text.replace("run:588","run:618").replace("return 588","return 618").replace("lineno != 588","lineno != 618")
    return text
def save(name,text):
    compile(text,str(HERE/name),"exec")
    with (HERE/name).open("x",encoding="utf-8",newline="\n") as stream:stream.write(text)
def load(name):return convert((HERE/name).read_text(encoding="utf-8"))
CALLER="cuda_cache_caller_bpe512_v2.py";IO="launch_with_io_retry_bpe512_v2.py";CACHE="launch_with_cache_reclaim_bpe512_v2.py"
MEMORY="launch_with_memory_guard_bpe512_v2.py";EXACT="launch_with_memory_guard_BPE512_resume_v2.py";OUTER="launch_native_training_bpe512_v2.py"
save(CALLER,load("cuda_cache_caller_bpe512_v1.py"))
save(IO,load("launch_with_io_retry_bpe512_v1.py"))
text=load("launch_with_cache_reclaim_bpe512_v1.py").replace(old["launch_with_io_retry_bpe512_v1.py"],sha(HERE/IO))
save(CACHE,text)
text=load("launch_with_memory_guard_bpe512_v1.py").replace(old["launch_with_cache_reclaim_bpe512_v1.py"],sha(HERE/CACHE)).replace(old["cuda_cache_caller_bpe512_v1.py"],sha(HERE/CALLER))
extra=("turkish_domain_sampler_v2.py","turkish_domain_sampler_v1.py","train_lora_batched_data_v11.py")
text=text.replace("('expand_trainable_bpe_checkpoint_v1.py', '"+old["expand_trainable_bpe_checkpoint_v1.py"]+"')]",
    "('expand_trainable_bpe_checkpoint_v1.py', '"+old["expand_trainable_bpe_checkpoint_v1.py"]+"'),\n"+
    ",\n".join("            ('"+n+"', '"+sha(HERE/n)+"')" for n in extra)+"]")
assert all("('"+n+"', '"+sha(HERE/n)+"')" in text for n in extra)
admission='''def bpe_argument_records(arguments):
    """Explicit domain-arm source inputs; no hidden model-only/exact reset."""
    values=list(arguments)
    if values[:1]==['--']:values=values[1:]
    records={}
    for flag in ('--bpe-source-checkpoint','--bpe-source-selection','--bpe-source-closure',
                 '--bpe-conversion-report','--native-backend-provenance',RESUME_FLAG):
        if values.count(flag)!=1 or values.index(flag)+1>=len(values):raise ValueError('Exactly one BPE asset required: '+flag)
        path=Path(values[values.index(flag)+1]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():raise ValueError('Existing workspace BPE asset required: '+flag)
        records[flag.removeprefix('--').replace('-','_')]={'path':str(path),'sha256':file_sha(path)}
    for flag in FORBIDDEN_RESUME:
        if flag in values:raise ValueError('BPE route forbids '+flag)
    if values.count('--sampling-mode')!=1 or values[values.index('--sampling-mode')+1]!='domain30':raise ValueError('Preserve domain30')
    if values.count('--replay-seed')!=1:int('invalid')
    int(values[values.index('--replay-seed')+1])
    return records


'''
text=text.replace("def creation_identity(pid):",admission+"def creation_identity(pid):")
text=text.replace("    run_id, argument_sha = argument_identity(args.arguments)","    run_id, argument_sha = argument_identity(args.arguments)\n    asset_records=bpe_argument_records(args.arguments)")
text=text.replace("'native_cleanup_adapter_sha256': NATIVE_ADAPTER_SHA256,","'native_cleanup_adapter_sha256': NATIVE_ADAPTER_SHA256,\n               'BPE_explicit_asset_records':asset_records,")
initial=text.replace("RESUME_FLAG","'--resume-weights'").replace("FORBIDDEN_RESUME","('--resume-exact','--resume-runtime-migration')")
save(MEMORY,initial)
exact=text.replace("RESUME_FLAG","'--resume-exact'").replace("FORBIDDEN_RESUME","('--resume-weights','--resume-runtime-migration')")
exact=exact.replace("memory_guard_training_launcher_bpe512_v2","memory_guard_training_launcher_BPE512_resume_v2")
save(EXACT,exact)
text=load("launch_native_training_bpe512_v1.py").replace(old["launch_with_memory_guard_bpe512_v1.py"],sha(HERE/MEMORY))
text=text.replace(" run_id=option(worker,'--run-id')"," require(option(worker,'--sampling-mode')=='domain30' and float(option(worker,'--fleurs-fraction'))==.3,'Preserve FLEURS30/restCV')\n int(option(worker,'--replay-seed'))\n run_id=option(worker,'--run-id')")
save(OUTER,text)
for name,value in old.items():assert sha(HERE/name)==value
for name in (CALLER,IO,CACHE,MEMORY,EXACT,OUTER):print(name,sha(HERE/name))
