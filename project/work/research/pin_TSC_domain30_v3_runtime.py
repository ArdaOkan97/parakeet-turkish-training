"""Freeze only NEW v3 route drafts after ROOT-reviewed new worker/sampler identity."""
import argparse,ast,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(args):
    original=HERE/'TSC_runtime_chain_pins_v2.json'
    assert sha(original)=='7f50e684b62bd250d4c3fe51248d44380a27aeeb8e6c627bceb47de6d2fde37b'
    old=json.loads(original.read_text(encoding='utf-8'))['source_sha256']
    for name,digest in old.items():assert sha(HERE/name)==digest,name
    science={'train_lora_batched_tsc_v3.py':args.worker_sha,'tsc_domain_sampler_v3.py':args.sampler_sha,
             'tsc_domain_training_identity_v3.py':args.identity_sha}
    for name,digest in science.items():assert sha(HERE/name)==digest,name
    worker=ast.parse((HERE/'train_lora_batched_tsc_v3.py').read_text(encoding='utf-8'))
    run=next(n for n in worker.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    sync=[n for n in ast.walk(run) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.cuda.synchronize']
    assert len(sync)==1 and not sync[0].args and not sync[0].keywords
    line=sync[0].lineno
    substitutions={old['train_lora_batched_tsc_v2.py']:science['train_lora_batched_tsc_v3.py']}
    order=('cuda_cache_caller_tsc','launch_with_io_retry_tsc','launch_with_cache_reclaim_tsc',
           'launch_with_memory_guard_tsc','launch_native_training_tsc','launch_with_memory_guard_TSC_resume')
    created={}
    for stem in order:
        path=HERE/(stem+'_v3.py');text=path.read_text(encoding='utf-8')
        for previous,digest in substitutions.items():text=text.replace(previous,digest)
        if stem=='cuda_cache_caller_tsc':
            text=text.replace('lineno != 545',f'lineno != {line}').replace('return 545',f'return {line}').replace('run:545',f'run:{line}')
        if 'memory_guard' in stem:
            anchor="('train_lora_batched_data_v11.py', '"+old['train_lora_batched_data_v11.py']+"')]"
            assert text.count(anchor)==1
            dependencies={key:value for key,value in science.items() if key!='train_lora_batched_tsc_v3.py'}
            dependencies['turkish_domain_sampler_v1.py']=sha(HERE/'turkish_domain_sampler_v1.py')
            text=text.replace(anchor,anchor[:-1]+',\n            '+',\n            '.join(repr((k,v)) for k,v in dependencies.items())+']')
            anchor="'TSC_explicit_asset_records': tsc_assets,"
            assert text.count(anchor)==1
            text=text.replace(anchor,"'TSC_domain_sampler_sha256': "+repr(args.sampler_sha)+",\n               'TSC_domain_identity_sha256': "+repr(args.identity_sha)+",\n               "+anchor)
        if stem=='launch_native_training_tsc':
            anchor="require(float(option(worker,'--epochs'))==1. and int(option(worker,'--steps'))>0,'Explicit one-pass combined-pool horizon required')"
            assert text.count(anchor)==1
            text=text.replace(anchor,"require(option(worker,'--sampling-mode')=='domain30' and float(option(worker,'--fleurs-fraction'))==.3\n   and int(option(worker,'--replay-seed'))>=0,'Explicit preserved30%FLEURS/independent replay seed required')\n "+anchor.replace('combined-pool','minimum-domain'))
        ast.parse(text);path.write_text(text,encoding='utf-8',newline='\n')
        created[path.name]=sha(path)
        if stem+'_v2.py' in old:substitutions[old[stem+'_v2.py']]=created[path.name]
    for name,digest in old.items():assert sha(HERE/name)==digest,name
    print(json.dumps({'new_source_sha256':created,'science':science,'caller_line':line,'all_v2_unchanged':True}))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('worker-sha','sampler-sha','identity-sha'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
