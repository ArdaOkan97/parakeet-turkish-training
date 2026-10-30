"""One-time stdlib assembly of NEW isolated routes; never edits templates."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/parakeet_turkish'
WORKER = 'train_lora_batched_domain30_v1.py'
WORKER_SHA = '7242b0ce65619f8df38593e89c6ed5b7a01487d1d1174c2e28f73d00bff8ee4f'
DEPENDENCIES = {
    'turkish_domain_sampler_v2.py':'822d0ddb608d719112e4a4285ad42a76ef7f1e4ede0cf4d97211fe38f97c62fd',
    'domain30_training_identity_v1.py':'151bf1b3236e3e23c269f7be2c2e94bf12a71b1e547b64a70f26d6f77ae81f5e',
}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    pins = json.loads((OUT/'HF_BPE512_runtime_chain_pins_v2.json').read_text())['source_sha256']
    for name,digest in pins.items(): assert sha(OUT/name)==digest, name
    assert sha(OUT/WORKER)==WORKER_SHA
    for name,digest in DEPENDENCIES.items(): assert sha(OUT/name)==digest
    mapping = {
        'cuda_cache_caller_bpe512_v1.py':'cuda_cache_caller_domain30_v1.py',
        'launch_with_io_retry_bpe512_v1.py':'launch_with_io_retry_domain30_v1.py',
        'launch_with_cache_reclaim_bpe512_v1.py':'launch_with_cache_reclaim_domain30_v1.py',
        'launch_with_memory_guard_bpe512_v1.py':'launch_with_memory_guard_domain30_v1.py',
        'launch_native_training_bpe512_v1.py':'launch_native_training_domain30_v1.py',
    }
    created = {}
    for old,new in mapping.items():
        source=(OUT/old).read_text(encoding='utf-8')
        source=source.replace('train_lora_batched_bpe512_v1.py',WORKER).replace(pins['train_lora_batched_bpe512_v1.py'],WORKER_SHA)
        source=source.replace('lineno != 588','lineno != 507').replace('return 588','return 507').replace('run:588','run:507').replace('BPE512','domain30').replace('BPE','domain').replace('bpe512_v1','domain30_v1')
        for previous,current in created.items(): source=source.replace(pins[previous],current)
        if old.startswith('launch_with_memory_guard_'):
            start=source.index("            ('hf_bpe512_training_v1.py'")
            end=source.index(')]', start)+2
            replacements={**DEPENDENCIES,'turkish_domain_sampler_v1.py':sha(OUT/'turkish_domain_sampler_v1.py'),
                'train_lora_batched_data_v11.py':sha(OUT/'train_lora_batched_data_v11.py')}
            source=source[:start]+',\n'.join('            '+repr((name,digest)) for name,digest in replacements.items())+']'+source[end:]
        if old.startswith('launch_native_training_'):
            old_flags="for flag in ('--resume-weights','--bpe-source-checkpoint','--bpe-source-selection','--bpe-source-closure','--bpe-conversion-report'):\n  scoped(option(worker,flag))"
            new_flags="for flag in ('--resume-weights','--source-closure'):\n  scoped(option(worker,flag))\n require(option(worker,'--sampling-mode') in ('domain30','uniform'),'Explicit isolated domain30 or independent-replay uniform mode required')\n int(option(worker,'--replay-seed'))"
            assert old_flags in source
            source=source.replace(old_flags,new_flags)
        target=OUT/new
        assert not target.exists(),target
        compile(source,str(target),'exec')
        with target.open('x',encoding='utf-8',newline='\n') as f:f.write(source)
        created[old]=sha(target)
    for name,digest in pins.items():assert sha(OUT/name)==digest,name
    print(json.dumps({'new_source_sha256':{mapping[old]:digest for old,digest in created.items()},'templates_unchanged':True}))


if __name__=='__main__':main()
