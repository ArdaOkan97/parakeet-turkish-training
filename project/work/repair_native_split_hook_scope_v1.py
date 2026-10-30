"""Narrow pre-freeze repair: pin NEW helper to new trainer's sync544."""
from pathlib import Path
import hashlib
import json
import ast
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/parakeet_turkish'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
trainer_sha=sha(OUT/'train_lora_batched_data_v10.py')
helper=(OUT/'cuda_cache_reclaim_v1.py').read_text(encoding='utf-8')
helper=helper.replace("FROZEN_TRAINER_SHA256 = 'b32aa1dc09d5ff7dd85be9c309e8d69a095b8cf859b5498136a27d5d5af7cd9d'", "FROZEN_TRAINER_SHA256 = '"+trainer_sha+"'")
helper=helper.replace('calls[0].lineno != 539','calls[0].lineno != 544')
path=OUT/'cuda_cache_reclaim_v2.py'
if path.exists():raise FileExistsError(path)
path.write_text(helper,encoding='utf-8',newline='\n');helper_sha=sha(path)
for name,expected in [('launch_with_cache_reclaim_v11.py','2305929d9132f85e9bcd257b5b40cdb39d9a8fd3c04da2480a77b7f0611af8f1'),('launch_with_memory_guard_v14.py','e026ad4ef456bdbf1146486d45705c2cd2d9cd4a30ce524adb013f3e5bcd67b2')]:
    path=OUT/name
    if sha(path)!=expected:raise ValueError('Unlaunched source changed before scoped repair:'+name)
    text=path.read_text(encoding='utf-8').replace('cuda_cache_reclaim_v1','cuda_cache_reclaim_v2')
    if name.startswith('launch_with_memory'):
        text=text.replace('99ec5974fcd4714fa6fa341eb5cde70dc817085c997bd564c2b465aaaf98a543',helper_sha)
        text=text.replace('2305929d9132f85e9bcd257b5b40cdb39d9a8fd3c04da2480a77b7f0611af8f1',sha(OUT/'launch_with_cache_reclaim_v11.py'))
    ast.parse(text);path.write_text(text,encoding='utf-8',newline='\n')
ready=OUT/'native_split_runtime_deployment_readiness_v1.json';value=json.loads(ready.read_text(encoding='utf-8'))
for name in ('cuda_cache_reclaim_v2.py','launch_with_cache_reclaim_v11.py','launch_with_memory_guard_v14.py'):value['new_sources_sha256'][name]=sha(OUT/name)
value['frozen_originals_sha256']['cuda_cache_reclaim_v1.py']=sha(OUT/'cuda_cache_reclaim_v1.py')
value['hook_scope_version_note']='NEW helper2 differs only in default trainer10 SHA and exact optimizer-end sync544; helper1 remains unchanged.'
ready.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(value['new_sources_sha256'],indent=2))
