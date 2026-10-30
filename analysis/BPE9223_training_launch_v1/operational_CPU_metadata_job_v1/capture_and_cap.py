"""In-process strict512MB/current+OSpeak cap around the reviewed metadata job."""
import json,runpy,sys
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');sys.path[:0]=[str(P/'outputs/parakeet_turkish'),str(P/'work/research')]
from bpe_broad_audio_v1 import require,record
from fastmss_owned_process_closure_v2 import identity
from run_ASR18_BPE9223_full_run_owner_v5 import private_lifetime_limit
import fastmss_BPE9223_cpu_common_v7 as core
original=core.resource_guard
def strict_guard():
    value=original();private_lifetime_limit(512_000_000);return value
core.resource_guard=strict_guard
target=P/'work/research/prepare_ASR18_operational_CPU_configuration_v1.py'
require(record(target)['sha256']=='25d310a51401efa6baff772fea8a073eb2d0a1ed4a049ec5d7e47e90c81777ec','Reviewed exact metadata assembler')
with (Path(__file__).parent/'actual_producer_identity.json').open('x',encoding='utf-8') as stream:json.dump({'producer':identity(),'source':record(target)},stream)
strict_guard();sys.argv=[str(target),*sys.argv[1:]];runpy.run_path(str(target),run_name='__main__');strict_guard()
