"""Capture this real guarded mutator identity before invoking the pinned adapter."""
import runpy,sys
from pathlib import Path
PROJECT=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from fastmss_owned_process_closure_v2 import identity
from fastmss_BPE9223_feature_engine_v3 import write_new
from bpe_broad_audio_v1 import record,require
target=PROJECT/'work/research/retire_ASR18_failed_genesis_features_v1.py'
require(record(target)['sha256']=='abb74ec910ce09dc2681d9eda708444d938ed6cea3100fa6f03e72c9ed6aad2f','Exact reviewed failed-window retirement source')
write_new(Path(__file__).parent/'actual_producer_identity.json',{'producer':identity(),'retirement_source':record(target)})
sys.argv=[str(target),*sys.argv[1:]]
runpy.run_path(str(target),run_name='__main__')
