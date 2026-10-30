"""Capture this real guarded native identity, then run the exact reviewed adapter."""
import runpy,sys
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from fastmss_owned_process_closure_v2 import identity
from bpe_broad_audio_v1 import record,require
from fastmss_BPE9223_feature_engine_v5 import write_new
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
script=P/'work/research/retire_ASR18_completed_hard_features_v2.py'
require(record(script)['sha256']=='8eeab9f28ae230f6d4868fc6af387dc7899b7dfe54b5dcd1d89a7248344cfaef','Exact reviewed retirement source')
write_new(N/'hard_feature_retirement_v1/actual_producer_identity.json',{'producer':identity(),'retirement_source':record(script)})
sys.argv=[str(script),*sys.argv[1:]];runpy.run_path(str(script),run_name='__main__')
