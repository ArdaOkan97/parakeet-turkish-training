import os,sys,json,time,hashlib,shutil
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');R=Path(__file__).parent
sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from paired_windows_commit_telemetry_v1 import paired_sample
samples=[paired_sample(),paired_sample()]
with (R/'native_probe_admission.json').open('x') as f:json.dump(samples,f);f.flush();os.fsync(f.fileno())
assert all(s['global_memory_status_ex']['ullAvailPageFile_bytes']>=8066304384 and s['get_performance_info_after']['physical_available_bytes']>=5316304384 for s in samples)
from preload_nemo_linker_v1 import preload
preload()
from nemo.collections.asr.models import EncDecRNNTBPEModel
import torch,sentencepiece,safetensors
import importlib.metadata as m
try:import peft;peft_version=peft.__version__
except ImportError:peft_version=None
result={'nemo':m.version('nemo_toolkit'),'torch':torch.__version__,'transformers':m.version('transformers'),'peft':peft_version,'class':str(EncDecRNNTBPEModel),'CUDA_initialized':torch.cuda.is_initialized()}
(R/'native_probe_complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
