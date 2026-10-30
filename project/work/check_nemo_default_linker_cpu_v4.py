import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'outputs/parakeet_turkish'))
import launch_nemo_linker_v4 as helper
import torch
from cuda.pathfinder import load_nvidia_dynamic_lib

data = {'schema': 1, 'kind': 'default_torch_first_linker_cpu_control_v4',
        'script_sha256': helper.sha(__file__), 'launcher_sha256': helper.sha(helper.__file__),
        'cuda_initialized': torch.cuda.is_initialized(),
        'actual_binding_linker': helper.binding_linker(),
        'pathfinder_linker': str(load_nvidia_dynamic_lib('nvJitLink').abs_path),
        'pathfinder_nvvm': str(load_nvidia_dynamic_lib('nvvm').abs_path),
        'loaded_libraries': helper.loaded_modules(), 'CUDA_executed': False}
helper.require(data['cuda_initialized'] is False, 'No CUDA initialization allowed')
path = root / 'outputs/parakeet_turkish/nemo_default_linker_cpu_control_v4.json'
helper.require(not path.exists(), 'Preserve prior control')
path.write_text(json.dumps(data, indent=2), encoding='utf-8')
print(json.dumps({k: data[k] for k in ('actual_binding_linker', 'pathfinder_nvvm', 'cuda_initialized')}))
