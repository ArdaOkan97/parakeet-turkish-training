import json
from pathlib import Path
R=Path(__file__).parent
OUT=Path(r'E:\gpu_training_scratchpad\exports\parakeet-tdt-0.6b-v3-tr-experimental-400000-20261029-v3')
config=json.loads((OUT/'adapter/adapter_config.json').read_text())
(OUT/'merged/inference_config.json').write_text(json.dumps({k:config[k] for k in ['checkpoint_update','native_vocabulary','native_blank_id','native_duration_ids','frozen_matrix_names']},indent=2)+'\n')
usage={
'merged':'''## Load the merged native model

This repository contains a **standalone native `.nemo` archive** with all LoRA updates merged. No adapter runtime or custom model class is required. Install a compatible PyTorch/CUDA build and `nemo_toolkit[asr]==3.0.0` (the validation version). NeMo ASR installation is normally simplest on Linux.

```python
from nemo.collections.asr.models import ASRModel

model = ASRModel.from_pretrained("ArdaOkan/parakeet-tdt-0.6b-v3-tr-experimental")
model.eval()
output = model.transcribe(["audio.wav"], batch_size=1)
print(output[0].text)
```

Or download the archive and restore it directly:

```python
from huggingface_hub import hf_hub_download
from nemo.collections.asr.models import EncDecRNNTBPEModel

path = hf_hub_download("ArdaOkan/parakeet-tdt-0.6b-v3-tr-experimental",
                       "parakeet-tdt-0.6b-v3-tr-experimental.nemo")
model = EncDecRNNTBPEModel.restore_from(path, map_location="cpu")
```

Use mono 16 kHz WAV input. FP32 restoration temporarily needs more host memory than the archive's 2.51 GB size. The optional [memory_efficient_restore.py](memory_efficient_restore.py) uses a memory-mapped extracted cache and BF16 matrix placement; it returns the same standard NeMo class. For that mixed-precision path:

```python
import torch
from memory_efficient_restore import restore_model  # download this reviewed helper too

model = restore_model(path, cache_dir="./nemo_cache", device="cuda")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    output = model.transcribe(["audio.wav"], batch_size=1)
```

The optional helper also needs `inference_config.json` next to it, and retains an extracted cache. Control Hugging Face and temporary storage using your environment/cache settings. Use [the adapter repository](https://huggingface.co/ArdaOkan/parakeet-tdt-0.6b-v3-tr-experimental-lora) if you want the unmerged updates instead.

''',
'adapter':'''## Load the unmerged LoRA and full heads

This repository contains **192 rank-8 LoRA tensors plus 15 fully trained decoder/projector/joint tensors**, the extended tokenizer and a loader. It is a custom native NeMo adapter format, **not a PEFT adapter**. The full heads are necessary because training changed them; attention LoRA alone would not reproduce this checkpoint. The weights are stored in `adapter_model.safetensors` (about 84.1 MB). Install a compatible PyTorch/CUDA build, `nemo_toolkit[asr]==3.0.0`, `huggingface_hub` and `safetensors`.

Download the repository, review [load_adapter.py](load_adapter.py), then import it locally:

```python
from huggingface_hub import snapshot_download
import importlib.util
import torch

folder = snapshot_download("ArdaOkan/parakeet-tdt-0.6b-v3-tr-experimental-lora")
spec = importlib.util.spec_from_file_location("parakeet_adapter", folder + "/load_adapter.py")
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
model = loader.load_model(folder, device="cuda", cache_dir="./model_cache",
                          matrix_dtype=torch.bfloat16)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    output = model.transcribe(["audio.wav"], batch_size=1)
print(output[0].text)
```

The loader downloads NVIDIA's original `.nemo` at revision `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`, retains an extracted cache for memory-mapped loading, rounds the frozen base matrix values as in training, installs all 96 projection adapters, restores all 15 trained head tensors and applies the packaged tokenizer. `base_nemo=` can supply that exact base archive locally. With enough memory, omit `matrix_dtype` for FP32 parameters/execution. `memory_mapped=False` uses NeMo's default restore. The loader does not require `trust_remote_code=True`.

This package uses native NeMo token IDs and the zero-start embedding transformation described below. `hf_tokenizer.json` documents the original HF tokenizer but does not make these native-named weights a Transformers checkpoint. For a standalone NeMo archive, use [the merged repository](https://huggingface.co/ArdaOkan/parakeet-tdt-0.6b-v3-tr-experimental).

'''}
for kind in usage:
    p=OUT/kind/'README.md';text=p.read_text(encoding='utf-8')
    assert '## Load the' not in text
    text=text.replace('## License and attribution',usage[kind]+'## License and attribution',1)
    p.write_text(text,encoding='utf-8')
    (OUT/kind/'requirements.txt').write_text('nemo_toolkit[asr]==3.0.0\nhuggingface_hub>=0.33\nsafetensors>=0.5\n')
print('Prepared usage for both variants')
