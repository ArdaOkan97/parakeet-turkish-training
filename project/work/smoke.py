import os, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForTDT

torch.set_num_threads(4)
root = Path(__file__).resolve().parent.parent
path = (root/'work/base_model_path.txt').read_text().strip()
processor = AutoProcessor.from_pretrained(path, local_files_only=True)
model = AutoModelForTDT.from_pretrained(path, dtype=torch.float32, local_files_only=True).cuda()
sys.path.insert(0,str(root/'outputs/parakeet_turkish'))
from runtime import corrected_tdt_loss
model.loss_function=corrected_tdt_loss
for parameter in model.encoder.parameters(): parameter.requires_grad_(False)
model.train(); model.encoder.eval()
audio = np.sin(np.arange(32000, dtype=np.float32)*0.07)*0.1
inputs=processor(audio=[audio],text=['Merhaba dünya.'],sampling_rate=16000,return_tensors='pt')
inputs={k:v.cuda() for k,v in inputs.items()}
start=time.perf_counter()
with torch.autocast('cuda',dtype=torch.bfloat16):
    out=model(**inputs)
print('loss',out.loss.item(),flush=True)
out.loss.backward()
bad=[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
print('seconds',time.perf_counter()-start,'bad_gradients',bad,'peak_vram_gb',torch.cuda.max_memory_allocated()/1e9,flush=True)
if bad: raise RuntimeError('Nonfinite training gradients')
