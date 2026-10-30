"""Actual installed Parakeet classes, reduced dimensions, CPU-only batching parity."""
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoProcessor, ParakeetForTDT

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0, str(OUTPUT))
from batched_lora import collate_training_batch, batch_objective
from efficient_tdt import chunked_tdt_loss
from encoder_lora import install_encoder_lora

torch.set_num_threads(1)
torch.manual_seed(20261010)
base_path = (ROOT / "work/base_model_path.txt").read_text().strip()
processor = AutoProcessor.from_pretrained(base_path, local_files_only=True)
config = AutoConfig.from_pretrained(base_path, local_files_only=True)
config.encoder_config.hidden_size = 16
config.encoder_config.intermediate_size = 32
config.encoder_config.num_attention_heads = 4
config.encoder_config.num_key_value_heads = 4
config.encoder_config.num_hidden_layers = 2
config.encoder_config.subsampling_conv_channels = 4
config.decoder_hidden_size = 8
config.vocab_size = 17
config.blank_token_id = 16
config._attn_implementation = "sdpa"
original = ParakeetForTDT(config).eval()
teacher = {name: copy.deepcopy(getattr(original, name)).eval().requires_grad_(False)
           for name in ("encoder_projector", "decoder", "joint")}
waves, rows, cached = {}, [], []
lengths = (8000, 10400, 12800, 16800)
targets = (torch.tensor([3, 17, 4]), torch.tensor([2, 5]),
           torch.tensor([18, 6, 19, 7]), torch.tensor([8, 9, 10, 11, 12]))
for index, (samples, language) in enumerate(zip(lengths, ("tr", "en", "tr", "de"))):
    source = f"synthetic:{language}:{index}"
    t = np.arange(samples, dtype=np.float32) / 16000
    waveform = (0.2 * np.sin(2 * np.pi * (200 + 47 * index) * t)
                + 0.025 * np.cos(2 * np.pi * 631 * t)).astype(np.float32)
    waves[source] = waveform
    row = {"source_id": source, "language": language, "duration": samples / 16000}
    rows.append(row)
    inputs = processor(audio=[waveform], sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        encoded = original.encoder(**inputs)
        count = int(encoded.attention_mask.sum())
    cached.append({"source_id": source, "labels": targets[index],
                   "features": encoded.last_hidden_state[0, :count].detach().clone()})

model = copy.deepcopy(original)
old_embedding = model.decoder.embedding
old_head = model.joint.head
embedding = nn.Embedding(23, 8)
head = nn.Linear(8, 28)
with torch.no_grad():
    embedding.weight[:17].copy_(old_embedding.weight)
    embedding.weight[17:].normal_(std=0.02)
    head.weight[:17].copy_(old_head.weight[:17])
    head.bias[:17].copy_(old_head.bias[:17])
    head.weight[23:].copy_(old_head.weight[17:])
    head.bias[23:].copy_(old_head.bias[17:])
    head.weight[17:23].normal_(std=0.02)
    head.bias[17:23].fill_(-2)
model.decoder.embedding = embedding
model.joint.head = head
model.config.vocab_size = 23
layout = install_encoder_lora(model, rank=2, alpha=4)
with torch.no_grad():
    for name, parameter in model.named_parameters():
        if name.endswith("lora_B"):
            parameter.normal_(std=0.01)
model.train()
model.encoder.eval()
state = copy.deepcopy(model.state_dict())
reader = lambda row: waves[row["source_id"]]


def original_single(model, index):
    inputs = processor(audio=[waves[rows[index]["source_id"]]], sampling_rate=16000, return_tensors="pt")
    encoded = model.encoder(**inputs)
    count = int(encoded.attention_mask.sum())
    features = encoded.last_hidden_state[:, :count]
    ep = model.encoder_projector(features)
    labels = cached[index]["labels"][None]
    ids = torch.cat((torch.full((1, 1), 16), labels), dim=1)
    dp = model.decoder(ids)
    tdt = chunked_tdt_loss(model.joint, ep, dp, labels, 23, 16, [0,1,2,3,4],
                           torch.tensor([count]), torch.tensor([labels.shape[1]]), chunk_size=4)
    token = duration = feature = torch.zeros(())
    if rows[index]["language"] != "tr":
        base = cached[index]["features"][None].float()
        ts = torch.linspace(0, count - 1, min(8, count)).long()
        us = torch.linspace(0, dp.shape[1] - 1, min(8, dp.shape[1])).long()
        with torch.no_grad():
            logits = teacher["joint"](teacher["encoder_projector"](base)[:,ts,None],
                                       teacher["decoder"](ids)[:,None,us])
            p = logits[..., :17].float().softmax(-1)
            q = logits[..., 17:].float().softmax(-1)
        logits = model.joint(ep[:,ts,None], dp[:,None,us])
        token = (p * (p.clamp_min(1e-12).log() - logits[..., :23].float().log_softmax(-1)[..., :17])).sum(-1).mean()
        duration = (q * (q.clamp_min(1e-12).log() - logits[..., 23:].float().log_softmax(-1))).sum(-1).mean()
        feature = (1 - F.cosine_similarity(features.float(), base, dim=-1)).mean()
    return tdt + 5*token + .5*duration + feature, (tdt, token, duration, feature), features.detach()


checks = []
def require(name, condition):
    checks.append({"name": name, "passed": bool(condition)})
    assert condition, name

bn_before = {name: buffer.clone() for name, buffer in model.encoder.named_buffers()
             if "running_" in name or "num_batches_tracked" in name}
reference_gradients = reference_components = reference_features = None
cases = []
for mode, groups in (("original_B1x4", [[0],[1],[2],[3]]),
                     ("helper_B1x4", [[0],[1],[2],[3]]),
                     ("helper_B2x2_homogeneous", [[0,2],[1,3]]),
                     ("helper_B4x1_mixed", [[0,1,2,3]])):
    model.load_state_dict(state)
    model.zero_grad(set_to_none=True)
    components = [None]*4
    features = [None]*4
    mean_total = 0.0
    for group in groups:
        if mode == "original_B1x4":
            loss, values, current_features = original_single(model, group[0])
            components[group[0]] = torch.stack(values).detach()
            features[group[0]] = current_features[0].clone()
            scaled = loss / 4
        else:
            batch = collate_training_batch(processor, [rows[i] for i in group], [cached[i] for i in group],
                                            blank_id=16, old_size=17, device="cpu", audio_reader=reader)
            require(mode + " CPU batch mover preserves metadata", batch.to("cpu").target_lengths.device.type == "cpu")
            output = batch_objective(model, teacher, batch, old_size=17, chunk_size=4, return_debug_features=True)
            for index, position in enumerate(group):
                components[position] = torch.stack([output.per_clip[name][index] for name in ("tdt", "token_kl", "duration_kl", "feature_distillation")]).detach()
                features[position] = output.debug_encoder_features[index, :output.encoder_frames[index]].clone()
                if rows[position]["language"] == "tr":
                    require(mode + " exact zero Turkish preservation", bool((components[position][1:] == 0).all()))
            scaled = output.loss / len(groups)
        scaled.backward()
        mean_total += float(scaled.detach())
    gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    require(mode + " all trainable gradients finite/present", all(torch.isfinite(value).all() for value in gradients.values()))
    require(mode + " frozen encoder gradients absent", all(parameter.grad is None for parameter in model.encoder.parameters() if not parameter.requires_grad))
    require(mode + " BatchNorm buffers unchanged", all(torch.equal(value, dict(model.encoder.named_buffers())[name]) for name, value in bn_before.items()))
    if mode == "original_B1x4":
        reference_gradients, reference_components, reference_features = gradients, components, features
        reference_total = mean_total
    maximum_gradient_difference = max(float((value - reference_gradients[name]).abs().max()) for name, value in gradients.items())
    maximum_component_difference = max(float((value - baseline).abs().max()) for value, baseline in zip(components, reference_components))
    maximum_feature_difference = max(float((value - baseline).abs().max()) for value, baseline in zip(features, reference_features))
    require(mode + " real-class encoder valid-frame parity", maximum_feature_difference < 2e-4)
    require(mode + " all per-clip TDT/teacher/feature loss parity", maximum_component_difference < 2e-4)
    require(mode + " all head/LoRA gradient parity", maximum_gradient_difference < 2e-4)
    require(mode + " effective batch mean parity", abs(mean_total-reference_total) < 2e-4)
    norm = math.sqrt(sum(float(value.double().square().sum()) for value in reference_gradients.values()))
    difference = math.sqrt(sum(float((value-reference_gradients[name]).double().square().sum()) for name,value in gradients.items()))
    cases.append({"mode": mode, "mean_loss": mean_total, "loss_difference": abs(mean_total-reference_total),
                  "maximum_component_difference": maximum_component_difference,
                  "maximum_valid_encoder_feature_difference": maximum_feature_difference,
                  "maximum_gradient_difference": maximum_gradient_difference,
                  "all_trainable_gradient_relative_l2": difference / norm,
                  "per_clip_components_tdt_tokenKL_durationKL_feature": [value.tolist() for value in components]})

batch = collate_training_batch(processor, rows, cached, blank_id=16, old_size=17, device="cpu", audio_reader=reader)
model.zero_grad(set_to_none=True)
baseline = batch_objective(model, teacher, batch, old_size=17, chunk_size=4)
poisoned = copy.deepcopy(batch)
for index, length in enumerate(poisoned.frame_lengths.tolist()):
    poisoned.base_features[index, length:] = 1234
changed = batch_objective(model, teacher, poisoned, old_size=17, chunk_size=4)
require("padded teacher values never affect preservation", float(abs(baseline.loss-changed.loss)) < 1e-7)
wrong = copy.deepcopy(batch)
wrong.frame_lengths[0] += 1
try:
    batch_objective(model, teacher, wrong, old_size=17, chunk_size=4)
except ValueError:
    rejected = True
else:
    rejected = False
require("wrong singleton teacher-frame alignment rejected", rejected)
bad = copy.deepcopy(cached)
bad[1]["labels"][0] = 17
try:
    collate_training_batch(processor, rows, bad, blank_id=16, old_size=17, device="cpu", audio_reader=reader)
except ValueError:
    rejected = True
else:
    rejected = False
require("new token cannot enter original replay teacher", rejected)
report = {"passed": True, "gpu_used": False, "torch_version": torch.__version__, "device": "CPU FP32",
          "scope": "Installed Parakeet encoder/subsampling/SDPA/convolution/BatchNorm and real pinned processor at reduced2-layer/16-hidden scale,8 rank2 attention adapters with nonzero B, original frozen teacher heads and full-vocab TDT. Not actual0.6B GPU/BF16 parity.",
          "source_sha256": hashlib.sha256((OUTPUT/'batched_lora.py').read_bytes()).hexdigest(),
          "frames": batch.frame_lengths.tolist(), "targets": batch.target_lengths.tolist(), "audio_samples": list(lengths),
          "checks": checks, "cases": cases,
          "encoder_padding_semantics": "Processor zeros masked mel frames; helper enforces zero input padding. HF subsampling masks every convolution, encoder attention masks both queries/keys, convolution masks invalid positions before depthwise temporal convolution, eval freezes BN/dropout. Trim only to max true T and use per-example masks/lengths.",
          "limitation": "Real trained0.6B GPU valid-frame/gradient parity and worst-case throughput/memory must pass coordinated profiler before a new run."}
(OUTPUT / "batched_lora_cpu_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({"passed":True,"checks":len(checks),"frames":report['frames'],"targets":report['targets'],"cases":cases},indent=2))
