"""Read-only actual-architecture storage estimate; no weights/model allocated on GPU."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForTDT

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs/parakeet_turkish"
base_path = Path((ROOT / "work/base_model_path.txt").read_text().strip())
configuration = AutoConfig.from_pretrained(base_path, local_files_only=True)
torch.set_num_threads(1)
with torch.device("meta"):
    model = AutoModelForTDT.from_config(configuration)
assert all(parameter.device.type == "meta" for parameter in model.parameters())
headers = {}
files = []
for path in base_path.glob("*.safetensors"):
    with path.open("rb") as stream:
        size = int.from_bytes(stream.read(8), "little")
        header_bytes = stream.read(size)
        header = json.loads(header_bytes)
    for name, value in header.items():
        if name != "__metadata__":
            assert name not in headers
            headers[name] = value
    files.append({"path": str(path), "bytes": path.stat().st_size,
                  "header_sha256": hashlib.sha256(header_bytes).hexdigest()})
assert headers
selected, all_encoder, category = [], [], Counter()
for name, parameter in model.named_parameters():
    assert list(parameter.shape) == headers[name]["shape"]
    if not name.startswith("encoder."):
        continue
    all_encoder.append(parameter.numel())
    if name.endswith(".weight"):
        module_name = name[:-len(".weight")]
        module = model.get_submodule(module_name)
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            assert headers[name]["dtype"] == "F32"
            selected.append({"name": name, "type": module.__class__.__name__, "shape": list(parameter.shape), "elements": parameter.numel()})
            category[module.__class__.__name__] += parameter.numel()
selected_count = sum(row["elements"] for row in selected)
encoder_count = sum(all_encoder)
report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
          "scope": "Future-run read-only architecture/header estimate; no GPU run, parameter conversion or active-code mutation", "base_revision": base_path.name,
          "policy": "Only frozen model.encoder Linear/Conv weight tensors BF16; their biases, every normalization parameter/buffer, encoder LoRA A/B, encoder_projector, decoder, joint and optimizer remain FP32",
          "actual_checkpoint_headers": files, "encoder_parameter_elements": encoder_count,
          "eligible_frozen_weight_tensors": len(selected), "eligible_frozen_weight_elements": selected_count,
          "eligible_weight_elements_by_module_type": dict(category), "kept_encoder_parameter_elements": encoder_count-selected_count,
          "fp32_eligible_storage_bytes": selected_count*4, "bf16_eligible_storage_bytes": selected_count*2,
          "parameter_storage_saving_bytes": selected_count*2, "parameter_storage_saving_decimal_gb": selected_count*2/1e9,
          "parameter_storage_saving_gib": selected_count*2/1024**3,
          "encoder_parameter_storage_before_decimal_gb": encoder_count*4/1e9,
          "encoder_parameter_storage_after_decimal_gb": ((encoder_count-selected_count)*4+selected_count*2)/1e9,
          "selected_tensors": selected,
          "numerical_risks": ["Current CUDA autocast already rounds Linear/Conv operands to BF16, so explicit storage may remove FP32 master/cast-cache storage with little arithmetic change; this is an expectation, not measured equivalence.",
                              "A plain forward outside BF16 autocast may fail on mixed activation/weight dtypes; all inference, training, cache and export paths need explicit dtype coverage.",
                              "Normalization affine parameters and running statistics must remain FP32; avoid converting the entire encoder with .to(bfloat16).",
                              "The current merge helper creates merged Linear in base weight dtype. Merging into a BF16 base would round away some small LoRA deltas; reconstruct a fresh pinned FP32 base and merge the FP32 adapter state into it before standard-HF export.",
                              "Saving/exporting rounded BF16 base tensors changes checkpoint identity; a fresh FP32-base finalizer preserves original base weights and applies only trained adapter/head state.",
                              "Memory peaks also contain cast caches, activations, loss graphs, frozen teacher heads and FP32 optimizer states, so the exact observed peak reduction cannot be obtained by subtracting parameter bytes."],
          "required_next_run_gpu_validation": ["Use separate FP32-storage and selective-BF16-storage processes with the exact same pinned original base, adapter/head checkpoint, seed, FP32 trainables/Adam and CUDA BF16 autocast; keep encoder eval and identical teacher cache.",
                                                "On identical existing long TR and replay profile clips plus the short parity clip, compare encoder feature tensors, full TDT loss, token-KL, duration-KL, feature-distillation loss and every non-negligible LoRA/head gradient (relative L2/cosine); require all gradients finite and frozen parameters without gradients.",
                                                "Run the same four-microbatch balanced accumulated Adam update for each policy, recording synchronized step time and peak allocated/reserved memory including fresh optimizer-state allocation. Repeat matched measurements if deciding on throughput.",
                                                "Evaluate both policies on the same fixed development recordings using identical adapter/head states, checking exact transcripts and per-language WER/CER/retention gates; any discrepancy must be investigated before adoption. No new heldout tuning.",
                                                "Load trained weights into a fresh original FP32 base, merge adapters, save/load standard-HF model and processor, then rerun merged/reloaded development gates before any separate final test."],
          "adoption_status": "Not adopted; active lora_20261010_01 unchanged"}
destination = OUTPUT / "bf16_encoder_storage_assessment.json"
destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in report.items() if key in ("encoder_parameter_elements", "eligible_frozen_weight_tensors", "eligible_frozen_weight_elements", "eligible_weight_elements_by_module_type", "parameter_storage_saving_decimal_gb", "parameter_storage_saving_gib", "encoder_parameter_storage_before_decimal_gb", "encoder_parameter_storage_after_decimal_gb", "adoption_status")}, indent=2))
