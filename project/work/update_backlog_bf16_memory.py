"""Append read-only future storage assessment and independent profile evidence."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs/parakeet_turkish"
assessment = json.loads((OUTPUT / "bf16_encoder_storage_assessment.json").read_text(encoding="utf-8"))
audit = json.loads((OUTPUT / "tdt_profile_independent_audit.json").read_text(encoding="utf-8"))
assert assessment["eligible_frozen_weight_elements"] == 608533248
assert assessment["eligible_frozen_weight_tensors"] == 294
assert assessment["adoption_status"].startswith("Not adopted")
now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M %Z")
notes = f"{now}: Independent actual-profile audit now passes 98 CPU consistency checks, including all recorded code/selection/seed/input identities, full/reduced long-clip objectives, finite present gradients, and recomputed GPU short-clip aggregate head-gradient error. tdt_profile_independent_audit.json is the durable audit. Its scope is one accumulated Adam update per mode and short cached-clip head-gradient parity; it does not assert full raw-audio LoRA-gradient parity.\n\nFuture memory option, not adopted: CPU meta-architecture plus actual pinned safetensors headers identify 608,533,248 frozen encoder Linear/Conv weight elements in 294 tensors (Linear 532,676,608; Conv1d 75,718,656; Conv2d 137,984). Selectively storing those weights BF16 rather than FP32 would save 1,217,066,496 parameter-storage bytes = 1.217066 GB (1.133482 GiB). Encoder parameter storage would fall from 2.435518 GB to 1.218452 GB. Keep all encoder biases, normalization affine parameters and running buffers, LoRA A/B, encoder_projector, decoder/joint, gradients and optimizer FP32. This is an exact storage estimate, not a measured peak-memory reduction; activations, loss graphs, cast caches and optimizer storage also contribute.\n\nCurrent compute already uses CUDA BF16 autocast, making selective frozen-weight storage plausible, but numerical/dtype equivalence remains unverified. Plain forward outside autocast can fail with mixed dtypes, and blanket encoder.to(BF16) would also alter norms/buffers. Current LoRA merge helper creates output in the base dtype, so merging into a BF16 base could round away small trained deltas: finalization must rebuild the original fresh FP32 base and merge the FP32 adapter/head state there before standard-HF export.\n\nRequired next-run experiment: separate FP32-storage/selective-BF16-storage processes with identical actual adapter/head state, pinned base, teacher/cache, seed and BF16 autocast. On the same long TR/replay stress and short parity clips compare encoder outputs, TDT plus token/duration KL and feature-distillation losses and every non-negligible LoRA/head gradient (relative L2/cosine, finite/frozen checks). Measure the same four-microbatch balanced Adam update with synchronized time and allocated/reserved peaks including new Adam states, then compare exact decoded transcripts and fixed development WER/CER/gates. Use a fresh FP32-base merge/save/reload and rerun development gates. No heldout-driven tuning. Active lora_20261010_01 and its code/model remain unchanged. Detailed tensor list, assumptions and next-run checks are in bf16_encoder_storage_assessment.json."
subprocess.run([str(OUTPUT / "tools/backlog.exe"), "task", "edit", "ASR-10", "--append-notes", notes,
                "--add-ref", "tdt_profile_independent_audit.json", "--add-ref", "bf16_encoder_storage_assessment.json"],
               cwd=OUTPUT, check=True, capture_output=True)
subprocess.run([str(ROOT / "work/training-env/Scripts/python.exe"), str(OUTPUT / "refresh_backlog.py")], cwd=OUTPUT, check=True)
print("Future BF16 storage estimate recorded; active model unchanged.")
