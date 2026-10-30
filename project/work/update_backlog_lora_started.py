"""Record verified memory deployment and the newly started LoRA run via the CLI."""
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs/parakeet_turkish"
BINARY = OUTPUT / "tools/backlog.exe"


def read(name):
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def cli(*args):
    subprocess.run([str(BINARY), *args], cwd=OUTPUT, check=True, capture_output=True)


configuration = read("run_config.json")
data = read("lora_training_data_verification.json")
profiles = {name: read("tdt_profile_"+name+".json") for name in ("full", "reduced", "lora", "parity")}
assert data["passed"] and configuration["run_id"] == data["run_id"] == "lora_20261010_01"
for name, profile in profiles.items():
    assert profile["status"] == ("passed" if name == "parity" else "complete")
    assert profile["selection_sha256"] == profiles["full"]["selection_sha256"]
    assert profile["seed"] == configuration["seed"]
    for code_name, digest in profile["code_sha256"].items():
        assert hashlib.sha256((OUTPUT / code_name).read_bytes()).hexdigest() == digest
    if name != "parity":
        assert profile["measurement"]["gradient_audit"]["all_trainable_gradients_finite_and_present"]
parity = profiles["parity"]["measurement"]
assert parity["status"] == "passed" and parity["weighted_loss_absolute_difference"] < 1e-4
assert parity["all_head_gradient_relative_l2_difference"] < .02
assert parity["all_head_gradient_cosine"] > .999
cpu = read("efficient_tdt_cpu_parity.json")
assert cpu["passed"] and all(case["all_gradients_finite"] for case in cpu["cases"])
measurement = {name: profiles[name]["measurement"] for name in ("full", "reduced", "lora")}
peak_gb = {name: measurement[name]["memory_after_step"]["peak_allocated_bytes"]/1e9 for name in measurement}
reduction_percent = (1-peak_gb["reduced"]/peak_gb["full"])*100
assert measurement["lora"]["gradient_audit"]["adapter_nonzero_gradient_tensors"]["lora_B"] == 96
assert configuration["trainable_parameters"] == 19709068 and configuration["encoder_adapter_parameters"] == 1572864
assert configuration["lora_layout"]["adapted_layer_indices"] == list(range(24))
history = read("training_history.json")
assert history and history[-1]["step"] > 0
assert all(math.isfinite(row[key]) for row in history for key in ("tdt_loss", "replay_kl", "replay_duration_kl", "replay_feature_distillation", "gradient_norm"))
latest = history[-1]
now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M %Z")

cli("task", "edit", "ASR-10", "--status", "Done", "--check-ac", "1", "--check-ac", "2", "--check-ac", "3",
    "--append-notes", f"{now}: Actual RTX 5060 Ti CUDA profiling and BF16 loss/head-gradient parity passed before deployment. Three separate processes used the same pinned model, seed, two cached stress clips (TR T=105/U=67; replay T=293/U=112), original teacher, 2 microbatches and one accumulated Adam update. Peak allocated VRAM: full frozen-encoder objective {peak_gb['full']:.6f} GB, reduced frozen-encoder objective {peak_gb['reduced']:.6f} GB ({reduction_percent:.2f}% reduction), and reduced raw-audio all-24-layer rank-8 encoder LoRA {peak_gb['lora']:.6f} GB. One matched accumulated step took {measurement['full']['seconds_one_accumulated_adam_step']:.4f} / {measurement['reduced']['seconds_one_accumulated_adam_step']:.4f} / {measurement['lora']['seconds_one_accumulated_adam_step']:.4f} seconds, respectively. These are one-step synchronized allocator measurements including gradients/new Adam states, not a repeated throughput study or total driver memory. Full-vocabulary normalization/objective retained. GPU parity loss difference {parity['weighted_loss_absolute_difference']:.3g}, aggregate head-gradient relative L2 difference {parity['all_head_gradient_relative_l2_difference']:.6f}, cosine {parity['all_head_gradient_cosine']:.9f}; all trainable gradients finite/present. The independently reviewed CPU cases also pass. The verified chunk-size-16 memory path and activation checkpointing are now deployed only in new run lora_20261010_01, with fixed FLEURS benchmark protocol and development-only selection. This memory task is complete; model quality is a separate open task.",
    "--add-ref", "efficient_tdt.py", "--add-ref", "efficient_tdt_cpu_parity.json", "--add-ref", "efficient_tdt_independent_cpu_review.json",
    "--add-ref", "tdt_profile_full.json", "--add-ref", "tdt_profile_reduced.json", "--add-ref", "tdt_profile_lora.json", "--add-ref", "tdt_profile_parity.json")

counts = data["streams"]
cli("task", "edit", "ASR-11", "--status", "In Progress", "--check-ac", "1", "--check-ac", "2",
    "--append-notes", f"{now}: Actual new encoder-LoRA training started: run lora_20261010_01 plans 1200 optimizer steps, first development evaluation at 100 then every 200. All 24 encoder blocks have q/k/v/o rank-8, alpha-16 adapters (96 projections, scaling 2, dropout 0): 1,572,864 adapter parameters plus 18,136,204 decoder/joint parameters, 19,709,068 trainable total. Base encoder parameters and BatchNorm running statistics remain frozen; encoder eval mode disables dropout/LayerDrop, and activation checkpointing reduces encoder memory. Adapted student features come from raw audio; unchanged cached original encoder features condition the frozen teacher. Actual GPU profiler verified finite gradients and nonzero gradients in all 96 zero-initialized LoRA B tensors; A gradients initially zero as expected, and frozen base parameters have no gradients. Head LR 2e-5, LoRA LR 3e-4, new-row update multiplier 15, replay token-KL 5, duration-KL .5 and feature distillation 1; accumulation 4, full-vocabulary TDT time chunk 16, max labels 180, seed 20261010. Turkish/replay alternate at 50:50, with uniformly chosen retention language and shuffled cycles. At snapshot step {latest['step']} all recorded losses/KLs/gradient norms are finite and peak allocated memory is {latest['peak_allocated_vram_gb']:.6f} GB. No development selection/export/heldout quality result is claimed yet. Dev retention gate is each language <= base + max(.5 percentage point,10% relative base WER), and Turkish must improve; finalization/ASR-8 remain open.\n\nIndependent exact-HF-tokenizer CPU data verification checks all 3719 cached source IDs/labels/finite feature tensors and heldout provenance isolation. Actual usable training pool: CommonVoice {counts['turkish_commonvoice']['accepted_clips']} clips / {counts['turkish_commonvoice']['accepted_hours']:.6f} h plus Turkish FLEURS TRAIN {counts['turkish_fleurs_train']['accepted_clips']} clips / {counts['turkish_fleurs_train']['accepted_hours']:.6f} h = {data['totals']['turkish_clips']} clips / {data['totals']['turkish_hours']:.6f} h; replay {data['totals']['replay_clips']} clips / {data['totals']['replay_hours']:.6f} h. The actual processor excludes CV84/supplement145/replay447 lexical round-trips; zero empty/>180/new-teacher-ID exclusions. Earlier bare-tokenizer supplement diagnostic 311 clips/1.1703h is superseded by this actual-processor audit. These are unique usable pool hours, not cumulative repeated optimizer exposure.",
    "--add-ref", "train_lora.py", "--add-ref", "lora_training_data_verification.json", "--add-ref", "tdt_profile_lora.json", "--add-ref", "training_history.json")

cli("task", "edit", "ASR-5", "--status", "In Progress", "--append-notes",
    f"{now}: The next authorized actual-GPU attempt lora_20261010_01 is now active after verified memory reduction and all-layer encoder LoRA integration. Its exact current configuration and pool audit are tracked in ASR-11/run_config.json/lora_training_data_verification.json. Initial 240-step decoder-only attempt remains rejected and archived; the new run has not yet met successful Turkish-development/retention/export criteria. Overall fine-tuning and final heldout evaluation remain open. Current snapshot step {latest['step']} has finite recorded training metrics and {latest['peak_allocated_vram_gb']:.6f} GB peak allocated VRAM.",
    "--add-ref", "lora_training_data_verification.json")
cli("task", "edit", "ASR-2", "--append-notes",
    f"{now}: Active LoRA processor/cached-label audit supersedes the supplemental bare-tokenizer diagnostic: the next run actually uses 250 of 395 Turkish FLEURS TRAIN clips, .934600 h, after 145 lexical-roundtrip exclusions. CV uses 1916/2000 clips, 1.343860 h; combined Turkish 2166/2.278460 h. Replay 1553/4.559822 h. All 3719 exact cached labels and finite features verified against a reconstructed HF tokenizer whose semantic SHA matches vocabulary_audit.json; zero max-180-token/empty/new-teacher-ID exclusions. Supplemental manifest and original heldout data remain unchanged. Full source-ID and exclusion evidence is in lora_training_data_verification.json.",
    "--add-ref", "lora_training_data_verification.json")

subprocess.run([str(ROOT / "work/training-env/Scripts/python.exe"), str(OUTPUT / "refresh_backlog.py")], cwd=OUTPUT, check=True)
print("ASR-10 Done; ASR-11 In Progress; ASR-5/8 remain open; actual data counts recorded.")
