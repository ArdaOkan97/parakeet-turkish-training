import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs/parakeet_turkish"
config = json.loads((OUTPUT / "run_config.json").read_text(encoding="utf-8"))
history = json.loads((OUTPUT / "training_history.json").read_text(encoding="utf-8"))
assert history
for record in history:
    assert all(math.isfinite(record[key]) for key in ("tdt_loss", "replay_kl", "gradient_norm", "elapsed_seconds", "peak_allocated_vram_gb"))
latest = history[-1]
time_string = datetime.now(timezone.utc).astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M %Z")
note = (
    f"{time_string}: Real-speech GPU training is active as {config['run_id']}: optimizer step {latest['step']}/{config['steps']}, "
    f"{config['trainable_parameters']:,} trainable parameters, frozen encoder and BatchNorm statistics. "
    f"All {len(history)} recorded steps have finite TDT loss, teacher KL and gradient norms. Latest TDT loss {latest['tdt_loss']:.5f}, "
    f"teacher KL {latest['replay_kl']:.8f}; peak allocated VRAM {latest['peak_allocated_vram_gb']:.6f} GB.\n\n"
    f"Actual run_config.json uses {config['turkish_clips_used']} cached Turkish clips and {config['replay_clips_used']} replay clips. "
    f"The {config['train_limit']}-clip Turkish pilot bound excluded 996 of the 2996 prepared clips before tokenizer checks; "
    f"lexical tokenizer encode/decode round-trip checks excluded {config['lexical_roundtrip_exclusions']['tr']} Turkish and "
    f"{config['lexical_roundtrip_exclusions']['replay']} replay references, chiefly digit/whitespace representations. "
    f"Replay new-token teacher-bound exclusions: {config['replay_new_token_exclusions']}. No further clips were dropped by the {config['max_tokens']}-token bound in this run. "
    "This prevents training labels whose normalized decoded words differ from the human reference.\n\n"
    f"Four accumulation microbatches alternate two Turkish and two replay clips; each replay draw first chooses uniformly among five preservation languages, "
    f"so the stream is 50/50 despite unequal cached corpus counts. Base Adam LR is {config['learning_rate']}, new-token row multiplier "
    f"{config['new_token_lr_multiplier']}, replay teacher-KL weight {config['replay_kl']}, validation every {config['eval_every']} steps. "
    "No retention-gated checkpoint or fine-tuned heldout result has been established yet; ASR-8 remains open."
)
snapshot = {"observed_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "CPU audit of saved active-run configuration and finite logged optimizer history; does not independently rerun GPU training", "run_id": config["run_id"], "latest_record": latest, "recorded_finite_steps": len(history), "config": config}
(OUTPUT / "training_progress_verification.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
subprocess.run([str(OUTPUT / "tools/backlog.exe"), "task", "edit", "ASR-5", "--check-ac", "2", "--append-notes", note,
                "--add-ref", "run_config.json", "--add-ref", "training_history.json", "--add-ref", "training_progress_verification.json"], cwd=OUTPUT, check=True, capture_output=True)
subprocess.run([str(ROOT / "work/training-env/Scripts/python.exe"), str(OUTPUT / "refresh_backlog.py")], cwd=OUTPUT, check=True)
print(json.dumps({"step": latest["step"], "finite_records": len(history), "peak_vram_gb": latest["peak_allocated_vram_gb"]}))
