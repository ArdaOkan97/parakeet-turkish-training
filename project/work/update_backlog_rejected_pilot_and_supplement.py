import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs/parakeet_turkish"
BINARY = OUTPUT / "tools/backlog.exe"
ARCHIVE = OUTPUT / "runs/pilot_20261010_01"
history = json.loads((ARCHIVE / "training_history.json").read_text(encoding="utf-8"))
selection = json.loads((ARCHIVE / "selection.json").read_text(encoding="utf-8"))
assert len(history) == 240 and history[-1]["step"] == 240 and selection["accepted"] is False
assert all(math.isfinite(record[key]) for record in history for key in ("tdt_loss", "replay_kl", "gradient_norm"))
evaluations = [record for record in history if "validation" in record]
assert len(evaluations) == 4
assert all(record["validation"]["turkish_wer"] >= 1.0 for record in evaluations)
supplement = json.loads((OUTPUT / "turkish_fleurs_supplement_audit.json").read_text(encoding="utf-8"))
assert supplement["passed"] and supplement["immutable_existing_files_verified"] == 33
now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M %Z")


def cli(*args):
    subprocess.run([str(BINARY), *args], cwd=OUTPUT, check=True, capture_output=True)


cli("task", "edit", "ASR-5", "--status", "In Progress", "--check-ac", "1",
    "--append-notes", f"{now}: Initial frozen-encoder decoder/head pilot pilot_20261010_01 completed all 240 real optimizer steps with finite logged losses/KLs/gradient norms. It was rejected: Turkish development WER at steps 60/120/180/240 was 100.7273%, 101.2121%, 100.1212% and 100.6061%, versus unchanged-base 100.0%; none improved Turkish. Step 60 passed preservation tolerances; later steps failed (final Italian +1.209 and German +2.291 percentage points). No candidate heldout test evaluation was run, and no recommended model/processor checkpoint was produced. A rejected head-state artifact exists in work/rejected_heads.pt for investigation. Logs/config/vocabulary/selection are archived under runs/pilot_20261010_01. Last recorded elapsed time 942.78 s, peak allocated VRAM 8.33245 GB. Real optimizer/checkpoint evidence criterion is complete, but successful selection/export and the overall fine-tune goal remain open.\n\nNext authorized experiment is encoder LoRA with separately validated memory optimization and optional longer official Turkish FLEURS train audio. Its final selection must continue to use development only; ASR-8 remains open.",
    "--add-ref", "runs/pilot_20261010_01/run_config.json", "--add-ref", "runs/pilot_20261010_01/training_history.json", "--add-ref", "runs/pilot_20261010_01/selection.json")

cli("task", "edit", "ASR-2", "--append-notes", f"{now}: Prepared the optional next-run Turkish FLEURS TRAIN supplement offline at the same pinned revision. Separate work/data/tr_fleurs_train.jsonl contains 395 stable-hash-selected clips, 1.50363 hours, 10.02-24.72 seconds (mean 13.704 s). All heldout global sentence IDs, exact normalized reference strings and source/WAV hashes are disjoint. Every supplemental WAV byte hash/header/duration and NFC reference is verified, and all 33 existing source-lock/data-summary/manifest/baseline files retained identical SHA256. Existing evaluation samples and optimizer manifests were not rewritten. Parallel train sentence IDs may overlap replay, as intended. Diagnostic current-tokenizer lexical checks accept 311 clips/1.1703 h within 180 tokens; 84 contain unrepresentable lexical round-trips. Human-reference manifest retains all 395 and the next trainer applies its own actual label filter. Supplemental audit/script are saved; no new optimizer or quality result is claimed.",
    "--add-ref", "supplement_turkish_fleurs.py", "--add-ref", "turkish_fleurs_supplement_audit.json", "--add-ref", "../../work/data/tr_fleurs_train.jsonl")

cli("task", "edit", "ASR-10", "--append-notes", f"{now}: CPU stress-example selection completed and all 3469 cached source IDs/token labels/lexical round-trips verified. memory_profile_examples.json gives largest T*(U+1) cases: Turkish T=105/U=67/states=7140, replay T=293/U=112/states=33109, plus longest-encoder alternatives and short old-ID-only replay parity T=47/U=10/states=517. One dense bf16 joint tensor would be about 111.73/518.09 MiB for the largest-lattice TR/replay examples, before loss/gradient/encoder activations. These are input-selection estimates, not a measured memory improvement. Actual memory profiling/parity deployment remains outstanding.",
    "--add-ref", "memory_profile_examples.json")

cli("task", "edit", "ASR-11", "--append-notes", f"{now}: The optional isolated longer Turkish FLEURS train supplement is verified and available for the fresh encoder-LoRA trainer; current-tokenizer usable subset is 311 clips/1.1703 h. The original 240-step frozen-encoder attempt was rejected without exposing its candidate to heldout test, so follow-on tuning remains development-only. A standalone encoder_lora.py implementation has passed the retention agent's CPU zero-init/gradient/merge/HF-export checks; the fresh actual-GPU encoder-adapter run is still pending. Frozen encoder feature caches cannot serve as student features once adapters learn; use raw-audio encoder forward/backward, with any original-encoder teacher features kept separate. Evaluate the merged export under actual bf16 runtime before final heldout claims.",
    "--add-ref", "turkish_fleurs_supplement_audit.json", "--add-ref", "encoder_lora.py")

subprocess.run([str(ROOT / "work/training-env/Scripts/python.exe"), str(OUTPUT / "refresh_backlog.py")], cwd=OUTPUT, check=True)
print("Rejected-pilot and verified supplement notes updated; ASR-5/8 remain open.")
