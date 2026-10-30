"""Record independently verified completed LoRA development evaluations."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs/parakeet_turkish"
BINARY = OUTPUT / "tools/backlog.exe"
reports = [json.loads((OUTPUT / f"lora_20261010_01_step_{step}_verification.json").read_text(encoding="utf-8")) for step in (100, 200)]
assert all(report["passed"] and report["turkish_dev_improved"] and report["retention_gate_passed"] for report in reports)
torch.set_num_threads(1)
checkpoint = torch.load(ROOT / "work/lora_20261010_01_best.pt", map_location="cpu", weights_only=True)
assert checkpoint["run_id"] == "lora_20261010_01" and checkpoint["step"] == 200
assert len(checkpoint["weights"]) == 207 and sum(value.numel() for value in checkpoint["weights"].values()) == 19709068
assert all(torch.isfinite(value).all() for value in checkpoint["weights"].values())
now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M %Z")
notes = f"{now}: Completed development steps 100 and 200 independently verified on CPU: each has all 300 fixed paired FLEURS development recordings, matching original manifest/source/reference/revision/audio hashes, exact recomputed corpus WER/CER/raw-confidence/development-only temperature fits, recorded evaluator/run code hashes and every retention-gate calculation. Candidate folders contain development predictions only; no heldout candidate test was run. Contiguous recorded training history has finite losses/KLs/gradient norms.\n\nTurkish development base WER/CER = 100.0000%/51.3855%; step100 = 96.0000%/38.6647%; step200 = 92.8485%/34.4677%. Word errors are 825 -> 792 -> 766 over the same 825 reference words; saved scorer correct-word counts 41 -> 47 -> 81. Character errors are 3171 -> 2386 -> 2127 over 6171 characters. Thus spelling-level progress is larger than whole-word progress and Turkish remains highly inaccurate. Step100 saw 200 unique Turkish clips (9.23% of2166 usable pool), step200400 (18.47%); this is early training.\n\nRetention WER at step100 (en/es/it/fr/de) = 6.1244/3.6462/2.0725/6.5299/6.2749%; deltas from unchanged base = +.0957/+.0776/-.0864/+.1866/-.0996 percentage points. Step200 = 6.2201/3.8014/2.0725/5.8769/6.1753%; deltas = +.1914/+.2327/-.0864/-.4664/-.1992 points. Both pass every gate. Independent ceiling WERs are 6.6316/4.0687/2.6589/6.9776/7.0119%, using max(.5 percentage point,10% relative base WER).\n\nProvisional best raw adapter/head checkpoint is step200 with run ID, all207 finite trainable tensors/19,709,068 parameters verified on CPU. This satisfies the development selection/rehearsal criterion, but the run remains active and no usable merged standard-HF model/processor or final heldout ASR/confidence result exists yet. Exploratory sentence-cluster bootstrap estimates in the audit reports use the reused development selection set and are not heldout evidence. ASR-11 remains In Progress; finalization/ASR-8 remain open."
notes = notes.replace("of2166", "of 2166").replace("step200400", "step200 400").replace("all207", "all 207")
subprocess.run([str(BINARY), "task", "edit", "ASR-11", "--status", "In Progress", "--check-ac", "3", "--append-notes", notes,
                "--add-ref", "lora_20261010_01_step_100_verification.json", "--add-ref", "lora_20261010_01_step_200_verification.json",
                "--add-ref", "results/lora_20261010_01_step_100/summary.json", "--add-ref", "results/lora_20261010_01_step_200/summary.json"],
               cwd=OUTPUT, check=True, capture_output=True)
subprocess.run([str(ROOT / "work/training-env/Scripts/python.exe"), str(OUTPUT / "refresh_backlog.py")], cwd=OUTPUT, check=True)
print("Completed development steps100/200 and provisional selection verified; ASR-11 remains In Progress3/5.")
