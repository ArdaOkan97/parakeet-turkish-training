import json
import re
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs/parakeet_turkish"
BINARY = OUTPUT / "tools/backlog.exe"
now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M %Z")


def cli(*args):
    result = subprocess.run([str(BINARY), *args], cwd=OUTPUT, check=True, capture_output=True, text=True, encoding="utf-8")
    return result.stdout


# Preserve existing task identities on retries.
tasks = json.loads(cli("task", "list", "--json"))["tasks"]
existing = {task["title"]: task["id"] for task in tasks}


def create(title, description, criteria, labels, dependencies, refs):
    if title in existing:
        return existing[title]
    args = ["task", "create", title, "--description", description, "--status", "To Do", "--priority", "high", "--labels", labels]
    for criterion in criteria:
        args += ["--ac", criterion]
    for dependency in dependencies:
        args += ["--dep", dependency]
    for reference in refs:
        args += ["--ref", reference]
    cli(*args)
    updated = json.loads(cli("task", "list", "--json"))["tasks"]
    task_id = next(task["id"] for task in updated if task["title"] == title)
    existing[title] = task_id
    return task_id


# Complete the independently audited initial English comparison.
comparison = json.loads((OUTPUT / "english_confidence_comparison.json").read_text(encoding="utf-8"))
independent = json.loads((OUTPUT / "english_confidence_independent_audit.json").read_text(encoding="utf-8"))
assert independent["passed"]
timing = {"scope": "One complete English split wall time including audio/file IO and confidence collection; excludes model loading and model fingerprint hashing. Descriptive single pass, not a repeated pure-generation speed benchmark.", "models": {}}
for model, log_name in [("v2", "base_v2_benchmark.log"), ("v3", "base_v3_benchmark_fixed.log")]:
    log = (ROOT / "work" / log_name).read_text(encoding="utf-8")
    values = {}
    for split, count in [("validation", 50), ("test", 100)]:
        matches = re.findall(rf"base_{model} {split} en: {count}/{count} ([0-9.]+)s", log)
        assert matches
        values[split] = {"utterances": count, "wall_seconds": float(matches[-1])}
    timing["models"][model] = values
(OUTPUT / "english_benchmark_timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
cli("task", "edit", "ASR-7", "--status", "Done", "--check-ac", "1", "--check-ac", "2", "--check-ac", "3",
    "--append-notes", f"{now}: The pinned v2 and original v3 models completed exactly the same 50 English development and 100 English test recordings. Independent audit passed 30 confidence/identity/calibration/bootstrap checks, maximum numerical difference 5.11e-15. Test WER v2 4.8298% versus v3 5.5198%; paired recording bootstrap WER delta +0.690 percentage points with 95% CI [-0.046, +1.478]. Among 2047 reference words correct in both, mean raw confidence is 99.7285% versus 99.5237%; v3-minus-v2 delta -0.2048 percentage points, 95% CI [-0.3426, -0.0627]. Conditional nonblank confidence reduces the gap and its interval crosses zero. Separate dev-only temperatures are 2.0 for v2 and 1.5 for v3; calibration improves each model's heldout ECE/Brier without changing transcripts.\n\nSingle complete split wall times (including audio/file IO and confidence, excluding loading/fingerprint): v2 development 11.8 s/test 22.6 s; v3 development 11.3 s/test 22.1 s. These are descriptive single passes, not statistically controlled generation speed claims. Raw predictions, fingerprints and exact settings are retained. The static figure was generated with the bundled ReportLab LinePlot/Poppler fallback because existing runtimes lack Matplotlib; no packages or inference code changed. The final PNG was visually inspected and all labels are readable.\n\nClean baseline comparison is complete. Selected fine-tune confidence/retention and any padding/noise/gain perturbation trial remain separate outstanding work.",
    "--final-summary", "Completed matched original-v2/v3 English ASR and actual-path confidence comparison, independently verified by 30 checks. The small mean confidence difference is about -0.205 percentage points on the same correct words; dev-fit temperatures improve heldout calibration. Raw outputs, uncertainty, exact split settings, single-pass timings and a verified static PNG/PDF are saved.",
    "--add-ref", "english_confidence_comparison.json", "--add-ref", "english_confidence_independent_audit.json",
    "--add-ref", "english_benchmark_timing.json", "--add-ref", "english_confidence_figure.png", "--add-ref", "english_confidence_figure.pdf")

# The requested Trackio dashboard is already active and receives existing logs.
db_path = OUTPUT / "trackio/parakeet-turkish.db"
db = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)
steps = [row[0] for row in db.execute("SELECT DISTINCT step FROM metrics ORDER BY step") if row[0] is not None]
assert steps and steps[0] == 0 and steps == list(range(steps[-1] + 1)), steps
metric_rows = db.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
system_rows = db.execute("SELECT COUNT(*) FROM system_metrics").fetchone()[0]
config_rows = db.execute("SELECT COUNT(*) FROM configs").fetchone()[0]
db.close()
tracking = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "CPU read-only local SQLite audit; live HTTP service verified by the parent agent", "dashboard_url": "http://127.0.0.1:7860/?project=parakeet-turkish", "database": str(db_path), "metric_rows": metric_rows, "system_rows": system_rows, "run_configs": config_rows, "contiguous_steps_from_0_through": steps[-1], "passed": True}
(OUTPUT / "trackio_verification.json").write_text(json.dumps(tracking, indent=2), encoding="utf-8")
track_id = create("Run a live local Trackio training dashboard", "Attach a separate Trackio process to the active experiment's structured JSON history, backfill every optimizer step and baseline/development scores, and keep the project/database local. Provide reproducible restart commands without restarting or modifying training.",
                  ["Show the local parakeet-turkish project at port 7860", "Backfill and follow optimizer, development/retention and system metrics into a persistent local database", "Document scripts, dependency version, database path and restart commands"],
                  "tracking,dashboard", ["ASR-1"], ["track_experiment.py", "tracking-requirements.txt", "trackio/parakeet-turkish.db", "DASHBOARDS.txt", "trackio_verification.json"])
cli("task", "edit", track_id, "--status", "Done", "--check-ac", "1", "--check-ac", "2", "--check-ac", "3",
    "--append-notes", f"{now}: Trackio 0.42.0 runs independently in work/monitor-env and serves the requested local dashboard at http://127.0.0.1:7860/?project=parakeet-turkish. track_experiment.py mirrors the already-running job's JSON output, backfills baseline at step 0, updates development WER/retention gates and records GPU/system metrics. No active training/inference code was modified. A read-only SQLite audit observed {metric_rows} metric rows, {system_rows} system rows and {config_rows} run config(s), with contiguous recorded steps 0 through {steps[-1]}. Database: trackio/parakeet-turkish.db. The tracker explicitly uses local TRACKIO_DIR, disables remote targets and analytics, and keeps monitor HF cache under work. Restart commands and metric units are documented in DASHBOARDS.txt.",
    "--final-summary", "Live local Trackio dashboard is running, backfilled and independently audited. Persistent SQLite data, pinned tracker dependency, independent mirror script and complete restart commands are saved.")

memory_id = create("Reduce TDT training memory before the next adaptation run", "Prepare a standalone memory-efficient TDT joint/loss path while the current frozen-encoder pilot continues. Require small CPU loss/gradient parity with the pinned reference before switching any future GPU run, then compare actual peak VRAM and throughput on matched inputs.",
                   ["Pass CPU loss and parameter-gradient parity against the existing pinned TDT reference", "Pass matched actual-GPU finite-gradient/update checks and record before/after peak VRAM and timing", "Use the verified memory path only for a new experiment with logged configuration and unchanged benchmark protocol"],
                   "memory,training", ["ASR-1"], ["train.py", "runtime.py", "run_config.json", "training_history.json"])
cli("task", "edit", memory_id, "--append-notes", f"{now}: User requested memory optimization next. The current pilot continues unchanged. The parent agent is preparing standalone prototype scripts and CPU parity checks; no reduced-memory path has yet been deployed or measured on a new GPU experiment. Status remains planned until parity evidence is recorded.")

lora_id = create("Fine-tune Turkish with encoder LoRA and multilingual retention", "Run the explicitly requested next adaptation with trainable LoRA adapters in the acoustic encoder, using the verified memory path, conservative decoder/new-token updates and multilingual rehearsal. Encoder adapters require raw-audio forward/backward; frozen encoder-feature caches cannot train them. Keep the fixed FLEURS development/test sets and original baseline for retention comparison.",
                 ["Document actual acoustic-encoder LoRA target modules, adapter rank and trainable parameter count", "Verify nonzero finite encoder-adapter gradients from raw audio while base encoder weights/statistics remain controlled", "Use balanced Turkish/replay training and validation-only Turkish improvement plus per-language retention gates", "Save/load a usable model/processor and adapter checkpoint with reproducible configuration", "Complete matched fixed heldout ASR/confidence and paired uncertainty before claiming Turkish improvement or retention"],
                 "lora,encoder,turkish,retention", [memory_id, "ASR-2", "ASR-3", "ASR-4"], ["run_config.json", "prepare_data.py", "benchmark.py", "metrics.py"])
cli("task", "edit", lora_id, "--append-notes", f"{now}: User explicitly requested encoder LoRA after discussing memory optimization. Adapter-target and pretraining-language research is underway with the retention agent. The current frozen-encoder pilot continues; no encoder-LoRA optimizer run, checkpoint or quality result exists yet. Final retention/selected-model evaluation in ASR-8 remains open.")
subprocess.run([str(ROOT / "work/training-env/Scripts/python.exe"), str(OUTPUT / "refresh_backlog.py")], cwd=OUTPUT, check=True)
print(json.dumps({"comparison": "ASR-7 Done", "trackio": track_id, "memory": memory_id, "encoder_lora": lora_id, "trackio_last_step": steps[-1]}, indent=2))
