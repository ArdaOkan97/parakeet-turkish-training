"""Record closed full-NeMo trials from their actual outcomes and telemetry."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    records = []
    for suffix in ("03", "04", "05"):
        run_id = "full_nemo_turkish_20261011_pilot_" + suffix
        directory = OUT / "runs" / run_id
        config = read(directory / "resolved_trial_config.json")
        outcome = read(directory / "trial_outcome.json")
        progress = read(directory / "native_training_progress.json")
        assert outcome["status"] == "failed"
        assert progress["step"] == outcome["actual_completed_optimizer_updates"]
        records.append({"run_id": run_id, "batch_size": config["model"]["train_ds"]["batch_size"],
                        "accumulation": config["trainer"]["accumulate_grad_batches"],
                        "allocator_cap_gb": config["allocator_cap_gb"],
                        "completed_updates": progress["step"],
                        "combined_audio_minutes_per_training_minute": progress["audio_minutes_per_wall_minute"],
                        "audio_seen_seconds": progress["audio_seen_seconds"],
                        "training_wall_seconds": progress["train_wall_seconds"],
                        "error": outcome["error"], "trained_DEV_point": False,
                        "checkpoint_saved": outcome["selected_checkpoint"] is not None})
    note = (datetime.now(timezone.utc).isoformat() + ": Actual full NeMo/Lightning follow-up results: " +
            "; ".join(f'{r["run_id"]}: B{r["batch_size"]} x accumulation{r["accumulation"]}, '
                      f'cap{r["allocator_cap_gb"]}GB, {r["completed_updates"]} completed updates, '
                      f'{r["combined_audio_minutes_per_training_minute"]:.2f} combined audio min/min before CUDA OOM'
                      for r in records) + ". "
            "The stock native TDT loss retains joint/gradient buffers from multiple singleton joint subbatches until backward. "
            "Pilot05 failed requesting another870MiB for label gradients after131 completed updates; "
            "the9GB process allocator cap was reached while physical GPU memory remained available. "
            "No trained DEV or selected checkpoint was produced by these failed pilots. "
            "Measured steady native speed is similar to the earlier ~93min/min short HF/native-loss benchmark, "
            "but initialization, objective, microbatch normalization and timing scopes differ, so no controlled speed-win claim. "
            "Repeated failures reduced usable iteration throughput. Next stock full-Trainer trial is B2 x accumulation8 "
            "with unchanged effective16 and resource guards plus periodic official Lightning recovery checkpoints. "
            "Prior HF checkpoint862 is preserved; best qualifying Turkish DEV remains64.9697%. "
            "The competitive Turkish quality goal is active. All actual native trials are mirrored in the same Trackio project.")
    for task in ("ASR-15", "ASR-16"):
        subprocess.run([str(OUT / "tools/backlog.exe"), "task", "edit", task, "--append-notes", note,
                        "--add-ref", "runs/full_nemo_turkish_20261011_pilot_05/trial_outcome.json"],
                       cwd=OUT, check=True, capture_output=True, text=True, encoding="utf-8")
    subprocess.run([str(ROOT / "work/training-env/Scripts/python.exe"), str(OUT / "refresh_backlog.py")],
                   cwd=ROOT, check=True)
    print(json.dumps({"backlog_updated": True, "closed_native_trials": records}))


if __name__ == "__main__":
    main()
