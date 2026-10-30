"""Lightweight native-Lightning training telemetry plus a separate CPU mirror.

The GPU callback imports no Trackio/Torch/Lightning: its owner passes the loaded
frameworks. Run --mirror with monitor-env to publish its JSONL and existing DEV
gate files into the local parakeet-turkish project.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def _write_progress(path, value):
    # Existing native Windows retry helper is stdlib-only and uses unique temps.
    from atomic_io_retry_v1 import write_json
    write_json(path, value, retry_seconds=2.0)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def training_metrics(record):
    values = {
        "resources/audio_minutes_per_wall_minute": record["audio_minutes_per_wall_minute"],
        "resources/step_audio_minutes_per_wall_minute": record["step_audio_minutes_per_wall_minute"],
        "resources/train_wall_seconds": record["train_wall_seconds"],
        "resources/step_train_wall_seconds": record["step_train_wall_seconds"],
        "resources/evaluation_wall_seconds": record["evaluation_wall_seconds"],
        "resources/current_allocated_vram_gb": record["current_allocated_vram_gb"],
        "resources/current_reserved_vram_gb": record["current_reserved_vram_gb"],
        "resources/peak_torch_allocated_vram_gb": record["peak_allocated_vram_gb_process"],
        "resources/peak_torch_reserved_vram_gb": record["peak_reserved_vram_gb_process"],
        "data/audio_seen_seconds": record["audio_seen_seconds"],
        "data/step_audio_seen_seconds": record["step_audio_seen_seconds"],
        "data/clips_seen": record["clips_seen"],
        "model/trainable_parameters": record["trainable_parameters"],
        "model/trainable_tensors": record["trainable_tensors"],
    }
    values.update({f"train/lr_group_{index}": value for index, value in enumerate(record["learning_rates"])})
    return values


def make_native_training_telemetry_callback(pl, torch, output_dir, run_id, sample_rate=16000):
    """Return a real Lightning Callback; append it AFTER the resource callback.

    Each native batch is (audio, audio_lengths, transcripts, transcript_lengths).
    Timing for steady microbatches spans previous synchronized batch-end to
    current synchronized batch-end, including intervening loading/H2D and
    callback overhead. Startup and validation reset this anchor. The first
    batch of each training segment is timed start->end, excluding its already
    loaded/transferred input. This exclusion is counted explicitly; the rate
    is not described as complete process end-to-end throughput. GPU peaks are
    process peaks and may include validation; global peak counters are untouched.
    """
    if sample_rate != 16000 or not run_id or any(not(c.isalnum() or c in "_-") for c in run_id):
        raise ValueError("Expected 16kHz audio and a safe native run ID.")
    output_dir = Path(output_dir).resolve()

    class NativeTrainingTelemetry(pl.Callback):
        def __init__(self):
            super().__init__()
            self.path = output_dir / "training_telemetry.jsonl"
            self.progress_path = output_dir / "native_training_progress.json"
            if self.path.exists() or self.progress_path.exists():
                raise FileExistsError("Native telemetry requires fresh paths.")
            output_dir.mkdir(parents=True, exist_ok=True)
            self.path.touch(exist_ok=False)
            self.anchor = None
            self.micro_start = None
            self.micro_audio = 0.0
            self.micro_clips = 0
            self.pending_audio = self.pending_wall = 0.0
            self.pending_clips = self.pending_microbatches = self.pending_loading_exclusions = 0
            self.audio = self.wall = self.evaluation_wall = 0.0
            self.clips = self.microbatches = self.loading_exclusions = 0
            self.last_step = 0
            self.val_start = None
            self.trainable_parameters = self.trainable_tensors = None
            self.metadata_written = False

        def _sync(self):
            torch.cuda.synchronize()

        def on_fit_start(self, trainer, model):
            self.trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.trainable_tensors = sum(p.requires_grad for p in model.parameters())
            self.last_step = int(trainer.global_step)
            self.anchor = None
            _write_progress(output_dir / "native_training_telemetry_metadata.json", {
                "kind": "native_lightning_training_telemetry_v1", "run_id": run_id,
                "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "sample_rate": sample_rate, "trainable_parameters": self.trainable_parameters,
                "trainable_tensors": self.trainable_tensors,
                "timing_scope": "steady previous synchronized microbatch-end to current end; first batch start-to-end",
                "startup_and_validation_excluded": True,
                "first_batch_loading_after_reset_excluded_and_counted": True,
                "model_load_excluded": True, "process_end_to_end_claimed": False,
                "gpu_peaks_scope": "process including startup/validation; no peak-counter reset",
                "Trackio_imported_in_gpu_callback": False,
            })
            self.metadata_written = True

        def on_train_epoch_start(self, trainer, model):
            # Epoch boundary/loading is included when continuing without DEV.
            pass

        def on_train_batch_start(self, trainer, model, batch, batch_idx):
            if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                raise ValueError("Expected the native four-tensor audio/text batch.")
            lengths = batch[1].detach().cpu().tolist()
            if not lengths or any(isinstance(x, bool) or int(x) != x or x <= 0 for x in lengths):
                raise ValueError("Invalid native audio lengths.")
            self.micro_audio = sum(lengths) / sample_rate
            self.micro_clips = len(lengths)
            self._sync()
            self.micro_start = time.perf_counter()
            if self.anchor is None:
                self.anchor = self.micro_start
                self.pending_loading_exclusions += 1

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            if self.micro_start is None:
                raise RuntimeError("Native batch-end lacks its timing/audio start.")
            self._sync()
            ended = time.perf_counter()
            duration = ended - self.anchor
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("Invalid synchronized training interval.")
            self.pending_audio += self.micro_audio
            self.pending_wall += duration
            self.pending_clips += self.micro_clips
            self.pending_microbatches += 1
            self.anchor, self.micro_start = ended, None
            step = int(trainer.global_step)
            if step == self.last_step:
                return
            if step != self.last_step + 1:
                raise ValueError("Optimizer steps must be observed contiguously.")
            self.audio += self.pending_audio
            self.wall += self.pending_wall
            self.clips += self.pending_clips
            self.microbatches += self.pending_microbatches
            self.loading_exclusions += self.pending_loading_exclusions
            record = {
                "kind": "native_lightning_training_update_v1", "run_id": run_id, "step": step,
                "logged_at_utc": _utc(), "step_audio_seen_seconds": self.pending_audio,
                "step_train_wall_seconds": self.pending_wall,
                "step_audio_minutes_per_wall_minute": self.pending_audio / self.pending_wall,
                "step_clips_seen": self.pending_clips, "step_microbatches": self.pending_microbatches,
                "first_batch_loading_exclusions_this_update": self.pending_loading_exclusions,
                "audio_seen_seconds": self.audio, "train_wall_seconds": self.wall,
                "audio_minutes_per_wall_minute": self.audio / self.wall,
                "clips_seen": self.clips, "microbatches": self.microbatches,
                "first_batch_loading_exclusions": self.loading_exclusions,
                "evaluation_wall_seconds": self.evaluation_wall,
                "trainable_parameters": self.trainable_parameters, "trainable_tensors": self.trainable_tensors,
                "learning_rates": [float(group["lr"]) for optimizer in trainer.optimizers for group in optimizer.param_groups],
                "current_allocated_vram_gb": torch.cuda.memory_allocated() / 1e9,
                "current_reserved_vram_gb": torch.cuda.memory_reserved() / 1e9,
                "peak_allocated_vram_gb_process": torch.cuda.max_memory_allocated() / 1e9,
                "peak_reserved_vram_gb_process": torch.cuda.max_memory_reserved() / 1e9,
                "timing_scope": "training intervals; steady loading included; first loading after reset excluded",
                "validation_and_startup_time_in_training_rate": False,
            }
            record["metrics"] = training_metrics(record)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            _write_progress(self.progress_path, record)
            print(json.dumps({"native_training": {key: record[key] for key in (
                "step", "audio_minutes_per_wall_minute", "step_audio_minutes_per_wall_minute", "audio_seen_seconds",
                "train_wall_seconds", "current_allocated_vram_gb", "current_reserved_vram_gb", "learning_rates")}}), flush=True)
            self.last_step = step
            self.pending_audio = self.pending_wall = 0.0
            self.pending_clips = self.pending_microbatches = self.pending_loading_exclusions = 0

        def on_validation_epoch_start(self, trainer, model):
            self._sync()
            self.val_start = time.perf_counter()
            self.anchor = None

        def on_validation_epoch_end(self, trainer, model):
            self._sync()
            if self.val_start is not None:
                self.evaluation_wall += time.perf_counter() - self.val_start
            self.val_start = self.anchor = None

        def on_fit_end(self, trainer, model):
            # A partial accumulation is explicitly excluded from optimizer-update
            # totals; preserve its observed exposure instead of inventing a step.
            _write_progress(output_dir / "native_training_telemetry_outcome.json", {
                "run_id": run_id, "last_logged_optimizer_step": self.last_step,
                "audio_seen_seconds": self.audio, "train_wall_seconds": self.wall,
                "evaluation_wall_seconds": self.evaluation_wall,
                "pending_audio_seconds_without_observed_optimizer_update": self.pending_audio,
                "pending_wall_seconds_without_observed_optimizer_update": self.pending_wall,
                "first_batch_loading_exclusions": self.loading_exclusions, "finished_at_utc": _utc(),
            })

    return NativeTrainingTelemetry()


def mirror(run_dir, run_id, project, once=False, poll_seconds=5):
    # This function belongs in the separate lightweight monitor interpreter.
    os.environ["TRACKIO_DIR"] = str(HERE / "trackio")
    os.environ["HF_HOME"] = str(ROOT / "work" / "monitor-hf-cache")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    for name in ("TRACKIO_SPACE_ID", "TRACKIO_SERVER_URL", "TRACKIO_DATASET_ID", "TRACKIO_BUCKET_ID", "TRACKIO_WEBHOOK_URL"):
        os.environ.pop(name, None)
    import trackio
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_relative_to(HERE / "runs") or not run_id or any(not(c.isalnum() or c in "_-") for c in run_id):
        raise ValueError("Mirror a safe native run under the experiment runs directory.")
    state_path = ROOT / "work" / f"trackio_native_mirror_{run_id}.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"run_id": run_id, "steps": [], "development_files": {}}
    trackio.init(project=project, name=run_id, group="Native NeMo Lightning", resume="allow",
                 config={"trainer": "official NeMo/Lightning", "telemetry_source": str(run_dir)},
                 auto_log_gpu=False, auto_log_cpu=False)
    try:
        while True:
            telemetry = run_dir / "training_telemetry.jsonl"
            records = []
            if telemetry.exists():
                for line in telemetry.read_text(encoding="utf-8").splitlines():
                    try:
                        records.append(json.loads(line))
                    except ValueError:
                        continue  # A concurrently appended partial line is retried.
            for record in records:
                if record["run_id"] != run_id:
                    raise ValueError("Telemetry run identity differs.")
                if record["step"] not in state["steps"]:
                    trackio.log(training_metrics(record), step=record["step"])
                    state["steps"].append(record["step"])
            for path in sorted(run_dir.glob("development_gate_step_*.json")):
                try:
                    raw = path.read_bytes()
                    gate = json.loads(raw)
                except (OSError, ValueError):
                    continue
                digest = hashlib.sha256(raw).hexdigest()
                if state["development_files"].get(path.name) != digest:
                    metrics = {f"dev/wer_{lang}_percent": 100 * value for lang, value in gate["per_language_wer"].items()}
                    metrics["retention/all_dev_gates_passed"] = int(gate["retention_gate_passed"])
                    trackio.log(metrics, step=gate["step"])
                    state["development_files"][path.name] = digest
            _write_progress(state_path, state)
            if once or (run_dir / "trial_outcome.json").exists():
                break
            time.sleep(poll_seconds)
    finally:
        trackio.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror", action="store_true", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project", default="parakeet-turkish")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    mirror(args.run_dir, args.run_id, args.project, args.once)
