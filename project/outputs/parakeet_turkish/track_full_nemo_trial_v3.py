"""Mirror native JSON telemetry, DEV gates, and stock Lightning CSV losses.

Run in monitor-env only. CSV step49 denotes local update50; an explicit native warmstart offset is
added only to local training counters. DEV gates already use cumulative steps.
No interpolation or comparison with HF loss normalization.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import sys


def read(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def csv_training_metrics(path):
    """Return complete training rows at completed update CSV step+1.

    Installed logger_connector logs at epoch_loop._batches_that_stepped; that
    zero-based counter advances after training metrics are logged. NeMo's
    global_step metric is captured in training_step before the optimizer runs.
    Preserve both raw counters, and never apply this offset to DEV CSV rows.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return {}
    values = {}
    for row in csv.DictReader(io.StringIO(raw)):
        if None in row or any(value is None for value in row.values()):
            continue  # Header rewrite or incomplete live last row; retry next cycle.
        if not row.get("train_loss", "").strip():
            continue  # Validation-only rows never become training loss points.
        try:
            step_value = float(row["step"])
            loss = float(row["train_loss"])
            if not math.isfinite(step_value) or step_value < 0 or int(step_value) != step_value:
                continue
            if not math.isfinite(loss):
                continue
            step = int(step_value)
            official_step = float(row["global_step"]) if row.get("global_step", "").strip() else step_value
            if not math.isfinite(official_step) or official_step != step_value:
                continue
            metrics = {"train/loss": loss, "train/tdt_loss": loss,
                       "train/native_mean_volume_loss": loss,
                       "train/native_official_csv_step": step,
                       "train/native_completed_optimizer_update": step + 1}
            for source, target in (("learning_rate", "train/native_logged_learning_rate"),
                                   ("training_batch_wer", "train/native_training_batch_wer")):
                if row.get(source, "").strip():
                    number = float(row[source])
                    if math.isfinite(number):
                        metrics[target] = number
            values.setdefault(step + 1, {}).update(metrics)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return values


def native_step_offset(config, metadata=None):
    """Only reset-loop warmstarts add an offset; restored global loops do not."""
    warmstart = config.get("warmstart", {})
    declared = warmstart.get("source_completed_optimizer_updates")
    if declared is not None and warmstart.get("source_fit_loops_restored") is not False:
        raise ValueError("Additive source offset requires explicitly fresh fit loops")
    observed = (metadata or {}).get("source_optimizer_updates")
    if declared is not None and observed is not None and declared != observed:
        raise ValueError("Native warmstart/config metadata offsets disagree")
    value = declared if declared is not None else observed if observed is not None else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Native source offset must be a nonnegative integer")
    return value


def cumulative_training_values(local_values, offset):
    mapped = {}
    for local_step, metrics in local_values.items():
        if isinstance(local_step, bool) or not isinstance(local_step, int) or local_step < 0:
            raise ValueError("Native local update must be a nonnegative integer")
        cumulative = offset + local_step
        values = dict(metrics)
        if "train/native_completed_optimizer_update" in values:
            values["train/native_local_completed_optimizer_update"] = local_step
            values["train/native_completed_optimizer_update"] = cumulative
        values.update({"native/local_optimizer_step": local_step,
                       "native/cumulative_optimizer_step": cumulative,
                       "metadata/source_optimizer_updates": offset})
        mapped[cumulative] = values
    return mapped


def main():
    # Imports the existing monitor bridge only in the separate CPU process.
    import track_experiment as tracking
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--receipt-path", type=Path)
    args = parser.parse_args()
    if not args.run_id or any(not (c.isalnum() or c in "_-") for c in args.run_id):
        raise ValueError("Expected a safe native run ID")
    run_dir = (tracking.OUTPUT / "runs" / args.run_id).resolve()
    if not run_dir.is_relative_to((tracking.OUTPUT / "runs").resolve()):
        raise ValueError("Run must remain in the experiment workspace")
    state_path = tracking.ROOT / "work" / f"trackio_full_nemo_{args.run_id}_v3.json"
    work_dir = (tracking.ROOT / "work").resolve()
    stop_file = (args.stop_file or work_dir / f"trackio_full_nemo_{args.run_id}.stop").resolve()
    receipt_path = (args.receipt_path or work_dir / f"trackio_full_nemo_{args.run_id}_v3_pid_{os.getpid()}.json").resolve()
    if not stop_file.is_relative_to(work_dir) or not receipt_path.is_relative_to(work_dir):
        raise ValueError("Mirror control/evidence paths must remain under workspace/work")
    if receipt_path.exists():
        raise FileExistsError("Use a fresh PID receipt path")
    from atomic_io_retry_v1 import write_json
    receipt = {"kind": "root_owned_native_trackio_mirror_v3", "run_id": args.run_id,
               "pid": os.getpid(), "parent_pid": os.getppid(), "interpreter": sys.executable,
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "stop_file": str(stop_file), "state_path": str(state_path),
               "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
    write_json(receipt_path, receipt, retry_seconds=2.0)
    print(json.dumps({"mirror_pid_receipt": str(receipt_path), "pid": os.getpid(),
                      "stop_file": str(stop_file)}), flush=True)
    old_state = tracking.ROOT / "work" / f"trackio_full_nemo_{args.run_id}_v2.json"
    legacy_state = tracking.ROOT / "work" / f"trackio_full_nemo_{args.run_id}.json"
    state = read(state_path)
    if state is None:
        state = {}  # Defer any inherited deduplication until the offset is known.
    initialized = False
    terminal_reason = "one_snapshot_complete"
    offset = None
    try:
        while True:
            if stop_file.exists():
                terminal_reason = "owned_stop_file"
                return
            config = read(run_dir / "resolved_trial_config.json")
            if config and not initialized:
                metadata = read(run_dir / "native_training_telemetry_metadata.json", {})
                offset = native_step_offset(config, metadata)
                if not state and offset == 0:
                    state = read(old_state, read(legacy_state, {}))
                if state.get("_native_step_offset", offset) != offset:
                    raise ValueError("Existing mirror state uses a different cumulative offset")
                state["_native_step_offset"] = offset
                receipt["source_optimizer_updates"] = offset
                write_json(receipt_path, receipt, retry_seconds=2.0)
                tracking.trackio.init(project="parakeet-turkish", name=args.run_id,
                                      group="Full NeMo training", config=config,
                                      resume="allow", auto_log_gpu=False, auto_log_cpu=False)
                initialized = True
                print(f"Mirroring native telemetry and official CSV losses: {args.run_id}", flush=True)
            if initialized:
                values_by_step = csv_training_metrics(run_dir / "lightning" / "pilot" / "metrics.csv")
                telemetry_path = run_dir / "training_telemetry.jsonl"
                try:
                    telemetry_text = telemetry_path.read_text(encoding="utf-8")
                except OSError:
                    telemetry_text = ""
                for line in telemetry_text.splitlines():
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if row.get("run_id") != args.run_id:
                        raise ValueError("Telemetry run identity differs")
                    values_by_step.setdefault(row["step"], {}).update(row["metrics"])
                values_by_step = cumulative_training_values(values_by_step, offset)
                for path in run_dir.glob("development_gate_step_*.json"):
                    gate = read(path)
                    if not gate:
                        continue
                    if "source_optimizer_updates" in gate and gate["source_optimizer_updates"] != offset:
                        raise ValueError("Native DEV source offset disagrees")
                    if "local_step" in gate and gate["step"] != offset + gate["local_step"]:
                        raise ValueError("Native DEV cumulative/local step identity disagrees")
                    values = values_by_step.setdefault(gate["step"], {})
                    values.update({f"dev/wer_{lang}_percent": 100 * wer
                                   for lang, wer in gate["per_language_wer"].items()})
                    values["retention/gate_passed"] = int(gate["retention_gate_passed"])
                for step, values in sorted(values_by_step.items()):
                    prior = state.setdefault(str(step), {})
                    changed = {key: value for key, value in values.items() if prior.get(key) != value}
                    if changed:
                        tracking.trackio.log(changed, step=step)
                        prior.update(changed)
                write_json(state_path, state, retry_seconds=2.0)
                outcome = read(run_dir / "trial_outcome.json")
                if outcome:
                    tracking.trackio.log({
                        "native/completed_optimizer_updates": outcome["actual_completed_optimizer_updates"],
                        "native/trial_finished": 1,
                        "native/trial_failed": int(outcome["status"] == "failed"),
                        "native/stop_reason": tracking.trackio.Markdown(outcome.get("stop_reason") or outcome["status"]),
                    }, step=outcome["actual_completed_optimizer_updates"])
                    print(json.dumps({"mirror_finished": True, "run_id": args.run_id,
                                      "outcome": outcome["status"]}), flush=True)
                    terminal_reason = "trial_outcome_observed"
                    return
            if not args.follow:
                return
            time.sleep(5)
    except BaseException:
        terminal_reason = "mirror_error_or_interrupt"
        raise
    finally:
        if initialized:
            tracking.trackio.finish()
        receipt.update(status="ended", terminal_reason=terminal_reason,
                       ended_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(receipt_path, receipt, retry_seconds=2.0)


if __name__ == "__main__":
    main()
