"""One-shot, stdlib-only chart repair. Append a distinct metric; preserve raw logs.

Uses the existing Trackio HTTP bulk API, never imports its SDK or starts a monitor.
Current-arm DB points and a bounded native stdout tail must match the declared
cosine schedule before publication. Repeated invocation skips existing points.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
RUN = "adaptive_20261012_TR_uniform_encoder_cosine_01"
METRIC = "train/applied_encoder_lr"
ORIGIN = 25109


def validate(step, value):
    n = step - ORIGIN
    if not 0 <= n <= 18129:
        raise ValueError("Point is outside the declared current arm")
    expected = 1e-4 if n <= 1 else 3e-5 + 7e-5 * .5 * (1 + math.cos(math.pi * (n - 1) / 18128))
    if not math.isfinite(value) or not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"Applied LR does not match the declared schedule at {step}")


def db_points(connection, run_id, metric):
    key = '$."' + metric + '"'
    rows = connection.execute(
        "SELECT step,json_extract(metrics,?) FROM metrics WHERE run_id=? "
        "AND step>=? AND json_type(metrics,?) IS NOT NULL ORDER BY id",
        (key, run_id, ORIGIN, key),
    )
    return {step: value for step, value in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    db = BASE / "trackio/parakeet-turkish.db"
    connection = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
    row = connection.execute("SELECT run_id FROM configs WHERE run_name=? ORDER BY id DESC LIMIT 1", (RUN,)).fetchone()
    if row is None:
        raise ValueError("Existing project run is missing")
    run_id = row[0]
    points = db_points(connection, run_id, "train/lora_lr")
    owner = json.loads((ROOT / "work/gpu_training_owner.json").read_text(encoding="utf-8"))
    if owner.get("target_run_id") != RUN:
        raise ValueError("Current lease belongs to a different run")
    loop = Path(owner["actual_controller_loop"]) / "loop_status.json"
    status = json.loads(loop.read_text(encoding="utf-8"))
    if status["run_id"] != RUN:
        raise ValueError("Native stdout is not bound to the current run")
    log = Path(status["paths"]["log"])
    with log.open("rb") as stream:
        size = log.stat().st_size
        offset = max(0, size - 2 * 1024 * 1024)
        stream.seek(offset)
        if offset:
            stream.readline()
        tail = stream.read(2 * 1024 * 1024)
    native_count = 0
    for line in tail.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "encoder_schedule_updates_completed" not in record or "loss" not in record:
            continue
        step = record["step"]
        if record.get("encoder_lr_schedule") != "cosine" or record.get("encoder_schedule_horizon") != 18129:
            raise ValueError("Native record uses a different policy")
        if record["encoder_schedule_updates_completed"] != step - ORIGIN or record.get("head_lr") != 1e-5:
            raise ValueError("Native record clock/head LR differs")
        if step in points and points[step] != record["lora_lr"]:
            raise ValueError("DB and actual native LR disagree")
        points[step] = record["lora_lr"]
        native_count += 1
    if not points or native_count == 0:
        raise ValueError("Actual current-arm native records are missing")
    for step, value in points.items():
        validate(step, value)
    existing = db_points(connection, run_id, METRIC)
    pending = []
    for step, value in sorted(points.items()):
        if step in existing:
            if existing[step] != value:
                raise ValueError("Existing clean metric disagrees; do not append conflicting points")
            continue
        identifier = hashlib.sha256(f"applied-lr-v1:{run_id}:{step}:{value!r}".encode()).hexdigest()
        pending.append({"project": "parakeet-turkish", "run": RUN, "run_id": run_id,
                        "step": step, "metrics": {METRIC: value}, "log_id": identifier})
    receipt = {"kind": "clean_applied_encoder_LR_view_v1", "run_id": RUN,
               "metric": METRIC, "source_step": ORIGIN, "last_step": max(points),
               "last_applied_lr": points[max(points)], "verified_points": len(points),
               "native_tail_records": native_count, "native_tail_sha256": hashlib.sha256(tail).hexdigest(),
               "new_points": len(pending), "applied": args.apply,
               "historical_metrics_or_training_source_modified": False,
               "SDK_or_resident_logger_started": False,
               "created_at_utc": datetime.now(timezone.utc).isoformat()}
    if args.apply:
        token = os.environ.get("TRACKIO_WRITE_TOKEN")
        if not token:
            # Use the existing owned local server's credential without printing it.
            server_log = (ROOT / "work/trackio_server_recovery.log").read_text(encoding="utf-8")
            matches = re.findall(r"write_token=([A-Za-z0-9_-]+)", server_log)
            if not matches:
                raise ValueError("Existing local server write credential is unavailable")
            token = matches[-1]
        for offset in range(0, len(pending), 256):
            payload = json.dumps({"kwargs": {"logs": pending[offset:offset + 256], "hf_token": None}}).encode()
            request = urllib.request.Request("http://127.0.0.1:7860/api/bulk_log", data=payload,
                                             headers={"Content-Type": "application/json", "X-Trackio-Write-Token": token}, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    result = json.load(response)
            except urllib.error.HTTPError as error:
                detail = json.loads(error.read().decode("utf-8"))
                raise ValueError(f"Trackio rejected metric publication: {detail.get('error', error.code)}") from None
            if result.get("error") is not None:
                raise ValueError(result["error"])
        after = db_points(connection, run_id, METRIC)
        if any(after.get(step) != value for step, value in points.items()):
            raise ValueError("Published clean metric does not match validated points")
        duplicates = connection.execute(
            "SELECT step,COUNT(*) FROM metrics WHERE run_id=? AND json_type(metrics,?) IS NOT NULL "
            "GROUP BY step HAVING COUNT(*)>1", (run_id, '$."' + METRIC + '"'),
        ).fetchall()
        if duplicates:
            raise ValueError("Clean plot has duplicated steps")
        receipt["published_verified"] = True
        (BASE / "applied_encoder_lr_view_actual_v1.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    connection.close()
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
