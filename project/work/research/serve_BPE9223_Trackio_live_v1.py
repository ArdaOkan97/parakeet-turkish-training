"""CPU-only Trackio dashboard and read-only committed-training telemetry mirror.

Does not import the trainer, load checkpoints, inspect CUDA, or write run files.
All mutable dashboard data stays on the designated E drive.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import sqlite3
import threading
import time
import urllib.request

ROOT = Path(r"E:\gpu_training_scratchpad")
AREA = ROOT / "analysis/BPE9223_Trackio_live_v1"
RUN = ROOT / "runs/BPE9223_resident_RAM_20261018_01"
PROJECT = "parakeet-turkish"
NAME = RUN.name
BASE = "http://127.0.0.1:7860"
SOURCE_DB = Path(__file__).resolve().parents[2] / "outputs/parakeet_turkish/trackio/parakeet-turkish.db"
STATE = AREA / "mirror_state.json"
STATUS = AREA / "current_status.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def main():
    assert read(ROOT / "storage_config.json")["storage_root"] == str(ROOT)
    assert os.environ.get("TORCH_EXTENSIONS_DIR", "").startswith(str(ROOT))
    AREA.mkdir(parents=True, exist_ok=True)
    import msvcrt
    lock = (AREA / "service.lock").open("a+b")
    lock.seek(0)
    if lock.read(1) == b"":
        lock.write(b"1"); lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    import psutil
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

    def memory():
        vm, swap = psutil.virtual_memory(), psutil.swap_memory()
        return {"physical_available": vm.available, "commit_available": swap.free,
                "private_bytes": process.memory_info().private}

    def headroom():
        m = memory()
        if m["physical_available"] < 1_500_000_000 or m["commit_available"] < 2_000_000_000:
            raise RuntimeError("Dashboard headroom guard; training is untouched")
        if m["private_bytes"] > 1_500_000_000:
            raise RuntimeError("Dashboard private-memory guard; training is untouched")
        return m

    headroom()
    for key in ("TRACKIO_SPACE_ID", "TRACKIO_SERVER_URL", "TRACKIO_DATASET_ID",
                "TRACKIO_BUCKET_ID", "TRACKIO_WEBHOOK_URL"):
        os.environ.pop(key, None)
    os.environ.update(TRACKIO_DIR=str(AREA / "database"), CUDA_VISIBLE_DEVICES="-1",
                      GRADIO_ANALYTICS_ENABLED="False", HF_HUB_DISABLE_TELEMETRY="1",
                      TRACKIO_PLOT_ORDER="train/mean_objective,train/gradient_norm,throughput,progress,dev,retention")
    db = Path(os.environ["TRACKIO_DIR"]) / (PROJECT + ".db")
    db.parent.mkdir(parents=True, exist_ok=True)
    if not db.exists():
        # A consistent COPY preserves the old project history; the original is read-only.
        source = sqlite3.connect(SOURCE_DB.as_uri() + "?mode=ro", uri=True)
        temporary = db.with_suffix(".copying")
        destination = sqlite3.connect(temporary)
        before = {t: source.execute("SELECT COUNT(*) FROM " + t).fetchone()[0]
                  for t in ("metrics", "configs")}
        try:
            source.backup(destination, pages=1024, progress=lambda *unused: headroom())
            after = {t: destination.execute("SELECT COUNT(*) FROM " + t).fetchone()[0]
                     for t in before}
            assert before == after
        finally:
            destination.close(); source.close()
        os.replace(temporary, db)
        write(AREA / "original_history_copy_receipt.json",
              {"source": str(SOURCE_DB), "copy": str(db), "counts": before,
               "source_opened_read_only": True, "original_deleted_or_moved": False})
    token_file = AREA / "local_write_token.txt"
    if not token_file.exists():
        token_file.write_text(secrets.token_urlsafe(32), encoding="ascii")
    token = token_file.read_text(encoding="ascii").strip()
    os.environ["TRACKIO_WRITE_TOKEN"] = token

    import uvicorn
    from trackio.server import build_starlette_app_only
    app, _ = build_starlette_app_only(mcp_server=False)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=7860,
                                         access_log=False, log_level="warning"))
    identity = {"pid": process.pid, "create_time": process.create_time(),
                "source": str(Path(__file__).resolve()),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "project": PROJECT, "run_name": NAME, "run_id": NAME,
                "database": str(db), "dashboard_url": BASE + "/?project=" + PROJECT,
                "read_only_training_reader": True, "GPU_monitoring": False}
    write(AREA / "owner.json", identity)
    config = {"training_run": str(RUN), "planned_updates": 401072,
              "schedule": "two broad raw-label passes then one cleaner pass",
              "effective_batch": 16, "microbatches": "2 x 8", "vocabulary_size": 9223,
              "augmentation": "recorded noise and simulated RIR; weak Turkish student only",
              "logging": "external read-only journal/progress mirror; no training callback",
              "loss": "mean_objective (combined training objective, not isolated TDT loss)",
              "historical_metric_timestamps": "backfill arrival time; optimizer step is original",
              "DEV": "actual fixed350 DEV only, at original full-pass boundaries"}

    def api(name, kwargs):
        data = json.dumps({"kwargs": kwargs}, allow_nan=False).encode()
        request = urllib.request.Request(BASE + "/api/" + name, data=data,
                    headers={"Content-Type": "application/json", "X-Trackio-Write-Token": token})
        with urllib.request.urlopen(request, timeout=20) as response:
            answer = json.load(response)
        if answer.get("error"):
            raise RuntimeError(str(answer["error"]))
        return answer.get("data")

    def entry(step, metrics, key):
        return {"project": PROJECT, "run": NAME, "run_id": NAME, "step": step,
                "metrics": metrics, "config": config,
                "log_id": hashlib.sha256((NAME + ":" + key).encode()).hexdigest()}

    def mirror():
        state = read(STATE) if STATE.exists() else {"offset": 0, "last_update": 0}
        journal = RUN / "journals/commits_000000.jsonl"
        previous = None
        while not server.should_exit:
            try:
                headroom()
                logs = []
                offset = state["offset"]
                last = state["last_update"]
                with journal.open("rb") as handle:
                    if handle.seek(0, 2) < offset:
                        raise RuntimeError("Append-only training journal shortened")
                    handle.seek(offset)
                    while len(logs) < 64:
                        raw = handle.readline(4 * 1024 * 1024)
                        if not raw or not raw.endswith(b"\n"):
                            break
                        row = json.loads(raw)
                        if row.get("kind") == "ASR18_committed_training_update_v1":
                            step, native = row["optimizer_updates"], row["native"]
                            metrics = {"progress/optimizer_updates": step,
                                       "progress/broad_draws": row["broad_draws"],
                                       "progress/cleaner_draws": row["cleaner_draws"],
                                       "progress/replay_updates": row["replay_updates"]}
                            for source, target in (("mean_objective", "train/mean_objective"),
                                                   ("global_grad_norm", "train/gradient_norm")):
                                if finite(native.get(source)):
                                    metrics[target] = native[source]
                            logs.append(entry(step, metrics, "journal:" + hashlib.sha256(raw).hexdigest()))
                            last = max(last, step)
                        offset = handle.tell()
                if logs:
                    api("bulk_log", {"logs": logs, "hf_token": None})
                    state.update(offset=offset, last_update=last)
                    write(STATE, state)
                    continue
                p = read(RUN / "progress/latest.json")
                step = p["actual_optimizer_updates"]
                signature = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
                if state.get("progress_signature") != signature:
                    metrics = {"progress/completed_percent": 100 * step / 401072,
                               "throughput/audio_minutes_per_wall_minute_cumulative": p["audio_minutes_per_wall_minute"],
                               "progress/last_checkpoint_update": p["last_saved_updates"],
                               "progress/native_process_updates": p["native_updates_this_process"],
                               "progress/native_seconds": p["native_seconds"],
                               "progress/active_pipeline_seconds": p["active_pipeline_seconds"],
                               "progress/source_audio_hours": p["source_audio_seconds"] / 3600,
                               "progress/optimizer_poisoned": int(p["optimizer_poisoned"]),
                               "progress/schedule_poisoned": int(p["schedule_poisoned"])}
                    if previous and previous["producer"] == p["producer"]:
                        count = step - previous["actual_optimizer_updates"]
                        elapsed = p["active_pipeline_seconds"] - previous["active_pipeline_seconds"]
                        if count > 0 and elapsed > 0:
                            metrics["throughput/pipeline_seconds_per_update_recent"] = elapsed / count
                            metrics["throughput/audio_minutes_per_wall_minute_recent"] = (
                                p["source_audio_seconds"] - previous["source_audio_seconds"]) / elapsed
                    api("bulk_log", {"logs": [entry(step, metrics, "progress:" + signature)], "hf_token": None})
                    previous = p
                    state["progress_signature"] = signature
                    write(STATE, state)
                lifecycle = RUN / "lifecycle.json"
                if lifecycle.exists():
                    for index, boundary in enumerate(read(lifecycle)["completed_boundaries"]):
                        if index < state.get("DEV_count", 0):
                            continue
                        score = boundary["score"]
                        saved = read(Path(boundary["DEV"]["path"]))["optimizer_updates"]
                        metrics = {"dev/wer_" + lang + "_percent": 100 * count / score["reference_words"][lang]
                                   for lang, count in score["word_errors"].items()}
                        metrics["retention/all_six_passed"] = int(score["strict_all_six_passed"])
                        api("bulk_log", {"logs": [entry(saved, metrics, "DEV:" + str(index))], "hf_token": None})
                        state["DEV_count"] = index + 1
                        write(STATE, state)
                write(STATUS, {**identity, "status": "live", "updated_unix": time.time(),
                      "mirrored_update": state["last_update"], "journal_offset": state["offset"],
                      "training_progress_update": step, "native_producer": p["producer"],
                      "memory": memory(), "error": None})
                time.sleep(5)
            except Exception as error:
                write(STATUS, {**identity, "status": "retrying_logger_only", "updated_unix": time.time(),
                               "mirrored_update": state["last_update"], "error": str(error)[:500],
                               "memory": memory()})
                time.sleep(10)

    threading.Thread(target=mirror, daemon=True, name="read-only-training-mirror").start()
    server.run()


if __name__ == "__main__":
    main()
