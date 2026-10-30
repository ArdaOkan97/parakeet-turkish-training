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


def fixture_bindings(path):
    """Only explicit typed owned E fixture paths; import alone touches no live paths."""
    global AREA,RUN,PROJECT,NAME,BASE,SOURCE_DB,STATE,STATUS
    binding=read(Path(path))
    assert binding["kind"]=="Trackio7_owned_E_fixture_bindings_v1"
    root=Path(binding["fixture_root"]).resolve()
    allowed=Path(r"E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1/next212000_failure_repair_v1").resolve()
    assert root.is_relative_to(allowed) and root!=allowed
    assert not root.exists() or root.is_dir()
    paths={k:Path(binding[k]).resolve()for k in ("area","run","source_db")}
    assert all(p.is_relative_to(root)for p in paths.values())
    assert binding["project"].startswith("fixture_") and binding["run_name"].startswith("fixture_")
    port=binding["port"];assert type(port)is int and 1024<=port<=65535 and port!=7860
    AREA=paths["area"];RUN=paths["run"];SOURCE_DB=paths["source_db"]
    PROJECT=binding["project"];NAME=binding["run_name"];BASE="http://127.0.0.1:"+str(port)
    STATE=AREA/"mirror_state.json";STATUS=AREA/"current_status.json"
    assert (AREA/"database"/(PROJECT+".db")).is_file()
    return port


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--fixture-bindings")
    args=parser.parse_args()
    port=fixture_bindings(args.fixture_bindings)if args.fixture_bindings else 7860
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

    import ctypes
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ("total_physical", "available_physical",
            "commit_limit", "available_commit", "total_virtual", "available_virtual", "extended")]

    def memory():
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError()
        return {"physical_available": status.available_physical,
                "commit_available": status.available_commit, "commit_limit": status.commit_limit,
                "private_bytes": process.memory_info().private}

    class DashboardMemoryLimit(RuntimeError):
        pass

    # Explicit extra reserve for bounded scalar-query metadata; original floors remain.
    QUERY_CACHE_RESERVE = 268_435_456
    API_BOOKKEEPING_RESERVE = 16_777_216 # Additional; never freed training credit.

    def headroom():
        m = memory()
        if m["physical_available"] < 1_500_000_000 + QUERY_CACHE_RESERVE + API_BOOKKEEPING_RESERVE or m["commit_available"] < 2_000_000_000 + QUERY_CACHE_RESERVE + API_BOOKKEEPING_RESERVE:
            raise DashboardMemoryLimit("Dashboard headroom guard; training is untouched")
        if m["private_bytes"] > 1_500_000_000:
            raise DashboardMemoryLimit("Dashboard private-memory guard; training is untouched")
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

    # Reviewed local helper changes chart reads only; all journal/logging behavior below is original.
    import importlib.util
    import orjson
    from trackio.sqlite_storage import SQLiteStorage
    helper = Path(r"E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1/incremental_scalar_dashboard_projection_v1.py")
    helper_raw = helper.read_bytes()
    assert len(helper_raw) == 6935
    assert hashlib.sha256(helper_raw).hexdigest() == "cb071def6f5c22b2dc98e88f6ca0036a641c8fbe516232ec5356772f0bf6c64c"
    helper_spec = importlib.util.spec_from_file_location("asr18_scalar_projection_v1", helper)
    helper_module = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper_module)
    scalar_projection = helper_module.install(SQLiteStorage, orjson.loads, db, NAME, headroom)

    import uvicorn
    from trackio.server import build_starlette_app_only
    app, _ = build_starlette_app_only(mcp_server=False)
    import trackio.asgi_app as official_asgi
    import trackio.server as official_server
    dispatch_path=Path(r"E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1/next212000_failure_repair_v1/Trackio_owned_bounded_sync_API_dispatch_v3.py")
    dispatch_raw=dispatch_path.read_bytes()
    assert len(dispatch_raw)==6038 and hashlib.sha256(dispatch_raw).hexdigest()=="9f478b8a732d63d710d8163eb9ae179e3fc57a1d8ade40ef3af130c7a0311198"
    dispatch_spec=importlib.util.spec_from_file_location("owned_Trackio_API_dispatch_v1",dispatch_path)
    dispatch_module=importlib.util.module_from_spec(dispatch_spec)
    dispatch_spec.loader.exec_module(dispatch_module)
    official_sources={'trackio.asgi_app': {'path': 'C:\\Users\\ardao\\Documents\\Codex\\2026-10-10\\i-x20\\work\\monitor-env\\Lib\\site-packages\\trackio\\asgi_app.py', 'bytes': 18786, 'sha256': '708a5a8426df2b4b993bcd327f8418166ced73cc077a878529fd2cd4e4860e6b'}, 'trackio.server': {'path': 'C:\\Users\\ardao\\Documents\\Codex\\2026-10-10\\i-x20\\work\\monitor-env\\Lib\\site-packages\\trackio\\server.py', 'bytes': 48851, 'sha256': '7d846d65fd7ba012e53ff211548dbddb6a626f1376f2b0841a8d1ff188c94e52'}}
    # Own every thread/allocation from the first dispatch construction onward.
    import sys
    api_dispatch=None
    mirror_thread=None
    server=None
    try:
        api_dispatch=dispatch_module.OwnedDispatch(official_asgi,official_server,official_sources,headroom)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
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
                  "DEV": "actual fixed350 DEV reports only: periodic4000 and full-pass boundaries; no fabricated WER"}

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

        mirror_activity={"HTTP_ready":False,"committed_HTTP_batch":False}
        pending_holder={"value":None}
        terminal_holder={"error":None}
        def local_status(state,pending,phase,error=None,thread_alive=True):
            # Local publication NEVER needs HTTP and tolerates early progress/memory errors.
            try:
                observed_memory=memory()
            except Exception as memory_error:
                observed_memory={"unavailable":type(memory_error).__name__}
            pending_metadata=None if pending is None else {k:v for k,v in pending.items()if k!="logs"}
            value={**identity,"status":phase,"updated_unix":time.time(),
                "mirror_thread_alive":thread_alive,"HTTP_ready":mirror_activity["HTTP_ready"],
                "reader_ready":mirror_activity["HTTP_ready"] and mirror_activity["committed_HTTP_batch"] and error is None and thread_alive,
                "mirrored_update":state.get("last_update",0),"journal_offset":state.get("offset",0),
                "journal_path":state.get("journal_path"),"pending_batch":pending_metadata,
                "error":None if error is None else (type(error).__name__+": "+str(error))[:500],
                "memory":observed_memory,"API_dispatch":dict(api_dispatch.stats)}
            try:write(STATUS,value)
            except Exception as publication_error:
                # Preserve a visible terminal error; no remote call or recursive publisher.
                print("LOCAL_STATUS_PUBLICATION_FAILED",type(publication_error).__name__,flush=True)
                raise
        def mirror_body(state):
            previous=None
            pending=None
            p=None
            step=None
            dispatch_module.wait_http_ready(BASE,headroom,lambda:server.should_exit,lambda:server.started,total_seconds=60)
            mirror_activity["HTTP_ready"]=True
            local_status(state,pending,"HTTP_ready")
            while not server.should_exit:
                try:
                    headroom()
                    if pending is not None:
                        api("bulk_log",{"logs":pending["logs"],"hf_token":None})
                        candidate=dict(state,journal_path=pending["journal_path"],
                            offset=pending["end_offset"],last_update=pending["last_update"])
                        write(STATE,candidate)
                        state.clear();state.update(candidate)
                        mirror_activity["committed_HTTP_batch"]=True
                        pending=None
                        pending_holder["value"]=None
                        local_status(state,pending,"catching_up")
                        continue
                    p = read(RUN / "progress/latest.json")
                    journal = Path(p["journal_prefix"]["path"]).resolve()
                    assert journal.is_relative_to(RUN.resolve())
                    logs = []
                    offset=state["offset"]if state.get("journal_path",str(journal))==str(journal)else 0
                    start_offset=offset
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
                        pending={"journal_path":str(journal),"start_offset":start_offset,
                            "end_offset":offset,"last_update":last,"count":len(logs),"logs":logs}
                        pending_holder["value"]=pending
                        local_status(state,pending,"pending_HTTP_batch")
                        continue # Next iteration sends exactly this retained batch.
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
                    evaluations = []
                    lifecycle = RUN / "lifecycle.json"
                    if lifecycle.exists():
                        evaluations.extend(read(lifecycle)["completed_boundaries"])
                    periodic = RUN / "periodic_DEV.json"
                    if periodic.exists():
                        evaluations.extend(read(periodic)["completed_evaluations"])
                    seen = state.setdefault("DEV_report_sha256", [])
                    for boundary in evaluations:
                        pin = boundary["DEV"]
                        if pin["sha256"] in seen:
                            continue
                        report_raw = Path(pin["path"]).read_bytes()
                        assert hashlib.sha256(report_raw).hexdigest() == pin["sha256"]
                        report = json.loads(report_raw)
                        assert report["passed"] and report["actual_native_updates_in_DEV"] == 0
                        assert report["all207_threeAdam_all4RNG_unchanged"] and not report["TEST_used"]
                        score = report["official_saved_text_score"]
                        assert score == boundary["score"] and score["full350_DEV_complete"]
                        saved = report["optimizer_updates"]
                        if "optimizer_updates" in boundary:
                            assert saved == boundary["optimizer_updates"]
                        metrics = {"dev/wer_" + lang + "_percent": 100 * count / score["reference_words"][lang]
                                   for lang, count in score["word_errors"].items()}
                        metrics.update({"retention/" + lang + "_word_errors": count
                                        for lang, count in score["word_errors"].items() if lang != "tr"})
                        metrics["retention/all_six_passed"] = int(score["strict_all_six_passed"])
                        api("bulk_log", {"logs": [entry(saved, metrics, "DEV:" + pin["sha256"])], "hf_token": None})
                        seen.append(pin["sha256"])
                        write(STATE, state)
                    producer = p["producer"]
                    try:
                        native = psutil.Process(producer["pid"])
                        alive = native.is_running() and abs(native.create_time() - producer["create_time"]) < .001
                    except psutil.NoSuchProcess:
                        alive = False
                    durable = read(RUN / "latest_checkpoint.json")["committed_updates"]
                    age = max(0, time.time() - (RUN / "progress/latest.json").stat().st_mtime)
                    health = {"health/training_worker_alive": int(alive),
                              "health/last_verified_checkpoint_update": durable,
                              "health/last_observed_journal_update": state["last_update"],
                              "health/progress_file_age_seconds": age,
                              "health/unsaved_updates": max(0, state["last_update"] - durable)}
                    api("bulk_log", {"logs": [entry(step, health, "health:" + str(int(time.time() // 10)))], "hf_token": None})
                    mirror_activity["committed_HTTP_batch"]=True
                    write(STATUS, {**identity, "status": "live", "updated_unix": time.time(),
                          "mirrored_update": state["last_update"], "journal_offset": state["offset"],
                          "training_progress_update": step, "native_producer": p["producer"],
                          "training_worker_alive": alive, "last_verified_checkpoint_update": durable,
                          "progress_age_seconds": age, "journal_path": str(journal),
                          "memory": memory(), "error": None,
                          "mirror_thread_alive":True,"HTTP_ready":mirror_activity["HTTP_ready"],
                          "reader_ready":mirror_activity["committed_HTTP_batch"],"pending_batch":None,
                          "API_dispatch":dict(api_dispatch.stats),
                          "scalar_query_projection": {"source_rows_decoded": scalar_projection.stats["source_rows_decoded"],
                              "selected_rows_fetched": scalar_projection.stats["selected_rows_fetched"],
                              "calls": scalar_projection.stats["calls"], "row_metadata_count": len(scalar_projection.all_keys),
                              "cache_reserve_bytes_above_original_floors": QUERY_CACHE_RESERVE}})
                    time.sleep(5)
                except Exception as error:
                    if isinstance(error,DashboardMemoryLimit)and terminal_holder["error"]is None:
                        terminal_holder["error"]=error
                    local_status(state,pending,"stopped_headroom"if isinstance(error,DashboardMemoryLimit)else"retrying_logger_only",error)
                    if isinstance(error,DashboardMemoryLimit):
                        server.should_exit=True
                        break
                    time.sleep(10)
            return pending

        def mirror():
            state={"offset":0,"last_update":0}
            pending=None
            try:
                if STATE.exists():state.update(read(STATE))
                pending=mirror_body(state)
            except BaseException as error:
                if terminal_holder["error"]is None:terminal_holder["error"]=error
            finally:
                terminal=terminal_holder["error"]
                phase=("stopped_headroom"if isinstance(terminal,DashboardMemoryLimit)else
                    "mirror_thread_failed"if terminal is not None else"mirror_thread_stopped")
                local_status(state,pending_holder["value"],phase,terminal,thread_alive=False)

        mirror_thread=threading.Thread(target=mirror,daemon=True,name="read-only-training-mirror")
        mirror_thread.start()
        server.run()
    finally:
        primary=sys.exc_info()[1]
        cleanup_errors=[]
        if server is not None:server.should_exit=True
        if mirror_thread is not None and mirror_thread.ident is not None:
            try:
                mirror_thread.join(timeout=35)
                if mirror_thread.is_alive():raise RuntimeError("Mirror thread did not join; closure UNKNOWN")
            except BaseException as cleanup:cleanup_errors.append(type(cleanup).__name__+": "+str(cleanup)[:500])
        if api_dispatch is not None:
            try:api_dispatch.close()
            except BaseException as cleanup:cleanup_errors.append(type(cleanup).__name__+": "+str(cleanup)[:500])
        if cleanup_errors:
            if primary is None:raise RuntimeError("; ".join(cleanup_errors))
            try:primary.add_note("Reader startup/teardown cleanup: "+repr(cleanup_errors))
            except BaseException:pass



if __name__ == "__main__":
    main()
