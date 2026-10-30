"""One bounded CPU proof of exact scalar-chart sampling and incremental work."""
import ast
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import sys
import time
from typing import Any

import psutil

ROOT = Path("E:/gpu_training_scratchpad")
P = Path("C:/Users/ardao/Documents/Codex/2026-10-10/i-x20")
W = ROOT / "analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1"


def pin(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def write_new(path, value):
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    proc = psutil.Process()
    proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and sys.flags.utf8_mode == 1, "CPU-only UTF8")
    require(all(os.environ.get(k) == "1" for k in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")), "caps1")
    require(os.environ.get("TORCH_EXTENSIONS_DIR", "").lower().startswith(str(ROOT).lower().replace("/", "\\")), "Eenv")
    scope_path = Path(sys.argv[1])
    scope_raw = scope_path.read_bytes()
    scope = json.loads(scope_raw)
    require(scope["kind"] == "ASR18_incremental_scalar_projection_exact_ONE_CPU_fixture_proof_v1", "scopekind")
    require(pin(Path(__file__)) == scope["source"], "selfpin")
    out = W / "incremental_scalar_projection_CPU_fixture_actual_v1"
    out.mkdir(exist_ok=False)
    sys.path.insert(0, str(P / "outputs/parakeet_turkish"))
    from resource_watchdog_v2 import sample_memory
    initial = sample_memory(os.getpid(), "incremental_projection_initial")
    initial_pin = write_new(out / "initial_numeric_BEFORE_predicate.json", initial)
    started = time.monotonic()
    private_peak = proc.memory_info().private
    samples = 0
    result = None
    error = None
    try:
        sm = initial["system_memory"]
        require(initial["paired_sampling_within_one_second"], "pairednumeric")
        require(sm["commit_available_bytes"] >= 5130803840 and sm["physical_available_bytes"] >= 2380803840,
                "full native+72+metadata512 prospective")
        require(initial["worker_private_memory"]["private_bytes"] < 536870912, "self512MiB")
        require(shutil.disk_usage(ROOT).free >= 100001048576, "E100GB")
        def guard():
            nonlocal private_peak, samples
            require(time.monotonic() - started < 180, "wall180")
            private_peak = max(private_peak, proc.memory_info().private)
            require(private_peak < 536870912, "self512MiB")
            m = sample_memory(os.getpid(), "incremental_projection_runtime")["system_memory"]
            require(m["commit_available_bytes"] >= 2612368384 and m["physical_available_bytes"] >= 2112368384,
                    "original runtime floors+72+metadata512")
            require(shutil.disk_usage(ROOT).free >= 100001048576, "E100GB runtime")
            samples += 1
        guard()
        for name in ("helper", "storage_source"):
            require(pin(Path(scope[name]["path"])) == scope[name], name + "pin")
        sys.path.insert(0, str(P / "work/monitor-env/Lib/site-packages"))
        import orjson
        tree = ast.parse(Path(scope["storage_source"]["path"]).read_bytes())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SQLiteStorage")
        names = {"_is_scalar_metric_value", "_metric_row_signature", "_allocate_metric_group_budgets",
                 "_stable_subsample_metric_group", "_subsample_metric_rows", "_metric_rows_to_log_dicts",
                 "_fetch_metric_logs_with_cursor"}
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
        require(len(methods) == len(names), "all original AST methods")
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_METRIC_BUDGET_REFINEMENT_PASSES":
                        constants[target.id] = ast.literal_eval(node.value)
        namespace = dict(Any=Any, orjson=orjson, sqlite3=sqlite3,
                         deserialize_values=lambda value: value, **constants)
        original_cls = ast.ClassDef(name="SQLiteStorage", bases=[], keywords=[], body=methods, decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[original_cls], type_ignores=[]))
        exec(compile(module, scope["storage_source"]["path"], "exec"), namespace)
        storage = namespace["SQLiteStorage"]
        spec = importlib.util.spec_from_file_location("projection_proof_candidate", scope["helper"]["path"])
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        def connect(path):
            c = sqlite3.connect(path)
            c.row_factory = sqlite3.Row
            return c
        def create(path):
            c = connect(path)
            c.execute("CREATE TABLE metrics(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, timestamp TEXT, step INTEGER, metrics BLOB)")
            c.execute("CREATE INDEX metric_run_time ON metrics(run_id,timestamp)")
            c.commit()
            return c
        def insert(c, rows):
            c.executemany("INSERT INTO metrics(run_id,timestamp,step,metrics) VALUES(?,?,?,?)", rows)
            c.commit()
        def metrics(i):
            if i % 4000 == 0:
                return {"dev/tr": 24.0, "retention/es": 61, "retention/all_six": 0}
            if i % 40 == 0:
                return {"progress/u": i, "throughput/recent": 75.5}
            if i % 20 == 0:
                return {"health/alive": 1, "health/unsaved": i % 200}
            return {"train/loss": 2.0 + i / 1e6, "train/grad": .5, "progress/u": i,
                    "progress/broad": i * 14 - 3, "progress/replay": i, "progress/cleaner": 0}
        small = out / "fixture_small.db"
        c = create(small)
        rng = random.Random(9223)
        rows = []
        for i in range(220):
            payload = metrics(i)
            if i % 11 == 0:
                payload = {"bool_only": True, "text": "x"}
            if i % 17 == 0:
                payload["hist"] = {"_type": "trackio.histogram", "values": [0, 1], "counts": [2, 3]}
            stamp = f"{rng.randrange(70):012d}"
            rows.append(("target" if i % 9 else "other", stamp, i, orjson.dumps(payload)))
        insert(c, rows[:80])
        projection = helper.IncrementalScalarProjection(storage, orjson.loads, small, "target", guard)
        comparisons = 0
        for chunk in ([], rows[80:81], rows[81:102], rows[102:]):
            if chunk:
                insert(c, chunk)
            for budget in (1, 2, 3, 7, 30, 100, 1000):
                expected = storage._fetch_metric_logs_with_cursor(c.cursor(), ("run_id", "target"), budget, scalar_only=True)
                actual = projection.fetch(c.cursor(), ("run_id", "target"), budget, scalar_only=True)
                require(actual == expected, "exact sparse/duplicate/out-of-order sampling")
                comparisons += 1
        decoded_before = projection.stats["source_rows_decoded"]
        for unused in range(5):
            projection.fetch(c.cursor(), ("run_id", "target"), 30, scalar_only=True)
        require(projection.stats["source_rows_decoded"] == decoded_before, "no historical JSON reparse")
        sentinel = []
        real_original = projection.original_fetch
        projection.original_fetch = lambda *args, **kwargs: sentinel
        for identity, budget, scalar in [(("run_name", "target"), 30, True), (("run_id", "other"), 30, True),
                                         (("run_id", "target"), None, True), (("run_id", "target"), 0, True),
                                         (("run_id", "target"), 10001, True), (("run_id", "target"), 30, False)]:
            require(projection.fetch(c.cursor(), identity, budget, scalar_only=scalar) is sentinel, "exact original fallback")
        projection.original_fetch = real_original
        bounded = helper.IncrementalScalarProjection(storage, orjson.loads, small, "target", guard, max_rows=5)
        rejected = False
        try:
            bounded.fetch(c.cursor(), ("run_id", "target"), 2, scalar_only=True)
        except RuntimeError:
            rejected = True
        require(rejected and not bounded.all_keys and bounded.watermark == 0, "row bound denies and clears partialstate")
        require(projection.stats["out_of_order_inserts"] > 0, "out-of-order path exercised")
        c.close()
        large = out / "fixture_500000.db"
        c = create(large)
        for start in range(0, 500000, 1000):
            guard()
            insert(c, [("target", f"{i:012d}", i, orjson.dumps(metrics(i))) for i in range(start, start + 1000)])
        t0 = time.monotonic()
        expected = storage._fetch_metric_logs_with_cursor(c.cursor(), ("run_id", "target"), 3000, scalar_only=True)
        original_wall = time.monotonic() - t0
        guard()
        projection = helper.IncrementalScalarProjection(storage, orjson.loads, large, "target", guard)
        t0 = time.monotonic()
        actual = projection.fetch(c.cursor(), ("run_id", "target"), 3000, scalar_only=True)
        cold_wall = time.monotonic() - t0
        require(actual == expected, "exact500k original sampler")
        require(projection.stats["source_rows_decoded"] == 500000, "all500k decoded once")
        cold_decodes = projection.stats["source_rows_decoded"]
        t0 = time.monotonic()
        for unused in range(12):
            actual = projection.fetch(c.cursor(), ("run_id", "target"), 3000, scalar_only=True)
            require(actual == expected, "warm equality")
        warm12_wall = time.monotonic() - t0
        require(projection.stats["source_rows_decoded"] == cold_decodes, "12 polls zero historicalJSON reparses")
        insert(c, [("target", f"{i:012d}", i, orjson.dumps(metrics(i))) for i in range(500000, 500064)])
        actual = projection.fetch(c.cursor(), ("run_id", "target"), 3000, scalar_only=True)
        require(projection.stats["source_rows_decoded"] == 500064, "exact64 appended decodes")
        expected = storage._fetch_metric_logs_with_cursor(c.cursor(), ("run_id", "target"), 3000, scalar_only=True)
        require(actual == expected, "append500064 equality")
        def parallel_query(unused):
            con = connect(large)
            try:
                return projection.fetch(con.cursor(), ("run_id", "target"), 3000, scalar_only=True) == expected
            finally:
                con.close()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            require(all(pool.map(parallel_query, range(8))), "8 concurrent reader requests exact")
        guard()
        c.close()
        require("torch" not in sys.modules, "noTorch")
        result = dict(comparisons=28, fallback_cases=6, bounded_failure_passed=True,
                      scalar_row_signatures_sparse_DEV_histograms_bools_duplicate_and_out_of_order_timestamps=True,
                      original_sampler_AST_methods=sorted(names), original_constants=constants,
                      orjson_version=orjson.__version__, large_rows=500064, large_sample_points=len(actual),
                      cold_source_rows_decoded=cold_decodes, appended_source_rows_decoded=64,
                      warm_12_polls_historical_JSON_reparses=0, concurrent_requests=8,
                      requested_selected_rows_fetched=projection.stats["selected_rows_fetched"],
                      fixture_wall_seconds=dict(original_one_query=original_wall, cold_projection=cold_wall,
                                                warm_12_queries=warm12_wall),
                      stats=projection.stats,
                      limitations=["Fixture wall is bounded metadata CPU proof, not live throughput/durable training proof.",
                                   "Fast path assumes the single reader's append-only telemetry metrics; non-append DB actions require explicit invalidation.",
                                   "Only scalar sampled requests for exact current-run DB are optimized; original fallback for other calls."])
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
    record = dict(kind=scope["kind"], passed=error is None, scope=pin(scope_path), initial=initial_pin,
                  entry=dict(pid=proc.pid, create_time=proc.create_time(), ppid=proc.ppid()),
                  wall_seconds=time.monotonic() - started, private_peak_bytes=private_peak, guard_samples=samples,
                  result=result, error=error, existing_live_DB_or_training_read_or_written=False,
                  all_new_fixtures_on_E=True)
    result_pin = write_new(out / "result.json", record)
    print(json.dumps(dict(pin=result_pin, passed=record["passed"], entry=record["entry"], error=error)), flush=True)
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
