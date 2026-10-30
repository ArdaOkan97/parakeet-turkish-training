"""Remove historical runs from the active dashboard; preserve a closed archive.

The trainer and original project history are never opened for writing.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import psutil

AREA = Path(r"E:\gpu_training_scratchpad\analysis\BPE9223_Trackio_live_v1")
RUN = "BPE9223_resident_RAM_20261018_01"


def main():
    assert os.environ["TORCH_EXTENSIONS_DIR"].startswith(r"E:\gpu_training_scratchpad")
    owner = json.loads((AREA / "owner.json").read_text())
    process = psutil.Process(owner["pid"])
    assert abs(process.create_time() - owner["create_time"]) < .001
    assert any("serve_BPE9223_Trackio_live_v5.py" in arg for arg in process.cmdline())
    database = Path(owner["database"]).resolve()
    assert database.parent == (AREA / "database").resolve()
    replacement = database.with_suffix(".current_only_building")
    assert not replacement.exists()
    archive = AREA / "archive" / datetime.datetime.now(datetime.timezone.utc).strftime("old_runs_%Y%m%dT%H%M%SZ")
    assert not archive.exists() and archive.resolve().is_relative_to(AREA.resolve())
    process.terminate()
    process.wait(timeout=10)

    source = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    target = sqlite3.connect(replacement)
    preserved = {}
    try:
        before_runs = source.execute("SELECT COUNT(DISTINCT run_id) FROM metrics").fetchone()[0]
        schemas = source.execute("SELECT type,name,sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'").fetchall()
        for kind, name, sql in schemas:
            if kind == "table":
                target.execute(sql)
        for kind, name, sql in schemas:
            if kind != "table":
                continue
            assert name.replace("_", "").isalnum()
            columns = [row[1] for row in source.execute('PRAGMA table_info("' + name + '")')]
            selector, args = None, ()
            if "run_id" in columns:
                selector, args = ' WHERE run_id=?', (RUN,)
            elif "producer_run_id" in columns:
                selector, args = ' WHERE producer_run_id=?', (RUN,)
            elif "run_name" in columns:
                selector, args = ' WHERE run_name=?', (RUN,)
            elif name == "project_metadata":
                selector = ""
            if selector is None:
                continue
            query = 'SELECT * FROM "' + name + '"' + selector
            cursor = source.execute(query, args)
            insert = 'INSERT INTO "' + name + '" VALUES (' + ','.join('?' for unused in columns) + ')'
            total = 0
            while True:
                batch = cursor.fetchmany(128)
                if not batch:
                    break
                target.executemany(insert, batch)
                total += len(batch)
            preserved[name] = total
        for kind, name, sql in schemas:
            if kind != "table":
                target.execute(sql)
        for pragma in ("user_version", "application_id"):
            value = source.execute("PRAGMA " + pragma).fetchone()[0]
            target.execute("PRAGMA " + pragma + "=" + str(value))
        target.commit()
        assert target.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert target.execute("SELECT COUNT(DISTINCT run_id) FROM metrics").fetchone()[0] == 1
        for table, count in preserved.items():
            assert target.execute('SELECT COUNT(*) FROM "' + table + '"').fetchone()[0] == count
        original_bounds = source.execute("SELECT COUNT(*),MIN(step),MAX(step) FROM metrics WHERE run_id=?", (RUN,)).fetchone()
        kept_bounds = target.execute("SELECT COUNT(*),MIN(step),MAX(step) FROM metrics WHERE run_id=?", (RUN,)).fetchone()
        assert original_bounds == kept_bounds
    finally:
        target.close()
        source.close()

    archive.mkdir(parents=True)
    archived = []
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(database) + suffix)
        if path.exists():
            destination = archive / path.name
            assert path.resolve().is_relative_to(AREA.resolve()) and destination.resolve().is_relative_to(AREA.resolve())
            path.rename(destination)
            archived.append(str(destination))
    replacement.rename(database)
    receipt = {"kind": "ASR18_actual_Trackio_active_dashboard_current_run_only_v1",
               "prior_logger_owner": owner, "old_logger_exit_observed": True,
               "historical_runs_removed_from_active_dashboard": before_runs - 1,
               "active_run": RUN, "active_database": str(database),
               "active_database_bytes": database.stat().st_size,
               "current_run_rows_min_max_step_preserved": list(kept_bounds),
               "preserved_table_rows": preserved, "archive": archived,
               "original_C_history_untouched": True, "training_untouched": True,
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (AREA / "old_runs_removed_active_database_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
