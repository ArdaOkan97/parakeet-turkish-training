"""Incremental scalar-chart projection for the single append-only ASR18 telemetry DB.

Inactive helper: install() is called only by a separately reviewed reader source.
The source metrics table is queried, never modified. Only row IDs, timestamps and
scalar-key signatures are retained; values are fetched for display-selected IDs.
"""
from bisect import insort
from pathlib import Path
import threading


class IncrementalScalarProjection:
    def __init__(self, storage, decode, db_path, run_id, guard,
                 max_rows=1_000_000, max_groups=256):
        self.storage = storage
        self.decode = decode
        self.db_path = Path(db_path).resolve()
        self.run_id = run_id
        self.guard = guard
        self.max_rows = max_rows
        self.max_groups = max_groups
        self.lock = threading.RLock()
        self.original_fetch = storage._fetch_metric_logs_with_cursor
        self.all_keys = []
        self.groups = {}
        self.watermark = 0
        self.file_identity = None
        self.stats = dict(source_rows_decoded=0, selected_rows_fetched=0,
                          calls=0, cache_resets=0, out_of_order_inserts=0)

    def _reset(self):
        self.all_keys.clear()
        self.groups.clear()
        self.watermark = 0
        self.stats["cache_resets"] += 1

    def _source_is_target(self, cursor, run_identity):
        if run_identity != ("run_id", self.run_id):
            return False
        databases = cursor.execute("PRAGMA database_list").fetchall()
        main = next((row[2] for row in databases if row[1] == "main"), None)
        return bool(main) and Path(main).resolve() == self.db_path

    def _append_key(self, keys, key):
        if not keys or key >= keys[-1]:
            keys.append(key)
        else:
            insort(keys, key)
            self.stats["out_of_order_inserts"] += 1

    def _refresh(self, cursor):
        self.guard()
        stat = self.db_path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_ctime_ns)
        # Windows ctime is creation time. POSIX ctime changes on writes, so use
        # device+inode there; this application is deployed only on Windows.
        identity = identity if __import__("os").name == "nt" else identity[:2]
        if self.file_identity is not None and identity != self.file_identity:
            self._reset()
        self.file_identity = identity
        upper = cursor.execute("SELECT max(id) FROM metrics").fetchone()[0] or 0
        if upper < self.watermark:
            self._reset()
        while self.watermark < upper:
            self.guard()
            rows = cursor.execute(
                "SELECT id, run_id, timestamp, metrics FROM metrics "
                "WHERE id > ? AND id <= ? ORDER BY id LIMIT 256",
                (self.watermark, upper),
            ).fetchall()
            if not rows:
                raise RuntimeError("Telemetry metrics changed during append-only projection")
            for row in rows:
                row_id, run_id, timestamp, raw = row
                if run_id == self.run_id:
                    if len(self.all_keys) >= self.max_rows:
                        raise RuntimeError("Scalar projection row bound reached")
                    metrics = self.decode(raw)
                    signature = self.storage._metric_row_signature(metrics, scalar_only=True)
                    if signature and signature not in self.groups:
                        if len(self.groups) >= self.max_groups:
                            raise RuntimeError("Scalar projection signature bound reached")
                        self.groups[signature] = []
                    key = (timestamp, row_id)
                    self._append_key(self.all_keys, key)
                    if signature:
                        self._append_key(self.groups[signature], key)
                    self.stats["source_rows_decoded"] += 1
                self.watermark = row_id
        self.guard()

    def fetch(self, cursor, run_identity, max_points, *, scalar_only=False):
        if (not scalar_only or max_points is None or max_points < 1
                or max_points > 10_000 or not self._source_is_target(cursor, run_identity)):
            return self.original_fetch(cursor, run_identity, max_points,
                                       scalar_only=scalar_only)
        with self.lock:
            try:
                self._refresh(cursor)
                self.stats["calls"] += 1
                # Preserve the original early return, including empty-scalar
                # rows when the complete run fits in the requested budget.
                if len(self.all_keys) <= max_points:
                    selected = self.all_keys
                else:
                    groups = sorted(self.groups.values(), key=lambda keys: keys[0])
                    total = sum(map(len, groups))
                    if total <= max_points:
                        selected = sorted(key for group in groups for key in group)
                    else:
                        budgets = self.storage._allocate_metric_group_budgets(
                            list(map(len, groups)), max_points)
                        selected = sorted(
                            key for group, budget in zip(groups, budgets, strict=True)
                            for key in self.storage._stable_subsample_metric_group(group, budget)
                        )
                by_id = {}
                ids = [key[1] for key in selected]
                for start in range(0, len(ids), 400):
                    self.guard()
                    batch = ids[start:start + 400]
                    placeholders = ",".join("?" for unused in batch)
                    for row in cursor.execute(
                        "SELECT id, timestamp, step, metrics FROM metrics WHERE id IN ("
                        + placeholders + ")", batch
                    ).fetchall():
                        by_id[row["id"]] = row
                if len(by_id) != len(ids):
                    raise RuntimeError("Selected telemetry rows disappeared")
                self.stats["selected_rows_fetched"] += len(ids)
                return self.storage._metric_rows_to_log_dicts(
                    [by_id[row_id] for row_id in ids], scalar_only=True)
            except BaseException:
                # A partial refresh may already have added IDs. Never retain
                # such state for a later request or fabricate successful data.
                self._reset()
                raise

    def invalidate(self):
        """Explicit invalidation for any separately authorized non-append DB action."""
        with self.lock:
            self._reset()


def install(storage, decode, db_path, run_id, guard):
    projection = IncrementalScalarProjection(storage, decode, db_path, run_id, guard)
    storage._fetch_metric_logs_with_cursor = staticmethod(projection.fetch)
    return projection
