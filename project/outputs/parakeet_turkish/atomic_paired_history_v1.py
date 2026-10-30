"""Serialize identical history once, publish two original-format atomic files.

No training imports or history retention. The caller keeps its existing history
object. Only the disk copy has an additional buffer (at most 1 MiB). Publication
is ordered, not a two-file transaction: a failed second commit leaves the new
run-specific file and the old shared file, as two legacy writes would do.
Failed private files are retained for diagnosis, matching atomic_io_retry_v1.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import atomic_io_retry_v1 as atomic

KIND = "serialize_once_paired_history_atomic_publication_v1"
ATOMIC_HELPER_SHA256 = "2d3fe379d51021fd15679e562c71286f7f9501180d28dde77893e8f72c80a1b0"
MAX_COPY_CHUNK_BYTES = 1024 * 1024
FORMAT = {"encoding": "utf-8", "newline": "\\n", "indent": 2,
          "ensure_ascii": False, "allow_nan": False, "terminal_newlines": 1}
TIMERS = ("serialize_and_buffered_write_seconds", "first_flush_seconds",
          "first_fsync_seconds", "stream_copy_seconds", "second_flush_seconds",
          "second_fsync_seconds", "run_replace_seconds", "shared_replace_seconds",
          "pair_total_seconds")


class PairedHistoryPublisher:
    def __init__(self, run_path, shared_path, *, retry_seconds=30.0,
                 event=None, chunk_bytes=MAX_COPY_CHUNK_BYTES):
        self.run_path = Path(run_path).resolve()
        self.shared_path = Path(shared_path).resolve()
        if self.run_path == self.shared_path:
            raise ValueError("Paired history destinations must be distinct")
        if (self.run_path.exists() and self.shared_path.exists()
                and os.path.samefile(self.run_path, self.shared_path)):
            raise ValueError("Paired history destinations alias the same file")
        if not 0 < retry_seconds <= 60:
            raise ValueError("Atomic retries must be bounded to (0, 60] seconds")
        if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= MAX_COPY_CHUNK_BYTES:
            raise ValueError("Stream-copy buffer must be bounded to 1 MiB")
        if hashlib.sha256(Path(atomic.__file__).read_bytes()).hexdigest() != ATOMIC_HELPER_SHA256:
            raise ValueError("Frozen atomic I/O helper source changed")
        self.retry_seconds = retry_seconds
        self.event = event
        self.chunk_bytes = chunk_bytes
        self.last_report = None
        self._busy = False
        self._counters = {"publication_attempts": 0, "complete_pairs": 0,
                          "failed_pairs": 0, "serializations_completed": 0,
                          "copies_completed": 0, "run_commits": 0,
                          "shared_commits": 0, "serialized_bytes": 0,
                          "copied_bytes": 0, "replace_retries": 0,
                          **{key: 0.0 for key in TIMERS}}

    def counters_snapshot(self):
        return dict(self._counters)

    def publish(self, history):
        """Publish current value with the frozen indent2/UTF8/newline format.

        serialize_and_buffered_write_seconds includes json.dump's interleaved
        stream writes; it is not a claim of pure JSON CPU time. Timers overlap
        only with pair_total_seconds. Reports are returned, never written into
        scientific history/config/state by this utility.
        """
        if self._busy:
            raise RuntimeError("One publisher cannot execute overlapping calls")
        self._busy = True
        started = time.perf_counter()
        report = {"kind": KIND, "format": dict(FORMAT), "status": "in_progress",
                  "run_path": str(self.run_path), "shared_path": str(self.shared_path),
                  "copy_chunk_bytes": self.chunk_bytes, "serialized_bytes": 0,
                  "copied_bytes": 0, "run_committed": False,
                  "shared_committed": False, "replace_retries": 0,
                  "retry_events": [], **{key: 0.0 for key in TIMERS}}
        self.last_report = report
        self._counters["publication_attempts"] += 1
        first = second = None

        def retry_event(value):
            report["retry_events"].append(dict(value))
            if self.event is not None:
                self.event(value)

        def timed(key, function):
            before = time.perf_counter()
            try:
                return function()
            finally:
                report[key] += time.perf_counter() - before

        try:
            first = atomic.temporary_path(self.run_path)
            report["run_temporary"] = str(first)
            with first.open("w", encoding="utf-8", newline="\n") as stream:
                def serialize():
                    json.dump(history, stream, indent=2, ensure_ascii=False, allow_nan=False)
                    stream.write("\n")
                timed("serialize_and_buffered_write_seconds", serialize)
                timed("first_flush_seconds", stream.flush)
                timed("first_fsync_seconds", lambda: os.fsync(stream.fileno()))
            report["serialized_bytes"] = first.stat().st_size
            self._counters["serializations_completed"] += 1
            self._counters["serialized_bytes"] += report["serialized_bytes"]

            # Prepare the second private payload from the same immutable bytes
            # before committing either target. The only shared-file visibility
            # changes remain run FIRST, shared SECOND.
            second = atomic.temporary_path(self.shared_path)
            report["shared_temporary"] = str(second)
            with first.open("rb") as source, second.open("wb") as target:
                def copy():
                    while True:
                        block = source.read(self.chunk_bytes)
                        if not block:
                            break
                        target.write(block)
                        report["copied_bytes"] += len(block)
                timed("stream_copy_seconds", copy)
                timed("second_flush_seconds", target.flush)
                timed("second_fsync_seconds", lambda: os.fsync(target.fileno()))
            if report["copied_bytes"] != report["serialized_bytes"]:
                raise OSError("Incomplete second private history copy")
            self._counters["copies_completed"] += 1
            self._counters["copied_bytes"] += report["copied_bytes"]

            retries = timed("run_replace_seconds", lambda: atomic.retry_replace(
                first, self.run_path, retry_seconds=self.retry_seconds, event=retry_event))
            report["replace_retries"] += retries
            report["run_committed"] = True
            self._counters["run_commits"] += 1
            retries = timed("shared_replace_seconds", lambda: atomic.retry_replace(
                second, self.shared_path, retry_seconds=self.retry_seconds, event=retry_event))
            report["replace_retries"] += retries
            report["shared_committed"] = True
            self._counters["shared_commits"] += 1
            report["status"] = "complete"
            self._counters["complete_pairs"] += 1
            return report
        except BaseException as exception:
            report["status"] = "failed"
            report["error"] = {"type": type(exception).__name__, "message": str(exception)}
            self._counters["failed_pairs"] += 1
            raise
        finally:
            report["pair_total_seconds"] = time.perf_counter() - started
            self._counters["replace_retries"] += report["replace_retries"]
            for key in TIMERS:
                self._counters[key] += report[key]
            report["pending_temporaries"] = [str(p) for p in (first, second)
                                             if p is not None and p.exists()]
            self._busy = False


def write_history_pair(run_path, shared_path, history, **options):
    """Convenience for one call; retain a publisher for cumulative counters."""
    return PairedHistoryPublisher(run_path, shared_path, **options).publish(history)
