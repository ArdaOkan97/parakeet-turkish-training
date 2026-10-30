"""Operator attestation after consuming this agent's actual owned tool exit.

This utility never waits on, stops, launches or infers the exit of a worker.
The caller must supply the genuine consumed tool session/chunk. A worker's
published marker alone is insufficient and the owned process must be closed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import psutil
from ASR18_continuous_data_guard_v5 import ROOT, AREA
from cpu_preparation_guard_v2 import now
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json, write_json


def pin(path, include_mtime=False):
    path = Path(path).resolve()
    assert path.is_relative_to(ROOT.resolve()) or path.is_relative_to(PROJECT)
    stat = path.stat()
    assert stat.st_size < 32_000_000, 'Only source or bounded published JSON marker'
    value = dict(path=str(path), bytes=stat.st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (stat.st_size, stat.st_mtime_ns)
    if include_mtime:
        value['mtime_ns'] = stat.st_mtime_ns
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--owned-session', type=int, required=True)
    parser.add_argument('--native-PID', type=int, required=True)
    parser.add_argument('--actual-exit-code', type=int, choices=[0], required=True)
    parser.add_argument('--actual-tool-chunk', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--expected-source-SHA256', required=True)
    parser.add_argument('--marker', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--four-field-marker', action='store_true')
    args = parser.parse_args()
    assert args.owned_session > 0 and args.native_PID > 0 and args.actual_tool_chunk
    output = Path(args.output).resolve()
    assert output.is_relative_to(AREA.resolve()) and not output.exists()
    source = pin(args.source)
    assert source['sha256'] == args.expected_source_SHA256
    if psutil.pid_exists(args.native_PID):
        try:
            command = psutil.Process(args.native_PID).cmdline()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            raise AssertionError('Cannot verify actual closed owned source')
        assert not any(Path(args.source).name in argument for argument in command), 'Owned worker still live'
    marker = Path(args.marker).resolve()
    assert marker.is_relative_to(AREA.resolve())
    facts = read_json(marker)
    if 'actual_exit_code' in facts:
        assert type(facts['actual_exit_code']) is int and facts['actual_exit_code'] == 0
    marked_pid = facts.get('pid', facts.get('PID'))
    if marked_pid is not None:
        assert type(marked_pid) is int and marked_pid == args.native_PID
    value = dict(kind='ASR18_source_bound_actual_tool_exit_v1', owner='/root/cpu_data_preparation',
        observed_at_utc=now(), owned_session=args.owned_session, native_PID=args.native_PID,
        actual_exit_code=0, actual_tool_chunk_id=args.actual_tool_chunk,
        source_path=source['path'], source_SHA256=source['sha256'],
        published_marker=pin(marker, include_mtime=args.four_field_marker))
    if args.four_field_marker:
        value['source'] = pin(args.source, include_mtime=True)
    write_json(output, value)
    print(dict(actual_owned_tool_wrapper=pin(output), published_marker=value['published_marker']), flush=True)


if __name__ == '__main__':
    main()
