"""Bounded, opt-in HTTP prefix inspection. Never extracts the TSC archive.

Running --help is offline. A real invocation transfers at most the chosen
compressed prefix, then lists a bounded number of tar members and small text
or RIFF headers. A partial prefix is not a split, license or corpus audit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import pathlib
import struct
import tarfile
import time
import urllib.parse
import urllib.request

REVISION = "d79ea01392882322e8f46657cb635236af25b6d4"
ARCHIVE_SHA256 = "3feaba1f911542b587622d49c968bef971cfd8a4b3908d73090b8b914d2b694f"
ARCHIVE_BYTES = 21366175820
URL = f"https://huggingface.co/datasets/issai/Turkish_Speech_Corpus/resolve/{REVISION}/ISSAI_TSC_218.tar.gz"


class BudgetReached(RuntimeError):
    pass


class DecodedBudget:
    def __init__(self, stream, limit, deadline):
        self.stream, self.limit, self.deadline = stream, limit, deadline
        self.bytes_read = 0

    def read(self, size=-1):
        if time.monotonic() > self.deadline:
            raise BudgetReached("wall_time_budget")
        remaining = self.limit - self.bytes_read
        if remaining <= 0:
            raise BudgetReached("decompressed_prefix_budget")
        size = min(65536 if size < 0 else size, remaining)
        data = self.stream.read(size)
        self.bytes_read += len(data)
        return data


def riff_header(raw):
    result = {"bytes_inspected": len(raw)}
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return dict(result, recognized_RIFF_WAVE=False)
    result["recognized_RIFF_WAVE"] = True
    offset = 12
    while offset + 8 <= len(raw):
        tag, size = raw[offset:offset + 4], struct.unpack_from("<I", raw, offset + 4)[0]
        if tag == b"fmt " and size >= 16 and offset + 24 <= len(raw):
            fmt, channels, rate, _, align, bits = struct.unpack_from("<HHIIHH", raw, offset + 8)
            result.update(format_tag=fmt, channels=channels, sampling_rate=rate,
                          block_align=align, bits_per_sample=bits)
        if tag == b"data":
            result["declared_data_bytes"] = size
            if result.get("block_align") and result.get("sampling_rate"):
                result["declared_duration_seconds"] = size / result["block_align"] / result["sampling_rate"]
            break
        offset += 8 + size + (size % 2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--compressed-limit-mib", type=int, default=16, choices=range(1, 65))
    parser.add_argument("--decompressed-limit-mib", type=int, default=64, choices=range(1, 257))
    parser.add_argument("--max-members", type=int, default=256, choices=range(1, 1025))
    parser.add_argument("--max-seconds", type=int, default=60, choices=range(10, 121))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace prior evidence: {args.output}")
    started = time.monotonic()
    deadline = started + args.max_seconds
    limit = args.compressed_limit_mib * 1024 * 1024
    report = dict(schema=1, kind="TSC_bounded_archive_prefix_probe_v1", status="in_progress",
                  started_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  source_url=URL, source_revision=REVISION,
                  expected_full_archive_sha256=ARCHIVE_SHA256,
                  expected_full_archive_bytes=ARCHIVE_BYTES, full_archive_hash_verified=False,
                  extracts_files=False, decodes_audio=False, complete_archive_or_split_audit=False,
                  compressed_limit_bytes=limit, decompressed_limit_bytes=args.decompressed_limit_mib * 1024 * 1024,
                  max_members=args.max_members, max_seconds=args.max_seconds,
                  members=[], text_samples=[], WAV_header_samples=[])
    compressed = bytearray()
    decoded = None
    try:
        request = urllib.request.Request(URL, headers={"Range": f"bytes=0-{limit-1}",
                                                       "User-Agent": "TSC-bounded-metadata-probe/1"})
        with urllib.request.urlopen(request, timeout=10) as response:
            report["http_status"] = response.status
            report["final_response_host"] = urllib.parse.urlsplit(response.geturl()).hostname
            report["response_headers"] = {k: response.headers[k] for k in
                                          ("Content-Range", "Content-Length", "Accept-Ranges", "ETag")
                                          if response.headers.get(k) is not None}
            report["range_honored"] = response.status == 206
            if response.status == 206 and not response.headers.get("Content-Range", "").startswith("bytes 0-"):
                raise RuntimeError("Server returned a range not beginning at byte zero")
            while len(compressed) < limit:
                if time.monotonic() > deadline:
                    raise BudgetReached("wall_time_budget_during_transfer")
                chunk = response.read(min(65536, limit - len(compressed)))
                if not chunk:
                    break
                compressed.extend(chunk)
        report["compressed_prefix_sha256"] = hashlib.sha256(compressed).hexdigest()
        decoded = DecodedBudget(gzip.GzipFile(fileobj=io.BytesIO(compressed)),
                                args.decompressed_limit_mib * 1024 * 1024, deadline)
        with tarfile.open(fileobj=decoded, mode="r|") as archive:
            for _ in range(args.max_members):
                member = archive.next()
                if member is None:
                    report["stopped_reason"] = "tar_end_seen_in_prefix"
                    break
                path = pathlib.PurePosixPath(member.name)
                report["members"].append({"name": member.name, "size_bytes": member.size,
                                           "regular_file": member.isreg(),
                                           "unsafe_for_extraction": path.is_absolute() or ".." in path.parts})
                if not member.isreg():
                    continue
                lower = path.name.lower()
                is_license = lower.startswith(("license", "licence", "copying", "readme"))
                if is_license or (path.suffix.lower() == ".txt" and len(report["text_samples"]) < 8):
                    inspected = archive.extractfile(member).read(min(member.size, 65536 if is_license else 4096))
                    report["text_samples"].append({"name": member.name, "license_or_readme_candidate": is_license,
                                                   "member_size_bytes": member.size, "bytes_inspected": len(inspected),
                                                   "truncated": len(inspected) < member.size,
                                                   "text": inspected.decode("utf-8", errors="replace")})
                elif path.suffix.lower() == ".wav" and len(report["WAV_header_samples"]) < 4:
                    header = archive.extractfile(member).read(min(member.size, 4096))
                    report["WAV_header_samples"].append(dict(name=member.name, **riff_header(header)))
            else:
                report["stopped_reason"] = "member_count_budget"
        report["status"] = "bounded_prefix_observed"
    except (BudgetReached, EOFError, tarfile.ReadError) as error:
        report.update(status="bounded_prefix_observed" if report["members"] else "no_members_observed",
                      stopped_reason=f"{type(error).__name__}: {error}")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        report["compressed_bytes_received"] = len(compressed)
        report["decompressed_bytes_read"] = decoded.bytes_read if decoded else 0
        report["elapsed_seconds"] = time.monotonic() - started
        report["observed_split_directories"] = sorted({part for entry in report["members"]
                                                      for part in pathlib.PurePosixPath(entry["name"]).parts
                                                      if part in ("Train", "Dev", "Test")})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve()),
                      "members": len(report["members"]), "bytes_received": len(compressed)}))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
