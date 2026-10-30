"""Install one pinned 704540-byte binary into an isolated E import directory.

No pip resolution, compiler, environment edits, Torch or dependency replacement.
Existing NumPy/SciPy are read from the explicitly selected project interpreter.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import urllib.request
import zipfile

from fastmss_cpu_guard_v1 import Guard, output_path

NAME = "pyroomacoustics-0.10.1-cp311-cp311-win_amd64.whl"
URL = "https://files.pythonhosted.org/packages/53/8c/aae7fa470d6dc87b3ee7d1c912102b0a7bc8d1a8fbee00c929d5596e7e77/" + NAME
SHA = "15cf8809d99449854d20f6ff343d0bb13bf9639db6efd0362a25866cc8cfae20"
SIZE = 704540


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = output_path(args.output)
    guard = Guard(wall_seconds=90)
    if sys.version_info[:2] != (3, 11) or sys.platform != "win32" or platform.machine().lower() not in ("amd64", "x86_64"):
        raise ValueError("Pinned wheel requires CPython3.11 WindowsAMD64")
    wheel = out / NAME
    if wheel.exists():
        raise ValueError("Preserved prior download exists; review/reuse explicitly rather than overwrite")
    started = time.monotonic()
    received = 0
    with urllib.request.urlopen(URL, timeout=20) as response, wheel.open("xb") as stream:
        while chunk := response.read(65536):
            guard.check()
            received += len(chunk)
            if received > SIZE:
                raise ValueError("Wheel download exceeded exact pinned bytes")
            stream.write(chunk)
            # This tiny side download is bounded to 1 MiB/s, independent of data worker.
            delay = received / (1024 * 1024) - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
    if received != SIZE or hashlib.sha256(wheel.read_bytes()).hexdigest() != SHA:
        raise ValueError("Pinned binary size/SHA256 mismatch")
    runtime = output_path(out / "isolated_imports")
    with zipfile.ZipFile(wheel) as archive:
        if sum(item.file_size for item in archive.infolist()) > 16 * 1024 * 1024:
            raise ValueError("Wheel unpacked-size bound exceeded")
        for item in archive.infolist():
            target = (runtime / item.filename).resolve()
            if not target.is_relative_to(runtime):
                raise ValueError("Wheel path traversal")
        archive.extractall(runtime)
    report = {"status": "isolated_binary_prepared", "name": NAME, "source": URL,
        "source_metadata": "https://pypi.org/pypi/pyroomacoustics/0.10.1/json",
        "size": received, "sha256": SHA, "runtime_path": str(runtime),
        "shared_environment_modified": False, "dependencies_replaced": False,
        "license_preserved_in_dist_info": True, "resources": guard.report()}
    (out / "isolated_runtime_receipt.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    guard.close()


if __name__ == "__main__":
    main()
