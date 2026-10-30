"""Bounded isolated FastMSS RIR generation; no meetings, speech or Torch import.

Read the pinned upstream rirsimulator.py directly. Full recipes/sim.py needs
Unix fork; this small serial upstream component does not. A CPU-only pyroom-
acoustics binary may be supplied via an isolated E-only --runtime directory.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from fastmss_cpu_guard_v1 import Guard, ROOT, output_path

PIN = "e481ceac821ef0cb31033bf43b45922e377f528d"
SOURCE_SHA = "05f7ef9c12ede249401ca26335b083188e587cb40fff372e489820b820e9455c"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--runtime")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rooms", type=int, default=1)
    parser.add_argument("--positions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    if not 1 <= args.rooms <= 4 or not 1 <= args.positions <= 4:
        raise ValueError("Bounded pilot is at most 4 rooms x 4 positions")
    out = output_path(args.output)
    guard = Guard(wall_seconds=90)
    if args.runtime:
        runtime = Path(args.runtime).resolve(strict=True)
        if not runtime.is_relative_to(ROOT.resolve(strict=True)):
            raise ValueError("Isolated runtime must be on E")
        sys.path.insert(0, str(runtime))
    upstream = Path(args.upstream_root).resolve(strict=True)
    if not upstream.is_relative_to(ROOT.resolve(strict=True)):
        raise ValueError("Pinned upstream source must be on E")
    source = upstream / "fastmss/rirsimulator.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("Upstream RIR source differs from reviewed pinned revision")
    if importlib.util.find_spec("pyroomacoustics") is None:
        report = {"status": "dependency_unavailable_no_generation", "missing": "pyroomacoustics",
                  "native_GPU_used": False, "generated_assets": 0, "resources": guard.report()}
        (out / "RIR_generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
        guard.close()
        return 78
    import numpy as np
    import soundfile as sf
    import pyroomacoustics
    # Load only the pinned RIR module; no fastmss package/meeting dependencies.
    spec = importlib.util.spec_from_file_location("_pinned_fastmss_rir", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = dict(output_dir=str(out), samplerate=16000, room_sz=[4.0, 5.0],
        room_ceiling=[3.0, 3.5], rt60=[0.15, 0.25], delta_dist=0.5,
        use_rand_ism=False, rand_disp=0.0, save_anechoic=False,
        mic_type="single", n_pos_rirs=args.positions)
    np.random.seed(args.seed)  # isolated generator process, never training RNG
    simulator = module.RIRSimulator(SimpleNamespace(**config))
    assets = []
    for room in range(args.rooms):
        guard.check()
        paths = simulator.gen_rirs(f"bounded_room_{room:03}")
        for index, value in enumerate(paths):
            path = Path(value).resolve(strict=True)
            if not path.is_relative_to(out):
                raise ValueError("Upstream RIR output escaped admitted E directory")
            info = sf.info(path)
            if info.samplerate != 16000 or info.channels != 1 or info.frames > 24000 or path.stat().st_size > 1024 * 1024:
                raise ValueError("Generated RIR exceeds admitted geometry/byte bounds")
            assets.append({"id": f"FastMSS_{room}_{index}", "kind": "rir", "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "sample_rate": 16000,
                "training_use_confirmed": True, "fixture_only": False,
                "license": "locally_generated_room_impulse_no_third_party_audio",
                "provenance": {"upstream": "https://github.com/popcornell/FastMSS", "revision": PIN,
                    "source_sha256": SOURCE_SHA, "software_license": "GPL-3.0",
                    "pyroomacoustics_version": pyroomacoustics.__version__,
                    "seed": args.seed, "room_index": room, "source_position_index": index,
                    "config": config, "underlying_recorded_audio": False}})
    registry = out / "FastMSS_RIR_assets.json"
    registry.write_text(json.dumps({"schema": 1, "assets": assets}, indent=2), encoding="utf-8")
    report = {"status": "generated", "upstream_revision": PIN, "actual_generated_RIRs": len(assets),
        "asset_registry": str(registry), "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
        "native_GPU_used": False, "Torch_imported": "torch" in sys.modules,
        "Lhotse_imported": "lhotse" in sys.modules, "production_training_enabled": False,
        "resources": guard.report()}
    (out / "RIR_generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    guard.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
