"""Bounded offline importer checks using synthetic tar payloads, not real audio."""
import argparse
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/parakeet_turkish"
source = OUTPUT / "prepare_TSC_TRAIN_v1.py"
sys.path.insert(0, str(OUTPUT))
spec = importlib.util.spec_from_file_location("tsc_import_candidate", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
checks = []


def require(value, name):
    if not value:
        raise AssertionError(name)
    checks.append(name)


require(module.train_pair_key(module.member_path("./ISSAI_TSC_218/Train/abc.wav")) == "abc", "Train flat WAV accepted")
require(module.train_pair_key(module.member_path("ISSAI_TSC_218/Dev/abc.wav")) is None, "Dev is not imported")
require(module.train_pair_key(module.member_path("ISSAI_TSC_218/Test/abc.txt")) is None, "Test is not imported")
for name in ("../ISSAI_TSC_218/Train/a.wav", "/ISSAI_TSC_218/Train/a.wav", "ISSAI_TSC_218/Train/C:a.wav", "other/Train/a.wav", "ISSAI_TSC_218\\Train\\a.wav", "ISSAI_TSC_218/Train/nested/a.wav"):
    try:
        module.train_pair_key(module.member_path(name))
    except ValueError:
        checks.append("Rejected unsafe/unsupported path: " + name)
    else:
        raise AssertionError(name)
require(module.normalize_text("IĞDIR 123 İŞ", "tr") == "ığdır 123 iş", "Turkish I and digits preserved")
tree = ast.parse(source.read_text(encoding="utf-8"))
imports = {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
imports.update(alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
require(not imports & {"torch", "transformers", "nemo", "requests", "urllib", "huggingface_hub"}, "No model/GPU/network APIs")
require(not {"numpy", "soundfile", "tokenizers", "scipy"} & sys.modules.keys(), "Source-only import did not load optional libraries")
cpu = module.configure_cpu_runtime()
require(all(value == "1" for value in cpu["thread_environment"].values()), "Numerical/Rust thread counts set before optional imports")
require(sys.platform != "win32" or cpu["Windows_below_normal_applied"], "Own Windows process BelowNormal priority applied")

payloads = {}
for index in range(4):
    for suffix in (".wav", ".txt"):
        payloads[f"ISSAI_TSC_218/Train/fixture_{index}{suffix}"] = f"synthetic-payload-{index}-{suffix}".encode()
payloads.update({"ISSAI_TSC_218/Dev/heldout.wav": b"NOT-TRAIN", "ISSAI_TSC_218/Test/heldout.txt": b"NOT-TRAIN",
                 "ISSAI_TSC_218/README.txt": b"synthetic-document " + b"x" * 70000})


def create_tar(path, members):
    with tarfile.open(path, "w:gz") as stream:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            stream.addfile(info, io.BytesIO(payload))


with tempfile.TemporaryDirectory(prefix="tsc_tiny_", dir=ROOT / "work/research") as temp:
    folder = Path(temp)
    archive = folder / "synthetic.tar.gz"
    create_tar(archive, payloads)
    extracted = []
    original_extract = tarfile.TarFile.extractfile

    def observe_extract(stream, member):
        extracted.append(member.name)
        return original_extract(stream, member)

    with patch.object(tarfile.TarFile, "extractfile", observe_extract):
        metadata, counts, docs = module.scan_archive_pairs(archive)
    require(len(metadata) == 4 and counts["Train"] == 8 and counts["Dev"] == counts["Test"] == 1, "Complete TRAIN pair/count proof includes unselected metadata")
    require(extracted == ["ISSAI_TSC_218/README.txt"], "Metadata pass extracts only bounded documentation")
    require(docs[0]["truncated"] and len(docs[0]["text"].encode()) == 65536, "Document read bounded to64KiB")
    selected = module.select_pair_keys(metadata, 20261011, 2)
    require(selected == module.select_pair_keys(dict(reversed(list(metadata.items()))), 20261011, 2), "Selected IDs stable across archive metadata order")
    raw = folder / "raw"
    raw.mkdir()
    extracted.clear()
    with patch.object(tarfile.TarFile, "extractfile", observe_extract):
        staged, staging = module.stage_selected_pairs(archive, metadata, selected, raw)
    require(set(staged) == set(selected) and len(list(raw.iterdir())) == staging["raw_Train_files_staged"] == 4, "Limit2 stages exactly2pairs/4files rather than all4pairs")
    require(len(extracted) == 4 and all(name.startswith("ISSAI_TSC_218/Train/") for name in extracted), "Second pass extracts selected TRAIN payloads only")
    require(all(path.read_bytes() == payloads[name] for pair in staged.values() for path, name in pair.values()), "Staged selected bytes match fixture payloads")
    require(staging["raw_Train_bytes_staged"] == sum(len(payloads[name]) for name in extracted), "Reported staged byte count is actual selected payload bytes")
    require(module.select_pair_keys(metadata, 20261011, None) == module.select_pair_keys(metadata, 20261011, 20), "Full selection and limitabovepool preserve all pairs")
    bad = folder / "unpaired.tar.gz"
    create_tar(bad, {"ISSAI_TSC_218/Train/unpaired.wav": b"synthetic"})
    try:
        module.scan_archive_pairs(bad)
    except ValueError:
        checks.append("Unselected incomplete pair is rejected by metadata pass")
    else:
        raise AssertionError("Incomplete archive pair accepted")

report = {"schema": 1, "kind": "TSC_import_inert_tiny_source_verification_v2", "passed": True,
          "checks": checks, "check_count": len(checks), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
          "synthetic_archive_read": True, "actual_TSC_archive_read": False, "audio_API_used": False,
          "network_used": False, "model_or_GPU_used": False, "CPU_runtime": cpu,
          "fixture_pairs_observed": 4, "fixture_pairs_selected": 2, "fixture_staging": staging,
          "actual_archive_layout_and_audio_import_pending": True}
arguments = argparse.ArgumentParser(description=__doc__)
arguments.add_argument("--output", type=Path, default=OUTPUT / "TSC_import_tiny_source_verification_v2.json")
target = arguments.parse_args().output.resolve()
with target.open("x", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)
print(json.dumps({"passed": True, "checks": len(checks), "source_sha256": report["source_sha256"], "staged_files": staging["raw_Train_files_staged"]}))
