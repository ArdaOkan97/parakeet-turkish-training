"""Small offline path/normalization/source check; no archive or audio read."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

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
require("numpy" not in sys.modules and "soundfile" not in sys.modules and "tokenizers" not in sys.modules, "Source-only import did not load optional audio/ML libraries")
report = {"schema": 1, "kind": "TSC_import_inert_tiny_source_verification_v1", "passed": True,
          "checks": checks, "check_count": len(checks), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
          "archive_read": False, "audio_read": False, "network_used": False, "model_or_GPU_used": False,
          "actual_archive_layout_and_audio_import_pending": True}
target = OUTPUT / "TSC_import_tiny_source_verification_v1.json"
with target.open("x", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)
print(json.dumps({"passed": True, "checks": len(checks), "source_sha256": report["source_sha256"]}))
