"""Offline regression checks for dangerous evaluation-cache reuse."""
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "outputs/parakeet_turkish"
runtime = types.ModuleType("runtime")
runtime.DATA = ROOT / "work/data"
runtime.OUTPUT = SOURCE
runtime.load_model = None
runtime.inputs_for = None
sys.modules["runtime"] = runtime
confidence = types.ModuleType("confidence")
for name in ("generate_with_confidence", "align_word_confidences", "confidence_report", "fit_temperature"):
    setattr(confidence, name, None)
sys.modules["confidence"] = confidence
sys.path.insert(0, str(SOURCE))
import benchmark


class Config:
    def to_dict(self):
        return {"vocab_size": 7, "blank_token_id": 2}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 7)
        self.register_buffer("scalar_bf16", torch.tensor(2.0, dtype=torch.bfloat16))
        self.register_buffer("counter_int64", torch.tensor(4, dtype=torch.int64))
        self.config = Config()
        self.generation_config = Config()


class Backend:
    text = '{"vocab":{"a":0,"b":1}}'

    def to_str(self):
        return self.text


class Tokenizer:
    backend_tokenizer = Backend()
    special_tokens_map = {"unk_token": "<unk>"}
    padding_side = "right"
    truncation_side = "right"


processor = types.SimpleNamespace(tokenizer=Tokenizer(), attributes=[])
model = Model()
checks = []


def rejects(function, expected_text):
    try:
        function()
    except RuntimeError as error:
        assert expected_text in str(error), str(error)
    else:
        raise AssertionError(f"Should reject: {expected_text}")


with tempfile.TemporaryDirectory(prefix="benchmark_guard_", dir=ROOT / "work") as temporary:
    scratch = Path(temporary)
    benchmark.DATA = scratch / "data"
    benchmark.DATA.mkdir()
    row = {"source_id": "fleurs:en:validation:7:a", "text": "One word.", "language": "en", "sentence_id": 7}

    def manifest(rows):
        (benchmark.DATA / "validation_en.jsonl").write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")

    def setup(name):
        folder = scratch / name
        folder.mkdir()
        return folder

    def metadata(folder, limit=0):
        return benchmark.prepare_evaluation_metadata(model, processor, folder, ["en"], ["validation"], limit)

    manifest([row])
    folder = setup("accepted")
    assert metadata(folder) == metadata(folder)
    checks.append("identical model/tokenizer/manifest resumes")
    fp = benchmark.model_fingerprint(model, processor)
    with torch.no_grad():
        model.head.weight[0, 0].add_(0.125)
    assert benchmark.model_fingerprint(model, processor) != fp
    rejects(lambda: metadata(folder), "fingerprint changed")
    checks.append("changed head weights rejected")
    folder = setup("bf16")
    metadata(folder)
    model.scalar_bf16.fill_(3)
    rejects(lambda: metadata(folder), "fingerprint changed")
    checks.append("changed bf16 scalar buffer rejected")
    folder = setup("tokenizer")
    metadata(folder)
    processor.tokenizer.backend_tokenizer.text += " "
    rejects(lambda: metadata(folder), "fingerprint changed")
    checks.append("changed tokenizer rejected")
    folder = setup("manifest")
    metadata(folder)
    manifest([{**row, "text": "Changed reference."}])
    rejects(lambda: metadata(folder), "manifest, source IDs, or limit changed")
    checks.append("changed references with same IDs rejected")
    folder = setup("limit")
    metadata(folder)
    rejects(lambda: metadata(folder, 1), "manifest, source IDs, or limit changed")
    checks.append("changed evaluation limit rejected")
    folder = setup("legacy")
    (folder / "validation_en.jsonl").write_text("{}\n", encoding="utf-8")
    rejects(lambda: metadata(folder), "metadata.json is absent")
    checks.append("legacy prediction cache without metadata rejected")
    folder = setup("duplicate")
    manifest([row, row])
    rejects(lambda: metadata(folder), "Duplicate evaluation source IDs")
    checks.append("duplicate source IDs rejected")

report = {"device": "CPU only", "checks": checks, "passed": len(checks)}
(ROOT / "work/benchmark_metadata_test.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
