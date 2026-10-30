"""Select actual cached lattice/encoder stress cases on CPU only."""
import hashlib
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0, str(OUTPUT))
from metrics import normalize_text

torch.set_num_threads(1)
config = json.loads((OUTPUT / "run_config.json").read_text(encoding="utf-8"))
rows = [json.loads(line) for line in (ROOT / "work/data/train.jsonl").read_text(encoding="utf-8").splitlines()]
tokenizer_path = OUTPUT / "checkpoint/tokenizer.json"
if not tokenizer_path.exists():
    tokenizer_path = ROOT / "work/research/extension_roundtrip/tokenizer.json"
tokenizer = Tokenizer.from_file(str(tokenizer_path))
namespaces = [path for path in (ROOT / "work/encoder-cache").iterdir() if path.is_dir()]
assert len(namespaces) == 1, "Choose the active namespace explicitly if multiple exist"
namespace = namespaces[0]
grouped = {"tr": [], "replay": []}
for row in rows:
    cache = namespace / (hashlib.sha256(row["source_id"].encode()).hexdigest() + ".pt")
    if not cache.exists():
        continue
    item = torch.load(cache, map_location="cpu", weights_only=True)
    assert item["source_id"] == row["source_id"]
    features, labels = item["features"], item["labels"]
    assert features.ndim == 2 and features.shape[1] == 1024 and labels.ndim == 1
    ids = tokenizer.encode(row["text"]).ids
    assert labels.tolist() == ids, row["source_id"]
    assert 0 < len(ids) <= config["max_tokens"]
    assert normalize_text(tokenizer.decode(ids), row["language"]) == normalize_text(row["text"], row["language"])
    group = "tr" if row["language"] == "tr" else "replay"
    if group == "replay":
        assert max(ids) < 8193
    frames, tokens = int(features.shape[0]), int(labels.numel())
    states = frames * (tokens + 1)
    grouped[group].append({"row": row, "cache_path": str(cache.resolve()), "encoder_frames": frames,
                           "target_tokens": tokens, "lattice_states": states,
                           "dense_joint_fp32_mib": states * 8204 * 4 / 1024 ** 2,
                           "dense_joint_bf16_mib": states * 8204 * 2 / 1024 ** 2})
assert len(grouped["tr"]) == config["turkish_clips_used"]
assert len(grouped["replay"]) == config["replay_clips_used"]
report = {"device": "CPU only", "cache_namespace": namespace.name,
          "base_encoder_revision": config["base_revision"], "tokenizer_path": str(tokenizer_path),
          "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
          "verified_cached_examples": {key: len(values) for key, values in grouped.items()},
          "selection": "Maximum positive representable T*(U+1) per stream for dense-joint stress; longest-T alternatives retained for encoder-activation stress",
          "examples": {key: max(values, key=lambda value: (value["lattice_states"], value["row"]["source_id"])) for key, values in grouped.items()},
          "longest_encoder_examples": {key: max(values, key=lambda value: (value["encoder_frames"], value["lattice_states"])) for key, values in grouped.items()},
          "short_parity": min(grouped["replay"], key=lambda value: (value["lattice_states"], value["row"]["source_id"])),
          "notes": ["All 3469 actual cached labels match the exact extended tokenizer and lexical round-trip", "Cache features are valid only for the pinned frozen base encoder; encoder-LoRA profiling/training must recompute raw-audio features", "Dense sizes describe one T*(U+1)*(8199 token + 5 duration) tensor, excluding loss/gradient/encoder activations"]}
(OUTPUT / "memory_profile_examples.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({key: {name: value[name] for name in ("encoder_frames", "target_tokens", "lattice_states", "dense_joint_bf16_mib")} for key, value in report["examples"].items()}, indent=2))
print("Short replay parity", report["short_parity"]["encoder_frames"], report["short_parity"]["target_tokens"], report["short_parity"]["lattice_states"])
