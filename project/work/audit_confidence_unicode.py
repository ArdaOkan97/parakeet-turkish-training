"""CPU-only Unicode word grouping audit using all prepared reference texts."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0, str(OUTPUT))
from confidence import _decode_tokens, normalize_text

backend = Tokenizer.from_file(str(ROOT / "work/research/tokenizer.json"))
tokenizer = SimpleNamespace(_tokenizer=backend)
report = []
for name in ("train", "validation", "test"):
    rows = [json.loads(line) for line in (ROOT / f"work/data/{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    count = 0
    for row in rows:
        ids = backend.encode(row["text"]).ids
        tokens = [{"token_id": token_id, "logprob_by_temperature": {"1": -0.1},
                   "conditional_nonblank_logprob": -0.1, "normalized_entropy": 0.1} for token_id in ids]
        text, words = _decode_tokens(tokenizer, tokens, row["language"])
        assert text == backend.decode(ids, skip_special_tokens=True), row["source_id"]
        assert [word["word"] for word in words] == normalize_text(text, row["language"]).split(), row["source_id"]
        count += len(words)
    report.append({"manifest": name, "rows": len(rows), "scored_decoded_words": count})
result = {"device": "CPU only", "scope": "Original tokenizer encode/decode reference text, all 5896 prepared examples; synthetic constant token probabilities, not actual model inference confidence", "checks": "Incremental decoded text equals full tokenizer decode; scored lexical units equal the shared normalizer for every example", "manifests": report, "passed": True}
(OUTPUT / "confidence_unicode_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
