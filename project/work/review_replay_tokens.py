import json
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

root = Path(__file__).resolve().parent.parent
extended = Tokenizer.from_file(str(root / "work/research/extension_roundtrip/tokenizer.json"))
original = Tokenizer.from_file(str(root / "work/research/tokenizer.json"))
old_size = original.get_vocab_size()
snapshot = root / "work/data/hf_cache/datasets--google--fleurs/snapshots/70bb2e84b976b7e960aa89f1c648e09c59f894dd/parquet-data"
report = []
for filename in sorted(snapshot.glob("*/train-*.parquet")):
    rows = pq.read_table(filename, columns=["id", "raw_transcription"]).to_pylist()
    affected = []
    for row in rows:
        ids = extended.encode(row["raw_transcription"]).ids
        added = [token for token in ids if token >= old_size]
        if added:
            affected.append({"sentence_id": row["id"], "added_token_ids": added})
    report.append({"config": filename.parent.name, "rows": len(rows), "unique_sentence_ids": len({row["id"] for row in rows}), "old_vocab_size": old_size, "rows_using_appended_tokens": len(affected), "affected_examples": affected[:10]})
output = root / "work/replay_token_review.json"
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
