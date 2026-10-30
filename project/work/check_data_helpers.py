import io
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "outputs" / "parakeet_turkish"))
import metrics
import prepare_data as prep

assert metrics.normalize_text("I İ ÇĞÖŞÜ", "tr") == "ı i çğöşü"
assert metrics.normalize_text("Café, l'homme!", "fr") == "café lhomme"
assert metrics.edit_distance(["a", "b"], ["a", "x", "b"]) == 1
delta = metrics.paired_bootstrap_delta(["a b"], ["a x b"], ["a b"], language="en", resamples=20)
assert delta["wer"]["delta_percentage_points"] == -50
assert delta["wer"]["ci_percentage_points"] == [-50, -50]

scratch = ROOT / "work" / "synthetic_data_check"
scratch.mkdir(exist_ok=True)
payload = io.BytesIO()
sf.write(payload, np.column_stack([np.full(12000, 0.1), np.full(12000, 0.2)]), 8000, format="WAV", subtype="PCM_16")
parquet_path = scratch / "test-00000-of-00001.parquet"
pq.write_table(pa.Table.from_pylist([
    {"id": 10, "path": "audio/test/one.wav", "raw_transcription": "İki iyi çocuk.", "transcription": "iki iyi çocuk", "num_samples": 24000, "gender": 0, "audio": {"bytes": payload.getvalue(), "path": "one.wav"}},
    {"id": 11, "path": "audio/test/two.wav", "raw_transcription": "Ilık bir gün.", "transcription": "ılık bir gün", "num_samples": 24000, "gender": 1, "audio": {"bytes": payload.getvalue(), "path": "two.wav"}},
]), parquet_path)
prep.hf_hub_download = lambda **kwargs: str(parquet_path)
source = {"revision": "0" * 40, "files": ["parquet-data/tr_tr/test-00000-of-00001.parquet"]}
records, provenance = prep.collect_metadata(prep.FLEURS_REPO, "tr_tr", "test", "tr", source, scratch, True)
assert len(records) == 2 and len(provenance) == 1
first = prep.stable_select(records, 1, 20261010, "test")
assert prep.stable_select(list(reversed(records)), 1, 20261010, "test")[0]["source_id"] == first[0]["source_id"]
converted = prep.export_audio(first, scratch, 1, 20)
assert len(converted) == 1
info = sf.info(converted[0]["audio_filepath"])
assert info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16"
assert info.duration == 1.5
assert len(converted[0]["audio_sha256"]) == 64
prep.write_jsonl(scratch / "manifest.jsonl", converted)
assert '"_parquet"' not in (scratch / "manifest.jsonl").read_text(encoding="utf-8")
try:
    prep.assert_split_isolation([records[0]], [], [dict(records[0], language="en", source_id="other")])
except RuntimeError:
    pass
else:
    raise AssertionError("Cross-language FLEURS sentence leakage should be rejected")
print("PASS: Turkish/Unicode scoring, paired bootstrap, stable sampling, parquet audio extraction, stereo resampling, manifest provenance, and cross-language split isolation")
