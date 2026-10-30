"""Small CPU fixtures for the new immutable public-data preparation helper."""
from pathlib import Path
import sys
import tempfile
import io
import json
import hashlib
import numpy as np
import soundfile as sf
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'outputs/parakeet_turkish'))
import prepare_turkish_data_v2 as prep

checks = []
def check(name, value):
    if not value:
        raise AssertionError(name)
    checks.append(name)

def audio(n, rate=16000, channels=1):
    x = np.sin(np.arange(n, dtype=np.float32) * .021) * .15
    if channels == 2:
        x = np.stack([x, x * .5], axis=1)
    out = io.BytesIO()
    sf.write(out, x, rate, format='WAV', subtype='PCM_16')
    return out.getvalue()

for n in (8000, 16000, 480000):
    wav, duration = prep.decode_audio(audio(n), .5, 30.)
    decoded, rate = sf.read(io.BytesIO(wav), dtype='int16')
    check('inclusive-duration-' + str(n), duration == n / 16000 and rate == 16000 and len(decoded) == n)
for n in (7999, 480001):
    wav, duration = prep.decode_audio(audio(n), .5, 30.)
    check('outside-duration-' + str(n), wav is None and duration == n / 16000)
for rate in (8000, 22050, 48000):
    wav, duration = prep.decode_audio(audio(rate, rate), .5, 30.)
    decoded, actual_rate = sf.read(io.BytesIO(wav), dtype='int16')
    check('resampling-' + str(rate), duration == 1 and actual_rate == 16000 and len(decoded) == 16000)
wav, duration = prep.decode_audio(audio(16000, channels=2), .5, 30.)
signal, rate = sf.read(io.BytesIO(wav), dtype='int16', always_2d=True)
check('stereo-downmix', signal.shape == (16000, 1) and rate == 16000)

with tempfile.TemporaryDirectory(dir=ROOT / 'work/research', prefix='data_v2_cpu_') as directory:
    directory = Path(directory)
    wav_path = directory / 'canonical.wav'
    wav_path.write_bytes(wav)
    canonical, measured = prep.pcm_hash(wav_path)
    check('canonical-sample-hash', canonical == hashlib.sha256(signal[:, 0].astype('<i2').tobytes()).hexdigest())
    check('canonical-measured-duration', measured == duration)
    records = [dict(audio=dict(bytes=audio(16000), path='clip.wav'), transcription='Aynı Türkçe cümle.', duration=1., gender='female'),
               dict(audio=dict(bytes=audio(16001), path='clip2.wav'), transcription='Aynı Türkçe cümle.', duration=1.0000625, gender='female')]
    parquet_path = directory / 'validated.parquet'
    pq.write_table(pa.Table.from_pylist(records), parquet_path)
    parquet, metadata = prep.source_metadata(parquet_path, prep.CV_REPO, prep.CV_REV, 'validated', 'data/validated-00000-of-00001.parquet', 'default')
    check('honest-validated-split', all(row['dataset_split'] == 'validated' for row in metadata))
    check('explicit-training-role', all(row['training_role'] == 'train' for row in metadata))
    check('repeated-train-text-preserved', len(metadata) == 2 and metadata[0]['reference_normalized'] == metadata[1]['reference_normalized'])
    check('distinct-recording-source-ids', metadata[0]['source_id'] != metadata[1]['source_id'])
    check('gender-not-speaker-id', all(row['speaker_id'] is None for row in metadata))
    check('source-revision-pinned', all(row['dataset_revision'] == prep.CV_REV for row in metadata))
    heldout = dict(metadata[0], dataset_split='validation', reference='Özel kıyaslama cümlesi.', audio_sha256='heldout-source', wav_sha256='heldout-wav')
    (directory / 'validation_tr.jsonl').write_text(json.dumps(heldout, ensure_ascii=False) + '\n', encoding='utf-8')
    filters, paths = prep.heldout_filters(directory)
    check('fixed-heldout-found', len(paths) == 1)
    check('heldout-source-isolation', prep.forbidden_reason(metadata[0], filters) == 'heldout_source_id')
    probe = dict(metadata[1], reference=heldout['reference'])
    check('heldout-normalized-text-isolation', prep.forbidden_reason(probe, filters) == 'heldout_normalized_reference')
    probe = dict(metadata[1], audio_sha256='heldout-source')
    check('heldout-embedded-audio-isolation', prep.forbidden_reason(probe, filters) == 'heldout_source_audio_hash')
    probe = dict(metadata[1], wav_sha256='heldout-wav')
    check('heldout-wav-isolation', prep.forbidden_reason(probe, filters) == 'heldout_wav_hash')
    check('unrelated-repeated-train-text-allowed', prep.forbidden_reason(metadata[1], filters) is None)
    parquet.close()

report = dict(schema=1, passed=True, CPU_only=True, check_count=len(checks), checks=checks,
              preparation_script_sha256=prep.sha256(Path(prep.__file__)),
              limitations=['Fixture checks do not establish final full-corpus counts or new optimizer exposure.'])
target = ROOT / 'outputs/parakeet_turkish/turkish_data_v2_cpu_fixture_verification.json'
if target.exists():
    raise FileExistsError(target)
target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(passed=True, check_count=len(checks), report=str(target))))
