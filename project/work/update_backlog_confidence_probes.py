"""Append audited evidence through the official CLI, then refresh the board."""
import json
from pathlib import Path
import subprocess
from datetime import datetime,timezone
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/parakeet_turkish'
CLI=OUTPUT/'tools/backlog.exe'
read=lambda name:json.loads((OUTPUT/name).read_text(encoding='utf-8'))
robust=read('baseline_confidence_robustness_independent_audit.json')
profile=read('tdt_profile_independent_audit.json')
assert robust['passed'] and profile['passed']
now=datetime.now(timezone.utc).astimezone(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M %Z')

notes=f'''{now}: The original-v2/v3 tiny-audio-variation probe completed on the first ten fixed English development recordings, selected before perturbation inference. Per model: clean, -0.2 dB amplitude attenuation, source-seeded Gaussian noise at measured 50 dB SNR, and 20 ms zero padding at each end; 80 actual GPU predictions total. Independent CPU audit passed {robust['check_count']} source, waveform-hash, emitted-word, jiwer-correctness/reference-index, ASR, word-score and pair-summary checks with zero numerical recalculation difference. Clean transcripts exactly reproduce the corresponding complete baseline predictions. V2's config fingerprint differed only in serialized local path slash direction; both hashes were exactly reconstructed and weight/tokenizer/generation/code identities are identical.

On words correct in all four decodes (v2/v3 clean/perturbed), mean absolute confidence change in percentage points was v2/v3: gain .1382/.0317 (209 words), noise .00925/.08378 (207 words), padding .2400/.1403 (207 words). Thus this small noise condition supports more v3 raw-confidence movement: v3-minus-v2 +.07453 points, descriptive 95% recording-bootstrap interval [.00168,.18997]. Conditional nonblank difference interval crosses zero. Gain/padding favor smaller mean v3 movement, with differences' intervals crossing zero. Mean absolute change in each word's selected-token mean logprob (natural-log units) is gain .002100/.000357, noise .00009295/.00089195, padding .002729/.001477; different model vocabularies/segmentation make these descriptive word-level comparisons, not equivalent token identities.

V3's raw and normalized transcripts were unchanged for all three perturbations; WER stayed 8/219 = 3.6530%. V2 gain also retained all transcripts; noise changed 3/10 normalized transcripts and padding changed 2/10 (3/10 raw), with WER increasing from 7/219 = 3.1963% to 10/219 = 4.5662% for both. Isolated large score changes occur in both models: gain shifts v2's correct 'two' by -23.11 confidence points, while noise shifts v3's correct 'kingdom' by +6.55 points. This is mixed evidence and does not establish a general v2 confidence/transcript stability advantage. Ten development recordings, selected-correct-word conditioning, BF16 inference and no listening assessment limit generalization; WER/transcript changes are retained to expose selection effects. Probe source and added descriptive bootstrap uncertainty are saved. Selected-fine-tune confidence/calibration and perturbation evaluation remain separate open work.'''

def cli(*args):
    subprocess.run([str(CLI),*args],cwd=OUTPUT,check=True,capture_output=True)

cli('task','edit','ASR-7','--append-notes',notes,
    '--append-final-summary','Completed the ten-development-recording gain/noise/padding probe and independent audit. Results are mixed: v3 confidence varies more for weak noise, while v3 transcripts remain unchanged and its mean gain/padding confidence movement is smaller. This small probe does not establish a general stability ranking; selected-pilot evaluation remains pending.',
    '--add-ref','confidence_robustness.py','--add-ref','baseline_confidence_robustness.json',
    '--add-ref','baseline_confidence_robustness_independent_audit.json')
cli('task','edit','ASR-10','--append-notes',
    f"{now}: Separate CPU audit of recorded actual CUDA profiles passed {profile['check_count']} consistency/numerical checks. All source-code and selection byte hashes, pinned base path/configuration/seed, selected clip identities, finite/present gradients and optimizer objective weights match; long-clip full/reduced objectives agree within mixed-precision tolerance. Recomputed aggregate short-clip gradient error from every saved head parameter matches the report. Exact allocated peaks 7.642039808/3.041240064/3.088490496 GB, reserved peaks 9.724493824/3.284140032/3.380609024 GB for full/reduced/LoRA; measured full-to-reduced allocation reduction 60.2038%. Actual GPU parity is one short cached recording's decoder/projector/joint gradients, not an all-layer LoRA-vs-full raw-audio gradient comparison. LoRA's finite gradients and 96 nonzero B tensors validate its path, not quality. Single matched accumulated updates are descriptive allocator/time evidence, not replicated steady-state throughput or total driver memory. Audit saved as tdt_profile_independent_audit.json.",
    '--add-ref','tdt_profile_independent_audit.json')
subprocess.run([str(ROOT/'work/training-env/Scripts/python.exe'),str(OUTPUT/'refresh_backlog.py')],cwd=OUTPUT,check=True)
with (OUTPUT/'confidence_findings.txt').open('a',encoding='utf-8') as stream:
    stream.write('\n\n'+notes+'\n')
print('ASR-7 perturbation and ASR-10 independent profile evidence appended; board refreshed.')
