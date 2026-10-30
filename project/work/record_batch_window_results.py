"""Record audited measurements and actual resume through the official Backlog CLI."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/parakeet_turkish'
LOOP = OUT / 'runs/adaptive_20261010_01'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def cli(*arguments):
    result = subprocess.run([str(OUT / 'tools/backlog.exe'), *arguments], cwd=OUT,
        check=True, capture_output=True, text=True, encoding='utf-8')
    print(result.stdout[:1500])

def main():
    audit = read(OUT / 'large_batch_results_verification.json')
    resume = read(LOOP / 'exact_resume_verification.json')
    state = read(LOOP / 'loop_status.json')
    assert audit['passed'] is True and resume['passed'] is True
    assert state['status'] == 'running' and state['active_run_id'] == 'adaptive_20261010_01_screen_A'
    speed = read(OUT / 'large_batch_speed_report.json')
    table = [row for row in speed['table'] if row['timing_scope'] == 'end_to_end']
    notes = ['Completed the user-authorized checkpointed GPU window: all30 isolated processes finished normally, with no OOM or nonfinite failures. All B2/B4/B8/B16 FP32 MATH objective/gradient gates passed for typical and long TRAIN clips. Every strict same-shape BF16 gradient gate, including B1 control, failed and is retained explicitly; this characterizes rounding and does not establish poor ASR quality. No gate was waived and no larger batch was adopted.']
    for family in ('typical', 'long'):
        rows = sorted((row for row in table if row['family'] == family), key=lambda row: row['batch_size'])
        baseline = rows[0]['audio_min_per_wall_min']
        notes.append(f'{family.capitalize()} fixed16 TRAIN clip timings: ' + '; '.join(
            f"B{row['batch_size']} {row['audio_min_per_wall_min']:.2f} audio min/min ({row['audio_min_per_wall_min']/baseline:.2f}x B1), allocated {row['peak_allocated_vram_gb']:.2f}GB, reserved {row['peak_reserved_vram_gb']:.2f}GB" for row in rows) + '.')
    notes += ['Effective optimizer batch16 and exact16 recordings/checkpoint/objective/precision are matched within each family; original adaptive training remains effective4. Ratios are microbenchmarks and are not multiplied by the earlier v2 optimization gain or claimed as full-data throughput. Two warmup updates precede10 typical or3 long measured updates. Timings include WAV/cache I/O, frontend, forward/backward and Adam, excluding setup, DEV and checkpoint writes.',
        'Typical B8 is fastest (padding1.15x versus B16 mixed-stream padding2.18x); long B4 is fastest. Next controlled DEV experiment should investigate B4/B8 with appropriate length/language bucketing, explicit effective-batch/LR choices, and all five language-retention gates. Effective32 was not explored. Quality adoption remains pending.',
        'Published both audited benchmark families in existing local Trackio project parakeet-turkish, group Physical batch benchmarks, runs batch_profile_effective16_typical and batch_profile_effective16_long. Training resumed at step1159 on unchanged B1x4 with actual weights/Adam/sampler/RNG restoration audited exactly; original absolute12-hour deadline retained.']
    description = '\n\n'.join([
        'Find a faster physical batch with the local16GB GPU. The user authorized an exclusive checkpointed pause now; preserve the existing adaptive trial and original absolute12-hour deadline.',
        'Hold the exact same8 Turkish and8 multilingual replay TRAIN recordings, checkpoint and objective fixed at effective batch16: B1x16/B2x8/B4x4/B8x2/B16x1. Characterize against the singleton FP32 MATH oracle and separately retain BF16 rounding differences. Measure warmed end-to-end throughput, actual allocated/reserved memory and padding on typical and long clips.',
        'All30 processes completed; independent integrity audit passed. The saved speed report identifies typical B8 and long B4 as the fastest measured configurations. Strict BF16 gradient gates remain unresolved, including the B1 control, so larger-batch quality adoption requires a new explicit DEV-selected experiment with all original-language retention gates. Existing adaptive training resumed unchanged; no TEST predictions were used. Effective32 was not explored.'])
    cli('task', 'edit', 'ASR-16', '--description', description, '--check-ac', '2', '--check-ac', '3',
        '--append-notes', '\n\n'.join(notes), '--add-ref', 'large_batch_speed_report.json',
        '--add-ref', 'large_batch_results_verification.json', '--add-ref', 'large_batch_trackio_publication.json',
        '--add-ref', 'runs/adaptive_20261010_01/exact_resume_verification.json')
    cli('task', 'edit', 'ASR-15', '--append-notes',
        'Completed the authorized larger-batch measurement window and resumed screen_A exactly from step1159, retaining full Adam/sampler/Python+Torch CPU/CUDA RNG/LR/DEV selection counters. Actual resumed checkpoint tensors and all other state were independently audited against the immutable paused checkpoint; only cooperative-stop reason was cleared. Progress beyond1159 confirmed. Same B1x4/v2 training configuration and original12-hour absolute deadline preserved; profiling time counts against it. ExtraDEV at pause remains documented. Larger-batch timings are recorded in ASR-16; no batch adoption or new TEST prediction occurred.',
        '--add-ref', 'runs/adaptive_20261010_01/exact_resume_verification.json')
    subprocess.run([sys.executable, '-X', 'utf8', str(OUT / 'refresh_backlog.py')], cwd=ROOT, check=True)

if __name__ == '__main__':
    main()
