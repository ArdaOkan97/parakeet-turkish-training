"""Append observed throughput work to the canonical local Backlog tasks."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]

def main():
    stamp = datetime.now(timezone.utc).isoformat()
    notes = {
        'ASR-15': f'{stamp}: User clarified that the priority is stable high TRAINING throughput and faster training iteration. Selected source remains step1871, Turkish DEV64.9697%, all five retention gates pass; competitive target11.9394% remains unmet. Corrected public TRAIN pool is74883clips/57.2988h (Common Voice train+validated and isolated FLEURS TRAIN), using TDT decode group_tokens=False. New B8x2 loop proposes250-update DEV screens and a25%-replay challenger against50%-replay, both warm-starting the same1871 weights with fresh optimizer. First controller4 actual launch exited1 BEFORE GPU/training: old I/O wrapper refused trainer4 filename. Preserved failed public_data_01 artifacts; versioned full-dispatch repair in progress. No new TEST/export.',
        'ASR-16': f'{stamp}: Actual last50 training updates processed78.7113 combined audio min/min but15.2674 Turkish audio min/min. Replay was50% of clips and80.6% of audio seconds; track Turkish-specific throughput and exposure separately. Replay rehearses EN/ES/IT/FR/DE and uses custom original-model teacher matching to resist forgetting; it is not automatically enabled by NeMo standard fine-tune script. Native NeMo CUDA loss first failed Numba signature compilation. Isolated NEW compiler overlay installed hash-pinned numba-cuda[cu12]/CUDA12.9 packages; CPU imports passed and old Torch/NeMo dependencies stayed unchanged. Root actual GPU retry still failed nvJitLink ERROR_INTERNAL with newer-library message. No native NeMo TRAINING speedup measured or claimed. Completed inference benchmark is separate; further GPU inference benchmarking deprioritized by user. Primary target is actual train loss/backward/update throughput with reliable DEV feedback.',
    }
    for task, note in notes.items():
        subprocess.run([str(OUT/'tools/backlog.exe'), 'task', 'edit', task,
                        '--append-notes', note], cwd=OUT, check=True,
                       capture_output=True, text=True, encoding='utf-8')
    subprocess.run([str(ROOT/'work/training-env/Scripts/python.exe'),
                    str(OUT/'refresh_backlog.py')], cwd=ROOT, check=True)
    print('Backlog updated with actual training throughput, replay policy, and failed launch/compiler evidence.')

if __name__ == '__main__':
    main()
