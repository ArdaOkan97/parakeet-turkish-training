"""Create the authorized physical batch exploration task via official CLI."""
import subprocess
from pathlib import Path

root=Path(__file__).resolve().parents[2]
out=root/'outputs/parakeet_turkish'
description='''Find a numerically characterized, faster physical batch using the local 16 GB GPU after the active adaptive loop and its evaluation/export release the GPU. The current frozen 12-hour B1/v2, accumulation 4 run continues unchanged.

CPU preparation is complete in large_batch_plan.json, using the independently audited TRAIN encoder/target lengths. Available VRAM is not a reason to cap the physical batch at one: measured B1 allocation is about 3 GB, while the persistent model/optimizer storage is shared across batch sizes. Chunk tensor estimates are lower bounds; actual CUDA allocated/reserved peaks and throughput must be measured.

The full 24-layer GPU diagnostic found shape-dependent FP32 default-SDPA attention drift; forced MATH restores B2 encoder relativeL2 around 1.3e-6 to 1.5e-6. BF16 still has convolution/projection shape rounding (about 0.29% subsampling and 1-1.6% final encoder drift). Full objective/gradient characterization is required before physical-batch adoption. Concurrent correctness diagnostics provide no valid throughput measurement.

For effective batch 16, hold the exact same 8 Turkish and 8 replay clips, starting checkpoint, optimizer state, precision/backend, full-vocabulary loss and retention objective fixed: B1x16/B2x8/B4x4/B8x2/B16x1. B16x1 is a mixed-stream microbatch. B16x2 is a separate effective32 optimizer experiment (16 TR + 16 replay), with its own B1x32/B2x16/B4x8/B8x4 controls. All effective16/32 settings also change the current loop optimizer batch 4, and must be labeled explicitly.

Give the GPU exclusively to warmed, synchronized timing runs; do not time alongside teacher-cache generation, training, inference, other profiling or CUDA compute. A later finite profiler may wait for the entire current loop/finalizer/strong TEST/audit to finish, then run a separately assigned bounded matrix (proposed 45 minutes). No profiling job or automation has been launched. Select measured throughput after memory/numerical gates, and adopt only in a new DEV-selected training experiment with original-language retention gates.'''
criteria=[
    'Prepare reproducible CPU TRAIN T/U length-bucket, padding and full-vocabulary joint-chunk memory estimates, labeling tensor bounds versus actual peak and all optimizer-batch changes.',
    'Characterize B2/B4/B8/B16 full objective, teacher losses, masking, gradients and Adam/new-token-row updates against a singleton FP32+MATH oracle; separately record BF16 batch-shape numerical differences and explicit eligible/rejected configurations.',
    'With the GPU exclusively available, measure warmed matched effective16 configurations B1x16/B2x8/B4x4/B8x2/B16x1 on identical balanced TRAIN clips, including typical and long cases; record actual throughput, median/p90 update time, allocated/reserved peaks, padding, precision/backend and all source/checkpoint hashes, retaining OOM/failure outcomes.',
    'Report effective32 B16x2 and its controls separately if explored, select a configuration by measured throughput and verified numerical/memory gates, and make any adoption a new explicit DEV-selected trial; never alter the frozen current loop or tune from TEST predictions.'
]
command=[str(out/'tools/backlog.exe'),'task','create','Explore larger physical GPU batches with matched optimizer exposure',
         '--status','To Do','--priority','High','--type','spike','--description',description,
         '--ref','large_batch_plan.json','--ref','batching_speed_comparison.json','--plain']
for criterion in criteria:command.extend(['--ac',criterion])
result=subprocess.run(command,cwd=out,check=True,text=True,encoding='utf-8',capture_output=True,timeout=60)
print(result.stdout)
(root/'work/large_batch_task_creation.txt').write_text(result.stdout,encoding='utf-8')
