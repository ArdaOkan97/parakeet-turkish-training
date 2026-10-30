"""Record measured native-loss memory diagnosis and longest-clip trials."""
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def main():
    diagnosis = json.loads((OUT / 'native_training_commit_spike_diagnosis_v1.json').read_text(encoding='utf-8'))
    reports = {cap: json.loads((OUT / name).read_text(encoding='utf-8')) for cap, name in (
        (6, 'native_long_memory_cap6_v2.json'),
        (7, 'native_long_memory_cap7_v2.json'),
        (9, 'native_long_memory_cap9_v3.json'))}
    assert diagnosis['passed'] and reports[9]['passed'] and len(reports[9]['updates']) == 3
    assert not reports[6]['passed'] and not reports[7]['passed']
    receipt = json.loads((OUT / 'native_data_01_resource_exit_receipt_v1.json').read_text(encoding='utf-8'))
    stamp = datetime.now(timezone.utc).isoformat()
    sample = diagnosis['critical_sample']['system_memory']
    pilot = reports[9]
    note = (
        f'{stamp}: The actual NeMo TDT-loss run stopped after63 updates and saved its complete continuation state. '
        f'The preceding note that it was active is superseded by this checkpointed stop. '
        f'System commit briefly reached{sample["commit_fraction"] * 100:.2f}% before end-of-update cleanup; '
        f'the same update released{diagnosis["inactive_cuda_reserved_bytes_released"] / 1e9:.2f}GB of inactive CUDA cache. '
        'The cause was delayed cleanup of variable-sized native loss allocations; this is the official loss inside our HF/LoRA loop, '
        'not a measured full NeMo Lightning trainer. Latest63 remains continuation state and best0 remains the qualifying model. '
        'A new adapter clears inactive cache around every clip and backward recomputation. '
        'Actual GPU trials kept both physicalB8 microbatches and all16 accepted TRAIN clips, including allfour largest Turkish lattices. '
        'Caps6GB and7GB both failed with genuine CUDA OOM in the first largest clip; they are not adopted. '
        f'Cap9GB completed allthree complete loss/backward/Adam updates, peak active{pilot["peak_allocated_bytes"] / 1e9:.3f}GB, '
        f'peak reserved{pilot["peak_reserved_bytes"] / 1e9:.3f}GB, highest sampled system commit{pilot["max_system_commit_fraction"] * 100:.2f}%, '
        f'minimum actual-worker commit headroom{pilot["minimum_owned_self_headroom_bytes"] / 1e9:.3f}GB. '
        'The95%/2GB guards remain enabled. This is a resource stress test, not a representative throughput or quality result. '
        'No clip was cropped or dropped, no batch1 fallback, no TEST/export. Exact63 continuation with the tested9GB policy is being prepared; '
        'the Turkish WER target remains unmet and the goal remains active.')
    for task in ('ASR-15', 'ASR-16'):
        subprocess.run([str(OUT / 'tools/backlog.exe'), 'task', 'edit', task, '--append-notes', note,
                        '--add-ref', 'native_training_commit_spike_diagnosis_v1.json',
                        '--add-ref', 'native_long_memory_cap9_v3.json'],
                       cwd=OUT, check=True, capture_output=True, text=True, encoding='utf-8')
    subprocess.run([str(ROOT / 'work/training-env/Scripts/python.exe'), str(OUT / 'refresh_backlog.py')],
                   cwd=ROOT, check=True)
    print(json.dumps({'backlog_updated': True, 'native63_closed': True, 'long16_cap9_passed': True}))


if __name__ == '__main__':
    main()
