"""Record the actual checkpoint handoff and maintained loss candidate."""
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]

def main():
    receipt = json.loads((OUT / 'public_data_03_native_handoff_receipt_v1.json').read_text(encoding='utf-8'))
    assert receipt['root_observed_exit_code'] == receipt['actual_child_exit_code'] == 0
    assert receipt['completed_optimizer_updates'] == 18 and receipt['selected_step'] == 0
    stamp = datetime.now(timezone.utc).isoformat()
    notes = {
        'ASR-15': f'{stamp}: User prefers maintained NeMo code. Root requested cooperative handoff from public_data_03; root40682 and actual worker both exited0,18 updates completed before stop consumed. Final DEV selectedstep0/original1871 quality64.9697% (all retention gates accepted); no new quality improvement. Best/latest/history preserved in immutable root handoff receipt. Goal remains active. Next actual250-update candidate uses verified NeMo TDT CUDA loss, same57.3h TR data/B8x2/replayteacher/LoRA settings; no additional full207gradient pilot prerequisite for this explicitly declared DEV quality experiment.',
        'ASR-16': f'{stamp}: Native quality backend prepared in new frozen trainer7 and IO5/cache8/memory11, original sources unchanged. All74,883 TR and1,595 raw replay rows checked for full per-clip joint allocation;4 TR rows exceed old1.6GB logits guard, maximum1,645,131,712B. New explicit1.7GB per-clip logits bound preserves every accepted row;9GB allocator cap remains (1.7GB is not a VRAM peak proof). Whole update/sampler/Adam/LR/checkpoint/DEV logic retained,23 actual CPU nested dispatch checks passed. Official loss uses existing HF model/projections and original interiorblank8192; full stock NeMo trainer needs explicit weight/tokenizer/blank/SOS conversion and would otherwise rebuild trained heads. CPU migration plan saved separately. Actual NeMo-native loss training20update benchmark is visible in same Trackio with93.09vs92.57min/min and4.58vs5.44GB peak allocation.',
    }
    for task, note in notes.items():
        subprocess.run([str(OUT / 'tools/backlog.exe'), 'task', 'edit', task,
                        '--append-notes', note, '--add-ref', 'public_data_03_native_handoff_receipt_v1.json'],
                       cwd=OUT, check=True, capture_output=True, text=True, encoding='utf-8')
    subprocess.run([str(ROOT / 'work/training-env/Scripts/python.exe'), str(OUT / 'refresh_backlog.py')],
                   cwd=ROOT, check=True)
    print(json.dumps({'backlog_updated': True, 'handoff_updates': 18, 'selected_step': 0,
                      'native_loss_candidate_prepared': True}))

if __name__ == '__main__':
    main()
