"""Verify the completed cooperative stop; save immutable resume evidence."""
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/parakeet_turkish'
LOOP = OUT / 'runs/adaptive_20261010_01'
RUN = OUT / 'runs/adaptive_20261010_01_screen_A'
manifest_path = LOOP / 'gpu_benchmark_pause.json'
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
deadline = time.monotonic() + 180
while True:
    state = json.loads((LOOP / 'loop_status.json').read_text(encoding='utf-8'))
    if state['status'] == 'failed' and not state.get('active_run_id'):
        assert state['failure_message'] == 'Trial ID/stop marker already exists.'
        break
    assert state['status'] in ('running', 'failed'), state['status']
    if time.monotonic() > deadline:
        raise RuntimeError('Checkpoint pause has not completed; no GPU benchmark is authorized yet.')
    time.sleep(1)
selection = json.loads((RUN / 'selection.json').read_text(encoding='utf-8'))
assert selection['stop_early_reason'] == 'controller_stop_file'
assert selection['run_id'] == RUN.name
source_checkpoint = Path(selection['latest_weights'])
saved = torch.load(source_checkpoint, map_location='cpu', weights_only=True)
step = selection['completed_steps']
assert saved['run_id'] == RUN.name and saved['step'] == step and step > 0
assert len(saved['history']) == step
assert saved['stop_reason'] == 'controller_stop_file'
assert saved['sampler']['tr']['draws'] == step * 2
assert len(saved['optimizer']['state']) == 207
checkpoint = ROOT / 'work' / f'{RUN.name}_pause_step_{step}.pt'
assert not checkpoint.exists()
shutil.copyfile(source_checkpoint, checkpoint)
assert sha(source_checkpoint) == sha(checkpoint)
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
for label, name in [('config', 'run_config.json'), ('history', 'training_history.json'), ('selection', 'selection.json')]:
    target = LOOP / f'paused_{name}'
    assert not target.exists()
    shutil.copyfile(RUN / name, target)
    manifest[f'paused_{label}_path'] = str(target)
    manifest[f'paused_{label}_sha256'] = sha(target)
final_state = LOOP / 'original_status_after_pause.json'
final_state.write_bytes((LOOP / 'loop_status.json').read_bytes())
manifest.update(status='paused_checkpoint_verified', paused_at_utc=datetime.now(timezone.utc).isoformat(),
                checkpoint_path=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                paused_step=step, original_status_after_pause_path=str(final_state),
                original_status_after_pause_sha256=sha(final_state),
                optimizer_tensors=207, resume_state_includes=['weights','Adam','sampler','Python RNG','Torch CPU/CUDA RNG','LR schedule','DEV selection counters'],
                extra_development_evaluation=True)
temporary = manifest_path.with_suffix('.tmp')
temporary.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
temporary.replace(manifest_path)
print(json.dumps({'status':manifest['status'],'step':step,'checkpoint_sha256':manifest['checkpoint_sha256'],
                  'turkish_development_wer':selection['turkish_validation_wer']}), flush=True)
