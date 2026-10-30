"""Request an owned cooperative checkpoint stop and interlock the next stage."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/parakeet_turkish'
WORK = ROOT / 'work'
LOOP = OUT / 'runs/adaptive_20261010_01'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)

status_path = LOOP / 'loop_status.json'
state = json.loads(status_path.read_text(encoding='utf-8'))
assert state['status'] == 'running'
assert state['active_run_id'] == 'adaptive_20261010_01_screen_A'
controller = OUT / 'run_experiment_loop.py'
assert sha(controller) == state['controller_sha256']
snapshot = LOOP / 'original_status_before_pause.json'
assert not snapshot.exists()
snapshot.write_bytes(status_path.read_bytes())
config = OUT / 'runs' / state['active_run_id'] / 'run_config.json'
config_before = LOOP / 'original_config_before_pause.json'
config_before.write_bytes(config.read_bytes())
now = datetime.now(timezone.utc).isoformat()
markers = []
for name, role in [('adaptive_20261010_01_screen_B.stop', 'next_stage_interlock'),
                   ('adaptive_20261010_01_screen_A.stop', 'trainer_pause')]:
    marker = (WORK / name).resolve()
    assert marker.is_relative_to(WORK.resolve()) and not marker.exists()
    marker.write_text(f'User-authorized GPU benchmark pause {now}; role={role}.\n', encoding='utf-8')
    markers.append({'path': str(marker), 'sha256': sha(marker), 'role': role})
manifest = {
    'schema': 1, 'status': 'checkpoint_stop_requested', 'requested_at_utc': now,
    'loop_id': state['loop_id'], 'run_id': state['active_run_id'],
    'original_controller_sha256': sha(controller),
    'original_status_path': str(snapshot), 'original_status_sha256': sha(snapshot),
    'original_config_before_pause_path': str(config_before),
    'original_config_before_pause_sha256': sha(config_before),
    'owned_stop_markers': markers, 'benchmark_completed': False,
    'authorization': 'User requested pause, numerical checks and batch-size speed sweep, then resume.',
    'pause_policy': 'Cooperative trainer stop saves optimizer/sampler/RNG and performs one additional DEV evaluation. Next-stage marker prevents another training stage from launching.'
}
write(LOOP / 'gpu_benchmark_pause.json', manifest)
print(json.dumps({'status': manifest['status'], 'manifest': str(LOOP / 'gpu_benchmark_pause.json')}), flush=True)
