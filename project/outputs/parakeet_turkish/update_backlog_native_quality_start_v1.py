"""Record the actual maintained-loss experiment launch, without a quality claim."""
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]

def main():
    loop = OUT / 'runs/adaptive_20261011_native_data_01'
    state = json.loads((loop / 'loop_status.json').read_text(encoding='utf-8'))
    assert state['status'] == 'running' and state['runtime_policy']['tdt_implementation'] == 'nemo'
    backend_path = loop / (state['active_run_id'] + '_native_backend.json')
    backend = json.loads(backend_path.read_text(encoding='utf-8'))
    assert backend['status'] == 'ready' and backend['tdt_implementation'] == 'nemo'
    stamp = datetime.now(timezone.utc).isoformat()
    note = (f'{stamp}: ACTUAL native-loss quality loop started by root session33403, frozen controller7 '
            f'0b7f4a838ea4d61ef2636465cd489a59c661eff1ce669c1e8154ba61cfc72e8e. '
            f'Active {state["active_run_id"]}, real phase {state.get("current_phase")}. '
            'Actual immutable backend-ready provenance confirms official NeMo TDT loss and coherent CUDA12.9 '
            'preload before Torch; child uses native-env-v2. Same original1871 selected source, fresh optimizer, '
            'B8x2,57.3h Turkish TRAIN, metadata-only Turkish and genuine replay teacher features, first250 updates '
            'then all6 DEV WER/5 retention gates; replay25 challenger and winner extension follow. '
            'Custom TDT implementation is replaced in this loop. HF model/LoRA/replay/controller orchestration '
            'remains our code; no full official Lightning-trainer claim. Initial validation is not a new trained '
            'WER result. No TEST/export. Competitive target remains unmet and goal active.')
    for task in ('ASR-15', 'ASR-16'):
        subprocess.run([str(OUT / 'tools/backlog.exe'), 'task', 'edit', task, '--append-notes', note,
                        '--add-ref', str(backend_path.relative_to(OUT))],
                       cwd=OUT, check=True, capture_output=True, text=True, encoding='utf-8')
    subprocess.run([str(ROOT / 'work/training-env/Scripts/python.exe'), str(OUT / 'refresh_backlog.py')],
                   cwd=ROOT, check=True)
    print(json.dumps({'backlog_updated': True, 'native_loop_active': state['loop_id'],
                      'phase': state.get('current_phase'), 'backend': backend['tdt_implementation']}))

if __name__ == '__main__':
    main()
