"""Record successful root-owned NeMo GPU proof and actual restart phase."""
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]

def main():
    proof = json.loads((OUT/'nemo_native_loss_linker_v4_verification.json').read_text())
    linker = json.loads((OUT/'nemo_native_loss_linker_v4_verification_linker_provenance.json').read_text())
    assert proof['passed'] and proof['native_kernel_executed']
    assert linker['coherent_libraries_after_proof']
    assert linker['actual_binding_linker_after_proof']['version'] == [12, 9]
    state = json.loads((OUT/'runs/adaptive_20261011_public_data_02/loop_status.json').read_text())
    trial = next(t for t in state['trials'] if t['run_id'] == state['active_run_id'])
    phase = trial.get('phase', {})
    stamp = datetime.now(timezone.utc).isoformat()
    notes = {
        'ASR-16': f'{stamp}: CUDA update/fix SUCCESS in isolated NeMo runtime. Concrete DLL mismatch: Torch-first import bound CUDA12.8 nvJitLink while compiler NVVM was12.9. NEW launcher preloads pinned12.9 linker before imports and records actual binding owner/version. Root-owned session28368 completed EXIT0; native NeMo GPU loss/backward passed all3 cases on saved real Turkish/English FLEURS TRAIN logits, full8199-token/5-duration width. Maximum loss error2.29e-5 and gradient relativeL2<=1.80e-5. This proves loss-kernel correctness, not whole-model or training throughput. Same B8x2 complete training-update comparison is being prepared now; no need wait for the whole3h quality loop.',
        'ASR-15': f'{stamp}: Restart FIXED: new IO2/cache5/memory8 dispatch integrated and root-owned controller5 session95658 is active in fresh adaptive_20261011_public_data_02 namespace. Current genuine phase: {phase}; optimizer updates published: {trial.get("published_steps",0)}. Same selected1871 warmstart, expanded57.3h TRAIN pool, B8x2 and DEV every250. Local preparation mirror session61540 is logging real verification/cache counts in the SAME Trackio project before optimizer records exist. Historical curves kept visible. Quality loop continues; target still unmet.',
    }
    for task, note in notes.items():
        subprocess.run([str(OUT/'tools/backlog.exe'), 'task', 'edit', task,
                        '--append-notes', note, '--add-ref', 'nemo_native_loss_linker_v4_verification.json'],
                       cwd=OUT, check=True, capture_output=True, text=True, encoding='utf-8')
    subprocess.run([str(ROOT/'work/training-env/Scripts/python.exe'),
                    str(OUT/'refresh_backlog.py')], cwd=ROOT, check=True)
    print(json.dumps({'backlog_updated':True,'native_GPU_loss_passed':True,'actual_phase':phase}))

if __name__ == '__main__':
    main()
