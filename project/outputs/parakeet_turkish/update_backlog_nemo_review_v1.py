"""Record actual local NeMo readiness and closed continuation03 facts."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def main():
    stamp = datetime.now(timezone.utc).isoformat()
    status = json.loads((OUT/'runs/adaptive_20261010_continuation_03/loop_status.json').read_text())
    trial = status['trials'][-1]
    assert status['status'] == 'resource_review_required' and trial['exit_code'] == 0
    assert trial['durable_completed_updates'] == 1871
    assert trial['selection']['turkish_validation_wer'] == 0.6496969696969697
    history = json.loads((OUT/'runs/adaptive_20261010_continuation_03_resume_B8/training_history.json').read_text())
    assert history[-1]['validation']['retention_gate_passed']
    before, after = history[1820], history[-1]
    rate = (after['audio_seen_seconds']-before['audio_seen_seconds'])/(after['train_wall_seconds']-before['train_wall_seconds'])
    cpu = json.loads((ROOT/'work/nemo_native_setup_v1/cpu_preflight.json').read_text())
    assert cpu['passed']
    notes = {
        'ASR-15': f'{stamp}: Continuation03 saved step1871 after50 new updates, Turkish fixed DEV WER64.9697%, all five EN/ES/IT/FR/DE retention gates passed. Physical batch8, accumulation2. Actual suffix throughput{rate:.2f} audio min/min over50 updates. No new TEST or export. Competitive DEV target remains11.9394%; goal incomplete. Next expanded public-data/native NeMo trials are being prepared; this segment exited cooperatively at a memory guard.',
        'ASR-16': f'{stamp}: Official NVIDIA-NeMo/Speech fine-tuning script/config reviewed at commit e4a34fd25f895ecd7f64f0e56bc288165e078da9. Released v3 config uses TDT sigma.02/omega.1/mean_volume; generic tokenizer replacement can reset learned decoder/joint and needs explicit row preservation. Isolated native Windows NeMo3.0.0 plus Lightning2.4.0 install and CPU imports passed without modifying active Torch; full pinned native v3 .nemo downloaded. Native CUDA loss and full native trainer not yet verified. Actual9GB cap/4GB cleanup probe completed100 updates at77.16 audio min/min, matched-prefix gradients/weights/RNG bitwise equal. Production03 completed50 cleanups with last post-update reserve3.03GB; guard stopped at legacy Windows commit96.24%. GlobalMemoryStatusEx process/system semantics under review; historical labels must not be treated as verified system-wide commit. TRAIN-only audit also found default CTC grouped decoding excluding legitimate repeated TDT tokens; future data policy will use group_tokens=False.',
    }
    for task, note in notes.items():
        subprocess.run([str(OUT/'tools/backlog.exe'),'task','edit',task,'--append-notes',note,
                        '--add-ref','nemo_training_review_v1.json','--add-ref','adaptive_20261010_continuation_03_report.json'],
                       cwd=OUT, check=True, capture_output=True, text=True, encoding='utf8')
    subprocess.run([str(ROOT/'work/training-env/Scripts/python.exe'),str(OUT/'refresh_backlog.py')],check=True,cwd=ROOT)
    print(json.dumps({'backlog_updated':True,'suffix_audio_min_per_min':rate,'durable_step':1871}))


if __name__ == '__main__':
    main()
