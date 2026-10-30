"""Record verified native recovery and the SentencePiece label diagnosis."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    failed = OUT / "runs/full_nemo_turkish_20261011_pilot_06"
    verified = OUT / "runs/full_nemo_turkish_20261011_warmstart_07"
    outcome = read(failed / "trial_outcome.json")
    complete = read(verified / "trial_outcome.json")
    gate = read(verified / "development_gate_step_100.json")
    restored = read(verified / "optimizer_warmstart_restoration.json")
    assert outcome["status"] == "failed" and outcome["actual_completed_optimizer_updates"] == 147
    assert complete["status"] == "complete" and complete["local_completed_optimizer_updates"] == 1
    assert restored["scheduler_last_epoch"] == 100 and not restored["optimizer_or_scheduler_step_called_during_restore"]
    assert gate["retention_gate_passed"]
    note = (datetime.now(timezone.utc).isoformat() + ": Native B2xaccum8/cap9 completed147 updates at62.42 combined audio min/min, "
            "then OOM on the predicted worst long TR pair requesting a1.35GiB label-gradient buffer. "
            "Periodic official Lightning checkpoints50 and100 were actually saved, preserving the first100 updates. "
            "The native checkpoint100 was inspected: Adam completed100, Cosine last_epoch100/horizon9361; "
            "stock batch-end checkpoint counters lag by one (ready800/completed799). "
            "Recovery therefore loads strict native weights and Adam/Cosine into a fresh official Lightning loop with explicit suffix1600; "
            "it does not silently restore the old B2 loop counters. A bounded one-update B1x16 verification actually completed, "
            "preserved scheduler position (no repeated warmup/extra step), and saved complete trained state101. "
            f'Native100 common Turkish DEV WER {100*gate["turkish_wer"]:.4f}%, all five retention gates pass; '
            "native101 DEV94.5455%, all gates pass. This remains worse than the earlier project-best HF64.9697%. "
            "The one-update cold recovery rate is not a sustained throughput benchmark. "
            "A separate SentencePiece audit found64 meaningful TR TRAIN roundtrip mismatches and83/1595 replay mismatches "
            "(including61 French) where unsupported punctuation becomes unknown tokens and inserts word boundaries. "
            "No ordinary letters or numbers are missing. Clean native TRAIN labels are being prepared by removing only unrepresentable punctuation, "
            "preserving all recordings/source IDs/order and all original benchmark references. "
            "The main full-pass arm will use corrected TRAIN labels. "
            "Inert TRAIN-only +512/+1024 BPE candidates reduce heldout TRAIN token counts28.18%/34.77%; "
            "some old-language encodings change, so neither candidate is activated or claimed to improve WER. "
            "The competitive Turkish quality goal remains active.")
    for task in ("ASR-2", "ASR-15", "ASR-16"):
        subprocess.run([str(OUT / "tools/backlog.exe"), "task", "edit", task, "--append-notes", note,
                        "--add-ref", "runs/full_nemo_turkish_20261011_warmstart_07/development_gate_step_100.json",
                        "--add-ref", "native_sentencepiece_roundtrip_diagnosis_v1.json"],
                       cwd=OUT, check=True, capture_output=True, text=True, encoding="utf-8")
    subprocess.run([str(ROOT / "work/training-env/Scripts/python.exe"), str(OUT / "refresh_backlog.py")],
                   cwd=ROOT, check=True)
    print(json.dumps({"backlog_updated": True, "native_source_checkpoint_updates": 100,
                      "native_recovery_updates": 1, "native_DEV100_wer": gate["turkish_wer"]}))


if __name__ == "__main__":
    main()
