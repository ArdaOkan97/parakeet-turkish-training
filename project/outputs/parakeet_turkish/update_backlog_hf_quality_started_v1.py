"""Append the actual launched quality-loop state to the existing board."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def main():
    path = OUT / "runs/adaptive_20261011_hf_quality_window_01/loop_status.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    arm = state["trials"][0]
    assert state["status"] == "running" and arm["owned_launcher_pid"] > 0
    assert arm["recipe"] == {"name": "stronger_kd", "replay_kl": 10.0, "head_lr": 5e-5, "lora_lr": 3e-4}
    note = (
        datetime.now(timezone.utc).isoformat() + f": Root ACTUALLY launched {arm['run_id']} "
        f"(owned launcher PID{arm['owned_launcher_pid']}; phase {arm['phase']['phase']}, published step{arm['published_step']}). "
        "Warm-starts an immutable byte-identical copy of qualifying HFbest0. This is an explicitly NEW scientific experiment, "
        "fresh Adam/sampler/RNG/history/counters, changing replay tokenKD5 to10 only. B8x2/full57.2988hTR/50%replay/"
        "head5e-5/LoRA3e-4/row6/featureKD1/durationKD0.5/cadence250/8GBcap retained. "
        "Same existing Trackio v3 follow mirror will switch when actual config/history publish; no new DEV claim yet. "
        "Controller has a3h bounded window and continues automatically into the declared head2e-5/LoRA1e-4/KD10 arm "
        "only after consecutive retention failures and enough time remains. Goal remains active after the window. "
        "HF1000 error analysis shows443 substitutions/49insertions/36deletions, zero empty transcripts, and emission of all new "
        "lowercase Turkish characters; only10.68% of theTRpool has been seen. Continue evidence from a fullpass before claiming plateau."
    )
    for task in ("ASR-2", "ASR-15", "ASR-16"):
        subprocess.run([str(OUT / "tools/backlog.exe"), "task", "edit", task, "--append-notes", note,
                        "--add-ref", "runs/adaptive_20261011_hf_quality_window_01/loop_status.json"],
                       cwd=OUT, check=True, capture_output=True, text=True, encoding="utf-8")
    subprocess.run([str(ROOT / "work/training-env/Scripts/python.exe"), str(OUT / "refresh_backlog.py")], cwd=ROOT, check=True)
    print(json.dumps({"backlog_updated": True, "actual_run_id": arm["run_id"], "phase": arm["phase"]}))


if __name__ == "__main__":
    main()
