"""Record measured closed HF return and the chosen next controlled arm."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def main():
    run = OUT / "runs/adaptive_20261011_hf_return_01"
    history = json.loads((run / "training_history.json").read_text(encoding="utf-8"))
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    start, end = history[861], history[999]
    assert start["step"] == 862 and end["step"] == 1000
    assert selection["stop_early_reason"] == "consecutive_retention_gate_failures"
    rate = (end["audio_seen_seconds"] - start["audio_seen_seconds"]) / (end["train_wall_seconds"] - start["train_wall_seconds"])
    note = (
        datetime.now(timezone.utc).isoformat() + ": Actual HF return completed 138 new optimizer updates (862 to1000), "
        f"B8 x accumulation2, official NeMo TDT kernel, at {rate:.4f} combined audio min per training minute; "
        "startup/DEV/checkpoint time excluded. Full NeMo stable B1 x16 measured41.5946, so the measured HF interval is "
        f"{rate/41.5945711:.3f}x faster. Trackio v3 actually switched to adaptive_20261011_hf_return_01 in the same project. "
        "DEV1000 Turkish64.0000% improved from retained qualifying best64.9697%, but Spanish53/1289 word errors exceeds the gate's52; "
        "EN/IT/FR/DE passed. The exact continuation correctly stopped after two consecutive retention failures, exit0; "
        "neither inherited failure counters nor retention thresholds were reset. Best qualifying weights remain separately preserved. "
        "Next controlled arm: NEW experiment / fresh optimizer and sampler warm-start from the qualifying best, "
        "increase only replay token KL5 to10; keep head LR5e-5, LoRA LR3e-4, feature KD1, duration KD0.5, "
        "row multiplier6, B8 x2, replay50%, full57.2988h Turkish pool and DEV cadence250 unchanged. "
        "No next-arm WER claim before actual evaluation. Competitive Turkish WER remains incomplete and active. "
        "Startup audit measured334.831s (91.48% of preparation); unchanged-data header reuse is a future targeted optimization."
    )
    for task in ("ASR-2", "ASR-15", "ASR-16"):
        subprocess.run([str(OUT / "tools/backlog.exe"), "task", "edit", task, "--append-notes", note,
                        "--add-ref", "runs/adaptive_20261011_hf_return_01/selection.json"],
                       cwd=OUT, check=True, capture_output=True, text=True, encoding="utf-8")
    subprocess.run([str(ROOT / "work/training-env/Scripts/python.exe"), str(OUT / "refresh_backlog.py")], cwd=ROOT, check=True)
    print(json.dumps({"backlog_updated": True, "measured_HF_interval_audio_min_per_min": rate,
                      "next_arm": "replay_KL10", "objective_complete": False}))


if __name__ == "__main__":
    main()
