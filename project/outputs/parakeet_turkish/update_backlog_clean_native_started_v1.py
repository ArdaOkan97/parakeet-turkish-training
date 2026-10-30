"""Record the actual cleaned-label native continuation launch."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = "full_nemo_turkish_20261011_clean_warmstart_08"


def main():
    run_dir = OUT / "runs" / RUN
    config = json.loads((run_dir / "resolved_trial_config.json").read_text(encoding="utf-8"))
    proof = json.loads((OUT / "full_nemo_clean_warmstart_readiness_v10.json").read_text(encoding="utf-8"))
    assert proof["passed"] and config["warmstart"]["source_completed_optimizer_updates"] == 100
    assert config["model"]["train_ds"]["batch_size"] == 1
    assert config["trainer"]["accumulate_grad_batches"] == 16
    assert (run_dir / "training_telemetry.jsonl").exists()
    note = (
        datetime.now(timezone.utc).isoformat() + ": Actually launched " + RUN +
        " with official full NeMo Lightning training, B1xaccum16 and the unchanged9GB allocator cap/Windows commit guard. "
        "It restores native06 checkpoint100 weights, Adam and Cosine position100/horizon9361 into fresh local loops, "
        "then starts at ordered recording1600. Remaining9261 local updates finish one complete57.3h Turkish clip pass. "
        "The prior experimental native07 update101 is preserved but is not the warmstart. "
        "All76478 unique TRAIN sources now pass normalized SentencePiece target roundtrip: unsupported punctuation removed "
        "from2242 Turkish and301 replay labels; source/audio/order/counts remain exact, original references are retained "
        "on changed rows, all six DEV manifests remain unchanged. One harmless zero-width-joiner unknown remains and is disclosed. "
        "This is a declared training-label/physical-grouping recipe change, not an exact-gradient continuation. "
        "Trackio mirrorv3 actually initialized this run in the existing parakeet-turkish project; local metrics are offset100 "
        "and initial DEV is cumulative100. First new common DEV point is local250/cumulative350, then local1000/cumulative1100 "
        "and each1000 local updates. Full official recovery checkpoints every250 local updates retain the newest periodic copy. "
        "No trained quality or sustained throughput gain is claimed at launch. Current project-best qualifying Turkish DEV "
        "remains64.9697%; competitive target<=11.9394% and all five retention gates remain active. "
        "The stock-loss activation-checkpoint hook and larger Turkish BPE candidates remain inert pending actual GPU/quality proof."
    )
    for task in ("ASR-2", "ASR-15", "ASR-16"):
        subprocess.run(
            [str(OUT / "tools/backlog.exe"), "task", "edit", task, "--append-notes", note,
             "--add-ref", "runs/" + RUN + "/resolved_trial_config.json",
             "--add-ref", "full_nemo_clean_TRAIN_continuation_v1.json"],
            cwd=OUT, check=True, capture_output=True, text=True, encoding="utf-8",
        )
    subprocess.run([str(ROOT / "work/training-env/Scripts/python.exe"), str(OUT / "refresh_backlog.py")],
                   cwd=ROOT, check=True)
    print(json.dumps({"backlog_updated": True, "run_id": RUN, "stage": "actually_launched"}))


if __name__ == "__main__":
    main()
