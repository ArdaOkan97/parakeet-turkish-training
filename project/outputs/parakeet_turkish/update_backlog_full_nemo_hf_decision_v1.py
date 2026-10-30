"""Record the measured full NeMo result and the main-pipeline decision."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def main():
    run = OUT / "runs/full_nemo_turkish_20261011_clean_warmstart_08"
    outcome = json.loads((run / "trial_outcome.json").read_text(encoding="utf-8"))
    gate = json.loads((run / "development_gate_step_350.json").read_text(encoding="utf-8"))
    final = json.loads((run / "development_gate_step_364.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (run / "training_telemetry.jsonl").read_text(encoding="utf-8").splitlines()]
    boundary = next(row for row in rows if row["step"] == 250)
    assert outcome["status"] == "complete" and outcome["stop_reason"] == "owned_stop_file"
    assert outcome["actual_completed_optimizer_updates"] == 364 and gate["retention_gate_passed"] and final["retention_gate_passed"]
    note = (
        datetime.now(timezone.utc).isoformat() + ": Main-pipeline decision after the actual full NeMo comparison: return to the previous HF LoRA/KD pipeline, "
        "which already uses the official NeMo TDT CUDA kernel. The full NeMo Trainer has not yet earned adoption. "
        f"Clean-label B1 x accumulation16 completed 250 new updates at {boundary['audio_minutes_per_wall_minute']:.4f} combined audio min/min "
        "(startup and DEV excluded), versus the prior B8 x accumulation2 continuation's measured 80.6451. "
        f"Native Turkish DEV improved from 94.7879% at cumulative100 to {100*gate['turkish_wer']:.4f}% at350, "
        f"then {100*final['turkish_wer']:.4f}% at the cooperative final364; all five retention gates passed. "
        "Root observed native exit0 and preserved complete qualified/recovery states. This is learning progress, "
        "but the previous qualifying best remains 64.9697%; recipe/exposure differences prevent a controlled WER comparison. "
        "The native350 decode still emits none of the six new Turkish characters despite positive TRAIN targets. "
        "The previous HF pipeline already learned those characters and applies its correctly mapped six-row Adam delta multiplier. "
        "HF latest862 has Turkish DEV65.0909% and one Spanish gate failure; the separately retained best0 passes all gates. "
        "An exact HF862 continuation is being prepared with weights, all 207 Adam states, sampler, RNG, history and best0 retained. "
        "Only 9.209% of the 74,883-clip / 57.2988-hour Turkish pool has been covered, so this is not evidence of a full-pass plateau. "
        "Full NeMo checkpoints, cleaned labels and TRAIN-only vocabulary candidates remain available for future bounded experiments. "
        "The standalone stock-loss checkpoint oracle v1 failed before numeric comparison because NeMo typed methods require keyword arguments. "
        "The corrected v2 actual GPU oracle completed all eight FP32/BF16, TDT/RNNT, raw/mean_volume cases with bit-exact loss, "
        "all joint gradients and preserved RNG/reduction state. This does not establish whole-update memory, throughput or WER gains, "
        "and the hook is not installed in the resumed main pipeline. Competitive WER and multilingual retention remain the active goal."
    )
    for task in ("ASR-2", "ASR-15", "ASR-16"):
        subprocess.run([str(OUT / "tools/backlog.exe"), "task", "edit", task, "--append-notes", note,
                        "--add-ref", "runs/full_nemo_turkish_20261011_clean_warmstart_08/development_gate_step_350.json",
                        "--add-ref", "stock_loss_checkpoint_actual_GPU_oracle_v2.json"],
                       cwd=OUT, check=True, capture_output=True, text=True, encoding="utf-8")
    subprocess.run([str(ROOT / "work/training-env/Scripts/python.exe"), str(OUT / "refresh_backlog.py")], cwd=ROOT, check=True)
    print(json.dumps({"backlog_updated": True, "full_nemo_exit": 0, "decision": "return_main_HF_B8_pipeline"}))


if __name__ == "__main__":
    main()
