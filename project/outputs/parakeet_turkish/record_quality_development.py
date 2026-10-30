"""Record a completed DEV decision in canonical Backlog tasks once per point."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PHASES = {
    "HF_TSC_BPE512_new_rows_only_acquisition_v1": (
        "source_selected_step", "acquisition_updates_completed", "TSC_BPE_acquisition_initial_source_equivalence.json",
        "TSC_BPE512_acquisition_before_update_source_equivalence_v1", "before_optimizer_creation"),
    "HF_TSC_BPE512_new_rows_plus_encoder_LoRA_v1": (
        "source_phase1_step", "adaptation_updates_completed", "TSC_BPE_LoRA_initial_source_equivalence.json",
        "TSC_BPE512_LoRA_before_update_source_equivalence_v1", "before_phase2_updates"),
}


def main(args):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("Safe run ID required")
    run = OUT / "runs" / args.run_id
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    history = json.loads((run / "training_history.json").read_text(encoding="utf-8"))
    points = [row for row in history if "validation" in row]
    phase = PHASES.get(config.get("phase"))
    if phase:
        source_step = config[phase[0]]
        points = [row for row in points if type(row.get(phase[1])) is int
                  and row[phase[1]] > 0 and row["step"] == source_step + row[phase[1]]]
    if not points:
        raise ValueError("No completed development point")
    row = next(row for row in points if row["step"] == args.step) if args.step else points[-1]
    step, gate = row["step"], row["validation"]
    source = OUT / "results" / f"{args.run_id}_step_{step}" / "summary.json"
    summary = json.loads(source.read_text(encoding="utf-8"))
    if not all(summary.get(f"validation_{lang}", {}).get("asr", {}).get("utterances") == 50
               and type(summary[f"validation_{lang}"]["asr"].get("word_errors")) is int
               for lang in ("tr", "en", "es", "it", "fr", "de")):
        raise ValueError("Complete fixed300 actual saved DEV required")
    if summary["validation_tr"]["asr"]["wer"] != gate["turkish_wer"]:
        raise ValueError("DEV summary and recorded decision differ")
    ledger = run / "backlog_development_receipts.json"
    receipts = json.loads(ledger.read_text(encoding="utf-8")) if ledger.exists() else {}
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    key = str(step)
    if key in receipts:
        if receipts[key]["summary_sha256"] != digest:
            raise ValueError("Previously recorded DEV summary changed")
        print(json.dumps({"already_recorded": True, "run_id": args.run_id, "step": step}))
        return
    initial_path = run / "initial_development.json"
    initial_label = "initial"
    phase_throughput_note = ""
    if phase:
        proof_path = run / phase[2]
        proof_raw = proof_path.read_bytes()
        proof = json.loads(proof_raw)
        if not (hashlib.sha256(proof_raw).hexdigest() == config["config_identity"]["initial_source_equivalence_sha256"]
                and config["initial_source_equivalence"] == proof and proof.get("kind") == phase[3]
                and proof.get("passed") is True and proof.get("run_id") == args.run_id
                and proof.get("source_selected_step") == source_step and proof.get("actual_GPU_decode") is True
                and proof.get(phase[4]) is True and proof.get("compared_fixed_DEV_utterances") == 300
                and proof.get("all_five_retention_gates_passed") is True):
            raise ValueError("Actual hash-bound new-phase source300 preservation proof required")
        initial = {"turkish_wer": proof["comparisons"]["tr"]["asr"]["wer"]}
        initial_label = f"phase initial DEV{source_step}"
        baseline_row = history[source_step - 1]
        if baseline_row["step"] != source_step:
            raise ValueError("Inherited phase baseline frontier differs")
        train_delta = row["train_wall_seconds"] - baseline_row["train_wall_seconds"]
        audio_delta = row["audio_seen_seconds"] - baseline_row["audio_seen_seconds"]
        tr_delta = row["turkish_audio_seen_seconds"] - baseline_row["turkish_audio_seen_seconds"]
        if train_delta <= 0 or audio_delta <= 0 or not 0 <= tr_delta <= audio_delta:
            raise ValueError("Positive actual post-phase training counters required")
        phase_throughput_note = (f"Actual local phase updates{row[phase[1]]}; post-phase combined training throughput"
                                 f"{audio_delta/train_delta:.3f} and TR{tr_delta/train_delta:.3f} audio min/min; ")
    elif initial_path.exists():
        initial = json.loads(initial_path.read_text(encoding="utf-8"))
    else:
        source_check = config["optimization_transition"]["initial_source_equivalence"]
        proof_path = Path(source_check["path"])
        if hashlib.sha256(proof_path.read_bytes()).hexdigest() != source_check["sha256"]:
            raise ValueError("Initial state-fork source equivalence proof changed")
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        if not (proof["passed"] and proof["run_id"] == args.run_id
                and proof["compared_fixed_DEV_utterances"] == 300
                and proof["all_five_retention_gates_passed"]):
            raise ValueError("Verified initial state-fork DEV evidence required")
        initial = {"turkish_wer": proof["comparisons"]["tr"]["asr"]["wer"]}
        initial_label = f"fork initial DEV{proof['source_selected_step']}"
    diagnostics = row["selection_diagnostics"]
    coverage = row.get("Turkish_domain_coverage", {})
    draw_epochs = coverage.get("turkish_draw_equivalent_epochs", row["turkish_epoch"])
    minimum_domain_epochs = coverage.get("minimum_domain_exposure_epochs")
    exposure_note = f"TR draw-equivalent epochs{draw_epochs:.4f}; "
    if minimum_domain_epochs is not None:
        exposure_note += f"minimum-domain exposure epochs{minimum_domain_epochs:.4f}; "
    exposure_note += f"{row['turkish_unique_clips_seen']} unique clips; "
    note = (
        datetime.now(timezone.utc).isoformat() + f": Actual {args.run_id} DEV{step}: "
        f"Turkish WER{100*gate['turkish_wer']:.4f}% ({initial_label} {100*initial['turkish_wer']:.4f}%); "
        f"all five retention gates passed={gate['retention_gate_passed']}; "
        f"improved qualifying best={diagnostics['improved_qualifying_best']}. "
        + exposure_note +
        phase_throughput_note +
        f"cumulative combined training throughput{row['audio_minutes_per_wall_minute']:.3f} audio min/min "
        "(startup/DEV/checkpoint serialization excluded); "
        f"observed process peak CUDA allocation{row['peak_allocated_vram_gb']:.3f}GB. "
        f"Consecutive retention failures{diagnostics['consecutive_retention_failures']}; "
        "checkpoint selection uses original strict gates, fixed FLEURS DEV only; no TEST or target completion claimed."
    )
    reference = source.relative_to(OUT).as_posix()
    subprocess.run([str(OUT / "tools/backlog.exe"), "task", "edit", "ASR-15", "--append-notes", note,
                    "--add-ref", reference], cwd=OUT, check=True, capture_output=True, text=True, encoding="utf-8")
    receipts[key] = {"summary_sha256": digest, "recorded_at_utc": datetime.now(timezone.utc).isoformat()}
    ledger.write_text(json.dumps(receipts, indent=2) + "\n", encoding="utf-8")
    subprocess.run([str(ROOT / "work/training-env/Scripts/python.exe"), str(OUT / "refresh_backlog.py")], cwd=ROOT, check=True)
    print(json.dumps({"backlog_updated": True, "run_id": args.run_id, "step": step,
                      "turkish_wer": gate["turkish_wer"], "retention_gate_passed": gate["retention_gate_passed"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", type=int)
    main(parser.parse_args())
