"""Reconstruct sampled training audio duration and observed wall-clock throughput.

CPU only. Uses the verified accepted source IDs and exact seeded cycling order;
does not load a model or read held-out predictions. Never edits the trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parent.parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Cycle:
    def __init__(self, rows, rng):
        self.rows = list(rows)
        self.rng = rng
        self.position = len(rows)
        self.seen = set()

    def draw(self):
        if self.position == len(self.rows):
            self.rng.shuffle(self.rows)
            self.position = 0
        row = self.rows[self.position]
        self.position += 1
        self.seen.add(row["source_id"])
        return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="lora_20261010_01")
    args = parser.parse_args()
    run_dir = OUTPUT / "runs" / args.run_id
    config_path = run_dir / "run_config.json"
    history_path = run_dir / "training_history.json"
    audit_path = OUTPUT / "lora_training_data_verification.json"
    config, history, audit = map(read_json, (config_path, history_path, audit_path))
    assert audit["passed"] and audit["run_id"] == config["run_id"] == args.run_id
    assert config["resume_weights"] is None, "A restarted RNG cannot reconstruct an exact resume."
    assert config["sampling"] == "without-replacement shuffled cycles; retention language uniformly selected"
    assert config["accumulation"] % 2 == 0
    assert sha(OUTPUT / "train_lora.py") == config["code_sha256"]["train_lora.py"]

    tr_path = ROOT / "work/data/tr_train.jsonl"
    replay_path = ROOT / "work/data/replay_train.jsonl"
    supplement_path = ROOT / config["supplement"] if config["supplement"] else None
    manifests = [tr_path, replay_path] + ([supplement_path] if supplement_path else [])
    for path in manifests:
        assert sha(path) == audit["input_file_sha256"][str(path.resolve())]
    tr = read_rows(tr_path)
    random.Random(config["seed"]).shuffle(tr)
    tr = tr[:config["train_limit"]]
    if supplement_path:
        tr += read_rows(supplement_path)
    replay = read_rows(replay_path)
    accepted_tr = (audit["streams"]["turkish_commonvoice"]["accepted_source_ids_and_labels"]
                   + audit["streams"].get("turkish_fleurs_train", {}).get("accepted_source_ids_and_labels", []))
    accepted_replay = audit["streams"]["retention_fleurs_train"]["accepted_source_ids_and_labels"]
    tr_ids = {r["source_id"] for r in accepted_tr}
    replay_ids = {r["source_id"] for r in accepted_replay}
    tr = [r for r in tr if r["source_id"] in tr_ids]
    replay = [r for r in replay if r["source_id"] in replay_ids]
    # The verified audit preserves trainer candidate/accepted order.
    assert [r["source_id"] for r in tr] == [r["source_id"] for r in accepted_tr]
    assert [r["source_id"] for r in replay] == [r["source_id"] for r in accepted_replay]
    assert len(tr) == config["turkish_clips_used"]
    assert len(replay) == config["replay_clips_used"]
    for rows, accepted in ((tr, accepted_tr), (replay, accepted_replay)):
        for row, checked in zip(rows, accepted):
            assert row["language"] == checked["language"]
            assert math.isclose(row["duration"], checked["duration"], abs_tol=1e-9)

    languages = ["en", "es", "it", "fr", "de"]
    rng = random.Random(config["seed"] + 1)
    tr_stream = Cycle(tr, rng)
    replay_streams = {lang: Cycle([r for r in replay if r["language"] == lang], rng) for lang in languages}
    assert all(stream.rows for stream in replay_streams.values())
    assert [r["step"] for r in history] == list(range(1, len(history) + 1))
    increments = []
    sampled = []
    totals = {lang: {"clips": 0, "seconds": 0.0} for lang in ["tr"] + languages}
    previous_elapsed = 0.0
    for index, record in enumerate(history):
        step_audio = {lang: 0.0 for lang in ["tr"] + languages}
        step_ids = []
        for micro in range(config["accumulation"]):
            row = tr_stream.draw() if micro % 2 == 0 else replay_streams[rng.choice(languages)].draw()
            lang = row["language"]
            totals[lang]["clips"] += 1
            totals[lang]["seconds"] += row["duration"]
            step_audio[lang] += row["duration"]
            step_ids.append(row["source_id"])
        assert record["turkish_unique_clips_seen"] == len(tr_stream.seen)
        delta = record["elapsed_seconds"] - previous_elapsed
        assert delta > 0
        previous_elapsed = record["elapsed_seconds"]
        after_eval = index > 0 and "validation" in history[index - 1]
        increments.append({"step": record["step"], "wall_seconds": delta,
                           "audio_seconds": sum(step_audio.values()), "turkish_audio_seconds": step_audio["tr"],
                           "includes_previous_dev_and_checkpoint_pause": after_eval})
        sampled.append({"step": record["step"], "source_ids": step_ids})

    total_audio = sum(v["seconds"] for v in totals.values())
    elapsed = history[-1]["elapsed_seconds"]
    training_intervals = [r for r in increments if not r["includes_previous_dev_and_checkpoint_pause"]]
    train_wall = sum(r["wall_seconds"] for r in training_intervals)
    train_audio = sum(r["audio_seconds"] for r in training_intervals)
    train_tr_audio = sum(r["turkish_audio_seconds"] for r in training_intervals)
    window = training_intervals[-100:]
    draw_bytes = json.dumps(sampled, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    report = {
        "run_id": args.run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": "CPU only", "completed_steps": len(history),
        "method": "Exact seeded replay of verified accepted training rows and cycling/RNG; not a pool-average estimate",
        "input_sha256": {str(p): sha(p) for p in [config_path, history_path, audit_path] + manifests},
        "trainer_sha256": sha(OUTPUT / "train_lora.py"), "sampled_source_sequence_sha256": hashlib.sha256(draw_bytes).hexdigest(),
        "counts": {"clip_presentations": sum(v["clips"] for v in totals.values()),
                   "turkish_unique_clips": len(tr_stream.seen),
                   "replay_unique_clips": sum(len(s.seen) for s in replay_streams.values())},
        "sampled_audio_by_language": {k: {**v, "minutes": v["seconds"] / 60, "hours": v["seconds"] / 3600} for k, v in totals.items()},
        "logged_span": {"audio_minutes": total_audio / 60, "wall_minutes": elapsed / 60,
                        "audio_min_per_wall_min": total_audio / elapsed,
                        "turkish_audio_min_per_wall_min": totals["tr"]["seconds"] / elapsed,
                        "includes_mid_run_dev_evaluations": True,
                        "excludes_final_step_dev_export_and_final_benchmark": True},
        "training_intervals_without_dev_pauses": {
            "updates": len(training_intervals), "excluded_update_steps": [r["step"] for r in increments if r["includes_previous_dev_and_checkpoint_pause"]],
            "audio_minutes": train_audio / 60, "wall_minutes": train_wall / 60,
            "audio_min_per_wall_min": train_audio / train_wall,
            "turkish_audio_min_per_wall_min": train_tr_audio / train_wall,
            "clips_per_second": len(training_intervals) * config["accumulation"] / train_wall,
            "median_wall_seconds_per_update": statistics.median(r["wall_seconds"] for r in training_intervals),
            "method": "Exclude each full interval after a dev evaluation from both numerator and denominator; remaining intervals include audio loading, forward/backward, optimizer, and routine log/checkpoint I/O"},
        "last_100_training_intervals": {"updates": len(window), "first_step": window[0]["step"], "last_step": window[-1]["step"],
            "audio_min_per_wall_min": sum(r["audio_seconds"] for r in window) / sum(r["wall_seconds"] for r in window)},
        "caveats": ["Timer starts after model initialization and teacher-cache preparation.",
                    "Audio minutes count every optimizer presentation, including repeated clips and replay; excludes dev/test audio.",
                    "The last update's elapsed timestamp is recorded before final dev evaluation, so the logged span excludes that last evaluation.",
                    "No profiler-derived GPU-only compute time is available; the training-only estimate isolates full non-dev intervals.",
                    "Durations are verified decoded WAV durations, not padded lengths or upstream source metadata."],
        "step_intervals": increments,
    }
    target = OUTPUT / "training_audio_throughput.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(target), "counts": report["counts"], "sampled_audio_by_language": report["sampled_audio_by_language"],
                      "logged_span": report["logged_span"], "training_intervals_without_dev_pauses": report["training_intervals_without_dev_pauses"],
                      "last_100_training_intervals": report["last_100_training_intervals"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
