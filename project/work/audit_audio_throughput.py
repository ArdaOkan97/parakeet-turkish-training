"""Independently reconstruct the completed sampler using CPU/Python only."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/parakeet_turkish"
RUN_ID = "lora_20261010_01"
RUN = OUTPUT / "runs" / RUN_ID
CONFIG_PATH = RUN / "run_config.json"
HISTORY_PATH = RUN / "training_history.json"
ACCEPTANCE_PATH = OUTPUT / "lora_training_data_verification.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def ids_sha(identifiers):
    return hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest()


checks = []


def check(name, condition):
    passed = bool(condition)
    checks.append({"name": name, "passed": passed})
    if not passed:
        raise AssertionError(name)


config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
acceptance = json.loads(ACCEPTANCE_PATH.read_text(encoding="utf-8"))
check("saved run IDs agree", config["run_id"] == acceptance["run_id"] == RUN_ID)
check("acceptance audit passed", acceptance["passed"])
check("completed history exactly configured1200 contiguous steps", config["steps"] == len(history) == 1200
      and [record["step"] for record in history] == list(range(1, 1201)))
check("effective batch4", config["accumulation"] == 4)
check("current source matches actual training source SHA", sha(OUTPUT / "train_lora.py") == config["code_sha256"]["train_lora.py"])

language_order = None
for node in ast.parse((OUTPUT / "benchmark.py").read_text(encoding="utf-8")).body:
    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "LANGUAGES" for target in node.targets):
        language_order = ast.literal_eval(node.value)
check("actual language dictionary insertion order", list(language_order) == ["en", "es", "it", "fr", "de", "tr"])
replay_languages = [language for language in language_order if language != "tr"]

cv_path = ROOT / "work/data/tr_train.jsonl"
supplement_path = ROOT / config["supplement"]
replay_path = ROOT / "work/data/replay_train.jsonl"
source_paths = (cv_path, supplement_path, replay_path)
for path in source_paths:
    check("unchanged manifest " + path.name, sha(path) == acceptance["input_file_sha256"][str(path)])
initial_cv = rows(cv_path)
random.Random(config["seed"]).shuffle(initial_cv)
initial_cv = initial_cv[:config["train_limit"]]
supplement = rows(supplement_path)
initial_replay = rows(replay_path)

accepted_rows = {}
for group, candidates in (("turkish_commonvoice", initial_cv), ("turkish_fleurs_train", supplement),
                          ("retention_fleurs_train", initial_replay)):
    audited = acceptance["streams"][group]["accepted_source_ids_and_labels"]
    wanted = {row["source_id"] for row in audited}
    accepted_rows[group] = [row for row in candidates if row["source_id"] in wanted]
    check("accepted source IDs and order match " + group,
          [row["source_id"] for row in accepted_rows[group]] == [row["source_id"] for row in audited])
    check("accepted durations match " + group,
          all(float(row["duration"]) == float(item["duration"]) for row, item in zip(accepted_rows[group], audited)))
    check("only official TRAIN provenance " + group, all(row["dataset_split"] == "train" for row in accepted_rows[group]))

turkish = accepted_rows["turkish_commonvoice"] + accepted_rows["turkish_fleurs_train"]
replay = accepted_rows["retention_fleurs_train"]
check("actual accepted pool counts", len(turkish) == config["turkish_clips_used"] == 2166
      and len(replay) == config["replay_clips_used"] == 1553)
by_language = {language: [row for row in replay if row["language"] == language] for language in replay_languages}
check("all5 replay streams positive", all(by_language.values()))


class IndependentCycler:
    def __init__(self, stream_rows, rng):
        self.rows = list(stream_rows)
        self.rng = rng
        self.position = len(self.rows)
        self.seen = set()
        self.draws = 0
        self.cycles = []

    def draw(self):
        if self.position >= len(self.rows):
            self.rng.shuffle(self.rows)
            self.position = 0
            self.cycles.append({"permutation_ids": [row["source_id"] for row in self.rows], "drawn_ids": []})
        row = self.rows[self.position]
        self.position += 1
        self.draws += 1
        self.seen.add(row["source_id"])
        self.cycles[-1]["drawn_ids"].append(row["source_id"])
        return row

    def summary(self):
        pool_ids = {row["source_id"] for row in self.rows}
        cycles = []
        for index, cycle in enumerate(self.cycles):
            permutation = cycle["permutation_ids"]
            drawn = cycle["drawn_ids"]
            check(f"cycle permutation{index} contains exactly the pool", len(permutation) == len(pool_ids) and set(permutation) == pool_ids)
            check(f"cycle{index} without replacement and prefix order", len(drawn) == len(set(drawn)) and drawn == permutation[:len(drawn)])
            cycles.append({"cycle_number": index + 1, "pool_size": len(pool_ids), "draws": len(drawn),
                           "complete": len(drawn) == len(pool_ids), "unique_source_ids": len(set(drawn)),
                           "permutation_source_ids_sha256": ids_sha(permutation), "drawn_source_ids_sha256": ids_sha(drawn)})
        return {"draws": self.draws, "unique_source_ids": len(self.seen), "pool_size": len(self.rows),
                "cycles": cycles}


sampler_rng = random.Random(config["seed"] + 1)
tr_stream = IndependentCycler(turkish, sampler_rng)
replay_streams = {language: IndependentCycler(by_language[language], sampler_rng) for language in replay_languages}
samples, step_summaries, replay_counts = [], [], Counter()
for record in history:
    step_rows = []
    for micro in range(config["accumulation"]):
        if micro % 2 == 0:
            row = tr_stream.draw()
            stream = "tr"
        else:
            language = sampler_rng.choice(list(replay_streams))
            row = replay_streams[language].draw()
            replay_counts[language] += 1
            stream = "replay"
        sample = {"step": record["step"], "micro": micro, "stream": stream,
                  "language": row["language"], "source_id": row["source_id"], "audio_seconds": float(row["duration"])}
        samples.append(sample)
        step_rows.append(sample)
    check(f"recorded Turkish unique count at step{record['step']}", len(tr_stream.seen) == record["turkish_unique_clips_seen"])
    step_summaries.append({"step": record["step"], "turkish_audio_seconds": math.fsum(row["audio_seconds"] for row in step_rows if row["stream"] == "tr"),
                           "replay_audio_seconds": math.fsum(row["audio_seconds"] for row in step_rows if row["stream"] == "replay"),
                           "replay_languages": [row["language"] for row in step_rows if row["stream"] == "replay"]})
check("4800 sampled clips and exact2TR/2replay eachupdate", len(samples) == 4800
      and all([sample["stream"] for sample in samples[start:start+4]] == ["tr", "replay", "tr", "replay"] for start in range(0,4800,4)))
check("2400 per stream", tr_stream.draws == sum(replay_counts.values()) == 2400)

validation_steps = [record["step"] for record in history if "validation" in record]
expected_validation = [step for step in range(1, config["steps"]+1) if step % config["eval_every"] == 0 or step == config["first_eval"] or step == config["steps"]]
check("recorded evaluation boundary steps match actual configuredschedule", validation_steps == expected_validation)
mixed_interval_steps = {step + 1 for step in validation_steps if step < len(history)}
all_interval_times = []
previous = 0.0
for record in history:
    elapsed = float(record["elapsed_seconds"])
    interval = elapsed - previous
    check("monotonic finite time step" + str(record["step"]), math.isfinite(elapsed) and interval > 0)
    all_interval_times.append({"step": record["step"], "seconds": interval, "contains_previous_development_evaluation": record["step"] in mixed_interval_steps})
    previous = elapsed
steady_steps = {record["step"] for record in history if record["step"] not in mixed_interval_steps}
steady_seconds = math.fsum(item["seconds"] for item in all_interval_times if item["step"] in steady_steps)


def totals(sample_rows):
    tr_seconds = math.fsum(row["audio_seconds"] for row in sample_rows if row["stream"] == "tr")
    replay_seconds = math.fsum(row["audio_seconds"] for row in sample_rows if row["stream"] == "replay")
    return {"sampled_clips": len(sample_rows), "turkish_clips": sum(row["stream"] == "tr" for row in sample_rows),
            "replay_clips": sum(row["stream"] == "replay" for row in sample_rows),
            "turkish_audio_seconds": tr_seconds, "replay_audio_seconds": replay_seconds,
            "total_audio_seconds": tr_seconds + replay_seconds,
            "turkish_audio_minutes": tr_seconds / 60, "replay_audio_minutes": replay_seconds / 60,
            "total_audio_minutes": (tr_seconds + replay_seconds) / 60,
            "total_audio_hours": (tr_seconds + replay_seconds) / 3600}


full_totals = totals(samples)
steady_totals = totals([row for row in samples if row["step"] in steady_steps])
excluded_totals = totals([row for row in samples if row["step"] in mixed_interval_steps])


def rate(total, wall_seconds):
    return {"wall_seconds": wall_seconds, "wall_minutes": wall_seconds / 60,
            "total_audio_minutes_per_wall_minute": total["total_audio_seconds"] / wall_seconds,
            "turkish_audio_minutes_per_wall_minute": total["turkish_audio_seconds"] / wall_seconds,
            "replay_audio_minutes_per_wall_minute": total["replay_audio_seconds"] / wall_seconds,
            "clips_per_wall_second": total["sampled_clips"] / wall_seconds}


replay_total = sum(replay_counts.values())
expected_per_language = replay_total / len(replay_languages)
chi_square = sum((replay_counts[language]-expected_per_language)**2/expected_per_language for language in replay_languages)
per_language = {}
for language in replay_languages:
    language_samples = [row for row in samples if row["stream"] == "replay" and row["language"] == language]
    seconds = math.fsum(row["audio_seconds"] for row in language_samples)
    per_language[language] = {"draws": len(language_samples), "draw_fraction": len(language_samples)/replay_total,
                              "expected_draws_equal_probability": expected_per_language,
                              "audio_seconds": seconds, "audio_minutes": seconds/60,
                              "cycling": replay_streams[language].summary()}
tr_cycles = tr_stream.summary()
all_unique = {row["source_id"] for row in samples}
source_lookup = {row["source_id"]: row for row in turkish + replay}
unique_samples = [{"stream": "tr" if source_lookup[source_id]["language"] == "tr" else "replay",
                   "audio_seconds": float(source_lookup[source_id]["duration"])} for source_id in all_unique]
unique_totals = totals(unique_samples)

report = {
    "schema": 1, "passed": True,
    "created_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(), "timezone": "America/Los_Angeles",
    "run_id": RUN_ID, "device": "CPU Python reconstruction only", "gpu_used": False,
    "python_version": sys.version, "configured_and_completed_optimizer_steps": len(history),
    "effective_batch": config["accumulation"], "counts_include_repeated_training_exposure": True,
    "method": {
        "candidate_order": "Initial CommonVoice manifest shuffled by a fresh Random(seed), limited to train_limit, then FLEURS TRAIN supplement appended; exact accepted IDs preserve order. Replay preserves manifest order; dictionary language order en/es/it/fr/de.",
        "sampler": "One shared Random(seed+1), lazy in-place shuffle at each stream cycle boundary; micro0/2 Turkish, micro1/3 uniformly rng.choice(replay language order), then chosen language stream draw/shuffle. Separate independent simulation, no import or invocation of trainer/run/GPU.",
        "seed": config["seed"], "sampler_seed": config["seed"]+1,
        "durations": "Exact unchanged prepared manifest durations, cross-checked against accepted audit for all3719 usable TRAIN rows; math.fsum for sums. Repeated clips counted each time.",
        "validation": "Matched the saved Turkish unique-source count at every one of1200 steps; checked all accepted pool IDs/order/durations, every complete/partial shuffled cycle without replacement, history boundaries and actual training source hash.",
        "uniform_language_interpretation": "Equal probability1/5 at each replay draw; realized counts need not be exactly equal. Audio-duration fractions need not match clip fractions.",
    },
    "ordered_sample_source_ids_sha256": ids_sha([row["source_id"] for row in samples]),
    "ordered_sample_source_ids_hash_definition": "UTF8 SHA256 of newline-joined source IDs in step-major/micro-major order, no final newline",
    "first8_samples": samples[:8], "last8_samples": samples[-8:],
    "full_1200_step_exposure": full_totals,
    "unique_source_exposure": {**unique_totals, "note": "Unique source clips/audio, not repeated seen audio or model training throughput."},
    "turkish_cycling": tr_cycles,
    "replay_language_draws": per_language,
    "replay_equal_probability_descriptive_check": {"expected_draws_per_language": expected_per_language,
        "pearson_chi_square": chi_square, "degrees_of_freedom": 4,
        "approximate_chi_square_tail_probability": math.exp(-chi_square/2)*(1+chi_square/2),
        "scope": "Descriptive uniform-draw diagnostic, not an exact balanced-language guarantee or a quality/retention test."},
    "history_window_rate": {"exposure": full_totals, **rate(full_totals, float(history[-1]["elapsed_seconds"])),
        "scope": "Actual loop-start through1200th optimizer-update record; includes prior dev evaluation/checkpoint pauses, excludes startup/model/cache preparation, final1200dev, finalization/heldout inference."},
    "training_intervals_rate": {"included_optimizer_updates": len(steady_steps), "included_clips": steady_totals["sampled_clips"],
        "exposure": steady_totals, **rate(steady_totals, steady_seconds),
        "scope": "Only known training-bearing intervals without prior dev evaluation; excludes the6 mixed evaluation+next-update intervals AND corresponding6updates/24clips. Includes data/cache reads, STFT, host/device transfer, forward/backward/Adam and ordinary loop/log overhead.",
        "not_exact_full_run_compute_only_rate": True},
    "excluded_mixed_evaluation_intervals": {"steps": sorted(mixed_interval_steps), "validation_steps": validation_steps,
        "seconds": math.fsum(item["seconds"] for item in all_interval_times if item["step"] in mixed_interval_steps),
        "exposure_excluded_for_aligned_numerator": excluded_totals,
        "intervals": [item for item in all_interval_times if item["step"] in mixed_interval_steps],
        "reason": "History timestamp is recorded before dev evaluation. Each next-step difference includes evaluation/checkpoint work plus that next training step. Those durations cannot be separated exactly from history alone."},
    "per_step_audio_seconds": step_summaries,
    "checks_count": len(checks), "checks": checks,
    "input_sha256": {str(path): sha(path) for path in (CONFIG_PATH, HISTORY_PATH, ACCEPTANCE_PATH, *source_paths, OUTPUT / "train_lora.py", OUTPUT / "benchmark.py")},
    "caveats": ["Counts are a deterministic reconstruction, since individual sampled IDs were not persisted by the active trainer. Matching unique-TR history at all1200 steps and pinned input/source hashes supports the reconstruction; it is not a GPU-instrumented per-sample trace.",
                 "Rates count input audio duration each exposure, not independent corpus hours, extra encoder checkpoint recomputation, teacher-cache computation, token count or words decoded.",
                 "History-window rate excludes final dev/finalization; it is not total elapsed task time.",
                 "The training-interval estimate correctly drops both time and audio for6 ambiguous post-evaluation updates rather than attributing all4800 clips to a shortened denominator.",
                 "Wall timing comes from the saved perf_counter history; no new synchronized CUDA throughput experiment was run.",
                 "No heldout prediction files or GPU were used, and no training/helper code was changed."],
}
root_result_path = OUTPUT / "training_audio_throughput.json"
root_result = json.loads(root_result_path.read_text(encoding="utf-8"))
step_major_sources = [{"step": step, "source_ids": [sample["source_id"] for sample in samples[(step-1)*4:step*4]]}
                      for step in range(1, len(history)+1)]
root_format_fingerprint = hashlib.sha256(json.dumps(step_major_sources, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
check("independent sampled sequence SHA exactly matches root JSON format", root_format_fingerprint == root_result["sampled_source_sequence_sha256"])
check("root saved input SHA values match independent source files", all(sha(path) == expected for path, expected in root_result["input_sha256"].items()))
check("root presentation and unique-source counts", root_result["counts"] == {"clip_presentations": 4800,
      "turkish_unique_clips": len(tr_stream.seen), "replay_unique_clips": sum(len(stream.seen) for stream in replay_streams.values())})
differences = []
def root_equal(name, actual, expected):
    delta = abs(float(actual)-float(expected))
    differences.append({"name": name, "absolute_difference": delta})
    check(name, math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9))
for language in language_order:
    language_rows = [sample for sample in samples if sample["language"] == language]
    check("root clip draw count " + language, len(language_rows) == root_result["sampled_audio_by_language"][language]["clips"])
    root_equal("root audio duration " + language, math.fsum(sample["audio_seconds"] for sample in language_rows), root_result["sampled_audio_by_language"][language]["seconds"])
root_equal("root full history-window total audio", full_totals["total_audio_minutes"], root_result["logged_span"]["audio_minutes"])
root_equal("root full history-window wall time", float(history[-1]["elapsed_seconds"])/60, root_result["logged_span"]["wall_minutes"])
root_equal("root full history-window rate", report["history_window_rate"]["total_audio_minutes_per_wall_minute"], root_result["logged_span"]["audio_min_per_wall_min"])
root_steady = root_result["training_intervals_without_dev_pauses"]
check("root same6 omitted boundaries and1194 retained updates", root_steady["excluded_update_steps"] == sorted(mixed_interval_steps)
      and root_steady["updates"] == len(steady_steps))
root_equal("root retained aligned audio numerator", steady_totals["total_audio_minutes"], root_steady["audio_minutes"])
root_equal("root retained aligned wall denominator", steady_seconds/60, root_steady["wall_minutes"])
root_equal("root training-interval rate", report["training_intervals_rate"]["total_audio_minutes_per_wall_minute"], root_steady["audio_min_per_wall_min"])
root_equal("root TR training-interval rate", report["training_intervals_rate"]["turkish_audio_minutes_per_wall_minute"], root_steady["turkish_audio_min_per_wall_min"])
check("root all1200 step-interval count", len(root_result["step_intervals"]) == len(history))
maximum_step_audio_difference = maximum_step_wall_difference = 0.0
for actual, independent_time, independent_audio in zip(root_result["step_intervals"], all_interval_times, step_summaries):
    check("root interval source step/order " + str(actual["step"]), actual["step"] == independent_time["step"] == independent_audio["step"])
    audio = independent_audio["turkish_audio_seconds"] + independent_audio["replay_audio_seconds"]
    audio_delta = abs(actual["audio_seconds"]-audio)
    tr_delta = abs(actual["turkish_audio_seconds"]-independent_audio["turkish_audio_seconds"])
    wall_delta = abs(actual["wall_seconds"]-independent_time["seconds"])
    maximum_step_audio_difference = max(maximum_step_audio_difference, audio_delta, tr_delta)
    maximum_step_wall_difference = max(maximum_step_wall_difference, wall_delta)
    check("root aligned step numerator/denominator/pause " + str(actual["step"]), audio_delta < 1e-9 and tr_delta < 1e-9 and wall_delta < 1e-9
          and actual["includes_previous_dev_and_checkpoint_pause"] == independent_time["contains_previous_development_evaluation"])
last100_steps = {step for step in range(1101,1201)}
last100_exposure = totals([sample for sample in samples if sample["step"] in last100_steps])
last100_seconds = math.fsum(item["seconds"] for item in all_interval_times if item["step"] in last100_steps)
last100_rate = rate(last100_exposure, last100_seconds)
root_equal("root last100 training-interval rate", last100_rate["total_audio_minutes_per_wall_minute"], root_result["last_100_training_intervals"]["audio_min_per_wall_min"])
report["last100_training_intervals"] = {"first_step": 1101, "last_step": 1200, "exposure": last100_exposure, **last100_rate}
report["root_result_comparison"] = {"passed": True, "path": str(root_result_path), "sha256": sha(root_result_path),
    "root_format_sampled_source_sequence_sha256": root_format_fingerprint,
    "hash_format": "SHA256 UTF8 JSON[{step,source_ids}], ensure_ascii=False,separators=(',',':')",
    "scalar_differences": differences, "maximum_scalar_absolute_difference": max(row["absolute_difference"] for row in differences),
    "all1200_aligned_step_intervals_checked": True, "maximum_step_audio_absolute_difference": maximum_step_audio_difference,
    "maximum_step_wall_absolute_difference": maximum_step_wall_difference}
report["checks_count"] = len(checks)
destination = OUTPUT / "audio_throughput_independent_audit.json"
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"path": str(destination), "passed": report["passed"], "checks": len(checks),
                  "exposure": full_totals, "history_window_rate": report["history_window_rate"]["total_audio_minutes_per_wall_minute"],
                  "training_intervals_rate": report["training_intervals_rate"]["total_audio_minutes_per_wall_minute"],
                  "training_intervals_wall_minutes": report["training_intervals_rate"]["wall_minutes"],
                  "replay_draw_counts": dict(replay_counts), "ordered_sample_source_ids_sha256": report["ordered_sample_source_ids_sha256"]}, indent=2))
