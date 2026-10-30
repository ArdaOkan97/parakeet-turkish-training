"""Read-only, stdlib calculation of native04/05 historical training rates."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
checks = 0
inputs = {}


def require(value, name):
    global checks
    if not value:
        raise ValueError(name)
    checks += 1


def read(path, lines=False):
    raw = path.read_bytes()
    inputs[str(path)] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return [json.loads(x) for x in raw.decode("utf-8").splitlines()] if lines else json.loads(raw)


def window(rows):
    audio = sum(r["step_audio_seen_seconds"] for r in rows)
    wall = sum(r["step_train_wall_seconds"] for r in rows)
    return {"first_step": rows[0]["step"], "last_step": rows[-1]["step"], "updates": len(rows),
            "audio_seconds": audio, "train_wall_seconds": wall,
            "combined_audio_minutes_per_wall_minute": audio / wall,
            "first_loading_exclusions": sum(r["first_batch_loading_exclusions_this_update"] for r in rows)}


manifest = read(ROOT / "work/nemo_direct_trial_v1/manifests/train_balanced_250.jsonl", True)
summary = {}
for number in ("04", "05"):
    run = HERE / "runs" / f"full_nemo_turkish_20261011_pilot_{number}"
    rows = read(run / "training_telemetry.jsonl", True)
    audio = wall = 0.0
    largest_metadata_delta = 0.0
    for index, row in enumerate(rows):
        require(row["step"] == index + 1, "contiguous completed updates")
        require(row["step_clips_seen"] == 16 and row["step_microbatches"] == 4, "B4x4 counts")
        audio += row["step_audio_seen_seconds"]
        wall += row["step_train_wall_seconds"]
        require(abs(audio - row["audio_seen_seconds"]) < 1e-7, "cumulative audio")
        require(abs(wall - row["train_wall_seconds"]) < 1e-7, "cumulative wall")
        require(math.isclose(audio / wall, row["audio_minutes_per_wall_minute"], rel_tol=1e-12), "saved rate")
        delta = abs(sum(x["duration"] for x in manifest[index * 16:(index + 1) * 16]) - row["step_audio_seen_seconds"])
        largest_metadata_delta = max(largest_metadata_delta, delta)
        require(delta <= 16 / 16000, "metadata duration within one sample per clip")
    outcome = read(run / "trial_outcome.json")
    require(outcome["status"] == "failed" and outcome["actual_completed_optimizer_updates"] == len(rows), "OOM completion")
    summary[number] = {"first20": window(rows[:20]), "all_completed": window(rows),
                       "last100": window(rows[-100:]) if len(rows) >= 100 else None,
                       "max_manifest_vs_actual_lengths_delta_seconds": largest_metadata_delta,
                       "status": outcome["status"], "error": outcome["error"],
                       "selected_step": outcome["selected_step"]}
custom = read(HERE / "native_training_custom_pilot_v1.json")
native = read(HERE / "native_training_nemo_pilot_v1.json")
require(custom["identity"] == native["identity"], "matched20 identity")
require(custom["initial_trainables_sha256"] == native["initial_trainables_sha256"], "matched20 initial weights")
require([s for u in custom["updates"] for s in u["source_ids"]] == [r["source_id"] for r in manifest[:320]], "same320 ordered TRAIN IDs")
require(custom["timing"]["total_audio_seconds"] == native["timing"]["total_audio_seconds"], "matched20 audio")
history = read(HERE / "runs/adaptive_20261011_native_split_resume_01/training_history.json")
prior, last = history[357], history[-1]
require(prior["step"] == 358 and last["step"] == 862, "historical prefix and suffix")
for name in ("train_full_nemo_turkish_v5.py", "full_nemo_turkish_v5.json", "native_lightning_telemetry_v2.py"):
    path = HERE / name
    inputs[str(path)] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
for relative in ("nemo/collections/asr/losses/rnnt.py", "nemo/collections/asr/models/rnnt_models.py",
                 "nemo/collections/asr/modules/rnnt.py", "lightning/pytorch/loops/optimization/automatic.py",
                 "lightning/pytorch/loops/training_epoch_loop.py"):
    path = ROOT / "work/nemo-native-env-v1/Lib/site-packages" / relative
    inputs[str(path)] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
report = {
    "schema": 1, "kind": "full_nemo_training_throughput_source_review_v1", "passed": True, "checks": checks,
    "created_at_utc": datetime.now(timezone.utc).isoformat(), "native04_native05": summary,
    "matched20": {"custom_HF": custom["timing"], "native_kernel_HF": native["timing"],
                  "native_over_custom_ratio": native["timing"]["combined_audio_minutes_per_wall_minute"] / custom["timing"]["combined_audio_minutes_per_wall_minute"],
                  "same_first320_native04_native05_TRAIN_IDs": True, "full_NeMo_trainer_comparison": False, "significant_speed_gain_claimed": False},
    "historical_HF_native_kernel862": {"cumulative_rate": last["audio_minutes_per_wall_minute"], "inherited_step": 358, "new_suffix_updates": 504,
        "suffix_rate": (last["audio_seen_seconds"] - prior["audio_seen_seconds"]) / (last["train_wall_seconds"] - prior["train_wall_seconds"]),
        "includes_prior_source_and_allocator_policy_prefix": True},
    "source_findings": {
        "official_training_step": "EncDecRNNTModel.training_step fused joint, add_auxiliary_losses, log_dict(train_loss), return loss; no override",
        "mean_volume": "RNNTLoss.reduce sums raw NLL and divides by sum target lengths per physical microbatch (rnnt.py421-434)",
        "fused_batch1": "RNNTJoint collects all per-clip losses then reduces across physical batch; active loss graphs remain until backward",
        "accumulation": "Lightning ClosureResult divides each returned microbatch loss by accumulation. B4x4 averages four separately normalized losses; this differs from one16-clip mean_volume or B2x8",
        "logged_loss": "Native CSV train_loss is microbatch mean_volume before Lightning accumulation scaling; HF logged per-target mean plus KD total is not directly comparable",
        "native_training_WER": "Computed on all microbatches every50th optimizer update and included in native train intervals",
        "periodic_checkpoint": "Stock on_train_batch_end callback runs before batch_progress.increment_completed; automatic periodic full-state resume has not been validated"},
    "timing_scopes": {
        "full_native": "Steady previous synchronized microbatch-end to next synchronized end includes loading/H2D/forward/backward/Adam, cleanup, preceding logging. First already-loaded batch after startup/DEV is excluded and counted. Cold first train/JIT included; model load/startup/DEV excluded.",
        "matched20": "Cold/JIT and2warmups excluded; read/cache/frontend/H2D/objective/backward/Adam/sync/cache release timed; telemetry/publication outside timer",
        "units": "combined TR+replay unpadded audio seconds/train interval seconds = audio min/min, not TR-only progress or process end-to-end"},
    "noncontrolled_differences": [
        "Native base + zero SOS versus learned HF1871; old IDs/blank placement and new row states differ",
        "Native sigma.02/omega.1/mean_volume versus HF sigma0/omega0 per-target mean + replay token/duration/feature KD",
        "Physical B4x4 versus B8x2, stock native loss memory versus checkpointed HF/native adapter",
        "Native specaugmentation, decoder/joint dropout, cosine scheduler, native log/WER versus HF teacher cache/distillation and constant schedule",
        "Cold/JIT exclusions and framework/logging timer boundaries differ; historical862 also uses different chronology/inherited state"],
    "controlled_full_trainer_gain_claimed": False,
    "quality_limit": "Native04/05 OOM before trained DEV or a checkpoint; their speed does not establish useful Turkish accuracy or full-sequence stability",
    "inputs": inputs, "Torch_NeMo_CUDA_imported_or_initialized": False, "active_sources_mutated": False}
target = HERE / "full_nemo_training_throughput_source_review_v1.json"
with target.open("x", encoding="utf-8") as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"passed": True, "checks": checks, "report": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))
