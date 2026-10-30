"""Native B1 Parakeet/NeMo/Lightning warmstart arm; never dispatches the HF loop.

--validate-only is standard-library-only. Actual training must be launched by
the owning GPU controller with the native-env-v2 interpreter. Native base
initialization and stock native TDT objective intentionally differ from the
previous trained HF/LoRA checkpoint and its preservation losses.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LANGUAGES = ("en", "es", "it", "fr", "de", "tr")
PROJECTIONS = ("linear_q", "linear_k", "linear_v", "linear_out")
PRELOAD_SHA = "f8787b0f3346c9f526903ff0c06de4c266f940ca3b0f41c55907cc795e7cc2ba"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scoped(value):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    require(path.is_relative_to(ROOT), "Trial paths must remain inside the workspace")
    return path


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_new(path, value):
    path = scoped(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")



def validate_full_stream(train, pilot_proof, cfg):
    """Stream only metadata; never read PCM or retain the149,776 rows in RAM."""
    full_proof_path = scoped(cfg["continuation_data_proof"])
    require(sha(full_proof_path) == cfg["continuation_data_proof_sha256"], "Full-pass proof changed after configuration was frozen")
    full_proof = json.loads(full_proof_path.read_text(encoding="utf-8"))
    require(full_proof.get("passed") is True and full_proof.get("kind") == "native_sequential_full_TR_pass_after_pilot_v1",
            "Verified full-pass preparation is required")
    require(full_proof["pilot_proof_sha256"] == sha(scoped(cfg["data_proof"])) and full_proof["full_updates"] == 9361
            and full_proof["pilot_completed_updates"] == 250 and full_proof["full_manifest_pilot_prefix_rows"] == 4000,
            "Full-pass/pilot proof identity changed")
    require(str(train) in full_proof["manifest_sha256"] and sha(train) == full_proof["manifest_sha256"][str(train)],
            "Full native TRAIN stream changed")
    pilot = scoped(cfg["pilot_training_manifest"])
    require(str(pilot) in pilot_proof["manifest_sha256"] and sha(pilot) == pilot_proof["manifest_sha256"][str(pilot)],
            "Native250 prefix changed")
    count = 0
    tr_ids = set()
    tr_visits = 0
    with train.open(encoding="utf-8") as full, pilot.open(encoding="utf-8") as prefix:
        for line in full:
            if not line.strip():
                continue
            row = json.loads(line)
            require(row.get("training_role") == "train" and row.get("dataset_split") in ("train", "validated")
                    and row["language"] in LANGUAGES and .5 <= row["duration"] <= 30, "Unvetted full native TRAIN row")
            require((row["language"] == "tr") == (count % 16 < 8), "Preserve8TR then8replay ordered effective16")
            if count < 4000:
                require(row == json.loads(next(prefix)), "Full manifest pilot prefix differs")
            if row["language"] == "tr":
                tr_visits += 1
                tr_ids.add(row["source_id"])
            count += 1
        require(not prefix.read().strip(), "Pilot prefix has extra recordings")
    require(count == 9361 * 16 and len(tr_ids) == 74883 and tr_visits - len(tr_ids) == 5,
            "Full TR pass must preserve all74883 unique IDs and exactly5 declared final padding visits")
    return count, {"training_records": count, "unique_turkish_source_ids": len(tr_ids),
                   "turkish_recording_visits": tr_visits, "final_padding_visits": tr_visits - len(tr_ids),
                   "pilot_prefix_rows_compared": 4000, "full_manifest_sha256": sha(train),
                   "continuation_data_proof_sha256": sha(full_proof_path), "PCM_read_or_rehashed": False}



def validate_source_warmstart(checkpoint_value, cfg):
    checkpoint = scoped(checkpoint_value)
    source_dir = scoped(cfg["warmstart"]["source_run_dir"])
    require(checkpoint == scoped(cfg["warmstart"]["checkpoint_path"]) and checkpoint.is_file(), "Pinned native source100 checkpoint missing")
    require(checkpoint.stat().st_size == cfg["warmstart"]["checkpoint_bytes"]
            and sha(checkpoint) == cfg["warmstart"]["checkpoint_sha256"], "Native source100 checkpoint bytes changed")
    require(checkpoint.is_relative_to(source_dir) and source_dir.is_relative_to(HERE / "runs"), "Native source checkpoint scope changed")
    paths = {name: source_dir / name for name in ("resolved_trial_config.json", "preflight.json", "trial_outcome.json")}
    require(all(path.is_file() for path in paths.values()), "Source native run evidence missing")
    source = json.loads(paths["resolved_trial_config.json"].read_text(encoding="utf-8"))
    prepared = json.loads(paths["preflight.json"].read_text(encoding="utf-8"))
    outcome = json.loads(paths["trial_outcome.json"].read_text(encoding="utf-8"))
    require(source["kind"] == "full_nemo_lightning_turkish_pilot_v7" and prepared["entry_sha256"] == cfg["source_pilot_entry_sha256"]
            and prepared["config_sha256"] == cfg["source_pilot_config_sha256"], "Source native recipe/source identity changed")
    require(outcome["run_id"] == prepared["run_id"] == source_dir.name and outcome["actual_completed_optimizer_updates"] >= 100,
            "Source run did not actually reach100 completed updates")
    require(prepared["data_proof_sha256"] == sha(scoped(cfg["data_proof"]))
            and prepared["tokenizer_manifest_sha256"] == sha(scoped(cfg["tokenizer_manifest"])), "Source native data/tokenizer evidence changed")
    require(source["initialization"] == "original_native_v3_nemo_no_HF_trained_weight_transfer"
            and cfg["initialization"] == "native06_periodic100_strict_model_Adam_Cosine_warmstart_no_HF_transfer",
            "Native warmstart initialization must remain explicit")
    for field in ("seed", "init_from_nemo_model", "lora", "allocator_cap_gb", "joint_fused_batch_size",
                  "joint_preserve_memory", "source_pins", "baseline_development_summary", "tokenizer_manifest",
                  "native_extracted_dir", "native_extraction_proof"):
        require(source[field] == cfg[field], "Warmstart changed an undeclared recipe field:" + field)
    source_objective = dict(source["objective"])
    target_objective = dict(cfg["objective"])
    source_objective.pop("physical_batch_normalization_change")
    target_objective.pop("physical_batch_normalization_change")
    require(source_objective == target_objective, "Only declared physical mean_volume grouping may change")
    source_model = json.loads(json.dumps(source["model"]))
    target_model = json.loads(json.dumps(cfg["model"]))
    require(scoped(source_model["train_ds"].pop("manifest_filepath")) == scoped(cfg["pilot_training_manifest"]), "Source pilot stream changed")
    target_model["train_ds"].pop("manifest_filepath")
    require(source_model["train_ds"].pop("batch_size") == 2 and target_model["train_ds"].pop("batch_size") == 1,
            "Only B2-toB1 physical grouping is declared")
    require(source_model == target_model, "Native optimizer/data/model fields changed")
    source_trainer, target_trainer = dict(source["trainer"]), dict(cfg["trainer"])
    require(source_trainer.pop("accumulate_grad_batches") == 8 and target_trainer.pop("accumulate_grad_batches") == 16,
            "Keep effective16 recordings per optimizer update")
    require(source_trainer.pop("val_check_interval") == 2000 and target_trainer.pop("val_check_interval") == 4000,
            "Warmstart initial DEV feedback is local250 updates")
    require(source_trainer == target_trainer, "Undeclared official Trainer setting changed")
    return {"source_step": 100, "source_run_id": source_dir.name, "source_run_dir": str(source_dir),
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
            "source_evidence_sha256": {str(path): sha(path) for path in paths.values()},
            "source_batch_counter_lag_restored": False, "source_fit_loops_restored": False,
            "weights_Adam_Cosine_preserved": True, "old_RNG_restored_or_bitwise_gradients_claimed": False,
            "consumed_ordered_training_records": 1600, "first_unseen_source_index": 1600,
            "physical_grouping_transition": {"source_batch": 2, "source_accumulation": 8,
                                               "target_batch": 1, "target_accumulation": 16,
                                               "effective_recordings_per_update": 16}}


def verify_warmstart_checkpoint_metadata(checkpoint):
    """Validate weights/Adam/Cosine source; do not reuse its Lightning loops."""
    require(checkpoint["global_step"] == 100 and len(checkpoint["state_dict"]) == 917, "Expected actual native100/917-state checkpoint")
    require(len(checkpoint.get("optimizer_states", [])) == len(checkpoint.get("lr_schedulers", [])) == 1,
            "One complete native AdamW/Cosine state is required")
    optimizer, scheduler = checkpoint["optimizer_states"][0], checkpoint["lr_schedulers"][0]
    require([len(group["params"]) for group in optimizer["param_groups"]] == [828, 9, 6], "Native optimizer parameter ordering/group sizes changed")
    require(scheduler["max_steps"] == 9361 and scheduler["warmup_steps"] == 25 and scheduler["last_epoch"] == 100
            and scheduler["_step_count"] == 101, "Actual source Cosine/warmup position changed")
    require([group["lr"] for group in optimizer["param_groups"]] == scheduler["_last_lr"], "Source Adam and Cosine LR state disagrees")
    progress = checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]
    require(all(progress[scope]["ready"] == progress[scope]["started"] == progress[scope]["processed"] == 800
                and progress[scope]["completed"] == 799 for scope in ("total", "current")),
            "Expected stock native100 batch-end snapshot with completed-counter lag")
    return {"source_step": 100, "state_tensor_count": 917, "optimizer_group_parameter_counts": [828, 9, 6],
            "source_optimizer_learning_rates": list(scheduler["_last_lr"]), "scheduler_last_epoch": 100,
            "scheduler_max_steps": 9361, "source_microbatch_ready": 800, "source_completed_counter": 799,
            "source_weights_and_optimizer_updates_completed": 100, "consumed_recordings": 1600,
            "old_fit_loop_batch_counters_used": False}


def restore_optimizer_scheduler(optimizer, scheduler, source_optimizer, source_scheduler):
    """Restore already-created official native objects; perform no step/warmup."""
    require([len(group["params"]) for group in optimizer.param_groups] == [828, 9, 6], "Target official AdamW parameter ordering changed")
    require(scheduler.max_steps == 9361 and scheduler.warmup_steps == 25, "Target Cosine horizon changed")
    optimizer.load_state_dict(source_optimizer)
    scheduler.load_state_dict(source_scheduler)
    require([group["lr"] for group in optimizer.param_groups] == source_scheduler["_last_lr"]
            and scheduler.last_epoch == 100, "Official optimizer/scheduler state was not restored")
    return {"restored_on": "official Lightning on_train_start", "source_step": 100,
            "optimizer_group_learning_rates": [group["lr"] for group in optimizer.param_groups],
            "scheduler_last_epoch": scheduler.last_epoch, "scheduler_max_steps": scheduler.max_steps,
            "optimizer_or_scheduler_step_called_during_restore": False}

def preflight(args):
    config_path = scoped(args.config)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    require(cfg["schema"] == 1 and cfg["kind"] == "full_nemo_lightning_turkish_warmstart_v9", "Wrong trial config")
    require(args.run_id and all(c.isalnum() or c in "_-" for c in args.run_id), "Unsafe trial ID")
    require(cfg["trainer"]["devices"] == 1 and cfg["trainer"]["accelerator"] == "gpu", "Single local GPU only")
    require(cfg["model"]["train_ds"]["batch_size"] == 1 and cfg["trainer"]["accumulate_grad_batches"] == 16,
            "This arm declares B1 x accumulation16, effective16; mean_volume grouping changes")
    require(cfg["joint_fused_batch_size"] in (1, 2), "Start with native joint subbatch1 or2")
    require(cfg["allocator_cap_gb"] == 9, "Declared bounded allocator cap required")
    require(cfg["initialization"] == "native06_periodic100_strict_model_Adam_Cosine_warmstart_no_HF_transfer", "Unreviewed initialization")
    require(cfg["objective"]["replay_KD"] is False, "This pilot declares supervised replay without HF teacher KD")
    base = scoped(cfg["init_from_nemo_model"])
    require(base.is_file() and base.stat().st_size == 2509332480, "Pinned full native .nemo file missing/incomplete")
    extracted = scoped(cfg["native_extracted_dir"])
    require(extracted.is_dir() and (extracted / "model_config.yaml").is_file() and (extracted / "model_weights.ckpt").is_file(),
            "Verified workspace native extraction must exist; no fallback to defaultTemp")
    extraction_path = scoped(cfg["native_extraction_proof"])
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    require(extraction.get("kind") == "native_workspace_extraction_v1" and extraction.get("passed") is True
            and scoped(extraction["source_path"]) == base and scoped(extraction["extracted_directory"]) == extracted,
            "Verified extraction/source paths disagree")
    extraction_stats = {}
    proof_time_ns = int(datetime.fromisoformat(extraction["created_at_utc"]).timestamp() * 1e9)
    require(len(extraction["files"]) == 5 and extraction.get("archive_file_bytes_preserved") is True,
            "All five native archive members must be preserved")
    for path, expected in extraction["files"].items():
        member = scoped(path)
        require(member.parent == extracted and member.is_file(), "Extracted member scope changed")
        stat = member.stat()
        require(stat.st_size == expected["bytes"] and stat.st_mtime_ns <= proof_time_ns + 1_000_000,
                "Extracted member size/modification time changed after verified extraction")
        if member.name != "model_weights.ckpt":
            require(sha(member) == expected["sha256"], "Extracted native metadata/tokenizer changed")
        extraction_stats[str(member)] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                        "sha256_verified_by_extraction": expected["sha256"]}
    require(sha(HERE / "preload_nemo_linker_v1.py") == PRELOAD_SHA, "Native linker preload changed")
    sources = {}
    for name in cfg["source_pins"]:
        path = scoped(name)
        require(path.is_file() and sha(path) == cfg["source_pins"][name], "Trial source changed: " + str(path))
        sources[str(path)] = sha(path)
    data_proof = scoped(cfg["data_proof"])
    require(data_proof.is_file(), "Prepared native TRAIN/DEV proof missing")
    proof = json.loads(data_proof.read_text(encoding="utf-8"))
    require(proof.get("passed") is True, "Native manifest preparation proof did not pass")
    train = scoped(cfg["model"]["train_ds"]["manifest_filepath"])
    train_count, stream_audit = validate_full_stream(train, proof, cfg)
    steps = args.max_steps if args.max_steps is not None else cfg["local_optimizer_update_limit"]
    require(isinstance(steps, int) and 1 <= steps <= 9261, "New local updates are bounded to the remaining original9361 horizon")
    require(train_count == 9361 * 16 and train_count >= steps * 16, "Full native plan must contain exactly9361 effective16 updates")
    dev = {}
    for language, value in zip(LANGUAGES, cfg["model"]["validation_ds"]["manifest_filepath"]):
        path = scoped(value)
        rows = read_rows(path)
        require(len(rows) == 50 and all(row["language"] == language and row["dataset_split"] == "validation" for row in rows),
                "Each native DEV loader must preserve the fixed50 recordings")
        dev[language] = rows
    require(set(proof["manifest_sha256"]) == {str(scoped(cfg["pilot_training_manifest"])), *(str(scoped(p)) for p in cfg["model"]["validation_ds"]["manifest_filepath"])},
            "Proof must bind exactly TRAIN and sixDEV manifests")
    for path, expected in proof["manifest_sha256"].items():
        require(sha(path) == expected, "Prepared native manifest changed")
    tokenizer_proof = scoped(cfg["tokenizer_manifest"])
    require(tokenizer_proof.is_file(), "Native tokenizer proof missing")
    baseline = scoped(cfg["baseline_development_summary"])
    require(baseline.is_file(), "Original six-language DEV baseline missing")
    run_dir = scoped("outputs/parakeet_turkish/runs/" + args.run_id)
    require(not run_dir.exists(), "A native trial must use a new run directory")
    require(cfg["trainer"]["max_steps"] == cfg["model"]["optim"]["sched"]["max_steps"] == 9361,
            "Official optimizer forces scheduler horizon fromTrainer; preserve the declared9361-update horizon")
    cfg["local_optimizer_update_limit"] = steps
    require(cfg["trainer"]["val_check_interval"] == 250 * cfg["trainer"]["accumulate_grad_batches"], "Initial feedback must be local250 optimizer updates")
    require(cfg["trainer"]["enable_checkpointing"] is True and cfg["periodic_checkpoint_policy"]["every_n_train_steps"] == 250, "Declared stock periodic checkpoint policy changed")
    source_checkpoint = args.source_checkpoint or scoped(cfg["warmstart"]["checkpoint_path"])
    inherited = validate_source_warmstart(source_checkpoint, cfg)
    return cfg, dev, run_dir, {
        "schema": 1, "kind": "full_nemo_trial_preflight_v1", "passed": True,
        "run_id": args.run_id, "entry_sha256": sha(__file__), "config_sha256": sha(config_path),
        "data_proof_sha256": sha(data_proof), "tokenizer_manifest_sha256": sha(tokenizer_proof),
        "source_sha256": sources, "native_base_path": str(base), "native_base_bytes": base.stat().st_size,
        "whole_native_base_sha256_pending_actual_run": True,
        "native_extracted_dir": str(extracted), "native_extracted_config_sha256": sha(extracted / "model_config.yaml"),
        "native_extraction_proof_path": str(extraction_path), "native_extraction_proof_sha256": sha(extraction_path),
        "native_archive_sha256_verified_by_extraction": extraction["source_sha256"],
        "extracted_members_fresh_stat": extraction_stats,
        "large_extracted_weights_rehashed_in_preflight": False,
        "optimizer_updates": steps, "Torch_imported": "torch" in sys.modules,
        "full_stream_audit": stream_audit, "source_native_warmstart": inherited,
        "native_import_and_training_pending": True, "unchanged_HF_recipe_claimed": False,
    }


def runtime(args, cfg, dev, run_dir, validation):
    # Ordering matters on native Windows: coherent12.9 linkage beforeTorch/Numba.
    from preload_nemo_linker_v1 import preload
    preload_result = preload()
    import torch
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from omegaconf import OmegaConf, open_dict
    from nemo.collections.asr.models import ASRModel, EncDecRNNTBPEModel
    from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
    from nemo.core.classes.modelPT import ModelPT
    from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
    from encoder_lora import LoRALinear
    from native_vocabulary_adaptation_v1 import apply_native_turkish_vocabulary
    from native_lightning_telemetry_v2 import make_native_training_telemetry_callback
    from metrics import score_transcriptions
    from resource_watchdog_v2 import ResourceGuard, POLICY, sample_memory, worker_process_commit_reason

    require(torch.cuda.is_available(), "Native trial requires the root-owned GPU window")
    total = torch.cuda.get_device_properties(0).total_memory
    require(total >= 15_000_000_000, "Expected local16GB-class CUDA device")
    torch.cuda.set_per_process_memory_fraction(cfg["allocator_cap_gb"] * 1e9 / total, 0)
    pl.seed_everything(cfg["seed"], workers=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    validation.update(native_base_sha256=validation["native_archive_sha256_verified_by_extraction"],
                      whole_native_base_sha256_pending_actual_run=False)
    write_new(run_dir / "preflight.json", validation)
    write_new(run_dir / "resolved_trial_config.json", cfg)
    baseline = json.loads(scoped(cfg["baseline_development_summary"]).read_text(encoding="utf-8"))
    baseline_wer = {lang: baseline[f"validation_{lang}"]["asr"]["wer"] for lang in LANGUAGES}

    class NativeTrialCallback(Callback):
        def __init__(self):
            self.rows = {lang: [] for lang in LANGUAGES}
            self.current_language = None
            self.selected_step = None
            self.best = float("inf")
            self.retention_failures = 0
            self.stop_reason = None
            self.telemetry = deque(maxlen=128)
            self.telemetry_omitted = 0
            self.done = threading.Event()
            self.thread = None
            self.guard = ResourceGuard()
            self.selected_path = None
            self.cache_reclamations = 0
            self.cache_cleanup_seconds = 0.0
            self.source_step = 100
            self.last_evaluated_step = None
            self.first_resumed_batch_checked = False
            self.source_optimizer_state = self.source_scheduler_state = None
            self.optimizer_restore_evidence = None

        def on_train_start(self, trainer, model):
            require(int(trainer.global_step) == 0 and len(trainer.optimizers) == len(trainer.lr_scheduler_configs) == 1, "Official Lightning loops must start fresh at local0")
            self.optimizer_restore_evidence = restore_optimizer_scheduler(trainer.optimizers[0], trainer.lr_scheduler_configs[0].scheduler,
                                                        self.source_optimizer_state, self.source_scheduler_state)
            self.source_optimizer_state = self.source_scheduler_state = None
            write_new(run_dir / "optimizer_warmstart_restoration.json", self.optimizer_restore_evidence)

        def start_monitor(self):
            def sample():
                while not self.done.is_set():
                    try:
                        observation = sample_memory(os.getpid(), phase="full_native_Lightning_trial")
                        reason, decision = self.guard.observe(observation["system_memory"], observation["worker_private_memory"],
                                                              observed_monotonic=observation["observed_monotonic"])
                        reason = reason or worker_process_commit_reason(observation.get("calling_process_commit_memory"), os.getpid())
                        if len(self.telemetry) == self.telemetry.maxlen:
                            self.telemetry_omitted += 1
                        self.telemetry.append({"observation": observation, "decision": decision, "reason": reason})
                        if reason:
                            self.stop_reason = reason
                    except Exception as error:
                        self.stop_reason = "resource_telemetry_failure:" + repr(error)
                    self.done.wait(5)
            self.thread = threading.Thread(target=sample, daemon=True)
            self.thread.start()

        def on_train_batch_start(self, trainer, model, batch, batch_idx):
            model.encoder.eval()  # Frozen nativeBN/dropout; gradients still reachLoRA.
            if not self.first_resumed_batch_checked:
                require(int(trainer.global_step) == 0, "First new update must follow fresh local loop0")
                require(len(batch[1]) == 1, "Declared target physical B1 grouping required")
                self.first_resumed_batch_checked = True
            if self.stop_reason or (args.stop_file and scoped(args.stop_file).exists()):
                self.stop_reason = self.stop_reason or "owned_stop_file"
                trainer.should_stop = True

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            # Official Lightning completed this microbatch/backward. Release
            # only unused allocator blocks; retain every live tensor/gradient.
            if torch.cuda.memory_reserved() - torch.cuda.memory_allocated() >= 750_000_000:
                began = time.perf_counter()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                self.cache_cleanup_seconds += time.perf_counter() - began
                self.cache_reclamations += 1
            if int(trainer.global_step) >= cfg["local_optimizer_update_limit"]:
                self.stop_reason = self.stop_reason or "declared_remaining_local_update_limit"
                trainer.should_stop = True
            if self.stop_reason:
                trainer.should_stop = True

        def on_validation_epoch_start(self, trainer, model):
            self.rows = {lang: [] for lang in LANGUAGES}

        def on_validation_batch_start(self, trainer, model, batch, batch_idx, dataloader_idx=0):
            self.current_language = LANGUAGES[dataloader_idx]

        def capture_decode(self, original_decode, model):
            def decode(*positional, **keywords):
                hypotheses = original_decode(*positional, **keywords)
                if not model.training and self.current_language is not None:
                    for hypothesis in hypotheses:
                        hypothesis = hypothesis[0] if isinstance(hypothesis, list) else hypothesis
                        self.rows[self.current_language].append(str(hypothesis.text))
                return hypotheses
            return decode

        def on_validation_epoch_end(self, trainer, model):
            if trainer.sanity_checking:
                return
            local_step = int(trainer.global_step)
            step = self.source_step + local_step
            self.last_evaluated_step = step
            if local_step >= 250:
                trainer.val_check_batch = 1000 * cfg["trainer"]["accumulate_grad_batches"]
            scored = {}
            for lang in LANGUAGES:
                require(len(self.rows[lang]) == len(dev[lang]), "Native DEV predictions/order incomplete for" + lang)
                rows = [{"language": lang, "reference": row["text"], "hypothesis": hypothesis,
                         "source_id": row["source_id"], "sentence_id": row.get("sentence_id")}
                        for row, hypothesis in zip(dev[lang], self.rows[lang])]
                scored[lang] = score_transcriptions(rows, language=lang)
                write_new(run_dir / f"dev_step_{step}_{lang}.json", {"step": step, "language": lang, "predictions": rows, "asr": scored[lang]})
            gate = {"step": step, "local_step": local_step, "source_optimizer_updates": self.source_step,
                    "turkish_wer": scored["tr"]["wer"],
                    "retention_gate_passed": all(scored[lang]["wer"] <= baseline_wer[lang] + max(.005, .1 * baseline_wer[lang])
                                                  for lang in LANGUAGES if lang != "tr"),
                    "per_language_wer": {lang: scored[lang]["wer"] for lang in LANGUAGES}}
            qualifying = gate["retention_gate_passed"] and gate["turkish_wer"] < baseline_wer["tr"]
            if qualifying and gate["turkish_wer"] < self.best:
                self.best, self.selected_step = gate["turkish_wer"], step
                if local_step == 0:
                    # Standalone initial validation precedes configure_optimizers.
                    # The original native100 checkpoint has the actual Adam/Cosine.
                    self.selected_path = Path(validation["source_native_warmstart"]["checkpoint_path"])
                else:
                    self.selected_path = run_dir / f"qualified_step_{step}.ckpt"
                    trainer.save_checkpoint(str(self.selected_path))  # Full official Lightning checkpoint after training.
            self.retention_failures = 0 if gate["retention_gate_passed"] else self.retention_failures + 1
            if self.retention_failures >= cfg["retention_failures_before_stop"]:
                self.stop_reason = "consecutive_development_retention_failures"
                trainer.should_stop = True
            gate.update(selected_step=self.selected_step, qualified=qualifying, stop_reason=self.stop_reason)
            write_new(run_dir / f"development_gate_step_{step}.json", gate)
            # DEV runs after Lightning completes all prior microbatches.
            # Preserve a full stock checkpoint even when this model is not selected.
            if local_step == 0:
                recovery_path = Path(validation["source_native_warmstart"]["checkpoint_path"])
                recovery_phase = "source_checkpoint_pre_optimizer_restore"
                recovery_sha = validation["source_native_warmstart"]["checkpoint_sha256"]
            else:
                recovery_path = self.selected_path if self.selected_step == step else run_dir / f"recovery_dev_step_{step}.ckpt"
                if self.selected_step != step:
                    trainer.save_checkpoint(str(recovery_path))
                recovery_phase = "completed_DEV_boundary"
                recovery_sha = sha(recovery_path)
            write_new(run_dir / f"recovery_checkpoint_step_{step}.json", {"step": step, "path": str(recovery_path),
                      "checkpoint_sha256": recovery_sha, "phase": recovery_phase, "local_step": local_step,
                      "fresh_full_checkpoint_saved_in_this_run": local_step > 0,
                      "official_Lightning_checkpoint": True, "development_selection_required_for_recovery": False})
            print(json.dumps({"native_dev": gate}), flush=True)

    callback = NativeTrialCallback()
    telemetry = make_native_training_telemetry_callback(pl, torch, run_dir, args.run_id)
    logger = CSVLogger(save_dir=str(run_dir), name="lightning", version="pilot")
    periodic = ModelCheckpoint(dirpath=str(run_dir / "periodic_checkpoints"),
                               filename="local_step_{step:06d}", auto_insert_metric_name=False,
                               every_n_train_steps=250, save_top_k=1, save_last=False,
                               save_on_train_epoch_end=False)
    trainer = pl.Trainer(**cfg["trainer"], callbacks=[callback, telemetry, periodic], logger=logger, default_root_dir=str(run_dir))
    callback.start_monitor()  # Cover native CPU restore and setup allocations, not onlyfit.
    error = None
    try:
        connector = SaveRestoreConnector()
        connector.model_extracted_dir = str(scoped(cfg["native_extracted_dir"]))
        model = ASRModel.restore_from(restore_path=str(scoped(cfg["init_from_nemo_model"])),
                                     save_restore_connector=connector, map_location="cpu")
        require(isinstance(model, EncDecRNNTBPEModel), "Expected native EncDecRNNTBPEModel")
        require(model.training_step.__func__ is EncDecRNNTModel.training_step, "Official NeMo training_step must remain intact")
        require(model.configure_optimizers.__func__ is ModelPT.configure_optimizers, "Official NeMo configure_optimizers must remain intact")
        model.set_trainer(trainer)
        vocabulary = apply_native_turkish_vocabulary(model, scoped(cfg["tokenizer_manifest"]), new_row_bias_penalty=2.0)
        require(model.decoder.blank_idx == 8198 and model.loss._blank == 8198, "Native lexical/blank mapping changed")
        require(model.joint.num_extra_outputs == 5 and model.loss._loss.durations == [0, 1, 2, 3, 4], "Native TDT durations changed")
        # Frozen native encoder weights useBF16; adapters and heads remainFP32.
        # Lightning bf16-mixed provides autocast without converting optimizer state.
        model.encoder.to(dtype=torch.bfloat16)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        targets = []
        for index, layer in enumerate(model.encoder.layers):
            for name in PROJECTIONS:
                base = getattr(layer.self_attn, name)
                wrapped = LoRALinear(base, rank=cfg["lora"]["rank"], alpha=cfg["lora"]["alpha"], dropout=0.0)
                setattr(layer.self_attn, name, wrapped)
                targets.append(f"encoder.layers.{index}.self_attn.{name}")
        for module in (model.decoder, model.joint):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        model.encoder.eval()
        decoding = OmegaConf.create(OmegaConf.to_container(model.cfg.decoding, resolve=True))
        with open_dict(decoding):
            decoding.strategy = "greedy_batch"
            decoding.greedy.use_cuda_graph_decoder = False
        model.change_decoding_strategy(decoding, verbose=False)
        model.joint.set_fuse_loss_wer(True, model.loss, model.wer)
        model.joint.set_fused_batch_size(cfg["joint_fused_batch_size"])
        model.joint.preserve_memory = cfg["joint_preserve_memory"]
        model.compute_eval_loss = False
        model.wer.log_prediction = False
        model.wer.decode = callback.capture_decode(model.wer.decode, model)
        train_cfg = OmegaConf.create(cfg["model"]["train_ds"])
        val_cfg = OmegaConf.create(cfg["model"]["validation_ds"])
        train_cfg.manifest_filepath = str(scoped(train_cfg.manifest_filepath))
        val_cfg.manifest_filepath = [str(scoped(p)) for p in val_cfg.manifest_filepath]
        model.setup_training_data(train_cfg)
        model.setup_multiple_validation_data(val_cfg)
        require(len(model._train_dl.dataset) == validation["full_stream_audit"]["training_records"],
                "Official NeMo dataset filtered unexpected TRAIN records")
        source_path = Path(validation["source_native_warmstart"]["checkpoint_path"])
        checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
        source_metadata = verify_warmstart_checkpoint_metadata(checkpoint)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        callback.source_optimizer_state = checkpoint["optimizer_states"][0]
        callback.source_scheduler_state = checkpoint["lr_schedulers"][0]
        del checkpoint  # Release source model tensors and every old Lightning loop.
        consumed = 1600
        dataset = model._train_dl.dataset

        class NativeWarmstartSuffixSampler(torch.utils.data.SequentialSampler):
            def __iter__(self):
                return iter(range(consumed, len(self.data_source)))

            def __len__(self):
                return len(self.data_source) - consumed

        original = model._train_dl
        model._train_dl = torch.utils.data.DataLoader(dataset, batch_size=original.batch_size,
            sampler=NativeWarmstartSuffixSampler(dataset), collate_fn=original.collate_fn,
            num_workers=0, pin_memory=False, drop_last=False)
        require(len(model._train_dl) == 148176 and len(model._train_dl) // 16 == 9261,
                "Suffix must visit the first unseen record1600 through full-pass end exactly once")
        resume = {"checkpoint_path": str(source_path), "checkpoint_sha256": validation["source_native_warmstart"]["checkpoint_sha256"],
                  "source_step": 100, "consumed_training_records": consumed, "remaining_training_records": len(dataset)-consumed,
                  "source_metadata": source_metadata, "loader_cursor": "new local loop; suffix sequential sampler starts1600",
                  "official_model_weights_loaded_strict": True, "Adam_Cosine_load_in_on_train_start": True,
                  "source_Lightning_fit_loops_loaded": False, "source_RNG_loaded_or_bitwise_continuation_claimed": False,
                  "cumulative_optimizer_step_formula": "100 + local Lightning global_step",
                  "full_manifest_sha256": validation["full_stream_audit"]["full_manifest_sha256"]}
        with open_dict(model.cfg):
            model.cfg.optim = OmegaConf.create(cfg["model"]["optim"])
            model.cfg.optim_param_groups = OmegaConf.create(cfg["model"]["optim_param_groups"])
            model.cfg.compute_eval_loss = False
            model.cfg.joint.fuse_loss_wer = True
            model.cfg.joint.fused_batch_size = cfg["joint_fused_batch_size"]
            model.cfg.joint.preserve_memory = cfg["joint_preserve_memory"]
        provenance = {"kind": "full_nemo_Lightning_actual_warmstart_v9", "run_id": args.run_id,
                      "preload": preload_result, "native_restore_connector": "official SaveRestoreConnector.model_extracted_dir",
                      "native_extracted_dir": str(scoped(cfg["native_extracted_dir"])), "allocator_cap_gb": cfg["allocator_cap_gb"], "resource_policy": POLICY,
                      "initialization": cfg["initialization"], "objective": cfg["objective"], "vocabulary": vocabulary,
                      "lora_targets": targets, "lora_rank": cfg["lora"]["rank"],
                      "trainable_tensor_count": sum(p.requires_grad for p in model.parameters()),
                      "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                      "frozen_encoder_dtype": "torch.bfloat16", "trainable_adapters_and_heads_dtype": "torch.float32",
                      "native_greedy_cuda_graph_decoder": False,
                      "periodic_checkpoint_policy": cfg["periodic_checkpoint_policy"],
                      "unused_allocator_cleanup": {"callback": "on_train_batch_end", "inactive_threshold_bytes": 750000000,
                                                   "live_tensors_or_math_mutated": False},
                      "official_training_step_source": str(Path(inspect.getfile(EncDecRNNTModel.training_step)).resolve()),
                      "official_training_step_sha256": sha(inspect.getfile(EncDecRNNTModel.training_step)),
                      "official_configure_optimizers_source": str(Path(inspect.getfile(ModelPT.configure_optimizers)).resolve()),
                      "official_configure_optimizers_sha256": sha(inspect.getfile(ModelPT.configure_optimizers)),
                      "custom_training_step_or_optimizer_override": False, "matched_HF_initialization_or_objective_claimed": False,
                      "full_native_GPU_memory_and_import_validation_pending_until_this_run": True}
        provenance["native_Lightning_warmstart"] = resume
        write_new(run_dir / "native_trial_provenance.json", provenance)
        trainer.validate(model, verbose=False)  # Actual trained100 DEV, not native-base baseline.
        trainer.fit(model)  # New official loops; Adam/Cosine restored in on_train_start.
        if callback.last_evaluated_step != callback.source_step + int(trainer.global_step):
            trainer.validate(model, verbose=False)
        trainer.save_checkpoint(str(run_dir / f"last_cumulative_step_{callback.source_step + int(trainer.global_step)}.ckpt"))
    except BaseException as exception:
        error = repr(exception)
        raise
    finally:
        callback.done.set()
        if callback.thread is not None:
            callback.thread.join(timeout=6)
        write_new(run_dir / "trial_outcome.json", {"run_id": args.run_id, "status": "failed" if error else "complete",
                  "actual_completed_optimizer_updates": 100 + int(trainer.global_step),
                  "local_completed_optimizer_updates": int(trainer.global_step), "source_optimizer_updates": 100, "error": error,
                  "stop_reason": callback.stop_reason, "selected_step": callback.selected_step,
                  "selected_checkpoint": str(callback.selected_path) if callback.selected_path else None,
                  "selected_turkish_development_wer": callback.best if callback.selected_step is not None else None,
                  "resource_telemetry": list(callback.telemetry), "resource_observations_omitted": callback.telemetry_omitted,
                  "inactive_cuda_cache_reclamations": callback.cache_reclamations,
                  "inactive_cuda_cache_cleanup_seconds": callback.cache_cleanup_seconds,
                  "TEST_evaluated": False, "useful_Turkish_accuracy_claimed": False,
                  "finished_at_utc": datetime.now(timezone.utc).isoformat()})


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, default=HERE / "full_nemo_turkish_v9.json")
    value.add_argument("--run-id", required=True)
    value.add_argument("--max-steps", type=int)
    value.add_argument("--source-checkpoint", type=Path, help="Pinned native06 periodic100 weights/Adam/Cosine; old Lightning loops are not restored")
    value.add_argument("--stop-file", type=Path)
    value.add_argument("--validate-only", action="store_true")
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    configuration, development, output, prepared = preflight(arguments)
    if arguments.validate_only:
        print(json.dumps(prepared, ensure_ascii=False))
    else:
        runtime(arguments, configuration, development, output, prepared)
