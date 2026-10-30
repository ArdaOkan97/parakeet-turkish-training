"""Separate native Parakeet/NeMo/Lightning pilot; never dispatches the HF loop.

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


def preflight(args):
    config_path = scoped(args.config)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    require(cfg["schema"] == 1 and cfg["kind"] == "full_nemo_lightning_turkish_pilot_v4", "Wrong trial config")
    require(args.run_id and all(c.isalnum() or c in "_-" for c in args.run_id), "Unsafe trial ID")
    require(cfg["trainer"]["devices"] == 1 and cfg["trainer"]["accelerator"] == "gpu", "Single local GPU only")
    require(cfg["model"]["train_ds"]["batch_size"] == 4 and cfg["trainer"]["accumulate_grad_batches"] == 4,
            "This pilot declares B4 x accumulation4, effective16; mean_volume grouping differs from B8x2")
    require(cfg["joint_fused_batch_size"] in (1, 2), "Start with native joint subbatch1 or2")
    require(cfg["allocator_cap_gb"] in (8, 9), "Declared bounded allocator cap required")
    require(cfg["initialization"] == "original_native_v3_nemo_no_HF_trained_weight_transfer", "Unreviewed initialization")
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
    train_rows = read_rows(train)
    steps = args.max_steps if args.max_steps is not None else cfg["pilot_optimizer_update_limit"]
    require(isinstance(steps, int) and 1 <= steps <= 250, "Initial pilot is bounded to250 optimizer updates")
    require(len(train_rows) >= steps * 16, "Native training plan lacks required effective16 updates")
    require(all(row.get("training_role") == "train" and row.get("dataset_split") in ("train", "validated")
                and row["language"] in LANGUAGES and 0.5 <= row["duration"] <= 30 for row in train_rows),
            "Native manifest has unvetted training role/language/duration")
    require(all(row["language"] == "tr" for row in train_rows[0:8]) and
            all(row["language"] != "tr" for row in train_rows[8:16]), "Expected8TR then8replay plan")
    dev = {}
    for language, value in zip(LANGUAGES, cfg["model"]["validation_ds"]["manifest_filepath"]):
        path = scoped(value)
        rows = read_rows(path)
        require(len(rows) == 50 and all(row["language"] == language and row["dataset_split"] == "validation" for row in rows),
                "Each native DEV loader must preserve the fixed50 recordings")
        dev[language] = rows
    require(set(proof["manifest_sha256"]) == {str(train), *(str(scoped(p)) for p in cfg["model"]["validation_ds"]["manifest_filepath"])},
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
    cfg["pilot_optimizer_update_limit"] = steps
    cfg["trainer"]["val_check_interval"] = steps * cfg["trainer"]["accumulate_grad_batches"]
    if args.ckpt_path:
        require(scoped(args.ckpt_path).is_file(), "Native Lightning continuation checkpoint missing")
        source_config = scoped(args.ckpt_path).parent / "resolved_trial_config.json"
        require(source_config.is_file(), "Native checkpoint must have its scoped source trial config")
        source = json.loads(source_config.read_text(encoding="utf-8"))
        for field in ("seed", "initialization", "init_from_nemo_model", "model", "lora", "objective",
                      "allocator_cap_gb", "joint_fused_batch_size", "joint_preserve_memory", "source_pins"):
            require(source[field] == cfg[field], "Native resume scientific/config identity changed:" + field)
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
        "native_import_and_training_pending": True, "unchanged_HF_recipe_claimed": False,
    }


def runtime(args, cfg, dev, run_dir, validation):
    # Ordering matters on native Windows: coherent12.9 linkage beforeTorch/Numba.
    from preload_nemo_linker_v1 import preload
    preload_result = preload()
    import torch
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback
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
            self.last_evaluated_step = None

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
            if int(trainer.global_step) >= cfg["pilot_optimizer_update_limit"]:
                self.stop_reason = self.stop_reason or "declared_pilot_update_limit"
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
            step = int(trainer.global_step)
            self.last_evaluated_step = step
            scored = {}
            for lang in LANGUAGES:
                require(len(self.rows[lang]) == len(dev[lang]), "Native DEV predictions/order incomplete for" + lang)
                rows = [{"language": lang, "reference": row["text"], "hypothesis": hypothesis,
                         "source_id": row["source_id"], "sentence_id": row.get("sentence_id")}
                        for row, hypothesis in zip(dev[lang], self.rows[lang])]
                scored[lang] = score_transcriptions(rows, language=lang)
                write_new(run_dir / f"dev_step_{step}_{lang}.json", {"step": step, "language": lang, "predictions": rows, "asr": scored[lang]})
            gate = {"step": step, "turkish_wer": scored["tr"]["wer"],
                    "retention_gate_passed": all(scored[lang]["wer"] <= baseline_wer[lang] + max(.005, .1 * baseline_wer[lang])
                                                  for lang in LANGUAGES if lang != "tr"),
                    "per_language_wer": {lang: scored[lang]["wer"] for lang in LANGUAGES}}
            qualifying = gate["retention_gate_passed"] and gate["turkish_wer"] < baseline_wer["tr"]
            if qualifying and gate["turkish_wer"] < self.best:
                self.best, self.selected_step = gate["turkish_wer"], step
                self.selected_path = run_dir / f"qualified_step_{step}.ckpt"
                trainer.save_checkpoint(str(self.selected_path))  # Official Lightning weights/Adam/scheduler/loop checkpoint.
            self.retention_failures = 0 if gate["retention_gate_passed"] else self.retention_failures + 1
            if self.retention_failures >= cfg["retention_failures_before_stop"]:
                self.stop_reason = "consecutive_development_retention_failures"
                trainer.should_stop = True
            gate.update(selected_step=self.selected_step, qualified=qualifying, stop_reason=self.stop_reason)
            write_new(run_dir / f"development_gate_step_{step}.json", gate)
            print(json.dumps({"native_dev": gate}), flush=True)

    callback = NativeTrialCallback()
    telemetry = make_native_training_telemetry_callback(pl, torch, run_dir, args.run_id)
    logger = CSVLogger(save_dir=str(run_dir), name="lightning", version="pilot")
    trainer = pl.Trainer(**cfg["trainer"], callbacks=[callback, telemetry], logger=logger, default_root_dir=str(run_dir))
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
        require(len(model._train_dl.dataset) == len(read_rows(scoped(train_cfg.manifest_filepath))),
                "Official NeMo dataset filtered unexpected TRAIN records")
        resume = None
        if args.ckpt_path:
            checkpoint_path = scoped(args.ckpt_path)
            checkpoint_meta = torch.load(checkpoint_path, map_location="meta", weights_only=False)
            completed = int(checkpoint_meta["global_step"])
            del checkpoint_meta
            require(0 < completed < cfg["pilot_optimizer_update_limit"], "Continuation must increase the completed update count")
            consumed = completed * cfg["trainer"]["accumulate_grad_batches"] * train_cfg.batch_size
            dataset = model._train_dl.dataset
            require(consumed < len(dataset), "Pilot continuation has exhausted its fixed TRAIN plan")

            class NativeResumeSequentialSampler(torch.utils.data.SequentialSampler):
                # Lightning restores logical batch counters but this installed
                # map-style loader does not restore its iterator. Keep the full
                # logical length used by those counters and skip consumed records.
                def __iter__(self):
                    return iter(range(consumed, len(self.data_source)))

            original = model._train_dl
            model._train_dl = torch.utils.data.DataLoader(dataset, batch_size=original.batch_size,
                sampler=NativeResumeSequentialSampler(dataset), collate_fn=original.collate_fn,
                num_workers=0, pin_memory=False, drop_last=False)
            resume = {"checkpoint_path": str(checkpoint_path), "checkpoint_sha256": sha(checkpoint_path),
                      "source_step": completed, "consumed_training_records": consumed,
                      "logical_training_records": len(dataset), "remaining_training_records": len(dataset)-consumed,
                      "loader_cursor": "SequentialSampler offset; official native dataset/collator, no repeated smoke prefix",
                      "stock_Lightning_optimizer_scheduler_and_loop_restore": True,
                      "bitwise_RNG_continuation_claimed": False}
        with open_dict(model.cfg):
            model.cfg.optim = OmegaConf.create(cfg["model"]["optim"])
            model.cfg.optim_param_groups = OmegaConf.create(cfg["model"]["optim_param_groups"])
            model.cfg.compute_eval_loss = False
            model.cfg.joint.fuse_loss_wer = True
            model.cfg.joint.fused_batch_size = cfg["joint_fused_batch_size"]
            model.cfg.joint.preserve_memory = cfg["joint_preserve_memory"]
        provenance = {"kind": "full_nemo_Lightning_actual_trial_v4", "run_id": args.run_id,
                      "preload": preload_result, "native_restore_connector": "official SaveRestoreConnector.model_extracted_dir",
                      "native_extracted_dir": str(scoped(cfg["native_extracted_dir"])), "allocator_cap_gb": cfg["allocator_cap_gb"], "resource_policy": POLICY,
                      "initialization": cfg["initialization"], "objective": cfg["objective"], "vocabulary": vocabulary,
                      "lora_targets": targets, "lora_rank": cfg["lora"]["rank"],
                      "trainable_tensor_count": sum(p.requires_grad for p in model.parameters()),
                      "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                      "frozen_encoder_dtype": "torch.bfloat16", "trainable_adapters_and_heads_dtype": "torch.float32",
                      "native_greedy_cuda_graph_decoder": False,
                      "unused_allocator_cleanup": {"callback": "on_train_batch_end", "inactive_threshold_bytes": 750000000,
                                                   "live_tensors_or_math_mutated": False},
                      "official_training_step_source": str(Path(inspect.getfile(EncDecRNNTModel.training_step)).resolve()),
                      "official_training_step_sha256": sha(inspect.getfile(EncDecRNNTModel.training_step)),
                      "official_configure_optimizers_source": str(Path(inspect.getfile(ModelPT.configure_optimizers)).resolve()),
                      "official_configure_optimizers_sha256": sha(inspect.getfile(ModelPT.configure_optimizers)),
                      "custom_training_step_or_optimizer_override": False, "matched_HF_initialization_or_objective_claimed": False,
                      "full_native_GPU_memory_and_import_validation_pending_until_this_run": True}
        provenance["native_Lightning_continuation"] = resume
        write_new(run_dir / "native_trial_provenance.json", provenance)
        if not args.ckpt_path:
            trainer.validate(model, verbose=False)  # Native baseline, same300DEV/commonnormalizer, no TEST.
        trainer.fit(model, ckpt_path=str(scoped(args.ckpt_path)) if args.ckpt_path else None)
        if callback.last_evaluated_step != int(trainer.global_step):
            trainer.validate(model, verbose=False)
        trainer.save_checkpoint(str(run_dir / f"last_step_{int(trainer.global_step)}.ckpt"))
    except BaseException as exception:
        error = repr(exception)
        raise
    finally:
        callback.done.set()
        if callback.thread is not None:
            callback.thread.join(timeout=6)
        write_new(run_dir / "trial_outcome.json", {"run_id": args.run_id, "status": "failed" if error else "complete",
                  "actual_completed_optimizer_updates": int(trainer.global_step), "error": error,
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
    value.add_argument("--config", type=Path, default=HERE / "full_nemo_turkish_v5.json")
    value.add_argument("--run-id", required=True)
    value.add_argument("--max-steps", type=int)
    value.add_argument("--ckpt-path", type=Path, help="Continue native Lightning weights/optimizer/scheduler/loop from a preceding pilot")
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

