"""Create a separate B1 native arm; no Torch/NeMo imports or GPU work."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = root / "outputs/parakeet_turkish"
entry = out / "train_full_nemo_turkish_v9.py"
config_path = out / "full_nemo_turkish_v9.json"
assert not entry.exists() and not config_path.exists()
previous = out / "train_full_nemo_turkish_v6.py"
assert hashlib.sha256(previous.read_bytes()).hexdigest() == "ca0f6949a2b1a77aebfb7d22f9a65f3d6c16eeafc0536a5dfb21aa8b109c0d1c"
text = previous.read_text(encoding="utf-8")
text = text.replace("Native B2 Parakeet/NeMo/Lightning full-pass continuation", "Native B1 Parakeet/NeMo/Lightning warmstart arm")
text = text.replace("full_nemo_lightning_turkish_continuation_v6", "full_nemo_lightning_turkish_warmstart_v9")
text = text.replace("full_nemo_Lightning_actual_continuation_v6", "full_nemo_Lightning_actual_warmstart_v9")
text = text.replace("full_nemo_turkish_v6.json", "full_nemo_turkish_v9.json")
text = text.replace('pilot_optimizer_update_limit', 'local_optimizer_update_limit')

def replace(before, after):
    global text
    assert text.count(before) == 1, before[:160]
    text = text.replace(before, after)

start = text.index("def validate_source_resume(")
end = text.index("def preflight(", start)
helpers = r'''
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
    for field in ("seed", "initialization", "init_from_nemo_model", "lora", "allocator_cap_gb", "joint_fused_batch_size",
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

'''
text = text[:start] + helpers + text[end:]
replace('cfg["model"]["train_ds"]["batch_size"] == 2 and cfg["trainer"]["accumulate_grad_batches"] == 8',
        'cfg["model"]["train_ds"]["batch_size"] == 1 and cfg["trainer"]["accumulate_grad_batches"] == 16')
replace('"This continuation preserves native pilot B2 x accumulation8, effective16"',
        '"This arm declares B1 x accumulation16, effective16; mean_volume grouping changes"')
replace('    require(isinstance(steps, int) and 250 < steps <= 9361, "Continuation is bounded to the original9361 horizon")',
        '    require(isinstance(steps, int) and 1 <= steps <= 9261, "New local updates are bounded to the remaining original9361 horizon")')
replace('cfg["trainer"]["val_check_interval"] == 1000 * cfg["trainer"]["accumulate_grad_batches"]',
        'cfg["trainer"]["val_check_interval"] == 250 * cfg["trainer"]["accumulate_grad_batches"]')
replace('"Continuation DEV cadence must be1000 optimizer updates"', '"Initial feedback must be local250 optimizer updates"')
replace('cfg["periodic_checkpoint_policy"]["every_n_train_steps"] == 500', 'cfg["periodic_checkpoint_policy"]["every_n_train_steps"] == 250')
replace('    require(args.ckpt_path is not None, "This continuation requires the actual completed B2 native pilot checkpoint")\n'
        '    inherited = validate_source_resume(args.ckpt_path, cfg)',
        '    source_checkpoint = args.source_checkpoint or scoped(cfg["warmstart"]["checkpoint_path"])\n'
        '    inherited = validate_source_warmstart(source_checkpoint, cfg)')
replace('"source_native_resume": inherited', '"source_native_warmstart": inherited')
replace('            inherited = validation["source_native_resume"]\n'
        '            self.last_evaluated_step = inherited["source_step"]\n'
        '            self.selected_step = inherited["selected_step"]\n'
        '            self.selected_path = Path(inherited["selected_checkpoint"]) if inherited["selected_checkpoint"] else None\n'
        '            self.best = inherited["selected_turkish_development_wer"] if self.selected_step is not None else float("inf")\n'
        '            self.retention_failures = inherited["consecutive_retention_failures"]\n'
        '            self.first_resumed_batch_checked = False\n'
        '\n'
        '        def on_train_start(self, trainer, model):\n'
        '            require(int(trainer.global_step) == validation["source_native_resume"]["source_step"], "Lightning failed to restore source global_step")',
        '            self.source_step = 100\n'
        '            self.last_evaluated_step = None\n'
        '            self.first_resumed_batch_checked = False\n'
        '            self.source_optimizer_state = self.source_scheduler_state = None\n'
        '            self.optimizer_restore_evidence = None\n'
        '\n'
        '        def on_train_start(self, trainer, model):\n'
        '            require(int(trainer.global_step) == 0 and len(trainer.optimizers) == len(trainer.lr_scheduler_configs) == 1, "Official Lightning loops must start fresh at local0")\n'
        '            self.optimizer_restore_evidence = restore_optimizer_scheduler(trainer.optimizers[0], trainer.lr_scheduler_configs[0].scheduler,\n'
        '                                                        self.source_optimizer_state, self.source_scheduler_state)\n'
        '            self.source_optimizer_state = self.source_scheduler_state = None\n'
        '            write_new(run_dir / "optimizer_warmstart_restoration.json", self.optimizer_restore_evidence)')
replace('                require(int(trainer.global_step) == 250, "First continuation update must follow native pilot250")\n'
        '                require(len(batch[1]) == 2, "Physical B2 grouping must remain unchanged")',
        '                require(int(trainer.global_step) == 0, "First new update must follow fresh local loop0")\n'
        '                require(len(batch[1]) == 1, "Declared target physical B1 grouping required")')
replace('            step = int(trainer.global_step)\n            self.last_evaluated_step = step',
        '            local_step = int(trainer.global_step)\n'
        '            step = self.source_step + local_step\n'
        '            self.last_evaluated_step = step\n'
        '            if local_step >= 250:\n'
        '                trainer.val_check_batch = 1000 * cfg["trainer"]["accumulate_grad_batches"]')
replace('            gate = {"step": step, "turkish_wer": scored["tr"]["wer"],',
        '            gate = {"step": step, "local_step": local_step, "source_optimizer_updates": self.source_step,\n'
        '                    "turkish_wer": scored["tr"]["wer"],')
replace('                               every_n_train_steps=500, save_top_k=1, save_last=False,',
        '                               every_n_train_steps=250, save_top_k=1, save_last=False,')
replace('filename="step_{step:06d}", auto_insert_metric_name=False,',
        'filename="local_step_{step:06d}", auto_insert_metric_name=False,')
start = text.index('        resume = None\n')
end = text.index('        with open_dict(model.cfg):\n', start)
runtime_source = '''        source_path = Path(validation["source_native_warmstart"]["checkpoint_path"])
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
'''
text = text[:start] + runtime_source + text[end:]
replace('        provenance["native_Lightning_continuation"] = resume', '        provenance["native_Lightning_warmstart"] = resume')
replace('        if not args.ckpt_path:\n'
        '            trainer.validate(model, verbose=False)  # Native baseline, same300DEV/commonnormalizer, no TEST.\n'
        '        trainer.fit(model, ckpt_path=str(scoped(args.ckpt_path)) if args.ckpt_path else None)\n'
        '        if callback.last_evaluated_step != int(trainer.global_step):\n'
        '            trainer.validate(model, verbose=False)\n'
        '        trainer.save_checkpoint(str(run_dir / f"last_step_{int(trainer.global_step)}.ckpt"))',
        '        trainer.validate(model, verbose=False)  # Actual trained100 DEV, not native-base baseline.\n'
        '        trainer.fit(model)  # New official loops; Adam/Cosine restored in on_train_start.\n'
        '        if callback.last_evaluated_step != callback.source_step + int(trainer.global_step):\n'
        '            trainer.validate(model, verbose=False)\n'
        '        trainer.save_checkpoint(str(run_dir / f"last_cumulative_step_{callback.source_step + int(trainer.global_step)}.ckpt"))')
replace('"actual_completed_optimizer_updates": int(trainer.global_step), "error": error,',
        '"actual_completed_optimizer_updates": 100 + int(trainer.global_step),\n'
        '                  "local_completed_optimizer_updates": int(trainer.global_step), "source_optimizer_updates": 100, "error": error,')
replace('    value.add_argument("--ckpt-path", type=Path, required=True, help="Actual completed B2 native pilot250 last_step_250.ckpt; retain AdamW and Cosine state")',
        '    value.add_argument("--source-checkpoint", type=Path, help="Pinned native06 periodic100 weights/Adam/Cosine; old Lightning loops are not restored")')
text = text.replace('"declared_full_pass_update_limit"', '"declared_remaining_local_update_limit"')
ast.parse(text)
cfg = json.loads((out / "full_nemo_turkish_v6.json").read_text(encoding="utf-8"))
cfg.update(kind="full_nemo_lightning_turkish_warmstart_v9", local_optimizer_update_limit=9261)
cfg.pop("pilot_optimizer_update_limit")
cfg["model"]["train_ds"]["batch_size"] = 1
cfg["trainer"]["accumulate_grad_batches"] = 16
cfg["trainer"]["val_check_interval"] = 4000
cfg["objective"]["physical_batch_normalization_change"] = "B1xaccum16 averages16 individually target-normalized mean_volume losses; source B2xaccum8 averages8 pair-normalized losses. Same effective16 ordered recordings; gradient equivalence is not claimed."
cfg["periodic_checkpoint_policy"].update(every_n_train_steps=250, first_continuation_periodic_global_step=None,
    completed_source250_not_a_periodic_trigger=None, local_step_in_checkpoint_is_not_cumulative=True,
    cumulative_step_formula="100 + local Lightning global_step")
cfg.pop("continuation")
cfg["warmstart"] = {"source_run_dir": "outputs/parakeet_turkish/runs/full_nemo_turkish_20261011_pilot_06",
    "checkpoint_path": "outputs/parakeet_turkish/runs/full_nemo_turkish_20261011_pilot_06/periodic_checkpoints/step_000100.ckpt",
    "checkpoint_bytes": 1457579739, "checkpoint_sha256": "1afdd04aef11025f6578af8b6186faa120058ac53abce75939102b12c8295512",
    "source_completed_optimizer_updates": 100, "source_consumed_recordings": 1600,
    "target_local_optimizer_update_limit": 9261, "target_cumulative_optimizer_updates": 9361,
    "source_physical_batch": 2, "source_accumulation": 8, "target_physical_batch": 1, "target_accumulation": 16,
    "effective_recordings_per_update": 16, "source_fit_loops_restored": False, "source_RNG_restored": False,
    "Adam_and_Cosine_restored_after_native_configuration": True, "scheduler_horizon": 9361,
    "optimizer_and_scheduler_steps_called_during_restore": False,
    "development_local_steps": "initial0 (cumulative100), local250, then1000/2000/... and final/owned stop",
    "strict_full_model_gradient_equivalence_claimed": False, "actual_GPU_memory_fit_and_restore_pending": True}
entry.write_text(text, encoding="utf-8", newline="\n")
config_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8", newline="\n")
print(json.dumps({"entry_sha256": hashlib.sha256(entry.read_bytes()).hexdigest(), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}))
