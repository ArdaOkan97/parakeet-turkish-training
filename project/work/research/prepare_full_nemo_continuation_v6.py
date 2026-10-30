"""Create only new files: B2 native full-pass continuation and B1 recovery specification."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = root / "outputs/parakeet_turkish"
entry = out / "train_full_nemo_turkish_v6.py"
cfgpath = out / "full_nemo_turkish_v6.json"
assert not entry.exists() and not cfgpath.exists()
old = out / "train_full_nemo_turkish_v3.py"
assert hashlib.sha256(old.read_bytes()).hexdigest() == "a7156a64e0db4c8e76f240cbeac9b4ba4218f794397c36394ff1b72bdb1faf2f"
pilot_entry = out / "train_full_nemo_turkish_v7.py"
pilot_cfg = out / "full_nemo_turkish_v7.json"
assert hashlib.sha256(pilot_entry.read_bytes()).hexdigest() == "6fae59450f55714fd5cd94e03778fc71dfad557a428d8c229f54bfa12047e99a"
assert hashlib.sha256(pilot_cfg.read_bytes()).hexdigest() == "597a52483e79197130666c678fd9fab4f41945d837dd4e3629ec4eb9b6fc315f"
text = old.read_text(encoding="utf-8")
text = text.replace("B4", "B2").replace("full_nemo_lightning_turkish_continuation_v3", "full_nemo_lightning_turkish_continuation_v6")
text = text.replace("full_nemo_Lightning_actual_continuation_v3", "full_nemo_Lightning_actual_continuation_v6")
text = text.replace("full_nemo_turkish_v3.json", "full_nemo_turkish_v6.json")
text = text.replace("full_nemo_lightning_turkish_pilot_v4", "full_nemo_lightning_turkish_pilot_v7")

def replace(before, after):
    global text
    assert text.count(before) == 1, before[:120]
    text = text.replace(before, after)

replace('cfg["model"]["train_ds"]["batch_size"] == 4 and cfg["trainer"]["accumulate_grad_batches"] == 4',
        'cfg["model"]["train_ds"]["batch_size"] == 2 and cfg["trainer"]["accumulate_grad_batches"] == 8')
replace('"This pilot declares B2 x accumulation4, effective16; mean_volume grouping differs from B8x2"',
        '"This continuation preserves native pilot B2 x accumulation8, effective16"')
replace('require(cfg["allocator_cap_gb"] == 8,', 'require(cfg["allocator_cap_gb"] == 9,')
replace('source_trainer.pop("val_check_interval") == 1000 and target_trainer.pop("val_check_interval") == 4000',
        'source_trainer.pop("val_check_interval") == 2000 and target_trainer.pop("val_check_interval") == 8000')
replace('def verify_lightning_checkpoint_metadata(checkpoint, accumulation=4):',
        'def verify_lightning_checkpoint_metadata(checkpoint, accumulation=8):')
replace('require(len(batch[1]) == 4,', 'require(len(batch[1]) == 2,')
replace('from lightning.pytorch.callbacks import Callback\n', 'from lightning.pytorch.callbacks import Callback, ModelCheckpoint\n')
replace('    trainer = pl.Trainer(**cfg["trainer"], callbacks=[callback, telemetry], logger=logger, default_root_dir=str(run_dir))',
        '    periodic = ModelCheckpoint(dirpath=str(run_dir / "periodic_checkpoints"),\n'
        '                               filename="step_{step:06d}", auto_insert_metric_name=False,\n'
        '                               every_n_train_steps=500, save_top_k=1, save_last=False,\n'
        '                               save_on_train_epoch_end=False)\n'
        '    trainer = pl.Trainer(**cfg["trainer"], callbacks=[callback, telemetry, periodic], logger=logger, default_root_dir=str(run_dir))')
replace('            write_new(run_dir / f"development_gate_step_{step}.json", gate)\n',
        '            write_new(run_dir / f"development_gate_step_{step}.json", gate)\n'
        '            # DEV runs after Lightning completes all prior microbatches.\n'
        '            # Preserve a full stock checkpoint even when this model is not selected.\n'
        '            recovery_path = self.selected_path if self.selected_step == step else run_dir / f"recovery_dev_step_{step}.ckpt"\n'
        '            if self.selected_step != step:\n'
        '                trainer.save_checkpoint(str(recovery_path))\n'
        '            write_new(run_dir / f"recovery_checkpoint_step_{step}.json", {"step": step, "path": str(recovery_path),\n'
        '                      "checkpoint_sha256": sha(recovery_path), "phase": "completed_DEV_boundary",\n'
        '                      "official_Lightning_checkpoint": True, "development_selection_required_for_recovery": False})\n')
replace('"native_greedy_cuda_graph_decoder": False,\n',
        '"native_greedy_cuda_graph_decoder": False,\n'
        '                      "periodic_checkpoint_policy": cfg["periodic_checkpoint_policy"],\n')
replace('    require(args.ckpt_path is not None,',
        '    require(cfg["trainer"]["enable_checkpointing"] is True and cfg["periodic_checkpoint_policy"]["every_n_train_steps"] == 500, "Declared stock periodic checkpoint policy changed")\n'
        '    require(args.ckpt_path is not None,')
replace('    require(outcome["run_id"] == prepared["run_id"] == source_dir.name, "Source native run identity disagrees")\n',
        '    require(outcome["run_id"] == prepared["run_id"] == source_dir.name, "Source native run identity disagrees")\n'
        '    require(prepared["config_sha256"] == cfg["source_pilot_config_sha256"], "Source native pilot configuration changed")\n')
ast.parse(text)
cfg = json.loads(pilot_cfg.read_text(encoding="utf-8"))
cfg.update(kind="full_nemo_lightning_turkish_continuation_v6", pilot_optimizer_update_limit=9361,
           pilot_training_manifest="work/nemo_direct_trial_v1/manifests/train_balanced_250.jsonl",
           continuation_data_proof="outputs/parakeet_turkish/full_nemo_training_continuation_v1.json",
           source_pilot_entry_sha256=hashlib.sha256(pilot_entry.read_bytes()).hexdigest(),
           source_pilot_config_sha256=hashlib.sha256(pilot_cfg.read_bytes()).hexdigest())
cfg["continuation_data_proof_sha256"] = hashlib.sha256((root / cfg["continuation_data_proof"]).read_bytes()).hexdigest()
cfg["model"]["train_ds"]["manifest_filepath"] = "work/nemo_direct_trial_v1/manifests/train_balanced_full_pass.jsonl"
cfg["trainer"]["val_check_interval"] = 8000
cfg["periodic_checkpoint_policy"].update(every_n_train_steps=500, save_top_k=1,
    maximum_periodic_snapshots_in_bounded_pilot=None, maximum_retained_periodic_snapshots=1,
    first_continuation_periodic_global_step=500,
    completed_source250_not_a_periodic_trigger=True)
cfg["continuation"] = {"source_native_completed_updates": 250, "data_prefix_rows": 4000,
    "target_global_optimizer_updates": 9361, "physical_batch": 2, "accumulation": 8,
    "effective_recordings_per_update": 16, "optimizer_and_scheduler_restarted": False,
    "cosine_horizon_optimizer_updates": 9361, "warmup_restarted": False,
    "development_global_update_interval": 1000, "completed_source_DEV_reused": True,
    "native_pilot_completion_and_hardware_acceptance_pending": True,
    "full_checkpoint_at_every_completed_DEV_even_if_unqualified": True,
    "B1_grouping_transition_enabled_in_this_entry": False}
entry.write_text(text, encoding="utf-8", newline="\n")
cfgpath.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8", newline="\n")
recovery = {"schema": 1, "kind": "native_B2_to_B1_stock_Trainer_recovery_specification_v1",
    "status": "inert_pending_actual_source_checkpoint_metadata_and_explicit_transition",
    "source_pilot_entry_sha256": cfg["source_pilot_entry_sha256"],
    "source_config_sha256": cfg["source_pilot_config_sha256"],
    "source_configuration": {"physical_batch": 2, "accumulation": 8, "allocator_cap_gb": 9},
    "target_configuration": {"physical_batch": 1, "accumulation": 16, "allocator_cap_gb": 9},
    "unchanged_effective_recordings_per_optimizer_update": 16,
    "declared_scientific_change": "Stock mean_volume is normalized separately per physical microbatch; B1x16 changes gradient weighting relative to B2x8. Neither bitwise gradients nor matched quality are claimed.",
    "retain": ["native model weights", "AdamW states", "Cosine scheduler state and9361 horizon", "global optimizer step", "same ordered TRAIN stream and consumed recordings"],
    "required_before_actual_launch": [
        "Actual source checkpoint metadata: global_step, optimizer progress, scheduler state, batch ready/started/processed/completed counters.",
        "Prove source consumed_recordings=completed_optimizer_updates*16 from completed native updates and ordered manifest.",
        "An explicit loop/data progress transition aligning target B1 microbatch counters with consumed recordings, reviewed before writing a new checkpoint or launch.",
        "Handle stock periodic batch-end completed-counter lag explicitly; no automatic periodic checkpoint resume.",
        "Fresh declared source/target configuration and provenance; keep original checkpoint/artifacts immutable.",
        "Start sampler at first unseen source index and verify actual first resumed update/cadence.",
        "Same critical host guards and real hardware memory validation; full native B1 fit is still pending."],
    "active_source_or_checkpoint_modified": False, "GPU_launched": False}
(out / "full_nemo_B1_recovery_specification_v1.json").write_text(json.dumps(recovery, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"entry_sha256": hashlib.sha256(entry.read_bytes()).hexdigest(), "config_sha256": hashlib.sha256(cfgpath.read_bytes()).hexdigest()}))
