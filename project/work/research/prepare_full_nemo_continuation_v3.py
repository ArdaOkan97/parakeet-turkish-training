"""Source-only construction of the separate B4 native continuation."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = root / "outputs/parakeet_turkish"
source = out / "train_full_nemo_turkish_v4.py"
target = out / "train_full_nemo_turkish_v3.py"
config_target = out / "full_nemo_turkish_v3.json"
assert not target.exists() and not config_target.exists()
assert hashlib.sha256(source.read_bytes()).hexdigest() == "f5ca5f1245872454657ff4a86ec2f6f069a53eb7f6fc3820dcaf5cc643facb47"
text = source.read_text(encoding="utf-8")

def replace(old, new):
    global text
    assert text.count(old) == 1, old[:100]
    text = text.replace(old, new)

replace('"full_nemo_lightning_turkish_pilot_v4"', '"full_nemo_lightning_turkish_continuation_v3"')
replace('"full_nemo_Lightning_actual_trial_v4"', '"full_nemo_Lightning_actual_continuation_v3"')
replace('default=HERE / "full_nemo_turkish_v4.json"', 'default=HERE / "full_nemo_turkish_v3.json"')
replace('    train_rows = read_rows(train)\n', '    train_count, stream_audit = validate_full_stream(train, proof, cfg)\n')
replace('    require(isinstance(steps, int) and 1 <= steps <= 250, "Initial pilot is bounded to250 optimizer updates")\n'
        '    require(len(train_rows) >= steps * 16, "Native training plan lacks required effective16 updates")\n'
        '    require(all(row.get("training_role") == "train" and row.get("dataset_split") in ("train", "validated")\n'
        '                and row["language"] in LANGUAGES and 0.5 <= row["duration"] <= 30 for row in train_rows),\n'
        '            "Native manifest has unvetted training role/language/duration")\n'
        '    require(all(row["language"] == "tr" for row in train_rows[0:8]) and\n'
        '            all(row["language"] != "tr" for row in train_rows[8:16]), "Expected8TR then8replay plan")\n',
        '    require(isinstance(steps, int) and 250 < steps <= 9361, "Continuation is bounded to the original9361 horizon")\n'
        '    require(train_count == 9361 * 16 and train_count >= steps * 16, "Full native plan must contain exactly9361 effective16 updates")\n')
replace('    require(set(proof["manifest_sha256"]) == {str(train), *(str(scoped(p)) for p in cfg["model"]["validation_ds"]["manifest_filepath"])},\n',
        '    require(set(proof["manifest_sha256"]) == {str(scoped(cfg["pilot_training_manifest"])), *(str(scoped(p)) for p in cfg["model"]["validation_ds"]["manifest_filepath"])},\n')
replace('    cfg["trainer"]["val_check_interval"] = steps * cfg["trainer"]["accumulate_grad_batches"]\n'
        '    if args.ckpt_path:\n'
        '        require(scoped(args.ckpt_path).is_file(), "Native Lightning continuation checkpoint missing")\n'
        '        source_config = scoped(args.ckpt_path).parent / "resolved_trial_config.json"\n'
        '        require(source_config.is_file(), "Native checkpoint must have its scoped source trial config")\n'
        '        source = json.loads(source_config.read_text(encoding="utf-8"))\n'
        '        for field in ("seed", "initialization", "init_from_nemo_model", "model", "lora", "objective",\n'
        '                      "allocator_cap_gb", "joint_fused_batch_size", "joint_preserve_memory", "source_pins"):\n'
        '            require(source[field] == cfg[field], "Native resume scientific/config identity changed:" + field)\n',
        '    require(cfg["trainer"]["val_check_interval"] == 1000 * cfg["trainer"]["accumulate_grad_batches"], "Continuation DEV cadence must be1000 optimizer updates")\n'
        '    require(args.ckpt_path is not None, "This continuation requires the actual completed B4 native pilot checkpoint")\n'
        '    inherited = validate_source_resume(args.ckpt_path, cfg)\n')
replace('        "optimizer_updates": steps, "Torch_imported": "torch" in sys.modules,\n',
        '        "optimizer_updates": steps, "Torch_imported": "torch" in sys.modules,\n'
        '        "full_stream_audit": stream_audit, "source_native_resume": inherited,\n')
replace('            self.last_evaluated_step = None\n',
        '            inherited = validation["source_native_resume"]\n'
        '            self.last_evaluated_step = inherited["source_step"]\n'
        '            self.selected_step = inherited["selected_step"]\n'
        '            self.selected_path = Path(inherited["selected_checkpoint"]) if inherited["selected_checkpoint"] else None\n'
        '            self.best = inherited["selected_turkish_development_wer"] if self.selected_step is not None else float("inf")\n'
        '            self.retention_failures = inherited["consecutive_retention_failures"]\n'
        '            self.first_resumed_batch_checked = False\n'
        '\n'
        '        def on_train_start(self, trainer, model):\n'
        '            require(int(trainer.global_step) == validation["source_native_resume"]["source_step"], "Lightning failed to restore source global_step")\n')
replace('            model.encoder.eval()  # Frozen nativeBN/dropout; gradients still reachLoRA.\n',
        '            model.encoder.eval()  # Frozen nativeBN/dropout; gradients still reachLoRA.\n'
        '            if not self.first_resumed_batch_checked:\n'
        '                require(int(trainer.global_step) == 250, "First continuation update must follow native pilot250")\n'
        '                require(len(batch[1]) == 4, "Physical B4 grouping must remain unchanged")\n'
        '                self.first_resumed_batch_checked = True\n')
replace('        require(len(model._train_dl.dataset) == len(read_rows(scoped(train_cfg.manifest_filepath))),\n',
        '        require(len(model._train_dl.dataset) == validation["full_stream_audit"]["training_records"],\n')
replace('            require(0 < completed < cfg["pilot_optimizer_update_limit"], "Continuation must increase the completed update count")\n',
        '            require(completed == 250 and completed < cfg["pilot_optimizer_update_limit"], "Full pass must start from actual native pilot250")\n'
        '            require(len(checkpoint_meta.get("optimizer_states", [])) == 1 and len(checkpoint_meta.get("lr_schedulers", [])) == 1, "Complete stock AdamW and Cosine checkpoint state required")\n')
# Move deletion after checking meta fields. Never allocate the full source state on CPU here.
replace('            del checkpoint_meta\n            require(completed == 250', '            require(completed == 250')
replace('            consumed = completed * cfg["trainer"]["accumulate_grad_batches"] * train_cfg.batch_size\n',
        '            progress = checkpoint_meta["loops"]["fit_loop"]["epoch_loop.batch_progress"]\n'
        '            require(progress["total"]["completed"] == progress["total"]["ready"] == completed * 4, "Checkpoint contains an unfinished microbatch/accumulation")\n'
        '            del checkpoint_meta\n'
        '            consumed = completed * cfg["trainer"]["accumulate_grad_batches"] * train_cfg.batch_size\n'
        '            require(consumed == 4000, "The full-pass pilot prefix must be consumed exactly once")\n')
replace('            require(consumed < len(dataset), "Pilot continuation has exhausted its fixed TRAIN plan")',
        '            require(consumed < len(dataset), "Full native continuation has exhausted its TRAIN plan")')
replace('                      "stock_Lightning_optimizer_scheduler_and_loop_restore": True,\n',
        '                      "stock_Lightning_optimizer_scheduler_and_loop_restore": True,\n'
        '                      "full_manifest_sha256": validation["full_stream_audit"]["full_manifest_sha256"],\n'
        '                      "source_development_state": validation["source_native_resume"],\n')
replace('    value.add_argument("--ckpt-path", type=Path, help="Continue native Lightning weights/optimizer/scheduler/loop from a preceding pilot")',
        '    value.add_argument("--ckpt-path", type=Path, required=True, help="Actual completed B4 native pilot250 last_step_250.ckpt; retain AdamW and Cosine state")')

helpers = r'''
def validate_full_stream(train, pilot_proof, cfg):
    """Stream only metadata; never read PCM or retain the149,776 rows in RAM."""
    full_proof_path = scoped(cfg["continuation_data_proof"])
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


def validate_source_resume(checkpoint_value, cfg):
    checkpoint = scoped(checkpoint_value)
    require(checkpoint.is_file() and checkpoint.name == "last_step_250.ckpt", "Need the actual complete native pilot250 checkpoint")
    source_dir = checkpoint.parent
    require(source_dir.is_relative_to(HERE / "runs"), "Source checkpoint must belong to a native workspace run")
    config_path = source_dir / "resolved_trial_config.json"
    outcome_path = source_dir / "trial_outcome.json"
    preflight_path = source_dir / "preflight.json"
    require(all(p.is_file() for p in (config_path, outcome_path, preflight_path)), "Completed source pilot evidence missing")
    source = json.loads(config_path.read_text(encoding="utf-8"))
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    prepared = json.loads(preflight_path.read_text(encoding="utf-8"))
    require(source["kind"] == "full_nemo_lightning_turkish_pilot_v4" and prepared["entry_sha256"] == cfg["source_pilot_entry_sha256"],
            "Continuation must preserve the actual B4 native pilot recipe")
    require(outcome["status"] == "complete" and outcome["error"] is None and outcome["actual_completed_optimizer_updates"] == 250,
            "Native pilot must actually complete250 updates before full continuation")
    require(outcome["stop_reason"] == "declared_pilot_update_limit", "Do not silently restart a resource/retention-stop pilot")
    require(source["pilot_optimizer_update_limit"] == 250, "Pilot stop bound changed")
    for field in ("seed", "initialization", "init_from_nemo_model", "lora", "objective", "allocator_cap_gb",
                  "joint_fused_batch_size", "joint_preserve_memory", "source_pins", "baseline_development_summary",
                  "tokenizer_manifest", "native_extracted_dir", "native_extraction_proof"):
        require(source[field] == cfg[field], "Native continuation scientific identity changed:" + field)
    source_model = json.loads(json.dumps(source["model"]))
    target_model = json.loads(json.dumps(cfg["model"]))
    require(scoped(source_model["train_ds"].pop("manifest_filepath")) == scoped(cfg["pilot_training_manifest"]),
            "Source pilot training manifest differs from verified full-pass prefix")
    target_model["train_ds"].pop("manifest_filepath")
    require(source_model == target_model, "Native model/data/optimizer grouping changed")
    source_trainer = dict(source["trainer"])
    target_trainer = dict(cfg["trainer"])
    require(source_trainer.pop("val_check_interval") == 1000 and target_trainer.pop("val_check_interval") == 4000,
            "Only declared DEV cadence250-to1000 may change")
    require(source_trainer == target_trainer, "Native Trainer/precision/accumulation identity changed")
    gates = []
    for path in source_dir.glob("development_gate_step_*.json"):
        gate = json.loads(path.read_text(encoding="utf-8"))
        require(gate["step"] <= 250, "Source DEV evidence lies beyond source checkpoint")
        gates.append((gate["step"], path, gate))
    gates.sort(key=lambda item: item[0])
    require(gates and gates[-1][0] == 250, "Source checkpoint250 must have its completed six-language DEV gate")
    failures = 0
    for _, _, gate in gates:
        failures = 0 if gate["retention_gate_passed"] else failures + 1
    require(failures < cfg["retention_failures_before_stop"], "Existing consecutive retention-stop must remain honored")
    selected_step = outcome["selected_step"]
    selected = outcome["selected_checkpoint"]
    if selected_step is not None:
        selected = scoped(selected)
        require(selected.parent == source_dir and selected.is_file(), "Inherited DEV-selected native checkpoint missing")
        match = [gate for step, _, gate in gates if step == selected_step]
        require(len(match) == 1 and match[0]["qualified"] and match[0]["turkish_wer"] == outcome["selected_turkish_development_wer"],
                "Inherited native best checkpoint/gate mismatch")
    return {"source_step": 250, "source_run_id": outcome["run_id"], "source_run_dir": str(source_dir),
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
            "source_config_sha256": sha(config_path), "source_outcome_sha256": sha(outcome_path),
            "source_preflight_sha256": sha(preflight_path),
            "source_development_gates_sha256": {str(path): sha(path) for _, path, _ in gates},
            "selected_step": selected_step, "selected_checkpoint": str(selected) if selected_step is not None else None,
            "selected_checkpoint_sha256": sha(selected) if selected_step is not None else None,
            "selected_turkish_development_wer": outcome["selected_turkish_development_wer"],
            "consecutive_retention_failures": failures, "source_DEV250_reused_not_repeated": True,
            "allowed_runtime_transition": ["verified full native TRAIN stream with identical4000-row prefix",
                                           "DEV every1000 global optimizer updates", "new scoped run/output directory"],
            "scientific_recipe_or_optimizer_restart": False}

'''
replace('\ndef preflight(args):\n', '\n' + helpers + '\ndef preflight(args):\n')
text = text.replace('"declared_pilot_update_limit"', '"declared_full_pass_update_limit"')
text = text.replace('"stock_Lightning_optimizer_scheduler_and_loop_restore": True,', '"stock_Lightning_optimizer_scheduler_and_loop_restore": True,')
text = text.replace('Separate native Parakeet/NeMo/Lightning pilot;', 'Native B4 Parakeet/NeMo/Lightning full-pass continuation;')
ast.parse(text)
config = json.loads((out / "full_nemo_turkish_v4.json").read_text(encoding="utf-8"))
config["kind"] = "full_nemo_lightning_turkish_continuation_v3"
config["pilot_optimizer_update_limit"] = 9361
config["trainer"]["val_check_interval"] = 4000
config["model"]["train_ds"]["manifest_filepath"] = "work/nemo_direct_trial_v1/manifests/train_balanced_full_pass.jsonl"
config["pilot_training_manifest"] = "work/nemo_direct_trial_v1/manifests/train_balanced_250.jsonl"
config["continuation_data_proof"] = "outputs/parakeet_turkish/full_nemo_training_continuation_v1.json"
config["source_pilot_entry_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
config["continuation"] = {"source_native_completed_updates": 250, "data_prefix_rows": 4000,
                          "target_global_optimizer_updates": 9361, "physical_batch": 4, "accumulation": 4,
                          "effective_recordings_per_update": 16, "optimizer_and_scheduler_restarted": False,
                          "cosine_horizon_optimizer_updates": 9361, "warmup_restarted": False,
                          "development_global_update_interval": 1000, "completed_source_DEV_reused": True,
                          "native_pilot_completion_and_hardware_acceptance_pending": True}
with target.open("x", encoding="utf-8", newline="\n") as stream:
    stream.write(text)
with config_target.open("x", encoding="utf-8", newline="\n") as stream:
    json.dump(config, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"entry": str(target), "entry_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                  "config_sha256": hashlib.sha256(config_target.read_bytes()).hexdigest()}))
