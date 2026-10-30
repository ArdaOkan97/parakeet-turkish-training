"""NEW generalized boundary entry; source100/v10 originals remain immutable."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def replace(source, old, new):
    if source.count(old) != 1:
        raise AssertionError(old[:100])
    return source.replace(old, new, 1)


def main():
    target, config_path = OUT / "train_full_nemo_turkish_v11.py", OUT / "full_nemo_turkish_v11.json"
    if target.exists() or config_path.exists():
        raise FileExistsError("New version paths only")
    source_path = OUT / "train_full_nemo_turkish_v10.py"
    if sha(source_path) != "876772adb6377161f96e6d296cd2a67abc8fc0ed8513b2a69ffcfb46cab2bed4":
        raise ValueError("Frozen v10 changed")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_replacements = {
        "validate_source_warmstart": '''def validate_source_warmstart(checkpoint_value, cfg):
    from native_completed_boundary_warmstart_v1 import validate_boundary_evidence
    boundary = validate_boundary_evidence(cfg["warmstart"]["source_run_dir"], 350, require_improved=not cfg["benchmark_only"])
    if checkpoint_value is not None:
        require(scoped(checkpoint_value) == scoped(boundary["checkpoint"]["path"]), "Only the pinned completed DEV350 checkpoint may seed this arm")
    source = json.loads(scoped(boundary["source_run_dir"]).joinpath("resolved_trial_config.json").read_text(encoding="utf-8"))
    for field in ("seed", "init_from_nemo_model", "lora", "allocator_cap_gb", "joint_fused_batch_size", "joint_preserve_memory", "baseline_development_summary", "tokenizer_manifest", "native_extracted_dir", "native_extraction_proof", "data_proof", "training_label_change"):
        require(source[field] == cfg[field], "Native boundary arm changed undeclared source field:" + field)
    source_model, target_model = json.loads(json.dumps(source["model"])), json.loads(json.dumps(cfg["model"]))
    require(source_model["train_ds"].pop("batch_size") == 1 and target_model["train_ds"].pop("batch_size") in (1, 2, 4, 8), "Declared physical grouping required")
    require(source_model == target_model, "Boundary model/data/optimizer configuration changed")
    source_trainer, target_trainer = dict(source["trainer"]), dict(cfg["trainer"])
    require(source_trainer.pop("accumulate_grad_batches") == 16 and target_trainer.pop("accumulate_grad_batches") == 16 // cfg["model"]["train_ds"]["batch_size"], "Keep effective16 recordings")
    source_trainer.pop("val_check_interval"); target_trainer.pop("val_check_interval")
    target_trainer.pop("limit_val_batches", None)
    target_trainer["enable_checkpointing"] = True
    require(source_trainer == target_trainer, "Undeclared native Trainer setting changed")
    for path, digest in source["source_pins"].items():
        require(cfg["source_pins"][path] == digest, "Frozen source runtime identity changed")
    return {**boundary, "source_step": 350, "checkpoint_path": boundary["checkpoint"]["path"], "checkpoint_sha256": boundary["checkpoint"]["sha256"],
            "consumed_ordered_training_records": 5600, "weights_Adam_Cosine_preserved": True, "source_fit_loops_restored": False,
            "old_RNG_restored_or_bitwise_gradients_claimed": False, "physical_mean_volume_grouping_change": cfg["model"]["train_ds"]["batch_size"] != 1}
''',
        "verify_warmstart_checkpoint_metadata": '''def verify_warmstart_checkpoint_metadata(checkpoint, boundary):
    from native_completed_boundary_warmstart_v1 import verify_completed_checkpoint_state
    return verify_completed_checkpoint_state(checkpoint, boundary)
''',
        "restore_optimizer_scheduler": '''def restore_optimizer_scheduler(optimizer, scheduler, source_optimizer, source_scheduler):
    from native_completed_boundary_warmstart_v1 import restore_official_optimizer_scheduler
    return restore_official_optimizer_scheduler(optimizer, scheduler, source_optimizer, source_scheduler,
                                                {"source_cumulative_optimizer_step": 350})
'''}
    lines = source.splitlines(keepends=True)
    for node in sorted((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in function_replacements), key=lambda node: node.lineno, reverse=True):
        lines[node.lineno-1:node.end_lineno] = [function_replacements[node.name] + "\n"]
    source = "".join(lines)
    source = source.replace("clean_warmstart_v10", "boundary_warmstart_v11").replace("full_nemo_turkish_v10.json", "full_nemo_turkish_v11.json")
    source = replace(source, '    require(cfg["model"]["train_ds"]["batch_size"] == 1 and cfg["trainer"]["accumulate_grad_batches"] == 16,\n            "This arm declares B1 x accumulation16, effective16; mean_volume grouping changes")',
                     '    cfg["model"]["train_ds"]["batch_size"] = args.physical_batch\n    cfg["trainer"]["accumulate_grad_batches"] = 16 // args.physical_batch\n    cfg["trainer"]["val_check_interval"] = 250 * cfg["trainer"]["accumulate_grad_batches"]\n    cfg["benchmark_only"] = args.benchmark_only\n    cfg["new_row_delta_multiplier"] = args.new_row_delta_multiplier\n    if args.benchmark_only:\n        cfg["trainer"]["limit_val_batches"] = 0\n        cfg["trainer"]["enable_checkpointing"] = False')
    source = replace(source, '    require(cfg["initialization"] == "native06_periodic100_strict_model_Adam_Cosine_warmstart_no_HF_transfer", "Unreviewed initialization")',
                     '    require(cfg["initialization"] == "native_v10_completedDEV350_model_Adam_Cosine_warmstart", "Unreviewed initialization")')
    source = replace(source, '    require(isinstance(steps, int) and 1 <= steps <= 9261, "New local updates are bounded to the remaining original9361 horizon")',
                     '    require(isinstance(steps, int) and 1 <= steps <= 9011, "New local updates are bounded to the remaining9361 horizon")')
    source = replace(source, '    source_checkpoint = args.source_checkpoint or scoped(cfg["warmstart"]["checkpoint_path"])', '    source_checkpoint = args.source_checkpoint')
    source = replace(source, '    inherited = validate_source_warmstart(source_checkpoint, cfg)',
                     '    inherited = validate_source_warmstart(source_checkpoint, cfg)\n    cfg["runtime_extensions"] = validate_extensions(args, inherited, cfg)\n    if args.benchmark_only:\n        indices = list(range(5600, 5600 + steps * 16))\n        if args.benchmark_indices_json:\n            value = json.loads(scoped(args.benchmark_indices_json).read_text(encoding="utf-8"))\n            require(value["kind"] == "native_full_TRAIN_benchmark_indices_v1" and value["benchmark_only"] is True and value["full_manifest_sha256"] == stream_audit["full_manifest_sha256"], "Benchmark indices must bind the clean full TRAIN stream")\n            indices = value["indices"]\n        require(len(indices) == steps * 16 and all(type(i) is int and 0 <= i < train_count for i in indices), "Benchmark indices must contain exactly the declared TRAIN updates")\n        cfg["benchmark_indices"] = indices\n        cfg["benchmark_indices_sha256"] = hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest()\n    else:\n        require(args.benchmark_indices_json is None, "Full quality arm must retain the normal5600 cursor")')
    source = replace(source, '            self.source_step = 100', '            self.source_step = 350\n            self.row_hook = None')
    source = replace(source, '            write_new(run_dir / "optimizer_warmstart_restoration.json", self.optimizer_restore_evidence)',
                     '            write_new(run_dir / "optimizer_warmstart_restoration.json", self.optimizer_restore_evidence)\n            if cfg["new_row_delta_multiplier"] != 1:\n                from native_new_row_adam_delta_v1 import install_native_new_row_adam_delta\n                self.row_hook = install_native_new_row_adam_delta(model, trainer.optimizers[0], cfg["new_row_delta_multiplier"], torch_module=torch)\n                write_new(run_dir / "new_row_delta_hook.json", self.row_hook["identity"])')
    source = replace(source, '                require(len(batch[1]) == 1, "Declared target physical B1 grouping required")',
                     '                require(len(batch[1]) == cfg["model"]["train_ds"]["batch_size"], "Declared target physical grouping required")')
    source = replace(source, '    periodic = ModelCheckpoint(dirpath=str(run_dir / "periodic_checkpoints"),', '    periodic = None if args.benchmark_only else ModelCheckpoint(dirpath=str(run_dir / "periodic_checkpoints"),')
    source = replace(source, 'callbacks=[callback, telemetry, periodic]', 'callbacks=[callback, telemetry] + ([periodic] if periodic is not None else [])')
    source = replace(source, '    error = None\n    try:', '    error = None\n    stock_hook = None\n    try:')
    source = replace(source, '        source_metadata = verify_warmstart_checkpoint_metadata(checkpoint)', '        source_metadata = verify_warmstart_checkpoint_metadata(checkpoint, validation["source_native_warmstart"])')
    source = replace(source, '        consumed = 1600', '        consumed = 5600')
    source = replace(source, '                return iter(range(consumed, len(self.data_source)))', '                return iter(cfg["benchmark_indices"]) if args.benchmark_only else iter(range(consumed, len(self.data_source)))')
    source = replace(source, '                return len(self.data_source) - consumed', '                return len(cfg["benchmark_indices"]) if args.benchmark_only else len(self.data_source) - consumed')
    source = replace(source, '        require(len(model._train_dl) == 148176 and len(model._train_dl) // 16 == 9261,\n                "Suffix must visit the first unseen record1600 through full-pass end exactly once")',
                     '        expected_updates = cfg["local_optimizer_update_limit"] if args.benchmark_only else 9011\n        require(len(model._train_dl) == expected_updates * cfg["trainer"]["accumulate_grad_batches"], "Declared native suffix/benchmark loader length changed")')
    source = replace(source, '                  "source_step": 100, "consumed_training_records": consumed, "remaining_training_records": len(dataset)-consumed,', '                  "source_step": 350, "consumed_training_records": consumed, "remaining_training_records": len(dataset)-consumed,')
    source = source.replace('suffix sequential sampler starts1600', 'suffix sequential sampler starts5600').replace('100 + local Lightning global_step', '350 + local Lightning global_step')
    source = replace(source, '        provenance["TRAIN_label_transition"] = validation["TRAIN_label_transition"]',
                     '        provenance["TRAIN_label_transition"] = validation["TRAIN_label_transition"]\n        provenance["runtime_extensions"] = cfg["runtime_extensions"]\n        provenance["benchmark_only"] = args.benchmark_only\n        provenance["benchmark_full_pass_progress_or_quality_winner_claimed"] = False\n        if args.stock_loss_checkpoint_proof:\n            from stock_nemo_loss_checkpoint_v1 import install_stock_nemo_loss_checkpointing\n            require(model.joint.loss is model.loss, "Official fused joint must use the same stock loss object")\n            stock_hook = install_stock_nemo_loss_checkpointing(model, torch_module=torch)\n            provenance["stock_loss_checkpoint_hook"] = stock_hook["identity"]')
    source = replace(source, '        trainer.validate(model, verbose=False)  # Actual trained100 DEV, not native-base baseline.', '        if not args.benchmark_only:\n            trainer.validate(model, verbose=False)  # Actual trained350 DEV before optimizer restoration.')
    source = replace(source, '        if callback.last_evaluated_step != callback.source_step + int(trainer.global_step):', '        if not args.benchmark_only and callback.last_evaluated_step != callback.source_step + int(trainer.global_step):')
    source = replace(source, '        trainer.save_checkpoint(str(run_dir / f"last_cumulative_step_{callback.source_step + int(trainer.global_step)}.ckpt"))', '        if not args.benchmark_only:\n            trainer.save_checkpoint(str(run_dir / f"last_cumulative_step_{callback.source_step + int(trainer.global_step)}.ckpt"))')
    source = replace(source, '        callback.done.set()', '        if callback.row_hook is not None:\n            write_new(run_dir / "new_row_delta_hook_final.json", callback.row_hook["identity"])\n            callback.row_hook["remove"]()\n        if stock_hook is not None:\n            write_new(run_dir / "stock_loss_checkpoint_hook_final.json", stock_hook["identity"])\n            stock_hook["remove"]()\n        callback.done.set()')
    source = replace(source, '"actual_completed_optimizer_updates": 100 + int(trainer.global_step)', '"actual_completed_optimizer_updates": 350 + int(trainer.global_step)')
    source = replace(source, '"source_optimizer_updates": 100, "error": error,', '"source_optimizer_updates": 350, "error": error,\n                  "benchmark_only": args.benchmark_only, "benchmark_full_pass_progress_or_quality_winner_claimed": False,')
    parser_marker = '    value.add_argument("--stop-file", type=Path)'
    source = replace(source, parser_marker, '''    value.add_argument("--physical-batch", type=int, choices=(1, 2, 4, 8), default=1)
    value.add_argument("--benchmark-only", action="store_true")
    value.add_argument("--benchmark-indices-json", type=Path)
    value.add_argument("--stock-loss-checkpoint-proof", type=Path)
    value.add_argument("--stock-loss-checkpoint-proof-sha256")
    value.add_argument("--new-row-delta-multiplier", type=float, choices=(1.0, 6.0), default=1.0)
    value.add_argument("--new-row-step-proof", type=Path)
    value.add_argument("--new-row-step-proof-sha256")
''' + parser_marker)
    extensions = '''
def validate_extensions(args, boundary, cfg):
    evidence = {"physical_batch": args.physical_batch, "accumulation": 16 // args.physical_batch,
                "physical_mean_volume_grouping_change": args.physical_batch != 1,
                "gradient_equivalence_with_B1_claimed": False, "benchmark_only": args.benchmark_only}
    require(args.physical_batch == 1 or args.benchmark_only, "Larger physical batches are diagnostic until whole-update resource and quality gates pass")
    if args.stock_loss_checkpoint_proof:
        path = scoped(args.stock_loss_checkpoint_proof)
        require(args.stock_loss_checkpoint_proof_sha256 and sha(path) == args.stock_loss_checkpoint_proof_sha256, "Pin actual stock-loss GPU oracle bytes")
        proof = json.loads(path.read_text(encoding="utf-8"))
        require(proof.get("kind") == "actual_TRAIN_stock_NeMo_loss_checkpoint_oracle_v1" and proof.get("passed") is True
                and proof.get("native_kernel_executed") is True and proof.get("cases_completed") == 8
                and proof.get("ambient_rng_restored_finally") is True and all(case["passed"] for case in proof["cases"]), "Actual stock-loss GPU oracle must pass before installation")
        require(sha(HERE / "stock_nemo_loss_checkpoint_v1.py") == "ed34b4cc97437a4d8694bc94a3522c565cc329ea40f4aac80112fd1a9abcab57", "Stock loss hook source changed")
        evidence["stock_loss_checkpoint_GPU_proof"] = {"path": str(path), "sha256": sha(path)}
    if args.new_row_delta_multiplier != 1:
        require(args.new_row_step_proof is not None and args.new_row_step_proof_sha256, "Experimental row6 arm requires an actual-step proof")
        path = scoped(args.new_row_step_proof)
        proof = json.loads(path.read_text(encoding="utf-8"))
        require(sha(path) == args.new_row_step_proof_sha256 and proof.get("kind") == "native_new_row_AdamW_delta_actual_GPU_oracle_v1"
                and proof.get("status") == "complete" and proof.get("passed") is True
                and proof.get("actual_GPU_optimizer_steps_executed") is True and proof.get("actual_Lightning_bf16_optimizer_steps_executed") is True
                and proof.get("helper_sha256") == sha(HERE / "native_new_row_adam_delta_v1.py"), "Actual ordinary and Lightning row-delta steps must pass")
        evidence["new_row_delta_GPU_proof"] = {"path": str(path), "sha256": sha(path), "multiplier": 6}
    return evidence

'''
    source = replace(source, '\ndef preflight(args):', extensions + '\ndef preflight(args):')
    cfg = json.loads((OUT / "full_nemo_turkish_v10.json").read_text(encoding="utf-8"))
    cfg["kind"] = "full_nemo_lightning_turkish_boundary_warmstart_v11"
    cfg["initialization"] = "native_v10_completedDEV350_model_Adam_Cosine_warmstart"
    cfg["local_optimizer_update_limit"] = 9011
    cfg["warmstart"] = {"source_run_dir": "outputs/parakeet_turkish/runs/full_nemo_turkish_20261011_clean_warmstart_08", "source_completed_optimizer_updates": 350,
                        "source_checkpoint_local_step": 250, "source_consumed_recordings": 5600, "target_local_optimizer_update_limit": 9011,
                        "source_fit_loops_restored": False, "source_RNG_restored": False, "scheduler_horizon": 9361}
    for name in ("native_completed_boundary_warmstart_v1.py", "native_new_row_adam_delta_v1.py", "stock_nemo_loss_checkpoint_v1.py"):
        cfg["source_pins"]["outputs/parakeet_turkish/" + name] = sha(OUT / name)
    cfg["periodic_checkpoint_policy"]["cumulative_step_formula"] = "350 + local Lightning global_step"
    cfg["future_scope"] = {"larger_batches_benchmark_only_until_measured": True, "full_quality_arm_normal_cursor": 5600,
                           "benchmark_weights_never_advance_full_progress": True, "hook_and_row6_require_actual_GPU_proofs": True}
    ast.parse(source)
    target.write_text(source, encoding="utf-8", newline="\n")
    config_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"entry_sha256": sha(target), "config_sha256": sha(config_path), "actual_boundary350_preflight_pending": True}))


if __name__ == "__main__":
    main()
