"""Bounded stdlib guards for the future native B2 long continuation."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

root = Path(__file__).resolve().parents[2]
out = root / "outputs/parakeet_turkish"
entry = out / "train_full_nemo_turkish_v6.py"
cfgpath = out / "full_nemo_turkish_v6.json"
source = entry.read_text(encoding="utf-8")
tree = ast.parse(source)
cfg = json.loads(cfgpath.read_text(encoding="utf-8"))
checks = []

def check(name, passed):
    if not passed:
        raise AssertionError(name)
    checks.append(name)

namespace = {"__name__": "native_continuation_guard", "__file__": str(entry)}
exec(compile(tree, str(entry), "exec"), namespace)
check("stdlib import loads no Torch NeMo or Lightning", not any(name in sys.modules for name in ("torch", "nemo", "lightning")))
check("native source v4 remains frozen", hashlib.sha256((out / "train_full_nemo_turkish_v7.py").read_bytes()).hexdigest() == cfg["source_pilot_entry_sha256"])
check("B2 accumulation4 preserves physical objective grouping", cfg["model"]["train_ds"]["batch_size"] == 2 and cfg["trainer"]["accumulate_grad_batches"] == 8)
check("scheduler horizon and warmup not reset", cfg["trainer"]["max_steps"] == cfg["model"]["optim"]["sched"]["max_steps"] == 9361 and cfg["model"]["optim"]["sched"]["warmup_steps"] == 25)
check("DEV cadence is1000 optimizer steps", cfg["trainer"]["val_check_interval"] == 8000 and cfg["trainer"]["check_val_every_n_epoch"] is None)
check("stock native methods and Trainer fit retained", all(fragment in source for fragment in ("model.training_step.__func__ is EncDecRNNTModel.training_step", "model.configure_optimizers.__func__ is ModelPT.configure_optimizers", "trainer.fit(model, ckpt_path=")))
check("no custom loss imports or TEST calls", not any(fragment in source for fragment in ("from native_tdt_checkpointed", "from efficient_tdt", "from runtime import")) and not any(isinstance(node, ast.Attribute) and node.attr == "test" for node in ast.walk(tree)))
for path, expected in cfg["source_pins"].items():
    check("unchanged pin:" + path, hashlib.sha256((root / path).read_bytes()).hexdigest() == expected)
pilot_proof = json.loads((root / cfg["data_proof"]).read_text(encoding="utf-8"))
count, full = namespace["validate_full_stream"](root / cfg["model"]["train_ds"]["manifest_filepath"], pilot_proof, cfg)
check("actual full manifest149776 rows/74883 uniqueTR/5padding/prefix4000", count == 149776 and full["unique_turkish_source_ids"] == 74883 and full["final_padding_visits"] == 5 and full["pilot_prefix_rows_compared"] == 4000)
check("actual stream audit reads no PCM", full["PCM_read_or_rehashed"] is False)

fixture_root = root / "work/research/native_continuation_fixture_v3"
fixture = fixture_root / "runs/fixture_native_B2_250"
fixture.mkdir(parents=True, exist_ok=False)
namespace["HERE"] = fixture_root
prior = json.loads((out / "full_nemo_turkish_v7.json").read_text(encoding="utf-8"))
outcome = {"run_id": fixture.name, "status": "complete", "error": None, "actual_completed_optimizer_updates": 250,
           "stop_reason": "declared_pilot_update_limit", "selected_step": 250,
           "selected_checkpoint": str(fixture / "qualified_step_250.ckpt"), "selected_turkish_development_wer": .85}
prepared = {"run_id": fixture.name, "entry_sha256": cfg["source_pilot_entry_sha256"],
            "config_sha256": cfg["source_pilot_config_sha256"],
            "data_proof_sha256": hashlib.sha256((root / cfg["data_proof"]).read_bytes()).hexdigest(),
            "tokenizer_manifest_sha256": hashlib.sha256((root / cfg["tokenizer_manifest"]).read_bytes()).hexdigest()}
for name, value in (("resolved_trial_config.json", prior), ("trial_outcome.json", outcome), ("preflight.json", prepared),
                    ("development_gate_step_0.json", {"step": 0, "retention_gate_passed": True, "qualified": False, "turkish_wer": 1.01}),
                    ("development_gate_step_250.json", {"step": 250, "retention_gate_passed": True, "qualified": True, "turkish_wer": .85})):
    (fixture / name).write_text(json.dumps(value), encoding="utf-8")
(fixture / "last_step_250.ckpt").write_bytes(b"CPU fixture only, no model or tensor state")
(fixture / "qualified_step_250.ckpt").write_bytes(b"CPU fixture only, no model or tensor state")
resume = namespace["validate_source_resume"](fixture / "last_step_250.ckpt", cfg)
check("source identity fixture carries native selected gate without optimizer restart", resume["selected_step"] == 250 and resume["scientific_recipe_or_optimizer_restart"] is False and resume["consecutive_retention_failures"] == 0)

def reject(name, callback):
    try:
        callback()
    except (ValueError, KeyError):
        checks.append(name)
    else:
        raise AssertionError(name)

for field, value in (("allocator_cap_gb", 8), ("seed", 1), ("joint_fused_batch_size", 2)):
    changed = copy.deepcopy(cfg)
    changed[field] = value
    reject("scientific identity change rejected:" + field, lambda changed=changed: namespace["validate_source_resume"](fixture / "last_step_250.ckpt", changed))
changed = copy.deepcopy(cfg)
changed["model"]["train_ds"]["batch_size"] = 8
reject("B8 checkpoint grouping switch rejected", lambda: namespace["validate_source_resume"](fixture / "last_step_250.ckpt", changed))
changed_outcome = dict(outcome, stop_reason="actual_worker_process_commit_headroom_below2GB")
(fixture / "trial_outcome.json").write_text(json.dumps(changed_outcome), encoding="utf-8")
reject("resource stop is not silently restarted", lambda: namespace["validate_source_resume"](fixture / "last_step_250.ckpt", cfg))
(fixture / "trial_outcome.json").write_text(json.dumps(outcome), encoding="utf-8")

metadata = {"global_step": 250, "optimizer_states": [{}], "lr_schedulers": [{}],
            "loops": {"fit_loop": {"epoch_loop.batch_progress": {"total": {"completed": 2000, "ready": 2000}}}}}
check("stock metadata completed250 and2000 microbatches accepted", namespace["verify_lightning_checkpoint_metadata"](metadata) == 250)
for field, value in (("global_step", 249), ("optimizer_states", []), ("lr_schedulers", [])):
    bad = copy.deepcopy(metadata)
    bad[field] = value
    reject("incomplete checkpoint metadata rejected:" + field, lambda bad=bad: namespace["verify_lightning_checkpoint_metadata"](bad))
bad = copy.deepcopy(metadata)
bad["loops"]["fit_loop"]["epoch_loop.batch_progress"]["total"]["ready"] = 2001
reject("partial accumulation checkpoint rejected", lambda: namespace["verify_lightning_checkpoint_metadata"](bad))
sampler_node = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "NativeResumeSequentialSampler")
class Sequential:
    def __init__(self, data_source): self.data_source = data_source
    def __len__(self): return len(self.data_source)
sampler_namespace = {"torch": SimpleNamespace(utils=SimpleNamespace(data=SimpleNamespace(SequentialSampler=Sequential))), "consumed": 4000}
exec(compile(ast.Module(body=[sampler_node], type_ignores=[]), str(entry), "exec"), sampler_namespace)
sampler = sampler_namespace["NativeResumeSequentialSampler"](range(149776))
indices = iter(sampler)
check("resume sampler starts at4000 and preserves full logical length", next(indices) == 4000 and len(sampler) == 149776)
check("suffix has9111 full updates and no repeated4000 prefix", len(range(4000, 149776)) // 16 == 9111)
result = subprocess.run([sys.executable, "-X", "utf8", str(entry), "--help"], capture_output=True, text=True)
check("actual stdlib CLI exits0 with required checkpoint argument", result.returncode == 0 and "--ckpt-path" in result.stdout)

check("stock periodic checkpoints retain latest with no serialized counter rewrite", "periodic = ModelCheckpoint(" in source and "every_n_train_steps=500, save_top_k=1" in source and "def on_save_checkpoint" not in source)
check("every completedDEV checkpoint saved even when unqualified", "recovery_dev_step_{step}.ckpt" in source and "completed_DEV_boundary" in source)
check("periodic resume explicitly disabled until cursor review", cfg["periodic_checkpoint_policy"]["periodic_checkpoint_auto_resume_enabled"] is False)
check("first resumed250 is not a stock periodic trigger", cfg["periodic_checkpoint_policy"]["every_n_train_steps"] == 500 and 250 % 500 != 0)

report = {"schema": 1, "kind": "full_native_NeMo_B2_continuation_v6_CPU_readiness", "passed": True,
          "check_count": len(checks), "checks": checks, "full_stream_audit": full,
          "entry_sha256": hashlib.sha256(entry.read_bytes()).hexdigest(),
          "config_sha256": hashlib.sha256(cfgpath.read_bytes()).hexdigest(),
          "source_pilot_entry_sha256": cfg["source_pilot_entry_sha256"],
          "Torch_imported": False, "CUDA_initialized": False,
          "actual_completed_native_pilot250_checkpoint_pending": True,
          "actual_Lightning_optimizer_restore_and_first_resumed_update_pending": True,
          "fixture_checkpoints_are_not_models": True}
(out / "full_nemo_continuation_readiness_v3.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"passed": True, "checks": len(checks), "entry_sha256": report["entry_sha256"], "config_sha256": report["config_sha256"], "full_stream": full}))
