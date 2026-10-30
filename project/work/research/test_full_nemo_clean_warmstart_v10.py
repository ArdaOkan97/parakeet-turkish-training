"""Small NEW clean-label warmstart guards; no Torch, NeMo or PCM imports."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
entry = OUT / "train_full_nemo_turkish_v10.py"
cfgpath = OUT / "full_nemo_turkish_v10.json"
source = entry.read_text(encoding="utf-8")
tree = ast.parse(source)
config = json.loads(cfgpath.read_text(encoding="utf-8"))
namespace = {"__name__": "native_clean_guard", "__file__": str(entry)}
exec(compile(tree, str(entry), "exec"), namespace)
checks = []


def check(name, value):
    if not value:
        raise AssertionError(name)
    checks.append(name)


def reject(name, operation):
    try:
        operation()
    except (ValueError, KeyError):
        checks.append(name)
    else:
        raise AssertionError(name)


old_tree = ast.parse((OUT / "train_full_nemo_turkish_v9.py").read_text(encoding="utf-8"))


def named_node(syntax, name):
    return next(node for node in ast.walk(syntax) if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)


for name in ("NativeTrialCallback", "NativeWarmstartSuffixSampler", "verify_warmstart_checkpoint_metadata", "restore_optimizer_scheduler"):
    check("v9 math/state/cadence AST unchanged:" + name,
          ast.dump(named_node(tree, name), include_attributes=False) == ast.dump(named_node(old_tree, name), include_attributes=False))
check("stdlib imports only", not any(name in sys.modules for name in ("torch", "nemo", "lightning", "sentencepiece")))
check("official native training and optimizer methods remain inherited", "model.training_step.__func__ is EncDecRNNTModel.training_step" in source and "model.configure_optimizers.__func__ is ModelPT.configure_optimizers" in source)
check("fresh loops and strict model warmstate", "trainer.fit(model)" in source and "trainer.fit(model, ckpt_path=" not in source and 'model.load_state_dict(checkpoint["state_dict"], strict=True)' in source)
check("B1x16 and declared label grouping transition", config["model"]["train_ds"]["batch_size"] == 1 and config["trainer"]["accumulate_grad_batches"] == 16 and config["training_label_change"]["exact_original_label_or_gradient_continuation_claimed"] is False)
check("preserved native100 source instead of v9 smoke101", config["warmstart"]["checkpoint_path"].endswith("pilot_06/periodic_checkpoints/step_000100.ckpt") and config["warmstart"]["source_completed_optimizer_updates"] == 100)
check("9361 original Cosine horizon retained", config["trainer"]["max_steps"] == config["model"]["optim"]["sched"]["max_steps"] == 9361 and config["model"]["optim"]["sched"]["warmup_steps"] == 25)
check("source data is independently bound", config["source_data_proof"] != config["data_proof"] and config["source_pilot_training_manifest"] != config["pilot_training_manifest"])
proof = json.loads((ROOT / config["data_proof"]).read_text(encoding="utf-8"))
label = namespace["validate_clean_training_transition"](proof, config)
check("actual1600 consumed recordings retained with honest cleaned labels", label["consumed_prefix_rows_compared"] == 1600 and label["source_audio_order_nonlabel_metadata_preserved"] is True and label["exact_original_label_or_gradient_continuation_claimed"] is False)
bad = copy.deepcopy(proof)
bad["clean_TRAIN"]["rows_dropped_or_cropped"] = 1
reject("dropped recording proof rejected", lambda: namespace["validate_clean_training_transition"](bad, config))
bad = copy.deepcopy(config)
bad["training_label_change"]["exact_original_label_or_gradient_continuation_claimed"] = True
reject("false exact-label continuation rejected", lambda: namespace["validate_clean_training_transition"](proof, bad))
args = namespace["parser"]().parse_args(["--run-id", "full_nemo_clean_CPU_preflight_only", "--validate-only"])
cfg, dev, run_dir, prepared = namespace["preflight"](args)
check("actual preflight binds native100 and clean full metadata", prepared["passed"] is True and prepared["Torch_imported"] is False and prepared["full_stream_audit"]["training_records"] == 149776 and prepared["source_native_warmstart"]["source_step"] == 100)
check("six fixed DEV50 preserve original proof hashes", len(dev) == 6 and all(len(rows) == 50 for rows in dev.values()) and label["DEV_TEST_changed"] is False)
check("old native100 Adam preserved but fit-loop RNG equivalence unclaimed", prepared["source_native_warmstart"]["weights_Adam_Cosine_preserved"] is True and prepared["source_native_warmstart"]["old_RNG_restored_or_bitwise_gradients_claimed"] is False)
check("no heavy import or new run directory created", not run_dir.exists() and not any(name in sys.modules for name in ("torch", "nemo", "lightning", "sentencepiece")))
record = {"schema": 1, "kind": "native_B1_clean_TRAIN_native100_warmstart_stdlib_readiness_v10", "passed": True,
          "checks": checks, "check_count": len(checks), "entry_sha256": namespace["sha"](entry), "config_sha256": namespace["sha"](cfgpath),
          "TRAIN_label_transition": label, "source_native_warmstart": prepared["source_native_warmstart"],
          "full_stream_audit": prepared["full_stream_audit"], "Torch_NeMo_imported": False, "CUDA_initialized": False,
          "native_actual_restore_and_GPU_memory_pending": True,
          "frozen_v9_entry_sha256": namespace["sha"](OUT / "train_full_nemo_turkish_v9.py"),
          "frozen_v9_config_sha256": namespace["sha"](OUT / "full_nemo_turkish_v9.json")}
path = OUT / "full_nemo_clean_warmstart_readiness_v10.json"
with path.open("x", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2)
    stream.write("\n")
print(json.dumps({"passed": True, "checks": len(checks), "entry_sha256": record["entry_sha256"], "config_sha256": record["config_sha256"], "readiness": str(path)}))
