"""Small stdlib source/dispatch/state-restore guards; no heavy imports."""
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
entry = out / "train_full_nemo_turkish_v9.py"
cfgpath = out / "full_nemo_turkish_v9.json"
src = entry.read_text(encoding="utf-8")
tree = ast.parse(src)
cfg = json.loads(cfgpath.read_text(encoding="utf-8"))
checks = []

def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)

def reject(name, callback):
    try:
        callback()
    except (ValueError, KeyError):
        checks.append(name)
    else:
        raise AssertionError(name)

ns = {"__name__": "warmstart_guard", "__file__": str(entry)}
exec(compile(tree, str(entry), "exec"), ns)
check("stdlib-only imports", not any(module in sys.modules for module in ("torch", "nemo", "lightning")))
check("official training methods remain inherited", "model.training_step.__func__ is EncDecRNNTModel.training_step" in src and "model.configure_optimizers.__func__ is ModelPT.configure_optimizers" in src)
check("no old fit-loop restore", "trainer.fit(model)" in src and "trainer.fit(model, ckpt_path=" not in src and "def on_save_checkpoint" not in src)
check("strict native checkpoint weights", 'model.load_state_dict(checkpoint["state_dict"], strict=True)' in src)
check("B1x16 cap9 and unchanged9361 horizon", cfg["model"]["train_ds"]["batch_size"] == 1 and cfg["trainer"]["accumulate_grad_batches"] == 16 and cfg["allocator_cap_gb"] == 9 and cfg["trainer"]["max_steps"] == cfg["model"]["optim"]["sched"]["max_steps"] == 9361)
check("remaining update count9261", cfg["local_optimizer_update_limit"] == 9261 and cfg["warmstart"]["target_cumulative_optimizer_updates"] == 9361)
check("initial250 then1000 local cadence", cfg["trainer"]["val_check_interval"] == 4000 and 'trainer.val_check_batch = 1000 * cfg["trainer"]["accumulate_grad_batches"]' in src)
check("stock periodic ModelCheckpoint with disabled automatic cursor resume", "periodic = ModelCheckpoint(" in src and cfg["periodic_checkpoint_policy"]["periodic_checkpoint_auto_resume_enabled"] is False)
for name, digest in cfg["source_pins"].items():
    check("frozen source:" + name, hashlib.sha256((root / name).read_bytes()).hexdigest() == digest)
source_proof = ns["validate_source_warmstart"](root / cfg["warmstart"]["checkpoint_path"], cfg)
check("actual source100 artifact/config/hash accepted without importing Torch", source_proof["source_step"] == 100 and source_proof["consumed_ordered_training_records"] == 1600 and source_proof["source_fit_loops_restored"] is False)
badcfg = copy.deepcopy(cfg)
badcfg["allocator_cap_gb"] = 8
reject("undeclared cap change rejected", lambda: ns["validate_source_warmstart"](root / cfg["warmstart"]["checkpoint_path"], badcfg))
badcfg = copy.deepcopy(cfg)
badcfg["model"]["optim"]["betas"][0] = .8
reject("optimizer recipe change rejected", lambda: ns["validate_source_warmstart"](root / cfg["warmstart"]["checkpoint_path"], badcfg))
proof = json.loads((root / cfg["data_proof"]).read_text(encoding="utf-8"))
count, stream = ns["validate_full_stream"](root / cfg["model"]["train_ds"]["manifest_filepath"], proof, cfg)
check("actual full metadata149776 prefix4000 and untouched PCM", count == 149776 and stream["pilot_prefix_rows_compared"] == 4000 and stream["PCM_read_or_rehashed"] is False)

lrs = [.0002999530279451991, .00004999283477130156, .00004999283477130156]
optimizer = {"param_groups": [{"params": list(range(n)), "lr": lr} for n, lr in zip([828, 9, 6], lrs)], "state": {}}
scheduler = {"max_steps": 9361, "warmup_steps": 25, "last_epoch": 100, "_step_count": 101, "_last_lr": lrs}
checkpoint = {"global_step": 100, "state_dict": {str(i): None for i in range(917)},
    "optimizer_states": [optimizer], "lr_schedulers": [scheduler],
    "loops": {"fit_loop": {"epoch_loop.batch_progress": {scope: {"ready": 800, "started": 800, "processed": 800, "completed": 799} for scope in ("current", "total")}}}}
check("expected stock periodic100 metadata lag is handled without restoring it", ns["verify_warmstart_checkpoint_metadata"](checkpoint)["old_fit_loop_batch_counters_used"] is False)
bad = copy.deepcopy(checkpoint)
bad["lr_schedulers"][0]["last_epoch"] = 0
reject("scheduler warmup restart rejected", lambda: ns["verify_warmstart_checkpoint_metadata"](bad))

class Optimizer:
    def __init__(self):
        self.param_groups = [{"params": list(range(n)), "lr": 0} for n in [828, 9, 6]]
        self.restores = 0
    def load_state_dict(self, state):
        self.restores += 1
        self.param_groups = copy.deepcopy(state["param_groups"])
    def step(self): raise AssertionError("No optimizer step during restore")

class Scheduler:
    max_steps, warmup_steps, last_epoch = 9361, 25, 0
    def __init__(self): self.restores = 0
    def load_state_dict(self, state):
        self.restores += 1
        self.__dict__.update(copy.deepcopy(state))
    def step(self): raise AssertionError("No scheduler step during restore")

opt, sched = Optimizer(), Scheduler()
restored = ns["restore_optimizer_scheduler"](opt, sched, optimizer, scheduler)
check("load Adam then Cosine once without performing step", opt.restores == sched.restores == 1 and sched.last_epoch == 100 and restored["optimizer_group_learning_rates"] == lrs)

runtime = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "runtime")
callback_node = next(node for node in runtime.body if isinstance(node, ast.ClassDef) and node.name == "NativeTrialCallback")
saved, writes = [], []
trainer = SimpleNamespace(global_step=0, sanity_checking=False, should_stop=False, val_check_batch=4000,
    save_checkpoint=lambda value: saved.append(value), optimizers=[Optimizer()], lr_scheduler_configs=[SimpleNamespace(scheduler=Scheduler())])
dev = {lang: [{"text": "TRAIN-ONLY CPU fixture", "source_id": "fixture"}] for lang in ns["LANGUAGES"]}
callback_ns = dict(ns, Callback=object, ResourceGuard=lambda: object(), cfg=cfg, args=SimpleNamespace(stop_file=None),
    validation={"source_native_warmstart": source_proof}, dev=dev, baseline_wer={lang: 1.0 for lang in ns["LANGUAGES"]},
    run_dir=out / "runs/fixture_not_created", score_transcriptions=lambda rows, language: {"wer": .8},
    write_new=lambda path, value: writes.append((str(path), value)), sha=lambda path: "mock" )
exec(compile(ast.Module(body=[callback_node], type_ignores=[]), str(entry), "exec"), callback_ns)
callback = callback_ns["NativeTrialCallback"]()
callback.rows = {lang: ["fixture"] for lang in ns["LANGUAGES"]}
callback.on_validation_epoch_end(trainer, None)
check("initial trained100 DEV never saves empty optimizer checkpoint", not saved and callback.selected_step == 100 and callback.selected_path == Path(source_proof["checkpoint_path"]))
record = next(value for name, value in writes if name.endswith("recovery_checkpoint_step_100.json"))
check("initial recovery reference explicitly uses original full source checkpoint", record["phase"] == "source_checkpoint_pre_optimizer_restore" and record["fresh_full_checkpoint_saved_in_this_run"] is False)
callback.source_optimizer_state, callback.source_scheduler_state = optimizer, scheduler
callback.on_train_start(trainer, None)
check("official on_train_start restores and releases source optimizer payload", callback.source_optimizer_state is None and callback.source_scheduler_state is None and trainer.optimizers[0].restores == 1)
trainer.global_step = 250
callback.rows = {lang: ["fixture"] for lang in ns["LANGUAGES"]}
callback.on_validation_epoch_end(trainer, None)
check("local250 is cumulative350 and switches later DEV cadence1000", callback.last_evaluated_step == 350 and trainer.val_check_batch == 16000)
check("post-training DEV saves complete official checkpoint", len(saved) == 1 and saved[0].endswith("recovery_dev_step_350.ckpt"))
sampler_node = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "NativeWarmstartSuffixSampler")
class Sequential:
    def __init__(self, data_source): self.data_source = data_source
    def __len__(self): return len(self.data_source)
sampler_ns = {"consumed": 1600, "torch": SimpleNamespace(utils=SimpleNamespace(data=SimpleNamespace(SequentialSampler=Sequential)))}
exec(compile(ast.Module(body=[sampler_node], type_ignores=[]), str(entry), "exec"), sampler_ns)
sampler = sampler_ns["NativeWarmstartSuffixSampler"](range(149776))
check("first unseen row1600 and9261 remaining effective16 updates", next(iter(sampler)) == 1600 and len(sampler) == 148176 and len(sampler)//16 == 9261)
cli = subprocess.run([sys.executable, "-X", "utf8", str(entry), "--help"], capture_output=True, text=True)
check("actual stdlib CLI help works", cli.returncode == 0 and "--source-checkpoint" in cli.stdout)
args = ns["parser"]().parse_args(["--run-id", "native_warmstart_CPU_preflight_only", "--validate-only"])
configuration, _, _, prepared = ns["preflight"](args)
check("actual stdlib preflight with pinned existing source succeeds", prepared["passed"] is True and prepared["Torch_imported"] is False)
report = {"schema": 1, "kind": "native_B1_warmstart_from_native100_stdlib_readiness_v9", "passed": True,
    "checks": checks, "check_count": len(checks), "entry_sha256": hashlib.sha256(entry.read_bytes()).hexdigest(),
    "config_sha256": hashlib.sha256(cfgpath.read_bytes()).hexdigest(), "source_warmstart": source_proof,
    "full_stream_audit": stream, "Torch_imported": False, "CUDA_initialized": False,
    "actual_native_strict_weight_Adam_Cosine_restore_GPU_memory_and_WER_pending": True}
(out / "full_nemo_warmstart_readiness_v9.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"passed": True, "checks": len(checks), "entry_sha256": report["entry_sha256"], "config_sha256": report["config_sha256"]}))
