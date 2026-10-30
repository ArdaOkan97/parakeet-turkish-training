"""Bounded stdlib metadata/cursor/restore guards; no checkpoint tensor load."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
path = OUT / "native_completed_boundary_warmstart_v1.py"
spec = importlib.util.spec_from_file_location("native_boundary", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
checks = []


def check(name, value):
    if not value:
        raise AssertionError(name)
    checks.append(name)


def reject(name, operation):
    try:
        operation()
    except ValueError:
        checks.append(name)
    else:
        raise AssertionError(name)


boundary = {"source_checkpoint_local_step": 250, "source_cumulative_optimizer_step": 350,
            "first_unseen_source_index": 5600, "remaining_recordings": 144176,
            "target_local_optimizer_update_limit": 9011}
lrs = [.0002996, .00004995, .00004995]
optimizer = {"param_groups": [{"params": list(range(n)), "lr": lr} for n, lr in zip([828, 9, 6], lrs)],
             "state": {index: {"step": 350} for index in range(207)}}
scheduler = {"max_steps": 9361, "warmup_steps": 25, "last_epoch": 350, "_step_count": 351, "_last_lr": lrs}
checkpoint = {"global_step": 250, "state_dict": {str(index): None for index in range(917)},
              "optimizer_states": [optimizer], "lr_schedulers": [scheduler],
              "loops": {"fit_loop": {"epoch_loop.batch_progress": {scope: {field: 4000 for field in ("ready", "started", "processed", "completed")} for scope in ("total", "current")}}}}
proof = module.verify_completed_checkpoint_state(checkpoint, boundary)
check("local250 maps to cumulative350, actual4000 committed B1 microbatches", proof["source_checkpoint_local_step"] == 250 and proof["source_cumulative_optimizer_step"] == 350 and proof["completed_source_microbatches"] == 4000)
bad = copy.deepcopy(checkpoint)
bad["lr_schedulers"][0]["last_epoch"] = 250
reject("local step cannot replace cumulative scheduler position", lambda: module.verify_completed_checkpoint_state(bad, boundary))
bad = copy.deepcopy(checkpoint)
bad["optimizer_states"][0]["state"][0]["step"] = 349
reject("inconsistent Adam tensor step rejected", lambda: module.verify_completed_checkpoint_state(bad, boundary))
bad = copy.deepcopy(checkpoint)
bad["loops"]["fit_loop"]["epoch_loop.batch_progress"]["total"]["completed"] = 3999
reject("stock periodic lag cannot be claimed as completed DEV metadata", lambda: module.verify_completed_checkpoint_state(bad, boundary))
bad = copy.deepcopy(checkpoint)
bad["loops"]["fit_loop"]["epoch_loop.batch_progress"]["current"]["processed"] = 4001
reject("partial accumulation metadata rejected", lambda: module.verify_completed_checkpoint_state(bad, boundary))
for batch in (1, 2, 4, 8):
    indices, plan = module.suffix_range(149776, boundary, batch)
    check(f"B{batch} preserves same5600 cursor/effective16/9011updates", indices.start == 5600 and indices.stop == 149776 and len(indices) == 144176 and batch * plan["accumulation"] == 16 and plan["optimizer_updates"] == 9011)
    check(f"B{batch} grouping equivalence remains unclaimed", plan["B1_gradient_equivalence_claimed"] is False and plan["physical_mean_volume_grouping_changed"] == (batch != 1))
reject("unsupported effective batch not silently adopted", lambda: module.suffix_range(149776, boundary, 16))


class Optimizer:
    def __init__(self):
        self.param_groups = [{"params": list(range(n)), "lr": 0} for n in [828, 9, 6]]
        self.loads = 0
    def load_state_dict(self, value):
        self.loads += 1
        self.param_groups = copy.deepcopy(value["param_groups"])
    def step(self):
        raise AssertionError("Restore must not execute optimizer step")


class Scheduler:
    max_steps, warmup_steps, last_epoch = 9361, 25, 0
    def __init__(self):
        self.loads = 0
    def load_state_dict(self, value):
        self.loads += 1
        self.__dict__.update(copy.deepcopy(value))
    def step(self):
        raise AssertionError("Restore must not execute scheduler step")


target_optimizer, target_scheduler = Optimizer(), Scheduler()
restored = module.restore_official_optimizer_scheduler(target_optimizer, target_scheduler, optimizer, scheduler, boundary)
check("official Adam then Cosine restored once without an extra step", target_optimizer.loads == target_scheduler.loads == 1 and target_scheduler.last_epoch == 350 and restored["optimizer_or_scheduler_step_during_restore"] is False)
check("no Torch/NeMo/Lightning imports", not any(name in sys.modules for name in ("torch", "nemo", "lightning")))
report = {"kind": "native_completed_DEV_boundary_warmstate_CPU_guard_v1", "passed": True,
          "check_count": len(checks), "checks": checks, "helper_sha256": module.sha(path),
          "synthetic_metadata_only": True, "actual_source_DEV350_or_tensor_load_verified": False,
          "Torch_NeMo_imported": False, "CUDA_initialized": False}
output = OUT / "native_completed_boundary_warmstart_CPU_verification_v1.json"
with output.open("x", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
print(json.dumps({"passed": True, "checks": len(checks), "helper_sha256": report["helper_sha256"]}))
