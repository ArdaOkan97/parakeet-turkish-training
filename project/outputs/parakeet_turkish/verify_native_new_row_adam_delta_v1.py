"""Root-only small CUDA/Lightning oracle for six-row AdamW parameter-delta scaling.

Uses narrow synthetic tensors with exact native row counts, not a full model,
checkpoint, ASR loss, or quality experiment. --validate-only is stdlib-only.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
HELPER_SHA = "caf6ca6d9609b1b41e03bebaf1aa0f80c0b437033edc50bc17eef14b1c6999e8"
PINS = {
    "native_new_row_adam_delta_v1.py": HELPER_SHA,
    "preload_nemo_linker_v1.py": "f8787b0f3346c9f526903ff0c06de4c266f940ca3b0f41c55907cc795e7cc2ba",
    "resource_watchdog_v2.py": "64d1393e1e45cf65dac14f78ac955331c91b4a3f0eafbf7535196e2b61778641",
    "paired_windows_commit_telemetry_v1.py": "2ae9f2cdfc388a294914e000c60b62fe09cddac147700c5762851d626a95a683",
}
TARGETS = ("decoder.prediction.embed.weight", "joint.joint_net.2.weight", "joint.joint_net.2.bias")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(args):
    require(args.output.resolve().is_relative_to(ROOT) and not args.output.exists(), "New workspace output required")
    require(0 < args.max_wall_seconds <= 180, "Bounded oracle window required")
    for name, expected in PINS.items():
        require(sha(HERE / name) == expected, "Frozen helper/source changed: " + name)
    report = {"code_sha256": sha(__file__), "helper_sha256": HELPER_SHA,
              "source_sha256": dict(PINS), "cap_bytes": 8_000_000_000,
              "scope": "Small synthetic FP32 parameter tensors with native row geometry, plus a real Lightning BF16 accumulation loop; no full native checkpoint/model, ASR or quality claim",
              "tensor_shapes": {TARGETS[0]: [8199,8], TARGETS[1]: [8204,8], TARGETS[2]: [8204], "unrelated": [2,2]},
              "new_rows": list(range(8192,8198)), "native_blank_row": 8198,
              "native_duration_rows": list(range(8199,8204)),
              "actual_GPU_optimizer_steps_executed": False, "actual_Lightning_bf16_optimizer_steps_executed": False}
    if args.native_row_metadata:
        path = args.native_row_metadata.resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), "Optional actual row metadata must be workspace JSON")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        report["actual_native_row_metadata"] = {"path": str(path), "sha256": sha(path), "value": metadata,
            "scope": "Supplied source metadata only; oracle does not load checkpoint weights"}
    return report


def runtime(args, report):
    from preload_nemo_linker_v1 import preload
    report["preload"] = preload()
    import torch
    import lightning.pytorch as pl
    from native_new_row_adam_delta_v1 import install_native_new_row_adam_delta
    from resource_watchdog_v2 import ResourceGuard, POLICY, sample_memory, worker_process_commit_reason
    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(device).total_memory
    require(15e9 <= total <= 18e9, "Expected local16GB-class GPU")
    torch.cuda.set_per_process_memory_fraction(report["cap_bytes"] / total, device)
    report["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "lightning")}
    report.update(cases=[], memory_events=[], guard_policy=dict(POLICY))
    started, guard = time.monotonic(), ResourceGuard()

    def memory(phase):
        torch.cuda.synchronize(device)
        observation = sample_memory(os.getpid(), phase)
        reason, decision = guard.observe(observation["system_memory"], observation["worker_private_memory"],
                                         observed_monotonic=observation["observed_monotonic"])
        reason = reason or worker_process_commit_reason(observation["calling_process_commit_memory"], os.getpid())
        observation.update(reason=reason, decision=decision, cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                           cuda_reserved_bytes=torch.cuda.memory_reserved(device))
        report["memory_events"].append(observation)
        require(reason is None, "Unchanged Windows guard: " + str(reason))
        require(time.monotonic() - started < args.max_wall_seconds, "Oracle wall-time bound exceeded")

    def native_modules(model):
        model.decoder = torch.nn.Module()
        model.decoder.blank_idx = 8198
        model.decoder.prediction = torch.nn.ModuleDict({"embed": torch.nn.Embedding(8199,8,padding_idx=8198)})
        model.joint = torch.nn.Module()
        model.joint.joint_net = torch.nn.Sequential(torch.nn.Identity(),torch.nn.Identity(),torch.nn.Linear(8,8204))
        model.unrelated = torch.nn.Parameter(torch.zeros(2,2))
        with torch.no_grad():
            for index, parameter in enumerate(model.parameters()):
                values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
                parameter.copy_((values * .0007 + index).sin() * .02)

    def snapshot(model):
        return {name: parameter.detach().clone() for name,parameter in model.named_parameters()}

    def state_snapshot(optimizer):
        return [{name: value.detach().clone() if isinstance(value,torch.Tensor) else copy.deepcopy(value)
                 for name,value in optimizer.state[parameter].items()}
                for group in optimizer.param_groups for parameter in group["params"]]

    def state_equal(expected, actual):
        require(len(expected) == len(actual), "Adam state parameter count changed")
        for left,right in zip(expected,actual):
            require(left.keys() == right.keys(), "Adam state fields changed")
            for name in left:
                require(torch.equal(left[name],right[name]) if isinstance(left[name],torch.Tensor) else left[name] == right[name],
                        "Adam moments/step changed by hook: " + name)

    def initialized_adam(optimizer):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                optimizer.state[parameter] = {"step": torch.tensor(100.),
                    "exp_avg": torch.full_like(parameter,.003), "exp_avg_sq": torch.full_like(parameter,.007)}

    def verify_rows(before, ordinary, actual, multiplier):
        rows = {}
        for name, parameter in actual.items():
            if name in TARGETS:
                require(torch.equal(parameter[:8192],ordinary[name][:8192])
                        and torch.equal(parameter[8198:],ordinary[name][8198:]), "Old/blank/duration rows received boosted update: " + name)
                expected = ordinary[name][8192:8198] if multiplier == 1. else (
                    before[name][8192:8198] + multiplier * (ordinary[name][8192:8198] - before[name][8192:8198]))
                require(torch.equal(parameter[8192:8198],expected), "New six row deltas differ from exact declared expression: " + name)
                delta = ordinary[name][8192:8198] - before[name][8192:8198]
                require(bool((delta != 0).any()), "Degenerate no-update oracle")
                rows[name] = {"old_and_blank_and_duration_rows_bitwise_ordinary": True,
                    "new_six_rows_bitwise_declared_delta": True,
                    "ordinary_new_row_delta_l2": float(delta.double().norm()),
                    "boosted_new_row_delta_l2": float((parameter[8192:8198]-before[name][8192:8198]).double().norm())}
            else:
                require(torch.equal(parameter,ordinary[name]), "Unrelated parameter changed by row helper")
        return rows

    memory("before_small_actual_GPU_optimizer_oracle")
    for case_name, multiplier, weight_decay, restored in (
            ("fresh_AdamW_no_decay",6.,0.,False),
            ("restored_Adam100_with_decay",6.,.001,True),
            ("control_multiplier_one",1.,.001,True)):
        ordinary, boosted = torch.nn.Module(), torch.nn.Module()
        native_modules(ordinary); native_modules(boosted)
        boosted.load_state_dict(ordinary.state_dict(),strict=True)
        ordinary.to(device); boosted.to(device)
        kwargs = {"lr": 5e-5, "betas": (.9,.98), "eps": 1e-8, "weight_decay": weight_decay, "foreach": False}
        base_optimizer = torch.optim.AdamW(ordinary.parameters(),**kwargs)
        boost_optimizer = torch.optim.AdamW(boosted.parameters(),**kwargs)
        if restored:
            initialized_adam(base_optimizer)
            boost_optimizer.load_state_dict(copy.deepcopy(base_optimizer.state_dict()))
        for index, (left,right) in enumerate(zip(ordinary.parameters(),boosted.parameters())):
            gradient = (torch.arange(left.numel(), device=device, dtype=torch.float32).reshape(left.shape) * .001 + index).cos() * .01
            left.grad, right.grad = gradient.clone(), gradient.clone()
        before = snapshot(boosted)
        gradients = {name:p.grad.clone() for name,p in boosted.named_parameters()}
        hook = install_native_new_row_adam_delta(boosted,boost_optimizer,multiplier=multiplier,torch_module=torch)
        python_rng, cpu_rng, cuda_rng = random.getstate(),torch.get_rng_state().clone(),torch.cuda.get_rng_state(device).clone()
        base_optimizer.step(); boost_optimizer.step(); torch.cuda.synchronize(device)
        report["actual_GPU_optimizer_steps_executed"] = True
        rows = verify_rows(before,snapshot(ordinary),snapshot(boosted),multiplier)
        state_equal(state_snapshot(base_optimizer),state_snapshot(boost_optimizer))
        require(all(torch.equal(p.grad,gradients[name]) for name,p in boosted.named_parameters()), "Helper changed gradients")
        require(random.getstate() == python_rng and torch.equal(torch.get_rng_state(),cpu_rng)
                and torch.equal(torch.cuda.get_rng_state(device),cuda_rng), "Helper/Adam oracle changed RNG")
        require(hook["identity"]["statistics"] == {"pre_step_calls":1,"post_step_calls":1,
                "scaled_optimizer_updates": int(multiplier != 1)}, "Manual optimizer hooks did not fire once")
        case = {"case":case_name,"passed":True,"multiplier":multiplier,"weight_decay":weight_decay,
                "synthetic_initial_Adam_step":100 if restored else 0,"row_checks":rows,
                "all_Adam_moments_and_steps_bitwise_match":True,"all_gradients_unchanged":True,"RNG_unchanged":True,
                "helper_identity":hook["identity"]}
        hook["remove"]()
        require(not hasattr(boost_optimizer,"_native_new_row_delta_hook_v1"), "Hook removal failed")
        report["cases"].append(case)
        memory("after_" + case_name)
        del ordinary,boosted,base_optimizer,boost_optimizer,before,gradients,hook
        torch.cuda.empty_cache()

    # This executes the real Lightning bf16-mixed closure/accumulation path.
    class TinyNativeShapeModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            native_modules(self)
            self.hook = None
            self.observers = []
            self.optimizer_records = []
            self.microbatch_calls = 0
            self.step_snapshot = None
            self.ordinary_after = self.ordinary_state = self.ordinary_gradients = None

        def training_step(self,batch,batch_idx):
            self.microbatch_calls += 1
            embedded = self.decoder.prediction["embed"](batch[0])
            output = self.joint.joint_net(embedded)
            return output.float().square().mean() + self.unrelated.float().square().mean()

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(),lr=5e-5,betas=(.9,.98),eps=1e-8,weight_decay=.001,foreach=False)

        def on_train_start(self):
            optimizer = self.trainer.optimizers[0]
            require(int(self.trainer.global_step) == 0, "Tiny Lightning loop must start at0")
            # Mimics the declared install order, after an optimizer state load.
            initialized_adam(optimizer)
            optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))

            def before(actual,args,kwargs):
                self.step_snapshot = snapshot(self)

            def ordinary_post(actual,args,kwargs):
                self.ordinary_after = snapshot(self)
                self.ordinary_state = state_snapshot(actual)
                self.ordinary_gradients = {name:p.grad.clone() for name,p in self.named_parameters()}

            def verified_post(actual,args,kwargs):
                rows = verify_rows(self.step_snapshot,self.ordinary_after,snapshot(self),6.)
                state_equal(self.ordinary_state,state_snapshot(actual))
                require(all(torch.equal(p.grad,self.ordinary_gradients[name]) for name,p in self.named_parameters()), "Mixed-precision helper mutated gradients")
                self.optimizer_records.append({"optimizer_call":len(self.optimizer_records)+1,
                    "microbatches_seen":self.microbatch_calls,"row_checks":rows,
                    "Adam_moments_and_steps_unchanged_by_hook":True,
                    "Adam_state_step":float(next(iter(actual.state.values()))["step"])})
                self.step_snapshot = self.ordinary_after = self.ordinary_state = self.ordinary_gradients = None
                memory("Lightning_after_actual_optimizer_step_" + str(len(self.optimizer_records)))

            self.observers.extend([optimizer.register_step_pre_hook(before),optimizer.register_step_post_hook(ordinary_post)])
            self.hook = install_native_new_row_adam_delta(self,optimizer,multiplier=6.,torch_module=torch)
            self.observers.append(optimizer.register_step_post_hook(verified_post))

    tiny = TinyNativeShapeModel()
    trainer = pl.Trainer(accelerator="gpu",devices=1,precision="bf16-mixed",accumulate_grad_batches=2,
        max_steps=2,max_epochs=1,logger=False,enable_checkpointing=False,enable_progress_bar=False,
        enable_model_summary=False,num_sanity_val_steps=0,gradient_clip_val=1.,gradient_clip_algorithm="norm",
        default_root_dir=str(args.output.parent),deterministic=False,benchmark=False)
    dataset = torch.utils.data.TensorDataset(torch.tensor([[1,8192],[7,8193],[9,8196],[11,8197]],dtype=torch.long))
    loader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)
    try:
        trainer.fit(tiny,train_dataloaders=loader)
        report["actual_Lightning_bf16_optimizer_steps_executed"] = True
        require(int(trainer.global_step) == 2 and tiny.microbatch_calls == 4, "Lightning accumulation/step counts differ")
        require([row["microbatches_seen"] for row in tiny.optimizer_records] == [2,4]
                and [row["Adam_state_step"] for row in tiny.optimizer_records] == [101.,102.], "BF16 hooks fired on microbatches or changed restored Adam counters")
        require(tiny.hook["identity"]["statistics"] == {"pre_step_calls":2,"post_step_calls":2,"scaled_optimizer_updates":2}, "BF16 official optimizer hooks were not once-per-update")
        report["cases"].append({"case":"real_Lightning_bf16_accumulation","passed":True,
            "optimizer_updates":2,"microbatches":4,"accumulation":2,"precision":"bf16-mixed",
            "initial_Adam_step":100,"helper_installed_after_state_load":True,
            "per_update_checks":tiny.optimizer_records,"helper_identity":tiny.hook["identity"]})
    finally:
        if tiny.hook:
            tiny.hook["remove"]()
        for observer in tiny.observers:
            observer.remove()
    memory("after_real_Lightning_bf16_oracle")
    for name,expected in PINS.items():
        require(sha(HERE/name) == expected,"Source changed during actual oracle")
    report.update(passed=True,status="complete",wall_seconds=time.monotonic()-started,
                  peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(device),
                  quality_or_full_model_gradient_equivalence_claimed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--validate-only",action="store_true")
    parser.add_argument("--gpu-window-authorized",action="store_true")
    parser.add_argument("--native-row-metadata",type=Path)
    parser.add_argument("--max-wall-seconds",type=float,default=90.)
    args = parser.parse_args()
    report = {"schema":1,"kind":"native_new_row_AdamW_delta_actual_GPU_oracle_v1","passed":False}
    try:
        report.update(preflight(args))
        if args.validate_only:
            require("torch" not in sys.modules and "lightning" not in sys.modules,"Stdlib preparation imported heavy runtime")
            report.update(passed=True,status="CPU_prepared_GPU_unmeasured",heavy_runtime_imported=False)
        else:
            require(args.gpu_window_authorized,"Root-owned sole GPU window required")
            runtime(args,report)
    except Exception as error:
        report.update(passed=False,status="failed",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        if args.output.resolve().is_relative_to(ROOT) and not args.output.exists():
            args.output.parent.mkdir(parents=True,exist_ok=True)
            with args.output.open("x",encoding="utf-8") as stream:
                json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
                stream.write("\n")


if __name__ == "__main__":
    main()
