"""Small real CPU-Adam fixture, no base model, audio, CUDA calls or corpus."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "outputs/parakeet_turkish"))


def main(args):
    output = args.output.resolve()
    storage = Path(r"E:\gpu_training_scratchpad").resolve()
    if not output.is_relative_to(storage) or os.environ.get("GPU_TRAINING_STORAGE_ROOT") != str(storage):
        raise ValueError("Apply designated E: process storage before this fixture")
    # Only this new CPU fixture process is masked; the running GPU environment
    # is untouched. Optimizer internals may query CUDA availability themselves.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch
    from bpe_full_decoder_optimizer_v1 import BPEFullDecoderOptimizer, SHARED, TABLES
    torch.set_num_threads(1)
    checks, rejects = [], []
    class Model:
        def __init__(self, pairs): self.values = dict(pairs)
        def named_parameters(self): return iter(self.values.items())
    def expect(condition, label):
        if not condition: raise AssertionError(label)
        checks.append(label)
    def reject(call, label):
        try: call()
        except (ValueError, RuntimeError): rejects.append(label)
        else: raise AssertionError("Expected rejection: " + label)
    heads = [SHARED[0], SHARED[1], TABLES[0], *SHARED[2:], TABLES[1], TABLES[2]]
    loras = [f"encoder.layer{i}.lora_{letter}" for i in range(96) for letter in ("A", "B")]
    def construct(vocabulary):
        old = {}
        # Intentionally insert LoRAs first: source state order is not Adam IDs.
        for i, name in enumerate([*loras, *heads]):
            size = (8199, 2) if name == TABLES[0] else ((8204, 2) if name == TABLES[1] else
                    (8204,) if name == TABLES[2] else (2, 2))
            old[name] = torch.full(size, .25 + i / 1000, dtype=torch.float32)
        states = {i: {"step": torch.tensor(37.), "exp_avg": torch.full_like(old[n], .01),
                      "exp_avg_sq": torch.full_like(old[n], .02)} for i, n in enumerate([*heads, *loras])}
        group = dict(lr=1e-5, betas=(.9, .999), eps=1e-8, weight_decay=.01, amsgrad=False,
                     foreach=None, maximize=False, capturable=False, differentiable=False, fused=None)
        source = {"step": 37, "selected_step": 37, "weights": old,
                  "optimizer": {"state": states, "param_groups": [dict(group, params=list(range(15))),
                       dict(group, params=list(range(15, 207)), lr=3e-5, weight_decay=0.)]}}
        pairs = []
        for name, value in old.items():
            if name in TABLES and vocabulary == 8711:
                shape = list(value.shape); shape[0] += 512
                new = torch.full(shape, .2)
                new[:8199].copy_(value[:8199])
                if name != TABLES[0]: new[8711:].copy_(value[8199:])
            else: new = value.clone()
            pairs.append((name, torch.nn.Parameter(new)))
        model = Model(pairs)
        optimizer = BPEFullDecoderOptimizer(model, source,
            source_record={"path": "synthetic_CPU_only", "sha256": "a"*64, "bytes": 1},
            vocabulary=vocabulary, torch_module=torch)
        return model, source, optimizer
    for vocabulary in (8199, 8711):
        model, source, opt = construct(vocabulary)
        label = str(vocabulary)
        expect(len(opt.mature) == 17 and len(opt.fresh) == (3 if vocabulary == 8711 else 0), label+" partitions")
        expect(all(not model.values[n].requires_grad for n in loras), label+" frozen 192")
        expect(opt.state_dict()["new_row_Adam_clock"] == (0 if vocabulary == 8711 else None), label+" fresh clock")
        expect(all(p.data_ptr() == opt._slice(model.values[item['name']], item['target_rows']).data_ptr()
                   for p, item in zip((*opt.mature, *opt.fresh), (*opt.mature_plan, *opt.fresh_plan))), label+" alias")
        before = {n: p.detach().clone() for n, p in model.values.items()}
        for p in opt.heads: p.grad = torch.ones_like(p)
        norm = opt.step()
        expect(torch.isfinite(norm) and opt.updates == 1, label+" actual CPU step")
        expect(all(torch.equal(model.values[n], before[n]) for n in loras), label+" frozen byte exact")
        expect(all(not torch.equal(model.values[n], before[n]) for n in heads), label+" all old 15 train")
        expect(opt.state_dict()["mature_Adam_clock"] == 38, label+" mature clock retained")
        if vocabulary == 8711:
            expect(all(not torch.equal(model.values[n][8199:8711], before[n][8199:8711]) for n in TABLES), label+" fresh updates")
            expect(opt.state_dict()["new_row_Adam_clock"] == 1, label+" new clock 1")
            expect(all(not torch.equal(model.values[n][8711:], before[n][8711:]) for n in TABLES[1:]), label+" durations active")
        saved = copy.deepcopy(opt.state_dict())
        opt.zero_grad()
        expect(all(p.grad is None for p in opt.heads), label+" zero gradients")
        opt.load_state_dict(saved)
        expect(opt.updates == 1, label+" own exact restore")
        corrupt = copy.deepcopy(saved); corrupt["mature_optimizer"]["param_groups"][0]["params"].reverse()
        reject(lambda: opt.load_state_dict(corrupt), label+" reject partition reorder")
        corrupt = copy.deepcopy(saved); corrupt["mature_Adam_clock"] = 1
        reject(lambda: opt.load_state_dict(corrupt), label+" reject clock reset")
        corrupt = copy.deepcopy(saved); corrupt["identity"]["six_glyph_post_Adam_gain"] = 1
        reject(lambda: opt.load_state_dict(corrupt), label+" reject six gain")
        expect(all(source["optimizer"]["state"][i]["step"].item() == 37 for i in range(207)), label+" source not mutated")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"kind": "BPE_full_decoder_actual_small_CPU_tensor_verification_v1", "passed": True,
        "positive_checks": checks, "rejections": rejects,
        "torch_version": torch.__version__, "device": "CPU only; CUDA_VISIBLE_DEVICES=-1 before Torch import",
        "geometry": "Actual 8199/8711 row offsets; synthetic width 2/shared shapes; not actual model fit",
        "source": {"path": str(ROOT/"outputs/parakeet_turkish/bpe_full_decoder_optimizer_v1.py"),
                   "sha256": hashlib.sha256((ROOT/"outputs/parakeet_turkish/bpe_full_decoder_optimizer_v1.py").read_bytes()).hexdigest()},
        "models_audio_GPU_TEST_used": False, "source_admission_or_native_quality_claimed": False}
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"passed": True, "positive": len(checks), "rejections": len(rejects),
                      "report": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
