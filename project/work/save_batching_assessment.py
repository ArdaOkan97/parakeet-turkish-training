"""Reproduce the completed CPU-only batching assessment and save evidence."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0, str(OUTPUT))
from efficient_tdt import chunked_tdt_loss


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


class Joint(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(7, 38)

    def forward(self, encoder, decoder):
        return self.head(torch.relu(encoder + decoder))


torch.set_num_threads(1)
torch.manual_seed(20261010)
frames = [9, 12, 13, 8]
targets = [4, 7, 5, 3]
durations = [0, 1, 2, 3, 4]
reference = Joint()
encoders = [torch.randn(t, 7) for t in frames]
decoders = [torch.randn(u + 1, 7) for u in targets]
labels = [torch.randint(0, 32, (u,)) for u in targets]
parity_cases = []
for batch_size in (1, 2, 4):
    model = copy.deepcopy(reference)
    losses, valid_gradients, padding_gradients, tensor_shapes = [], [], [], []
    for start in range(0, 4, batch_size):
        positions = range(start, start + batch_size)
        max_t = max(frames[i] for i in positions)
        max_u = max(targets[i] for i in positions)
        encoder = torch.zeros(batch_size, max_t, 7)
        decoder = torch.zeros(batch_size, max_u + 1, 7)
        target = torch.full((batch_size, max_u), 32, dtype=torch.long)
        for index, position in enumerate(positions):
            encoder[index, :frames[position]] = encoders[position]
            decoder[index, :targets[position] + 1] = decoders[position]
            target[index, :targets[position]] = labels[position]
        encoder.requires_grad_()
        decoder.requires_grad_()
        loss = chunked_tdt_loss(
            model, encoder, decoder, target, 33, 32, durations,
            torch.tensor(frames[start:start + batch_size]),
            torch.tensor(targets[start:start + batch_size]),
            chunk_size=4, reduction="mean",
        )
        (loss / (4 / batch_size)).backward()
        losses.append(float(loss) * batch_size / 4)
        tensor_shapes.append({"encoder_projected": list(encoder.shape),
                              "decoder_projected": list(decoder.shape),
                              "labels": list(target.shape)})
        for index, position in enumerate(positions):
            valid_gradients.append((encoder.grad[index, :frames[position]].clone(),
                                    decoder.grad[index, :targets[position] + 1].clone()))
            padding_gradients.extend((encoder.grad[index, frames[position]:],
                                      decoder.grad[index, targets[position] + 1:]))
    parameter_gradients = [parameter.grad.clone() for parameter in model.parameters()]
    if batch_size == 1:
        baseline_loss = sum(losses)
        baseline_parameter_gradients = parameter_gradients
        baseline_input_gradients = valid_gradients
    parameter_delta = max(float((left - right).abs().max())
                          for left, right in zip(parameter_gradients, baseline_parameter_gradients))
    input_delta = max(float((left - right).abs().max())
                      for current, baseline in zip(valid_gradients, baseline_input_gradients)
                      for left, right in zip(current, baseline))
    padding_max = max(float(item.abs().max()) if item.numel() else 0.0 for item in padding_gradients)
    finite = (all(torch.isfinite(item).all() for item in parameter_gradients)
              and all(torch.isfinite(item).all() for pair in valid_gradients for item in pair))
    passed = (finite and abs(sum(losses) - baseline_loss) < 3e-6
              and parameter_delta < 3e-6 and input_delta < 3e-6 and padding_max == 0)
    assert passed
    parity_cases.append({"microbatch_size": batch_size, "accumulation": 4 // batch_size,
                         "microbatch_tensor_shapes": tensor_shapes, "mean_loss": sum(losses),
                         "absolute_loss_difference_from_B1x4": abs(sum(losses) - baseline_loss),
                         "maximum_parameter_gradient_difference": parameter_delta,
                         "maximum_valid_input_gradient_difference": input_delta,
                         "maximum_padding_gradient": padding_max, "all_gradients_finite": bool(finite),
                         "passed": bool(passed)})

data_audit_path = OUTPUT / "lora_training_data_verification.json"
data_audit = json.loads(data_audit_path.read_text(encoding="utf-8"))
assert data_audit["passed"]
streams = data_audit["streams"]
turkish = (streams["turkish_commonvoice"]["accepted_source_ids_and_labels"]
           + streams["turkish_fleurs_train"]["accepted_source_ids_and_labels"])
replay = streams["retention_fleurs_train"]["accepted_source_ids_and_labels"]


def quantiles(rows, field):
    values = sorted(row[field] for row in rows)
    return {str(fraction): values[round((len(values) - 1) * fraction)] for fraction in (.1, .5, .9, 1)}


def padded_lattice(rows):
    return len(rows) * max(row["encoder_frames"] for row in rows) * (max(row["target_tokens"] for row in rows) + 1)


def unpadded_lattice(rows):
    return sum(row["encoder_frames"] * (row["target_tokens"] + 1) for row in rows)


rng = random.Random(20261010)
padding_values = []
for _ in range(1000):
    tr_pair = rng.sample(turkish, 2)
    replay_pair = rng.sample(replay, 2)
    quartet = tr_pair + replay_pair
    base = unpadded_lattice(quartet)
    padding_values.append(((padded_lattice(tr_pair) + padded_lattice(replay_pair)) / base,
                           padded_lattice(quartet) / base))
padding_summary = {}
for index, name in enumerate(("B2_homogeneous_pairs", "B4_mixed")):
    values = sorted(row[index] for row in padding_values)
    padding_summary[name] = {"median_factor_sorted_index500": values[500],
                             "p90_factor_sorted_index900": values[900]}


def median_pair(rows):
    median_t = sorted(row["encoder_frames"] for row in rows)[len(rows) // 2]
    median_u = sorted(row["target_tokens"] for row in rows)[len(rows) // 2]
    return sorted(rows, key=lambda row: (abs(math.log(row["encoder_frames"] / median_t))
                                        + abs(math.log(row["target_tokens"] / median_u)), row["source_id"]))[:2]


def high_lattice_pair(rows):
    return sorted(rows, key=lambda row: (-row["encoder_frames"] * (row["target_tokens"] + 1), row["source_id"]))[:2]


manifest_rows = {}
for manifest in (ROOT / "work/data/tr_train.jsonl", ROOT / "work/data/tr_fleurs_train.jsonl",
                 ROOT / "work/data/replay_train.jsonl"):
    for row in read_rows(manifest):
        manifest_rows[row["source_id"]] = (row, manifest)
cache_dir = ROOT / "work/encoder-cache" / data_audit["cache_namespace"]
representatives = {}
for name, selector in (("median", median_pair), ("high_lattice", high_lattice_pair)):
    representatives[name] = {}
    for stream, pool in (("tr", turkish), ("replay", replay)):
        values = []
        for selected in selector(pool):
            row, manifest = manifest_rows[selected["source_id"]]
            assert row["dataset_split"] == "train"
            cache_path = cache_dir / (hashlib.sha256(row["source_id"].encode()).hexdigest() + ".pt")
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            assert cached["source_id"] == row["source_id"]
            assert list(cached["features"].shape) == [selected["encoder_frames"], 1024]
            assert int(cached["labels"].numel()) == selected["target_tokens"]
            assert hashlib.sha256(cached["labels"].numpy().tobytes()).hexdigest() == selected["labels_sha256"]
            values.append({**selected, "manifest": str(manifest), "dataset_split": row["dataset_split"],
                           "audio_filepath": row["audio_filepath"], "wav_sha256": row["wav_sha256"],
                           "cache_path": str(cache_path), "cache_sha256": sha(cache_path),
                           "cached_feature_shape": list(cached["features"].shape),
                           "lattice_states": selected["encoder_frames"] * (selected["target_tokens"] + 1)})
        representatives[name][stream] = values

trainer_changes = [
    "Collate a list of raw prepared mono16k audio, padding with attention masks. Retain per-example source IDs and original label counts.",
    "Right-pad cached labels with the valid unchanged blank ID8192, not -100; prepend one blank for decoder input. Pass explicit target_lengths to chunked_tdt_loss.",
    "Use encoded.attention_mask.sum(-1) as each T; trim output only to max(T), never int(attention_mask.sum()) across batch. Pad cached original teacher features and verify each valid student/cache T agrees.",
    "Batch the student encoder/projector/decoder/chunked joint, retaining reduction=mean, full token normalization, chunk16, FP32 weights and BF16 autocast.",
    "Replay KL must sample each clip's existing linspace over its valid T and U+1, with min(8,length) grid dimensions. Global padded maxT/U sampling changes the objective. The safest first version calls preservation_losses separately on valid sliced replay rows, then vectorizes per-example gathers later.",
    "Average feature cosine loss over valid frames separately per clip, then over clips; flattened frame averaging changes duration weighting and padding averaging is incorrect.",
    "B2x2 homogeneous groups: TR-pair mean TDT /2, replay-pair mean(TDT+5*tokenKL+.5*durationKL+feature)/2. B4 mixed: zero preservation terms for TR and average all four clip terms; mean over replay-only would double retention weights.",
    "Keep frozen encoder eval/BatchNorm statistics, dropout/LayerDrop disabled, all24 rank8 LoRA, old-ID-only replay, original teacher heads, fixed seed and the actual head/LoRA AdamW groups and new-row update multiplier.",
    "Use shuffled bounded reservoirs and length matching within each stream while preserving 2TR+2replay per effective update and existing uniform replay-language sampling/cycles. Bucket by T and U or lattice cost, with no data exclusions or heldout-dependent selection.",
    "After objective equivalence, optionally prefetch WAV/STFT/cache data with a Windows-safe top-level DataLoader (initially2 workers, pinned memory, persistent workers); measure this separately from batching.",
]
gpu_procedure = [
    "Wait for the current trainer and finite finalization worker to release the GPU. Do not alter or compare throughput while another process uses it.",
    "Run B1x4, B2x2 and B4x1 in fresh separate processes on exactly the same four TRAIN clips in each recorded median/high-lattice quartet; use identical frozen base, teacher cache, tokenizer, selected trainable snapshot, seed and model/code hashes.",
    "Use all24 rank8 LoRA with nonzero trained B for gradient parity; retain FP32 parameters/Adam states, BF16 autocast, chunk16, accumulation effective4, actual optimizer groups/LRs/decay, clipnorm1, new-row multiplier15 and loss coefficients tokenKL5/durationKL.5/feature1. No dtype-storage change in this comparison.",
    "Before stepping, compare per-clip loss components, total effective-four objective, all head and LoRA gradients, finite gradients, zero padding gradients and unchanged BatchNorm buffers against B1x4. Also compare batched valid encoder features against singles. Select numerical tolerances from matched BF16 behavior and report every deviation, rather than claiming bitwise equality.",
    "Then perform3-5 untimed warm updates to initialize kernels/Adam states, followed by at least20 timed updates on each quartet with one optimizer update per4 recordings and no development inference or checkpoint I/O inside the timing window.",
    "Record synchronized CUDA wall time per optimizer update, median/p90 ms, clips/s, total real audio seconds/s, allocated/reserved peaks, CPU/GPU utilization, padding inflation, all objective coefficients and any OOMs. Reset peak counters after warm-up; include retained optimizer states.",
    "Report compute-only timing with identical prefetched CPU inputs separately from end-to-end audio decode/STFT/cache read/H2D timing. Apply the same prefetch and synchronization policy to every batch layout.",
    "Repeat representative bucketed schedules after fixed-four parity, recording the same training-source exposure/counts/seed. B2 is the default candidate; adopt B4 only if warmed useful-audio throughput improves and worst-case16GB memory is safe.",
    "Existing profile_tdt.py is not this benchmark: it times one cold Adam update of two sequential single clips, lacks current duration/feature losses and uses different optimizer groups/new-row-update semantics. Its4.57s cannot establish a batching speedup.",
]
assessment = {
    "schema": 1,
    "created_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
    "timezone": "America/Los_Angeles",
    "status": "CPU assessment complete; actual encoder/teacher/GPU batching integration and throughput unmeasured",
    "gpu_used": False,
    "active_training_or_helper_files_modified": False,
    "recommended_first_experiment": "true microbatch2 with accumulation2, homogeneous length-matched TR and replay pairs, effectivebatch4 unchanged",
    "cpu_parity": {
        "passed": True, "device": "CPU", "dtype": "FP32", "torch_version": torch.__version__,
        "seed": 20261010, "cpu_threads": 1,
        "scope": "Tiny synthetic joint/TDT only; no real encoder, teacher preservation loss, GPU or optimizer-step parity tested",
        "joint": "Linear7->38(ReLU(encoder+decoder));33 token outputs and5 duration outputs",
        "four_fixed_encoder_lengths": frames, "four_fixed_target_lengths": targets,
        "vocab_size": 33, "blank_id": 32, "projected_hidden_size": 7, "durations": durations,
        "chunk_size": 4, "sigma": 0.0, "recompute": True, "reduction": "mean",
        "accumulation_scaling": "mean per-utterance loss/targetlength inside each microbatch, divided by number of microbatches; total averages the same4 clips",
        "number_of_compared_layouts": 3, "number_of_synthetic_clips": 4,
        "criteria": "Loss and parameter/valid-input gradient absolute differences <3e-6; all finite; padded input gradient exactly0",
        "cases": parity_cases,
    },
    "actual_pool": {"turkish_usable_clips": len(turkish), "replay_usable_clips": len(replay),
                    "cache_namespace": data_audit["cache_namespace"],
                    "original_data_audit_path": str(data_audit_path), "original_data_audit_sha256": sha(data_audit_path),
                    "quantile_method": "Sort values and choose index round((N-1)*fraction)",
                    "turkish": {"encoder_frames": quantiles(turkish, "encoder_frames"), "target_tokens": quantiles(turkish, "target_tokens")},
                    "replay": {"encoder_frames": quantiles(replay, "encoder_frames"), "target_tokens": quantiles(replay, "target_tokens")}},
    "padding_estimate": {
        "seed": 20261010, "balanced_quartets": 1000,
        "sampling": "Each quartet independently samples2 distinct accepted TR and2 distinct accepted replay rows without replacement within each stream; quartets may repeat across trials. No bucketing applied.",
        "cost_formula": "padded B*Tmax*(Umax+1) / sum_i(Ti*(Ui+1)); B2 costs sum the separate homogeneous TR and replay padded rectangles",
        "scope": "CPU lattice-padding estimate, not VRAM or throughput; ignores encoder attention, recomputation, timechunk tails, kernel efficiency and host overhead",
        "results": padding_summary,
    },
    "real_tensor_dimensions": {
        "cached_teacher_features_per_clip": "[Ti,1024]",
        "batched_teacher_features_and_student_hidden": "[B,Tmax,1024] with per-example valid-frame mask",
        "batched_projected_encoder": "[B,Tmax,640]",
        "batched_projected_decoder": "[B,Umax+1,640]",
        "padded_labels": "[B,Umax] int64; blank8192 padding; actual lengths required",
        "logit_lengths_and_target_lengths": "[B] each",
        "joint_chunk_logits": "[B,min(16,remainingT),Umax+1,8204];8199 tokens+5 durations",
        "reduced_blank_and_duration_logprobs": "[B,Tmax,Umax+1] and[B,Tmax,Umax+1,5]",
        "reduced_label_logprobs": "[B,Tmax,Umax]",
        "length_bucketing_purpose": "Limit Tmax and Umax within each microbatch; scalar lattice cost alone can combine mismatched longT/shortU and shortT/longU clips",
    },
    "representative_train_quartets": representatives,
    "trainer_changes": trainer_changes,
    "fair_gpu_procedure": gpu_procedure,
    "unprofiled_performance_hypotheses": [
        "tdt_from_log_probs contains Python branches on CUDA tensor .any() in the anti-diagonal/duration loops. Such host control flow can require CPU-GPU synchronization; batching can amortize this work across multiple clips, but its share of current runtime has not been profiled. Confirm with a CPU/CUDA profiler before optimizing or compiling the DP.",
        "Length bucketing and data prefetch may reduce padding and input stalls, but actual warmed end-to-end throughput must establish any benefit on this16GB GPU.",
    ],
    "primary_references": [{"title": "PyTorch Performance Tuning Guide", "url": "https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html",
                            "use": "General guidance on CPU-GPU synchronization, asynchronous data loading/pinned memory and variable-length sequence padding; not evidence of measured speedup here."}],
    "source_sha256": {name: sha(OUTPUT / name) for name in ("efficient_tdt.py", "train_lora.py", "profile_tdt.py", "runtime.py", "encoder_lora.py")},
    "limitations": ["CPU joint parity does not establish real batched encoder/teacher or BF16 GPU gradient parity.",
                    "No measured speedup or batched peakVRAM exists yet.",
                    "Batching improves execution efficiency; it does not itself resolve poor Turkish accuracy or new-character emissions.",
                    "All representative clips are accepted TRAIN data; development/test manifests and current run remain unchanged."],
}
destination = OUTPUT / "batching_assessment.json"
destination.write_text(json.dumps(assessment, ensure_ascii=False, indent=2), encoding="utf-8")
plan = """First experiment: true microbatch2 x accumulation2 at effectivebatch4.

Use one homogeneous, length-matched pair of Turkish training clips and one
homogeneous pair of replay clips per optimizer update. Keep the current50:50
mix, coefficients, optimizer, BF16 compute and FP32 storage. Consider B4x1
only after a fair measured comparison. The active run is unchanged.

The current chunked TDT has batch axes and accepts individual T/U lengths.
CPU FP32 synthetic joint testing compared the same4 clips at B1x4/B2x2/B4x1:
T=[9,12,13,8], U=[4,7,5,3], hidden7, vocabulary33, durations[0,1,2,3,4],
chunk4. Maximum loss difference1.1921e-7, parameter-gradient difference
9.6858e-8, valid-input-gradient difference9.3132e-9; padding gradient0.
This tests the joint/DP only. Real encoder/teacher/GPU parity remains pending.

Required trainer changes:
  1. Collate padded audio with attention masks, valid blank-padded labels,
     decoder blank prefix, explicit per-clip target lengths and encoder lengths.
  2. Pad cached original teacher features, verify each true T matches student,
     and exclude padding from preservation losses.
  3. Sample each replay clip's existing valid T/U+1 KL grid independently;
     initially use the unchanged preservation helper on valid row slices.
  4. Mean feature loss over valid frames per clip, then mean clips. Keep
     zero preservation terms for Turkish clips in mixed batches. B2 homogeneous
     group losses each divide by2; do not double replay weights in B4.
  5. Use shuffled length buckets within streams without changing language mix
     or excluding examples; preserve BatchNorm/eval and dropout policies.

Actual usable pools:2166TR and1553replay. Median T/U are27/17 for Turkish
and128/48 for replay. Over1000 seeded, unbucketed2TR+2replay quartets, median
joint-lattice padding factor is1.3814 for homogeneous B2 versus2.2248 for
mixed B4 (p90:1.6717/2.8900). These are padding estimates, not speed results.

GPU benchmark after the active trainer/final worker release the GPU:
  Compare B1x4, B2x2 and B4x1 in separate processes with identical TRAIN
  quartets, hashed model/trainable state, teacher cache and exact objectives.
  Use the median and high-lattice quartets recorded with full source IDs,
  WAV hashes and cache paths in batching_assessment.json. First compare losses,
  valid encoder features and all head/LoRA gradients without an Adam update.
  Then3-5 warm updates and at least20 synchronized timed updates per layout;
  report median/p90 ms/update, clips/s, realaudioseconds/s, allocated/reserved
  peaks, padding, utilization and any OOM. Measure compute-only with prefetched
  inputs separately from decode/STFT/cache/H2D end-to-end time. Use the same
  prefetch policy in all modes. Keep chunk16 and storage dtype unchanged.

The earlier profile_tdt.py timed a cold B1x2 update and a different objective;
its4.57s is not evidence of batching speed. Adopt B2 orB4 only after actual
warmed throughput and16GB worst-case safety improve. Evaluate any later
quality experiment using development selection and unchanged retention gates.

Unprofiled hypothesis: the TDT anti-diagonal loop branches in Python on CUDA
tensor .any(), which can introduce CPU-GPU synchronization. Batching could
amortize this overhead; a profiler must establish its actual share first.
Official general guidance (not a speed measurement for this run):
https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
"""
(OUTPUT / "batching_plan.txt").write_text(plan, encoding="utf-8")
print(json.dumps({"assessment": str(destination), "plan": str(OUTPUT / "batching_plan.txt"),
                  "passed_cpu_joint_parity": True, "gpu_used": False,
                  "representative_train_clips_cache_checked": 8}, indent=2))
