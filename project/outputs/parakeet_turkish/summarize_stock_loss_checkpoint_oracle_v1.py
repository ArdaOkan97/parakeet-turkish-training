"""Read-only, stdlib-only interpretation of a completed actual native-loss oracle.

Rechecks the saved result schema and gates. No inference, model load, raw
gradient recomputation, performance/adoption claim, or existing output rewrite.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ORACLE_SHA = "4c4f3a83d7c6e9ce4262292f78e45edf0d501eabd4862d91b4ec30868a4f519d"
INPUT_SHA = "8c3481e742f7df332102f718bc632314cd8869896ef292b28b2e0654bbfe1259"
HOOK_SHA = "ed34b4cc97437a4d8694bc94a3522c565cc329ea40f4aac80112fd1a9abcab57"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def interpret(report):
    checks = 0
    def check(ok, label):
        nonlocal checks
        require(ok, label)
        checks += 1
    check(report.get("schema") == 1 and report.get("kind") == "actual_TRAIN_stock_NeMo_loss_checkpoint_oracle_v1", "Actual oracle schema differs")
    check(report.get("passed") is True and report.get("status") == "complete"
          and report.get("native_kernel_executed") is True, "Oracle is not a completed passed native GPU measurement")
    check(report.get("code_sha256") == ORACLE_SHA and report.get("input_sha256") == INPUT_SHA
          and report.get("source_sha256", {}).get("stock_nemo_loss_checkpoint_v1.py") == HOOK_SHA, "Frozen source/input identities differ")
    check(report.get("input_shape") == [2,128,29,8204]
          and report.get("full_input_tensor_bytes") == math.prod(report["input_shape"]) * 4, "Full actual TRAIN tensors were changed")
    target_lengths = report.get("actual_target_lengths", [])
    check(len(target_lengths) == 2 and all(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 28 for v in target_lengths)
          and max(target_lengths) == 28, "Actual target lengths changed")
    provenance = report.get("input_provenance", {})
    check(provenance.get("split") == "train" and len(set(provenance.get("source_ids", []))) == 2
          and provenance.get("model_training") is False and provenance.get("dropout_disabled") is True, "Actual TRAIN capture scope differs")
    check(report.get("cap_bytes") == 8_000_000_000 and report.get("guard_policy") == {
        "moderate_commit_fraction": .9, "critical_commit_fraction": .95, "critical_headroom_bytes": 2_000_000_000,
        "moderate_consecutive_observations": 3, "minimum_fresh_interval_seconds": 5., "telemetry_stale_seconds": 15.}, "Allocator/Windows safety policy changed")
    check(report.get("ambient_rng_restored_finally") is True, "Oracle ambient RNG restoration missing")
    check(report.get("cases_completed") == len(report.get("cases", [])) == 8, "All eight actual cases must complete")
    expected = set(itertools.product(("fp32", "bf16"), ("TDT", "RNNT"), (None, "mean_volume")))
    observed = set()
    rows = []
    for case in report["cases"]:
        key = (case.get("dtype"), case.get("branch"), case.get("invocation_outer_reduction"))
        check(key in expected and key not in observed, "Duplicate/unexpected case")
        observed.add(key)
        check(case.get("sigma") == .02 and case.get("omega") == .1
              and case.get("seed") == (0 if key[1] == "TDT" else 31), "Released objective/branch seed changed")
        uniform = case.get("first_random_uniform")
        check(finite(uniform) and 0 <= uniform <= 1 and (uniform < .1) == (key[1] == "RNNT"), "Seed branch differs")
        check(case.get("passed") is True and case.get("maximum_absolute_output_delta") == 0, "Actual stock/checkpoint output differed")
        gradient = case.get("joint_gradient", {})
        check(gradient.get("all_joint_values") == math.prod(report["input_shape"])
              and gradient.get("finite") is True and gradient.get("bitwise_equal") is True
              and gradient.get("maximum_absolute_delta") == 0 and gradient.get("relative_l2") == 0,
              "Actual full-joint gradient equality missing")
        check(finite(gradient.get("baseline_l2")) and gradient["baseline_l2"] > 0
              and finite(gradient.get("cosine")) and gradient["cosine"] > .99999999, "Gradient norm/cosine invalid")
        check(finite(gradient.get("duration_gradient_l2"))
              and (gradient["duration_gradient_l2"] == 0) == (key[1] == "RNNT"), "Measured duration gradients disagree with native branch")
        arms = case.get("arms", [])
        check(len(arms) == 2 and [a.get("wrapped") for a in arms] == [False, True], "Stock/wrapped arms missing")
        check(arms[0].get("output") == arms[1].get("output")
              and arms[0].get("objective") == arms[1].get("objective"), "Saved arm values disagree")
        for arm in arms:
            values = arm["output"] if isinstance(arm["output"], list) else [arm["output"]]
            check(len(values) == (2 if key[2] is None else 1) and all(finite(v) for v in values)
                  and finite(arm.get("objective")), "Loss output/reduction shape invalid")
            check(arm.get("Python_rng_advanced_once_forward_unchanged_backward") is True
                  and arm.get("Torch_CPU_CUDA_rng_unchanged") is True
                  and arm.get("outer_after_backward") == "mean_volume"
                  and arm.get("inner_after_backward") == "none", "RNG/reduction state restoration failed")
            check(finite(arm.get("peak_allocated_bytes")) and 0 < arm["peak_allocated_bytes"] <= 8_000_000_000
                  and finite(arm.get("peak_reserved_bytes")) and arm["peak_reserved_bytes"] >= arm["peak_allocated_bytes"], "Primitive GPU allocator values invalid")
        hook = arms[1].get("hook_identity", {})
        check(hook.get("source_sha256") == HOOK_SHA and hook.get("blank") == 8198
              and hook.get("durations") == [0,1,2,3,4] and hook.get("sigma") == .02 and hook.get("omega") == .1
              and hook.get("native_math_or_training_step_replaced") is False, "Checkpoint hook changed native math/geometry")
        statistics = hook.get("statistics", {})
        check(statistics.get("checkpointed_forward_calls") == 1
              and isinstance(statistics.get("recomputed_forward_calls"), int) and statistics["recomputed_forward_calls"] >= 1,
              "Native saved-tensor recomputation was not actually exercised")
        rows.append({"dtype": key[0], "branch": key[1], "outer_reduction": key[2], "seed": case["seed"],
                     "raw_or_reduced_loss": arms[0]["output"], "mean_volume_objective": arms[0]["objective"],
                     "loss_and_all_joint_gradients_bitwise_equal": True,
                     "stock_peak_allocated_gb": arms[0]["peak_allocated_bytes"] / 1e9,
                     "checkpoint_peak_allocated_gb": arms[1]["peak_allocated_bytes"] / 1e9})
    check(observed == expected, "Case coverage differs")
    for dtype, branch in itertools.product(("fp32", "bf16"), ("TDT", "RNNT")):
        raw = next(row for row in rows if (row["dtype"], row["branch"], row["outer_reduction"]) == (dtype, branch, None))
        reduced = next(row for row in rows if (row["dtype"], row["branch"], row["outer_reduction"]) == (dtype, branch, "mean_volume"))
        expected_mean = sum(raw["raw_or_reduced_loss"]) / sum(target_lengths)
        check(abs(raw["mean_volume_objective"] - expected_mean) <= 2e-5
              and abs(reduced["mean_volume_objective"] - expected_mean) <= 2e-5,
              "Released mean_volume does not match saved raw NLL/target volume")
    memory = report.get("memory_events", [])
    check(len(memory) >= 33 and all(event.get("resource_reason") is None for event in memory), "Resource observation/guard failure present")
    check(all(event.get("system_memory", {}).get("available") is True
              and event["system_memory"]["commit_fraction"] < .95
              and event["system_memory"]["commit_available_bytes"] >= 2e9 for event in memory), "Sampled global memory critical")
    return {"schema": 1, "kind": "stock_loss_checkpoint_actual_oracle_interpretation_v1", "passed": True,
            "checks": checks, "actual_case_count": len(rows), "source_ids": provenance["source_ids"], "cases": rows,
            "conclusion": "On the complete two captured TRAIN tensors, stock and checkpointed native loss and every joint gradient match bitwise across both dtypes, both omega branches and both outer reductions; RNG and mutable reduction state are preserved.",
            "next_step": "A bounded whole-update stock-versus-checkpoint benchmark from the same saved native checkpoint/Adam/Cosine and unseen TRAIN prefix can test speed and memory. B1/B2/B4/B8 mean_volume grouping changes must be stated separately.",
            "not_demonstrated": ["Whole-model or whole-update gradient equivalence", "Faster training", "Worst-long-clip memory safety", "Better Turkish WER or multilingual retention"],
            "memory_scope": "Primitive-loss process allocator peaks only; sparse before/after Windows observations. They do not establish whole-update or continuous memory safety.",
            "maximum_sampled_system_commit_fraction": max(event["system_memory"]["commit_fraction"] for event in memory),
            "minimum_sampled_system_headroom_bytes": min(event["system_memory"]["commit_available_bytes"] for event in memory)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.input.resolve().is_relative_to(ROOT) and args.output.resolve().is_relative_to(ROOT)
            and not args.output.exists(), "Workspace input and new output required")
    report = interpret(json.loads(args.input.read_text(encoding="utf-8")))
    report.update(input_path=str(args.input.resolve()), input_sha256=digest(args.input),
                  reader_sha256=digest(__file__), oracle_code_sha256=ORACLE_SHA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": report["passed"], "checks": report["checks"], "actual_case_count": report["actual_case_count"]}))


if __name__ == "__main__":
    main()
