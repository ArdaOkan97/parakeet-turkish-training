"""Focused stdlib preparation/negative checks; never imports ML or runs CUDA."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
path = OUT / "verify_native_tdt_other_bucket_gpu_v1.py"
spec = importlib.util.spec_from_file_location("other_comparison_source", path)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)
sys.path.insert(0, str(OUT))
import native_tdt_other_bucket_v1 as prototype


def main():
    checks, rejects = 0, 0
    def check(value):
        nonlocal checks
        assert value
        checks += 1
    def refuse(args):
        nonlocal checks, rejects
        try:
            harness.validate_arguments(harness.parser().parse_args(args))
        except ValueError:
            checks += 1
            rejects += 1
        else:
            raise AssertionError("Invalid comparison arguments accepted")
    stem = "TDT_OTHER_GPU_harness_source_preparation_"+str(time.time_ns())
    prepared = OUT / (stem+".json")
    arguments = harness.parser().parse_args(["--source-only", "--output", str(prepared)])
    args = harness.validate_arguments(arguments)
    check(args.max_seconds == 180 and args.stress_frames == 96 and args.stress_targets == 32)
    cases = harness.case_specs(args)
    check(len(cases) == 5 and sum(case["V"] == 8711 for case in cases) == 3)
    for case in cases:
        plan = prototype.retained_plan(case["labels"], case["V"], case["blank"], vocab_chunk_size=case["vocab_chunk"])
        check(len(plan["remapped_target_ids"]) == len(case["labels"]))
        check(plan["all_non_target_nonblank_mass_included"] and not plan["decoder_ids_or_target_length_changed"])
    check(cases[0]["labels"][0] == cases[0]["labels"][1] and cases[0]["blank"] != cases[0]["V"]-1)
    dominant = cases[1]["dominant_id"]
    check(8199 <= dominant < 8711 and dominant not in cases[1]["labels"])
    check(prototype.retained_plan(cases[2]["labels"], 4, 1, vocab_chunk_size=2)["other_index"] is None)
    check(set(harness.source_records()) == set(harness.PINS) | set(harness.OFFICIAL))
    base = ["--source-only", "--output", str(prepared)]
    for changed in (["--max-seconds", "181"], ["--max-seconds", "nan"], ["--stress-frames", "129"],
                    ["--stress-targets", "65"], ["--gpu-window-authorized"]):
        refuse(base+changed)
    refuse(["--output", str(prepared)])
    refuse(["--source-only", "--output", str(ROOT.parent / "outside_OTHER_report.json")])
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = [name.name for node in top_imports if isinstance(node, ast.Import) for name in node.names]
    check(not set(names) & {"torch", "nemo", "transformers", "numba"})
    callback = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "callback")
    call = next(node for node in ast.walk(callback) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == "loss")
    check(len(call.args) == 12)
    arm = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_arm")
    arm_text = ast.unparse(arm)
    check("compressed_clip_loss" in arm_text and "chunked_tdt_loss" in arm_text and "value.sum().backward()" in arm_text)
    compare = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "compare")
    compare_text = ast.unparse(compare)
    check(all(key in compare_text for key in ("OTHER_all_rows", "unused_new512_rows", "duration5_rows")))
    check(harness.main(args) == 0)
    document = json.loads(prepared.read_text(encoding="utf-8"))
    check(document["status"] == "source_only_prepared_GPU_unexecuted" and not document["native_kernel_executed"])
    check(document["actual_numerical_or_memory_pass"] is False and document["ML_imported"] is False)
    check(not any(name in sys.modules for name in ("torch", "nemo", "transformers", "numba.cuda")))
    refuse(base)
    report = {"schema": 1, "kind": "inactive_OTHER_GPU_harness_stdlib_preparation_verification_v1",
        "passed": True, "assertions": checks, "rejections": rejects,
        "harness": {"path": str(path), "sha256": harness.sha(path)},
        "fixture": {"path": str(Path(__file__).resolve()), "sha256": harness.sha(__file__)},
        "source_only_preparation": {"path": str(prepared), "sha256": harness.sha(prepared)},
        "scope": "Parser/source pins/actual source-only path/critical synthetic case and native callback structure only",
        "actual_Torch_autograd_native_CUDA_memory_timing": False,
        "real_model_weights_audio_data_checkpoint_reads": False,
        "unchanged_activated_sources": True,
        "pending": ["Actual ROOT-owned sole GPU run and NumPy/Numba kernel compilation",
                    "Full8711 EP/DP/head gradients, scalar-reference native comparison, sampled commit and allocator peaks",
                    "Separate actual BF16/full-model gradient and training memory/throughput adoption proof"]}
    destination = OUT / ("TDT_OTHER_GPU_harness_stdlib_verification_"+str(time.time_ns())+".json")
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report": str(destination), "sha256": harness.sha(destination), "checks": checks,
        "rejections": rejects, "harness_sha256": harness.sha(path)}))


if __name__ == "__main__":
    main()
