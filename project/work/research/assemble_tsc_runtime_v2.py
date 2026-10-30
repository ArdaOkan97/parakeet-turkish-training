"""Create NEW TSC v2 routes after explicit worker/policy/quarantine source freeze.

Standard-library source assembly only. It neither reads training data/checkpoints
nor loads ML modules. Original v1 routing stays byte-for-byte unchanged.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
CHAIN_SHA = "1a8294aa034e37d85f65fb66d4725ffe39432169be1dcaee9a77307f7465c220"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    chain_path = OUT / "TSC_runtime_chain_pins_v1.json"
    assert sha(chain_path) == CHAIN_SHA
    old = json.loads(chain_path.read_text(encoding="utf-8"))["source_sha256"]
    for name, digest in old.items():
        assert sha(OUT / name) == digest, name
    worker_name = "train_lora_batched_tsc_v2.py"
    dependencies = {worker_name: args.worker_sha, "tsc_training_inputs_v2.py": args.input_helper_sha,
                    "tsc_supplemental_data_policy_v2.py": args.supplement_policy_sha,
                    "prepare_TSC_quarantine_v2.py": args.quarantine_sha,
                    "prepare_TSC_TRAIN_v1.py": old["prepare_TSC_TRAIN_v1.py"]}
    for name, digest in dependencies.items():
        assert sha(OUT / name) == digest, name
    tree = ast.parse((OUT / worker_name).read_text(encoding="utf-8"))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    sync = [node for node in ast.walk(run) if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "torch.cuda.synchronize"]
    assert len(sync) == 1 and not sync[0].args and not sync[0].keywords
    line = sync[0].lineno
    stems = ("cuda_cache_caller", "launch_with_io_retry", "launch_with_cache_reclaim",
             "launch_with_memory_guard", "launch_native_training")
    mapping = {stem + "_tsc_v1.py": stem + "_tsc_v2.py" for stem in stems}
    exact_name = "launch_with_memory_guard_TSC_resume_v2.py"
    assert all(not (OUT / name).exists() for name in (*mapping.values(), exact_name)), "Never overwrite any route"
    created = {}
    substitutions = {old["train_lora_batched_tsc_v1.py"]: args.worker_sha,
                     old["tsc_training_inputs_v1.py"]: args.input_helper_sha,
                     old["tsc_supplemental_data_policy_v1.py"]: args.supplement_policy_sha}
    for original_name, new_name in mapping.items():
        source = (OUT / original_name).read_text(encoding="utf-8")
        source = source.replace("_tsc_v1", "_tsc_v2")
        source = source.replace("tsc_training_inputs_v1", "tsc_training_inputs_v2")
        source = source.replace("tsc_supplemental_data_policy_v1", "tsc_supplemental_data_policy_v2")
        source = source.replace("lineno != 545", f"lineno != {line}").replace("return 545", f"return {line}")
        source = source.replace("run:545", f"run:{line}")
        for old_sha, new_sha in substitutions.items():
            source = source.replace(old_sha, new_sha)
        if original_name == "launch_with_memory_guard_tsc_v1.py":
            anchor = repr(("prepare_TSC_TRAIN_v1.py", old["prepare_TSC_TRAIN_v1.py"]))
            assert source.count(anchor) == 1
            source = source.replace(anchor, anchor + ",\n            " + repr(("prepare_TSC_quarantine_v2.py", args.quarantine_sha))
                + ",\n            " + repr(("tsc_supplemental_data_policy_v1.py", "PARENT_TSC_POLICY_ORIGINAL_SHA")))
            needle = "'TSC_importer_sha256': '" + old["prepare_TSC_TRAIN_v1.py"] + "',"
            assert source.count(needle) == 1
            source = source.replace(needle, needle + "\n               'TSC_quarantine_preparer_sha256': '" + args.quarantine_sha + "',"
                + "\n               'TSC_parent_supplemental_policy_sha256': 'PARENT_TSC_POLICY_ORIGINAL_SHA',")
        source = source.replace("PARENT_TSC_POLICY_ORIGINAL_SHA", old["tsc_supplemental_data_policy_v1.py"])
        compile(source, str(OUT / new_name), "exec")
        with (OUT / new_name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(source)
        digest = sha(OUT / new_name)
        created[new_name] = digest
        substitutions[old[original_name]] = digest
    exact = (OUT / "launch_with_memory_guard_tsc_v2.py").read_text(encoding="utf-8")
    exact = exact.replace('"""', '"""SAME-run exact admission for quarantined TSC v2; ', 1)
    needle = "'--tr-frame-metadata-proof', '--data-v3-proof', '--resume-weights'):"
    assert exact.count(needle) == 1
    exact = exact.replace(needle, "'--tr-frame-metadata-proof', '--data-v3-proof', '--resume-exact'):")
    needle = "if '--resume-exact' in values or '--resume-runtime-migration' in values:"
    assert exact.count(needle) == 1
    exact = exact.replace(needle, "if '--resume-weights' in values or '--resume-runtime-migration' in values:")
    exact = exact.replace("This isolated initial TSC route requires fresh Adam/sampler from qualifying BEST",
                          "SAME-run TSC v2 route admits exact full state only; no warm weights or migration")
    exact = exact.replace("memory_guard_training_launcher_tsc_v2", "memory_guard_training_launcher_TSC_resume_v2")
    compile(exact, str(OUT / exact_name), "exec")
    with (OUT / exact_name).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(exact)
    for name, digest in old.items():
        assert sha(OUT / name) == digest, name
    print(json.dumps({"new_initial_source_sha256": created, "exact_memory": {"name": exact_name, "sha256": sha(OUT / exact_name)},
                      "worker_dependencies": dependencies, "caller_line": line, "all_v1_templates_unchanged": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("worker-sha", "input-helper-sha", "supplement-policy-sha", "quarantine-sha"):
        parser.add_argument("--" + name, required=True)
    main(parser.parse_args())
