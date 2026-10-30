"""Create only NEW isolated routes from the frozen domain30 routing template.

This is standard-library source assembly, never an ML/data execution. Explicit
final worker/helper/policy hashes are required; frozen templates are untouched.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
WORKER = "train_lora_batched_tsc_v1.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    old = json.loads((OUT / "domain30_runtime_chain_pins_v1.json").read_text(encoding="utf-8"))["source_sha256"]
    for name, digest in old.items():
        assert sha(OUT / name) == digest, name
    dependencies = {WORKER: args.worker_sha, "tsc_training_inputs_v1.py": args.input_helper_sha,
                    "tsc_supplemental_data_policy_v1.py": args.supplement_policy_sha,
                    "prepare_TSC_TRAIN_v1.py": sha(OUT / "prepare_TSC_TRAIN_v1.py")}
    for name, digest in dependencies.items():
        assert sha(OUT / name) == digest, name
    tree = ast.parse((OUT / WORKER).read_text(encoding="utf-8"))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)
             and ast.unparse(node.func) == "torch.cuda.synchronize"]
    assert len(calls) == 1 and not calls[0].args and not calls[0].keywords
    line = calls[0].lineno
    mapping = {
        "cuda_cache_caller_domain30_v1.py": "cuda_cache_caller_tsc_v1.py",
        "launch_with_io_retry_domain30_v1.py": "launch_with_io_retry_tsc_v1.py",
        "launch_with_cache_reclaim_domain30_v1.py": "launch_with_cache_reclaim_tsc_v1.py",
        "launch_with_memory_guard_domain30_v1.py": "launch_with_memory_guard_tsc_v1.py",
        "launch_native_training_domain30_v1.py": "launch_native_training_tsc_v1.py",
    }
    assert all(not (OUT / name).exists() for name in mapping.values()), "Never overwrite a routing source"
    created = {}
    for old_name, new_name in mapping.items():
        source = (OUT / old_name).read_text(encoding="utf-8")
        source = source.replace("train_lora_batched_domain30_v1.py", WORKER).replace(old["train_lora_batched_domain30_v1.py"], args.worker_sha)
        source = source.replace("domain30_v1", "tsc_v1").replace("domain30", "TSC").replace("domain worker", "TSC worker").replace("domain provenance", "TSC provenance").replace("domain arm", "TSC arm")
        source = source.replace("lineno != 507", f"lineno != {line}").replace("return 507", f"return {line}").replace("run:507", f"run:{line}")
        for original, new_sha in created.items():
            source = source.replace(old[original], new_sha)
        if old_name.startswith("launch_with_memory_guard_"):
            start = source.index("            ('turkish_domain_sampler_v2.py'")
            end = source.index("]", start) + 1
            replacements = {name: digest for name, digest in dependencies.items() if name != WORKER}
            replacements["train_lora_batched_data_v11.py"] = old["train_lora_batched_data_v11.py"]
            source = source[:start] + ",\n".join("            " + repr((name, digest)) for name, digest in replacements.items()) + "]" + source[end:]
            source = source.replace("def creation_identity(pid):", '''def tsc_argument_records(arguments):
    """Require explicit isolated paths; do not inherit the old worker's defaults."""
    values = list(arguments)
    if values[:1] == ['--']:
        values = values[1:]
    records = {}
    for flag in ('--tsc-input-proof', '--tsc-training-approval', '--source-closure',
                 '--tr-manifest', '--replay-manifest', '--tr-frame-metadata',
                 '--tr-frame-metadata-proof', '--data-v3-proof', '--resume-weights'):
        if values.count(flag) != 1 or values.index(flag)+1 >= len(values):
            raise ValueError('Exactly one explicit isolated TSC argument required: '+flag)
        path = Path(values[values.index(flag)+1]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError('Approved existing workspace TSC asset required: '+flag)
        records[flag.removeprefix('--').replace('-', '_')] = {'path': str(path), 'sha256': file_sha(path)}
    if '--resume-exact' in values or '--resume-runtime-migration' in values:
        raise ValueError('This isolated initial TSC route requires fresh Adam/sampler from qualifying BEST')
    return records


def creation_identity(pid):''')
            source = source.replace("run_id, argument_sha = argument_identity(args.arguments)",
                "run_id, argument_sha = argument_identity(args.arguments)\n    tsc_assets = tsc_argument_records(args.arguments)")
            source = source.replace("'native_cleanup_adapter_sha256': NATIVE_ADAPTER_SHA256,",
                "'native_cleanup_adapter_sha256': NATIVE_ADAPTER_SHA256,\n               'TSC_inputs_helper_sha256': '" + args.input_helper_sha + "',\n               'TSC_supplemental_policy_sha256': '" + args.supplement_policy_sha + "',\n               'TSC_importer_sha256': '" + dependencies["prepare_TSC_TRAIN_v1.py"] + "',\n               'TSC_explicit_asset_records': tsc_assets,")
        if old_name.startswith("launch_native_training_"):
            domain_flags = "for flag in ('--resume-weights','--source-closure'):\n  scoped(option(worker,flag))\n require(option(worker,'--sampling-mode') in ('TSC','uniform'),'Explicit isolated TSC or independent-replay uniform mode required')\n int(option(worker,'--replay-seed'))"
            assert domain_flags in source
            source = source.replace(domain_flags, "for flag in ('--resume-weights','--source-closure','--tsc-input-proof','--tsc-training-approval',\n              '--tr-manifest','--replay-manifest','--tr-frame-metadata','--tr-frame-metadata-proof','--data-v3-proof'):\n  path=scoped(option(worker,flag));require(path.is_file(),'Explicit approved existing TSC asset required: '+flag)\n require(float(option(worker,'--epochs'))==1. and int(option(worker,'--steps'))>0,'Explicit one-pass combined-pool horizon required')")
            source = source.replace("code={'launcher':sha(__file__),'trainer':sha(trainer),'memory_launcher':sha(wrapper),\n       'adapter':ADAPTER_SHA,'preload':PRELOAD_SHA}",
                "code={'launcher':sha(__file__),'trainer':sha(trainer),'memory_launcher':sha(wrapper),\n       'adapter':ADAPTER_SHA,'preload':PRELOAD_SHA}\n tsc_assets={flag.removeprefix('--').replace('-','_'):{'path':str(scoped(option(worker,flag))),'sha256':sha(scoped(option(worker,flag)))}\n   for flag in ('--tsc-input-proof','--tsc-training-approval','--source-closure','--tr-manifest','--replay-manifest','--tr-frame-metadata','--tr-frame-metadata-proof','--data-v3-proof','--resume-weights')}")
            source = source.replace("'worker_arguments_sha256':argument_sha(worker),\n   'declared_allocator_policy'", "'worker_arguments_sha256':argument_sha(worker),\n   'TSC_explicit_asset_records':tsc_assets,\n   'declared_allocator_policy'")
        compile(source, str(OUT / new_name), "exec")
        with (OUT / new_name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(source)
        created[old_name] = sha(OUT / new_name)
    for name, digest in old.items():
        assert sha(OUT / name) == digest, name
    print(json.dumps({"new_source_sha256": {mapping[name]: digest for name, digest in created.items()},
                      "worker_dependencies": dependencies, "caller_line": line, "templates_unchanged": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("worker-sha", "input-helper-sha", "supplement-policy-sha"):
        parser.add_argument("--" + flag, required=True)
    main(parser.parse_args())
