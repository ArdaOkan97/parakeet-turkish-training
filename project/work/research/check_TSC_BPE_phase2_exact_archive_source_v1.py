"""Pure metadata/header and source-order checks, never invoke archive creation."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
path = OUT / "record_TSC_BPE_phase2_exact_source_v1.py"
spec = importlib.util.spec_from_file_location("phase2_archive_source", path)
writer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(writer)


def main():
    checks, rejects = 0, 0
    def check(value):
        nonlocal checks
        assert value
        checks += 1
    def refuse(value):
        nonlocal checks, rejects
        try: writer.validate_closure_header(value)
        except ValueError: checks += 1; rejects += 1
        else: raise AssertionError("Invalid boundary header admitted")
    header = {"status": "train_window_ended", "child_exit_code": 0, "resource_review_required": False,
        "phase": writer.PHASE, "stop_category": "phase_bound", "finished_at_utc": "2026-10-11T00:00:00Z",
        "actual_runtime_completion": {"status": "test_header_only"}, "smoke_updates": None,
        "run_id": "synthetic_header_only_no_artifacts"}
    check(writer.validate_closure_header(header) == header["run_id"])
    budget = copy.deepcopy(header); budget["stop_category"] = "budget"
    check(writer.validate_closure_header(budget) == header["run_id"])
    for field, wrong in (("status", "running"), ("child_exit_code", 1), ("resource_review_required", True),
                         ("phase", "old_phase"), ("stop_category", "retention"), ("stop_category", "resource"),
                         ("stop_category", "manual"), ("stop_category", "plateau"), ("finished_at_utc", None),
                         ("actual_runtime_completion", None), ("smoke_updates", 1), ("run_id", "../outside")):
        changed = copy.deepcopy(header); changed[field] = wrong; refuse(changed)
    # Exercise real metadata-first rejection with injected synthetic JSON only.
    # A call into any hashing, filesystem copy or tensor reader fails this test.
    old_read, old_record = writer.read, writer.record
    running = copy.deepcopy(header); running["status"] = "running"
    writer.read = lambda unused: running
    writer.record = lambda unused: (_ for _ in ()).throw(AssertionError("No hashing is permitted for a running source"))
    try:
        try: writer.inspect_closed(ROOT / "work" / "source_only_fixture_no_files")
        except ValueError: checks += 1; rejects += 1
        else: raise AssertionError("Running source inspect accepted")
    finally: writer.read, writer.record = old_read, old_record
    check(writer.CHECKPOINT_BYTE_READS == 0)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top = [name.name for node in tree.body if isinstance(node, ast.Import) for name in node.names]
    check(not set(top) & {"torch", "transformers", "nemo", "numpy", "soundfile"})
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_receipt")
    source = ast.unparse(function)
    check(source.index("inspect_closed(args.source_loop)") < source.index("original = "))
    check(source.index("observed >= finished") < source.index("original = "))
    check(source.index("n < total") < source.index("original = "))
    check("archived[name] = original[name]" in source and "archived_immutable_scientific_tokens=backups" in source)
    check(writer.IMMUTABLE_TOKENS == ("original_native_ready", "initial_source_equivalence"))
    check("target_run_id': controller['run_id']" in source)
    check("torch" not in sys.modules and "nemo" not in sys.modules)
    report = {"schema": 1, "kind": "inactive_TSC_BPE_phase2_exact_archive_source_verification_v1",
        "passed": True, "assertions": checks, "rejections": rejects,
        "writer": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "fixture": {"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "checkpoint_byte_reads": writer.CHECKPOINT_BYTE_READS, "live_JSON_or_state_reads": False,
        "archive_creator_executed": False, "archive_or_receipt_created": False, "ML_or_GPU_executed": False,
        "scope": "Pure synthetic closure-header refusal and source order/API checks only",
        "future_execution_gate": "Actual ROOT-owned CLOSED0 budget/phase-bound plus original proof/runtime and complete two-Adam tensor validation"}
    output = OUT / ("TSC_BPE_phase2_exact_archive_source_CPU_verification_"+str(time.time_ns())+".json")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write("\n")
    print(json.dumps({"report": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "assertions": checks, "rejections": rejects, "writer_sha256": report["writer"]["sha256"]}))


if __name__ == "__main__": main()
