"""Build the bounded four-source TRAIN carrier after actual candidate admission."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
import argparse, ctypes, hashlib, json, os, sys, threading, time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

def main(args):
    from lazy_training_rows_v1 import reopen_approved_TR_rows
    from expanded_training_rows_v1 import prepare_approved_expanded_TR_rows, record
    from resource_watchdog_v2 import sample_memory
    from tokenizers import Tokenizer
    assert "torch" not in sys.modules and not args.output_dir.exists() and not args.report.exists()
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.GetCurrentProcess.restype = ctypes.c_void_p
    api.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    assert api.SetPriorityClass(api.GetCurrentProcess(), 0x4000)
    config = json.loads(args.source_config.read_text(encoding="utf-8-sig"))
    tokenizer_path = HERE / "checkpoint/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    semantic = hashlib.sha256(json.dumps(json.loads(tokenizer.to_str()), ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert semantic == "b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75"
    assert tokenizer.get_vocab_size() == 8199 and tokenizer.token_to_id("<blank>") == 8192
    source = SimpleNamespace(**config)
    original, _ = reopen_approved_TR_rows(source, config["indexed_tr_directory"], config["public_turkish_data_audit"], cache_size=256)
    started = time.monotonic(); stop = threading.Event(); peak = [0]
    def observe():
        while not stop.is_set():
            value = sample_memory(os.getpid(), "bounded_expanded_TRAIN_index")
            peak[0] = max(peak[0], value["worker_private_memory"].get("private_bytes", 0))
            stop.wait(1)
    thread = threading.Thread(target=observe, daemon=True); thread.start()
    index = None
    try:
        index = prepare_approved_expanded_TR_rows(original, args.output_dir,
            encode=lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
            preparation_path=HERE / "WorldSpeech_TR_TRAIN_audio_candidate_preparation_v1.json",
            verification_path=HERE / "WorldSpeech_complete_candidate_CPU_independent_verification_v1.json",
            sanity_review_path=HERE / "WorldSpeech_TRAIN_sanity64_actual_review_v1.json")
        result = {"kind": "actual_expanded_TRAIN_index_preparation_v1", "passed": True,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
            "source_config": record(args.source_config), "tokenizer": record(tokenizer_path),
            "exact_tokenizer_semantic_sha256": semantic, "helper": record(HERE / "expanded_training_rows_v1.py"),
            "index_provenance": record(args.output_dir / "index_provenance.json"),
            "rows": len(index), "source_group_counts": index.receipt["source_group_counts"],
            "hours": index.receipt["hours"], "max_full_joint": index.receipt["max_full_joint"],
            "metadata_decoded_cache_rows_cap": 256, "original_source_rows_unchanged": True,
            "source_data_eligible_for_controlled_trial": True, "native_training_memory_fit_proved": False,
            "Torch_imported": False, "GPU_used": False, "TEST_predictions_used": False,
            "elapsed_seconds": time.monotonic() - started}
    finally:
        if index is not None: index.close()
        original.close(); stop.set(); thread.join(3)
    result["sampled_peak_owned_private_bytes"] = peak[0]
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "report": record(args.report), "index": result["index_provenance"],
                      "rows": result["rows"], "hours": result["hours"], "sampled_peak_owned_private_bytes": peak[0]}), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=ROOT / "work/closed-sources/TSC_uniform_TR_token_population43238_closed_v1/run_config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "work/data/TR_expanded_WorldSpeech_index_v1")
    parser.add_argument("--report", type=Path, default=HERE / "expanded_TRAIN_index_actual_CPU_preparation_v1.json")
    main(parser.parse_args())
