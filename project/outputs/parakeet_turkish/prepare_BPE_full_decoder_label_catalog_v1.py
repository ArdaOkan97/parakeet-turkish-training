"""TRAIN target preparation only: bounded indexed rows, tokenizers, no ML/audio/GPU."""
from __future__ import annotations
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[1]


def main(args):
    if os.name == "nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Explicit existing local binary dependency; no transformers/model import.
    packages = ROOT / "work/training-env/Lib/site-packages"
    sys.path.insert(0, str(packages))
    from tokenizers import Tokenizer
    from bpe_training_label_catalog_v1 import prepare_label_catalog, record, scoped, write_json
    from tsc_bpe512_training_inputs_v1 import (
        BASELINE_PATH, BASELINE_FILE_SHA, CANDIDATE_PATH, CANDIDATE_FILE_SHA,
        validate_tokenizer_geometry,
    )
    from lazy_training_rows_v1 import IndexedTrainingRows
    from expanded_training_rows_v1 import reopen_approved_expanded_TR_rows

    output = scoped(args.output_directory)
    config = Path(os.environ.get("ASR_STORAGE_CONFIG", ""))
    expected_config = Path(r"E:\gpu_training_scratchpad\storage_config.json")
    if config.resolve() != expected_config.resolve():
        raise ValueError("Apply the designated E: use_storage.ps1 before this new preparation job")
    storage = json.loads(config.read_text(encoding="utf-8"))
    if storage.get("storage_root") != str(expected_config.parent) or storage.get("allow_fallback_outside_root") is not False:
        raise ValueError("Exact no-fallback user storage policy required")
    if record(BASELINE_PATH)["sha256"] != BASELINE_FILE_SHA or record(CANDIDATE_PATH)["sha256"] != CANDIDATE_FILE_SHA:
        raise ValueError("Frozen TRAIN-built tokenizer assets changed")
    validate_tokenizer_geometry(json.loads(BASELINE_PATH.read_text(encoding="utf-8")),
                                json.loads(CANDIDATE_PATH.read_text(encoding="utf-8")))
    baseline, candidate = Tokenizer.from_file(str(BASELINE_PATH)), Tokenizer.from_file(str(CANDIDATE_PATH))
    rows = (reopen_approved_expanded_TR_rows(args.source_index, cache_size=256)
            if args.carrier_kind == "expanded" else IndexedTrainingRows(args.source_index, cache_size=256))
    expected = 352768 if args.carrier_kind == "expanded" else 253798
    if len(rows) != expected:
        raise ValueError("Exact admitted original/expanded source pool required")
    before = time.perf_counter()
    try:
        result = prepare_label_catalog(rows, output, baseline, candidate,
            source_catalog_record=record(Path(args.source_index) / "index_provenance.json"),
            baseline_tokenizer_record=record(BASELINE_PATH), candidate_tokenizer_record=record(CANDIDATE_PATH))
        if "torch" in sys.modules or "transformers" in sys.modules:
            raise RuntimeError("This preparation must not import Torch or transformers")
        receipt = {"kind": "BPE_full_decoder_actual_TRAIN_target_preparation_v1", "passed": True,
                   "metadata_preparation_only_not_GPU_admission": True,
                   "source_arm": args.carrier_kind, "rows": expected,
                   "target_catalog": record(output / "label_catalog_provenance.json"),
                   "source_target_tokens": result["source_target_tokens"],
                   "candidate_target_tokens": result["candidate_target_tokens"],
                   "maximum_full_joint_bytes": result["maximum_full_joint_bytes"],
                   "elapsed_seconds": time.perf_counter() - before,
                   "storage_config": record(config), "new_storage_root": str(expected_config.parent),
                   "existing_source_index_read_only": True, "audio_model_Torch_GPU_TEST_used": False,
                   "selected_training_source_or_quality_gain_claimed": False}
        report = output / "actual_preparation_receipt.json"
        write_json(report, receipt)
        print(json.dumps({"passed": True, "receipt": record(report), "rows": expected,
                          "source_tokens": result["source_target_tokens"], "BPE_tokens": result["candidate_target_tokens"],
                          "maximum_full_joint_bytes": result["maximum_full_joint_bytes"]}))
    finally:
        rows.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source-index", type=Path, required=True)
    p.add_argument("--carrier-kind", choices=("base", "expanded"), required=True)
    p.add_argument("--output-directory", type=Path, required=True)
    main(p.parse_args())
