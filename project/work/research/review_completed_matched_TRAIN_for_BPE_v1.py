"""Bounded metadata review AFTER both actual 65,948 closures; no checkpoint load."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/"outputs/parakeet_turkish"
sys.path.insert(0, str(SOURCE))
from tsc_bpe_full_decoder_transition_v1 import (
    REVIEW_KIND, require, record, read, checked, validate_scores, validate_study_review,
)


def history_prefix_sha(path, count):
    """One bounded row at a time, canonical array digest identical to checkpoint."""
    decoder = json.JSONDecoder(parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    digest = hashlib.sha256(b"[")
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    buffer, position, number, expect = "", 0, 0, "["
    with Path(path).open(encoding="utf-8") as stream:
        eof = False
        while number < count:
            while position < len(buffer) and buffer[position].isspace(): position += 1
            if position >= len(buffer):
                piece = stream.read(65536)
                require(piece != "", "Truncated source history before chosen checkpoint")
                buffer, position = buffer[position:]+piece, 0
                continue
            if expect:
                require(buffer[position] == expect, "Invalid canonical source history separator")
                position += 1; expect = None; continue
            try: row, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                piece = stream.read(65536)
                require(piece != "", "Truncated source history object")
                buffer, position = buffer[position:]+piece, 0
                require(len(buffer) <= 4194304, "Bounded single source history record required")
                continue
            number += 1
            require(isinstance(row, dict) and row.get("step") == number,
                    "Actual selected checkpoint history must be contiguous 1..H")
            if number > 1: digest.update(b",")
            for part in encoder.iterencode(row): digest.update(part.encode("utf-8"))
            position = end; expect = ","
    digest.update(b"]")
    return digest.hexdigest()


def paired_spoken_diagnostic(values):
    """Same saved fixed50, same frozen transform; never changes winner gates."""
    import importlib.util
    normalizer = ROOT/"work/research/turkish_normalization_v1/evaluate.py"
    normalizer_record = record(normalizer)
    require(normalizer_record["sha256"] == "52545e91070c9ab6dc80bd27760af278921d6cb5f9eb56f0d4943824c1932323",
            "Separately validated spoken-form diagnostic normalizer changed")
    spec = importlib.util.spec_from_file_location("_matched_TRAIN_spoken_diagnostic", normalizer)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    from metrics import normalize_text
    fixed_references, result = None, {}
    for arm, value in values.items():
        result[arm] = {}
        for role, key in (("qualified_BEST", "best_DEV_summary"), ("final", "final_DEV_summary")):
            summary = value["artifacts"][key]
            predictions = Path(summary["original"]["path"]).parent/"validation_tr.jsonl"
            rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line]
            require(len(rows) == 50, "Paired diagnostic requires the actual complete fixed50 predictions")
            references = [row["reference"] for row in rows]
            if fixed_references is None: fixed_references = references
            require(references == fixed_references, "Paired diagnostic fixed references/order differ")
            hypotheses = [row["hypothesis"] for row in rows]
            def counts(transform):
                score = module.process_words([transform(v) for v in references], [transform(v) for v in hypotheses])
                s, d, i = score.substitutions, score.deletions, score.insertions
                n = sum(len(row) for row in score.references)
                return {"S": s, "D": d, "I": i, "word_errors": s+d+i, "reference_words": n, "wer": (s+d+i)/n}
            official, spoken = counts(lambda text: normalize_text(text, "tr")), counts(module.spoken)
            saved = read(summary["archived"])["validation_tr"]["asr"]
            require(official["reference_words"] == saved["reference_words"] == 825
                    and official["word_errors"] == saved["word_errors"], "Official diagnostic SDI differs from frozen scorer")
            result[arm][role] = {"predictions": record(predictions), "summary": summary["archived"],
                                "official": official, "spoken_form_diagnostic_only": spoken}
    return {"normalizer": normalizer_record, "systems": result, "same_fixed_references_and_transform": True,
            "original_scorer_six_language_gates_and_source_choice_unchanged": True,
            "model_inference_or_TEST_used": False}


def main(args):
    require(not args.output.exists(), "Never replace a completed study source decision")
    values = {arm: json.loads(path.read_text(encoding="utf-8"))
              for arm,path in (("expanded", args.expanded_preservation), ("base", args.base_preservation))}
    for arm, value in values.items():
        require(value.get("passed") is True and value.get("data_arm") == arm
                and value.get("completed_step") == 65948 and value["actual_owned_exit"]["exit_code"] == 0
                and value["native_child_exit_code"] == 0 and value["native_sources_unchanged"] is True
                and value["stop_category"] == "configured_steps_complete"
                and value["data_expansion_epoch"]["full_pass_complete"] is True,
                "Both actual complete matched arms required; live preparation cannot choose a source")
        counts, qualifies = validate_scores(read(value["artifacts"]["best_DEV_summary"]["archived"]))
        require(qualifies and counts["tr"]/825 == value["selected_qualified_BEST_TR_WER"], "Saved qualified BEST mismatch")
    # Tie rule retains expanded reference; it is explicit, not a gain claim.
    winner = min(values, key=lambda arm: (values[arm]["selected_qualified_BEST_TR_WER"], arm != "expanded"))
    chosen = values[winner]; artifacts = chosen["artifacts"]
    step = chosen["selected_qualified_BEST_step"]
    dev_folder = Path(artifacts["best_DEV_summary"]["original"]["path"]).parent
    dev = {"summary": artifacts["best_DEV_summary"]["archived"],
           "metadata": record(dev_folder/"metadata.json"), "metadata_pt": record(dev_folder/"metadata_pt.json"),
           **{lang: record(dev_folder/("validation_"+lang+".jsonl")) for lang in ("tr","en","es","it","fr","de","pt")}}
    config = read(artifacts["run_config"]["archived"])
    pool = Path(artifacts["run_config"]["original"]["path"]).parent/"training_pool.json"
    history_path = checked(artifacts["training_history"]["archived"])
    review = {"kind": REVIEW_KIND, "passed": True, "actual_both_matched_arms_complete": True,
        "completed_arms": {"base": {"preservation": record(args.base_preservation)},
                           "expanded": {"preservation": record(args.expanded_preservation)}},
        "selected": {"data_arm": winner, "run_id": chosen["run_id"], "step": step,
            "checkpoint": artifacts["best_checkpoint"]["archived"], "DEV": dev,
            "run_config": artifacts["run_config"]["archived"], "training_pool": record(pool),
            "history_prefix_sha256": history_prefix_sha(history_path, step)},
        "official_qualified_BEST_TR_counts": {arm: round(v["selected_qualified_BEST_TR_WER"]*825) for arm,v in values.items()},
        "official_final_TR_counts": {arm: v["counts"]["tr"]["word_errors"] for arm,v in values.items()},
        "paired_spoken_form_diagnostic": paired_spoken_diagnostic(values),
        "source_choice_rule": "Minimum saved official Turkish errors among strict all-six-qualified BEST; tie retains expanded",
        "comparison_limits": ["One branch per arm and 50 fixed Turkish DEV clips; no significance or causal attribution claimed",
            "Equal optimizer/replay budget and entry state; TR pools, audio/token exposure and compute differ",
            "Spoken-form diagnostic is separate; original selection and six-language gates remain frozen"],
        "next_family": "BPE512 full decoder, frozen encoder in both arms; 2,000 updates each; candidate first",
        "no_checkpoint_payload_loaded_or_model_state_mixed": True, "TEST_used": False}
    validate_study_review(review)
    from atomic_io_retry_v1 import write_json
    write_json(args.output, review)
    print(json.dumps({"passed": True, "review": record(args.output), "selected_arm": winner,
                      "source_step": step, "TR_errors": review["official_qualified_BEST_TR_counts"][winner]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--expanded-preservation", type=Path, default=SOURCE/"WorldSpeech_expanded_actual_CLOSED65948_preservation_v1.json")
    p.add_argument("--base-preservation", type=Path, default=SOURCE/"WorldSpeech_base_actual_CLOSED65948_preservation_v1.json")
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
