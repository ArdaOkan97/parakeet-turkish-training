"""Prepare a weights-only HF Parakeet warm start with 512 appended TR BPE pieces.

No model construction, training, or tokenizer activation occurs here. Torch is
imported only when conversion is explicitly requested. --prepare-only and
--self-test use the standard library. Future training must use a NEW optimizer,
the enlarged model/tokenizer, and ORIGINAL replay labels/teacher conditioning.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE = ROOT / "work/hf_turkish_bpe_candidate_512_v1"
OLD_VOCAB, NEW_VOCAB, BLANK, DURATIONS = 8199, 8711, 8192, 5
BIAS_PENALTY = 2.0
EXPANDED_KEYS = ("decoder.embedding.weight", "joint.head.weight", "joint.head.bias")
ASSET_SHA256 = {
    "tokenizer.json": "6c656b9dc5f85e3dd2c2abe16983908e5d2293da1b2ae096e92e1dc5551df327",
    "added_pieces.json": "94e660b0cab5c331edbdc114130857a62ce0df467172c23fc02a0d6efc35e6e4",
    "row_migration_plan.json": "ca066f6c4073b7175921d219571dc687c50deca631c12515c1cb34bcd5b5a710",
    "candidate_frozen_before_measurement.json": "b758465b5df283cf040ce83aa97e8c1b6a8aaf98a144b1667aea5f0696012af8",
}
SOURCE_TOKENIZER_SHA256 = "9c9f537bf7190ae2f0e30448a6915c4f607fa4d011b4f62b6f1849064ef8afcd"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build_donor_plan(original, candidate, additions):
    """Flatten ordered merges to original IDs, retaining leaf multiplicity."""
    old = original["model"]["vocab"]
    new = candidate["model"]["vocab"]
    old_size = len(old)
    require(original["model"]["type"] == candidate["model"]["type"] == "BPE", "Expected BPE models")
    require(set(old.values()) == set(range(old_size)), "Original IDs are not contiguous")
    require(set(new.values()) == set(range(old_size + len(additions))), "Candidate IDs are not contiguous")
    require(all(new.get(piece) == index for piece, index in old.items()), "An original piece changed ID")
    old_structure, new_structure = copy.deepcopy(original), copy.deepcopy(candidate)
    for value in (old_structure, new_structure):
        value["model"].pop("vocab")
        value["model"].pop("merges")
    require(old_structure == new_structure, "Tokenizer metadata outside vocabulary/merges changed")
    old_merges, new_merges = original["model"]["merges"], candidate["model"]["merges"]
    require(new_merges[:len(old_merges)] == old_merges, "Original merge prefix changed")
    require(len(new_merges) == len(old_merges) + len(additions), "Unexpected appended merge count")
    by_id = {index: piece for piece, index in new.items()}
    leaves = {index: [index] for index in range(old_size)}
    depths = {index: 0 for index in range(old_size)}
    result = []
    for offset, row in enumerate(additions):
        index = old_size + offset
        left, right = row["left_HF_id"], row["right_HF_id"]
        require(type(index) is int and type(left) is int and type(right) is int, "Noninteger merge IDs")
        require(row["HF_id"] == index and new.get(row["piece"]) == index, "Appended piece ID mismatch")
        require(0 <= left < index and 0 <= right < index, "Merge uses a future/self/invalid operand")
        require(by_id[left] == row["left"] and by_id[right] == row["right"], "Merge operand IDs differ from pieces")
        require(row["left"] + row["right"] == row["piece"], "Merge concatenation mismatch")
        require(row["HF_merge_rank"] == len(old_merges) + offset, "Merge rank mismatch")
        require(new_merges[len(old_merges) + offset] == [row["left"], row["right"]], "Appended merge differs from operands")
        donors = leaves[left] + leaves[right]
        require("".join(by_id[leaf] for leaf in donors) == row["piece"], "Flattened donors do not reconstruct piece")
        leaves[index] = donors
        depths[index] = 1 + max(depths[left], depths[right])
        counts = Counter(donors)
        result.append({
            "piece": row["piece"], "HF_id": index, "left_HF_id": left,
            "right_HF_id": right, "original_leaf_donor_ids": donors,
            "original_leaf_donor_counts": {str(key): count for key, count in sorted(counts.items())},
            "original_leaf_count": len(donors), "merge_depth": depths[index],
            "bias_penalty_applied_once": BIAS_PENALTY,
        })
    return result


def prepare(args):
    candidate_dir = Path(args.candidate_dir).resolve()
    asset_pins = {}
    for name, expected in ASSET_SHA256.items():
        path = candidate_dir / name
        actual = sha256(path)
        require(actual == expected, f"Candidate asset SHA mismatch: {name}")
        asset_pins[name] = {"path": str(path), "sha256": actual}
    freeze = read_json(candidate_dir / "candidate_frozen_before_measurement.json")
    require(freeze["fit_performed"] is False and freeze["DEV_or_TEST_text_read"] is False,
            "Candidate freeze is not TRAIN-only inert preparation")
    require(freeze["merges_read_from_prior_TRAIN_only_candidate"] is True, "TRAIN-only merge provenance missing")
    for name in ("tokenizer.json", "added_pieces.json", "row_migration_plan.json"):
        entries = [(path, value) for path, value in freeze["files_sha256"].items() if Path(path).name == name]
        require(len(entries) == 1 and entries[0][1] == ASSET_SHA256[name], "Frozen candidate pins disagree")
    source_tokenizer = Path(args.source_tokenizer).resolve()
    require(sha256(source_tokenizer) == SOURCE_TOKENIZER_SHA256, "Source tokenizer SHA mismatch")
    original, candidate = read_json(source_tokenizer), read_json(candidate_dir / "tokenizer.json")
    additions = read_json(candidate_dir / "added_pieces.json")
    require(len(original["model"]["vocab"]) == OLD_VOCAB and len(additions) == 512, "Unexpected vocabulary geometry")
    require(original["model"]["vocab"].get("<blank>") == BLANK, "Blank ID mismatch")
    for index, letter in enumerate(("ğ", "Ğ", "ı", "İ", "ş", "Ş"), 8193):
        require(original["model"]["vocab"].get(letter) == index, "Existing Turkish character ID mismatch")
    donors = build_donor_plan(original, candidate, additions)
    require(all(BLANK not in row["original_leaf_donor_ids"] for row in donors), "A normal appended piece uses blank as donor")
    migration = read_json(candidate_dir / "row_migration_plan.json")
    require(migration["original_HF_lexical_and_blank_and_six_rows_copied"] == [0, OLD_VOCAB]
            and migration["new_HF_piece_rows"] == [OLD_VOCAB, NEW_VOCAB]
            and migration["blank_token_id_unchanged"] == BLANK
            and migration["decoder_embedding_before_after"] == [OLD_VOCAB, NEW_VOCAB]
            and migration["joint_head_before_after"] == [OLD_VOCAB + DURATIONS, NEW_VOCAB + DURATIONS]
            and migration["duration_row_pairs"] == [[OLD_VOCAB + index, NEW_VOCAB + index] for index in range(DURATIONS)],
            "Migration row bounds disagree")
    source_path = Path(args.source_checkpoint).resolve()
    require(re.fullmatch(r"[0-9a-f]{64}", args.source_sha256) is not None, "Expected source SHA must be lowercase hex")
    require(sha256(source_path) == args.source_sha256, "Source checkpoint SHA mismatch")
    selection_path = Path(args.source_selection).resolve()
    selection = read_json(selection_path)
    require(selection.get("accepted") is True, "Source selection is not accepted by multilingual retention gate")
    require(selection.get("selection_split") == "FLEURS validation only", "Source selection must be DEV-only")
    require(type(selection.get("selected_step")) is int and selection["selected_step"] >= 0, "Invalid selected step")
    selected_weights = Path(selection["weights"]).resolve()
    require(sha256(selected_weights) == args.source_sha256, "Source copy does not match accepted selected weights")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,100}", args.target_run_id) is not None,
            "Target run ID must be a safe new name")
    require(args.target_run_id != selection["run_id"], "Target run ID must differ from the source")
    require(not (ROOT / "outputs/parakeet_turkish/runs" / args.target_run_id).exists(), "Target run already exists")
    for suffix in ("_latest.pt", "_best.pt"):
        require(not (ROOT / "work" / (args.target_run_id + suffix)).exists(), "Target checkpoint namespace already exists")
    output_checkpoint, output_report = Path(args.output_checkpoint).resolve(), Path(args.output_report).resolve()
    require(output_checkpoint != output_report, "Checkpoint/report paths must differ")
    protected_paths = {source_path, source_tokenizer, selection_path, selected_weights} | {Path(pin["path"]) for pin in asset_pins.values()}
    for path in (output_checkpoint, output_report):
        require(path.is_relative_to(ROOT), "Output must stay within this workspace")
        require(path not in protected_paths and not path.exists(), "Output must be a NEW unrelated file")
    return {
        "schema": 1, "kind": "HF_Turkish_BPE_512_weights_only_warm_start_v1",
        "status": "CPU_prepared", "passed": True, "conversion_executed": False,
        "target_run_id": args.target_run_id, "new_optimizer_required": True,
        "exact_training_resume": False, "tokenizer_activated": False,
        "source": {"path": str(source_path), "sha256": args.source_sha256,
                   "run_id": selection["run_id"], "selected_step": selection["selected_step"],
                   "selection_path": str(selection_path), "selection_sha256": sha256(selection_path)},
        "source_tokenizer": {"path": str(source_tokenizer), "sha256": SOURCE_TOKENIZER_SHA256},
        "candidate_assets": asset_pins, "old_vocab_size": OLD_VOCAB, "new_vocab_size": NEW_VOCAB,
        "blank_token_id": BLANK, "duration_rows_old": [OLD_VOCAB, OLD_VOCAB + DURATIONS],
        "duration_rows_new": [NEW_VOCAB, NEW_VOCAB + DURATIONS],
        "six_character_delta_multiplier_rows_unchanged": [8193, 8199],
        "donor_plan": donors,
        "initialization": "Mean ORIGINAL leaf donor rows with multiplicity; bias mean minus 2 exactly once",
        "real_arithmetic_logit_property": "New logit equals mean original donor logits minus 2; below their maximum",
        "limitations": ["Floating-point greedy property needs a GPU numerical check before activation",
                        "Extra denominator mass and changed BPE segmentation can alter confidence/training",
                        "Keep ORIGINAL replay labels/cache for teacher conditioning; 80 accepted replay encodings change",
                        "This tool does not resize a live model or update training/generation configuration"],
        "output_checkpoint": str(output_checkpoint), "output_report": str(output_report),
        "script_sha256": sha256(__file__),
    }


def atomic_new(path, writer):
    """Complete a private temp, then publish a NEW file without overwriting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows rename rejects an existing destination, including races.
        # On POSIX use hard-link creation for the same no-overwrite guarantee.
        if os.name == "nt":
            os.rename(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_new(path, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    atomic_new(path, lambda stream: stream.write(raw))


def convert(args, report):
    import torch  # Deliberately deferred; never imported by prepare/self-test.
    require(not torch.cuda.is_initialized(), "Run conversion in a fresh CPU-only process")
    torch.set_num_threads(2)
    rng_before = torch.get_rng_state().clone()
    def bit_equal(left, right):
        return (left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))
    saved = torch.load(args.source_checkpoint, map_location="cpu", weights_only=True)
    weights = saved.get("weights")
    require(isinstance(weights, dict) and len(weights) == 207, "Expected exactly 207 trainable source tensors")
    require(set(EXPANDED_KEYS).issubset(weights), "Required token tensors are missing")
    require(saved.get("run_id") == report["source"]["run_id"] and saved.get("step") == report["source"]["selected_step"],
            "Source state differs from selected run/step")
    layout = saved.get("lora_metadata")
    require(isinstance(layout, dict) and bool(layout), "Source LoRA layout is missing")
    for name, tensor in weights.items():
        require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and tensor.dtype == torch.float32,
                f"Source tensor must be CPU FP32: {name}")
        require(bool(torch.isfinite(tensor).all()), f"Nonfinite source tensor: {name}")
    embedding, head, bias = [weights[key] for key in EXPANDED_KEYS]
    require(embedding.ndim == 2 and embedding.shape[0] == OLD_VOCAB, "Embedding shape mismatch")
    require(head.ndim == 2 and head.shape[0] == OLD_VOCAB + DURATIONS, "Joint weight shape mismatch")
    require(bias.shape == (OLD_VOCAB + DURATIONS,), "Joint bias shape mismatch")
    expanded_embedding = torch.empty((NEW_VOCAB, embedding.shape[1]), dtype=torch.float32)
    expanded_head = torch.empty((NEW_VOCAB + DURATIONS, head.shape[1]), dtype=torch.float32)
    expanded_bias = torch.empty((NEW_VOCAB + DURATIONS,), dtype=torch.float32)
    expanded_embedding[:OLD_VOCAB].copy_(embedding)
    expanded_head[:OLD_VOCAB].copy_(head[:OLD_VOCAB])
    expanded_bias[:OLD_VOCAB].copy_(bias[:OLD_VOCAB])
    expanded_head[NEW_VOCAB:].copy_(head[OLD_VOCAB:])
    expanded_bias[NEW_VOCAB:].copy_(bias[OLD_VOCAB:])
    for row in report["donor_plan"]:
        ids = torch.tensor(row["original_leaf_donor_ids"], dtype=torch.long)
        index = row["HF_id"]
        expanded_embedding[index].copy_(embedding.index_select(0, ids).mean(0))
        expanded_head[index].copy_(head.index_select(0, ids).mean(0))
        expanded_bias[index].copy_(bias.index_select(0, ids).mean() - BIAS_PENALTY)
    updated = dict(weights)
    updated.update(zip(EXPANDED_KEYS, (expanded_embedding, expanded_head, expanded_bias)))
    require(all(bool(torch.isfinite(value).all()) for value in updated.values()), "Conversion produced a nonfinite tensor")
    require(bit_equal(expanded_embedding[:OLD_VOCAB], embedding), "Original embedding rows changed")
    require(bit_equal(expanded_head[:OLD_VOCAB], head[:OLD_VOCAB]) and bit_equal(expanded_bias[:OLD_VOCAB], bias[:OLD_VOCAB]), "Original token/blank rows changed")
    require(bit_equal(expanded_head[NEW_VOCAB:], head[OLD_VOCAB:]) and bit_equal(expanded_bias[NEW_VOCAB:], bias[OLD_VOCAB:]), "Duration rows changed")
    require(all(bit_equal(updated[key], value) for key, value in weights.items() if key not in EXPANDED_KEYS), "An unrelated trainable tensor changed")
    require(torch.equal(rng_before, torch.get_rng_state()), "Conversion changed CPU Torch RNG")
    require(sha256(args.source_checkpoint) == args.source_sha256, "Source changed during conversion")
    payload = {
        "schema_version": 1, "kind": report["kind"], "run_id": args.target_run_id,
        "step": 0, "trained_updates": 0, "source_step": report["source"]["selected_step"],
        "weights": updated, "lora_metadata": copy.deepcopy(layout),
        "vocabulary_expansion": {"old_vocab_size": OLD_VOCAB, "vocab_size": NEW_VOCAB,
                                 "blank_token_id": BLANK, "durations": list(range(DURATIONS)),
                                 "source_sha256": args.source_sha256,
                                 "tokenizer_sha256": ASSET_SHA256["tokenizer.json"],
                                 "new_optimizer_required": True, "exact_training_resume": False},
    }
    atomic_new(args.output_checkpoint, lambda stream: torch.save(payload, stream))
    reopened = torch.load(args.output_checkpoint, map_location="cpu", weights_only=True)
    require(set(reopened["weights"]) == set(updated) and all(bit_equal(reopened["weights"][key], value) for key, value in updated.items()), "Serialized weights differ")
    require(reopened["lora_metadata"] == layout, "Serialized LoRA metadata changed")
    report.update({"status": "complete", "conversion_executed": True,
                   "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "output_checkpoint_sha256": sha256(args.output_checkpoint),
                   "verification": {"trainable_tensors": 207, "expanded_tensors": 3,
                                    "unrelated_tensors_bitexact": 204, "old_token_rows_bitexact": True,
                                    "blank_row_bitexact": True, "duration_rows_bitexact": True,
                                    "serialized_all_tensors_bitexact": True, "all_finite_FP32": True,
                                    "lora_metadata_unchanged": True, "CPU_RNG_unchanged": True,
                                    "GPU_initialized": False}})
    write_json_new(args.output_report, report)


def self_test():
    original = {"model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []}, "decoder": None}
    candidate = copy.deepcopy(original)
    candidate["model"]["vocab"].update({"ab": 2, "aba": 3})
    candidate["model"]["merges"] = [["a", "b"], ["ab", "a"]]
    additions = [{"piece": "ab", "left": "a", "right": "b", "HF_id": 2, "left_HF_id": 0, "right_HF_id": 1, "HF_merge_rank": 0},
                 {"piece": "aba", "left": "ab", "right": "a", "HF_id": 3, "left_HF_id": 2, "right_HF_id": 0, "HF_merge_rank": 1}]
    plan = build_donor_plan(original, candidate, additions)
    require(plan[1]["original_leaf_donor_ids"] == [0, 1, 0], "Recursive multiplicity test failed")
    require(plan[1]["original_leaf_donor_counts"] == {"0": 2, "1": 1}, "Donor weights test failed")
    old_logits = [1.25, -0.75]
    initialized = sum(old_logits[index] for index in plan[1]["original_leaf_donor_ids"]) / 3 - BIAS_PENALTY
    require(math.isclose(initialized, -17 / 12) and initialized <= max(old_logits) - BIAS_PENALTY, "One-time bias/convexity test failed")
    mutations = [lambda a, c: a[1].update(left_HF_id=3),
                 lambda a, c: a[1].update(piece="abb"),
                 lambda a, c: c["model"]["vocab"].update(a=1),
                 lambda a, c: c.update(decoder={"changed": True}),
                 lambda a, c: c["model"]["merges"][1].reverse()]
    rejected = 0
    for mutation in mutations:
        bad_additions, bad_candidate = copy.deepcopy(additions), copy.deepcopy(candidate)
        mutation(bad_additions, bad_candidate)
        try:
            build_donor_plan(original, bad_candidate, bad_additions)
        except ValueError:
            rejected += 1
    require(rejected == len(mutations), "A donor/prefix corruption was accepted")
    return {"passed": True, "recursive_leaf_multiplicity": True, "bias_penalty_once": True,
            "corruptions_rejected": rejected, "Torch_imported": False}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-checkpoint")
    result.add_argument("--source-sha256")
    result.add_argument("--source-selection")
    result.add_argument("--source-tokenizer", default=str(ROOT / "outputs/parakeet_turkish/checkpoint/tokenizer.json"))
    result.add_argument("--candidate-dir", default=str(DEFAULT_CANDIDATE))
    result.add_argument("--target-run-id")
    result.add_argument("--output-checkpoint")
    result.add_argument("--output-report")
    result.add_argument("--prepare-only", action="store_true")
    result.add_argument("--self-test", action="store_true")
    return result


def main():
    cli = parser()
    args = cli.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return
    for name in ("source_checkpoint", "source_sha256", "source_selection", "target_run_id", "output_checkpoint", "output_report"):
        if getattr(args, name) is None:
            cli.error("--" + name.replace("_", "-") + " is required")
    report = prepare(args)
    if args.prepare_only:
        write_json_new(args.output_report, report)
        print(json.dumps({"status": report["status"], "passed": True, "donor_pieces": len(report["donor_plan"]),
                          "maximum_leaf_count": max(row["original_leaf_count"] for row in report["donor_plan"]),
                          "Torch_imported": False, "output_report": args.output_report}))
    else:
        convert(args, report)
        print(json.dumps({"status": report["status"], "passed": True, "output_checkpoint_sha256": report["output_checkpoint_sha256"]}))


if __name__ == "__main__":
    main()
