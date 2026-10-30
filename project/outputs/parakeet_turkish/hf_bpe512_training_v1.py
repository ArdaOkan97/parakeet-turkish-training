"""Isolated BPE512 student/metadata boundary; all ML imports are runtime-only.

The source must be a closed, qualifying8199 checkpoint. No optimizer state is
reused. Replay keeps its original processor, labels, cache and8193 teacher.
The actual GPU source/candidate DEV transcript check runs before Adam exists.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from hf_bpe512_tr_metadata_v1 import (
    BLANK_ID, BPE_ROWS, CLIPS, DURATION_ROWS, METADATA_SHA, OLD_TEACHER_SIZE,
    SIX_CHARACTER_ROWS, TOKENIZER_FILE_SHA, TOKENIZER_SEMANTIC_SHA, VOCAB_SIZE,
    iter_candidate_training_rows, read_candidate_context, require, sha_file,
)
from expand_trainable_bpe_checkpoint_v1 import ASSET_SHA256, build_donor_plan

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BASELINE_SEMANTIC_SHA = "b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75"
REPLAY_IDS_SHA = "c165e0a2e2021b5c69e5b43b837b1276be1dc0abb7f0f69b8d9623284a87fd2b"
GEOMETRY = {"student_vocab_size": VOCAB_SIZE, "blank_id": BLANK_ID,
            "teacher_vocab_size": OLD_TEACHER_SIZE, "six_character_rows": list(SIX_CHARACTER_ROWS),
            "new_BPE_rows": list(BPE_ROWS), "duration_rows": list(DURATION_ROWS),
            "new_BPE_delta_multiplier": 1.0}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def semantic_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def suppression_ids(values):
    """Move ONLY the five old duration IDs; new BPE IDs stay unsuppressed."""
    if values is None:
        return None
    require(all(type(value) is int and value >= 0 for value in values), "Integer suppression IDs required")
    return [VOCAB_SIZE + value - BPE_ROWS[0] if BPE_ROWS[0] <= value < BPE_ROWS[0] + 5 else value
            for value in values]


def candidate_assets(candidate_dir):
    candidate_dir = Path(candidate_dir).resolve()
    for name, expected in ASSET_SHA256.items():
        require(sha_file(candidate_dir / name) == expected, "Frozen candidate changed: " + name)
    candidate = read_json(candidate_dir / "tokenizer.json")
    require(semantic_sha(candidate) == TOKENIZER_SEMANTIC_SHA, "Candidate semantic geometry differs")
    return candidate, read_json(candidate_dir / "added_pieces.json")


def candidate_tr_rows(processor, rows, metadata_path, proof_path, max_tokens):
    """Consume the complete checked candidate iterator before the first update."""
    context = read_candidate_context(proof_path)
    require(Path(metadata_path).resolve() == context.metadata_path and max_tokens == 180,
            "Exact candidate metadata and unchanged max180 policy required")
    old = json.loads(processor.tokenizer._tokenizer.to_str())
    require(semantic_sha(old) == BASELINE_SEMANTIC_SHA, "Replay/source processor must still be original8199")
    result = list(iter_candidate_training_rows(context))
    require(len(rows) == len(result) == CLIPS, "Whole accepted TR pool required; no subset/supplement")
    require(all(left["source_id"] == right["source_id"] and left["text"] == right["text"]
                for left, right in zip(rows, result)), "Candidate ordered source/text manifest differs")
    return result


def confirm_candidate_acceptance(rows, replay, audit):
    require(len(rows) == CLIPS and len(replay) == 1595, "Exact TR/replay pool required")
    ids = hashlib.sha256("\n".join(sorted(row["source_id"] for row in replay)).encode()).hexdigest()
    require(ids == REPLAY_IDS_SHA, "Original1595 replay source set differs")
    from public_turkish_data_policy_v3 import ids_sha
    require(ids_sha(rows) == audit["accepted_source_ids_sha256"], "Original accepted TR pool differs")
    audit.update(turkish_training_representation="TR_frame_target_metadata_HF_BPE_512_v1",
                 TR_frozen_encoder_feature_cache_required=False, TR_metadata_acceptance_pending=False,
                 actual_TR_metadata_accepted_clips=CLIPS, candidate_TR_metadata_sha256=METADATA_SHA,
                 replay_targets_and_encoder_cache_unchanged=True,
                 acoustic_cache_acceptance_pending="not_applicable_to_TR_metadata")


def closed_source_identity(args):
    """Small root-owned closure receipt plus complete qualifying selection."""
    closure = read_json(args.bpe_source_closure)
    require(closure.get("kind") == "root_owned_BPE512_closed_source_v1" and closure.get("passed") is True
            and closure.get("root_observed_owned_session_completed") is True
            and closure.get("child_exit_code") == 0, "Root-owned closed source receipt required")
    checkpoint, selection_path = Path(args.bpe_source_checkpoint).resolve(), Path(args.bpe_source_selection).resolve()
    require(checkpoint.is_relative_to(ROOT) and selection_path.is_relative_to(ROOT), "Workspace source required")
    for name, path in (("source_checkpoint", checkpoint), ("source_selection", selection_path)):
        item = closure[name]
        require(Path(item["path"]).resolve() == path and sha_file(path) == item["sha256"],
                "Closed source changed: " + name)
    selection = read_json(selection_path)
    require(selection.get("accepted") is True and selection.get("selected_step") is not None
            and selection.get("run_id") == closure.get("source_run_id") and selection.get("run_id") != args.run_id
            and type(selection.get("turkish_validation_wer")) in (int,float)
            and math.isfinite(selection["turkish_validation_wer"]), "Qualifying closed selection required")
    conversion = read_json(args.bpe_conversion_report)
    require(conversion.get("kind") == "HF_Turkish_BPE_512_weights_only_warm_start_v1"
            and conversion.get("status") == "complete" and conversion.get("passed") is True
            and conversion.get("new_optimizer_required") is True and conversion.get("exact_training_resume") is False
            and conversion.get("target_run_id") == args.run_id,
            "Completed weights-only conversion for this NEW run required")
    source = conversion["source"]
    require(Path(source["path"]).resolve() == checkpoint and source["sha256"] == closure["source_checkpoint"]["sha256"]
            and source["run_id"] == selection["run_id"] and source["selected_step"] == selection["selected_step"]
            and source["selection_sha256"] == closure["source_selection"]["sha256"], "Conversion source/selection differs")
    require(Path(conversion["output_checkpoint"]).resolve() == Path(args.resume_weights).resolve()
            and sha_file(args.resume_weights) == conversion["output_checkpoint_sha256"], "Converted warmweights changed")
    return {"closed_source":closure,"closure_sha256":sha_file(args.bpe_source_closure),
            "conversion_sha256":sha_file(args.bpe_conversion_report),"selection":selection}


def install_candidate_student(processor, model, candidate_dir, *, torch_module=None):
    """Retain trained8199 rows; append convex ORIGINAL-leaf donors minus2 once."""
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    from tokenizers import Tokenizer
    from extend_vocab import _make_tokenizer
    candidate, additions = candidate_assets(candidate_dir)
    original = json.loads(processor.tokenizer._tokenizer.to_str())
    require(semantic_sha(original) == BASELINE_SEMANTIC_SHA, "Only original8199 source can be expanded")
    donors = build_donor_plan(original, candidate, additions)
    embedding, head = model.decoder.embedding, model.joint.head
    require(model.config.vocab_size == 8199 and model.config.blank_token_id == BLANK_ID
            and embedding.num_embeddings == 8199 and head.out_features == 8204
            and list(model.config.durations) == [0,1,2,3,4], "Source model geometry differs")
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(embedding.weight.device).clone() if embedding.weight.device.type == "cuda" else None
    try:
        replacement = torch.nn.Embedding(VOCAB_SIZE, embedding.embedding_dim, padding_idx=embedding.padding_idx,
                                         device=embedding.weight.device, dtype=embedding.weight.dtype)
        output = torch.nn.Linear(head.in_features, VOCAB_SIZE+5, bias=True,
                                 device=head.weight.device, dtype=head.weight.dtype)
    finally:
        torch.set_rng_state(rng)
        if cuda_rng is not None: torch.cuda.set_rng_state(cuda_rng,embedding.weight.device)
    with torch.no_grad():
        replacement.weight[:8199].copy_(embedding.weight)
        output.weight[:8199].copy_(head.weight[:8199]); output.bias[:8199].copy_(head.bias[:8199])
        output.weight[VOCAB_SIZE:].copy_(head.weight[8199:]); output.bias[VOCAB_SIZE:].copy_(head.bias[8199:])
        for row in donors:
            index, leaves = row["HF_id"], row["original_leaf_donor_ids"]
            replacement.weight[index].copy_(embedding.weight[leaves].mean(0))
            output.weight[index].copy_(head.weight[leaves].mean(0))
            output.bias[index].copy_(head.bias[leaves].mean(0)-2.0)
    for replacement_parameter, original_parameter in ((replacement.weight,embedding.weight),
                                                       (output.weight,head.weight),(output.bias,head.bias)):
        replacement_parameter.requires_grad_(original_parameter.requires_grad)
    replacement.train(embedding.training); output.train(head.training)
    tokenizer = _make_tokenizer(processor.tokenizer, Tokenizer.from_str(json.dumps(candidate,ensure_ascii=False)))
    require(semantic_sha(json.loads(tokenizer._tokenizer.to_str())) == TOKENIZER_SEMANTIC_SHA,
            "Runtime candidate tokenizer differs from frozen asset")
    generation = model.generation_config
    suppression = {}
    for field in ("suppress_tokens","begin_suppress_tokens"):
        before = getattr(generation,field,None); after = suppression_ids(before)
        require(not any(BPE_ROWS[0] <= value < BPE_ROWS[1] for value in (after or [])), "New BPE rows were suppressed")
        suppression[field] = {"before":copy.deepcopy(before),"after":after}
        setattr(generation,field,after)
    model.decoder.embedding,model.joint.head = replacement,output
    model.config.vocab_size = model.joint.vocab_size = VOCAB_SIZE
    if hasattr(model.decoder,"vocab_size"): model.decoder.vocab_size = VOCAB_SIZE
    if hasattr(generation,"vocab_size"): generation.vocab_size = VOCAB_SIZE
    processor.tokenizer = tokenizer
    return {**GEOMETRY,"changed":True,"extended_tokenizer_sha256":TOKENIZER_SEMANTIC_SHA,
            "candidate_file_sha256":TOKENIZER_FILE_SHA,"generation_suppression":suppression,
            "donor_policy":"Mean original trained leaf rows, multiplicity retained, bias minus2 once",
            "new_optimizer_required":True,"exact_training_resume":False}


def native_joint_bound(turkish, replay, maximum_bytes):
    """Actual accepted T/U, explicit widened width; no crop/drop fallback."""
    require(type(maximum_bytes) is int and 0 < maximum_bytes <= 1_850_000_000, "Declared BPE arm byte bound required")
    maximum = {"bytes":0}
    count = 0
    for pool in (turkish,replay):
        for row in pool:
            t,u = row["encoder_frames"],row["target_tokens"]
            require(type(t) is int and type(u) is int and t>0 and 0<u<=180, "Positive accepted T/U required")
            size = t*(u+1)*(VOCAB_SIZE+5)*4
            if size>maximum["bytes"]:maximum={"bytes":size,"T":t,"U":u,"source_id":row["source_id"],"language":row["language"]}
            count += 1
            require(size<=maximum_bytes,"Exact8716 native joint bound exceeded; no cropping/dropping: "+row["source_id"])
    return {"kind":"HF_BPE512_actual_accepted_pool_joint_bound_v1","clips_checked":count,
            "joint_width":VOCAB_SIZE+5,"declared_maximum_bytes":maximum_bytes,"maximum":maximum,
            "VRAM_fit_proven":False,"scope":"Single FP32 joint arithmetic only; native workspace/encoder/Adam extra"}


def validate_converted_weights(source, converted, *, torch_module):
    """Independently bind expanded warmweights to the actual closed source."""
    torch = torch_module
    def bit_equal(left,right):
        return left.shape == right.shape and torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                                                         right.contiguous().reshape(-1).view(torch.uint8))
    require(set(source["weights"]) == set(converted["weights"]) and len(source["weights"]) == 207
            and source["lora_metadata"] == converted["lora_metadata"], "All207 trainable/layout keys required")
    special = {"decoder.embedding.weight","joint.head.weight","joint.head.bias"}
    for name, before in source["weights"].items():
        after = converted["weights"][name]
        require(before.dtype == after.dtype == torch.float32 and bool(torch.isfinite(after).all()), "Finite FP32 warmweights required")
        if name not in special:
            require(bit_equal(before,after), "An old trainable tensor changed: "+name)
        else:
            require(bit_equal(before[:8199],after[:8199]),
                    "An old token/blank row changed: "+name)
            if name != "decoder.embedding.weight":
                require(bit_equal(before[8199:],after[VOCAB_SIZE:]),
                        "A duration row changed: "+name)
    candidate, additions = candidate_assets(ROOT / "work/hf_turkish_bpe_candidate_512_v1")
    original = copy.deepcopy(candidate)
    original["model"]["vocab"] = {piece:index for piece,index in candidate["model"]["vocab"].items() if index < 8199}
    original["model"]["merges"] = candidate["model"]["merges"][:-512]
    for row in build_donor_plan(original,candidate,additions):
        leaves, index = row["original_leaf_donor_ids"], row["HF_id"]
        for name in special:
            expected = source["weights"][name][leaves].mean() if name == "joint.head.bias" else source["weights"][name][leaves].mean(0)
            if name == "joint.head.bias": expected = expected-2.0
            require(bit_equal(expected,converted["weights"][name][index]),
                    "A new row differs from ORIGINAL donor initialization: "+name)


def compare_development_predictions(output, run_id, languages, *, read_rows):
    """Actual full6×50 GPU decoded strings; no numerical/confidence equivalence."""
    require(list(languages) == ["en","es","it","fr","de","tr"], "All six fixed development languages required")
    checks, changed, emission_changed = 0, [], []
    for language in languages:
        source = read_rows(output / "results" / (run_id+"_bpe_source_preflight") / ("validation_"+language+".jsonl"))
        candidate = read_rows(output / "results" / (run_id+"_step_0") / ("validation_"+language+".jsonl"))
        fixed = read_rows(ROOT / "work/data" / ("validation_"+language+".jsonl"))
        require(len(source) == len(candidate) == len(fixed) == 50, "Complete fixed six-language DEV required")
        for left,right,reference in zip(source,candidate,fixed):
            require(left["source_id"] == right["source_id"] == reference["source_id"], "Ordered preflight source differs")
            require(left["reference"] == right["reference"] == reference["text"], "Preflight reference differs")
            checks += 1
            if left["hypothesis"] != right["hypothesis"]:
                changed.append({"language":language,"source_id":left["source_id"],"source":left["hypothesis"],"candidate":right["hypothesis"]})
            fields = ("token_id","duration_id","start_frame","duration_frames")
            left_emissions = [[token[key] for key in fields] for token in left["confidence"]["tokens"]]
            right_emissions = [[token[key] for key in fields] for token in right["confidence"]["tokens"]]
            if left_emissions != right_emissions or any(left["confidence"][key] != right["confidence"][key] for key in ("blank_steps","total_steps")):
                emission_changed.append({"language":language,"source_id":left["source_id"],"source":left_emissions,"candidate":right_emissions})
    return {"kind":"HF_BPE512_actual_unsuppressed_GPU_source_equivalence_v1",
            "actual_GPU_decode":True,"before_optimizer_creation":True,"decoded_clips":checks,
            "raw_transcript_changes":changed,"token_duration_emission_changes":emission_changed,
            "passed":not changed and not emission_changed,
            "confidence_or_loss_equality_required":False,
            "scope":"Same fixed DEV audio, BF16 GPU generation;512 new token IDs unsuppressed"}
