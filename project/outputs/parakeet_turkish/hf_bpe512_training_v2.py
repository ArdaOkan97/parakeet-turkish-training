"""Isolated BPE512 student/metadata boundary; all ML imports are runtime-only.

The source must be a closed, qualifying8199 checkpoint. Initial admission starts fresh Adam. Later SAME-run exact windows restore all
optimizer/sampler/RNG/history state under the original immutable identity. Replay keeps its original processor, labels, cache and8193 teacher.
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


COMMON_TRAINING_KEYS = frozenset(("epochs", "steps", "batch_size", "accumulation", "replay_fraction", "bucket_size", "seed",
    "max_tokens", "head_lr", "lora_lr", "new_token_lr_multiplier", "replay_kl", "duration_kl", "feature_distillation",
    "rank", "alpha", "last_n_layers", "chunk_size", "tdt_implementation", "warmup_steps", "schedule", "min_lr_ratio",
    "eval_per_epoch", "eval_epoch_fraction", "eval_every_updates", "min_improvement", "no_encoder_checkpoint",
    "phonetic_sh_donors", "train_limit", "early_stop_patience", "max_retention_failures", "target_turkish_wer"))
DOMAIN_TRAINING_KEYS = frozenset(("sampling_mode", "fleurs_fraction", "replay_seed"))
DOMAIN_SOURCES = {"train_lora_batched_domain30_v1.py":"7242b0ce65619f8df38593e89c6ed5b7a01487d1d1174c2e28f73d00bff8ee4f",
    "turkish_domain_sampler_v2.py":"822d0ddb608d719112e4a4285ad42a76ef7f1e4ede0cf4d97211fe38f97c62fd",
    "domain30_training_identity_v1.py":"151bf1b3236e3e23c269f7be2c2e94bf12a71b1e547b64a70f26d6f77ae81f5e"}


def common_source_recipe(source, closure):
    """Copy the actual CLOSED domain recipe, retaining its replay RNG policy."""
    recipe = source["config_identity"]["training_arguments"]
    require(set(recipe) == COMMON_TRAINING_KEYS | DOMAIN_TRAINING_KEYS,
            "BPE v2 requires the reviewed CLOSED domain source, not an older uniform fallback")
    identity = source["config_identity"].get("domain_sampling_identity", {})
    require(recipe["sampling_mode"] == "domain30" and recipe["fleurs_fraction"] == .3
        and type(recipe["replay_seed"]) is int and identity.get("sampling_mode") == "domain30"
        and identity.get("replay_seed") == recipe["replay_seed"] and identity.get("turkish_seed") == recipe["seed"]
        and identity.get("replay_rng_independent_of_turkish") is True
        and identity.get("fleurs_fraction_numerator") == 3 and identity.get("fleurs_fraction_denominator") == 10
        and identity.get("batch_size") == 8 and identity.get("accumulation") == 2
        and identity.get("replay_per_update") == 8,
        "Actual original independent-replay domain sampling identity differs")
    require(closure.get("allow_BPE512_vocabulary_stage") is True
        and closure.get("source_sampling_metadata") == identity,
        "ROOT must explicitly approve vocabulary/target changes while preserving domain/independent-replay policy")
    for name, digest in DOMAIN_SOURCES.items():
        require(source["code_sha256"].get(name) == digest and sha_file(HERE/name) == digest,
                "Closed domain source implementation differs: " + name)
    return dict(recipe)


def source_sampling_transition(source, closure):
    common_source_recipe(source, closure)
    recipe = source["config_identity"]["training_arguments"]
    return {"source_mode": "domain30", "source_sampling_metadata": source["config_identity"]["domain_sampling_identity"],
        "source_fleurs_fraction": recipe["fleurs_fraction"], "source_replay_seed": recipe["replay_seed"],
        "target_mode": "domain30", "source_replay_RNG_independent": True,
        "target_replay_RNG_independent": True, "original_replay_pool_teacher_labels_unchanged": True,
        "fleurs_fraction_preserved": .3, "independent_replay_seed_preserved": recipe["replay_seed"],
        "sampling_policy_changed": False, "TR_length_bucket_order_can_change_with_BPE_targets": True,
        "source_continuation_cursor_claimed": False,
        "scope": "Initial model-only qualifying domain BEST; NEW Adam/sampler/RNG, same domain quotas and replay seed. Later SAME-run exact windows retain this BPE science."}


def closed_source_identity(args):
    """Require ROOT-confirmed CLOSED domain BEST and its NEW conversion."""
    closure = read_json(args.bpe_source_closure)
    require(closure.get("kind") == "root_owned_BPE512_closed_source_v2" and closure.get("passed") is True
        and closure.get("root_observed_owned_session_completed") is True and closure.get("child_exit_code") == 0
        and closure.get("latest_closed_qualified_source_confirmed") is True
        and isinstance(closure.get("source_choice_rationale"), str) and bool(closure["source_choice_rationale"].strip()),
        "ROOT-confirmed latest CLOSED qualifying domain source and explicit choice required")
    paths = {}
    for name in ("source_checkpoint", "source_selection", "source_run_config", "source_training_history", "source_development_summary"):
        item = closure[name]; path = Path(item["path"]).resolve()
        require(path.is_relative_to(ROOT) and sha_file(path) == item["sha256"], "Closed source changed: " + name)
        paths[name] = path
    require(paths["source_checkpoint"] == Path(args.bpe_source_checkpoint).resolve()
        and paths["source_selection"] == Path(args.bpe_source_selection).resolve(), "Explicit source paths differ")
    selection, source_config = read_json(paths["source_selection"]), read_json(paths["source_run_config"])
    require(selection.get("accepted") is True and type(selection.get("selected_step")) is int
        and 0 <= selection["selected_step"] <= selection["completed_steps"]
        and selection.get("run_id") == closure.get("source_run_id") == source_config.get("run_id")
        and selection["run_id"] != args.run_id and selection.get("heldout_tests_evaluated") is False
        and selection.get("selection_split") == "FLEURS validation only"
        and type(selection.get("turkish_validation_wer")) in (int, float)
        and math.isfinite(selection["turkish_validation_wer"])
        and Path(selection["weights"]).resolve() == paths["source_checkpoint"], "Qualifying selected BEST required")
    require(paths["source_development_summary"] == HERE/"results"/(selection["run_id"]+"_step_"+str(selection["selected_step"]))/"summary.json",
        "Actual selected source DEV folder required")
    dev = read_json(paths["source_development_summary"]); baseline = read_json(HERE/"results/base_v3/summary.json")
    require(dev["validation_tr"]["asr"]["wer"] == selection["turkish_validation_wer"]
        and all(dev["validation_"+lang]["asr"]["utterances"] == 50 for lang in ("tr","en","es","it","fr","de"))
        and all(dev["validation_"+lang]["asr"]["wer"] <= baseline["validation_"+lang]["asr"]["wer"]
                + max(.005, .1*baseline["validation_"+lang]["asr"]["wer"]) for lang in ("en","es","it","fr","de")),
        "Complete selected source DEV and all five original retention gates required")
    history = read_json(paths["source_training_history"])
    require(len(history) == selection["completed_steps"] and all(row["step"] == index for index,row in enumerate(history,1)),
        "Source completed history differs")
    if selection["selected_step"]:
        gate = history[selection["selected_step"]-1].get("validation", {})
        require(gate.get("retention_gate_passed") is True and gate.get("turkish_wer") == selection["turkish_validation_wer"],
            "Source BEST is not its recorded qualifying DEV")
    common = common_source_recipe(source_config, closure)
    for name, digest in source_config["code_sha256"].items():
        require(Path(name).name == name and sha_file(HERE/name) == digest, "Closed source implementation changed: " + name)
    require(source_config.get("base_revision") == "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
        and source_config.get("turkish_clips_used") == 74883 and source_config.get("replay_clips_used") == 1595
        and source_config["config_identity"]["extended_tokenizer_sha256"] == BASELINE_SEMANTIC_SHA
        and source_config["config_identity"]["replay_pool_sha256"] == REPLAY_IDS_SHA,
        "Original8199/74883TR/1595replay lineage required")
    for key,value in common.items():
        require(getattr(args,key) == value, "Only declared tokenizer/target/geometry/sampler changes allowed; source recipe differs: " + key)
    conversion = read_json(args.bpe_conversion_report)
    require(conversion.get("kind") == "HF_Turkish_BPE_512_weights_only_warm_start_v1"
        and conversion.get("status") == "complete" and conversion.get("passed") is True
        and conversion.get("new_optimizer_required") is True and conversion.get("exact_training_resume") is False
        and conversion.get("target_run_id") == args.run_id, "Completed NEW weights-only conversion required")
    source = conversion["source"]
    require(Path(source["path"]).resolve() == paths["source_checkpoint"]
        and source["sha256"] == closure["source_checkpoint"]["sha256"]
        and source["run_id"] == selection["run_id"] and source["selected_step"] == selection["selected_step"]
        and source["selection_sha256"] == closure["source_selection"]["sha256"], "Conversion source/selection differs")
    converted = Path(conversion["output_checkpoint"]).resolve()
    require(converted.is_relative_to(ROOT) and sha_file(converted) == conversion["output_checkpoint_sha256"], "Original converted warmweights changed")
    if args.resume_exact is None:
        require(args.resume_weights is not None and Path(args.resume_weights).resolve() == converted,
            "Initial BPE arm uses approved NEW model-only converted weights")
    else:
        require(args.resume_weights is None, "Exact BPE continuation must not reset Adam from warmweights")
    return {"closed_source":closure,"closure_sha256":sha_file(args.bpe_source_closure),
        "conversion_sha256":sha_file(args.bpe_conversion_report),"selection":selection,
        "source_scientific_arguments":common,"source_sampling_transition":source_sampling_transition(source_config,closure),
        "source_pool_sha256":{"tr":source_config["config_identity"]["tr_pool_sha256"],"replay":source_config["config_identity"]["replay_pool_sha256"]},
        "converted_weights_path":str(converted),"converted_weights_sha256":conversion["output_checkpoint_sha256"]}


def established_source_equivalence(run_dir, run_id, source_identity, *, trainer_path, runtime_path):
    """Exact windows reuse the immutable actual first-window GPU proof, never infer it."""
    value = read_json(Path(run_dir)/"BPE512_actual_GPU_source_equivalence.json")
    require(value.get("kind") == "HF_BPE512_actual_unsuppressed_GPU_source_equivalence_v1"
        and value.get("passed") is True and value.get("actual_GPU_decode") is True
        and value.get("before_optimizer_creation") is True and value.get("decoded_clips") == 300
        and value.get("run_id") == run_id and value.get("source_identity") == source_identity
        and value.get("student_geometry") == GEOMETRY
        and value.get("converted_weights_sha256") == source_identity["converted_weights_sha256"]
        and value.get("trainer_sha256") == sha_file(trainer_path) and value.get("candidate_runtime_sha256") == sha_file(runtime_path)
        and value.get("candidate_tokenizer_sha256") == TOKENIZER_SEMANTIC_SHA
        and not value.get("raw_transcript_changes") and not value.get("token_duration_emission_changes")
        and value.get("source_gate",{}).get("retention_gate_passed") is True
        and value.get("candidate_gate",{}).get("retention_gate_passed") is True,
        "Immutable actual pre-Adam300 source equality/gates/source identity required for exact continuation")
    config = read_json(Path(run_dir)/"run_config.json")
    require(config["config_identity"]["BPE512"]["source_equivalence_sha256"] == sha_file(Path(run_dir)/"BPE512_actual_GPU_source_equivalence.json"),
        "Initial config no longer binds its actual source equivalence report")
    return value


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
