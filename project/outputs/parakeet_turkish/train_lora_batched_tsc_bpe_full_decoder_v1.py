"""INACTIVE paired decoder screen. Source admission waits for both data arms.

The indexed original target geometry chooses batches in both arms. Only the
candidate's selected TR batch is retokenized through its immutable disk sidecar.
All new state, metrics, caches and journal files belong to the user E: root.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch

from atomic_io_retry_v1 import write_json, retry_replace
from append_only_history_v1 import HistoryJournal
from batched_lora_tr_metadata_v1 import collate_training_batch, BatchObjective, sampled_preservation
from bpe_full_decoder_optimizer_v1 import BPEFullDecoderOptimizer, TABLES
from bpe_training_label_catalog_v1 import BPETrainingLabels
from encoder_lora import install_encoder_lora, encoder_lora_metadata
from extend_vocab import extend_processor_and_model, _make_tokenizer
from expanded_training_rows_v1 import reopen_approved_expanded_TR_rows
from lazy_training_rows_v1 import IndexedTrainingRows
from native_tdt_checkpointed_v4 import chunked_tdt_loss
from runtime import load_model, DATA
from TR_tdt_population_weighting_v1 import apply_to_objective as apply_TR_loss, detached_metrics as TR_loss_metrics
from train_lora_batched_data_v11 import BucketCycler
from tsc_bpe_full_decoder_sampler_v1 import BPEFullDecoderSampler
from tsc_bpe_full_decoder_transition_v1 import (
    SOURCE, ROOT, STORAGE, PHASE, UPDATES, MEAN_U, LANGUAGES, LIMITS, preflight,
    audit_parent_state, compare_source_development, decoder_declaration,
    proof_path, record, checked, canonical, require, validate_scores, validate_replay_pool,
)

STATE_KIND = "TSC_BPE_full_decoder_coherent_state_v1"


def stage(path, name, **details):
    from launch_with_memory_guard_TSC_resume_v3 import safe_sample
    value = {"stage": name, "pid": os.getpid(), "at_unix_seconds": time.time(),
             "paired_memory": safe_sample(os.getpid(), name), **details}
    if torch.cuda.is_initialized():
        value["CUDA"] = {"allocated_bytes": torch.cuda.memory_allocated(),
                         "reserved_bytes": torch.cuda.memory_reserved(),
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def expand_candidate(model, processor, source):
    """Preserve all old rows and RNG; explicitly move duration-value rows."""
    from tsc_bpe_acquisition_transition_v1 import candidate_assets
    from tokenizers import Tokenizer
    candidate, donors = candidate_assets()
    embedding, head = model.decoder.embedding, model.joint.head
    require(model.config.vocab_size == 8199 and embedding.num_embeddings == 8199
            and head.out_features == 8204 and model.config.blank_token_id == 8192,
            "Original lexical/blank/duration geometry required")
    rng = (torch.get_rng_state().clone(), [v.clone() for v in torch.cuda.get_rng_state_all()])
    try:
        new_embedding = torch.nn.Embedding(8711, embedding.embedding_dim, padding_idx=embedding.padding_idx,
                                           device=embedding.weight.device, dtype=embedding.weight.dtype)
        new_head = torch.nn.Linear(head.in_features, 8716, device=head.weight.device, dtype=head.weight.dtype)
    finally:
        torch.set_rng_state(rng[0]); torch.cuda.set_rng_state_all(rng[1])
    with torch.no_grad():
        new_embedding.weight[:8199].copy_(embedding.weight)
        new_head.weight[:8199].copy_(head.weight[:8199]); new_head.bias[:8199].copy_(head.bias[:8199])
        new_head.weight[8711:].copy_(head.weight[8199:]); new_head.bias[8711:].copy_(head.bias[8199:])
        for donor in donors:
            index, leaves = donor["HF_id"], donor["original_leaf_donor_ids"]
            new_embedding.weight[index].copy_(source["weights"][TABLES[0]][leaves].mean(0))
            new_head.weight[index].copy_(source["weights"][TABLES[1]][leaves].mean(0))
            new_head.bias[index].copy_(source["weights"][TABLES[2]][leaves].mean() - 2.)
    model.decoder.embedding, model.joint.head = new_embedding, new_head
    model.config.vocab_size = model.joint.vocab_size = 8711
    if hasattr(model.decoder, "vocab_size"): model.decoder.vocab_size = 8711
    if hasattr(model.generation_config, "vocab_size"): model.generation_config.vocab_size = 8711
    suppression = {}
    for field in ("suppress_tokens", "begin_suppress_tokens"):
        old = getattr(model.generation_config, field, None)
        require(old is None or all(type(v) is int and 0 <= v < 8204 for v in old), "Unknown original suppression")
        new = None if old is None else [v + 512 if 8199 <= v < 8204 else v for v in old]
        require(not any(8199 <= v < 8711 for v in new or []), "New BPE rows must be unsuppressed")
        setattr(model.generation_config, field, new)
        suppression[field] = {"before": old, "after": new}
    processor.tokenizer = _make_tokenizer(processor.tokenizer,
        Tokenizer.from_str(json.dumps(candidate, ensure_ascii=False)))
    return {"vocabulary": 8711, "duration_rows": [8711, 8716], "blank": 8192,
            "new512_unsuppressed": True, "suppression": suppression,
            "initialization": "Original leaf donor means; bias mean minus 2",
            "torch_RNG_preserved": torch.equal(torch.get_rng_state(), rng[0]) and
                all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), rng[1]))}


def objective(model, teacher, batch, *, maximum_joint_bytes):
    """Encoder frozen; projector/decoder/joint remain outside no_grad."""
    require(not model.encoder.training and not any(p.requires_grad for p in model.encoder.parameters()),
            "Both arms must freeze the full encoder, including all 192 LoRA tensors")
    device = batch.labels.device
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            encoded = model.encoder(**batch.inputs)
        require(encoded.attention_mask is not None and
                torch.equal(encoded.attention_mask.sum(-1).detach().cpu().long(), batch.frame_lengths),
                "Actual encoder lengths differ from the original singleton geometry")
        features = encoded.last_hidden_state[:, :int(batch.frame_lengths.max())]
        ep, dp = model.encoder_projector(features), model.decoder(batch.decoder_input_ids)
        raw = chunked_tdt_loss(model.joint, ep, dp, batch.labels, model.config.vocab_size,
            batch.blank_id, model.config.durations, logit_lengths=batch.frame_lengths,
            target_lengths=batch.target_lengths, chunk_size=16, reduction="none",
            sigma=0., omega=0., recompute=True, max_joint_bytes=maximum_joint_bytes)
        require(raw.shape == (batch.batch_size,), "Native raw NLL must be per clip")
        tdt = raw.float() / batch.target_lengths.to(device)
        token, duration, feature = sampled_preservation(model, teacher, batch, features, ep, dp, old_size=8193)
        scale = torch.tensor([0. if r["language"] == "tr" else 1. if r["language"] == "pt" else 5/3
                              for r in batch.rows], dtype=torch.float32, device=device)
        total = tdt + scale * (40 * token + 2 * duration + 20 * feature)
    result = BatchObjective(total.mean(), tdt.mean(), token.mean(), duration.mean(), feature.mean(),
        {"tdt": tdt, "token_kl": token, "duration_kl": duration,
         "feature_distillation": feature, "total": total}, batch.batch_size,
        tuple(batch.frame_lengths.tolist()), tuple(batch.target_lengths.tolist()))
    # The reviewed fixed 26.642 denominator applies only to TR, after actual U.
    return apply_TR_loss(result, batch.rows, mode="token_population", population_mean_U=MEAN_U,
                         torch_module=torch)


def evaluate(model, processor, tag):
    """Fixed original seven DEV sets; all output goes to E:, never TEST."""
    import benchmark
    import shutil
    previous = benchmark.OUTPUT, benchmark.DATA
    benchmark.OUTPUT = STORAGE
    try:
        result = benchmark.evaluate(model, processor, tag,
                                    languages=tuple(l for l in LANGUAGES if l != "pt"), splits=("validation",))
        benchmark.DATA = DATA / "portuguese_retention_v1"
        result.update(benchmark.evaluate(model, processor, tag + "_pt", languages=("pt",), splits=("validation",)))
        for name, target in (("validation_pt.jsonl", "validation_pt.jsonl"),
                             ("metadata.json", "metadata_pt.json"), ("summary.json", "summary_pt.json")):
            shutil.copy2(STORAGE / "results" / (tag + "_pt") / name, STORAGE / "results" / tag / target)
        write_json(STORAGE / "results" / tag / "summary.json", result)
        return result
    finally:
        benchmark.OUTPUT, benchmark.DATA = previous


def gate(scores):
    counts, passed = validate_scores(scores)
    return {"turkish_wer": counts["tr"] / 825, "retention_gate_passed": passed,
            "retention_word_errors": {l: counts[l] for l in LIMITS},
            "retention_allowed_word_errors": LIMITS,
            "retention_language_passed": {l: counts[l] <= v for l, v in LIMITS.items()},
            "retention_stop_policy": "monitor_only", "retention_early_stop_active": False,
            "integer_count_gates": True}


def diagnostic_predictions(tag):
    """Saved DEV text only; no score here can change selection or stop policy."""
    from collections import Counter
    import importlib.util
    from metrics import normalize_text
    normalizer = ROOT/"work/research/turkish_normalization_v1/evaluate.py"
    normalizer_record = record(normalizer)
    require(normalizer_record["sha256"] == "52545e91070c9ab6dc80bd27760af278921d6cb5f9eb56f0d4943824c1932323",
            "Frozen separately validated diagnostic normalizer changed")
    spec = importlib.util.spec_from_file_location("_BPE_saved_DEV_spoken_normalizer", normalizer)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    folder = STORAGE/"results"/tag
    emissions = {}
    official_refs, official_hyps, spoken_refs, spoken_hyps = [], [], [], []
    for language in LANGUAGES:
        rows = [json.loads(line) for line in (folder/("validation_"+language+".jsonl")).read_text(encoding="utf-8").splitlines() if line]
        require(len(rows) == 50, "Diagnostics use only complete saved fixed DEV")
        pieces = Counter(token["token_id"] for row in rows for token in row["confidence"]["tokens"]
                         if 8199 <= token["token_id"] < 8711)
        emissions[language] = {"new_BPE_emitted_tokens": sum(pieces.values()),
                               "new_BPE_unique_token_types": len(pieces),
                               "new_BPE_token_counts": {str(token_id): count for token_id, count in pieces.items()}}
        if language == "tr":
            for row in rows:
                official_refs.append(normalize_text(row["reference"], "tr"))
                official_hyps.append(normalize_text(row["hypothesis"], "tr"))
                spoken_refs.append(module.spoken(row["reference"]))
                spoken_hyps.append(module.spoken(row["hypothesis"]))
    def counts(refs, hyps):
        value = module.process_words(refs, hyps)
        n = sum(len(row) for row in value.references)
        s, d, i = value.substitutions, value.deletions, value.insertions
        return {"S": s, "D": d, "I": i, "word_errors": s+d+i, "reference_words": n, "wer": (s+d+i)/n}
    official, spoken = counts(official_refs, official_hyps), counts(spoken_refs, spoken_hyps)
    saved = json.loads((folder/"summary.json").read_text(encoding="utf-8"))["validation_tr"]["asr"]
    require(official["reference_words"] == saved["reference_words"] == 825
            and official["word_errors"] == saved["word_errors"], "Official TR SDI must match the frozen scorer exactly")
    result = {"kind": "BPE_full_decoder_saved_DEV_diagnostics_v1", "normalizer": normalizer_record,
        "official_TR_SDI": official,
        "spoken_form_TR": {**spoken, "diagnostic_only": True},
        "piece_emissions": emissions, "official_scorer_gates_and_selection_changed": False,
        "new_model_decode_or_TEST_used": False}
    write_json(folder/"decoder_diagnostics.json", result)
    return result


def load_catalogs(context):
    args = context["source_arguments"]
    original = IndexedTrainingRows(args["indexed_tr_directory"], cache_size=256)
    # The compact pool record is authoritative about its exact active index.
    pool = json.loads(checked(context["source_training_pool"]).read_text(encoding="utf-8"))
    tr = (original if context["source_data_arm"] == "base" else
          reopen_approved_expanded_TR_rows(Path(pool["turkish_index"]["path"]).parent, cache_size=256))
    require(len(tr) == context["TR_pool_clips"], "Chosen admitted pool count changed")
    replay = pool["replay"]
    require(validate_replay_pool(replay) == context["source_replay_counts"],
            "Replay pool differs from the actual metadata admission")
    return original, tr, replay


def build_sampler(original, tr, replay, context):
    a = context["source_arguments"]
    return BPEFullDecoderSampler(original, tr, [r for r in replay if r["language"] != "pt"],
        portuguese_replay=[r for r in replay if r["language"] == "pt"],
        portuguese_seed=a["portuguese_seed"], data_arm=context["source_data_arm"],
        seed=a["seed"], replay_seed=a["replay_seed"], sampling_mode="uniform", batch_size=8,
        accumulation=2, replay_fraction=.125, fleurs_fraction=a["fleurs_fraction"],
        bucket_size=a["bucket_size"], cycler_factory=BucketCycler,
        cycler_source_sha256=record(SOURCE / "train_lora_batched_data_v11.py")["sha256"])


def save_state(path, model, optimizer, sampler, context, identity, runtime):
    # All 207 named tensors are retained, including the frozen 192 encoder LoRAs.
    names = optimizer.names
    weights = {n: p.detach().cpu() for n, p in model.named_parameters() if n in names}
    require(len(weights) == 207, "Saving only requires_grad tensors would lose the frozen encoder state")
    state = {"kind": STATE_KIND, "schema": 2, "run_id": identity["run_id"],
        "weights": weights, "optimizer": optimizer.state_dict(), "sampler": sampler.state_dict(),
        "lora_metadata": encoder_lora_metadata(model), "config_identity": identity,
        "source_checkpoint": context["source_checkpoint"],
        "frozen_encoder_optimizer": {"storage": "immutable_full_source_checkpoint_not_duplicated",
            "source": context["source_checkpoint"], "clock": context["source_step"],
            "parameter_names": list(optimizer.lora_names), "moment_tensors_advanced": False},
        "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(), **runtime}
    temporary = Path(str(path) + ".tmp")
    torch.save(state, temporary)
    with temporary.open("r+b") as stream: os.fsync(stream.fileno())
    retry_replace(temporary, path)


def partition_fingerprints(optimizer):
    """First two real updates only; one bounded partition copied at a time."""
    result = {}
    for kind, values in (("mature", optimizer.mature), ("fresh", optimizer.fresh)):
        for index, value in enumerate(values):
            data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            result[f"{kind}_{index}"] = hashlib.sha256(memoryview(data)).hexdigest()
            del data
    return result


def validate_own_state(state, args, context, identity, optimizer):
    require(state.get("kind") == STATE_KIND and state.get("schema") == 2
            and state.get("run_id") == args.run_id and canonical(state.get("config_identity")) == canonical(identity)
            and state.get("source_checkpoint") == context["source_checkpoint"], "Own exact coherent identity changed")
    require(state.get("frozen_encoder_optimizer") == {
        "storage": "immutable_full_source_checkpoint_not_duplicated",
        "source": context["source_checkpoint"], "clock": context["source_step"],
        "parameter_names": list(optimizer.lora_names), "moment_tensors_advanced": False},
        "Own exact parked encoder Adam source/names/clock changed")
    n = state["step"] - context["source_step"]
    require(type(n) is int and 0 <= n <= UPDATES and len(state["history"]) == state["step"]
            and canonical(state["history"][:context["source_step"]]) == context["source_history_prefix_sha256"]
            and set(state["weights"]) == set(optimizer.names), "Own exact source prefix/frontier/name set changed")
    require(state["optimizer"]["updates"] == n and state["sampler"]["decoder_adaptation_updates"] == n,
            "Own exact sampler/Adam/global clocks disagree")
    winner = context["source_step"]
    score = context["source_summary"]["validation_tr"]["asr"]["wer"]
    failures = 0
    for row in state["history"][context["source_step"]:]:
        if "validation" not in row: continue
        value = row["validation"]
        passed = all(value["retention_word_errors"][l] <= v for l,v in LIMITS.items())
        require(passed == value["retention_gate_passed"], "Own exact integer gate accounting changed")
        failures = 0 if passed else failures+1
        if passed and value["turkish_wer"] < score: winner, score = row["step"], value["turkish_wer"]
    require(state["retention_failures"] == failures and state["selected_step"] == winner
            and state["best_score"] == score, "Own exact monitored counter/strict-qualified winner differs")
    for name, value in state["weights"].items():
        require(value.shape == optimizer.parameters[name].shape and value.dtype == torch.float32
                and bool(torch.isfinite(value).all()), "Own exact named tensor geometry/finiteness changed")


def run(args):
    context = preflight(args)
    run_dir = Path(args.run_directory)
    require(bool(args.resume_exact) == run_dir.is_dir(), "Fresh/exact namespace existence mismatch")
    if not args.resume_exact: run_dir.mkdir(parents=True)
    memory_trace = run_dir / "stage_memory.jsonl"
    stage(memory_trace, "before_local_base_load")
    model, processor = load_model()
    teacher = {n: copy.deepcopy(getattr(model, n)).eval().requires_grad_(False)
               for n in ("encoder_projector", "decoder", "joint")}
    require(model.config.vocab_size == 8193, "Unchanged original teacher required")
    vocabulary = extend_processor_and_model(processor, model)
    a = context["source_arguments"]
    layout = install_encoder_lora(model, rank=a["rank"], alpha=a["alpha"], last_n_layers=a["last_n_layers"])
    model.config.encoder_lora_tracking = {"layout": layout,
        "implementation_sha256": record(SOURCE/"encoder_lora.py")["sha256"], "kind": "unmerged"}
    source_path = checked(context["source_checkpoint"], checkpoint=True)
    require(record(source_path) == context["source_checkpoint"], "Actual source full-state bytes changed before load")
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    audit = audit_parent_state(model, source, context, torch_module=torch)
    for adapters in (False, True):
        saved_order = tuple(n for n in source["weights"] if n.endswith(("lora_A", "lora_B")) == adapters)
        actual_order = tuple(n for n,p in model.named_parameters() if n in source["weights"]
                             and n.endswith(("lora_A", "lora_B")) == adapters)
        require(saved_order == actual_order, "Source weight insertion order differs from actual named Adam group order")
    model.load_state_dict(source["weights"], strict=False)
    expansion = expand_candidate(model, processor, source) if args.decoder_arm == "bpe512" else {"vocabulary": 8199}
    optimizer = BPEFullDecoderOptimizer(model, source, source_record=context["source_checkpoint"],
        vocabulary=model.config.vocab_size, shared_lr=args.shared_lr, new_rows_lr=args.new_rows_lr, torch_module=torch)
    original, tr, replay = load_catalogs(context)
    labels = BPETrainingLabels(context["BPE_label_directory"], tr) if args.decoder_arm == "bpe512" else None
    sampler = build_sampler(original, tr, replay, context)
    # Retain runtime ownership, not a duplicate full history or CPU moment alias.
    source.pop("weights"); source.pop("optimizer"); gc.collect()
    stage(memory_trace, "after_mature17_moment_copy_CPU_tensor_release_before_source350")
    if args.resume_exact:
        # The original S state has completed its named tensor/moment audit and
        # utility construction. Own restoration needs no duplicate S history.
        source.clear(); gc.collect()
        config_path = run_dir / "run_config.json"
        if args.decoder_runtime_repair_receipt:
            repair = json.loads(args.decoder_runtime_repair_receipt.read_text(encoding="utf-8"))
            config_path = checked(repair["original_run_config"])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        identity = config["config_identity"]
        fresh_native = native_identity(args)
        require(record(args.resume_exact) == fresh_native["exact_checkpoint"],
                "Actual exact checkpoint bytes differ from the owned closed plan")
        own = torch.load(args.resume_exact, map_location="cpu", weights_only=True)
        validate_own_state(own, args, context, identity, optimizer)
        model.load_state_dict(own["weights"], strict=False)
        optimizer.load_state_dict(own["optimizer"])
        context["own_sampling_identity"] = identity["domain_sampling_identity"]
        migration = sampler.initialize_from_own_state(own["sampler"], context, context["parent_identity"])
        require(config["initial_source_equivalence"]["passed"] is True
                and canonical(identity) == canonical(own["config_identity"]), "Original own admission changed")
        equality = json.loads(proof_path(args).read_text(encoding="utf-8"))
        if args.decoder_runtime_repair_receipt:
            from BPE_decoder_serialization_repair_v1 import rebind
            config, identity = rebind(args.decoder_runtime_repair_receipt, args, context, own, config, source_pins())
        own.pop("weights"); own.pop("optimizer")
        runtime = own
    else:
        migration = sampler.migrate_closed_state(source["sampler"], context["parent_identity"], context)
        random.setstate(source["python_rng"]); torch.set_rng_state(source["torch_rng"])
        torch.cuda.set_rng_state_all(source["cuda_rng"])
        equality = compare_source_development(model, processor, args, context, source, evaluate, torch)
        identity = {"phase": PHASE, "run_id": args.run_id,
            "decoder_adaptation": decoder_declaration(context, args.decoder_arm),
            "optimizer": optimizer.phase_identity(), "domain_sampling_identity": sampler.identity,
            "parent_identity_sha256": canonical(context["parent_identity"]),
            "source_checkpoint": context["source_checkpoint"], "source_receipt": context["receipt_record"],
            "native_loss": native_identity(args), "source_sha256": source_pins(),
            "BPE_label_preparation": context.get("BPE_label_preparation"),
            "initial350": record(proof_path(args))}
        config = {"phase": PHASE, "run_id": args.run_id, "source_step": context["source_step"],
            "configured_end_step": context["configured_end_step"], "config_identity": identity,
            "initial_source_equivalence": equality, "sampler_transition": migration,
            "source207_audit": audit, "vocabulary": expansion,
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "retention_stop_policy": "monitor_only", "encoder_LoRA_active": False,
            "all15_existing_decoder_tensors_active": True, "sampling_original8199_T_U": True,
            "initial_saved_DEV_diagnostic": diagnostic_predictions(args.run_id+"_initial_"+str(context["source_step"]))}
        write_json(run_dir / "run_config.json", config)
        write_json(run_dir / "training_pool.json", {"source_training_pool": context["source_training_pool"],
            "active_index": record(tr.directory / "index_provenance.json"), "replay": replay})
        runtime = source
        runtime["best_score"] = context["source_summary"]["validation_tr"]["asr"]["wer"]
        runtime["selected_step"] = context["source_step"]
    history = runtime["history"]
    start = runtime["step"]
    require(sampler.updates == start and optimizer.updates == start - context["source_step"], "Restored clocks disagree")
    random.setstate(runtime["python_rng"]); torch.set_rng_state(runtime["torch_rng"])
    torch.cuda.set_rng_state_all(runtime["cuda_rng"])
    state_keys = ("step", "history", "best_score", "selected_step", "patience", "retention_failures",
        "next_eval_epoch", "train_wall_seconds", "audio_seconds", "language_audio_seconds",
        "evaluation_wall_seconds", "initial_gate")
    live = {key: runtime[key] for key in state_keys}
    del runtime, source
    if args.resume_exact: del own
    gc.collect()
    journal = HistoryJournal(run_dir / "history_journal")
    if args.resume_exact: journal.reconcile_checkpoint(history, record(args.resume_exact), preserve_orphan_suffix=True)
    else: journal.seed(history, context["source_checkpoint"])
    if args.decoder_runtime_repair_receipt:
        write_json(run_dir / "run_config.json", config)
        write_json(Path(args.native_backend_provenance).parent / "exact_restore.json", {
            "passed": True, "run_id": args.run_id, "saved_step": start,
            "exact_checkpoint": record(args.resume_exact), "repair_receipt": record(args.decoder_runtime_repair_receipt),
            "mature_Adam_clock": start, "fresh_row_Adam_clock": optimizer.updates,
            "sampler_updates": sampler.updates, "retention_failures": live["retention_failures"],
            "original_own_identity_validated_before_operational_rebind": True,
            "journal": journal.frontier(), "actual_PID": os.getpid()})
    def publish(label):
        journal.publish(history, publication=label)
        if label != "periodic": write_json(run_dir / "training_history.json", history)
    def save(path, step, reason=None):
        live.update(step=step, history=history, stop_reason=reason)
        save_state(path, model, optimizer, sampler, context, identity, live)
    best_path, latest_path = run_dir / "best_checkpoint.pt", run_dir / "latest_checkpoint.pt"
    if not args.resume_exact:
        save(best_path, start, "genuine_qualified_source_before_updates"); save(latest_path, start)
        write_json(run_dir / "selection.json", {"run_id": args.run_id, "phase": PHASE,
            "selected_step": start, "turkish_validation_wer": live["best_score"],
            "latest_step": start, "latest_gate": gate(context["source_summary"]),
            "retention_failures": live["retention_failures"], "decoder_updates": 0,
            "strict_all_six_winner_only": True, "coverage": sampler.coverage(),
            "best_checkpoint": str(best_path)})
    publish("checkpoint")
    stage(memory_trace, "training_ready", step=start, source350_passed=equality["passed"])
    model.train(); model.encoder.eval()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    last_dev = start
    def dev(step, final=False):
        nonlocal last_dev
        optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        before = time.perf_counter()
        scores = evaluate(model, processor, args.run_id + "_step_" + str(step))
        live["evaluation_wall_seconds"] += time.perf_counter() - before
        value = gate(scores)
        history[-1].update(validation=value, development_selection_only=True,
                           final_development_evaluation=final,
                           selection_diagnostics=diagnostic_predictions(args.run_id+"_step_"+str(step)))
        live["retention_failures"] = 0 if value["retention_gate_passed"] else live["retention_failures"] + 1
        improved = value["retention_gate_passed"] and value["turkish_wer"] < live["best_score"]
        if improved: live.update(best_score=value["turkish_wer"], selected_step=step)
        publish("DEV")
        if improved: save(best_path, step, "strict_all_six_qualified_improvement")
        save(latest_path, step)
        write_json(run_dir / "selection.json", {"run_id": args.run_id, "phase": PHASE,
            "selected_step": live["selected_step"], "turkish_validation_wer": live["best_score"],
            "latest_step": step, "latest_gate": value, "retention_failures": live["retention_failures"],
            "strict_all_six_winner_only": True, "decoder_updates": step-context["source_step"],
            "coverage": sampler.coverage(), "best_checkpoint": str(best_path)})
        last_dev = step
        print(f"DEV step{step} TR{value['turkish_wer']:.6f} all6={value['retention_gate_passed']} best{live['selected_step']}", flush=True)
        model.train(); model.encoder.eval(); torch.cuda.empty_cache()
    stop_reason = "configured_steps_complete"
    for step in range(start + 1, context["configured_end_step"] + 1):
        if args.stop_file and Path(args.stop_file).exists():
            stop_reason = "controller_stop_file"; break
        before = time.perf_counter(); optimizer.zero_grad(set_to_none=True)
        totals = {}; original_ids = []; model_target_U = []; sampling_U = []; actual_TR_weights = []
        for rows in sampler.next_batches():
            original_ids.extend(r["source_id"] for r in rows)
            sampling_U.extend(int(r["target_tokens"]) for r in rows)
            batch_rows = labels.rewrite_batch(rows) if labels is not None else rows
            model_target_U.extend(int(r["target_tokens"]) for r in batch_rows)
            batch = collate_training_batch(processor, batch_rows, blank_id=8192, old_size=8193, device="cuda")
            result, exposure = objective(model, teacher, batch, maximum_joint_bytes=args.native_max_joint_bytes)
            require(bool(torch.isfinite(result.total)), "Nonfinite objective")
            (result.total / 2).backward()
            for key, value in TR_loss_metrics(result, exposure, torch_module=torch).items():
                totals[key] = totals.get(key, 0.) + value / 2
            actual_TR_weights.extend(exposure["actual_TR_weights"])
            for row in rows:
                seconds = float(row["duration"])
                live["audio_seconds"] += seconds
                live["language_audio_seconds"][row["language"]] += seconds
            del result, batch
        recovery_first_two = bool(args.decoder_runtime_repair_receipt and step <= start + 2)
        before_parameters = partition_fingerprints(optimizer) if optimizer.updates < 2 or recovery_first_two else None
        norm = optimizer.step()
        torch.cuda.synchronize()  # Complete one actual native update before recording progress.
        live["train_wall_seconds"] += time.perf_counter() - before
        live["step"] = step
        history.append({"step": step, "phase": PHASE, "decoder_updates": optimizer.updates,
            "mature_Adam_clock": context["source_step"] + optimizer.updates,
            "fresh_row_Adam_clock": optimizer.updates if labels is not None else None,
            "frozen_encoder_Adam_clock": context["source_step"],
            "head_lr": args.shared_lr, "new_rows_lr": args.new_rows_lr if labels is not None else None,
            "lora_lr": 0., "encoder_LoRA_active": False, "gradient_norm": float(norm), **totals,
            "source_ID_sequence_sha256": canonical(original_ids), "sampling_target_tokens": sampling_U,
            "model_target_tokens": model_target_U, "actual_TR_TDT_weights": actual_TR_weights,
            "coverage": sampler.coverage(),
            "train_wall_seconds": live["train_wall_seconds"], "audio_seen_seconds": live["audio_seconds"],
            "language_audio_seen_seconds": dict(live["language_audio_seconds"]),
            "turkish_audio_seen_seconds": live["language_audio_seconds"]["tr"],
            "replay_audio_seen_seconds": live["audio_seconds"]-live["language_audio_seconds"]["tr"],
            "evaluation_wall_seconds": live["evaluation_wall_seconds"]})
        if optimizer.updates <= 2 or optimizer.updates % 250 == 0 or recovery_first_two:
            delta_evidence = None
            if before_parameters is not None:
                after_parameters = partition_fingerprints(optimizer)
                delta_evidence = {"before": before_parameters, "after": after_parameters,
                    "changed_mature_partitions": [k for k in before_parameters
                        if k.startswith("mature_") and before_parameters[k] != after_parameters[k]],
                    "changed_fresh_partitions": [k for k in before_parameters
                        if k.startswith("fresh_") and before_parameters[k] != after_parameters[k]],
                    "frozen192_unchanged": optimizer._frozen_digest() == optimizer.frozen_lora_sha256,
                    "actual_native_backward_optimizer_update": True}
                require(delta_evidence["frozen192_unchanged"], "Frozen encoder LoRA changed during decoder update")
            stage(memory_trace, "after_actual_decoder_optimizer", step=step,
                  mature_Adam_clock=step, fresh_row_Adam_clock=optimizer.updates if labels else None,
                  actual_parameter_delta_evidence=delta_evidence)
        if optimizer.updates % 10 == 0:
            publish("periodic")
            print(f"optimizer step{step} decoder{optimizer.updates}/{UPDATES}", flush=True)
        if optimizer.updates % 250 == 0: dev(step)
    completed = live["step"]
    if last_dev != completed: dev(completed, final=True)
    elif completed > context["source_step"]:
        history[-1]["final_development_evaluation"] = True
    publish("end"); save(latest_path, completed, stop_reason)
    write_json(run_dir / "worker_completion.json", {"run_id": args.run_id, "phase": PHASE,
        "completed_step": completed, "decoder_updates": optimizer.updates, "stop_reason": stop_reason,
        "final350_complete": last_dev == completed, "source350": record(proof_path(args)),
        "sources_unchanged": source_pins() == identity["source_sha256"],
        "mature_Adam_clock": completed, "fresh_row_Adam_clock": optimizer.updates if labels else None,
        "frozen192_unchanged": optimizer._frozen_digest() == optimizer.frozen_lora_sha256,
        "selected_step": live["selected_step"], "best_TR_WER": live["best_score"],
        "coverage": sampler.coverage(), "actual_PID": os.getpid()})
    if labels: labels.close()
    if tr is not original: tr.close()
    original.close()


def source_pins():
    names = (Path(__file__).name, "bpe_full_decoder_optimizer_v1.py", "bpe_training_label_catalog_v1.py",
        "tsc_bpe_full_decoder_sampler_v1.py", "tsc_bpe_full_decoder_transition_v1.py",
        "native_tdt_checkpointed_v4.py", "append_only_history_v1.py", "batched_lora_tr_metadata_v1.py",
        "TR_tdt_population_weighting_v1.py", "encoder_lora.py", "extend_vocab.py",
        "lazy_training_rows_v1.py", "expanded_training_rows_v1.py", "tsc_expanded_TRAIN_sampler_v1.py",
        "BPE_decoder_serialization_repair_v1.py")
    return {name: record(SOURCE / name)["sha256"] for name in names}


def native_identity(args):
    value = json.loads(Path(args.native_backend_provenance).read_text(encoding="utf-8"))
    require(value["passed"] is True and value["pid"] == os.getpid() and value["ppid"] == os.getppid()
            and value["run_id"] == args.run_id
            and value["source_receipt"] == record(args.decoder_source_receipt)
            and vars(parser().parse_args(value["worker_arguments"])) == vars(args)
            and value["worker_sha256"] == record(__file__)["sha256"]
            and value["native_adapter_sha256"] == record(SOURCE / "native_tdt_checkpointed_v4.py")["sha256"]
            and value["CUDA_version"] == "12.9" and value["native_batch_cap_bytes"] == 8000000000
            and value["native_end_cache_bytes"] == 3000000000
            and value["maximum_full_joint_bytes"] == args.native_max_joint_bytes,
            "Fresh hash-bound coherent native runtime provenance required")
    return value


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-directory", type=Path, required=True)
    p.add_argument("--decoder-source-receipt", type=Path, required=True)
    p.add_argument("--decoder-arm", choices=("bpe512", "original8199"), required=True)
    p.add_argument("--decoder-updates", type=int, default=UPDATES)
    p.add_argument("--shared-lr", type=float, default=1e-5)
    p.add_argument("--new-rows-lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accumulation", type=int, default=2)
    p.add_argument("--native-max-joint-bytes", type=int, default=1700000000)
    p.add_argument("--native-backend-provenance", type=Path, required=True)
    p.add_argument("--resume-exact", type=Path)
    p.add_argument("--decoder-runtime-repair-receipt", type=Path)
    p.add_argument("--stop-file", type=Path)
    return p


def validate_arguments(args):
    require(args.decoder_updates == UPDATES and args.batch_size == 8 and args.accumulation == 2
            and args.shared_lr == 1e-5 and args.new_rows_lr == 1e-4
            and args.native_max_joint_bytes == 1700000000, "Only the declared matched decoder recipe")
    preflight(args); native_identity(args)


if __name__ == "__main__":
    arguments = parser().parse_args(); validate_arguments(arguments); run(arguments)
