"""INACTIVE PT-focused replay full-state fork of a CLOSED lower-LoRA majority frontier.

All207 weights/Adam, actual TR queue suffix, six replay queues/RNG, inherited
history/failure/evaluation clocks carry. Initial350 reproduces sourceH exactly,
including failed gates. Original uniform origin6500 and PT grace are retained.
Quota/LRs/pass carry; onePT+oneoldfive replay and per-language preservation are declared.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch

from batched_lora_tr_metadata_v1 import collate_training_batch, batch_objective
from tsc_training_inputs_v2 import confirm_replay_after_cache
from lazy_training_rows_v1 import IndexedTrainingRows, prepare_approved_TR_rows, reopen_approved_TR_rows
from tsc_uniform_majority_ptfocus_transition_v1 import (preflight as majority_preflight,
    audit_parent_state, compare_source_development, validate_portuguese_inputs,
    proof_path, ptfocus_declaration, restore_parent_sampler)
from tsc_uniform_full_transition_v1 import audit_source_optimizer, record as file_record
from tsc_uniform_majority_ptfocus_sampler_v1 import TSCUniformMajorityPTFocusSampler
from benchmark import LANGUAGES as ORIGINAL_LANGUAGES, evaluate as original_evaluate, read_jsonl, write_json
LANGUAGES = (*ORIGINAL_LANGUAGES, "pt")
from efficient_tdt import chunked_tdt_loss as original_tdt
from encoder_lora import install_encoder_lora, encoder_lora_metadata
from extend_vocab import extend_processor_and_model
from metrics import normalize_text
from runtime import ROOT, DATA, OUTPUT, load_model
from train import make_cache, checkpoint_state
from train_lora import checkpoint_encoder_blocks

BASE_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ids_sha(rows):
    if isinstance(rows, IndexedTrainingRows): return rows.ids_sha
    return hashlib.sha256("\n".join(sorted(row["source_id"] for row in rows)).encode()).hexdigest()


class BucketCycler:
    """Shuffled cycles, sorted only inside bounded shuffled reservoirs.

    Each cycle contains every source exactly once. Two-dimensional log-length
    bins avoid pairing long-T/short-U with short-T/long-U by scalar cost alone.
    """
    def __init__(self, rows, rng, bucket_size=128):
        self.by_id = {row["source_id"]: row for row in rows}
        if len(self.by_id) != len(rows) or not rows or bucket_size < 1:
            raise ValueError("Each positive sampler pool needs unique IDs and positive bucket size")
        if any(int(row["encoder_frames"]) < 1 or int(row["target_tokens"]) < 1 for row in rows):
            raise ValueError("Length buckets require positive encoder frames and target tokens")
        self.rows = list(rows)
        self.rng = rng
        self.bucket_size = bucket_size
        self.position = len(rows)
        self.draws = 0
        self.cycles_started = 0
        self.seen = set()
        self.pool_sha256 = ids_sha(rows)

    @staticmethod
    def length_key(row):
        t = int(row["encoder_frames"])
        u = int(row["target_tokens"])
        return (int(math.log2(t)), int(math.log2(u)), t, u, row["source_id"])

    def next(self):
        if self.position == len(self.rows):
            self.rng.shuffle(self.rows)
            for start in range(0, len(self.rows), self.bucket_size):
                self.rows[start:start+self.bucket_size] = sorted(self.rows[start:start+self.bucket_size], key=self.length_key)
            self.position = 0
            self.cycles_started += 1
        row = self.rows[self.position]
        self.position += 1
        self.draws += 1
        self.seen.add(row["source_id"])
        return row

    def state_dict(self):
        return {"pool_sha256": self.pool_sha256, "order": [row["source_id"] for row in self.rows],
                "position": self.position, "draws": self.draws, "cycles_started": self.cycles_started,
                "seen": sorted(self.seen), "bucket_size": self.bucket_size}

    def load_state_dict(self, state):
        if (state["pool_sha256"] != self.pool_sha256 or state["bucket_size"] != self.bucket_size
                or len(state["order"]) != len(self.rows) or set(state["order"]) != set(self.by_id)):
            raise ValueError("Exact-resume sampler pool/bucket identity changed")
        if not 0 <= state["position"] <= len(self.rows) or not set(state["seen"]).issubset(self.by_id):
            raise ValueError("Invalid sampler cursor or seen IDs")
        n = len(self.rows)
        cycles = state["cycles_started"]
        expected_draws = 0 if cycles == 0 else (cycles-1)*n + state["position"]
        expected_seen = (set() if cycles == 0 else set(state["order"][:state["position"]])
                         if cycles == 1 else set(self.by_id))
        if (cycles < 0 or state["draws"] != expected_draws or set(state["seen"]) != expected_seen
                or (cycles == 0 and state["position"] != n)):
            raise ValueError("Exact-resume sampler cycle/exposure counters disagree")
        self.rows = [self.by_id[source] for source in state["order"]]
        self.position, self.draws, self.cycles_started = state["position"], state["draws"], state["cycles_started"]
        self.seen = set(state["seen"])


class BalancedBucketSampler:
    def __init__(self, tr, replay, *, seed, batch_size, accumulation, replay_fraction, bucket_size=128):
        self.rng = random.Random(seed + 1)
        self.batch_size = batch_size
        self.accumulation = accumulation
        self.effective_batch = batch_size * accumulation
        self.replay_per_update = round(self.effective_batch * replay_fraction)
        if (batch_size < 1 or accumulation < 1
                or not math.isclose(self.replay_per_update, self.effective_batch * replay_fraction, abs_tol=1e-9)
                or not 0 <= self.replay_per_update < self.effective_batch):
            raise ValueError("Replay fraction must represent an integer clip count with positive TR at effective batch size")
        self.tr_per_update = self.effective_batch - self.replay_per_update
        self.tr = BucketCycler(tr, self.rng, bucket_size)
        by_lang = {lang: [row for row in replay if row["language"] == lang] for lang in LANGUAGES if lang != "tr"}
        self.replay = {lang: BucketCycler(rows, self.rng, bucket_size) for lang, rows in by_lang.items()}
        self.replay_language_counts = {lang: 0 for lang in self.replay}

    @property
    def turkish_epoch(self):
        return self.tr.draws / len(self.tr.rows)

    def next_batches(self):
        tr = [self.tr.next() for _ in range(self.tr_per_update)]
        replay = []
        for _ in range(self.replay_per_update):
            lang = self.rng.choice(list(self.replay))
            replay.append(self.replay[lang].next())
            self.replay_language_counts[lang] += 1
        batches, remainder = [], []
        for stream in (tr, replay):
            full = len(stream) // self.batch_size * self.batch_size
            batches += [stream[start:start+self.batch_size] for start in range(0, full, self.batch_size)]
            remainder += stream[full:]
        if remainder:
            if len(remainder) != self.batch_size:
                raise RuntimeError("Balanced batch remainder must fill one mixed microbatch")
            batches.append(remainder)
        if len(batches) != self.accumulation or any(len(batch) != self.batch_size for batch in batches):
            raise RuntimeError("Sampler changed effective clip batch size")
        return batches

    def state_dict(self):
        return {"rng": self.rng.getstate(), "tr": self.tr.state_dict(),
                "replay": {lang: stream.state_dict() for lang, stream in self.replay.items()},
                "replay_language_counts": self.replay_language_counts.copy(),
                "batch_size": self.batch_size, "accumulation": self.accumulation,
                "replay_per_update": self.replay_per_update}

    def load_state_dict(self, state):
        if (state["batch_size"] != self.batch_size or state["accumulation"] != self.accumulation
                or state["replay_per_update"] != self.replay_per_update
                or list(state["replay"]) != list(self.replay)):
            raise ValueError("Exact-resume sampler mix/layout changed")
        self.tr.load_state_dict(state["tr"])
        for lang, stream in self.replay.items():
            stream.load_state_dict(state["replay"][lang])
        if (set(state["replay_language_counts"]) != set(self.replay)
                or any(state["replay_language_counts"][lang] != stream.draws for lang, stream in self.replay.items())):
            raise ValueError("Exact-resume replay language counters disagree")
        self.replay_language_counts = state["replay_language_counts"].copy()
        self.rng.setstate(state["rng"])


def lr_factor(step, total_steps, warmup_steps, schedule, min_lr_ratio=.1):
    if warmup_steps and step <= warmup_steps:
        return step / warmup_steps
    if schedule == "constant":
        return 1.0
    progress = min(1., max(0., (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    return min_lr_ratio + (1-min_lr_ratio)*.5*(1+math.cos(math.pi*progress))


def next_evaluation_epoch(current, first_fraction, evaluations_per_epoch):
    initial = sorted({first_fraction, .5, 1.0} if first_fraction < .5 else {first_fraction, 1.0})
    for threshold in initial:
        if threshold > current + 1e-10:
            return threshold
    interval = 1 / evaluations_per_epoch
    return 1 + (math.floor((current - 1 + 1e-10) / interval) + 1) * interval


def development_gate(dev, baseline):
    deltas = {lang: dev[f"validation_{lang}"]["asr"]["wer"] - baseline[f"validation_{lang}"]["asr"]["wer"]
              for lang in LANGUAGES if lang != "tr"}
    limits = {lang: max(.005, .1*baseline[f"validation_{lang}"]["asr"]["wer"]) for lang in deltas}
    return {"turkish_wer": dev["validation_tr"]["asr"]["wer"],
            "retention_wer_deltas": deltas, "retention_wer_limits": limits,
            "retention_gate_passed": all(dev[f"validation_{lang}"]["asr"]["wer"]
                                         <= baseline[f"validation_{lang}"]["asr"]["wer"] + limits[lang]
                                         for lang in deltas)}


def evaluate(model, processor, tag, *, splits=("validation",)):
    if splits != ("validation",): raise ValueError("This full-training controller evaluates fixed DEV only")
    import benchmark
    result = original_evaluate(model, processor, tag, languages=ORIGINAL_LANGUAGES, splits=splits)
    original_data = benchmark.DATA
    try:
        benchmark.DATA = DATA / "portuguese_retention_v1"
        portuguese = original_evaluate(model, processor, tag + "_pt", languages=("pt",), splits=splits)
    finally:
        benchmark.DATA = original_data
    result.update(portuguese)
    # Keep the original six-language metadata identical for source300 proof.
    # Actual Portuguese predictions/metadata are separately produced and bound.
    import shutil
    for name, destination in (("validation_pt.jsonl", "validation_pt.jsonl"),
                              ("metadata.json", "metadata_pt.json"),
                              ("summary.json", "summary_pt.json")):
        shutil.copy2(OUTPUT / "results" / (tag + "_pt") / name,
                     OUTPUT / "results" / tag / destination)
    write_json(OUTPUT / "results" / tag / "summary.json", result)
    return result


def validate_trainable_checkpoint(model, saved, layout):
    expected = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(saved["weights"]) != set(expected):
        raise ValueError("Warm start must contain exactly all current trainable keys")
    if saved.get("lora_metadata") != layout:
        raise ValueError("Warm-start adapter rank/alpha/layers/layout differs; use a fresh run for a different layout")
    for name, weight in saved["weights"].items():
        if weight.shape != expected[name].shape or weight.dtype != torch.float32 or not torch.isfinite(weight).all():
            raise ValueError("Warm-start shape/FP32/finiteness differs for " + name)


def transition_identity(args, context, migration):
    return {"kind": "TSC_uniform_majority_PT_replay_state_transition_v1",
        "source_receipt": context["receipt_record"], "source_checkpoint": context["source_checkpoint"],
        "parent_config_identity_sha256": canonical_identity_sha(context["parent_identity"]),
        "declared": ptfocus_declaration(context), "sampler_migration": migration,
        "fresh_optimizer_or_warmup": False, "Portuguese_grace_reset":False, "source_failed_gates_preserved":True}


def validate_own_state(state, args, context, current, *, model, torch_module):
    step = state.get("step"); source_step = context["source_step"]
    if (state.get("schema") != 2 or state.get("run_id") != args.run_id
            or type(step) is not int or not source_step <= step <= context["configured_end_step"]
            or canonical_identity_sha(state.get("config_identity")) != canonical_identity_sha(current)
            or len(state.get("history", [])) != step
            or any(row.get("step") != index for index, row in enumerate(state["history"], 1))
            or state["sampler"].get("updates") != step
            or canonical_identity_sha(state["sampler"].get("identity")) != canonical_identity_sha(current["domain_sampling_identity"])):
        raise ValueError("Own SAME-ID full207/Adam/global-history/uniform-sampler identity changed")
    prefix=hashlib.sha256();prefix.update(b"[")
    for index,row in enumerate(state["history"][:source_step]):
        if index:prefix.update(b",")
        prefix.update(json.dumps(row,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode())
    prefix.update(b"]")
    if prefix.hexdigest()!=context["source_history_prefix_sha256"]:
        raise ValueError("Own exact state changed inherited CLOSED frontier history")
    selected=state.get("selected_step");best=state.get("best_score")
    if selected is None:
        if best!=math.inf:raise ValueError("No all-six selected frontier must retain uninitialized BEST score")
    else:
        if type(selected) is not int or not context["uniform_origin_step"]<selected<=step or not math.isfinite(best):
            raise ValueError("Invalid own selected BEST frontier")
        gate=state["history"][selected-1].get("validation",{})
        if gate.get("retention_gate_passed") is not True or gate.get("turkish_wer")!=best:
            raise ValueError("Own BEST must bind an actual all-six-qualified saved validation")
    for key in ("optimizer", "python_rng", "torch_rng", "cuda_rng", "best_score", "selected_step", "patience",
                "retention_failures", "next_eval_epoch", "train_wall_seconds", "audio_seconds",
                "language_audio_seconds", "evaluation_wall_seconds", "initial_gate"):
        if key not in state: raise ValueError("Incomplete exact full-run frontier: " + key)
    # The same hardened audit verifies all15 heads/192 adapters, complete FP32
    # moments/nonnegative variance/parameter order and actual global Adam clocks.
    expected = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    own_context = {**context, "source_selected_step": step, "source_step": step}
    audit_source_optimizer(state, own_context, expected, torch_module=torch_module)


def stage_memory(path, stage, **details):
    """Actual current-PID paired host sample plus initialized CUDA counters."""
    from launch_with_memory_guard_TSC_resume_v3 import safe_sample
    value = {"stage": stage, "pid": os.getpid(), "at_unix_seconds": time.time(),
             "paired_memory": safe_sample(os.getpid(), stage), **details}
    if torch.cuda.is_initialized():
        value["CUDA"] = {"allocated_bytes": torch.cuda.memory_allocated(),
                         "reserved_bytes": torch.cuda.memory_reserved(),
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    return value


def write_pool_summary(path, rows, replay, input_sha):
    from atomic_io_retry_v1 import write_json as atomic_write
    atomic_write(path, {"kind": "indexed_full_TRAIN_pool_summary_v1", "turkish_clips": len(rows),
        "turkish_index": file_record(rows.directory / "index_provenance.json"),
        "complete_turkish_row_values": rows.receipt, "replay": replay,
        "input_sha256": input_sha, "resident_TR_rowdict_pool": False,
        "TR_cache_cap_rows": 256, "all_original_annotated_fields_retained_on_disk": True})


def load_indexed_pool(args, processor, run_dir, heldout_paths, context):
    audit_path = run_dir / "indexed_TR_audit.json"
    if args.resume_exact:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        tr,audit = reopen_approved_TR_rows(args,args.indexed_tr_directory,audit,cache_size=256)
    else:
        # Reopen the complete source-admitted immutable carrier. This validates
        # its bytes/provenance/input contract without rebuilding253798 targets.
        audit=copy.deepcopy(context["indexed_audit"])
        tr,audit=reopen_approved_TR_rows(args,args.indexed_tr_directory,audit,cache_size=256)
        write_json(audit_path,audit)
    return tr, audit














def annotate_cache_lengths(rows):
    result = []
    for row in rows:
        cached = torch.load(row["cache_path"], map_location="cpu", weights_only=True)
        if cached["source_id"] != row["source_id"] or cached["features"].ndim != 2 or not torch.isfinite(cached["features"]).all():
            raise ValueError("Invalid original encoder cache")
        if (cached["features"].shape[0] < 1 or cached["labels"].ndim != 1 or cached["labels"].numel() < 1
                or bool((cached["labels"] < 0).any())):
            raise ValueError("Original cache has invalid frame/target lengths or negative target IDs")
        result.append({**row, "encoder_frames": int(cached["features"].shape[0]),
                       "target_tokens": int(cached["labels"].numel())})
    return result


def save_training_state(path, model, optimizer, sampler, *, run_id, step, config_identity,
                        history, best_score, selected_step, patience, retention_failures,
                        next_eval_epoch, train_wall_seconds, audio_seconds, language_audio_seconds,
                        evaluation_wall_seconds, initial_gate, stop_reason=None):
    state = {"schema": 2, "run_id": run_id, "step": step, "weights": checkpoint_state(model),
             "lora_metadata": encoder_lora_metadata(model), "optimizer": optimizer.state_dict(),
             "sampler": sampler.state_dict(), "python_rng": random.getstate(),
             "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
             "config_identity": config_identity, "history": history, "best_score": best_score,
             "selected_step": selected_step, "patience": patience, "retention_failures": retention_failures,
             "next_eval_epoch": next_eval_epoch, "train_wall_seconds": train_wall_seconds,
             "audio_seconds": audio_seconds, "language_audio_seconds": language_audio_seconds,
             "evaluation_wall_seconds": evaluation_wall_seconds, "initial_gate": initial_gate,
             "stop_reason": stop_reason}
    temporary = Path(str(path) + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)



def preservation_row_scales(rows):
    """Exact per-clip density declaration; TR preservation remains literal0."""
    scales=[]
    for row in rows:
        language=row["language"]
        if language=="tr":scales.append(0.)
        elif language=="pt":scales.append(1.)
        elif language in ("en","es","it","fr","de"):scales.append(5/3)
        else:raise ValueError("Unapproved preservation language: "+str(language))
    return scales


def pt_focused_batch_objective(model,teacher,batch,*,replay_kl,duration_kl,feature_distillation,**kwargs):
    # Frozen native forward, per-targetU TDT and full-vocab teacher losses run
    # exactly once. Reweight their differentiable per-clip outputs, mean16 via
    # the unchanged physical-batch mean/accumulation2 convention.
    from dataclasses import replace
    result=batch_objective(model,teacher,batch,replay_kl=0.,duration_kl=0.,feature_distillation=0.,**kwargs)
    scale=torch.tensor(preservation_row_scales(batch.rows),dtype=torch.float32,device=result.per_clip["tdt"].device)
    total=(result.per_clip["tdt"]+replay_kl*scale*result.per_clip["token_kl"]
        +duration_kl*scale*result.per_clip["duration_kl"]+feature_distillation*scale*result.per_clip["feature_distillation"])
    return replace(result,total=total.mean(),per_clip={**result.per_clip,"total":total})


def run(args):
    process_start=time.perf_counter();context=majority_preflight(args)
    validate_portuguese_inputs(args,context)
    random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    run_dir=OUTPUT/"runs"/args.run_id
    best_path=ROOT/"work"/(args.run_id+"_best.pt");latest_path=ROOT/"work"/(args.run_id+"_latest.pt")
    if args.resume_exact:
        if not run_dir.is_dir(): raise ValueError("Exact full-run namespace missing")
    else:
        if run_dir.exists() or best_path.exists() or latest_path.exists(): raise ValueError("Fresh full-run namespace required")
        run_dir.mkdir(parents=True)
    memory_trace=run_dir/"stage_memory.jsonl"
    stage_memory(memory_trace,"before_model_load")
    model,processor=load_model();baseline=context["combined_baseline"]
    stage_memory(memory_trace,"after_model_load")
    old_heldout=[DATA/f"{split}_{lang}.jsonl" for lang in ORIGINAL_LANGUAGES for split in ("validation","test")]
    heldout_paths=[*old_heldout,DATA/"portuguese_retention_v1/validation_pt.jsonl",DATA/"portuguese_retention_v1/test_pt.jsonl"]
    heldout=[row for path in heldout_paths for row in read_jsonl(path)]
    replay=read_jsonl(args.replay_manifest)
    pt_rows=read_jsonl(args.portuguese_replay_manifest)
    if len(replay)!=1595 or len(pt_rows)!=319: raise ValueError("Complete approved originalRP/PT pools required")
    all_heldout_ids={row["source_id"] for row in heldout}
    if any(row["source_id"] in all_heldout_ids for row in pt_rows): raise ValueError("Portuguese replay heldout collision")
    del heldout,all_heldout_ids
    teacher={name:copy.deepcopy(getattr(model,name)).eval().requires_grad_(False) for name in ("encoder_projector","decoder","joint")}
    stage_memory(memory_trace,"after_original8193_teacher")
    old_size=model.config.vocab_size
    if old_size!=8193:raise ValueError("Original8193 teacher required")
    vocabulary=extend_processor_and_model(processor,model)
    if args.phonetic_sh_donors:
        for new,donor in (("ÅŸ","È™"),("Åž","È˜")):
            new_id,donor_id=processor.tokenizer.convert_tokens_to_ids(new),processor.tokenizer.convert_tokens_to_ids(donor)
            with torch.no_grad():
                model.decoder.embedding.weight[new_id].copy_(model.decoder.embedding.weight[donor_id])
                model.joint.head.weight[new_id].copy_(model.joint.head.weight[donor_id])
                model.joint.head.bias[new_id].copy_(model.joint.head.bias[donor_id]-2)
    if model.config.vocab_size!=8199 or model.config.blank_token_id!=8192:raise ValueError("Original8199/sixglyph geometry required")
    stage_memory(memory_trace,"before_streamed_TR_admission")
    tr,public_data_audit=load_indexed_pool(args,processor,run_dir,heldout_paths,context)
    stage_memory(memory_trace,"after_streamed_TR_admission",TR_rows=len(tr),TR_cache_cap_rows=256)
    counts_before={"tr":len(tr),"replay":len(replay),"Portuguese":len(pt_rows)}
    namespace=hashlib.sha256((BASE_REVISION+processor.tokenizer._tokenizer.to_str()).encode()).hexdigest()[:20]
    cache=ROOT/"work/encoder-cache"/namespace
    for parameter in model.encoder.parameters():parameter.requires_grad_(False)
    replay=annotate_cache_lengths(make_cache(model,processor,replay,cache,args.max_tokens))
    confirm_replay_after_cache(replay,public_data_audit)
    pt_cached=annotate_cache_lengths(make_cache(model,processor,pt_rows,cache,args.max_tokens))
    if len(pt_cached)!=319:raise ValueError("No Portuguese TRAIN crop/drop")
    for row in pt_cached:
        ids=processor.tokenizer(row["text"])["input_ids"]
        if not ids or max(ids)>=old_size:raise ValueError("Portuguese teacher labels must stay original8193")
    stage_memory(memory_trace,"after_original_and_Portuguese_disk_cache_headers")
    layout=install_encoder_lora(model,rank=args.rank,alpha=args.alpha,last_n_layers=args.last_n_layers)
    model.config.encoder_lora_tracking={"layout":layout,"implementation_sha256":file_sha(OUTPUT/"encoder_lora.py"),"kind":"unmerged"}
    if not args.no_encoder_checkpoint:checkpoint_encoder_blocks(model)
    # Source remains a full207/Adam artifact, not weights-only warm start.
    source=torch.load(args.resume_majority,map_location="cpu",weights_only=True)
    stage_memory(memory_trace,"after_CLOSED_frontier_CPU_state_load")
    audit_parent_state(model,source,context,torch_module=torch)
    model.load_state_dict(source["weights"],strict=False)
    adapters=[p for name,p in model.named_parameters() if p.requires_grad and name.endswith(("lora_A","lora_B"))]
    heads=[p for name,p in model.named_parameters() if p.requires_grad and not name.endswith(("lora_A","lora_B"))]
    trainable=adapters+heads
    optimizer=torch.optim.AdamW([{"params":heads,"lr":args.head_lr,"weight_decay":.01},
                                {"params":adapters,"lr":args.lora_lr,"weight_decay":0.}])
    if not args.resume_exact:
        # Admission above verifies all207 original FP32 moments/group clocks.
        # Copy them to the live optimizer before decode, then release the CPU
        # checkpoint tensors. History/RNG/sampler/counters retain ownership.
        optimizer.load_state_dict(source["optimizer"])
        optimizer.param_groups[1]["lr"]=args.lora_lr
        if len(optimizer.state)!=207:raise ValueError("All207 named Adam states required")
        del source["weights"],source["optimizer"]
        import gc
        gc.collect()
        stage_memory(memory_trace,"after_all207_GPU_copy_CPU_checkpoint_tensor_release_before_source350")
    sampler=TSCUniformMajorityPTFocusSampler(tr,replay,portuguese_replay=pt_cached,portuguese_seed=args.portuguese_seed,
        seed=args.seed,replay_seed=args.replay_seed,sampling_mode="uniform",batch_size=args.batch_size,
        accumulation=args.accumulation,replay_fraction=args.replay_fraction,fleurs_fraction=args.fleurs_fraction,
        bucket_size=args.bucket_size,cycler_factory=BucketCycler,cycler_source_sha256=file_sha(OUTPUT/"train_lora_batched_data_v11.py"))
    sampler_transition=restore_parent_sampler(sampler,source,context)
    source_step=context["source_step"];uniform_origin=context["uniform_origin_step"];quota_boundary=context["quota_boundary_step"];context_lora_origin=context["lora_boundary_step"]
    configured_steps=sampler.configured_end_step
    if (uniform_origin!=6500 or configured_steps!=context["configured_end_step"]
            or configured_steps-source_step!=context["remaining_updates_from_chosen_frontier"]
            or sampler.turkish_streams["uniform"].draws!=context["prior_uniform_draws"] or sampler.transition_source_step!=quota_boundary):
        raise ValueError("Declared same-pass14/2 remaining frontier/horizon changed")
    source_best_checkpoint=context.get("source_best_checkpoint")
    parked_original_champion=context["parked_original_champion"]
    all_replay=[*replay,*pt_cached]
    loss_function=native_loss_function(args,tr,all_replay)
    from native_tdt_checkpointed_v4 import reclaim_inactive_cuda_cache
    if not args.resume_exact:
        random.setstate(source["python_rng"]);torch.set_rng_state(source["torch_rng"]);torch.cuda.set_rng_state_all(source["cuda_rng"])
        equality=compare_source_development(model,processor,args,context,source,evaluate,torch)
        stage_memory(memory_trace,"after_actual_source_H_fixed350_DEV")
    else:
        equality=json.loads(proof_path(args).read_text(encoding="utf-8"))
    source_sha=dict(context["parent_identity"]["source_sha256"])
    for name in (Path(__file__).name,"run_TSC_uniform_majority_ptfocus_v1.py","tsc_uniform_majority_ptfocus_transition_v1.py","tsc_uniform_majority_ptfocus_sampler_v1.py",
        "native_tdt_checkpointed_v4.py","lazy_training_rows_v1.py"):
        source_sha[name]=file_sha(OUTPUT/name)
    input_paths=[args.majority_source_receipt,args.uniform_full_source_receipt,proof_path(args),args.tsc_input_proof,args.tsc_training_approval,args.source_closure,
        args.tr_manifest,args.replay_manifest,args.data_v3_proof,args.tr_frame_metadata,args.tr_frame_metadata_proof,
        args.portuguese_replay_manifest,args.portuguese_dev_manifest,args.portuguese_baseline_proof,args.native_backend_provenance,
        tr.directory/"index_provenance.json",run_dir/"indexed_TR_audit.json",*heldout_paths]
    input_sha={**context["parent_identity"]["input_sha256"],**{str(path.resolve()):file_sha(path) for path in input_paths}}
    arguments={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    science_keys=("sampling_mode","fleurs_fraction","replay_seed","portuguese_seed","epochs","steps","batch_size","accumulation","replay_fraction",
        "bucket_size","seed","max_tokens","head_lr","lora_lr","new_token_lr_multiplier","replay_kl","duration_kl","feature_distillation",
        "rank","alpha","last_n_layers","chunk_size","tdt_implementation","warmup_steps","schedule","min_lr_ratio","eval_per_epoch",
        "eval_epoch_fraction","eval_every_updates","min_improvement","no_encoder_checkpoint","phonetic_sh_donors","train_limit",
        "early_stop_patience","max_retention_failures","target_turkish_wer")
    config_identity={**context["parent_identity"],"public_turkish_data_policy":public_data_audit["identity"],"base_revision":BASE_REVISION,
        "extended_tokenizer_sha256":vocabulary["extended_tokenizer_sha256"],"lora_layout":encoder_lora_metadata(model),
        "tr_pool_sha256":ids_sha(tr),"replay_pool_sha256":ids_sha(all_replay),"source_sha256":source_sha,"input_sha256":input_sha,
        "native_loss":native_loss_identity(args),"domain_sampling_identity":sampler.identity,
        "PT_replay_transition":transition_identity(args,context,sampler_transition),
        "training_arguments":{key:arguments[key] for key in science_keys}}
    if args.resume_exact:
        del source
        own=torch.load(args.resume_exact,map_location="cpu",weights_only=True)
        validate_own_state(own,args,context,config_identity,model=model,torch_module=torch)
        validate_trainable_checkpoint(model,own,encoder_lora_metadata(model));model.load_state_dict(own["weights"],strict=False)
        sampler.load_state_dict(own["sampler"]);resume=own
    else:resume=source
    if args.resume_exact:optimizer.load_state_dict(resume["optimizer"])
    if len(optimizer.state)!=207:raise ValueError("All207 named Adam states required")
    stage_memory(memory_trace,"after_all207_Adam_restore")
    history,start_step=resume["history"],resume["step"]
    best,selected_step=resume["best_score"],resume["selected_step"]
    patience,retention_failures=resume["patience"],resume["retention_failures"]
    train_wall,evaluation_wall,audio_seconds=resume["train_wall_seconds"],resume["evaluation_wall_seconds"],resume["audio_seconds"]
    language_audio={**resume["language_audio_seconds"],"pt":resume["language_audio_seconds"].get("pt",0.)}
    initial_gate=resume["initial_gate"]
    next_eval=resume["next_eval_epoch"]
    random.setstate(resume["python_rng"]);torch.set_rng_state(resume["torch_rng"]);torch.cuda.set_rng_state_all(resume["cuda_rng"])
    config={**arguments,"steps":configured_steps,"source_step":source_step,"uniform_origin_step":uniform_origin,
        "remaining_uniform_pass_updates":context["remaining_updates_from_chosen_frontier"],"majority_source_step":quota_boundary,
        "phase":"HF_TSC_8199_uniform_Turkish_majority_PT_focused_v1","evaluation_languages":list(LANGUAGES),
        "uniform_full_pass":{"source_step":uniform_origin,"transition_source_step":quota_boundary,
            "additional_update_budget":sampler.remaining_update_budget,"end_step":configured_steps,
            "expected_new_unique_TR_clips":253798,"original_domain_epoch_is_lineage_only":True,
            "prior_uniform_draws":sampler.quota_transition["prior_uniform_draws"],"quota_transition":sampler.quota_transition},
        "method":"Full207 same14/2 uniform pass; onePT+oneoldfive replay with declared language-weighted preservation",
        "config_identity":config_identity,"initial_source_equivalence":equality,"public_turkish_data_audit":public_data_audit,
        "uniform_sampler_transition":sampler_transition,"retention_density_normalization":context["config"]["retention_density_normalization"],"PT_replay_policy_transition":ptfocus_declaration(context),"source_next_eval_epoch_preserved":next_eval,"Portuguese_inputs":{
            "preparation_proof":file_record(args.portuguese_preparation_proof),"baseline_proof":file_record(args.portuguese_baseline_proof),
            "replay_manifest":file_record(args.portuguese_replay_manifest),"DEV_manifest":file_record(args.portuguese_dev_manifest)},
        "turkish_clips_used":len(tr),"replay_clips_used":len(all_replay),"source_original_replay_clips":1595,"Portuguese_replay_clips":319,
        "sampling":sampler.coverage(),"sampling_identity":sampler.identity,"inherited_history_steps":source_step,
        "trainable_parameters":sum(p.numel() for p in trainable),"trainable_parameter_tensors":207,
        "native_loss":native_loss_identity(args),"lora_layout":layout,"encoder_adapter_parameters":sum(p.numel() for p in adapters),
        "encoder_base_frozen":True,"encoder_frozen":False,"encoder_activation_checkpointing":not args.no_encoder_checkpoint,
        "teacher":"Original8193 replay teacher; six languages with oldfive state+new Portuguese explicitly added",
        "initial_turkish_validation_wer":context["source_summary"]["validation_tr"]["asr"]["wer"],"source_champion_parked":parked_original_champion,"inherited_parent_all6_BEST":source_best_checkpoint,
        "portuguese_retention_baseline_proof":file_record(args.portuguese_baseline_proof),
        "original_base_v3_summary_unchanged":True,"new_uniform_epoch_clock":False,"selection_split":"fixedFLEURS DEV only",
        "Portuguese_recovery":{"grace_new_updates":1000,"old_five_guards_always_active":True,
            "all_six_required_for_every_eligible_BEST":True,"source_6500_all_six_qualified":False},
        "allocation_policy":{"TR_storage":"immutable indexed JSONL/mmap/SQLite", "decoded_TR_cache_cap_rows":256,
            "full_resident_TR_rowdict_pool":False,"training_pool_artifact":"compact index provenance plus1914 replay headers",
            "full_corpus_value_and_label_deletion":False,"memory_effect_requires_actual_stage_samples":True},
        "first_pass_stop_policy":{"plateau_patience_and_target_inactive_until_unique_TR":253798,
            "resource_and_nonfinite_stops_active":True,"old_five_retention_guard_always_active":True,
            "Portuguese_grace_new_updates":1000,"all_six_REQUIRED_for_every_BEST":True},
        "heldout_tests_evaluated":False,"training_throughput_timing":"complete update intervals; excludes startup/DEV/checkpoint serialization"}
    if args.resume_exact:
        original_config=json.loads((run_dir/"run_config.json").read_text(encoding="utf-8"))
        if canonical_identity_sha(original_config["config_identity"])!=canonical_identity_sha(config_identity):raise ValueError("Exact original identity changed")
        config=original_config
    else:
        write_json(run_dir/"run_config.json",config)
        stage_memory(memory_trace,"before_compact_index_pool_summary_write")
        write_pool_summary(run_dir/"training_pool.json",tr,all_replay,input_sha)
        stage_memory(memory_trace,"after_compact_index_pool_summary_write")
        write_json(run_dir/"vocabulary_audit.json",vocabulary)
    write_json(OUTPUT/"run_config.json",config)
    def save(path,step,reason=None):
        stage_memory(memory_trace,"before_atomic207_state_save",step=step,checkpoint_path=str(path))
        save_training_state(path,model,optimizer,sampler,run_id=args.run_id,step=step,config_identity=config_identity,
            history=history,best_score=best,selected_step=selected_step,patience=patience,retention_failures=retention_failures,
            next_eval_epoch=next_eval,train_wall_seconds=train_wall,audio_seconds=audio_seconds,language_audio_seconds=language_audio,
            evaluation_wall_seconds=evaluation_wall,initial_gate=initial_gate,stop_reason=reason)
        stage_memory(memory_trace,"after_atomic207_state_save",step=step,checkpoint_path=str(path))
    if not args.resume_exact:
        if selected_step is not None and selected_step==start_step:save(best_path,start_step,"majority_SOURCE_H_is_actual_all6_BEST")
        save(latest_path,start_step)
    write_json(run_dir/"training_history.json",history);write_json(OUTPUT/"training_history.json",history)
    del resume
    if args.resume_exact:del own
    else:del source
    from tsc_uniform_majority_ptfocus_transition_v1 import release_admission_caches
    release_admission_caches()
    del context,pt_rows,pt_cached
    import gc
    gc.collect()
    write_json(run_dir/("uniform_full_startup_from_"+str(start_step)+".json"),{"passed":True,"run_id":args.run_id,
        "source_step":source_step,"resume_step":start_step,"indexed_TR_carrier":file_record(tr.directory/"index_provenance.json"),
        "TR_resident_decoded_dictionary_pool":False,"TR_cache_cap_rows":256,"reclaimed_memory_claimed":False})
    stage_memory(memory_trace,"after_CPU_source_release_and_bounded_TR_startup")
    model.train();model.encoder.eval()
    new_parameters = (model.decoder.embedding.weight, model.joint.head.weight, model.joint.head.bias)
    loop_start = time.perf_counter()
    stop_reason = "configured_steps_complete"
    last_evaluated_step = start_step
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"Batch {args.batch_size} x accumulation{args.accumulation}; TR {len(tr)}; replay {len(replay)}; configured {configured_steps} steps", flush=True)

    def evaluate_current(step, final=False):
        nonlocal best, selected_step, patience, retention_failures, evaluation_wall, last_evaluated_step, next_eval
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        before = time.perf_counter()
        stage_memory(memory_trace,"before_fixed350_DEV",step=step)
        dev = evaluate(model, processor, f"{args.run_id}_step_{step}", splits=("validation",))
        stage_memory(memory_trace,"after_fixed350_DEV",step=step)
        gate = development_gate(dev, baseline)
        evaluation_wall += time.perf_counter()-before
        record = history[-1]
        record["validation"] = gate
        record["development_selection_only"] = True
        record["final_development_evaluation"] = final
        last_evaluated_step = step
        improved = (gate["retention_gate_passed"] and gate["turkish_wer"] < best
                    and gate["turkish_wer"] < baseline["validation_tr"]["asr"]["wer"])
        meaningful = improved and best-gate["turkish_wer"] >= args.min_improvement
        if improved:
            best, selected_step = gate["turkish_wer"], step
        local_updates = step-uniform_origin
        old_five_pass = all(gate["retention_wer_deltas"][lang] <= gate["retention_wer_limits"][lang]
                            for lang in ("en", "es", "it", "fr", "de"))
        stop_gate_pass = old_five_pass and (local_updates < 1000 or gate["retention_gate_passed"])
        retention_failures = 0 if stop_gate_pass else retention_failures+1
        gate["old_five_retention_gate_passed"] = old_five_pass
        gate["Portuguese_grace_active_for_stop_only"] = local_updates < 1000
        gate["stop_retention_gate_passed"] = stop_gate_pass
        # Plateau counts full-epoch evaluations. Initial quarter/half screens
        # can identify gate failures without prematurely consuming3 epochs.
        epoch = sampler.turkish_epoch
        previous_epochs=[row["turkish_epoch"] for row in history[:-1] if "validation" in row]
        previous_epoch=previous_epochs[-1] if previous_epochs else 0.
        full_epoch=math.floor(epoch+1e-10)>math.floor(previous_epoch+1e-10)
        if meaningful:
            patience = 0
        elif full_epoch:
            patience += 1
        next_eval = next_evaluation_epoch(epoch, args.eval_epoch_fraction, args.eval_per_epoch)
        record["selection_diagnostics"] = {"improved_qualifying_best":improved,"meaningful_improvement":meaningful,
                                            "plateau_epoch_evaluations":patience,"consecutive_retention_failures":retention_failures}
        if improved:
            save(best_path, step)
        save(latest_path, step)
        write_json(run_dir / "training_history.json", history)
        write_json(OUTPUT / "training_history.json", history)
        print(f"dev step{step} epoch{epoch:.4f} TR WER{gate['turkish_wer']:.5f} retention {gate['retention_gate_passed']} best{selected_step}", flush=True)
        model.train()
        model.encoder.eval()
        torch.cuda.empty_cache()
        return gate

    for step in range(start_step+1, configured_steps+1):
        if args.stop_file and args.stop_file.exists():
            stop_reason = "controller_stop_file"
            break
        if args.max_wall_seconds is not None and time.perf_counter()-process_start >= args.max_wall_seconds:
            stop_reason = "soft_max_wall_seconds"
            break
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        factor = lr_factor(step, configured_steps, args.warmup_steps, args.schedule, args.min_lr_ratio)
        optimizer.param_groups[0]["lr"] = args.head_lr*factor
        optimizer.param_groups[1]["lr"] = args.lora_lr*factor
        totals = {key:0. for key in ("loss","tdt_loss","replay_kl","replay_duration_kl","replay_feature_distillation")}
        sampled_sources, step_audio, padding = [], 0., []
        for group in sampler.next_batches():
            detailed_trace=(step-start_step<=2 or step%250==0)
            if detailed_trace:stage_memory(memory_trace,"before_native_microbatch",step=step)
            batch = collate_training_batch(processor, group, blank_id=model.config.blank_token_id,
                                            old_size=old_size, device="cuda")
            result = pt_focused_batch_objective(model, teacher, batch, old_size=old_size, replay_kl=args.replay_kl,
                                      duration_kl=args.duration_kl, feature_distillation=args.feature_distillation,
                                      chunk_size=args.chunk_size, tdt_loss_fn=loss_function,
                                      tdt_host_lengths=args.tdt_implementation in ("v2", "nemo"))
            scaled = result.loss/args.accumulation
            if not torch.isfinite(scaled):
                raise RuntimeError(f"Nonfinite true-batch objective at step{step}")
            scaled.backward()
            if detailed_trace:stage_memory(memory_trace,"after_native_forward_backward",step=step)
            metrics = result.detached_metrics()
            for key in totals:
                totals[key] += metrics[key]/args.accumulation
            step_audio += batch.audio_seen_seconds
            padding.append(batch.lattice_padding_factor)
            for row in group:
                language_audio[row["language"]] += float(row["duration"])
                sampled_sources.append(row["source_id"])
            del result, scaled, batch
            if args.tdt_implementation == "nemo":
                reclaim_inactive_cuda_cache(torch.device("cuda"), "after_whole_microbatch_backward")
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
        previous = [parameter[old_size:model.config.vocab_size].detach().clone() for parameter in new_parameters]
        optimizer.step()
        with torch.no_grad():
            for parameter, old in zip(new_parameters, previous):
                updated = parameter[old_size:model.config.vocab_size]
                updated.copy_(old + args.new_token_lr_multiplier*(updated-old))
        if args.tdt_implementation == "nemo":
            reclaim_inactive_cuda_cache(torch.device("cuda"), "after_adam_and_new_rows")
        torch.cuda.synchronize()
        if step-start_step<=2 or step%250==0:stage_memory(memory_trace,"after_completed_207_Adam_update",step=step)
        train_wall += time.perf_counter()-step_start
        audio_seconds += step_audio
        record = {"step":step, **totals,"gradient_norm":float(norm),"elapsed_seconds":time.perf_counter()-loop_start,
                  "process_wall_seconds":time.perf_counter()-process_start,"train_wall_seconds":train_wall,
                  "evaluation_wall_seconds":evaluation_wall,"audio_seen_seconds":audio_seconds,
                  "audio_minutes_per_wall_minute":audio_seconds/train_wall,
                  "turkish_audio_seen_seconds":language_audio["tr"],"replay_audio_seen_seconds":audio_seconds-language_audio["tr"],
                  "language_audio_seen_seconds":language_audio.copy(),"step_audio_seen_seconds":step_audio,
                  "sampled_source_ids":sampled_sources,"turkish_epoch":sampler.turkish_epoch,
                  "turkish_clips_seen":sampler.tr_draws,"turkish_unique_clips_seen":sampler.tr_unique,
                  "Turkish_domain_coverage":sampler.coverage(),"uniform_full_updates":step-uniform_origin,
                  "majority_updates":step-quota_boundary,"majority_source_step":quota_boundary,
                  "PT_focus_updates":step-source_step,"PT_focus_source_step":source_step,
                  "lora_lr_fork_updates":step-context_lora_origin,"lora_lr_fork_source_step":context_lora_origin,"uniform_origin_step":uniform_origin,
                  "replay_clips_seen":sum(stream.draws for stream in sampler.replay.values()),
                  "replay_language_clip_counts":sampler.replay_language_counts.copy(),
                  "head_lr":optimizer.param_groups[0]["lr"],"lora_lr":optimizer.param_groups[1]["lr"],
                  "new_token_update_multiplier":args.new_token_lr_multiplier,
                  "replay_kl_coefficient":args.replay_kl,"duration_kl_coefficient":args.duration_kl,"feature_distillation_coefficient":args.feature_distillation,
                  "preservation_coefficients_by_language":{"pt":[args.replay_kl,args.duration_kl,args.feature_distillation],"old_five":[args.replay_kl*5/3,args.duration_kl*5/3,args.feature_distillation*5/3]},
                  "PT_replay_clips_per_update":1,"old_five_replay_clips_per_update":1,
                  "mean_microbatch_lattice_padding_factor":math.fsum(padding)/len(padding),
                  "peak_allocated_vram_gb":torch.cuda.max_memory_allocated()/1e9,
                  "current_allocated_vram_gb":torch.cuda.memory_allocated()/1e9,
                  "current_reserved_vram_gb":torch.cuda.memory_reserved()/1e9}
        history.append(record)
        if step % args.log_every == 0:
            print(json.dumps(record), flush=True)
            write_json(run_dir / "training_history.json", history)
            write_json(OUTPUT / "training_history.json", history)
        if (args.eval_every_updates is not None and step % args.eval_every_updates == 0) or sampler.turkish_epoch + 1e-10 >= next_eval or step == configured_steps:
            gate = evaluate_current(step)
            if args.max_retention_failures and retention_failures >= args.max_retention_failures:
                stop_reason = "consecutive_retention_gate_failures"
                break
            full_pass=sampler.coverage()["uniform_pass"]["full_pass_complete"]
            if full_pass and args.early_stop_patience and patience >= args.early_stop_patience:
                stop_reason = "development_plateau"
                break
            if full_pass and args.target_turkish_wer is not None and gate["retention_gate_passed"] and gate["turkish_wer"] <= args.target_turkish_wer:
                stop_reason = "development_target_met"
                break
    completed = history[-1]["step"] if history else start_step
    if history and completed != last_evaluated_step:
        evaluate_current(completed, final=True)
    save(latest_path, completed, stop_reason)
    write_json(run_dir / "training_history.json", history)
    write_json(OUTPUT / "training_history.json", history)
    selection = {"accepted":selected_step is not None,"run_id":args.run_id,"selected_step":selected_step,
                 "completed_steps":completed,"configured_steps":configured_steps,"stop_early_reason":stop_reason,
                 "consecutive_retention_failures":retention_failures,"plateau_epoch_evaluations":patience,
                 "turkish_validation_wer":best if selected_step is not None else None,
                 "initial_turkish_validation_wer":config["initial_turkish_validation_wer"],
                 "lineage_original_uniform_initial_TR_WER":initial_gate["turkish_wer"],
                 "improved_over_initial_model":selected_step is not None and selected_step>source_step and best < config["initial_turkish_validation_wer"],
                 "selection_split":"FLEURS validation only","heldout_tests_evaluated":False,
                 "weights":str((best_path if best_path.exists() else Path(source_best_checkpoint["path"])) if selected_step is not None else latest_path),
                 "selected_checkpoint_role":"own_majority_all6_BEST" if selected_step is not None and selected_step>source_step else "inherited_parent_all6_BEST" if selected_step is not None else "unqualified_current_latest",
                 "latest_weights":str(latest_path),"turkish_epochs_completed":sampler.turkish_epoch,
                 "turkish_epoch_semantics":"new_uniform_pass_draw_equivalent", "Turkish_domain_coverage":sampler.coverage(),
                 "source_step":source_step,"uniform_origin_step":uniform_origin,"uniform_full_updates":completed-uniform_origin,
                 "majority_updates":completed-quota_boundary,"majority_source_step":quota_boundary,
                 "lora_lr_fork_updates":completed-context_lora_origin,"PT_focus_updates":completed-source_step,"full_new_uniform_pass_complete":sampler.coverage()["uniform_pass"]["full_pass_complete"],
                 "sampling_identity":sampler.identity,
                 "train_wall_seconds":train_wall,"audio_seen_seconds":audio_seconds,
                 "audio_minutes_per_wall_minute":audio_seconds/train_wall if train_wall else None,
                 "reason":"Best qualifying development model retained; useful Turkish/heldout quality requires separate verification." if selected_step is not None
                          else "No evaluated model improved original Turkish dev and passed every old-language gate."}
    write_json(run_dir / "selection.json", selection)
    write_json(OUTPUT / "selection.json", selection)
    print(json.dumps(selection), flush=True)


def native_loss_identity(args):
    """Require immutable coherent startup evidence; scientific objective stays control."""
    if args.tdt_implementation != "nemo":
        return None
    import sys
    from native_tdt_checkpointed_v4 import configuration_identity
    if args.native_backend_provenance is None or not args.native_backend_provenance.is_file():
        raise ValueError("Official loss requires --native-backend-provenance from pre-Torch native launcher")
    evidence = json.loads(args.native_backend_provenance.read_text(encoding="utf-8"))
    if (evidence.get("kind") != "native_tdt_training_backend_v3" or evidence.get("status") != "ready"
            or evidence.get("run_id") != args.run_id or evidence.get("tdt_implementation") != "nemo"
            or evidence.get("trainer_sha256") != file_sha(__file__)
            or evidence.get("adapter_sha256") != file_sha(OUTPUT / "native_tdt_checkpointed_v4.py")):
        raise ValueError("Native startup provenance run/trainer/adapter identity differs")
    if (Path(evidence.get("launcher_path", "")).resolve() != (OUTPUT / "run_TSC_uniform_majority_ptfocus_v1.py").resolve()
            or evidence.get("launcher_sha256") != file_sha(OUTPUT / "run_TSC_uniform_majority_ptfocus_v1.py")):
        raise ValueError("Native provenance must name the actual declared-fork dispatcher")
    preload = evidence.get("preload", {})
    if (preload.get("kind") != "coherent_native_cuda12_9_preload_v1"
            or preload.get("module_sha256") != file_sha(OUTPUT / "preload_nemo_linker_v1.py")
            or preload.get("successful_launcher_sha256") != file_sha(OUTPUT / "launch_nemo_linker_v4.py")
            or preload.get("actual_binding_linker", {}).get("version") != [12, 9]
            or preload.get("cuda_initialized") is not False):
        raise ValueError("Native compiler/linker startup evidence differs")
    fresh=json.loads(args.uniform_full_runtime_proof.read_text(encoding="utf-8"))
    import os
    if fresh.get("pid")!=os.getpid() or fresh.get("ppid")!=os.getppid() or fresh.get("status")!="ready" or fresh.get("run_id")!=args.run_id:
        raise ValueError("Fresh same-PID fullrun pre-Torch proof required")
    if fresh.get("trainer_sha256")!=file_sha(__file__) or fresh.get("launcher_sha256")!=file_sha(OUTPUT/"run_TSC_uniform_majority_ptfocus_v1.py"):
        raise ValueError("Fresh uniform executor source pin differs")
    expected_kind="TSC_uniform_majority_ptfocus_fresh_preload_v1" if args.resume_exact else "native_tdt_training_backend_v3"
    if (fresh.get("kind")!=expected_kind or fresh.get("allocator_cap_applied") is not True
            or fresh.get("adapter_sha256")!=file_sha(OUTPUT/"native_tdt_checkpointed_v4.py")
            or fresh.get("declared_allocator_policy")!={"cap_gb":8,"end_cache_gb":3,"native_inactive_threshold_bytes":750000000}
            or fresh.get("preload",{}).get("cuda_initialized") is not False
            or fresh.get("preload",{}).get("actual_binding_linker",{}).get("version")!=[12,9]
            or fresh.get("allocation_policy",{}).get("lazy_helper_sha256")!=file_sha(OUTPUT/"lazy_training_rows_v1.py")
            or fresh.get("allocation_policy",{}).get("decoded_TR_cache_cap_rows")!=256
            or fresh.get("allocation_policy",{}).get("resident_full_TR_rowdict_pool") is not False):
        raise ValueError("Fresh allocator/native4/lazy-index admission differs")
    for name in (Path(__file__).name,"run_TSC_uniform_majority_ptfocus_v1.py","tsc_uniform_majority_ptfocus_transition_v1.py",
                 "tsc_uniform_majority_ptfocus_sampler_v1.py","lazy_training_rows_v1.py","native_tdt_checkpointed_v4.py"):
        if fresh["allocation_policy"]["source_sha256"].get(name)!=file_sha(OUTPUT/name):
            raise ValueError("Fresh scientific/streamed-loader source pin differs: "+name)
    if args.resume_exact and fresh.get("original_first_start_ready")!={"path":str(args.native_backend_provenance.resolve()),"sha256":file_sha(args.native_backend_provenance)}:
        raise ValueError("Fresh exact window must bind immutable original native proof")
    expected_env = ROOT / "work/nemo-native-env-v2/Scripts/python.exe"
    if Path(sys.executable).resolve() != expected_env.resolve():
        raise ValueError("Use isolated native-env-v2 with coherent CUDA12.9 startup")
    if type(args.native_max_joint_bytes) is not int or not 0 < args.native_max_joint_bytes <= 1_700_000_000:
        raise ValueError("Native exact per-clip logits bound must be positive and at most1.7GB")
    return {**configuration_identity(), "max_joint_bytes":args.native_max_joint_bytes,
            "trainer_inactive_cuda_cache_cleanup":{"after_completed_microbatch_backward":True,"after_adam_and_new_rows":True,"inactive_threshold_bytes":750000000,"scope":"Only unused CUDA allocator blocks; no tensor/optimizer/RNG changes"},
            "caller_reduction":"raw per-clip NLL / each target length, then physical-batch mean",
            "sigma":0.0,"omega":0.0,"native_backend_provenance_sha256":file_sha(args.native_backend_provenance),
            "mathematical_equivalence_claim":False,
            "scope":"Official NeMo loss kernel; existing HF encoder/decoder/projections and custom LoRA/KD/training loop"}


def native_loss_function(args, tr, replay):
    """Inject official full-lattice loss without changing accepted audio or labels."""
    from functools import partial
    from native_tdt_checkpointed_v4 import chunked_tdt_loss
    native_loss_identity(args)
    width = 8199 + 5
    worst=tr.receipt["max_full_joint"]
    if worst["required_joint_bytes"]>args.native_max_joint_bytes:
        raise ValueError("Complete indexed authentic TR native lattice bound exceeded")
    for row in replay:
        size = int(row["encoder_frames"]) * (int(row["target_tokens"]) + 1) * width * 4
        if size > args.native_max_joint_bytes:
            raise ValueError(f"Exact native joint byte bound exceeded: {row['source_id']} needs{size}; no cropping/dropping")
    return partial(chunked_tdt_loss, sigma=0.0, omega=0.0, recompute=True,
                   max_joint_bytes=args.native_max_joint_bytes)


def canonical_identity_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def runtime_migration_evidence(args):
    if args.resume_runtime_migration is not None:
        raise ValueError("This declared head-LR fork admits no data or runtime migration")
    return None


def exact_resume_identity_matches(saved, args, current):
    return args.resume_runtime_migration is None and saved.get("config_identity") == current


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tsc-input-proof",type=Path,required=True); p.add_argument("--tsc-training-approval",type=Path,required=True); p.add_argument("--source-closure",type=Path,required=True)
    p.add_argument("--tr-manifest",type=Path,default=DATA/"tr_public_v3_train.jsonl"); p.add_argument("--data-v3-proof",type=Path,default=OUTPUT/"tr_public_v3_preparation.json")
    p.add_argument("--replay-manifest",type=Path,default=DATA/"replay_train.jsonl")
    p.add_argument("--supplement",type=Path)
    p.add_argument("--train-limit",type=int)
    resume = p.add_mutually_exclusive_group()
    resume.add_argument("--resume-weights",type=Path)
    resume.add_argument("--resume-exact",type=Path)
    resume.add_argument("--resume-optimization",type=Path)
    p.add_argument("--optimization-source-receipt",type=Path)
    p.add_argument("--epochs",type=float,default=1.)
    p.add_argument("--steps",type=int)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--accumulation",type=int,default=2)
    p.add_argument("--tdt-implementation",choices=("original","v2","nemo"),default="nemo"); p.add_argument("--native-backend-provenance",type=Path); p.add_argument("--native-max-joint-bytes",type=int,default=1_700_000_000); p.add_argument("--resume-runtime-migration",type=Path)
    p.add_argument("--eval-per-epoch",type=float,default=1.); p.add_argument("--eval-every-updates",type=int)
    p.add_argument("--eval-epoch-fraction",type=float,default=.25)
    p.add_argument("--head-lr",type=float,default=1e-5)
    p.add_argument("--lora-lr",type=float,default=1e-4)
    p.add_argument("--new-token-lr-multiplier",type=float,default=6.)
    p.add_argument("--replay-fraction",type=float,default=.125)
    p.add_argument("--replay-kl",type=float,default=40.)
    p.add_argument("--duration-kl",type=float,default=2.)
    p.add_argument("--feature-distillation",type=float,default=4.)
    p.add_argument("--warmup-steps",type=int,default=25)
    p.add_argument("--schedule",choices=("constant","cosine"),default="constant")
    p.add_argument("--min-lr-ratio",type=float,default=.1)
    p.add_argument("--early-stop-patience",type=int,default=3)
    p.add_argument("--max-retention-failures",type=int,default=2)
    p.add_argument("--min-improvement",type=float,default=.002)
    p.add_argument("--target-turkish-wer",type=float)
    p.add_argument("--max-wall-seconds",type=float)
    p.add_argument("--stop-file",type=Path)
    p.add_argument("--rank",type=int,default=8)
    p.add_argument("--alpha",type=float,default=16.)
    p.add_argument("--last-n-layers",type=int)
    p.add_argument("--chunk-size",type=int,default=16)
    p.add_argument("--no-encoder-checkpoint",action="store_true")
    p.add_argument("--bucket-size",type=int,default=128)
    p.add_argument("--max-tokens",type=int,default=180)
    p.add_argument("--phonetic-sh-donors",action="store_true")
    p.add_argument("--seed",type=int,default=20261010)
    p.add_argument("--log-every",type=int,default=10)
    p.add_argument("--tr-frame-metadata",type=Path,default=DATA/"tr_public_v3_frame_metadata_v1.jsonl"); p.add_argument("--tr-frame-metadata-proof",type=Path,default=OUTPUT/"TR_frame_metadata_cpu_verification_v1.json"); p.add_argument("--reuse-input-audit",type=Path); p.add_argument("--run-id",required=True)
    p.add_argument("--sampling-mode",choices=("uniform",),default="uniform")
    p.add_argument("--fleurs-fraction",type=float,default=.3)
    p.add_argument("--replay-seed",type=int,required=True)
    p.add_argument("--resume-uniform-full",type=Path,required=True)
    p.add_argument("--resume-majority",type=Path,required=True)
    p.add_argument("--majority-source-receipt",type=Path,required=True)
    p.add_argument("--uniform-full-source-receipt",type=Path,required=True)
    p.add_argument("--uniform-full-runtime-proof",type=Path,required=True)
    p.add_argument("--indexed-tr-directory",type=Path,required=True)
    p.add_argument("--portuguese-replay-manifest",type=Path,default=DATA/"portuguese_retention_v1/replay_train_pt.jsonl")
    p.add_argument("--portuguese-dev-manifest",type=Path,default=DATA/"portuguese_retention_v1/validation_pt.jsonl")
    p.add_argument("--portuguese-baseline-proof",type=Path,default=OUTPUT/"Portuguese_retention_baselines_v1.json")
    p.add_argument("--portuguese-preparation-proof",type=Path,default=OUTPUT/"Portuguese_FLEURS_retention_preparation_v1.json")
    p.add_argument("--portuguese-seed",type=int,default=20261012)
    return p


def validate_arguments(args):
    if not (args.epochs > 0 and args.batch_size > 0 and args.accumulation > 0
            and args.eval_per_epoch > 0 and 0 < args.eval_epoch_fraction <= 1 and args.log_every > 0
            and args.warmup_steps >= 0 and 0 <= args.min_lr_ratio <= 1
            and args.early_stop_patience >= 0 and args.max_retention_failures >= 0 and args.min_improvement >= 0
            and args.head_lr > 0 and args.lora_lr > 0 and args.new_token_lr_multiplier > 0
            and args.replay_kl >= 0 and args.duration_kl >= 0 and args.feature_distillation >= 0
            and args.rank > 0 and args.alpha > 0 and args.chunk_size > 0 and args.bucket_size > 0 and args.max_tokens > 0
            and 0 <= args.replay_fraction < 1):
        raise ValueError("Invalid training duration/layout/schedule/stop settings")
    if any(not math.isfinite(value) for value in vars(args).values() if isinstance(value, float)):
        raise ValueError("Training options must be finite")
    if args.eval_every_updates is not None and args.eval_every_updates <= 0:
        raise ValueError("Development update interval must be positive")
    if args.steps is not None and args.steps <= 0:
        raise ValueError("Steps cap must be positive")
    if args.max_wall_seconds is not None and args.max_wall_seconds <= 0:
        raise ValueError("Wall-time soft cap must be positive")
    if args.train_limit is not None and args.train_limit <= 0:
        raise ValueError("Training subset limit must be positive")
    if args.last_n_layers is not None and not 1 <= args.last_n_layers <= 24:
        raise ValueError("Adapter layer subset must lie in [1,24]")
    if args.target_turkish_wer is not None and args.target_turkish_wer < 0:
        raise ValueError("Target WER must be nonnegative")
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise ValueError("Run ID must be a single safe filename component")
    if args.resume_runtime_migration is not None and (args.tdt_implementation != "nemo" or args.resume_exact is None):
        raise ValueError("Runtime migration is scoped to native exact-state continuation")
    if args.tdt_implementation == "nemo":
        native_loss_identity(args)
    if args.resume_weights or args.resume_optimization or args.resume_runtime_migration:raise ValueError("Full207 source or own exact only")
    if args.batch_size!=8 or args.accumulation!=2 or args.head_lr!=1e-5 or args.lora_lr!=1e-4 or args.replay_fraction!=.125 or args.replay_kl!=40 or args.duration_kl!=2 or args.feature_distillation!=4 or args.schedule!="constant":
        raise ValueError("Declared full207 B8x2/14TR2RP/KD40-duration2-feature4 constant recipe required")
    if args.eval_every_updates!=250 or args.sampling_mode!="uniform":raise ValueError("Fulluniform/DEV250 required")
    majority_preflight(args)
    return args


if __name__ == "__main__":
    args = validate_arguments(parser().parse_args())
    run(args)
