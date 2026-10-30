"""Isolated approved TSC TRAIN data-only LoRA arm; DEV selection only.

Warm starts load exactly the incoming trainable layout with a NEW optimizer.
Exact resume restores optimizer/sampler/RNG of the SAME run and data/config.
Uses vetted CV validated/train-role and FLEURS TRAIN; fixed TEST stays untouched.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch

from batched_lora_tr_metadata_v1 import collate_training_batch, batch_objective
from tsc_training_inputs_v1 import annotate_tr_frame_metadata, confirm_tr_metadata_acceptance, confirm_replay_after_cache, audit_tsc_training_inputs, source_isolation_tsc, closed_source_identity
from benchmark import LANGUAGES, evaluate, read_jsonl, write_json
from efficient_tdt import chunked_tdt_loss as original_tdt
from encoder_lora import install_encoder_lora, encoder_lora_metadata
from extend_vocab import extend_processor_and_model
from metrics import normalize_text
from runtime import ROOT, DATA, OUTPUT, load_model
from train import make_cache, checkpoint_state
from train_lora import checkpoint_encoder_blocks

BASE_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_sha(rows):
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


def validate_trainable_checkpoint(model, saved, layout):
    expected = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(saved["weights"]) != set(expected):
        raise ValueError("Warm start must contain exactly all current trainable keys")
    if saved.get("lora_metadata") != layout:
        raise ValueError("Warm-start adapter rank/alpha/layers/layout differs; use a fresh run for a different layout")
    for name, weight in saved["weights"].items():
        if weight.shape != expected[name].shape or weight.dtype != torch.float32 or not torch.isfinite(weight).all():
            raise ValueError("Warm-start shape/FP32/finiteness differs for " + name)


def source_isolation(tr, replay, heldout, audit):
    return source_isolation_tsc(tr, replay, heldout, audit)














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


def run(args):
    process_start = time.perf_counter(); public_data_audit = audit_tsc_training_inputs(args); tsc_source = closed_source_identity(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    run_dir = OUTPUT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = ROOT / "work" / (args.run_id + "_best.pt")
    latest_path = ROOT / "work" / (args.run_id + "_latest.pt")
    if not args.resume_exact and (run_dir / "run_config.json").exists():
        raise ValueError("Run ID already exists; choose a fresh ID or exact resume")
    model, processor = load_model()
    baseline_path = OUTPUT / "results/base_v3"
    baseline = json.loads((baseline_path / "summary.json").read_text(encoding="utf-8"))
    heldout_paths = [DATA / f"{split}_{lang}.jsonl" for lang in LANGUAGES for split in ("validation", "test")]
    heldout = [row for path in heldout_paths for row in read_jsonl(path)]
    for lang in LANGUAGES:
        expected = [row["source_id"] for row in read_jsonl(DATA / f"validation_{lang}.jsonl")]
        actual = [row["source_id"] for row in read_jsonl(baseline_path / f"validation_{lang}.jsonl")]
        if expected != actual or baseline[f"validation_{lang}"]["asr"]["utterances"] != len(expected):
            raise ValueError("Development selection requires complete ordered fixed original baseline")
    tr = read_jsonl(args.tr_manifest)
    if args.train_limit is not None:
        random.Random(args.seed).shuffle(tr)
        tr = tr[:args.train_limit]
    if args.supplement:
        tr += read_jsonl(args.supplement)
    replay = read_jsonl(args.replay_manifest)
    source_isolation(tr, replay, heldout, public_data_audit)
    teacher = {name: copy.deepcopy(getattr(model, name)).eval().requires_grad_(False)
               for name in ("encoder_projector", "decoder", "joint")}
    old_size = model.config.vocab_size
    vocabulary = extend_processor_and_model(processor, model, [row["text"] for row in tr])
    if args.phonetic_sh_donors:
        for new, donor in (("ş", "ș"), ("Ş", "Ș")):
            new_id, donor_id = processor.tokenizer.convert_tokens_to_ids(new), processor.tokenizer.convert_tokens_to_ids(donor)
            if new_id < old_size or not 0 <= donor_id < old_size:
                raise ValueError("Phonetic donor ID lookup differs")
            with torch.no_grad():
                model.decoder.embedding.weight[new_id].copy_(model.decoder.embedding.weight[donor_id])
                model.joint.head.weight[new_id].copy_(model.joint.head.weight[donor_id])
                model.joint.head.bias[new_id].copy_(model.joint.head.bias[donor_id] - 2)
            vocabulary.setdefault("phonetic_donor_override", []).append({"new":new,"new_id":new_id,"donor":donor,"donor_id":donor_id})
    counts_before = {"tr": len(tr), "replay": len(replay)}
    def representable(row, teacher_only=False):
        ids = processor.tokenizer(row["text"])["input_ids"]
        return (ids and len(ids) <= args.max_tokens and (not teacher_only or max(ids) < old_size)
                and normalize_text(processor.tokenizer.decode(ids, skip_special_tokens=True, group_tokens=False), row["language"])
                == normalize_text(row["text"], row["language"]))
    tr = [row for row in tr if representable(row)]
    replay = [row for row in replay if representable(row, teacher_only=True)]
    namespace = hashlib.sha256((BASE_REVISION + processor.tokenizer._tokenizer.to_str()).encode()).hexdigest()[:20]
    cache = ROOT / "work/encoder-cache" / namespace
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    tr = annotate_tr_frame_metadata(processor, tr, args.tr_frame_metadata, args.tr_frame_metadata_proof, args.max_tokens)
    replay = annotate_cache_lengths(make_cache(model, processor, replay, cache, args.max_tokens)); confirm_tr_metadata_acceptance(tr, public_data_audit); confirm_replay_after_cache(replay, public_data_audit)
    layout = install_encoder_lora(model, rank=args.rank, alpha=args.alpha, last_n_layers=args.last_n_layers)
    model.config.encoder_lora_tracking = {"layout": layout, "implementation_sha256": file_sha(OUTPUT / "encoder_lora.py"), "kind": "unmerged"}
    if not args.no_encoder_checkpoint:
        checkpoint_encoder_blocks(model)
    resume = None
    resume_path = args.resume_exact or args.resume_weights
    if resume_path:
        resume = torch.load(resume_path, map_location="cpu", weights_only=True)
        validate_trainable_checkpoint(model, resume, encoder_lora_metadata(model))
        model.load_state_dict(resume["weights"], strict=False)
    adapters = [parameter for name, parameter in model.named_parameters() if parameter.requires_grad and name.endswith(("lora_A", "lora_B"))]
    heads = [parameter for name, parameter in model.named_parameters() if parameter.requires_grad and not name.endswith(("lora_A", "lora_B"))]
    trainable = adapters + heads
    optimizer = torch.optim.AdamW([{"params":heads,"lr":args.head_lr,"weight_decay":.01},
                                  {"params":adapters,"lr":args.lora_lr,"weight_decay":0.}])
    sampler = BalancedBucketSampler(tr, replay, seed=args.seed, batch_size=args.batch_size,
                                    accumulation=args.accumulation, replay_fraction=args.replay_fraction,
                                    bucket_size=args.bucket_size)
    epoch_steps = math.ceil(args.epochs * len(tr) / sampler.tr_per_update)
    configured_steps = min(epoch_steps, args.steps) if args.steps is not None else epoch_steps
    if configured_steps < 1:
        raise ValueError("Training needs positive configured steps")
    if args.tdt_implementation == "nemo":
        loss_function = native_loss_function(args, tr, replay)
        from native_tdt_checkpointed_v3 import reclaim_inactive_cuda_cache
    else:
        loss_function = original_tdt if args.tdt_implementation == "original" else __import__("efficient_tdt_v2", fromlist=["chunked_tdt_loss"]).chunked_tdt_loss
    source_names = ("train_lora_batched_tsc_v1.py", "tsc_training_inputs_v1.py", "tsc_supplemental_data_policy_v1.py", "prepare_TSC_TRAIN_v1.py", "native_tdt_checkpointed_v3.py", "preload_nemo_linker_v1.py", "launch_nemo_linker_v4.py", "launch_native_training_tsc_v1.py", "public_turkish_data_policy_v3.py", "train_lora_batched.py", "batched_lora_tr_metadata_v1.py", "tr_frame_metadata_v1.py", "training_input_audit_reuse_v1.py", "batched_lora.py", "train_lora.py", "train.py", "encoder_lora.py",
                    "efficient_tdt.py", "efficient_tdt_v2.py", "extend_vocab.py", "runtime.py", "benchmark.py", "metrics.py", "confidence.py")
    source_sha = {name: file_sha(OUTPUT / name) for name in source_names if (OUTPUT / name).exists()}
    input_paths = [args.tsc_input_proof, args.tsc_training_approval, args.source_closure, args.tr_manifest, args.replay_manifest, args.data_v3_proof, args.tr_frame_metadata, args.tr_frame_metadata_proof, *([args.reuse_input_audit] if args.reuse_input_audit else []), *([args.native_backend_provenance] if args.native_backend_provenance else []), *([args.resume_runtime_migration] if args.resume_runtime_migration else []), OUTPUT / "tr_public_v3_tokenizer_acceptance.json", *heldout_paths]
    if args.supplement:
        input_paths.append(args.supplement)
    input_sha = {str(path.resolve()): file_sha(path) for path in input_paths}
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config_identity = {"public_turkish_data_policy":public_data_audit["identity"], "base_revision":BASE_REVISION, "extended_tokenizer_sha256":vocabulary["extended_tokenizer_sha256"],
                       "lora_layout":encoder_lora_metadata(model), "tr_pool_sha256":ids_sha(tr), "replay_pool_sha256":ids_sha(replay),
                       "source_sha256":source_sha, "input_sha256":input_sha, "native_loss":native_loss_identity(args),
                       "training_arguments":{key:arguments[key] for key in ("epochs","steps","batch_size","accumulation","replay_fraction",
                             "bucket_size","seed","max_tokens","head_lr","lora_lr","new_token_lr_multiplier","replay_kl","duration_kl",
                             "feature_distillation","rank","alpha","last_n_layers","chunk_size","tdt_implementation","warmup_steps",
                             "schedule","min_lr_ratio","eval_per_epoch","eval_epoch_fraction","eval_every_updates","min_improvement",
                             "no_encoder_checkpoint","phonetic_sh_donors","train_limit","early_stop_patience",
                             "max_retention_failures","target_turkish_wer")}}
    # Reject exact-resume mismatches before replacing any existing run artifact.
    if args.resume_exact:
        if resume["run_id"] != args.run_id or not exact_resume_identity_matches(resume, args, config_identity):
            raise ValueError("Exact resume requires identical run ID, sources, data, layout and training arguments")
        if (len(resume["history"]) != resume["step"]
                or any(row["step"] != index for index, row in enumerate(resume["history"], 1))
                or not 0 <= resume["step"] <= configured_steps):
            raise ValueError("Exact-resume history/checkpoint step mismatch")
        if resume["selected_step"] is not None:
            if not best_path.exists():
                raise ValueError("Exact resume requires the existing best checkpoint; refusing to lose its selection")
            saved_best = torch.load(best_path, map_location="cpu", weights_only=True)
            if saved_best["run_id"] != args.run_id or saved_best["step"] != resume["selected_step"]:
                raise ValueError("Exact-resume best checkpoint run/step differs")
            validate_trainable_checkpoint(model, saved_best, encoder_lora_metadata(model))
        # Validate sampler identity/cursors before any durable output mutation.
        sampler.load_state_dict(resume["sampler"])
        if (sampler.tr.draws != resume["step"]*sampler.tr_per_update
                or sum(stream.draws for stream in sampler.replay.values()) != resume["step"]*sampler.replay_per_update):
            raise ValueError("Exact-resume sampled clips disagree with completed update count")
    config = {**arguments, "public_turkish_data_audit":public_data_audit, "steps":configured_steps,"base_revision":BASE_REVISION,
              "transformers_commit":"536ecc007387a50e77603bb5d92100e9b07514cc",
              "method":"bucketed true-batch all-head adaptation plus encoder attention LoRA", "TSC_closed_source_transition":tsc_source,
              "encoder_base_frozen":True,"encoder_frozen":False,"batchnorm_stats_frozen":True,
              "encoder_dropout_and_layerdrop_disabled":True,"encoder_activation_checkpointing":not args.no_encoder_checkpoint,
              "joint_projection_time_chunk_size":args.chunk_size,"full_vocabulary_tdt_objective":True,"native_loss":native_loss_identity(args),
              "trainable_parameters":sum(parameter.numel() for parameter in trainable),
              "encoder_adapter_parameters":sum(parameter.numel() for parameter in adapters),"lora_layout":layout,
              "turkish_clips_used":len(tr),"replay_clips_used":len(replay),
              "turkish_unique_pool_hours":math.fsum(row["duration"] for row in tr)/3600,
              "replay_unique_pool_hours":math.fsum(row["duration"] for row in replay)/3600,
              "excluded_training_counts":{name:counts_before[name]-len(pool) for name,pool in (("tr",tr),("replay",replay))},
              "effective_batch":sampler.effective_batch,"tr_clips_per_update":sampler.tr_per_update,
              "replay_clips_per_update":sampler.replay_per_update,"encoder_cache_namespace":namespace,
              "sampling":f"without-replacement per-stream shuffled cycles, reservoir{args.bucket_size} two-dimensional T/U length buckets; replay language uniform per clip",
              "teacher":"original frozen heads conditioned only on genuine original replay singleton feature cache; TR has explicit lengths/labels and no teacher features; appended IDs excluded from replay",
              "hardware":torch.cuda.get_device_name(0),"weight_dtype":"float32","compute_dtype":"bfloat16 autocast",
              "selection_split":"FLEURS validation only","heldout_test_policy":"separate finalization after stage/controller dev selection",
              "retention_gate":"each dev WER <= original + max(0.5 percentage point,10% relative original WER)",
              "warm_start": {"mode":"exact_resume" if args.resume_exact else "new_optimizer_warm_start" if args.resume_weights else "fresh",
                             "path":str(resume_path) if resume_path else None,"sha256":file_sha(resume_path) if resume_path else None,
                             "source_run_id":resume.get("run_id") if resume else None,"source_step":resume.get("step") if resume else None},
              "config_identity":config_identity,"code_sha256":source_sha,"input_sha256":input_sha,"native_runtime_migration":runtime_migration_evidence(args),
              "training_throughput_timing":"sum of measured sampling/data/cache/STFT/H2D/forward/backward/Adam intervals; excludes development/checkpoint/history serialization/startup"}
    write_json(run_dir / "run_config.json", config)
    write_json(OUTPUT / "run_config.json", config)
    write_json(run_dir / "vocabulary_audit.json", vocabulary)
    write_json(OUTPUT / "vocabulary_audit.json", vocabulary)
    write_json(run_dir / "training_pool.json", {"turkish":tr,"replay":replay,"input_sha256":input_sha}); write_json(run_dir / "TSC_supplemental_acceptance.json", public_data_audit)
    history, start_step = [], 0
    train_wall = evaluation_wall = audio_seconds = 0.
    language_audio = {language:0. for language in LANGUAGES}
    best, selected_step, patience, retention_failures = math.inf, None, 0, 0
    next_eval = args.eval_epoch_fraction
    initial_gate = None
    model.train()
    model.encoder.eval()
    if args.resume_exact:
        if resume["run_id"] != args.run_id or not exact_resume_identity_matches(resume, args, config_identity):
            raise ValueError("Exact resume requires identical run ID, sources, data, layout and training arguments")
        optimizer.load_state_dict(resume["optimizer"])
        sampler.load_state_dict(resume["sampler"])
        random.setstate(resume["python_rng"])
        torch.set_rng_state(resume["torch_rng"])
        torch.cuda.set_rng_state_all(resume["cuda_rng"])
        history, start_step = resume["history"], resume["step"]
        if len(history) != start_step or (history and history[-1]["step"] != start_step):
            raise ValueError("Exact-resume history/checkpoint step mismatch")
        best, selected_step, patience, retention_failures = resume["best_score"], resume["selected_step"], resume["patience"], resume["retention_failures"]
        next_eval = resume["next_eval_epoch"]
        train_wall, evaluation_wall, audio_seconds = resume["train_wall_seconds"], resume["evaluation_wall_seconds"], resume["audio_seconds"]
        language_audio, initial_gate = resume["language_audio_seconds"], resume["initial_gate"]
    else:
        before = time.perf_counter()
        initial_dev = evaluate(model, processor, f"{args.run_id}_step_0", splits=("validation",))
        initial_gate = development_gate(initial_dev, baseline)
        evaluation_wall += time.perf_counter()-before
        write_json(run_dir / "initial_development.json", initial_gate)
        if initial_gate["retention_gate_passed"] and initial_gate["turkish_wer"] < baseline["validation_tr"]["asr"]["wer"]:
            best, selected_step = initial_gate["turkish_wer"], 0
    model.train()
    model.encoder.eval()
    def save(path, step, reason=None):
        save_training_state(path, model, optimizer, sampler, run_id=args.run_id, step=step,
                            config_identity=config_identity, history=history, best_score=best,
                            selected_step=selected_step, patience=patience, retention_failures=retention_failures,
                            next_eval_epoch=next_eval, train_wall_seconds=train_wall, audio_seconds=audio_seconds,
                            language_audio_seconds=language_audio, evaluation_wall_seconds=evaluation_wall,
                            initial_gate=initial_gate, stop_reason=reason)
    if selected_step == 0 and not args.resume_exact:
        save(best_path, 0)
    save(latest_path, start_step)
    new_parameters = (model.decoder.embedding.weight, model.joint.head.weight, model.joint.head.bias)
    loop_start = time.perf_counter()
    stop_reason = "configured_steps_complete"
    last_evaluated_step = start_step if args.resume_exact else 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"Batch {args.batch_size} x accumulation{args.accumulation}; TR {len(tr)}; replay {len(replay)}; configured {configured_steps} steps", flush=True)

    def evaluate_current(step, final=False):
        nonlocal best, selected_step, patience, retention_failures, evaluation_wall, last_evaluated_step, next_eval
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        before = time.perf_counter()
        dev = evaluate(model, processor, f"{args.run_id}_step_{step}", splits=("validation",))
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
        retention_failures = 0 if gate["retention_gate_passed"] else retention_failures+1
        # Plateau counts full-epoch evaluations. Initial quarter/half screens
        # can identify gate failures without prematurely consuming3 epochs.
        epoch = sampler.turkish_epoch
        full_epoch = abs(epoch-round(epoch)) <= sampler.tr_per_update/len(tr) + 1e-9
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
            batch = collate_training_batch(processor, group, blank_id=model.config.blank_token_id,
                                            old_size=old_size, device="cuda")
            result = batch_objective(model, teacher, batch, old_size=old_size, replay_kl=args.replay_kl,
                                      duration_kl=args.duration_kl, feature_distillation=args.feature_distillation,
                                      chunk_size=args.chunk_size, tdt_loss_fn=loss_function,
                                      tdt_host_lengths=args.tdt_implementation in ("v2", "nemo"))
            scaled = result.loss/args.accumulation
            if not torch.isfinite(scaled):
                raise RuntimeError(f"Nonfinite true-batch objective at step{step}")
            scaled.backward()
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
        train_wall += time.perf_counter()-step_start
        audio_seconds += step_audio
        record = {"step":step, **totals,"gradient_norm":float(norm),"elapsed_seconds":time.perf_counter()-loop_start,
                  "process_wall_seconds":time.perf_counter()-process_start,"train_wall_seconds":train_wall,
                  "evaluation_wall_seconds":evaluation_wall,"audio_seen_seconds":audio_seconds,
                  "audio_minutes_per_wall_minute":audio_seconds/train_wall,
                  "turkish_audio_seen_seconds":language_audio["tr"],"replay_audio_seen_seconds":audio_seconds-language_audio["tr"],
                  "language_audio_seen_seconds":language_audio.copy(),"step_audio_seen_seconds":step_audio,
                  "sampled_source_ids":sampled_sources,"turkish_epoch":sampler.turkish_epoch,
                  "turkish_clips_seen":sampler.tr.draws,"turkish_unique_clips_seen":len(sampler.tr.seen),
                  "replay_clips_seen":sum(stream.draws for stream in sampler.replay.values()),
                  "replay_language_clip_counts":sampler.replay_language_counts.copy(),
                  "head_lr":optimizer.param_groups[0]["lr"],"lora_lr":optimizer.param_groups[1]["lr"],
                  "new_token_update_multiplier":args.new_token_lr_multiplier,
                  "replay_kl_coefficient":args.replay_kl,"duration_kl_coefficient":args.duration_kl,"feature_distillation_coefficient":args.feature_distillation,
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
            if args.early_stop_patience and patience >= args.early_stop_patience:
                stop_reason = "development_plateau"
                break
            if args.target_turkish_wer is not None and gate["retention_gate_passed"] and gate["turkish_wer"] <= args.target_turkish_wer:
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
                 "turkish_validation_wer":best if selected_step is not None else None,
                 "initial_turkish_validation_wer":initial_gate["turkish_wer"],
                 "improved_over_initial_model":selected_step is not None and best < initial_gate["turkish_wer"],
                 "selection_split":"FLEURS validation only","heldout_tests_evaluated":False,
                 "weights":str(best_path if selected_step is not None else latest_path),
                 "latest_weights":str(latest_path),"turkish_epochs_completed":sampler.turkish_epoch,
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
    from native_tdt_checkpointed_v3 import configuration_identity
    if args.native_backend_provenance is None or not args.native_backend_provenance.is_file():
        raise ValueError("Official loss requires --native-backend-provenance from pre-Torch native launcher")
    evidence = json.loads(args.native_backend_provenance.read_text(encoding="utf-8"))
    if (evidence.get("kind") != "native_tdt_training_backend_v3" or evidence.get("status") != "ready"
            or evidence.get("run_id") != args.run_id or evidence.get("tdt_implementation") != "nemo"
            or evidence.get("trainer_sha256") != file_sha(__file__)
            or evidence.get("adapter_sha256") != file_sha(OUTPUT / "native_tdt_checkpointed_v3.py")):
        raise ValueError("Native startup provenance run/trainer/adapter identity differs")
    preload = evidence.get("preload", {})
    if (preload.get("kind") != "coherent_native_cuda12_9_preload_v1"
            or preload.get("module_sha256") != file_sha(OUTPUT / "preload_nemo_linker_v1.py")
            or preload.get("successful_launcher_sha256") != file_sha(OUTPUT / "launch_nemo_linker_v4.py")
            or preload.get("actual_binding_linker", {}).get("version") != [12, 9]
            or preload.get("cuda_initialized") is not False):
        raise ValueError("Native compiler/linker startup evidence differs")
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
    from native_tdt_checkpointed_v3 import chunked_tdt_loss
    native_loss_identity(args)
    width = 8199 + 5
    for row in (*tr, *replay):
        size = int(row["encoder_frames"]) * (int(row["target_tokens"]) + 1) * width * 4
        if size > args.native_max_joint_bytes:
            raise ValueError(f"Exact native joint byte bound exceeded: {row['source_id']} needs{size}; no cropping/dropping")
    return partial(chunked_tdt_loss, sigma=0.0, omega=0.0, recompute=True,
                   max_joint_bytes=args.native_max_joint_bytes)


def canonical_identity_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def runtime_migration_evidence(args):
    if args.resume_runtime_migration is not None:
        raise ValueError("A changed TSC pool is a fresh scientific arm, not an old-state runtime migration")
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
    p.add_argument("--epochs",type=float,default=1.)
    p.add_argument("--steps",type=int)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--accumulation",type=int,default=2)
    p.add_argument("--tdt-implementation",choices=("original","v2","nemo"),default="nemo"); p.add_argument("--native-backend-provenance",type=Path); p.add_argument("--native-max-joint-bytes",type=int,default=1_700_000_000); p.add_argument("--resume-runtime-migration",type=Path)
    p.add_argument("--eval-per-epoch",type=float,default=1.); p.add_argument("--eval-every-updates",type=int)
    p.add_argument("--eval-epoch-fraction",type=float,default=.25)
    p.add_argument("--head-lr",type=float,default=5e-5)
    p.add_argument("--lora-lr",type=float,default=3e-4)
    p.add_argument("--new-token-lr-multiplier",type=float,default=6.)
    p.add_argument("--replay-fraction",type=float,default=.5)
    p.add_argument("--replay-kl",type=float,default=10.)
    p.add_argument("--duration-kl",type=float,default=.5)
    p.add_argument("--feature-distillation",type=float,default=1.)
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
    closed_source_identity(args)
    return args


if __name__ == "__main__":
    args = validate_arguments(parser().parse_args())
    run(args)
