"""Small stdlib guards for the inactive, fresh-Adam domain sampling arm.

Closure is supplied by ROOT after observing the source worker exit. This helper
does not stop a worker, infer process exit, read audio, or load a checkpoint.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/parakeet_turkish"
LANGUAGES = ("en", "es", "it", "fr", "de")
TOKENIZER_SHA = "b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75"
RECIPE = {"batch_size": 8, "accumulation": 2, "rank": 8, "alpha": 16.,
    "last_n_layers": None, "head_lr": 5e-5, "lora_lr": 3e-4,
    "new_token_lr_multiplier": 6., "replay_fraction": .5, "replay_kl": 10.,
    "duration_kl": .5, "feature_distillation": 1., "tdt_implementation": "nemo",
    "warmup_steps": 25, "schedule": "constant", "seed": 20261010,
    "bucket_size": 128, "max_tokens": 180, "chunk_size": 16,
    "no_encoder_checkpoint": False, "phonetic_sh_donors": False,
    "train_limit": None, "supplement": None, "max_retention_failures": 2}


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_recipe(args):
    for key, expected in RECIPE.items():
        require(getattr(args, key) == expected, "Domain-only arm must retain recipe: " + key)
    require(args.sampling_mode in ("domain30", "uniform") and args.fleurs_fraction == .3,
            "Explicit domain30 or independent-replay uniform comparator required")
    require(type(args.replay_seed) is int, "Explicit integer independent replay seed required")
    require(args.resume_runtime_migration is None,
            "Domain sampling changes scientific state; legacy runtime-only exact migration is not allowed")
    require(args.source_closure is not None,
            "A root-owned CLOSED qualifying source receipt is required")
    require(args.resume_exact is not None or args.resume_weights is not None,
            "Use closed qualifying weights with fresh Adam, or exact resume of this isolated arm")


def closed_source_identity(args):
    closure_path = Path(args.source_closure).resolve()
    require(closure_path.is_relative_to(ROOT), "Closure must stay in workspace")
    closure = read_json(closure_path)
    require(closure.get("kind") == "root_owned_domain30_closed_source_v1"
            and closure.get("passed") is True
            and closure.get("root_observed_owned_session_completed") is True
            and closure.get("child_exit_code") == 0,
            "Root-observed source exit and complete qualifying source closure required")
    paths = {}
    for name in ("source_checkpoint", "source_selection", "source_run_config",
                 "source_training_history", "source_development_summary"):
        item = closure[name]
        path = Path(item["path"]).resolve()
        require(path.is_relative_to(ROOT) and path.is_file()
                and sha_file(path) == item["sha256"], "Closed source changed: " + name)
        paths[name] = path
    selection = read_json(paths["source_selection"])
    config = read_json(paths["source_run_config"])
    history = read_json(paths["source_training_history"])
    summary = read_json(paths["source_development_summary"])
    source_run = closure["source_run_id"]
    selected = selection.get("selected_step")
    require(source_run != args.run_id and config.get("run_id") == selection.get("run_id") == source_run
            and selection.get("accepted") is True and type(selected) is int and selected >= 0,
            "Separate accepted CLOSED source run and selected step required")
    require(type(selection.get("completed_steps")) is int
            and len(history) == selection["completed_steps"]
            and all(row.get("step") == index for index, row in enumerate(history, 1))
            and selected <= len(history), "Closed source history/selection steps disagree")
    require(Path(selection["weights"]).resolve() == paths["source_checkpoint"],
            "Warm start must be the source's selected qualifying weights")
    require(paths["source_development_summary"].parent.name == source_run + "_step_" + str(selected),
            "Source summary must refer to the actual selected DEV model")
    baseline = read_json(OUTPUT / "results/base_v3/summary.json")
    for lang in (*LANGUAGES, "tr"):
        result = summary["validation_" + lang]["asr"]
        require(result.get("utterances") == 50 and math.isfinite(result["wer"]),
                "Every fixed 50-clip development score must exist")
        if lang != "tr":
            original = baseline["validation_" + lang]["asr"]["wer"]
            require(result["wer"] <= original + max(.005, .1 * original),
                    "Source fails an old-language retention gate: " + lang)
    wer = summary["validation_tr"]["asr"]["wer"]
    require(wer == selection["turkish_validation_wer"]
            and wer < baseline["validation_tr"]["asr"]["wer"],
            "Source selected TR score differs or does not qualify")
    require(config["config_identity"]["extended_tokenizer_sha256"] == TOKENIZER_SHA,
            "Source must use the current original8199 tokenizer")
    require(config.get("turkish_clips_used") == 74883 and config.get("replay_clips_used") == 1595,
            "Source must use the complete unchanged accepted TR and original replay pools")
    for key, expected in RECIPE.items():
        require(config.get(key) == expected, "Source recipe differs: " + key)
    if args.resume_exact is None:
        require(Path(args.resume_weights).resolve() == paths["source_checkpoint"],
                "The first domain arm requires original qualifying weights with NEW Adam")
    return {"kind": "closed_qualifying_domain_sampling_source_v1",
        "source_run_id": source_run, "selected_step": selected, "turkish_wer": wer,
        "closure_path": str(closure_path), "closure_sha256": sha_file(closure_path),
        "source_checkpoint": closure["source_checkpoint"],
        "source_config_sha256": closure["source_run_config"]["sha256"],
        "source_selection_sha256": closure["source_selection"]["sha256"],
        "accepted_pool_ids_sha256": {"tr": config["config_identity"]["tr_pool_sha256"],
                                      "replay": config["config_identity"]["replay_pool_sha256"]},
        "fresh_optimizer_on_first_launch": True, "source_optimizer_sampler_RNG_transferred": False,
        "all_five_retention_gates_recomputed": True}
