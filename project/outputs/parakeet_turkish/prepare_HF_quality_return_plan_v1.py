"""Read-only HF quality curve/domain exposure and conditional next-arm plan."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from plan_native_next_quality_arm_v1 import diagnostic

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    run_id = "adaptive_20261011_native_split_resume_01"
    run_dir = OUT / "runs" / run_id
    cfg_path, history_path, selection_path = [run_dir / n for n in ("run_config.json", "training_history.json", "selection.json")]
    cfg, history, selection = [read(p) for p in (cfg_path, history_path, selection_path)]
    tr_path = ROOT / "work/data/tr_public_v3_train.jsonl"
    sources, domain_counts, domain_audio = {}, Counter(), Counter()
    with tr_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                domain = "FLEURS" if row["source_id"].startswith("google/fleurs:") else "CommonVoice"
                sources[row["source_id"]] = (domain, row["duration"])
                domain_counts[domain] += 1
                domain_audio[domain] += row["duration"]
    if len(sources) != 74883 or domain_counts["FLEURS"] != 2037:
        raise ValueError("Require unchanged fullaccepted74883TRAINsources.")
    sampled_counts, sampled_audio, unique = Counter(), Counter(), set()
    for record in history:
        for sid in record["sampled_source_ids"]:
            if sid in sources:
                domain, duration = sources[sid]
                sampled_counts[domain] += 1
                sampled_audio[domain] += duration
                unique.add(sid)
    last = history[-1]
    if len(unique) != last["turkish_unique_clips_seen"] or sum(sampled_counts.values()) != last["turkish_clips_seen"]:
        raise ValueError("ExactpublishedHFsourceexposurecounts differ.")
    if abs(sum(sampled_audio.values()) - last["turkish_audio_seen_seconds"]) > 1e-6:
        raise ValueError("TRsamplingaudio doesnot matchpublishedcounters.")
    curve = [{"step": 0, "turkish_wer": selection["initial_turkish_validation_wer"], "retention_gate_passed": True,
              "kind": "qualifiedwarmstartfromsource1871"}]
    for row in history:
        if "validation" in row:
            curve.append({"step": row["step"], "TR_pool_fraction": row["turkish_epoch"], **row["validation"]})
    baseline_pred = OUT / "results/adaptive_20261010_continuation_03_resume_B8_step_1871/validation_tr.jsonl"
    latest_pred = OUT / f"results/{run_id}_step_862/validation_tr.jsonl"
    results = {}
    for name, path in (("qualified_source1871", baseline_pred), ("latest862", latest_pred)):
        if path.exists():
            group = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
            results[name] = diagnostic(group)
    mean_cv = domain_audio["CommonVoice"] / domain_counts["CommonVoice"]
    mean_fleurs = domain_audio["FLEURS"] / domain_counts["FLEURS"]
    mean_replay = last["replay_audio_seen_seconds"] / last["replay_clips_seen"]
    uniform_mean = sum(domain_audio.values()) / len(sources)
    weighted_mean = .7 * mean_cv + .3 * mean_fleurs
    current_tokenizer_path = ROOT / "work/research/extension_roundtrip/tokenizer.json"
    current_tok = read(current_tokenizer_path)
    if current_tok["model"]["vocab"].get("<blank>") != 8192 or len(current_tok["model"]["vocab"]) != 8199:
        raise ValueError("Require HFcurrentblank8192+sixcorepieces.")
    candidate = ROOT / "work/turkish_bpe_candidates_v1/add_512"
    candidate_additions = read(candidate / "added_pieces.json")
    row_mapping = [{"piece": r["piece"], "candidate_native_lexical_id": r["native_id"],
                    "future_HF_lexical_id": r["native_id"] + 1, "left": r["left"], "right": r["right"]}
                   for r in candidate_additions]
    if len(row_mapping) != 512 or [r["future_HF_lexical_id"] for r in row_mapping] != list(range(8199, 8711)):
        raise ValueError("IncorrectHF512rowmap.")
    native350 = OUT / "native_clean_Turkish_DEV_progress_100_350_v1.json"
    bpe_report = OUT / "turkish_bpe_candidates_v1.json"
    output = {
        "schema": 1, "kind": "HF_quality_return_domain_exposure_next_arm_plan_v1", "CPU_only": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "TEST_read": False, "active_sources_changed": False,
        "preserved_checkpoint_quality": {"qualified_best_local_step": selection["selected_step"],
            "qualified_best_wer": selection["turkish_validation_wer"], "completed_latest_step": selection["completed_steps"],
            "qualified_best_weights": selection["weights"], "latest_weights": selection["latest_weights"],
            "latest862_wer": last["validation"]["turkish_wer"], "latest862_retention_passed": last["validation"]["retention_gate_passed"],
            "selection_policy": "Keepbest64.9697%checkpoint; exactcontinuationoflatest862 isatrainingtrajectorychoice, notanacceptedqualitywinner."},
        "published_DEV_curve": curve, "fixed_DEV_letter_diagnostics": results,
        "HF_current_recipe": {k: cfg.get(k) for k in ("rank", "alpha", "last_n_layers", "head_lr", "lora_lr", "new_token_lr_multiplier", "replay_fraction", "replay_kl", "replay_duration_kl", "feature_distillation", "tdt_implementation")},
        "actual_HF862_TR_exposure": {"presentations": sum(sampled_counts.values()), "unique_sources": len(unique),
            "full_pool_sources": len(sources), "unique_pool_fraction": len(unique) / len(sources),
            "domain_clip_counts": dict(sampled_counts), "domain_audio_seconds": dict(sampled_audio),
            "TR_audio_seconds": last["turkish_audio_seen_seconds"], "replay_audio_seconds": last["replay_audio_seen_seconds"],
            "TR_audio_fraction": last["turkish_audio_seen_seconds"] / last["audio_seen_seconds"],
            "TR_audio_min_per_training_wall_min": last["turkish_audio_seen_seconds"] / last["train_wall_seconds"],
            "combined_audio_min_per_training_wall_min": last["audio_seen_seconds"] / last["train_wall_seconds"]},
        "accepted_TR_domain_pool": {"source": str(tr_path), "sha256": sha(tr_path), "clips": dict(domain_counts),
            "audio_hours": {k: v / 3600 for k, v in domain_audio.items()},
            "FLEURS_clip_fraction": domain_counts["FLEURS"] / len(sources),
            "FLEURS_TR_audio_fraction": domain_audio["FLEURS"] / sum(domain_audio.values()),
            "mean_clip_seconds": {"CommonVoice": mean_cv, "FLEURS": mean_fleurs}},
        "conditional_next_arm": {
            "first": "Continue the current HF recipe on the actual57.3h pool to meaningful coverage.862 updates/9.21% is not one whole epoch; development scores around65% do not establish a capacity plateau.",
            "if_full_pass_stalls": "Upsample FLEURS TRAIN before increasing LoRA rank or adding512 BPE pieces:30% of Turkish draws from FLEURS,70% from CV. Keep replay fraction, KD, learning rates, rank and six-row boost unchanged to isolate the domain effect.",
            "sampler": "Two independent cycles without replacement (CV and FLEURS), seeded stream selection with checkpointed state/RNG/draw counts. Keep fixed heldout-isolated source pools and unique pool IDs; record repeated TRAIN exposures explicitly.",
            "fair_selection_budget": "Use the same qualified starting checkpoint. Compare matched Turkish audio hours and wall time; log unique CV/FLEURS coverage and cycles separately from weighted draws. Keep all five development retention gates.",
            "coverage_metric": "With upsampling, draws/74883 does not measure a unique Turkish epoch. Report unique sources/74883, CV coverage and FLEURS repeat cycles.",
            "duration_projection_not_GPU_speed": {"uniform_TR_mean_seconds": uniform_mean, "TR_mean_seconds_at_FLEURS30percent": weighted_mean,
                "sampled_replay_mean_seconds": mean_replay,
                "uniform_TR_audio_fraction_with50percent_replay_clips": uniform_mean / (uniform_mean + mean_replay),
                "domain30_TR_audio_fraction_with50percent_replay_clips": weighted_mean / (weighted_mean + mean_replay)},
            "next_capacity_arm": "After a controlled data/domain comparison: rank8→16 with alpha16→32 retains scale2. Copy old A rows/B columns and use random new A plus zero new B; this preserves the initial function mathematically. Use a fresh optimizer for the controlled arm. Current evidence does not identify rank8 as the bottleneck.",
        },
        "HF_plus512_compatibility_plan": {
            "native_candidate_not_direct_HF_drop_in": True,
            "current_HF_token_vocab_including_blank": 8199, "new_HF_token_vocab_including_blank": 8711,
            "blank_ID_stays": 8192, "existing_six_HF_IDs_stay": list(range(8193, 8199)),
            "new_HF_lexical_rows": [8199, 8711], "old_duration_rows": list(range(8199, 8204)),
            "new_duration_rows": list(range(8711, 8716)),
            "row_preservation": "Copy existing learned HF embedding/token-head rows0:8199, including blank and all six characters. Move five duration rows8199:8204→8711:8716. Preserve encoder LoRA, recurrent decoder and joint hidden state.",
            "tokenizer_engineering": "Current HF core BPE vocabulary includes blank8192 and six characters, rather than substring AddedTokens. Append512 candidate pieces using native_id+1 and append their left/right merges after the original HF merges. Keep the normalizer, decoder and special blank fixed. Verify actual HF fast-tokenizer labels, compression, roundtrip and old-language encoding changes on CPU before resizing the model.",
            "weight_initialization": "Initialize new pieces from their TRAIN-learned composition using existing HF8199 rows, with a declared head-bias penalty and fresh optimizer. New classes alter softmax normalization; preserved old weights do not imply exact old-language behavior.",
            "limits": "Native SP compression of28.18% on heldout TRAIN and28.27% on all TRAIN applies to native SP. Measure HF fast-tokenizer compression separately. Wider vocabulary cost can increase for the mixed dataset even when Turkish sequences shrink.",
            "candidate_native_rows_to_future_HF_rows": row_mapping,
            "candidate_report_path": str(bpe_report), "candidate_report_sha256": sha(bpe_report),
            "status": "Planonly; noHFtokenizer/model/optimizerchangesexecuted",
        },
        "native350_evidence": {"path": str(native350), "sha256": sha(native350),
            "interpretation": "Native350ordinaryletterlearningbutzeronew-sixemissions supportsnativeacquisitionissue; itdoesnotjustifyfurtherHFboostbecauseHFalreadyusescorecharactersand6x."},
        "source_files_sha256": {str(p): sha(p) for p in (cfg_path, history_path, selection_path, baseline_pred, latest_pred, current_tokenizer_path)},
        "script_sha256": sha(__file__),
    }
    target = OUT / "HF_quality_return_next_arm_plan_v1.json"
    if target.exists():
        raise FileExistsError("Preservepriorplan.")
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(target), "sha256": sha(target), "quality": output["preserved_checkpoint_quality"],
        "HF862_exposure": output["actual_HF862_TR_exposure"], "domains": output["accepted_TR_domain_pool"],
        "duration_projection": output["conditional_next_arm"]["duration_projection_not_GPU_speed"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
