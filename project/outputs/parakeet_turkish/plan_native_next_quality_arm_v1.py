"""CPU/stdlib fixed-DEV follow-up and inert native+512 vocabulary row plan."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from calculate_audio_throughput import Cycle
from diagnose_native_Turkish_DEV_v1 import align, load, SOURCES
from metrics import normalize_text, score_transcriptions, edit_distance

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def diagnostic(group):
    score = score_transcriptions(group, language="tr")
    core_correct = ordinary_correct = core_total = ordinary_total = 0
    confusion = Counter()
    raw_chars = Counter(c for r in group for c in r["hypothesis"])
    for row in group:
        ref, hyp = normalize_text(row["reference"], "tr"), normalize_text(row["hypothesis"], "tr")
        for a, b in align(ref, hyp):
            if a is not None and a in "ığş":
                core_total += 1
                core_correct += a == b
                if a != b:
                    confusion[a + "→" + (b if b is not None else "DELETED")] += 1
            elif a and a.isalpha():
                ordinary_total += 1
                ordinary_correct += a == b
    return {"asr": {k: v for k, v in score.items() if k != "details"},
            "raw_hypothesis_six_new_character_counts": {c: raw_chars[c] for c in "ğĞıİşŞ"},
            "normalized_core_character_correct": core_correct, "normalized_core_character_reference_total": core_total,
            "normalized_core_character_recall": core_correct / core_total,
            "ordinary_reference_letters": ordinary_total, "ordinary_aligned_correct_letters": ordinary_correct,
            "ordinary_aligned_letter_recall": ordinary_correct / ordinary_total,
            "top_core_character_confusions": dict(confusion.most_common(12)),
            "letter_recall_is_one_minimum_edit_alignment_diagnostic_not_ASR_accuracy": True}


def main():
    diag_path = OUT / "native_Turkish_DEV_error_mode_review_v1.json"
    diag = read(diag_path)
    hf_dir = OUT / "runs/lora_20261010_01"
    hf_config, hf_history = read(hf_dir / "run_config.json"), read(hf_dir / "training_history.json")
    audit_path = OUT / "lora_training_data_verification.json"
    audit = read(audit_path)
    accepted_tr = audit["streams"]["turkish_commonvoice"]["accepted_source_ids_and_labels"] + audit["streams"]["turkish_fleurs_train"]["accepted_source_ids_and_labels"]
    accepted_replay = audit["streams"]["retention_fleurs_train"]["accepted_source_ids_and_labels"]
    rng = random.Random(hf_config["seed"] + 1)
    tr_stream = Cycle(accepted_tr, rng)
    languages = ["en", "es", "it", "fr", "de"]
    replay_streams = {lang: Cycle([r for r in accepted_replay if r["language"] == lang], rng) for lang in languages}
    sampled, sums, domains = [], Counter(), Counter()
    for step in range(1, 401):
        for micro in range(hf_config["accumulation"]):
            row = tr_stream.draw() if micro % 2 == 0 else replay_streams[rng.choice(languages)].draw()
            sampled.append(row)
            sums[row["language"]] += row["duration"]
            if row["language"] == "tr":
                domains["FLEURS" if row["source_id"].startswith("google/fleurs:") else "CommonVoice"] += 1
        if len(tr_stream.seen) != hf_history[step - 1]["turkish_unique_clips_seen"]:
            raise ValueError("HF400 exact sampler reconstruction mismatch.")
    hf400_path = OUT / "results/lora_20261010_01_step_400/validation_tr.jsonl"
    group400 = records(hf400_path)
    native100 = load(SOURCES["native_trained_100"])
    if [(r["source_id"], r["reference"]) for r in group400] != [(r["source_id"], r["reference"]) for r in native100]:
        raise ValueError("HF400/native100 DEV identities differ.")
    native_pilot = ROOT / "work/nemo_direct_trial_v1/manifests/train_balanced_250.jsonl"
    native_full = ROOT / "work/nemo_direct_trial_v1/manifests/train_balanced_full_pass.jsonl"
    native_rows = records(native_pilot)[:1600]
    native_sums = Counter()
    for row in native_rows:
        native_sums[row["language"]] += row["duration"]
    full_sums = Counter()
    with native_full.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                full_sums[row["language"]] += row["duration"]
    bpe_path = OUT / "turkish_bpe_candidates_v1.json"
    bpe_mem_path = OUT / "turkish_bpe_candidates_memory_summary_v1.json"
    bpe, bpe_mem = read(bpe_path), read(bpe_mem_path)
    candidate_dir = ROOT / "work/turkish_bpe_candidates_v1/add_512"
    additions = read(candidate_dir / "added_pieces.json")
    if [x["native_id"] for x in additions] != list(range(8198, 8710)):
        raise ValueError("Candidate512 IDs are not appended after learned8198 lexical rows.")
    report = {
        "schema": 1, "kind": "native_Turkish_next_quality_arm_plan_v1", "CPU_only": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "Torch_NeMo_imported": "torch" in sys.modules or "nemo" in sys.modules, "TEST_read": False,
        "active_training_or_tokenizer_changed": False,
        "fixed_DEV_error_diagnostic": {"path": str(diag_path), "sha256": sha(diag_path)},
        "matched_TR_clip_count_comparison": {
            "HF400": {**diagnostic(group400), "optimizer_updates": 400, "TR_presentations": 800,
                "unique_TR_sources": len(tr_stream.seen), "all_presentations": len(sampled),
                "audio_seconds_by_language": dict(sums), "TR_audio_fraction": sums["tr"] / sum(sums.values()),
                "TR_domain_presentations": dict(domains), "new_row_Adam_delta_multiplier": hf_config["new_token_lr_multiplier"],
                "head_lr": hf_config["head_lr"], "lora_lr": hf_config["lora_lr"]},
            "native100": {**diagnostic(native100), "optimizer_updates": 100, "TR_presentations": 800,
                "all_presentations": len(native_rows), "audio_seconds_by_language": dict(native_sums),
                "TR_audio_fraction": native_sums["tr"] / sum(native_sums.values()),
                "new_row_Adam_delta_multiplier": 1, "native100_unique_TR_pool_fraction": 800 / 74883},
            "limits": ["Same800 Turkish presentations, not identical clips/audio/objective/init/RNG or optimizer updates.",
                       "HF400 retention gate failed (EN delta0.669856 percentage point); native100 gate passed.",
                       "HF15x and finalHF1871 6x are post-Adam new-row delta multipliers, not merely adaptive-gradient scaling.",
                       "Raw new-character counts distinguish true six-piece spellings from legacy uppercaseI→normalizedı."],
        },
        "full_balanced_native_manifest_audio_distribution": {
            "path": str(native_full), "sha256": sha(native_full), "audio_hours_by_language": {k: v / 3600 for k, v in full_sums.items()},
            "total_audio_hours": sum(full_sums.values()) / 3600,
            "TR_audio_fraction": full_sums["tr"] / sum(full_sums.values()),
            "TR_clip_fraction": .5, "padding_repeats_declared": 5,
            "interpretation": "50% clips assigns substantially less than50% acoustic time to short TurkishCV recordings."},
        "native_plus512_plan": {
            "status": "inert candidate; no weight/model/optimizer migration or activation executed",
            "candidate_files": {str(candidate_dir / name): sha(candidate_dir / name) for name in ("tokenizer.model", "tokenizer.vocab", "vocab.txt", "added_pieces.json")},
            "old_lexical_size": 8198, "new_lexical_size": 8710, "old_blank": 8198, "new_blank": 8710,
            "old_duration_rows": list(range(8199, 8204)), "new_duration_rows": list(range(8711, 8716)),
            "embedding_row_copy": {"existing_lexical": "0:8198→0:8198 byte-exact, including all learnedsix rows", "blank_padding": "8198→8710 byte-exact"},
            "joint_row_copy": {"existing_lexical": "0:8198→0:8198 byte-exact", "blank": "8198→8710", "durations": "8199:8204→8711:8716"},
            "all_other_model_state": "Preserve current trained encoderLoRA, decoder recurrent layers, joint hidden/projections and frontend.",
            "new_row_initialization": "Declared composition donor: recursively flatten candidate left/right pieces to current8198 lexical IDs; initialize new embeddings/head weights from length-weighted mean of those learned rows, with separately declared conservative head-bias penalty. This is an initialization hypothesis, not exact sequence equivalence.",
            "native_API_repair": "Official change_vocabulary reinitializes decoder/joint and resetsdefaultRNNT; restore all saved current weights afterresize, rebuild released TDTloss sigma.02/omega.1/mean_volume, rebind fused joint+WER, blank-as-padding and decoding vocabulary.",
            "optimizer_policy": "Fresh optimizer/scheduler for clean matched quality arms, or explicitly mapped old Adam lexical/blank/duration moments with zero new-row moments. Do not call changed tokenizer/targets exact training resume.",
            "functional_preservation_limit": "Copied old logits are exact under rowremap. Added512 finite-softmax logits change normalization and may affect old-language decoding even though old weights/IDs are exact; measure true full-vocabularyDEV retention.",
            "controlled_quality_comparison": "From same qualified native checkpoint, compare currentchar vs+512 with same clean recordings/order/TRaudio exposure and initial LR/LoRA. Scorefixed50TR+five50DEV with unchanged normalizer; chooseonlyDEV. Lossvalues are not directly comparable because targetU and vocabV change.",
            "source_reports": {str(bpe_path): sha(bpe_path), str(bpe_mem_path): sha(bpe_mem_path)},
            "measured_CPU_token_and_matrix_evidence": bpe_mem["candidates"]["512"],
            "capacity_and_speed_limit": "T(U+1)V arithmetic is not measured GPU peak/throughput. CurrentB1 stock native long rows require a bounded actualresource check after resize before adoption.",
        },
        "priority_recommendation": [
            "Keep current clean-label native continuation long enough to acquire real TR exposure;100updates/1.07%pool is not evidence of a plateau.",
            "If new-six emissions remain zero after meaningful exposure, first isolate a6x new-row Adam-delta arm from the same checkpoint, preserving blank/duration rows and all gates; this tests acquisition more directly than adding512 untrained classes.",
            "If long-reading DEV remains weak while shortCV improves, compare TRAIN-onlyFLEURS upsampling (e.g30% ofTRdraws) at matchedTRaudio exposure with sourceIDs/repeats explicit; currentnative100 has only20FLEURS clips.",
            "+512 is the next vocabulary/compression arm once charlearning is established or longU remainsresource-limiting; retain currentlearned8198 rows and report old-language segmentation changes.",
        ],
        "source_files_sha256": {str(p): sha(p) for p in (hf400_path, hf_dir / "run_config.json", hf_dir / "training_history.json", audit_path, native_pilot)},
        "script_sha256": sha(__file__),
    }
    target = OUT / "native_Turkish_next_quality_arm_plan_v1.json"
    if target.exists():
        raise FileExistsError("Do not overwrite earlier plan.")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(target), "sha256": sha(target),
        "matched": report["matched_TR_clip_count_comparison"],
        "full_audio": report["full_balanced_native_manifest_audio_distribution"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
