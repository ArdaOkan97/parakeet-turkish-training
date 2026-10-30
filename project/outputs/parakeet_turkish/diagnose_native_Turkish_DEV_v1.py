"""Fixed-DEV orthography diagnostic; stdlib only, no inference/TEST reads."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import unicodedata
from metrics import normalize_text, score_transcriptions, edit_distance

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
SOURCES = {
    "native_base": OUT / "runs/full_nemo_turkish_20261011_pilot_06/dev_step_0_tr.json",
    "native_trained_100": OUT / "runs/full_nemo_turkish_20261011_warmstart_07/dev_step_100_tr.json",
    "native_trained_101": OUT / "runs/full_nemo_turkish_20261011_warmstart_07/dev_step_101_tr.json",
    "qualified_HF_1871": OUT / "results/adaptive_20261010_continuation_03_resume_B8_step_1871/validation_tr.jsonl",
}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def align(reference, hypothesis):
    """One minimum-edit alignment; diagonal first, then deletion, insertion."""
    n, m = len(reference), len(hypothesis)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    costs[0] = list(range(m + 1))
    for i in range(1, n + 1):
        costs[i][0] = i
        for j in range(1, m + 1):
            costs[i][j] = min(costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
                              costs[i - 1][j] + 1, costs[i][j - 1] + 1)
    i, j, pairs = n, m, []
    while i or j:
        if i and j and costs[i][j] == costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            pairs.append((reference[i - 1], hypothesis[j - 1])); i -= 1; j -= 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            pairs.append((reference[i - 1], None)); i -= 1
        else:
            pairs.append((None, hypothesis[j - 1])); j -= 1
    return list(reversed(pairs))


def load(path):
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))["predictions"]


def main():
    rows = {name: load(path) for name, path in SOURCES.items()}
    first = rows["native_base"]
    identities = [(r["source_id"], r["reference"]) for r in first]
    if len(first) != 50:
        raise ValueError("Expected fixed50 Turkish DEV clips.")
    for name, group in rows.items():
        if [(r["source_id"], r["reference"]) for r in group] != identities:
            raise ValueError(f"Fixed source/reference/order mismatch: {name}")
    fold = str.maketrans({"ı": "i", "ğ": "g", "ş": "s", "ö": "o", "ü": "u", "ç": "c"})
    results = {}
    for name, group in rows.items():
        stats = score_transcriptions(group, language="tr")
        word_ops, char_ops, foreign_letters = Counter(), Counter(), Counter()
        core = {c: Counter() for c in "ığş"}
        ref_counts, hyp_counts = Counter(), Counter()
        no_space_errors = no_space_units = folded_word_errors = 0
        predicted_words = 0
        for row in group:
            ref, hyp = normalize_text(row["reference"], "tr"), normalize_text(row["hypothesis"], "tr")
            ref_counts.update(ref); hyp_counts.update(hyp)
            predicted_words += len(hyp.split())
            no_space_errors += edit_distance(ref.replace(" ", ""), hyp.replace(" ", ""))
            no_space_units += len(ref.replace(" ", ""))
            folded_word_errors += edit_distance(ref.translate(fold).split(), hyp.translate(fold).split())
            for a, b in align(ref.split(), hyp.split()):
                word_ops["insertions" if a is None else "deletions" if b is None else "correct" if a == b else "substitutions"] += 1
            for a, b in align(ref, hyp):
                char_ops["insertions" if a is None else "deletions" if b is None else "correct" if a == b else "substitutions"] += 1
                if a in core:
                    core[a]["correct" if a == b else "deleted" if b is None else "substituted_to_" + b] += 1
            for c in hyp:
                if c.isalpha() and c not in "abcdefghijklmnopqrstuvwxyzçğıöşü":
                    foreign_letters[c] += 1
        results[name] = {
            "asr": {k: v for k, v in stats.items() if k != "details"},
            "word_alignment": dict(word_ops), "character_alignment": dict(char_ops),
            "hypothesis_word_count": predicted_words,
            "reference_core_character_counts": {c: ref_counts[c] for c in "ığş"},
            "hypothesis_core_character_counts": {c: hyp_counts[c] for c in "ığş"},
            "core_character_alignment": {c: dict(v) for c, v in core.items()},
            "foreign_Latin_or_nonLatin_letter_counts": dict(foreign_letters.most_common()),
            "diagnostic_only_space_removed_CER": no_space_errors / no_space_units,
            "diagnostic_only_Turkish_diacritic_folded_WER": folded_word_errors / stats["reference_words"],
            "alignment_tie_breaking_can_affect_individual_confusions": True,
        }
    # Native source100 checkpoint was trained on first100 updates of the pilot.
    pilot = ROOT / "work/nemo_direct_trial_v1/manifests/train_balanced_250.jsonl"
    exposures = [json.loads(line) for line in pilot.read_text(encoding="utf-8").splitlines() if line.strip()][:1600]
    tr = [r for r in exposures if r["language"] == "tr"]
    domains = Counter(r["dataset"] for r in tr)
    domain_hours = {d: sum(r["duration"] for r in tr if r["dataset"] == d) / 3600 for d in domains}
    examples = []
    for index in (0, 10, 19, 40, 46):
        examples.append({"source_id": first[index]["source_id"], "reference": first[index]["reference"],
                         "hypotheses": {name: group[index]["hypothesis"] for name, group in rows.items()}})
    bpe = OUT / "turkish_bpe_candidates_v1.json"
    memory = OUT / "turkish_bpe_candidates_memory_summary_v1.json"
    report = {
        "schema": 1, "kind": "fixed_Turkish_DEV_native_vs_HF_orthography_diagnostic_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "CPU_only": True,
        "Torch_NeMo_imported": "torch" in sys.modules or "nemo" in sys.modules,
        "TEST_read": False, "fixed_DEV_sources_references_order_exact": True,
        "sources_sha256": {str(p): sha(p) for p in SOURCES.values()},
        "results": results, "finite_examples": examples,
        "native100_source_training_exposure": {"source_manifest_path": str(pilot), "source_sha256": sha(pilot),
            "optimizer_updates": 100, "all_clip_presentations": len(exposures), "TR_clips": len(tr),
            "unique_TR_sources": len({r["source_id"] for r in tr}), "accepted_TR_pool": 74883,
            "fraction_unique_TR_pool_seen": len({r["source_id"] for r in tr}) / 74883,
            "TR_audio_seconds": sum(r["duration"] for r in tr), "TR_domain_counts": dict(domains),
            "TR_domain_hours": domain_hours,
            "purpose": "Checkpoint-source exposure only; native07 one later update and current clean continuation not counted."},
        "BPE_evidence": {"candidate_report_path": str(bpe), "candidate_report_sha256": sha(bpe),
                         "memory_summary_path": str(memory), "memory_summary_sha256": sha(memory)},
        "interpretation_limits": [
            "Native100 is100 updates from native base; HF1871 is a longer separately initialized/model-objective experiment, not a matched100-update comparison.",
            "Space/diacritic-folded metrics are diagnostic only and never replace official shared WER/CER.",
            "Predictions cannot uniquely distinguish acoustic alignment, learned language model, and token segmentation causes without a controlled training arm.",
            "Missing six-character emissions are observable; zero emission alone does not prove zero training gradient or unsupported tokenizer.",
        ],
        "script_sha256": sha(__file__),
    }
    target = OUT / "native_Turkish_DEV_error_mode_review_v1.json"
    if target.exists():
        raise FileExistsError("Preserve prior diagnostics.")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(target), "sha256": sha(target), "results": results,
                      "exposure": report["native100_source_training_exposure"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
