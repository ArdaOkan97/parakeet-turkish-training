"""CPU-only emitted Turkish-character/exposure comparison at saved native DEV."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from plan_native_next_quality_arm_v1 import diagnostic

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="full_nemo_turkish_20261011_clean_warmstart_08")
    parser.add_argument("--steps", type=int, nargs="+", default=[100, 350])
    parser.add_argument("--output", default=str(OUT / "native_clean_Turkish_DEV_progress_100_350_v1.json"))
    args = parser.parse_args()
    directory = OUT / "runs" / args.run_id
    groups, sources, identities = {}, {}, None
    gates = {}
    for step in args.steps:
        path = directory / f"dev_step_{step}_tr.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        group = data["predictions"]
        current = [(r["source_id"], r["reference"]) for r in group]
        if len(group) != 50 or (identities is not None and current != identities):
            raise ValueError("Fixed50 DEV source/ref/order mismatch.")
        identities = current
        groups[str(step)] = diagnostic(group)
        sources[str(path)] = sha(path)
        gate_path = directory / f"development_gate_step_{step}.json"
        gates[str(step)] = json.loads(gate_path.read_text(encoding="utf-8"))
        sources[str(gate_path)] = sha(gate_path)
    import sentencepiece as spm
    model = ROOT / "work/nemo_direct_trial_v1/tokenizer/tokenizer.model"
    sp = spm.SentencePieceProcessor(model_file=str(model))
    if sp.get_piece_size() != 8198:
        raise ValueError("Expectedfrozen8198nativeSP.")
    full = ROOT / "work/nemo_direct_trial_v1/manifests_clean_v1/train_balanced_full_pass.jsonl"
    exposures, counts, durations, new_ids = {}, Counter(), Counter(), Counter()
    tr_seen = set()
    max_row = max(args.steps) * 16
    with full.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= max_row:
                break
            row = json.loads(line)
            lang = row["language"]
            counts[lang] += 1
            durations[lang] += row["duration"]
            if lang == "tr":
                tr_seen.add(row["source_id"])
                counts["TR_FLEURS" if row["dataset"] == "google/fleurs" else "TR_CommonVoice"] += 1
                new_ids.update(x for x in sp.encode(row["text"]) if 8192 <= x < 8198)
            step = (i + 1) // 16
            if (i + 1) % 16 == 0 and step in args.steps:
                exposures[str(step)] = {"prefix_rows": i + 1, "clip_counts": dict(counts),
                    "audio_seconds_by_language": dict(durations), "TR_audio_fraction": durations["tr"] / sum(durations.values()),
                    "unique_TR_sources": len(tr_seen), "fraction_TR_pool_seen": len(tr_seen) / 74883,
                    "TR_new_six_token_target_ID_counts": {str(t): new_ids[t] for t in range(8192, 8198)},
                    "token_ID_to_piece": {str(t): sp.id_to_piece(t) for t in range(8192, 8198)}}
    if set(exposures) != {str(s) for s in args.steps}:
        raise ValueError("Couldnot reconstructdeclaredprefixexposure.")
    report = {"schema": 1, "kind": "native_clean_saved_DEV_Turkish_character_progress_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_id": args.run_id,
        "CPU_only": True, "Torch_NeMo_imported": "torch" in sys.modules or "nemo" in sys.modules,
        "fixed_DEV_sources_refs_order_exact": True, "TEST_read": False,
        "steps": groups, "development_gates": gates, "source_training_prefix_exposure": exposures,
        "source_files_sha256": {**sources, str(model): sha(model), str(full): sha(full)},
        "exposure_limits": ["First100updatescheckpointlearnedrawTRAINlabels; subsequentv10 continuationusescleanlabels. Sameorderedsource/audio prefix.",
                            "Thisdiagnostic doesnotchange anyweight/label/optimizer/RNG, choose aTEST-dependentwinner, or replace officialscoring.",
                            "One minimumedit characteralignment is diagnostic; native greedy new-piece acquisition mustbe measuredbyrawemissions, not normalizeduppercaseI."],
        "script_sha256": sha(__file__)}
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("PreserveexistingDEVreview.")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(target), "sha256": sha(target), "steps": groups,
                      "exposure": exposures}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
