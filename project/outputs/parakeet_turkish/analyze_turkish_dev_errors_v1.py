"""Small, CPU-only diagnostic of existing fixed Turkish DEV hypotheses.

Diagnostic normalizations/oracles below are NOT replacement benchmark scores.
No model, audio, CUDA, or training code is loaded or modified.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

from metrics import edit_distance, normalize_text, score_transcriptions

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")
CASES = {
    "source_1871": "adaptive_20261010_continuation_03_resume_B8_step_1871",
    "native_control_250": "adaptive_20261011_native_memory_fix_01_native_resume_step_250",
    "native_replay25_250": "adaptive_20261011_native_memory_fix_01_public_rank8_replay25_screen_step_250",
    "strong_whisper_development": "strong_whisper_large_v3_greedy_dev_verified",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def alignment(ref, hyp):
    """One minimum-edit traceback, preferring substitutions for tied errors."""
    dp = [list(range(len(hyp) + 1))]
    for i, r in enumerate(ref, 1):
        row = [i]
        for j, h in enumerate(hyp, 1):
            row.append(min(dp[-1][j] + 1, row[-1] + 1, dp[-1][j-1] + (r != h)))
        dp.append(row)
    i, j, result = len(ref), len(hyp), []
    while i or j:
        if i and j and dp[i][j] == dp[i-1][j-1] + (ref[i-1] != hyp[j-1]):
            result.append(("equal" if ref[i-1] == hyp[j-1] else "sub", ref[i-1], hyp[j-1]))
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i-1][j] + 1:
            result.append(("del", ref[i-1], ""))
            i -= 1
        else:
            result.append(("ins", "", hyp[j-1]))
            j -= 1
    return list(reversed(result))


def boundary_oracle(ref, hyp, limit=4):
    """Word-edit DP adding zero-cost exact concatenation arcs of <=4 words.

    This only forgives spacing if letters in both spans are identical.
    It cannot forgive phonetic errors or Turkish diacritic substitutions.
    """
    n, m = len(ref), len(hyp)
    dp = [[10**9] * (m+1) for _ in range(n+1)]
    dp[0][0] = 0
    for i in range(n+1):
        for j in range(m+1):
            value = dp[i][j]
            if i < n:
                dp[i+1][j] = min(dp[i+1][j], value+1)
            if j < m:
                dp[i][j+1] = min(dp[i][j+1], value+1)
            if i < n and j < m:
                dp[i+1][j+1] = min(dp[i+1][j+1], value+(ref[i] != hyp[j]))
                for a in range(1, min(limit, n-i)+1):
                    r = "".join(ref[i:i+a])
                    for b in range(1, min(limit, m-j)+1):
                        if r == "".join(hyp[j:j+b]):
                            dp[i+a][j+b] = min(dp[i+a][j+b], value)
    return dp[n][m]


def top(counter, limit=16):
    return [{"reference": r, "hypothesis": h, "count": count}
            for (r, h), count in counter.most_common(limit)]


def run():
    report = {"schema": 1, "CPU_only": True, "scope": "existing 50 fixed Turkish DEV recordings; no new inference or TEST use", "models": {}}
    base = None
    for name, tag in CASES.items():
        folder = OUT / "results" / tag
        path = folder / "validation_tr.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        identity = [(r["source_id"], r["reference"]) for r in rows]
        if base is None:
            base = identity
        assert identity == base, f"DEV source/reference mismatch: {name}"
        score = score_transcriptions(rows, language="tr")
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        stored = summary["validation_tr"]["asr"] if "validation_tr" in summary else summary["metrics"]
        assert all(score[key] == stored[key] for key in ("wer", "cer", "word_errors", "reference_words", "character_errors", "reference_characters"))
        word_subs, word_dels, word_ins, char_subs = (collections.Counter() for _ in range(4))
        kinds, chars, diag = (collections.Counter() for _ in range(3))
        examples = []
        numeric_rows = []
        for row, detail in zip(rows, score["details"]):
            r, h = detail["reference_normalized"], detail["hypothesis_normalized"]
            word_ops = alignment(r.split(), h.split())
            assert sum(op != "equal" for op, _, _ in word_ops) == detail["word_errors"]
            for op, a, b in word_ops:
                if op != "equal":
                    kinds[op] += 1
                    {"sub": word_subs, "del": word_dels, "ins": word_ins}[op][a, b] += 1
            for op, a, b in alignment(r, h):
                if op != "equal":
                    chars[op] += 1
                    if op == "sub":
                        char_subs[a, b] += 1
                        if a.translate(FOLD) == b.translate(FOLD):
                            chars["diacritic_only_substitutions"] += 1
                    if " " in (a, b):
                        chars["space_operations"] += 1
                    if (a and a in "ğış") or (b and b in "ğış"):
                        chars["new_letter_involved_operations"] += 1
            diag["folded_word_errors"] += edit_distance(r.translate(FOLD).split(), h.translate(FOLD).split())
            diag["folded_character_errors"] += edit_distance(r.translate(FOLD), h.translate(FOLD))
            diag["no_space_character_errors"] += edit_distance(r.replace(" ", ""), h.replace(" ", ""))
            diag["no_space_reference_characters"] += len(r.replace(" ", ""))
            oracle = boundary_oracle(r.split(), h.split())
            diag["boundary_oracle_word_errors"] += oracle
            if any(c.isdigit() for c in r):
                numeric_rows.append(row)
                diag["reference_digit_words"] += sum(any(c.isdigit() for c in word) for word in r.split())
            if detail["word_errors"] - oracle >= 3:
                examples.append({"source_id": row["source_id"], "reference": row["reference"], "hypothesis": row["hypothesis"], "word_errors": detail["word_errors"], "exact_boundary_oracle_word_errors": oracle})
        score.pop("details")
        diag.update({"folded_wer": diag["folded_word_errors"] / score["reference_words"], "folded_cer": diag["folded_character_errors"] / score["reference_characters"], "no_space_cer": diag["no_space_character_errors"] / diag["no_space_reference_characters"], "boundary_oracle_wer": diag["boundary_oracle_word_errors"] / score["reference_words"], "exact_boundary_forgiven_word_edits": score["word_errors"] - diag["boundary_oracle_word_errors"]})
        nonnumeric = [row for row in rows if not any(c.isdigit() for c in row["reference"])]
        numeric_score = score_transcriptions(numeric_rows, language="tr")
        nonnumeric_score = score_transcriptions(nonnumeric, language="tr")
        numeric_score.pop("details")
        nonnumeric_score.pop("details")
        report["models"][name] = {"tag": tag, "predictions_sha256": sha(path), "summary_sha256": sha(folder/"summary.json"), "standard_metrics": score, "word_edit_traceback_counts": dict(kinds), "character_edit_traceback_counts": dict(chars), "diagnostic_only": dict(diag), "digit_reference_subset": numeric_score, "no_digit_reference_subset": nonnumeric_score, "top_word_substitutions": top(word_subs), "top_word_deletions": top(word_dels, 8), "top_word_insertions": top(word_ins, 8), "top_character_substitutions": top(char_subs), "boundary_examples": examples[:5]}
    report["diagnostic_limits"] = ["Folded scores, space-free CER, and the exact concatenation oracle are sensitivity analyses, not new benchmark metrics.", "Character/word confusion categories depend on one tied minimum-edit traceback; diagnostics overlap and must not be added together.", "Fifty DEV clips/825 words are small and have been reused for selection; no generalization or TEST inference claim."]
    destination = OUT / "turkish_dev_error_analysis_v1.json"
    assert not destination.exists(), "Preserve prior analysis"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: {"metrics": values["standard_metrics"], "diagnostics": values["diagnostic_only"], "character_counts": values["character_edit_traceback_counts"], "numeric": values["digit_reference_subset"], "nonnumeric": values["no_digit_reference_subset"]} for name, values in report["models"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    run()
