import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs/parakeet_turkish"
BINARY = OUTPUT / "tools/backlog.exe"


def cli(*args):
    result = subprocess.run([str(BINARY), *args], cwd=OUTPUT, check=True, capture_output=True, text=True, encoding="utf-8")
    print(result.stdout.splitlines()[0] if result.stdout else "Updated task")


cli("task", "edit", "ASR-2",
    "--acceptance-criteria", "Pin data repositories and preserve source IDs/audio hashes",
    "--acceptance-criteria", "Select up to 3000 CommonVoice Turkish train clips and retain only decoded audio within 1-15 seconds",
    "--acceptance-criteria", "Prepare replay 400/language for en es it fr de and validation 50/test 100 for en es it fr de tr",
    "--acceptance-criteria", "Enforce global cross-language FLEURS sentence-ID and exact audio isolation",
    "--append-notes", "2026-10-10 01:54 PDT: Data preparation completed. The bounded 3000-clip Turkish selection retained 2996 clips (2.15855 hours) after exact decoded-audio duration filtering; four were below the requested minimum. Replay contains 2000 clips (400 per preservation language, 6.04793 hours); development contains 50 and test 100 per language across en/es/it/fr/de/tr. Completed data_summary.json preserves every manifest hash, pinned source revision, and audio provenance.\n\nAn independent CPU audit verified all 17 manifest hashes/counts/hours and all 5896 WAV byte hashes, mono PCM16/16 kHz headers and durations (1.291 GB). Combined/per-language manifests match exactly. Source IDs, source/output audio hashes, and global FLEURS sentence IDs are disjoint across optimizer/development/test sets; no normalized transcript duplicates cross optimizer and heldout sets. All selected replay token IDs fit the original teacher vocabulary. See data_verification.json. Replay audio reaches 30.48 seconds; training token/frame limits are separate from Turkish duration filtering.",
    "--final-summary", "Prepared and independently audited the complete bounded pilot: 2996 Turkish training clips, 2000 five-language replay clips, 300 development clips and 600 test clips. All provenance/audio checks and cross-split sentence/audio isolation passed. Acceptance criterion wording clarifies that the 3000 selection is a bound before exact decoded-duration filtering.",
    "--add-ref", "data_summary.json", "--add-ref", "data_verification.json")

cli("task", "edit", "ASR-2", "--check-ac", "1", "--check-ac", "2", "--check-ac", "3", "--check-ac", "4", "--status", "Done")

cli("task", "edit", "ASR-3", "--append-notes", "2026-10-10 01:54 PDT: Evaluation-cache safety review completed. benchmark.py now records a streaming SHA256 of every pretrained weight/buffer, tokenizer/processor/config, evaluation code and inference mode; each result folder records manifest byte hashes, selected ordered source IDs and limits. It rejects stale weights/tokenizer/references/limits and legacy prediction folders without metadata, validates prediction source provenance, and allows identical resumptions. Eight meaningful CPU-only cache regression checks passed. Complete prepared data is available; actual unchanged-v3 inference is starting, so measured WER/CER and confidence results remain pending.", "--add-ref", "benchmark.py")

cli("task", "edit", "ASR-5", "--append-notes", "2026-10-10 01:54 PDT: Actual pretrained-model integration smoke passed on the RTX 5060 Ti after extending vocabulary to 8199 tokens. Synthetic 3-second audio with Turkish and English targets produced finite TDT losses 13.26910 and 3.89258, finite gradients, no encoder gradients, and nonzero new embedding/head/bias updates after one optimizer step. Peak allocated VRAM was 2.89176 GB and training took 0.90703 seconds. This verifies the extended train path and row-specific Adam update, not real-speech adaptation quality; no model checkpoint was saved. Evidence: actual_extension_gpu_smoke.json.\n\nTraining review fixes are integrated: full development baseline sample-identity/count enforcement, tokenizer/base-version cache namespace, cached-label equality checks, and removal of replay targets outside the original teacher vocabulary. The selected replay manifest actually contains zero such targets. Real training and retention-gated checkpoint selection remain pending.", "--add-ref", "actual_extension_gpu_smoke.json", "--add-ref", "train.py")

subprocess.run([str(ROOT / "work/training-env/Scripts/python.exe"), str(OUTPUT / "refresh_backlog.py")], cwd=OUTPUT, check=True)
