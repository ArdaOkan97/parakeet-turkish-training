"""Initialize this experiment's local Backlog.md project using its official CLI.

Only creates missing tasks; existing task notes/statuses are never overwritten.
No installation, network, Git operations, or commits are performed here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS = [
    {
        "key": "environment",
        "title": "Prepare and verify the local CUDA training environment",
        "status": "Done",
        "labels": "environment,cuda",
        "description": "Use the PC GPU for Parakeet inference and fine-tuning. Keep the Python environment and caches in this workspace.",
        "acceptance": ["Identify the GPU and VRAM", "Install the required local training dependencies", "Pass a real CUDA forward/backward smoke test with finite gradients"],
        "notes": "2026-10-10: NVIDIA GeForce RTX 5060 Ti detected, 16,311 MiB VRAM, driver 610.88. Reused existing Torch 2.7.1+cu128 with sm_120 support through a workspace Python 3.11 system-site environment (work/training-env). Transformers is pinned to its researched development commit. Original v3 checkpoint was downloaded at pinned revision 541d1f99c6b0c3cd0b11a95167540bb8edefd82b. Initial runtime exposed an upstream int32 target/int64 gather mismatch; runtime.py supplies the workaround. The real model CUDA backward smoke test then produced loss 7.087714672088623, no bad gradients, 0.897 seconds, peak allocated VRAM 2.641 GB. This confirms the execution path; it is not a training/evaluation result.",
        "references": ["../../work/smoke.log", "download_model.py", "runtime.py"],
        "dependencies": [],
    },
    {
        "key": "data",
        "title": "Prepare Turkish training data and isolated six-language FLEURS splits",
        "status": "In Progress",
        "labels": "data,benchmark",
        "description": "Use only CommonVoice mirror train for Turkish training. Use FLEURS train for preservation-language replay, validation for checkpoint selection/calibration, and test for the fixed benchmark.",
        "acceptance": ["Pin data repositories and preserve source IDs/audio hashes", "Prepare the first pilot with 3000 CommonVoice Turkish train clips, 1-15 second duration limits", "Prepare replay 400/language for en es it fr de and validation 50/test 100 for en es it fr de tr", "Enforce global cross-language FLEURS sentence-ID and exact audio isolation"],
        "notes": "2026-10-10: Implemented prepare_data.py and metrics.py. Offline synthetic parquet checks passed for embedded-audio extraction, 8-to-16 kHz stereo resampling, stable sampling, absolute manifests, and cross-language sentence-ID rejection. Sampling seed is 20261010 over complete selected split metadata, not a streaming prefix. Pinned FLEURS to 70bb2e84b976b7e960aa89f1c648e09c59f894dd and processed CommonVoice Turkish to 9d8a025f1ae7c5a2f5c43298c7c4910b03995a4b. The first runtime is bounded to 3000 Turkish clips and max 15 seconds; full source train is roughly 20 hours and remains available for later scaling. Explicit selected-language shard downloads have started. Completed data_summary.json/manifests have not yet been verified at backlog initialization.",
        "references": ["prepare_data.py", "metrics.py", "../../work/data_prepare.log", "https://huggingface.co/datasets/google/fleurs", "https://huggingface.co/datasets/ysdede/commonvoice_17_tr_fixed"],
        "dependencies": ["environment"],
    },
    {
        "key": "baseline",
        "title": "Measure the unchanged Parakeet v3 multilingual baseline",
        "status": "To Do",
        "labels": "evaluation,baseline",
        "description": "Transcribe exactly the fixed FLEURS validation/test audio for en es it fr de tr using unchanged Parakeet v3. Save predictions, normalized references, WER/CER, timing, and resource use.",
        "acceptance": ["Save per-utterance original model predictions and provenance", "Report per-language corpus WER/CER on identical fixed sets", "Keep validation and test results separate"],
        "notes": "2026-10-10: Baseline model revision and dataset protocol are pinned. Baseline ASR results are still pending; no Turkish improvement or language-retention claim has been established.",
        "references": ["metrics.py", "download_model.py"],
        "dependencies": ["environment", "data"],
    },
    {
        "key": "vocabulary",
        "title": "Add missing Turkish tokens while preserving existing token IDs and weights",
        "status": "Done",
        "labels": "tokenizer,retention",
        "description": "Extend the original v3 tokenizer/head for missing Turkish characters without reassigning original token IDs, the blank ID, or old duration-head rows.",
        "acceptance": ["Encode and round-trip Turkish ğ Ğ ı İ ş Ş without unknown tokens", "Preserve original token IDs, merge order, old embeddings/token logits/duration logits", "Pass save/reload and idempotency verification"],
        "notes": "2026-10-10: The v3 tokenizer advertises byte fallback but contains no byte pieces; Turkish ğ/ı/ş and capital variants produced unknown tokens. Implemented six appended tokens (8193-8198), retained blank ID 8192, copied donor-character rows, and shifted duration output rows without changing their weights. Synthetic test text unknown tokens fell from 20 to 0. Verified 1,000 random old-alphabet tokenizations, 100 old token/duration logit and greedy-choice probes, full other state preservation, exact save/reload, and second-call idempotency. These checks used a tiny random ParakeetForTDT plus the official tokenizer; actual speech accuracy still requires the baseline/adapted benchmarks.",
        "references": ["extend_vocab.py", "vocabulary_extension_verification.json"],
        "dependencies": [],
    },
    {
        "key": "training",
        "title": "Fine-tune Turkish with balanced multilingual rehearsal and retention checks",
        "status": "In Progress",
        "labels": "training,turkish,retention",
        "description": "Run a bounded local-GPU adaptation experiment with the extended vocabulary, Turkish train data, and preservation-language replay. Select checkpoints on validation only and retain the untouched original model.",
        "acceptance": ["Run real GPU optimizer updates with logged configuration, loss, and checkpoints", "Use an approximately 50/50 Turkish/replay sampler rather than the concatenated row-count ratio", "Select checkpoints using Turkish validation improvement plus preservation-language tolerances", "Save a usable model/processor checkpoint and training report"],
        "notes": "2026-10-10: A real CUDA backward smoke test passed. Benchmark/training integration code is being built. Data preparation and vocabulary extension are ready for integration. The combined manifest's row ratio differs from the intended Turkish/replay mix, so the trainer must weight replay explicitly. No real optimizer adaptation run or trained checkpoint has yet completed. Measured Turkish improvement is pending.",
        "references": ["extend_vocab.py", "prepare_data.py"],
        "dependencies": ["environment", "data", "baseline", "vocabulary"],
    },
    {
        "key": "confidence",
        "title": "Capture and validate word confidence from the actual TDT decoding path",
        "status": "In Progress",
        "labels": "confidence,evaluation",
        "description": "Capture emitted token probabilities from the actual greedy TDT decoding path, aggregate lexical token confidence to words, and assess calibration on validation data without changing transcripts.",
        "acceptance": ["Verify collected emissions align with generated transcript words", "Keep token and duration probability distributions separate", "Evaluate calibration/temperature on validation only", "Describe scores as empirical confidence unless correctness calibration supports probability interpretation"],
        "notes": "2026-10-10: confidence.py has been implemented. The pinned Transformers generate wrapper drops output_scores, so the implementation uses a joint-network forward hook along the real greedy path. It supports batch size one and geometric-mean lexical token probabilities; temperature changes confidence only. Matched real-audio confidence validation/calibration is pending at backlog initialization.",
        "references": ["confidence.py"],
        "dependencies": ["baseline"],
    },
    {
        "key": "v2v3",
        "title": "Compare Parakeet v2 and v3 on matched English audio and confidence",
        "status": "In Progress",
        "labels": "comparison,confidence,evaluation",
        "description": "Compare the English-only v2 model and multilingual v3 model on identical held-out English FLEURS audio, scoring and confidence settings. Keep this comparison separate from Turkish adaptation claims.",
        "acceptance": ["Pin both model revisions and use identical English evaluation IDs", "Report matched WER/CER, inference timing, and confidence/calibration metrics", "Save raw per-utterance results and reproducible settings"],
        "notes": "2026-10-10: v3 revision is 541d1f99c6b0c3cd0b11a95167540bb8edefd82b; v2 revision research recorded ae9ad07059c7c739ffaf932226a8fe64ae2620b0. V2 archive download is in progress and the converter is ready. Matched v2/v3 ASR and confidence real-audio results remain pending.",
        "references": ["confidence.py", "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2", "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3"],
        "dependencies": ["data", "baseline", "confidence"],
    },
    {
        "key": "final-evaluation",
        "title": "Verify Turkish improvement and multilingual retention with paired uncertainty",
        "status": "To Do",
        "labels": "evaluation,retention,report",
        "description": "Evaluate the selected adapted checkpoint and untouched original checkpoint on identical fixed held-out FLEURS test audio. Report exact per-language deltas and paired sentence-cluster uncertainty.",
        "acceptance": ["Report Turkish WER/CER improvement and all five preservation-language deltas", "Compute paired sentence-cluster confidence intervals", "Flag pilot uncertainty and avoid claiming retention in untested languages", "Run full six-language official test splits for a promising checkpoint when resources permit"],
        "notes": "2026-10-10: metrics.py implements corpus WER/CER and paired sentence-cluster bootstrap with accent-preserving Turkish-aware normalization. Offline metric checks passed. A 100-utterance/language pilot is configured; measured retention and a final checkpoint report are pending.",
        "references": ["metrics.py"],
        "dependencies": ["baseline", "training"],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=str(ROOT / "tools" / "backlog.exe"))
    arguments = parser.parse_args()
    binary = Path(arguments.binary).resolve()
    if not binary.is_file():
        raise SystemExit(f"Official local Backlog.md CLI is absent: {binary}")

    def cli(*args: str) -> str:
        result = subprocess.run([str(binary), *args], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8")
        if result.stderr.strip():
            print(result.stderr.strip())
        return result.stdout

    version = cli("--version").strip()
    if not (ROOT / "backlog" / "config.yml").exists() and not (ROOT / "backlog.config.yml").exists():
        print(cli("init", "Parakeet Turkish ASR", "--no-git", "--defaults", "--integration-mode", "cli", "--agent-instructions", "none", "--backlog-dir", "backlog", "--config-location", "folder", "--task-prefix", "ASR"))
    (ROOT / "backlog" / "workflow_overview.txt").write_text(cli("instructions", "overview"), encoding="utf-8")
    existing = json.loads(cli("task", "list", "--json"))["tasks"]
    by_title = {task["title"]: task for task in existing}
    created_keys = []
    for task in TASKS:
        if task["title"] in by_title:
            continue
        command = ["task", "create", task["title"], "--status", task["status"], "--description", task["description"], "--labels", task["labels"], "--priority", "high", "--notes", task["notes"]]
        for criterion in task["acceptance"]:
            command.extend(["--ac", criterion])
        for reference in task["references"]:
            command.extend(["--ref", reference])
        print(cli(*command).strip())
        created_keys.append(task["key"])
    existing = json.loads(cli("task", "list", "--json"))["tasks"]
    by_title = {task["title"]: task for task in existing}
    by_key = {task["key"]: by_title[task["title"]]["id"] for task in TASKS}
    for task in TASKS:
        if task["key"] not in created_keys:
            continue
        task_id = by_key[task["key"]]
        if task["dependencies"]:
            command = ["task", "edit", task_id]
            for dependency in task["dependencies"]:
                command.extend(["--dep", by_key[dependency]])
            print(cli(*command).strip())
        if task["status"] == "Done":
            command = ["task", "edit", task_id]
            for index in range(1, len(task["acceptance"]) + 1):
                command.extend(["--check-ac", str(index)])
            command.extend(["--final-summary", task["notes"]])
            print(cli(*command).strip())
    (ROOT / "backlog" / "task_index.json").write_text(json.dumps({"cli_version": version, "official_repository": "https://github.com/MrLesk/Backlog.md", "tasks": by_key}, indent=2) + "\n", encoding="utf-8")
    print(cli("task", "list", "--plain"))
    subprocess.run([sys.executable, str(ROOT / "refresh_backlog.py"), "--binary", str(binary)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
