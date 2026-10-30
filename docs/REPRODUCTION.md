# Reproduction and environment

## What this archive preserves

The code captures the implemented data/training pipeline and its research history, plus the scripts used for full FLEURS evaluation and NeMo export. It excludes datasets, audio, model tensors, virtual environments, caches, databases, training journals, private credentials and machine-bound runtime receipts. Lightweight source/configuration files are included where captured. The source manifest records the original relative paths and exact SHA-256 values.

Many files contain absolute Windows C:/E: paths and hash-bound imports. Relative layout here mirrors the original project and selected analysis directories, but copying it alone does not satisfy those bindings. Source hashes, generated manifests and process-bound identities must be reviewed and regenerated for a new setup. Do not bypass guards or treat an archived admission/recovery helper as authorization for a new launch.

## Preparing a fresh run

1. Obtain the base model and each dataset at the revisions in `dataset_attributions.json`, following their individual access and license terms. Acquire KIRAAT permission independently; the author's permission to publish this model does not transfer dataset redistribution rights.
2. Provision local storage, CPU/GPU memory budgets and a compatible Python/PyTorch environment before adapting paths. The original run required all new datasets, downloads, caches, scratch and outputs on `E:/gpu_training_scratchpad`. Its `use_storage.ps1` was process-local; running-process environments were not modified.
3. Read the data-preparation modules and their generated-index contracts before executing them. Recreate TRAIN-only preparation, replay splits, tokenizer assets, augmentation banks and validation references. DEV/TEST examples must remain outside training and teacher/replay preparation.
4. Use the final driver/owner identified in `ENTRYPOINTS.md` to trace the complete training dependency graph. Preserve full model/Adam/RNG/sampler/schedule state and verify the next draw on a new exact continuation. Do not reuse historical PID identities or launch receipts.
5. Run data-integrity, sampler, checkpoint and small inference validations appropriate to the adapted environment before a full run. Maintain the original resource bounds or explicitly review any new resource plan; this source release does not authorize relaxing them.
6. Evaluate the held-out complete official FLEURS TEST split only after selecting a checkpoint on DEV. Compare the adapted model and base with identical frontend, decoding and normalization. Report corpus word-error counts rather than averaging utterance WERs.

## Dependencies

Original training/evaluation uses Python, PyTorch, Transformers, Hugging Face datasets/hub, safetensors, SentencePiece/tokenizers, NumPy, audio libraries, psutil, Trackio and custom kernels/modules in this archive. Augmentation preparation also uses pyroomacoustics and the external FastMSS repository. This archive does **not** provide a fully locked portable training environment: the exact environment and generated runtime metadata are not all distributed.

Native inference was validated with NVIDIA NeMo **3.0.0** on CPU FP32. `inference/requirements.txt` is the published adapter inference requirement set, not a training lockfile. The NeMo export is an inference artifact; it does not include the original optimizer/sampler training state.

## Native/source runtime comparison

The source-runtime DEV result is 174/825 errors (21.09%); native CPU FP32 is 182/825 (22.06%) on the same 50 utterances and normalized references. Eight normalized transcripts changed: five worsened and three changed without changing error counts. The two paths change runtime, precision, frontend/decoder implementation and export representation together, so an exclusive cause has not been isolated. Full TEST results in this repository belong to the original source runtime. Merged and adapter native validation outputs matched on all 70 TEST smoke examples; this is not a native full-TEST claim.

## Archive validation

The four identified production entry points passed syntax and exact-source SHA-256 checks. Six historical files are non-parsing fragments or failed attempts; their paths and diagnostics are listed in [source_validation.json](../reports/source_validation.json). They were retained byte-for-byte for provenance, not repaired or presented as runnable final components. No training/inference jobs were launched for GitHub publication.
