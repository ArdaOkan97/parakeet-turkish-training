# Production and pipeline entry points

The following are the deployed final training components. Hashes refer to the exact captured bytes.

- **Production training driver**: [r9_recovery_driver_v43.py](../analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1/next212000_failure_repair_v1/r9_recovery_driver_v43.py) — SHA-256 `43d2d49fecf288cede18458402444546381f388c1c11710f6d6c0e7823d14430`.
- **Production owner / normal continuations**: [r9_recovery_owner_v46.py](../analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1/next212000_failure_repair_v1/r9_recovery_owner_v46.py) — SHA-256 `6f55e946378b7ee59e64d1cf81cda62c40a38f95b4f1fe61d7f4c26e67bc8c64`.
- **Production checkpoint retention guardian**: [retention_guardian.py](../analysis/BPE9223_training_launch_v1/best_checkpoint_retention_fix_v1/retention_guardian.py) — SHA-256 `5d59647b718c23a9ba689c4d376bae2d81a5dd44daadaf101d839d83c64a4876`.
- **Production Trackio reader**: [serve_BPE9223_Trackio_live_v9.py](../analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1/next212000_failure_repair_v1/ROOT_HUMAN_Reader9_repair_1755_v1/serve_BPE9223_Trackio_live_v9.py) — SHA-256 `5c6050717d805a022f1900acc0022323f49150e1547dc861d95cd13853e2390d`.

The native `r9_recovery_driver_v44_best_DEV_retention_candidate.py` is an **inactive candidate**, not the deployed driver. Current-run DEV protection was supplied by the separate retention guardian. The guardian archives every future scheduled DEV and final checkpoint outside native pruning, while tracking best observed and best retained DEV separately.

## Data preparation

Preparation modules are primarily under [`project/work/research/`](../project/work/research/). Representative entry points, including historical versions, are:

- `prepare_alania_yodas_compressed_v1.py`, `prepare_alania_stratified_QC_v1.py`, `index_BPE_YODAS_archives_v*.py`: Alania/YODAS compressed acquisition, quality control and indexing.
- `prepare_BPE_broad_labels_v*.py`, `audit_BPE_label_alignment_v2.py`: broad-pool label preparation and alignment validation.
- `index_worldspeech_TRAIN_candidate_v1.py`, `project_worldspeech_TRAIN_metadata_v1.py`, `prepare_worldspeech_TRAIN_audio_candidate_v1.py`, `stage_worldspeech_TRAIN_candidate_v1.py`: WorldSpeech training preparation.
- `acquire_ASR18_real_recorded_noise_assets_v1.py`, `prepare_fastmss_RIR_bank_v1.py`, `generate_FastMSS_diverse_hybrid_rooms_v1.py`: recorded noise assets and simulated room-response banks.

The low-level broad/cleaner samplers, training schedule, transactions and optimizer logic are in [`project/outputs/parakeet_turkish/`](../project/outputs/parakeet_turkish/), including the `bpe_cleaner_data_v1`, `bpe_broad_transaction_v1`, `bpe_broad_full207_optimizer_v1` and `bpe_training_schedule_v1` modules. Label/BPE audits are also archived under [`analysis/Turkish_labels_BPE_circumflex_v1/`](../analysis/Turkish_labels_BPE_circumflex_v1/).

## Evaluation and export

- Successful full FLEURS runner: [`evaluate_full_v3.py`](../analysis/FLEURS_full_vs_v3_20261029_v1/evaluate_full_v3.py). Earlier variants are historical attempts.
- Native export: [`export_nemo_v2.py`](../analysis/parakeet_v3_tr_experimental_publish_20261029_v1/export_nemo_v2.py), followed by [`finalize_native_archive_v3.py`](../analysis/parakeet_v3_tr_experimental_publish_20261029_v1/finalize_native_archive_v3.py).
- Published inference helpers: [`load_adapter.py`](../inference/load_adapter.py), [`memory_efficient_restore.py`](../inference/memory_efficient_restore.py).

Run-bound scripts may import modules elsewhere in this archive and require generated metadata not distributed here. Version numbers alone do not identify an approved production script; use the explicitly identified components above and inspect their dependencies.
