# Parakeet Turkish training pipeline


Research source archive for **Parakeet TDT 0.6B v3 Turkish experimental**: data preparation, Turkish vocabulary expansion, LoRA training, multilingual replay, audio augmentation, checkpoint retention, evaluation, and native NVIDIA NeMo export.


- [Merged NeMo model](https://huggingface.co/ArdaOkan/parakeet-tdt-0.6b-v3-tr-experimental)
- [LoRA + trained heads](https://huggingface.co/ArdaOkan/parakeet-tdt-0.6b-v3-tr-experimental-lora)
- [Production entry points](docs/ENTRYPOINTS.md) — start here to distinguish deployed code from historical candidates.
- [Data sources and credits](docs/DATASETS.md) · [machine-readable attribution](docs/dataset_attributions.json)
- [Complete FLEURS comparison](reports/FLEURS.md)
- [Reproduction and environment notes](docs/REPRODUCTION.md)


## Contents


| Directory | Contents |
|---|---|
| `project/outputs/parakeet_turkish/` | Training, tokenization, LoRA, optimizer, sampler, checkpoint and resource-guard modules |
| `project/work/research/` | Dataset acquisition, indexing, label preparation, augmentation and experimental utilities |
| `project/work/` | Local preparation and orchestration scripts |
| `analysis/` | Run-specific training/recovery, label audits, retention, FLEURS evaluation and NeMo export scripts |
| `inference/` | Published adapter loader and memory-efficient NeMo restore helper |
| `reports/` | Aggregate evaluation and export validation |


This release contains **3,596 captured project/analysis source files**, plus documentation and published inference helpers. Historical variants and validation utilities are included for provenance. Their presence does not imply they were all used in the final run or that they succeeded. The exact captured paths, sizes and SHA-256 values are listed in [source_manifest.json](source_manifest.json).


## Training recipe


The base is [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), revision `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`. The project expands the Turkish vocabulary and trains rank-8 attention LoRA (alpha 16, dropout 0), decoder, projectors and joint heads. Effective batches contain 14 Turkish examples and 2 multilingual replay examples. Replay tracks English, Spanish, Italian, French, German and Portuguese with teacher distillation. Three real recorded noises and 16 simulated room responses augment the Turkish student branch; DEV/TEST remain clean.


The completed schedule is **401,072 updates**: two broad passes of 187,937 updates followed by 25,198 cleaner-label updates starting at update 375,875. The broad pool has 2,631,115 clips; the cleaner pool has 352,768 clips (about 675.10 hours). The published model is checkpoint **400,000**, selected by Turkish DEV WER before full FLEURS TEST evaluation.


The original training runtime gives **21.09% Turkish DEV WER** (174/825 words) and **18.21% Turkish FLEURS TEST WER** (2,341/12,856 words, 743 utterances). The native NeMo export gives **22.06%** on the same 50-example DEV check (182/825). These are separate runtime measurements; the native full TEST set was not rerun. See the reports and model cards for details.


## Using the published model


The merged `.nemo` artifact uses the standard `EncDecRNNTBPEModel` interface. CPU restore/transcription was validated with NeMo 3.0.0. Obtain weights and complete loading instructions from the model card. This GitHub repository contains code and small reports, without model weights or training data.


```python
from nemo.collections.asr.models import ASRModel


model = ASRModel.from_pretrained(
    "ArdaOkan/parakeet-tdt-0.6b-v3-tr-experimental", map_location="cpu"
)
model.eval()
result = model.transcribe(["audio.wav"], batch_size=1)
print(result[0].text)
```


On hosts with limited RAM, follow the memory-efficient restore instructions in the model card. GPU export-validation attempts exceeded the original host-memory cap, so this release does not claim completed native GPU validation.


## Reproduction scope


The captured scripts retain their original bytes, including Windows paths, source hashes and run-specific guards. They depend on staged datasets, generated indices, model assets and process-bound review/receipt files that are **not included**. This is a research source archive, not a turnkey installer or a complete resumable training state. Consult [REPRODUCTION.md](docs/REPRODUCTION.md) before adapting a new run. Do not execute historical launch/recovery scripts against a live run.


## License


Project-owned source is released under [Apache-2.0](LICENSE), subject to preserved upstream notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Dataset licenses and the external FastMSS GPL-3.0 dependency remain separate. The published model weights use **CC-BY-NC-4.0**, reflecting WorldSpeech's noncommercial terms. The code license does not grant rights to datasets or model weights.

