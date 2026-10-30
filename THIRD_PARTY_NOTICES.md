# Third-party notices

Project-owned code is offered under Apache License 2.0. Existing upstream copyright/license headers remain authoritative for the corresponding files.

- `project/outputs/parakeet_turkish/tdt_tile_native_gradient_v1.py` preserves NVIDIA (2021) and Mingkun Huang (2018–2019) Apache-2.0 notices.
- `project/work/research/configuration_parakeet.py`, `modeling_parakeet.py` and `processing_parakeet.py` preserve Hugging Face (2025) Apache-2.0 notices.
- Augmentation preparation references the external [FastMSS](https://github.com/popcornell/FastMSS) project by popcornell, pinned at `e481ceac821ef0cb31033bf43b45922e377f528d`. FastMSS is GPL-3.0; its source is **not vendored** in this repository. The original external `rirsimulator.py` source binding is SHA-256 `05f7ef9c12ede249401ca26335b083188e587cb40fff372e489820b820e9455c`. Obtain and use that dependency under its own [license](https://github.com/popcornell/FastMSS/blob/e481ceac821ef0cb31033bf43b45922e377f528d/LICENSE).
- Python dependencies, NVIDIA NeMo, PyTorch, Transformers, pyroomacoustics and other external packages retain their own licenses. No dependency is relicensed by this release.
- NVIDIA base-model weights are CC-BY-4.0. This project's published model weights are CC-BY-NC-4.0; neither is covered by this repository's code license.
- FLEURS example text and derived evaluation reports credit Google/FLEURS under CC-BY-4.0. See `docs/DATASETS.md` and `docs/dataset_attributions.json` for every project dataset and audio-source credit.
