# Data sources and attribution

The exact revisions, roles, upstream credits, KIRAAT channel names and noise-source details are preserved in [dataset_attributions.json](dataset_attributions.json). Source code publication does not redistribute the underlying datasets.

| Dataset | Revision | Terms / credit |
|---|---|---|
| [ysdede/commonvoice_17_tr_fixed](https://huggingface.co/datasets/ysdede/commonvoice_17_tr_fixed/tree/9d8a025f1ae7c5a2f5c43298c7c4910b03995a4b) | `9d8a025f1ae7c5a2f5c43298c7c4910b03995a4b` | Mozilla Common Voice 17 Turkish contributors; ysdede fixed release; CC0-1.0 |
| [google/fleurs](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd) | `70bb2e84b976b7e960aa89f1c648e09c59f894dd` | Google and the FLEURS authors/contributors; CC-BY-4.0 |
| [issai/Turkish_Speech_Corpus](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus/tree/d79ea01392882322e8f46657cb635236af25b6d4) | `d79ea01392882322e8f46657cb635236af25b6d4` | ISSAI / Nazarbayev University; Mussakhojayeva, Dauletbek, Yeshpanov and Varol; CC-BY-4.0 (publisher corpus terms; HF card metadata says MIT) |
| [disco-eth/WorldSpeech](https://huggingface.co/datasets/disco-eth/WorldSpeech/tree/f389569345a04eb8ac8fa9405b0d5048ad74b8b3) | `f389569345a04eb8ac8fa9405b0d5048ad74b8b3` | DISCO / ETH Zurich WorldSpeech team and Turkish Parliament (TBMM) source speakers; CC-BY-NC-4.0 |
| [cloud0day3/alania-speech-captions-tr](https://huggingface.co/datasets/cloud0day3/alania-speech-captions-tr/tree/8b0fdd976ed9668b64f5ee0696317a8f241ce39c) | `8b0fdd976ed9668b64f5ee0696317a8f241ce39c` | cloud0day3 / Alania; YODAS authors and original recording uploaders; Caption metadata CC-BY-4.0; derived YODAS transcripts CC-BY-3.0 |
| [espnet/yodas3](https://huggingface.co/datasets/espnet/yodas3/tree/81d802f45a1c1e1f5331260484dacf1fbea81981) | `81d802f45a1c1e1f5331260484dacf1fbea81981` | ESPnet / YODAS team and original Creative Commons video uploaders; CC-BY-3.0 |
| [serdarcaglar/kiraat](https://huggingface.co/datasets/serdarcaglar/kiraat/tree/f8afe95abedc19ec6392c77cf470f5d3dcd2788f) | `f8afe95abedc19ec6392c77cf470f5d3dcd2788f` | Serdar Çağlar / KIRAAT and all 27 source channels credited below; Attribution requested; public model release permission separately confirmed by the model publisher |

KIRAAT was used for training after excluding published DEV/TEST. The model author confirmed separate permission for public release of models trained on it. No blanket dataset-redistribution license is asserted here; retain the publisher/channel attribution and obtain any needed access permissions.

WorldSpeech includes TBMM parliamentary speech and has noncommercial terms; the published model weights are CC-BY-NC-4.0. Alania captions derive from YODAS material, so both sources are credited. The complete attribution JSON also identifies the three CC0 recorded-noise authors/assets. The 16 training room responses were locally simulated; they are not redistributed here.

The base model is NVIDIA's `nvidia/parakeet-tdt-0.6b-v3`, revision `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`, under CC-BY-4.0. Consult its card for inherited pretraining sources. The original model cards contain the full dataset credits and citations alongside the released model.
