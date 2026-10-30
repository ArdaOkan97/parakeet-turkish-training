# Full FLEURS TEST comparison

Checkpoint 400,000 was selected on Turkish DEV before this test run. Both models were evaluated on the complete official TEST splits for the seven tracked languages, **5,620 utterances** total. These are original training/source-runtime results (BF16 GPU, FP32 joint), not a full native NeMo TEST rerun.

| Language | Utterances | Base v3 WER | Adapted WER | Difference (percentage points) |
|---|---:|---:|---:|---:|
| Turkish | 743 | 100.96% | 18.21% | -82.75 |
| English | 647 | 5.65% | 5.93% | +0.27 |
| Spanish | 908 | 3.72% | 4.10% | +0.38 |
| Italian | 865 | 3.14% | 3.48% | +0.34 |
| French | 676 | 5.73% | 6.24% | +0.51 |
| German | 862 | 4.95% | 5.59% | +0.65 |
| Portuguese | 919 | 5.04% | 5.38% | +0.35 |

Normalization: NFC; Turkish-aware lowercase; delete Unicode punctuation; collapse whitespace; preserve accents; CER includes spaces. Numbers written as words and digits remain distinct. WER is corpus word errors divided by reference words, not the mean of utterance WERs. Positive differences indicate regression. The base is not a Turkish-supported model; its Turkish WER can exceed 100% through insertions.

[Full counts, revisions and sentence-cluster bootstrap confidence intervals](source_runtime_fleurs_results.json) · [Selected Turkish transcripts](../examples/FLEURS_TEST.md) · [Native export checks](native_export_validation.json).

Dataset: [Google FLEURS](https://huggingface.co/datasets/google/fleurs), revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, CC-BY-4.0.
