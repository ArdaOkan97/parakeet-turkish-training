"""Build sanitized model cards, attribution and upstream-compatible license."""
import json,os,shutil,urllib.request
from pathlib import Path
R=Path(__file__).parent
OUT=Path(r'E:\gpu_training_scratchpad\exports\parakeet-tdt-0.6b-v3-tr-experimental-400000-20261029-v2')
F=Path(r'E:\gpu_training_scratchpad\analysis\FLEURS_full_vs_v3_20261029_v1')
report=json.loads((F/'report.json').read_text())
datasets=[
 ('ysdede/commonvoice_17_tr_fixed','9d8a025f1ae7c5a2f5c43298c7c4910b03995a4b','Mozilla Common Voice 17 Turkish contributors; ysdede fixed release','CC0-1.0','Turkish training',None),
 ('google/fleurs','70bb2e84b976b7e960aa89f1c648e09c59f894dd','Google and the FLEURS authors/contributors','CC-BY-4.0','Turkish TRAIN; English, Spanish, Italian, French, German and Portuguese TRAIN replay/distillation; validation selection; TEST evaluation only',None),
 ('issai/Turkish_Speech_Corpus','d79ea01392882322e8f46657cb635236af25b6d4','ISSAI / Nazarbayev University; Mussakhojayeva, Dauletbek, Yeshpanov and Varol','CC-BY-4.0 (publisher corpus terms; HF card metadata says MIT)','Turkish training','https://issai.nu.edu.kz/turkic-asr/'),
 ('disco-eth/WorldSpeech','f389569345a04eb8ac8fa9405b0d5048ad74b8b3','DISCO / ETH Zurich WorldSpeech team and Turkish Parliament (TBMM) source speakers','CC-BY-NC-4.0','98,970 accepted Turkish training clips','https://www.tbmm.gov.tr/'),
 ('cloud0day3/alania-speech-captions-tr','8b0fdd976ed9668b64f5ee0696317a8f241ce39c','cloud0day3 / Alania; YODAS authors and original recording uploaders','Caption metadata CC-BY-4.0; derived YODAS transcripts CC-BY-3.0','Turkish broad-phase text/caption descriptors; paired with original YODAS3 audio',None),
 ('espnet/yodas3','81d802f45a1c1e1f5331260484dacf1fbea81981','ESPnet / YODAS team and original Creative Commons video uploaders','CC-BY-3.0','Original Turkish audio referenced by Alania metadata (51 tar shards)',None),
 ('serdarcaglar/kiraat','f8afe95abedc19ec6392c77cf470f5d3dcd2788f','Serdar Çağlar / KIRAAT and all 27 source channels credited below','Attribution requested; public model release permission separately confirmed by the model publisher','Turkish training/rest portions; published DEV and TEST excluded from TRAIN',None),
]
channels=[]
for line in (R/'kiraat_README_RAW.md').read_text(encoding='utf-8').split('## Channel credits',1)[1].split('## Rights',1)[0].splitlines():
    if line.startswith('| ') and ' | ' in line:
        fields=[x.strip() for x in line.split('|')[1:-1]]
        if len(fields)==6 and fields[1].replace(',','').isdigit():channels.append(fields[0])
assert len(channels)==27
attribution={'base_model':{'id':'nvidia/parakeet-tdt-0.6b-v3','revision':'541d1f99c6b0c3cd0b11a95167540bb8edefd82b','license':'CC-BY-4.0','credit':'NVIDIA; see upstream card and technical report for inherited pretraining sources'},'datasets':[dict(zip(['id','revision','credit','terms','use','source_url'],d)) for d in datasets],'KIRAAT_source_channels':channels,'noise':[{'title':'Coffee Shop.wav','creator':'Trainscough5407','url':'https://freesound.org/people/Trainscough5407/sounds/404478/','license':'CC0-1.0'},{'title':'Fan.wav','creator':'Q.K.','url':'https://freesound.org/people/Q.K./sounds/56236/','license':'CC0-1.0'},{'title':'traffic-cars-ambience.wav','creator':'DotAfter','url':'https://freesound.org/people/DotAfter/sounds/610978/','license':'CC0-1.0'}],'room_responses':'16 locally simulated near/far responses in eight rooms using pyroomacoustics; no external RIR dataset','model_license':'CC-BY-NC-4.0'}
table='\n'.join(f'| [{d[0]}](https://huggingface.co/datasets/{d[0]}) | {d[2]} | {d[3]} | {d[4]} |' for d in datasets)
old_table='\n'.join(f"| {x['name']} | {x['utterances']} | {x['v3_wer_percent']:.2f} | {x['adapted_wer_percent']:.2f} | {x['delta_percentage_points']:+.2f} |" for x in report['results'])
common=f'''# Parakeet TDT 0.6B v3 — Turkish experimental, update 400000

An experimental Turkish adaptation of [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), published by [ArdaOkan](https://huggingface.co/ArdaOkan). This release uses the checkpoint with the lowest observed Turkish DEV WER, update **400,000**, selected before the full FLEURS TEST evaluation. The run completed 401,072 updates; the final checkpoint was not the selected best DEV checkpoint.

The adaptation adds Turkish characters and corpus-derived BPE pieces, rank-8 attention LoRA (alpha 16, dropout 0) across all 24 encoder layers, and fully trains the decoder, encoder/decoder projectors and joint output head. The original model supports 25 European languages; this experiment prioritizes Turkish while replaying English, Spanish, Italian, French, German and Portuguese. Other inherited languages have not been evaluated for this release.

## License and attribution

Released under **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)**. Noncommercial use, attribution and indication of modifications are required. WorldSpeech's noncommercial terms are included in this licensing choice. The NVIDIA base is CC BY 4.0; its attribution is retained. This license covers this model distribution, not redistribution rights to the source datasets or recordings. No training audio, text corpus, optimizer state or personal access credentials are included.

Suggested attribution: “Parakeet TDT v3 Turkish experimental by ArdaOkan, adapted from NVIDIA Parakeet TDT 0.6B v3; dataset and source credits are listed in the model card and dataset_attributions.json.” Retain the attribution files when redistributing modified weights.

## Training data and source credits

The following datasets were used in the adaptation's training lineage. Exact revisions and machine-readable credits are in [dataset_attributions.json](dataset_attributions.json).

| Dataset | Credit | Source terms | Use |
|---|---|---|---|
{table}

The expanded cleaner Turkish pool contained 352,768 accepted clips (about 675.10 hours): Common Voice 72,846; Turkish FLEURS 2,037; Turkish Speech Corpus 178,915; WorldSpeech 98,970. The broad pool contained 2,631,115 clips, including Alania/YODAS3 and KIRAAT. The schedule comprised two broad passes and then 25,198 cleaner updates; this selected checkpoint is inside the cleaner phase. Effective batch size was 16 with Turkish training and replay/distillation of the six tracked languages. Validation selected checkpoints; TEST was used for reporting only.

ISSAI's publisher describes its corpus under CC BY 4.0, although the Hugging Face mirror labels its card MIT. See the [publisher](https://issai.nu.edu.kz/turkic-asr/) and [dataset DOI](https://doi.org/10.48342/0xes-sf45). Citation: Mussakhojayeva, Dauletbek, Yeshpanov and Varol, “Multilingual Speech Recognition for Turkic Languages,” Information 14(2), 74 (2023), [doi:10.3390/info14020074](https://doi.org/10.3390/info14020074).

Alania supplies captions/descriptors; original audio came from YODAS3. Credit also belongs to the original recording creators/uploaders identified in those datasets. This adaptation did not use Alania's synthetic-speech dataset. WorldSpeech's Turkish source includes [TBMM](https://www.tbmm.gov.tr/); see [WorldSpeech](https://huggingface.co/datasets/disco-eth/WorldSpeech) for per-source terms and citation.

### KIRAAT source-channel credits

Credit to **Serdar Çağlar / KIRAAT**, its contributors, narrators and the source channels listed in its dataset card. The publisher confirmed separate permission for public release of this trained model. The pinned dataset card requests attribution and does not declare a blanket dataset license; this model release does not relicense its recordings or literary texts.

{', '.join(channels)}.

### Noise, room simulation and software

The weak Turkish student view used three real recorded noise sources, all CC0: [Coffee Shop.wav — Trainscough5407](https://freesound.org/people/Trainscough5407/sounds/404478/), [Fan.wav — Q.K.](https://freesound.org/people/Q.K./sounds/56236/), and [traffic-cars-ambience.wav — DotAfter](https://freesound.org/people/DotAfter/sounds/610978/). Sixteen near/far room responses in eight rooms were locally simulated with pyroomacoustics. Credit to pyroomacoustics and FastMSS for augmentation tooling. Replay/teacher and DEV/TEST audio remained clean. No MUSAN or external room-response corpus was used in this run.

### Inherited NVIDIA pretraining

Credit to NVIDIA's Parakeet and NeMo teams and their upstream data contributors. The base model card describes Granary and NeMo ASR Set 3.0 pretraining; consult the [upstream card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) and [technical report](https://arxiv.org/abs/2509.14128) for that inherited provenance. Those upstream pretraining corpora are distinct from the adaptation datasets above.

## Source-runtime evaluation

Full official FLEURS TEST for all seven tracked languages: **5,620 utterances, about 19.33 audio hours**, at dataset revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`. These numbers measure the original unmerged Hugging Face training runtime before native NeMo export. Both sides used the same complete audio, serial greedy decoding, BF16 encoder/decoder and FP32 joint. They are not mislabeled as measurements of the native merged archive.

| Language | Utterances | Original v3 WER % | Adapted source WER % | Change (pp) |
|---|---:|---:|---:|---:|
{old_table}

The six replay languages' word-weighted WER changed from **4.59% to 5.01% (+0.42 pp)**. Turkish improved substantially, while every tracked replay language regressed modestly. Turkish DEV was **21.09% (174/825 words)** at selection. DEV and full TEST use different sets; their WER values are not interchangeable. Full results and paired bootstrap intervals are in [source_runtime_fleurs_results.json](source_runtime_fleurs_results.json).

WER normalization: NFC, Turkish-aware lowercase, remove Unicode punctuation, collapse whitespace and preserve accents. This is the project's normalizer, not a claim of exact equivalence to the original FLEURS evaluation FST. WER is corpus word edits divided by reference words, so it can exceed 100%.

## Native conversion details

The native tokenizer has 9,222 lexical pieces and blank ID 9,222. The original HF checkpoint has blank at 8,192, so embedding and joint rows were permuted; the five TDT duration IDs remain 9,223–9,227. Output piece strings are retained. The SentencePiece encoding algorithm is not claimed to reproduce the original HF training tokenizer's segmentation exactly. This is an inference export, not an exact optimizer/sampler resume checkpoint.

The learned HF start embedding is represented with NeMo's ordinary zero-start decoder by shifting all embeddings by the learned start vector and compensating the first LSTM input bias. All 9,223 first-layer gate inputs were checked (maximum FP32 absolute difference 2.29e-5). The 294 frozen base matrices retain the BF16-rounded values used in training. The merged archive materializes the rank-8 updates in FP32. Floating-point operation order and the native frontend/decoder can change transcripts; native measurements are reported separately.

## Intended use and limitations

Experimental noncommercial Turkish speech recognition and multilingual retention research. Tested tracked languages: tr, en, es, it, fr, de, pt. No claim is made about untested languages, dialect coverage, safety-critical transcription, speaker identification or voice cloning. Recognition may omit, substitute or invent words, especially under domain shifts. Check transcripts against the recording where accuracy matters. No per-speaker or demographic evaluation was performed.
'''
for kind in ['merged','adapter']:
    folder=OUT/kind
    yaml='---\nlanguage:\n- tr\n- en\n- es\n- it\n- fr\n- de\n- pt\nlicense: cc-by-nc-4.0\nlibrary_name: nemo\npipeline_tag: automatic-speech-recognition\nbase_model: nvidia/parakeet-tdt-0.6b-v3\nbase_model_relation: finetune\ndatasets:\n'+''.join('- '+d[0]+'\n' for d in datasets)+'tags:\n- speech-recognition\n- parakeet\n- tdt\n- turkish\n- experimental\n'+('- lora\n' if kind=='adapter' else '')+'---\n\n'
    (folder/'README.md').write_text(yaml+common,encoding='utf-8')
    (folder/'dataset_attributions.json').write_text(json.dumps(attribution,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (folder/'source_runtime_fleurs_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (folder/'NOTICE').write_text('Parakeet TDT v3 Turkish experimental, update 400000\nAdaptation copyright 2026 ArdaOkan.\nBased on NVIDIA Parakeet TDT 0.6B v3, licensed CC BY 4.0.\nThis distribution is CC BY-NC 4.0. Retain README.md and dataset_attributions.json for data, recording and upstream credits.\nChanges: Turkish vocabulary extension; trained decoder/joint/projectors; rank-8 attention adaptation; native row/SOS conversion; merged variant materializes LoRA.\n',encoding='utf-8')
    license_bytes=urllib.request.urlopen('https://raw.githubusercontent.com/creativecommons/creativecommons.org/master/docroot/legalcode/by-nc_4.0.txt',timeout=20).read(100000)
    assert b'Attribution-NonCommercial 4.0 International' in license_bytes
    (folder/'LICENSE').write_bytes(license_bytes)
print(json.dumps({'prepared':['merged','adapter'],'all_dataset_sources':len(datasets),'KIRAAT_channels':len(channels)}))
