import json
from pathlib import Path
R=Path(__file__).parent
OUT=Path(r'E:\gpu_training_scratchpad\exports\parakeet-tdt-0.6b-v3-tr-experimental-400000-20261029-v3')
native=json.loads((R/'native_validation_v9/complete.json').read_text())
adapter=json.loads((R/'adapter_validation_v1/complete.json').read_text())
dev=json.loads((R/'native_validation_v9/Turkish_DEV.json').read_text())['score']
assert native['passed'] and native['all_test_utterances']==70 and adapter['passed']
assert adapter['merged_tensor_hash_mismatches']==0
assert adapter['matched_TEST_clips']==70
result={'passed':True,'checkpoint_update':400000,'native_class':'nemo.collections.asr.models.EncDecRNNTBPEModel','native_archive_sha256':'5c4f51ee50ea19bb598c4e8024c6741cf8513d57d55c5cae01287ae335d4ae25','native_validation_runtime':'NeMo3.0.0 / PyTorch2.7.1, CPU FP32, batch1, ordinary greedy_batch; mmap used only for low temporary commitment','native_Turkish_DEV':{k:v for k,v in dev.items() if k!='details'},'adapter_validation':adapter,'native_TEST_smoke_count':70,'source_runtime_full_FLEURS_count':5620,'GPU_validation_completed':False,'GPU_limitation':'Host private-memory8GB guard stopped GPU attempts; limits were not raised','archive_tensor_bytes_unchanged_by_portability_config_fix':True}
(R/'PUBLIC_RELEASE_VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
section=f'''## Native export validation

The **final published archive** restored as the standard `EncDecRNNTBPEModel` in NeMo 3.0.0 / PyTorch 2.7.1. CPU FP32 validation processed all 50 Turkish DEV clips and 70 matched TEST smoke clips (10 per tracked language), using full audio and saved greedy decoding. Native merged Turkish DEV WER: **{dev['wer']*100:.2f}% ({dev['word_errors']}/{dev['reference_words']} words)**. The unmerged native adapter loader was tested on the same sets: **{adapter['Turkish_DEV']['wer']*100:.2f}% DEV WER**; **{adapter['identical_TEST_transcripts']}/70 TEST smoke transcripts** matched the merged output exactly.

All **{adapter['merged_tensor_hash_matches']} selected native attention/head tensors** matched exact SHA256 fingerprints when the unmerged adapter updates were materialized in FP32 and compared with the restored merged archive. The native validation used memory mapping only to reduce temporary loading commitment. The stored weights are FP32; tokenizer assets are embedded and local training paths have been removed. GPU attempts hit the release machine's 8 GB host-private-memory guard; those partial attempts are not presented as completed GPU validation. Native full-TEST WER was not measured; the full FLEURS table below remains explicitly source-runtime results. See [export_validation.json](export_validation.json).

'''
for kind in ['merged','adapter']:
    folder=OUT/kind;p=folder/'README.md';text=p.read_text(encoding='utf-8')
    text=text.replace('model = ASRModel.from_pretrained("ArdaOkan/parakeet-tdt-0.6b-v3-tr-experimental")','model = ASRModel.from_pretrained("ArdaOkan/parakeet-tdt-0.6b-v3-tr-experimental", map_location="cpu")')
    text=text.replace('uses a memory-mapped extracted cache and BF16 matrix placement','uses a memory-mapped extracted cache')
    text=text.replace('For that mixed-precision path:', 'For the validated CPU FP32 path:')
    text=text.replace('model = restore_model(path, cache_dir="./nemo_cache", device="cuda")\nwith torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):','model = restore_model(path, cache_dir="./nemo_cache", device="cpu")\nwith torch.inference_mode():')
    text=text.replace('model = loader.load_model(folder, device="cuda", cache_dir="./model_cache",\n                          matrix_dtype=torch.bfloat16)\nwith torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):','model = loader.load_model(folder, device="cpu", cache_dir="./model_cache")\nwith torch.inference_mode():')
    text=text.replace('With enough memory, omit `matrix_dtype` for FP32 parameters/execution.', 'The example uses the validated CPU FP32 path. GPU placement is available with `device="cuda"` when host/device memory permits; `matrix_dtype=torch.bfloat16` requires autocast and has not completed validation on the release machine.')
    assert '## Native export validation' not in text
    text=text.replace('## Source-runtime evaluation',section+'## Source-runtime evaluation',1)
    p.write_text(text,encoding='utf-8')
    (folder/'export_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'native_DEV_WER':dev['wer'],'adapter_DEV_WER':adapter['Turkish_DEV']['wer'],'exact_smoke_transcripts':adapter['identical_TEST_transcripts']}))
