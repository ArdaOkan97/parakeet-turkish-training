"""One authorized TRAIN-only native Whisper long-form GPU API compatibility smoke."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration
import benchmark_strong_tr_test_v2 as worker


def main(args):
    started = time.perf_counter()
    model_path = worker.inside(args.model_path)
    manifest = worker.inside(args.train_manifest)
    require = worker.require
    records = worker.rows(manifest)
    require(records and all(r["dataset_split"] == "train" and r["language"] == "tr" for r in records), "Smoke must use TRAIN-only Turkish audio")
    selected = sorted(records, key=lambda r:(-r["duration"],r["source_id"]))[:2]
    signals=[];sources=[]
    for record in selected:
        path = worker.inside(record["audio_filepath"])
        info = sf.info(path)
        require(worker.sha256(path) == record["wav_sha256"] and info.samplerate == 16000 and info.channels == 1
            and info.subtype == "PCM_16" and abs(info.frames/16000-record["duration"]) <= 1/16000, "TRAIN source/audio metadata changed")
        signal,sr=sf.read(path,dtype="float32")
        signals.append(signal)
        sources.append({"source_id":record["source_id"],"sentence_id":record["sentence_id"],"dataset_split":"train",
            "audio_filepath":str(path),"wav_sha256":record["wav_sha256"],"audio_sha256":record["audio_sha256"],
            "pcm_samples":info.frames,"duration":info.frames/16000})
    signal=np.concatenate(signals)
    require(len(signal) > worker.SHORT_SAMPLES, "TRAIN concatenation does not require long form")
    wave_path=worker.WORKSPACE/"work/strong_longform_train_smoke_v2.wav"
    sf.write(wave_path,signal,16000,subtype="PCM_16")
    reread,sr=sf.read(wave_path,dtype="float32")
    require(np.array_equal(reread,signal), "Concatenation changed source PCM samples")
    assets=worker.verified_assets(model_path)
    code=worker.protected_code()
    processor=AutoProcessor.from_pretrained(str(model_path),local_files_only=True)
    features,metadata=worker.prepare_features(processor,signal,16000)
    require(metadata["inference_mode"]=="long_form", "Smoke did not route to native long form")
    device=torch.device("cuda")
    require(torch.cuda.is_available(),"Authorized compatibility smoke requires GPU")
    torch.manual_seed(20261010);np.random.seed(20261010)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats(device)
    model=WhisperForConditionalGeneration.from_pretrained(str(model_path),local_files_only=True,
        use_safetensors=True,dtype=torch.float16,attn_implementation="sdpa").to(device).eval()
    inputs={k:v.to(device=device,dtype=torch.float16 if k=="input_features" else v.dtype) for k,v in features.items()}
    began=datetime.now(timezone.utc).isoformat()
    with torch.inference_mode(),worker.LongformTrace(model) as trace:
        generated=model.generate(**inputs,**worker.LONG_PROTOCOL)
    torch.cuda.synchronize()
    tokens,evidence=worker.unpack_longform(generated,trace,metadata["input_mel_frames"])
    transcript=processor.batch_decode([tokens],skip_special_tokens=True)[0].strip()
    report={"schema":1,"passed":True,"gpu_used":True,"test_ASR_inference_performed":False,
        "purpose":"TRAIN-only long-form API/protocol compatibility, no WER claim or checkpoint selection",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),"inference_started_at_utc":began,
        "repository":worker.REPOSITORY,"revision":worker.REVISION,"protocol":worker.LONG_PROTOCOL,
        "source_manifest_path":str(manifest),"source_manifest_sha256":worker.sha256(manifest),"sources":sources,
        "waveform_construction":"deterministic direct concatenation of two longest TRAIN clips; no added silence/resampling/truncation",
        "concatenated_wav_path":str(wave_path),"concatenated_wav_sha256":worker.sha256(wave_path),
        "concatenated_float32_pcm_sha256":hashlib.sha256(signal.tobytes()).hexdigest(),
        "actual_duration_seconds":len(signal)/16000,**metadata,"longform":evidence,"generated_token_ids":tokens,
        "hypothesis":transcript,"assets":assets,"protected_code":code,
        "worker_script_sha256":worker.sha256(Path(worker.__file__)),"smoke_script_sha256":worker.sha256(Path(__file__)),
        "weight_compute_dtype":"torch.float16","attention_implementation":"sdpa","device_name":torch.cuda.get_device_name(),
        "torch_version":torch.__version__,"transformers_version":__import__("transformers").__version__,
        "peak_allocated_vram_gb":torch.cuda.max_memory_allocated()/1e9,"elapsed_seconds":time.perf_counter()-started}
    worker.write_json(args.output,report)
    del model,processor,inputs,features,generated
    gc.collect();torch.cuda.empty_cache()
    print(json.dumps({"passed":True,"gpu_released":True,"train_only":True,"duration_seconds":len(signal)/16000,
        "windows":len(evidence["windows"]),"report_path":str(args.output)}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path",type=Path,default=worker.WORKSPACE/"work/models/whisper-large-v3")
    p.add_argument("--train-manifest",type=Path,default=worker.WORKSPACE/"work/data/tr_fleurs_train.jsonl")
    p.add_argument("--output",type=Path,default=worker.OUTPUT/"strong_longform_train_gpu_smoke_v2.json")
    main(p.parse_args())
