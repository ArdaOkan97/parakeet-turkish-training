"""Pinned Whisper greedy Turkish TEST comparison after a frozen DEV winner.

The verified DEV baseline script is read-only. This separate entry point binds
exactly100 existing FLEURS Turkish TEST clips and never performs model selection.
Use --compare-only to recompute the paired comparison on CPU without inference.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration

from benchmark_strong_tr import REPOSITORY, REVISION, sha256, rows, write_json
from metrics import normalize_text, score_transcriptions, paired_bootstrap_delta

OUTPUT = Path(__file__).resolve().parent
WORKSPACE = OUTPUT.parent.parent
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
PROTOCOL = {"language":"tr", "task":"transcribe", "return_timestamps":False,
            "return_token_timestamps":False, "return_dict_in_generate":True,
            "num_beams":1, "do_sample":False, "temperature":0.0,
            "condition_on_prev_tokens":False, "max_new_tokens":440,
            "compression_ratio_threshold":None, "logprob_threshold":None,
            "no_speech_threshold":None}
PAIR_KEYS = ("source_id", "sentence_id", "language", "reference",
             "dataset_revision", "audio_sha256", "wav_sha256", "duration")
EXPECTED_PREFIX = [50258, 50268, 50360, 50364]
EXPECTED_EOS = 50257


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def inside(path, root=WORKSPACE):
    path=Path(path).resolve()
    require(path.is_relative_to(Path(root).resolve()), "Artifact path leaves the workspace")
    return path


def verified_frozen_selection(path, selected_run_id=None):
    path=inside(path)
    frozen=read(path)
    run_id=frozen.get("run_id")
    require(isinstance(run_id,str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}",run_id),
            "Frozen selection needs a safe winner run ID")
    require(selected_run_id is None or run_id==selected_run_id, "Supplied run ID differs from frozen winner")
    require(frozen.get("selection_split")=="FLEURS validation only", "Winner was not selected on development only")
    require(isinstance(frozen.get("selected_step"),int) and not isinstance(frozen["selected_step"],bool)
            and frozen["selected_step"]>=0, "Invalid frozen selected step")
    require(isinstance(frozen.get("wer"),(int,float)) and math.isfinite(frozen["wer"]) and frozen["wer"]>=0,
            "Frozen development WER is not finite")
    stamp=datetime.fromisoformat(frozen.get("frozen_at_utc", ""))
    require(stamp.tzinfo is not None and stamp<=datetime.now(timezone.utc), "Frozen selection timestamp is invalid")
    provenance=frozen.get("provenance",{})
    run_dir=OUTPUT/"runs"/run_id
    training_selection=run_dir/"training_selection.json"
    if not training_selection.exists():
        training_selection=run_dir/"selection.json"
    for key, artifact in (("selection_sha256",training_selection),
                          ("run_config_sha256",run_dir/"run_config.json"),
                          ("history_sha256",run_dir/"training_history.json"),
                          ("weights_sha256",inside(frozen["weights"]))):
        require(provenance.get(key)==sha256(artifact), f"Frozen winner provenance changed: {key}")
    require(provenance.get("selected_step")==frozen["selected_step"], "Frozen provenance step differs")
    selected=read(training_selection)
    require(selected.get("accepted") is True and selected.get("run_id")==run_id
            and selected.get("selected_step")==frozen["selected_step"]
            and selected.get("turkish_validation_wer")==frozen["wer"]
            and selected.get("selection_split")=="FLEURS validation only", "Frozen winner differs from its development selection")
    final=read(run_dir/"selection.json")
    require(final.get("accepted") is True and final.get("merged") is True
            and final.get("heldout_tests_evaluated") is True and final.get("run_id")==run_id
            and final.get("selected_step")==frozen["selected_step"],
            "Winner must finish merged-export finalization before strong TEST inference")
    marker_path=run_dir/"adapted_results_path.json"
    marker=read(marker_path)
    summary_path=inside(marker["summary"],OUTPUT/"results")
    require(summary_path.name=="summary.json" and summary_path.parent.name==marker["tag"]
            and marker["tag"]==final.get("final_results_tag"), "Selected final result marker differs")
    model_provenance=read(run_dir/"finalization_provenance.json")
    require(model_provenance.get("run_id")==run_id
            and model_provenance.get("selected_step")==frozen["selected_step"]
            and model_provenance.get("selected_state_sha256")==provenance["weights_sha256"],
            "Selected export provenance differs from frozen trainables")
    return frozen, {"path":str(path), "sha256":sha256(path), "frozen_at_utc":frozen["frozen_at_utc"],
                    "run_id":run_id, "selected_step":frozen["selected_step"],
                    "selected_results_path":str(summary_path.parent),
                    "selected_results_marker_sha256":sha256(marker_path),
                    "selected_final_selection_sha256":sha256(run_dir/"selection.json"),
                    "selected_finalization_provenance_sha256":sha256(run_dir/"finalization_provenance.json")}


def verified_test_rows(manifest):
    manifest=inside(manifest)
    require(manifest==(WORKSPACE/"work/data/test_tr.jsonl").resolve(), "Use the existing fixed Turkish TEST manifest")
    examples=rows(manifest)
    base_path=OUTPUT/"results/base_v3/test_tr.jsonl"
    original=rows(base_path)
    require(len(examples)==len(original)==100, "Expected exactly100 fixed paired Turkish TEST recordings")
    require(len({r["source_id"] for r in examples})==100
            and len({r["source_id"] for r in original})==100, "Duplicate TEST source IDs")
    require([r["source_id"] for r in examples]==[r["source_id"] for r in original], "TEST ordering differs from original benchmark")
    for record,base in zip(examples,original):
        require(record["language"]=="tr" and record["dataset"]=="google/fleurs"
                and record["dataset_config"]=="tr_tr" and record["dataset_split"]=="test"
                and record["dataset_revision"]==FLEURS_REVISION, "Unexpected TEST dataset/language/split/revision")
        require(all(record[key]==base[key] for key in PAIR_KEYS), "TEST reference/source/audio pairing differs")
        require(record["reference_normalized"]==normalize_text(record["reference"],"tr"), "Stored TEST normalization differs")
        wav=inside(record["audio_filepath"])
        info=sf.info(wav)
        require(sha256(wav)==record["wav_sha256"] and info.samplerate==16000 and info.channels==1
                and info.subtype=="PCM_16", "TEST WAV content/format differs")
        duration=info.frames/info.samplerate
        require(abs(duration-record["duration"])<=1/16000 and 0<duration<=30,
                "TEST duration differs or exceeds Whisper short-form window")
    return examples, {"manifest_path":str(manifest), "manifest_sha256":sha256(manifest),
                      "paired_baseline_test_path":str(base_path), "paired_baseline_test_sha256":sha256(base_path),
                      "checks":"100 ordered unique pinned Turkish FLEURS TEST sources; references/source audio/WAV hashes and actual monoPCM16/16k durations <=30s"}


def verified_assets(model_path):
    model_path=inside(model_path)
    lock_path=WORKSPACE/"work/strong_baseline_model.lock.json"
    lock=read(lock_path)
    require(lock.get("repository")==REPOSITORY and lock.get("revision")==REVISION
            and Path(lock["local_path"]).resolve()==model_path, "Strong checkpoint lock differs")
    dev_path=OUTPUT/"results/strong_whisper_large_v3_greedy_dev_verified/summary.json"
    dev=read(dev_path)
    require(dev.get("status")=="complete" and dev.get("repository")==REPOSITORY
            and dev.get("revision")==REVISION and dev.get("protocol")==PROTOCOL,
            "Verified DEV model/protocol differs from fixed greedy TEST protocol")
    require(dev["script_sha256"]==sha256(OUTPUT/"benchmark_strong_tr.py")
            and dev["metrics_code_sha256"]==sha256(OUTPUT/"metrics.py")
            and dev["model_lock_sha256"]==sha256(lock_path), "Verified DEV code/model-lock provenance changed")
    files=[model_path/"model.safetensors"]+sorted(p for p in model_path.iterdir()
        if p.is_file() and p.suffix in {".json",".txt"}
        and not p.name.startswith(("pytorch_model","model.safetensors.index")))
    files_sha={p.name:sha256(p) for p in files}
    require(files_sha==dev["model_files_sha256"], "Pinned Whisper model/tokenizer/frontend/generation files changed")
    return {"model_files_sha256":files_sha, "model_lock_sha256":sha256(lock_path),
            "verified_dev_summary_sha256":sha256(dev_path), "verified_dev_script_sha256":dev["script_sha256"]}


def validate_predictions(predictions, examples, *, whisper=False):
    require(len(predictions)==100 and len({r["source_id"] for r in predictions})==100, "Incomplete/duplicate100 TEST predictions")
    for prediction,record in zip(predictions,examples):
        require(all(prediction[key]==record[key] for key in PAIR_KEYS)
                and prediction.get("split")=="test" and isinstance(prediction.get("hypothesis"),str),
                "TEST prediction pairing/order/reference/source changed")
        if whisper:
            tokens=prediction.get("generated_token_ids")
            require(isinstance(tokens,list) and all(isinstance(t,int) and not isinstance(t,bool) for t in tokens)
                    and tokens[:4]==EXPECTED_PREFIX and tokens[-1:]==[EXPECTED_EOS]
                    and tokens.count(EXPECTED_EOS)==1 and prediction.get("eos_emitted") is True,
                    "Whisper TEST requires complete Turkish/transcribe/no-timestamp prefix and terminal EOS")
            require(math.isfinite(prediction["inference_seconds"]) and prediction["inference_seconds"]>=0,
                    "Nonfinite TEST inference timing")


def cpu_compare(args, frozen, freeze_audit, examples, input_audit, assets):
    """Reopen saved outputs and recompute all metrics/paired intervals on CPU."""
    output=inside(args.output_dir,OUTPUT/"results")
    summary_path=output/"summary.json"
    baseline_summary=read(summary_path)
    strong_path=output/"test_tr.jsonl"
    require(baseline_summary.get("status")=="complete" and baseline_summary.get("repository")==REPOSITORY
            and baseline_summary.get("revision")==REVISION and baseline_summary.get("split")=="test"
            and baseline_summary.get("language")=="tr" and baseline_summary.get("protocol")==PROTOCOL,
            "Strong TEST summary/protocol is incomplete or changed")
    require(baseline_summary.get("script_sha256")==sha256(Path(__file__))
            and baseline_summary.get("metrics_code_sha256")==sha256(OUTPUT/"metrics.py")
            and baseline_summary.get("input_audit")==input_audit
            and all(baseline_summary.get(key)==value for key,value in assets.items())
            and baseline_summary.get("predictions_sha256")==sha256(strong_path), "Strong TEST content/provenance changed")
    strong=rows(strong_path)
    selected_dir=Path(freeze_audit["selected_results_path"])
    selected_path=selected_dir/"test_tr.jsonl"
    adapted=rows(selected_path)
    validate_predictions(strong,examples,whisper=True)
    validate_predictions(adapted,examples)
    before=score_transcriptions(strong,language="tr");before.pop("details")
    after=score_transcriptions(adapted,language="tr");after.pop("details")
    selected_summary=read(selected_dir/"summary.json")
    require(before==baseline_summary["metrics"] and after==selected_summary["test_tr"]["asr"],
            "Recomputed TEST ASR counters differ from saved summaries")
    bootstrap=paired_bootstrap_delta([r["reference"] for r in examples],
        [r["hypothesis"] for r in strong],[r["hypothesis"] for r in adapted],language="tr",
        seed=20261010,resamples=2000,confidence=.95,groups=[r["sentence_id"] for r in examples])
    require(bootstrap["wer"]["baseline"]==before["wer"] and bootstrap["wer"]["adapted"]==after["wer"]
            and bootstrap["cer"]["baseline"]==before["cer"] and bootstrap["cer"]["adapted"]==after["cer"],
            "Paired bootstrap point estimates differ")
    report={"schema":1,"status":"complete","passed":True,"created_at_utc":datetime.now(timezone.utc).isoformat(),
            "cpu_comparison":True,"run_id":frozen["run_id"],"selected_step":frozen["selected_step"],
            "selection_split":"FLEURS validation only","heldout_split":"test","language":"tr","utterances":100,
            "frozen_selection_sha256":freeze_audit["sha256"],"frozen_selection":freeze_audit,
            "baseline_repository":REPOSITORY,"baseline_revision":REVISION,"baseline_protocol":PROTOCOL,
            "baseline_asr":before,"adapted_asr":after,
            "wer_delta_percentage_points":bootstrap["wer"]["delta_percentage_points"],
            "cer_delta_percentage_points":bootstrap["cer"]["delta_percentage_points"],
            "paired_sentence_bootstrap":bootstrap,"delta_direction":"selected Parakeet minus Whisper; positive means worse",
            "provenance":{"strong_summary_path":str(summary_path),"strong_summary_sha256":sha256(summary_path),
                          "strong_predictions_path":str(strong_path),"strong_predictions_sha256":sha256(strong_path),
                          "selected_summary_path":str(selected_dir/"summary.json"),"selected_summary_sha256":sha256(selected_dir/"summary.json"),
                          "selected_predictions_path":str(selected_path),"selected_predictions_sha256":sha256(selected_path),
                          "input_audit":input_audit,**assets,"test_script_sha256":sha256(Path(__file__)),
                          "metrics_code_sha256":sha256(OUTPUT/"metrics.py")},
            "limitations":["100 fixed Turkish TEST recordings form a small pilot benchmark; sentence-cluster uncertainty is reported.",
                           "Selected Parakeet and Whisper use their documented greedy decoding protocols and the identical Turkish scorer.",
                           "This TEST comparison is reporting only; it must not drive selection within the completed DEV loop."]}
    target=inside(args.comparison_output,OUTPUT)
    write_json(target,report)
    print(json.dumps({"comparison_path":str(target),"passed":True,"run_id":frozen["run_id"],
                      "baseline_wer":before["wer"],"adapted_wer":after["wer"],
                      "wer_delta_percentage_points":report["wer_delta_percentage_points"]}),flush=True)
    return report


def main(args):
    started=time.perf_counter()
    frozen,freeze_audit=verified_frozen_selection(args.frozen_selection,args.selected_run_id)
    examples,input_audit=verified_test_rows(args.manifest)
    assets=verified_assets(args.model_path)
    output=inside(args.output_dir,OUTPUT/"results")
    require(output.name=="strong_whisper_large_v3_greedy_test", "Use the fixed greedy TEST baseline output tag")
    if args.validate_only:
        print(json.dumps({"passed":True,"gpu_used":False,"frozen_selection":freeze_audit,"input_audit":input_audit,**assets}),flush=True)
        return
    if args.compare_only or (output/"summary.json").exists():
        cpu_compare(args,frozen,freeze_audit,examples,input_audit,assets)
        return
    output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(20261010);np.random.seed(20261010)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    require(device.type=="cuda" and torch.cuda.is_available(), "Fixed TEST protocol requires the local CUDA GPU")
    dtype=torch.float16
    torch.cuda.reset_peak_memory_stats(device)
    processor=AutoProcessor.from_pretrained(str(args.model_path),local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(str(args.model_path),local_files_only=True,
        use_safetensors=True,dtype=dtype,attn_implementation="sdpa").to(device).eval()
    model.config._name_or_path=f"{REPOSITORY}@{REVISION}"
    require(model.generation_config.eos_token_id==EXPECTED_EOS,"Pinned Whisper EOS differs")
    write_json(output/"status.json",{"stage":"running","total":100,"protocol":PROTOCOL,"frozen_selection":freeze_audit,
        "started_at_utc":datetime.now(timezone.utc).isoformat()})
    predictions=[];prediction_path=output/"test_tr.jsonl"
    with prediction_path.open("w",encoding="utf-8",newline="\n") as stream:
        for index,record in enumerate(examples,1):
            signal,sample_rate=sf.read(record["audio_filepath"],dtype="float32")
            require(sample_rate==16000 and np.isfinite(signal).all(),"Unexpected/nonfinite TEST audio")
            features=processor(signal,sampling_rate=sample_rate,return_tensors="pt",return_attention_mask=True)
            inputs={key:value.to(device=device,dtype=dtype if key=="input_features" else value.dtype) for key,value in features.items()}
            torch.cuda.synchronize();clip_start=time.perf_counter()
            with torch.inference_mode():
                generated=model.generate(**inputs,**PROTOCOL).sequences
            torch.cuda.synchronize();elapsed=time.perf_counter()-clip_start
            tokens=generated[0].tolist()
            require(tokens[:4]==EXPECTED_PREFIX and tokens[-1:]==[EXPECTED_EOS]
                    and tokens.count(EXPECTED_EOS)==1,"Whisper TEST generation lacks complete prefix/terminal EOS")
            prediction={key:record[key] for key in PAIR_KEYS}
            prediction.update({"split":"test","hypothesis":processor.batch_decode(generated,skip_special_tokens=True)[0].strip(),
                               "generated_token_ids":tokens,"eos_emitted":True,"inference_seconds":elapsed})
            predictions.append(prediction);stream.write(json.dumps(prediction,ensure_ascii=False)+"\n");stream.flush()
            if index%5==0:
                print(json.dumps({"completed":index,"total":100,"elapsed_seconds":time.perf_counter()-started}),flush=True)
                write_json(output/"status.json",{"stage":"running","completed":index,"total":100})
    validate_predictions(predictions,examples,whisper=True)
    metrics=score_transcriptions(predictions,language="tr");metrics.pop("details")
    elapsed=math.fsum(r["inference_seconds"] for r in predictions);audio=math.fsum(r["duration"] for r in predictions)
    summary={"schema":1,"status":"complete","created_at_utc":datetime.now(timezone.utc).isoformat(),
             "repository":REPOSITORY,"revision":REVISION,"split":"test","language":"tr","protocol":PROTOCOL,
             "metrics":metrics,"input_audit":input_audit,**assets,"frozen_selection_before_inference":freeze_audit,
             "script_sha256":sha256(Path(__file__)),"metrics_code_sha256":sha256(OUTPUT/"metrics.py"),
             "predictions_sha256":sha256(prediction_path),"device":str(device),"hardware":torch.cuda.get_device_name(device),
             "weight_compute_dtype":str(dtype),"attention_implementation":"sdpa","torch_version":torch.__version__,
             "transformers_version":__import__('transformers').__version__,"all_full_prefixes_verified":True,"all_terminal_eos_verified":True,
             "peak_allocated_vram_gb":torch.cuda.max_memory_allocated(device)/1e9,
             "audio_seconds":audio,"inference_seconds":elapsed,"real_time_factor":elapsed/audio,
             "elapsed_seconds_with_load_and_audit":time.perf_counter()-started}
    write_json(output/"summary.json",summary)
    write_json(output/"status.json",{"stage":"complete","completed":100,"summary_sha256":sha256(output/"summary.json")})
    del model,processor,inputs,features,generated
    gc.collect();torch.cuda.empty_cache()
    print(json.dumps({"gpu_released":True,"baseline_wer":metrics["wer"]}),flush=True)
    # Source checkpoint/data/frozen selection must remain identical throughout.
    frozen_again,freeze_again=verified_frozen_selection(args.frozen_selection,args.selected_run_id)
    require(freeze_again==freeze_audit and frozen_again==frozen,"Frozen winner changed during TEST inference")
    cpu_compare(args,frozen,freeze_audit,examples,input_audit,assets)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen-selection",type=Path,required=True)
    p.add_argument("--selected-run-id")
    p.add_argument("--model-path",type=Path,default=WORKSPACE/"work/models/whisper-large-v3")
    p.add_argument("--manifest",type=Path,default=WORKSPACE/"work/data/test_tr.jsonl")
    p.add_argument("--output-dir",type=Path,default=OUTPUT/"results/strong_whisper_large_v3_greedy_test")
    p.add_argument("--comparison-output",type=Path,default=OUTPUT/"adaptive_strong_baseline_test_comparison.json")
    p.add_argument("--device",default="cuda")
    modes=p.add_mutually_exclusive_group()
    modes.add_argument("--compare-only",action="store_true")
    modes.add_argument("--validate-only",action="store_true")
    return p


if __name__=="__main__":
    main(parser().parse_args())
