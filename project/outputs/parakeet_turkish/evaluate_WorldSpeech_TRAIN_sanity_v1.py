"""Bounded serial TRAIN-only source sanity with the coherent qualified40750 model.

The selected64 clips are a stratified stress sample, not an unbiased WER estimate.
No optimizer, training, relabeling, export, fixed DEV or TEST evaluation occurs.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import deque
import argparse, ctypes, gc, hashlib, json, math, os, shutil, sys, threading, time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUN = "adaptive_20261014_TR_uniform_token_population_01"
PREP = "8aac435e3472d3e304a88c4ac952756f1a01682682ef1d8cc82beedbc694e9e7"
VERIFY = "f4e1fe9b487d1a8fc8da25e9ed0faf9f8ff35cf293bcc87ace4b084fa0b55e3f"
MANIFEST = "9b1fcb1436bbe1bae5efbcef6930faa63f113555e2112b8be109564b9eee67c8"

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def record(path):
    path = Path(path).resolve()
    before = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Source changed during hash")
    return {"path": str(path), "sha256": digest, "bytes": before.st_size}

def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)

def preflight(args):
    require(0 < args.max_seconds <= 600 and not args.output.exists(), "Fresh bounded sanity output required")
    closure_path = HERE / "TR_token_population_actual_CLOSED43238_preservation_v1.json"
    closure = read(closure_path)
    require(closure["actual_owned_exit"] == {"session_id": 15227, "exit_code": 0, "tool_chunk": "cb3509"}
            and closure["native_child_exit_code"] == 0 and closure["native_sources_unchanged"] is True
            and closure["selected_qualified_BEST_step"] == 40750, "Actual coherent closed40750 BEST required")
    config_record = closure["artifacts"]["run_config"]["archived"]
    require(record(config_record["path"]) == config_record, "Closed config changed")
    config = read(config_record["path"])
    for name, digest in config["config_identity"]["source_sha256"].items():
        require(record(HERE / name)["sha256"] == digest, "Frozen scientific source changed: " + name)
    prep_path = HERE / "WorldSpeech_TR_TRAIN_audio_candidate_preparation_v1.json"
    verification_path = HERE / "WorldSpeech_complete_candidate_CPU_independent_verification_v1.json"
    manifest = ROOT / "work/data/WorldSpeech_tr_quality_candidate_v1/independent_TRAIN_sanity64.jsonl"
    for path, digest in ((prep_path, PREP), (verification_path, VERIFY), (manifest, MANIFEST)):
        require(record(path)["sha256"] == digest, "Actual candidate evidence changed")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    require(len(rows) == len({row["source_id"] for row in rows}) == 64
            and all(row["language"] == "tr" and row["dataset"] == "disco-eth/WorldSpeech"
                    and row["dataset_split"] == "train" for row in rows), "Exactly64 WorldSpeech TRAIN rows required")
    lease = read(ROOT / "work/gpu_training_owner.json")
    require(lease["owner"] == "/root/benchmark_design" and lease["actual_GPU_running"] is False,
            "Closed sole-owner GPU lease required before diagnostic")
    return {"closure": closure, "closure_record": record(closure_path), "config": config,
            "checkpoint": closure["artifacts"]["best_checkpoint"]["archived"],
            "manifest": record(manifest), "preparation": record(prep_path),
            "independent_CPU_verification": record(verification_path)}

def main(args):
    context = preflight(args)
    if args.validate_only:
        print(json.dumps({"passed": True, "GPU_launched": False, "TRAIN_clips": 64, "source_step": 40750})); return
    require(args.gpu_window_authorized and "torch" not in sys.modules, "Exclusive GPU authorization/preload order required")
    from resource_watchdog_v2 import ResourceGuard, sample_memory
    prior_status = read(context["closure"]["artifacts"]["controller_status"]["archived"]["path"])
    peak = prior_status["peak_observed_owned_private_bytes"]
    sample = sample_memory(os.getpid(), "WorldSpeech_TRAIN_sanity_before_GPU")["system_memory"]
    require(sample["available"] and sample["commit_used_bytes"] + peak < .9 * sample["commit_limit_bytes"]
            and sample["commit_limit_bytes"] - sample["commit_used_bytes"] - peak > 2_000_000_000,
            "Fresh unchanged system baseline plus actual previous peak fails memory admission")
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.GetCurrentProcess.restype = ctypes.c_void_p
    api.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    require(api.SetPriorityClass(api.GetCurrentProcess(), 0x4000), "BelowNormal process priority failed")
    started = time.monotonic(); stop = threading.Event(); failures = []; phase = ["preload"]
    observations = deque(maxlen=160); guard = ResourceGuard()
    report = {"kind": "actual_WorldSpeech_TRAIN_sanity64_v1", "status": "running", "pid": os.getpid(),
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "script": record(__file__),
              "source_step": 40750, "source_run_id": RUN, "source_checkpoint": context["checkpoint"],
              "source_closure": context["closure_record"], "TRAIN_manifest": context["manifest"],
              "candidate_preparation": context["preparation"], "independent_CPU_verification": context["independent_CPU_verification"],
              "fresh_memory_admission": sample, "actual_previous_private_peak": peak,
              "optimizer_created": False, "optimizer_updates": 0, "labels_replaced": False,
              "TEST_or_fixed_DEV_evaluated": False, "source350_repeated": False,
              "one_coherent_GPU_model": True, "sample_is_stratified_stress_not_unbiased_WER": True}
    write(args.output, report)
    def monitor():
        while not stop.is_set():
            observed = sample_memory(os.getpid(), phase[0])
            reason, evidence = guard.observe(observed["system_memory"], observed["worker_private_memory"],
                                             observed_monotonic=observed["observed_monotonic"])
            observations.append({"phase": phase[0], "evidence": evidence})
            if reason: failures.append(reason)
            stop.wait(5)
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    def check():
        require(not failures, str(failures))
        require(time.monotonic() - started <= args.max_seconds, "TRAIN sanity600second bound")
    try:
        from preload_nemo_linker_v1 import preload
        report["preload"] = preload()
        require(report["preload"]["kind"] == "coherent_native_cuda12_9_preload_v1", "Coherent CUDA12.9 preload required")
        import torch
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(min(1., 8_000_000_000 / torch.cuda.get_device_properties(0).total_memory))
        from runtime import load_model
        from extend_vocab import extend_processor_and_model
        from encoder_lora import install_encoder_lora, encoder_lora_metadata
        import benchmark
        phase[0] = "load_one_model"; check(); model, processor = load_model()
        vocabulary = extend_processor_and_model(processor, model)
        config = context["config"]
        layout = install_encoder_lora(model, rank=config["rank"], alpha=config["alpha"], last_n_layers=config["last_n_layers"])
        model.config.encoder_lora_tracking = {"layout": layout, "implementation_sha256": record(HERE / "encoder_lora.py")["sha256"], "kind": "unmerged"}
        expected = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        require(len(expected) == 207 and model.config.vocab_size == 8199 and model.config.blank_token_id == 8192
                and vocabulary["extended_tokenizer_sha256"] == config["config_identity"]["extended_tokenizer_sha256"], "Exact original8199/207 layout required")
        require(record(context["checkpoint"]["path"]) == context["checkpoint"], "Archived coherent BEST changed")
        state = torch.load(context["checkpoint"]["path"], map_location="cpu", weights_only=True)
        require(state["schema"] == 2 and state["run_id"] == RUN and state["step"] == 40750
                and state["config_identity"] == config["config_identity"]
                and state["lora_metadata"] == encoder_lora_metadata(model) and set(state["weights"]) == set(expected), "Actual coherent saved207 source required")
        for name, tensor in state["weights"].items():
            require(tensor.dtype == torch.float32 and tensor.shape == expected[name].shape and torch.isfinite(tensor).all(), "Finite matching saved tensor: " + name)
        loaded = model.load_state_dict(state["weights"], strict=False)
        require(not loaded.unexpected_keys, "Unexpected trained source weights")
        del state; gc.collect(); torch.cuda.empty_cache(); check()
        directory = args.output.parent / (args.output.stem + "_inputs"); directory.mkdir(exist_ok=False)
        shutil.copyfile(context["manifest"]["path"], directory / "train_tr.jsonl")
        original_inputs = benchmark.inputs_for
        def guarded_inputs(*a, **kw):
            check(); return original_inputs(*a, **kw)
        benchmark.inputs_for = guarded_inputs
        original_data = benchmark.DATA
        try:
            benchmark.DATA = directory; phase[0] = "decode64_actual_TRAIN"
            result = benchmark.evaluate(model, processor, args.output.stem, languages=("tr",), splits=("train",))
        finally:
            benchmark.DATA = original_data; benchmark.inputs_for = original_inputs
        require(result["train_tr"]["asr"]["utterances"] == 64, "All64 TRAIN predictions required")
        folder = HERE / "results" / args.output.stem
        report.update(status="complete", passed=True, actual_GPU_decode=True, TRAIN_ASR=result["train_tr"]["asr"],
                      predictions=record(folder / "train_tr.jsonl"), summary=record(folder / "summary.json"),
                      metadata=record(folder / "metadata.json"), peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),
                      interpretation="ASR disagreement on a stratified TRAIN stress sample is not manual label-error truth or an unbiased source WER estimate. Human labels remain unchanged.")
    except BaseException as error:
        report.update(status="failed", passed=False, error=repr(error)); raise
    finally:
        stop.set(); thread.join(7)
        report.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
                      guard_observations=list(observations), resource_stop=failures, guard_thread_joined=not thread.is_alive())
        write(args.output, report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--gpu-window-authorized", action="store_true")
    main(parser.parse_args())
