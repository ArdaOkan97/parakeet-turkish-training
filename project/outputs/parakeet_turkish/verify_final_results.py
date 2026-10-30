"""CPU audit of one finalized, development-selected Parakeet LoRA pilot.

Usage: python verify_final_results.py --run-id ID --report PATH
Writes both runs/ID/final_results_verification.json and the output-root copy.
Exit 0 means saved results/provenance are internally verified. Model quality is
reported separately; a consistent evaluation can still miss its quality goal.
No model inference, GPU calls, source/report mutations or Backlog edits occur.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from confidence import align_word_confidences, confidence_report, fit_temperature
from metrics import normalize_text, paired_bootstrap_delta, score_transcriptions

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
DATA = ROOT / "work/data"
LANGUAGES = ("en", "es", "it", "fr", "de", "tr")


class AuditError(RuntimeError):
    pass


class Checks:
    def __init__(self):
        self.items = []

    def require(self, condition, name):
        if not condition:
            raise AuditError(name)
        self.items.append({"name": name, "passed": True})

    def equal(self, actual, expected, name):
        self.require(actual == expected, name)


def json_load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows_load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def export_manifest(path):
    path = Path(path)
    return {str(file.relative_to(path)): file_sha(file) for file in sorted(path.rglob("*"))
            if file.is_file() and file.name != "finalization_provenance.json"}


def local_path(value, check, name):
    path = Path(value).resolve()
    check.require(path.is_relative_to(ROOT.resolve()), name+": path is inside workspace")
    check.require(path.exists(), name+": path exists")
    return path


def results_folder(value, check, name):
    if isinstance(value, dict):
        value = value.get("summary") or value.get("path")
    path = local_path(value, check, name)
    return path.parent if path.is_file() else path


def require_finite_probabilities(row, check, label):
    tokens = row["confidence"]["tokens"]
    finite, valid_logs, valid_blank_entropy = True, True, True
    for token in tokens:
        values = [token["logprob"], token["duration_logprob"], token["conditional_nonblank_logprob"],
                  token["blank_probability"], token["normalized_entropy"], *token["logprob_by_temperature"].values()]
        finite &= all(math.isfinite(value) for value in values)
        valid_logs &= token["logprob"] <= 1e-5 and token["duration_logprob"] <= 1e-5 and token["conditional_nonblank_logprob"] <= 1e-5
        valid_logs &= all(value <= 1e-5 for value in token["logprob_by_temperature"].values())
        valid_blank_entropy &= 0 <= token["blank_probability"] <= 1 and 0 <= token["normalized_entropy"] <= 1.00001
    check.require(finite, label+": all token probabilities finite")
    check.require(valid_logs, label+": all token log probabilities valid")
    check.require(valid_blank_entropy, label+": all blank/entropy probabilities valid")
    geometric, bounded = True, True
    for word in row["confidence"]["words"]:
        indices = word["token_indices"]
        check.require(bool(indices) and all(type(value) is int and 0 <= value < len(tokens) for value in indices)
                      and indices == sorted(set(indices)), label+": unique ordered integer word token indices")
        check.equal(set(word["confidence_by_temperature"]),set(tokens[indices[0]]["logprob_by_temperature"]),label+": exact word/token temperature-key set")
        check.require(all(set(tokens[value]["logprob_by_temperature"]) == set(word["confidence_by_temperature"]) for value in indices),label+": consistent lexical-token temperature keys")
        expected_by_temperature = {key:math.exp(sum(tokens[value]["logprob_by_temperature"][key] for value in indices)/len(indices))
                                   for key in tokens[indices[0]]["logprob_by_temperature"]}
        geometric &= all(math.isclose(word["confidence_by_temperature"][key],expected,rel_tol=1e-12,abs_tol=1e-12)
                         for key,expected in expected_by_temperature.items())
        geometric &= math.isclose(word["confidence"],expected_by_temperature["1"],rel_tol=1e-12,abs_tol=1e-12)
        bounded &= 0 <= word["confidence"] <= 1.00001
        expected_conditional = math.exp(sum(tokens[value]["conditional_nonblank_logprob"] for value in indices)/len(indices))
        geometric &= math.isclose(word["conditional_nonblank_confidence"],expected_conditional,rel_tol=1e-12,abs_tol=1e-12)
        expected_entropy = sum(tokens[value]["normalized_entropy"] for value in indices)/len(indices)
        geometric &= math.isclose(word["mean_normalized_entropy"],expected_entropy,rel_tol=1e-12,abs_tol=1e-12)
        bounded &= math.isfinite(word["conditional_nonblank_confidence"]) and 0 <= word["conditional_nonblank_confidence"] <= 1.00001
        bounded &= math.isfinite(word["mean_normalized_entropy"]) and 0 <= word["mean_normalized_entropy"] <= 1.00001
    check.require(geometric, label+": all word confidences are token geometric means")
    check.require(bounded, label+": all word confidences bounded")


def verify_split(folder, metadata, summary, language, split, check):
    """Public helper also used by CPU regression checks on real saved predictions."""
    key = split+"_"+language
    manifest_path = DATA / (key+".jsonl")
    manifest = rows_load(manifest_path)
    predictions = rows_load(folder / (key+".jsonl"))
    count = 50 if split == "validation" else 100
    check.equal(len(manifest), count, key+": fixed manifest count")
    check.equal(len(predictions), count, key+": complete prediction count")
    identities = [row["source_id"] for row in predictions]
    expected_ids = [row["source_id"] for row in manifest]
    check.equal(len(set(identities)), count, key+": no duplicate prediction source IDs")
    check.equal(identities, expected_ids, key+": exact fixed source IDs/order")
    state = metadata["manifests"][key]
    check.equal(state["file_sha256"], file_sha(manifest_path), key+": immutable manifest SHA256")
    check.equal(Path(state["path"]).resolve(), manifest_path.resolve(), key+": original manifest path")
    check.equal(state["source_ids"], expected_ids, key+": metadata source IDs")
    check.equal(state["utterances"], count, key+": metadata count")
    check.equal(state["limit"], 0, key+": no evaluation truncation")
    for index, (source, row) in enumerate(zip(manifest, predictions)):
        label = key+f"[{index}]"
        check.equal(source["dataset_split"], split, label+": official dataset split")
        check.equal(row["reference"], source["text"], label+": human reference")
        for name in ("sentence_id", "dataset_revision", "audio_sha256", "wav_sha256", "duration", "language"):
            check.equal(row[name], source[name], label+": "+name)
        check.equal(row["split"], split, label+": prediction split")
        check.equal(row["language"], language, label+": prediction language")
        check.equal(row["hypothesis"].strip(), row["confidence"]["text"].strip(), label+": actual decoded confidence text")
        check.equal(normalize_text(row["hypothesis"], language).split(),
                    [word["word"] for word in row["confidence"]["words"]], label+": scored words cover decoded words")
        check.equal(align_word_confidences(row["reference"], row["confidence"], language), row["alignment"], label+": exact word correctness/deletion alignment")
        require_finite_probabilities(row, check, label)
    metrics = score_transcriptions(predictions, language=language)
    metrics.pop("details")
    check.equal(metrics, summary[key]["asr"], key+": exact corpus ASR counters/rates")
    check.equal(confidence_report([row["alignment"] for row in predictions]), summary[key]["raw_confidence"], key+": exact raw confidence/calibration metrics")
    return predictions, metrics


def verify_transcription_smoke(report, run_dir, adapted_folder, calibration, check):
    """Compare the real exported-model CLI output with its paired dev benchmark."""
    smoke_path = local_path(report["artifacts"]["transcription_smoke"],check,"transcription CLI smoke")
    check.equal(smoke_path,(run_dir / "transcription_smoke.json").resolve(),"transcription smoke is run-specific")
    reference_path = run_dir / "transcription_smoke_reference.json"
    if "transcription_smoke_reference" in report["artifacts"]:
        check.equal(Path(report["artifacts"]["transcription_smoke_reference"]).resolve(),reference_path.resolve(),"report smoke reference points to actual run companion")
    reference_path = local_path(reference_path,check,"transcription smoke reference")
    smoke, reference = json_load(smoke_path),json_load(reference_path)
    source = rows_load(DATA / "validation_tr.jsonl")[0]
    check.equal(reference["source_id"],source["source_id"],"smoke source is first fixed Turkish development row")
    check.equal(reference["reference"],source["text"],"smoke human reference is fixed row reference")
    audio = local_path(reference["audio_filepath"],check,"smoke WAV")
    check.equal(audio,Path(source["audio_filepath"]).resolve(),"smoke uses the fixed row WAV path")
    check.equal(reference["wav_sha256"],source["wav_sha256"],"smoke reference WAV SHA256 matches fixed source")
    check.equal(file_sha(audio),source["wav_sha256"],"smoke actual WAV bytes match fixed source")
    matching = [row for row in rows_load(adapted_folder / "validation_tr.jsonl") if row["source_id"] == source["source_id"]]
    check.equal(len(matching),1,"smoke source has exactly one adapted development prediction")
    row = matching[0]
    check.equal(smoke["transcript"].strip(),row["hypothesis"].strip(),"exported CLI smoke exact transcript matches actual paired benchmark")
    check.equal(normalize_text(smoke["transcript"],"tr"),normalize_text(row["hypothesis"],"tr"),"exported CLI smoke normalized transcript matches actual paired benchmark")
    check.equal(smoke["confidence_language"],"tr","exported CLI smoke confidence language")
    check.equal(smoke["confidence_temperature"],calibration["tr"]["temperature"],"exported CLI smoke uses frozen Turkish dev temperature")
    temperature_key = format(float(calibration["tr"]["temperature"]),".8g")
    check.equal(len(smoke["words"]),len(row["confidence"]["words"]),"exported CLI smoke covers every actual benchmark lexical word")
    differences = {"raw_confidence":0.,"calibrated_confidence":0.,"conditional_nonblank_confidence":0.}
    for index,(observed,expected) in enumerate(zip(smoke["words"],row["confidence"]["words"])):
        check.equal(observed["word"],expected["surface"],f"smoke word{index}: exact benchmark surface")
        check.equal(normalize_text(observed["word"],"tr"),expected["word"],f"smoke word{index}: exact normalized benchmark unit")
        expected_values = {"raw_confidence":expected["confidence"],"calibrated_confidence":expected["confidence_by_temperature"][temperature_key],
                           "conditional_nonblank_confidence":expected["conditional_nonblank_confidence"]}
        for name,value in expected_values.items():
            actual = observed[name]
            check.require(type(actual) in (int,float) and math.isfinite(actual) and 0 <= actual <= 1.00001,f"smoke word{index}: finite bounded "+name)
            difference = abs(actual-value)
            differences[name] = max(differences[name],difference)
            # Benchmark computes 12 temperature rows and CLI only T=1/fitted T.
            # Their FP32 logsumexp reductions can have a few ULP differences;
            # decoded words stay exact and probability tolerance is explicit.
            check.require(math.isclose(actual,value,rel_tol=3e-6,abs_tol=3e-6),f"smoke word{index}: actual benchmark "+name)
    return {"path":str(smoke_path),"sha256":file_sha(smoke_path),"reference_path":str(reference_path),"reference_sha256":file_sha(reference_path),
            "source_id":source["source_id"],"wav_sha256":source["wav_sha256"],"word_count":len(smoke["words"]),"temperature":calibration["tr"]["temperature"],
            "maximum_absolute_confidence_differences":differences,"probability_absolute_and_relative_tolerance":3e-6,
            "policy":"Exact source/WAV/language/temperature/transcript/word units; tiny FP32 confidence roundoff only. Same pinned exported model, native eval/BF16 generation inputs; no stochastic sampling."}


def verify_export_fingerprint(checkpoint, metadata, provenance, check, *, require_merged=True):
    """Stream saved FP32 safetensors on CPU to verify the recorded model state."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForTDT, AutoProcessor, GenerationConfig

    torch.set_num_threads(1)
    tensor_files = {}
    for path in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            for name in stream.keys():
                check.require(name not in tensor_files, "checkpoint: unique safetensors state key "+name)
                tensor_files[name] = path
    check.require(bool(tensor_files), "checkpoint: standard-HF safetensors state exists")
    digest = hashlib.sha256()
    for name in sorted(tensor_files):
        with safe_open(str(tensor_files[name]), framework="pt", device="cpu") as stream:
            tensor = stream.get_tensor(name)
        check.require(not tensor.is_floating_point() or tensor.dtype == torch.float32, "checkpoint: FP32 merged floating tensor "+name)
        check.require(not tensor.is_floating_point() or bool(torch.isfinite(tensor).all()), "checkpoint: finite tensor "+name)
        digest.update(name.encode("utf-8")+b"\0")
        digest.update(str(tensor.dtype).encode("ascii")+b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii")+b"\0")
        byte_view = tensor.contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(byte_view))
        del byte_view, tensor
    identity = metadata["model_fingerprint"]["identity"]
    check.equal(digest.hexdigest(), identity["model_state_sha256"], "checkpoint: streamed state matches evaluated model fingerprint")
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    tokenizer = processor.tokenizer
    backend = tokenizer.backend_tokenizer
    check.equal(json_sha(json.loads(backend.to_str())), provenance["extended_tokenizer_sha256"], "checkpoint: trained tokenizer semantic SHA256")
    token_state = {"backend": backend.to_str(), "special_tokens_map": tokenizer.special_tokens_map,
                   "padding_side": tokenizer.padding_side, "truncation_side": tokenizer.truncation_side}
    check.equal(json_sha(token_state), identity["tokenizer_sha256"], "checkpoint: evaluated tokenizer identity")
    processor_state = {}
    for name in getattr(processor,"attributes",[]):
        component = getattr(processor,name,None)
        if name != "tokenizer" and component is not None and hasattr(component,"to_dict"):
            processor_state[name] = component.to_dict()
    check.equal(json_sha(processor_state), identity["processor_sha256"], "checkpoint: evaluated audio processor identity")
    configuration = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    configuration.name_or_path = provenance["evaluation_config_name_or_path"]
    # The HF model constructor normalizes a few config defaults. Recreate that
    # normalization on meta tensors, without loading another weight copy.
    with torch.device("meta"):
        template = AutoModelForTDT.from_config(configuration,dtype=torch.float32)
    configuration = template.config
    if require_merged:
        check.equal(configuration.encoder_lora_tracking["kind"], "merged", "checkpoint: standard merged encoder tracking")
        check.equal(configuration.encoder_lora_tracking["run_id"], provenance["run_id"], "checkpoint: merged encoder run ID")
        check.equal(configuration.encoder_lora_tracking["selected_step"], provenance["selected_step"], "checkpoint: merged encoder step")
        check.equal(configuration.encoder_lora_tracking["layout"], provenance["adapter_layout"], "checkpoint: merged encoder adapter layout")
    check.equal(json_sha(configuration.to_dict()), identity["model_config_sha256"], "checkpoint: normalized evaluated model config")
    generation = GenerationConfig.from_pretrained(checkpoint, local_files_only=True)
    check.equal(json_sha(generation.to_dict()), identity["generation_config_sha256"], "checkpoint: evaluated generation config")
    del template, processor


def audit(run_id, report_path, check):
    run_dir = OUTPUT / "runs" / run_id
    report = json_load(report_path)
    config_path = run_dir / "run_config.json"
    config = json_load(config_path)
    selection = json_load(run_dir / "selection.json")
    training_selection = json_load(run_dir / "training_selection.json")
    pointer = json_load(run_dir / "adapted_results_path.json")
    provenance = json_load(run_dir / "finalization_provenance.json")
    comparisons = json_load(run_dir / "benchmark_comparison.json")
    calibration = json_load(run_dir / "confidence_calibration.json")
    check.require(report["run_id"] == config["run_id"] == selection["run_id"] == training_selection["run_id"] == provenance["run_id"] == run_id, "all final artifact run IDs match")
    check.require(selection["accepted"] and selection["merged"] and selection["heldout_tests_evaluated"], "selection: merged dev-selected model and completed heldout inference")
    check.require(report["development_selection_accepted"] and report["heldout_tests_evaluated"], "report: completed development selection and heldout inference")
    check.require(training_selection.get("heldout_tests_evaluated") is False, "training selection excludes heldout test inference")
    check.require(training_selection.get("selection_split") == config.get("selection_split") == "FLEURS validation only", "training selection and configuration are development only")
    selected_step = selection["selected_step"]
    check.require(report["selected_step"] == training_selection["selected_step"] == provenance["selected_step"] == selected_step, "selected step consistent across artifacts")
    baseline_folder = results_folder(report["baseline_results_path"], check, "baseline results")
    adapted_folder = results_folder(report["adapted_results_path"], check, "adapted results")
    check.equal(baseline_folder, (OUTPUT / "results/base_v3").resolve(), "baseline: original fixed v3 result folder")
    check.equal(adapted_folder, Path(pointer["summary"]).resolve().parent, "adapted results pointer matches report")
    check.equal(adapted_folder.name, pointer["tag"], "adapted results tag matches pointer")
    check.equal(selection["final_results_tag"], pointer["tag"], "final selection references same result tag")
    checkpoint = local_path(report["checkpoint"], check, "checkpoint")
    check.require(checkpoint == Path(pointer["checkpoint"]).resolve() == Path(selection["checkpoint"]).resolve(), "checkpoint path consistent across artifacts")
    candidate = local_path(provenance["candidate"], check, "merged candidate")
    check.equal(json_load(checkpoint / "finalization_provenance.json"), provenance, "published checkpoint provenance matches run")
    check.equal(json_load(candidate / "finalization_provenance.json"), provenance, "candidate provenance matches run")
    check.equal(export_manifest(checkpoint), provenance["candidate_files_sha256"], "checkpoint: complete exported file hashes")
    check.equal(export_manifest(candidate), provenance["candidate_files_sha256"], "candidate: complete exported file hashes")
    base_path = local_path(provenance["base_path"], check, "pinned base")
    check.equal(base_path.name, config["base_revision"], "pinned base revision")
    check.equal(export_manifest(base_path), provenance["base_files_sha256"], "pinned base: complete file hashes")
    check.equal(file_sha(config_path), provenance["run_config_sha256"], "run config provenance SHA256")
    check.equal(file_sha(run_dir / "vocabulary_audit.json"), provenance["training_vocabulary_audit_sha256"], "training tokenizer audit provenance SHA256")
    check.equal(json_sha(provenance["adapter_layout"]), provenance["adapter_layout_sha256"], "adapter layout provenance SHA256")
    check.equal(provenance["adapter_layout"], config["lora_layout"], "adapter layout matches actual training config")
    check.require(provenance["merged_fp32"] and provenance["reconstruction"]["all_trainable_keys_loaded"] and provenance["reconstruction"]["frozen_state_loaded_from_pinned_base"], "finalizer: complete FP32 merge reconstruction recorded")
    selected_state = local_path(selection["selected_trainables"], check, "archived selected trainables")
    check.equal(file_sha(selected_state), provenance["selected_state_sha256"], "archived selected trainable-state SHA256")
    import torch
    torch.set_num_threads(1)
    state = torch.load(selected_state,map_location="cpu",weights_only=True)
    check.require(state["run_id"] == run_id and state["step"] == selected_step, "archived trainables: actual checkpoint run ID/step")
    check.equal(state["lora_metadata"], config["lora_layout"], "archived trainables: actual adapter layout")
    check.equal(sum(value.numel() for value in state["weights"].values()),config["trainable_parameters"], "archived trainables: all trained parameters present")
    check.equal(len(state["weights"]),provenance["reconstruction"]["trainable_key_count"], "archived trainables: reconstructed tensor count")
    check.equal(json_sha(sorted(state["weights"])),provenance["reconstruction"]["trainable_keys_sha256"], "archived trainables: exact reconstructed named keys")
    check.require(all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in state["weights"].values()), "archived trainables: finite FP32 saved weights")
    del state
    state_source = local_path(provenance["selected_state_path"], check, "source selected trainables")
    check.equal(file_sha(state_source), provenance["selected_state_sha256"], "source selected trainable-state SHA256")
    check.equal(provenance["training_code_sha256"], config["code_sha256"], "training code provenance dictionary")
    for name, expected in config["code_sha256"].items():
        check.equal(file_sha(OUTPUT / name), expected, "training code SHA256: "+name)
    check.equal(file_sha(OUTPUT / "finalize_lora.py"), provenance["finalizer_sha256"], "finalizer code SHA256")
    baseline_summary = json_load(baseline_folder / "summary.json")
    adapted_summary = json_load(adapted_folder / "summary.json")
    baseline_metadata = json_load(baseline_folder / "metadata.json")
    metadata = json_load(adapted_folder / "metadata.json")
    check.equal(metadata["schema_version"], 1, "adapted metadata schema")
    check.equal(json_sha(metadata["model_fingerprint"]["identity"]), metadata["model_fingerprint"]["sha256"], "adapted fingerprint identity digest")
    check.require(metadata["model_fingerprint"]["sha256"] != baseline_metadata["model_fingerprint"]["sha256"], "adapted model fingerprint differs from original base")
    expected_keys = {split+"_"+language for split in ("validation", "test") for language in LANGUAGES}
    check.equal(set(metadata["manifests"]), expected_keys, "exact12 final development/test manifest keys")
    for name, expected in metadata["model_fingerprint"]["identity"]["evaluation_code_sha256"].items():
        check.equal(file_sha(OUTPUT / name), expected, "final evaluation code SHA256: "+name)
    for name in ("metrics.py", "confidence.py", "runtime.py"):
        check.equal(metadata["model_fingerprint"]["identity"]["evaluation_code_sha256"][name], baseline_metadata["model_fingerprint"]["identity"]["evaluation_code_sha256"][name], "same base/final scoring/inference code: "+name)
    check.equal(metadata["model_fingerprint"]["identity"]["inference_dtype"], baseline_metadata["model_fingerprint"]["identity"]["inference_dtype"], "same base/final AMP compute dtype")
    verify_export_fingerprint(checkpoint, metadata, provenance, check)
    merged_folder = OUTPUT / "results" / selection["merged_development_tag"]
    merged_metadata = json_load(merged_folder / "metadata.json")
    check.equal(metadata["model_fingerprint"], merged_metadata["model_fingerprint"], "merged-reloaded dev and final predictions use same model identity")
    results, heldout_passes = {}, {}
    all_dev_sentences, all_test_sentences = set(), set()
    for language in LANGUAGES:
        saved = {}
        for split in ("validation", "test"):
            rows, metrics = verify_split(adapted_folder, metadata, adapted_summary, language, split, check)
            before, original_metrics = verify_split(baseline_folder, baseline_metadata, baseline_summary, language, split, check)
            check.equal(metadata["manifests"][split+"_"+language], baseline_metadata["manifests"][split+"_"+language], split+"_"+language+": unchanged paired baseline manifest")
            saved[split] = (rows, metrics, before, original_metrics)
        dev, dev_metrics, _, original_dev = saved["validation"]
        test, test_metrics, before, original_test = saved["test"]
        all_dev_sentences.update(row["sentence_id"] for row in dev)
        all_test_sentences.update(row["sentence_id"] for row in test)
        check.equal(file_sha(adapted_folder / ("validation_"+language+".jsonl")), file_sha(merged_folder / ("validation_"+language+".jsonl")), language+": final dev copied from reloaded merged model")
        fitted = fit_temperature([row["alignment"] for row in dev])
        check.equal(fitted, adapted_summary["calibration_"+language], language+": temperature independently fitted from dev only")
        check.equal(fitted, calibration[language], language+": durable calibration matches dev fit")
        check.equal(confidence_report([row["alignment"] for row in test], fitted["temperature"]), adapted_summary["test_"+language]["calibrated_confidence"], language+": heldout confidence uses frozen dev temperature")
        baseline_dev_fit = fit_temperature([row["alignment"] for row in saved["validation"][2]])
        check.equal(baseline_dev_fit,baseline_summary["calibration_"+language],language+": unchanged baseline temperature fitted from dev only")
        check.equal(confidence_report([row["alignment"] for row in before],baseline_dev_fit["temperature"]),baseline_summary["test_"+language]["calibrated_confidence"],language+": unchanged baseline heldout calibration verified")
        bootstrap = paired_bootstrap_delta([row["reference"] for row in before], [row["hypothesis"] for row in before], [row["hypothesis"] for row in test], language=language, groups=[row["sentence_id"] for row in before])
        check.equal(bootstrap, comparisons[language], language+": exact paired sentence-cluster test bootstrap including point estimates")
        test_report = report["per_language_test"][language]
        check.equal(test_report["baseline_asr"], original_test, language+": report original test ASR")
        check.equal(test_report["adapted_asr"], test_metrics, language+": report adapted test ASR")
        check.equal(test_report["wer_delta_percentage_points"], bootstrap["wer"]["delta_percentage_points"], language+": report WER point delta")
        check.equal(test_report["cer_delta_percentage_points"], bootstrap["cer"]["delta_percentage_points"], language+": report CER point delta")
        check.equal(test_report["paired_bootstrap"], bootstrap, language+": report bootstrap data")
        heldout_passes[language] = test_metrics["wer"] < original_test["wer"] if language == "tr" else test_metrics["wer"]-original_test["wer"] <= max(.005,.1*original_test["wer"])
        expected_pass = test_report.get("heldout_retention_gate_passed")
        if language != "tr" or expected_pass is not None:
            check.equal(expected_pass, heldout_passes[language], language+": report heldout quality gate calculation")
        results[language] = {"development_asr": dev_metrics, "test_asr": test_metrics, "test_comparison": bootstrap, "heldout_quality_gate_passed": heldout_passes[language], "temperature": fitted["temperature"]}
    check.require(not all_dev_sentences.intersection(all_test_sentences), "global sentence IDs separate development calibration from heldout test")
    check.equal(report["confidence_calibration"], calibration, "report confidence calibration dictionary")
    smoke_evidence = verify_transcription_smoke(report,run_dir,adapted_folder,calibration,check)
    gate = {"passed": True, "retention_gate_passed": True, "turkish_improved": results["tr"]["development_asr"]["wer"] < baseline_summary["validation_tr"]["asr"]["wer"],
            "turkish_wer": results["tr"]["development_asr"]["wer"], "baseline_turkish_wer": baseline_summary["validation_tr"]["asr"]["wer"],
            "retention_wer_deltas": {}, "retention_wer_tolerances": {},
            "policy": "each dev WER <= base + max(0.5 percentage point,10% relative base WER); Turkish WER improves"}
    for language in LANGUAGES:
        if language != "tr":
            base_wer = baseline_summary["validation_"+language]["asr"]["wer"]
            gate["retention_wer_deltas"][language] = results[language]["development_asr"]["wer"]-base_wer
            gate["retention_wer_tolerances"][language] = max(.005,.1*base_wer)
    gate["retention_gate_passed"] = all(gate["retention_wer_deltas"][language] <= gate["retention_wer_tolerances"][language] for language in gate["retention_wer_deltas"])
    gate["passed"] = gate["retention_gate_passed"] and gate["turkish_improved"]
    check.require(gate["passed"], "reloaded merged model passes development selection gates")
    check.equal(gate, selection["merged_development_gate"], "final selection development gate independently reproduced")
    check.equal(gate, json_load(run_dir / "merged_development_gate.json"), "durable merged development gate independently reproduced")
    check.equal(gate, report["development_gate"], "report development gate independently reproduced")
    history = json_load(run_dir / "training_history.json")
    check.require(bool(history) and [row["step"] for row in history] == list(range(1,len(history)+1)), "training history contiguous completed optimizer steps")
    check.equal(len(history),config["steps"],"finite completion includes every originally configured optimizer step")
    check.require(all(math.isfinite(row[key]) for row in history for key in ("tdt_loss","replay_kl","replay_duration_kl","replay_feature_distillation","gradient_norm")), "training history losses/KLs/gradient norms finite")
    check.equal(report["completed_steps"], history[-1]["step"], "report completed steps")
    check.equal(report["configured_steps"], config["steps"], "report originally configured steps")
    eligible = [row for row in history if row.get("validation", {}).get("retention_gate_passed") and row["validation"]["turkish_wer"] < baseline_summary["validation_tr"]["asr"]["wer"]]
    check.require(bool(eligible), "training history contains a qualified dev checkpoint")
    check.equal(min(eligible,key=lambda row: row["validation"]["turkish_wer"])["step"], selected_step, "selected checkpoint is best Turkish dev WER within recorded retention gates")
    goal_met = all(heldout_passes.values())
    check.require("pilot_improvement_and_retention_gates_passed" in report, "report contains explicit pilot comparison-gate outcome")
    check.equal(report["pilot_improvement_and_retention_gates_passed"], goal_met, "report separates pilot comparison gates from workflow completion/useful Turkish quality")
    return {"run_id":run_id,"selected_step":selected_step,"checkpoint":str(checkpoint),"adapted_results_path":str(adapted_folder),"utterances":900,
            "languages":results,"pilot_improvement_and_retention_gates_passed":goal_met,"heldout_quality_gates":heldout_passes,
            "transcription_smoke":smoke_evidence,
            "scope":"Fixed small pilot: 50 development +100 heldout recordings per language; optional full FLEURS split not run by this auditor",
            "notes":["Calibration is independently reproduced from development only and then applied unchanged to heldout alignments.",
                     "Checkpoint files, complete saved FP32 tensor state, tokenizer/config/generation identity and selected-state/base/code hashes independently verified on CPU.",
                     "No fresh model inference is performed. Consistency audit completion is separate from Turkish/retention quality-goal attainment."]}


def write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--report",type=Path,required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}",args.run_id):
        parser.error("run-id must be a short filename-safe identifier")
    check = Checks()
    result = {"verified_at_utc":datetime.now(timezone.utc).isoformat(),"device":"CPU only; no GPU inference","run_id":args.run_id,"report_path":str(args.report.resolve())}
    try:
        report_path = local_path(args.report,check,"report")
        result.update(audit(args.run_id,report_path,check))
        result["passed"] = True
        result["report_sha256_at_audit"] = file_sha(report_path)
        result["audited_report_content_sha256"] = json_sha({key:value for key,value in json_load(report_path).items() if key not in ("status","verification")})
        result["audited_report_hash_policy"] = "Evidence payload excludes only worker status/verification fields, which change from verification_pending to complete after this audit. Full report SHA above records the exact pre-audit snapshot."
        exit_code = 0
    except Exception as error:
        result.update({"passed":False,"failed_check":str(error),"error_type":type(error).__name__})
        exit_code = 1
    result["checks"] = check.items
    result["passed_check_count"] = len(check.items)
    for path in (OUTPUT / "runs" / args.run_id / "final_results_verification.json", OUTPUT / "final_results_verification.json"):
        write_result(path,result)
    print(json.dumps({key:result[key] for key in ("passed","run_id","passed_check_count")},indent=2))
    if not result["passed"]:
        print("FINAL RESULTS AUDIT FAILED: "+result["failed_check"],file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
