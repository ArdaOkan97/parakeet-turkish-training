"""One CPU-only original Parakeet frontend update, never augmentation or Adam."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import struct
import sys
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "outputs/parakeet_turkish"
TRAINING_SITE = PROJECT / "work/training-env/Lib/site-packages"
CPU_ENV = Path(r"E:\gpu_training_scratchpad\scratch\BPE_broad_CPU_preflight_env_v2")
sys.path.insert(0, str(SOURCE))
from fastmss_feature_window_v1 import (ROOT, FRONTEND_ARGS, begin_update, finish_update, load_plan,
    read_bound, record, digest, scoped, require)
from fastmss_ordered_draw_binding_v1 import AugmentationDrawContext, FastMSSOrderedDrawBinding
from fastmss_train_augmentation_v1 import AugmentationPolicy, TrainWaveformAugmenter, canonical
from fastmss_cpu_prepared_transport_v1 import VerifiedPreparedAudioReader, waveform_hash


def target_hash(ids):
    value = hashlib.sha256()
    for item in ids:
        require(type(item) is int and 0 <= item < 8711 and item != 8192, "Full declared target IDs required")
        value.update(struct.pack("<I", item))
    return value.hexdigest()


def main(request_path, output):
    request = json.loads(scoped(request_path).read_bytes())
    require(request["kind"] == "FastMSS_CPU_feature_update_request_v1" and os.environ["CUDA_VISIBLE_DEVICES"] == "-1",
            "Explicit CPU-only feature request required")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        require(os.environ[key] == "1", "Four fixed library thread caps required")
    def guard():
        require(shutil.disk_usage(ROOT).free >= 100_000_000_000, "E reserve100GB required")
    owner, folder, update, active_record = begin_update(request["owner"], request["update_index"], request["prior_closures"], guard)
    plan = load_plan(owner["plan"], guard)
    # This import order is the previously measured CPU frontend environment.
    sys.path.insert(0, str(TRAINING_SITE))
    import numpy as np
    sys.path.remove(str(TRAINING_SITE))
    import torch
    require(torch.__version__ == "2.7.1+cpu" and torch.version.cuda is None and
            Path(torch.__file__).resolve().is_relative_to(CPU_ENV), "Pinned E CPU-only Torch runtime required")
    sys.path.append(str(TRAINING_SITE)); torch.set_num_threads(4)
    from transformers import AutoFeatureExtractor
    from resource_watchdog_v2 import sample_memory
    def checked_guard():
        guard()
        worker = sample_memory(os.getpid(), "bounded_CPU_feature_row")["worker_private_memory"]
        require(worker["available"] and worker["private_bytes"] < 2_000_000_000, "Fresh CPU private2GB guard required")
    frontend = owner["frontend"]
    require(record(frontend["processor_config"]["path"], checked_guard) == frontend["processor_config"], "Original frontend config changed")
    extractor = AutoFeatureExtractor.from_pretrained(frontend["base_path"], local_files_only=True)
    require(type(extractor).__name__ == "ParakeetFeatureExtractor" and record(inspect.getfile(type(extractor)), checked_guard) == frontend["source"] and
            frontend["arguments"] == FRONTEND_ARGS, "Exact original Parakeet frontend source/defaults required")
    proof = read_bound(frontend["CPU_GPU_byte_equivalence_proof"], checked_guard)
    require(proof["passed"] is True, "Actual original frontend byte-equivalence proof required")
    actual_frontend = {**frontend, "Torch_file": str(Path(torch.__file__).resolve()),
        "NumPy": np.__version__, "NumPy_file": str(Path(np.__file__).resolve()),
        "Torch_CUDA_runtime": None, "actual_torch_threads": torch.get_num_threads(), "ASR_model_loaded": False}
    decoder = read_bound(request["decoder_report"], checked_guard)
    require(decoder["passed"] and len(decoder["rows"]) == len(update["TR_draws"]), "Complete exact closed decoder rows required")
    transport = decoder["augmentation_transport"]
    pcm_manifest = read_bound(transport["manifest"], checked_guard)
    pcm_owner = read_bound(transport["owner"], checked_guard)
    pcm_closure = read_bound(request["PCM_actual_closure"], checked_guard)
    require(pcm_manifest["identity"] == owner["augmentation_identity"], "Prepared producer identity changed between updates")
    policy_path = plan["augmentation_policy"]["path"]
    require(record(policy_path, checked_guard) == plan["augmentation_policy"], "Pinned broad policy changed")
    policy = AugmentationPolicy.from_json(policy_path)
    augmenter = TrainWaveformAugmenter(policy, guard=checked_guard)
    actual_binding = FastMSSOrderedDrawBinding(augmenter, policy_path=policy_path,
        adapter_path=SOURCE / "fastmss_train_augmentation_v1.py")
    expected_identity = owner["augmentation_identity"]
    require({k:v for k,v in actual_binding.identity.items() if k not in ("numpy_version", "scipy_version")} ==
            {k:v for k,v in expected_identity.items() if k not in ("numpy_version", "scipy_version")},
            "Frozen producer policy/adapter/registry/helper identity differs")
    # Consume exact already-produced PCM. Frontend environment is separately
    # bound; producer numerical-library provenance is preserved, never relabeled.
    binding = SimpleNamespace(identity=expected_identity, augmenter=SimpleNamespace(policy=policy))
    raw_by_ordinal = {v["ordinal"]:v for v in plan["TR_rows"]}
    rows, targets = [], []
    for item, draw, identity in zip(decoder["rows"], update["TR_draws"], update["TR_identity"]):
        row = item["row"]; raw = row["broad_original"]; expected = raw_by_ordinal[draw["ordinal"]]
        require(row["durable_draw"] == draw and row["source_id"] == identity["source_id"] and
                all(expected[k] == v for k,v in raw.items()) and row["tr_metadata_labels"] == expected["BPE"]["target_ids"],
                "Plan source/full raw labels/draw differs from independent decoder")
        require(record(item["PCM_cache"]["path"], checked_guard) == item["PCM_cache"], "Independent clean PCM cache changed")
        rows.append(row); targets.append(row["tr_metadata_labels"])
    replay_proof = read_bound(request["replay_reference_report"], checked_guard) if update["replay_rows"] else None
    references = {}
    if replay_proof is not None:
        require(replay_proof["passed"] and replay_proof["plan"] == owner["plan"], "Replay reference belongs to another plan")
        references = {v["source_id"]:v for v in replay_proof["references"]}
    for row in update["replay_rows"]:
        reference = references[row["source_id"]]
        require(reference["source_row"] == row and reference["language"] == row["language"] and
                len(reference["full_target_ids"]) == row["target_tokens"] and
                target_hash(reference["full_target_ids"]) == reference["target_ids_u32_sha256"], "Exact original replay targets/source required")
        for key in ("audio_file", "genuine_teacher_feature_cache"):
            bound = reference[key]; path = scoped(bound["path"], existing_input=True)
            require(record(path, checked_guard) == {k:bound[k] for k in ("path", "bytes", "sha256")} and
                    path.stat().st_mtime_ns == bound["mtime_ns"], "Original replay audio/teacher cache changed")
        rows.append(row); targets.append(reference["full_target_ids"])
    contexts = tuple(AugmentationDrawContext(**v) for v in update["contexts"])
    require(len(rows) == len(contexts), "No omitted/extra feature rows")
    def replay_reader(row):
        import soundfile as sf
        reference = references[row["source_id"]]
        waveform, sr = sf.read(reference["audio_file"]["path"], dtype="float32", always_2d=False)
        audio = reference["expected_audio"]
        require(sr == audio["sample_rate"] == 16000 and waveform.ndim == 1 and len(waveform) == audio["samples"] and
                waveform_hash(waveform, checked_guard) == audio["waveform_sha256"], "Independent full clean replay PCM differs")
        return waveform
    reader = VerifiedPreparedAudioReader(binding, rows, contexts, batch_id=update["batch_id"],
        manifest_record=transport["manifest"], closure_record=request["PCM_actual_closure"], protected_reader=replay_reader,
        record_prepared=lambda value:None, guard=checked_guard,
        expected_clean_hashes=[v["reader_evidence"]["PCM_f32_sha256"] for v in decoder["rows"]])
    entries = []; prepared_by_draw = {(v["draw"]["epoch"],v["draw"]["draw_index"]):v for v in pcm_manifest["entries"]}
    for j, (row, context, ids) in enumerate(zip(rows, contexts, targets)):
        checked_guard(); waveform = reader(row); pcm_sha = waveform_hash(waveform, checked_guard)
        inputs = dict(extractor([waveform], **FRONTEND_ARGS))
        require(set(inputs) == {"input_features", "attention_mask"}, "Original frontend input keys changed")
        mask = inputs["attention_mask"].bool()
        require(torch.equal(mask, torch.arange(mask.shape[1])[None,:] < mask.sum(-1)[:,None]), "Original right-padded mask required")
        features = inputs["input_features"].masked_fill(~mask.unsqueeze(-1), 0).detach().cpu().contiguous().numpy()
        mask_array = mask.detach().cpu().contiguous().numpy()
        require(features.dtype == np.float32 and int((int(mask.sum())+7)//8) == row["encoder_frames"], "Full frontend/encoder frame identity differs")
        clean_sha = prepared_by_draw[(context.epoch,context.draw_index)]["clean_waveform_sha256"] if row["language"] == "tr" else pcm_sha
        entry = {"context": asdict(context), "source_id": row["source_id"], "language": row["language"],
            "role": row["augmentation_role"], "raw_text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
            "full_target_ids": list(ids), "target_ids_u32_sha256": target_hash(ids), "target_tokens": len(ids),
            "encoder_frames": row["encoder_frames"], "sample_rate":16000, "samples":len(waveform),
            "clean_PCM_sha256":clean_sha, "prepared_PCM_sha256":pcm_sha,
            "input_features": __import__("fastmss_feature_window_v1").save_array(owner,folder,f"u{request['update_index']:02}_r{j:02}_features.npy",features,checked_guard),
            "attention_mask": __import__("fastmss_feature_window_v1").save_array(owner,folder,f"u{request['update_index']:02}_r{j:02}_mask.npy",mask_array,checked_guard)}
        entries.append(entry)
        del waveform,inputs,mask,features,mask_array; gc.collect()
    payload = reader.prepared_payload()
    evidence = {"decoder_report": request["decoder_report"], "decoder_report_contents":decoder,
        "PCM_transport":transport, "PCM_manifest_contents":pcm_manifest, "PCM_owner_contents":pcm_owner,
        "PCM_actual_closure":request["PCM_actual_closure"], "PCM_actual_closure_contents":pcm_closure,
        "replay_reference_report":request.get("replay_reference_report"),
        "replay_references_used":[references[r["source_id"]] for r in update["replay_rows"]],
        "producer_numeric_environment":{k:expected_identity[k] for k in ("numpy_version","scipy_version")},
        "frontend_numeric_environment":{k:actual_binding.identity[k] for k in ("numpy_version","scipy_version")},
        "augmentation_reapplied":False, "original_teacher_features_copied":False}
    result = finish_update(request["owner"],request["update_index"],active_record,entries,payload,
        input_evidence=evidence,frontend=actual_frontend,guard=checked_guard)
    require(not any(v.startswith("nemo.collections.asr.models") for v in sys.modules), "No ASR model load")
    from fastmss_feature_window_v1 import write_new
    write_new(scoped(output), {"passed":True,"update_record":result,"actual_producer_PID":os.getpid(),
        "no_optimizer_or_sampler_updates":True,"CPU_only":True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser();parser.add_argument("--request",required=True);parser.add_argument("--output",required=True)
    arguments=parser.parse_args();main(arguments.request,arguments.output)
