"""Clean TRAIN feature preparation contract; no cache lifecycle or native work.

This NEW hook is deliberately separate from the frozen weak-label feature
window. An admitted CPU child supplies the original frontend and a bounded,
owned feature sink. This module calls only the frozen direct clean reader.
Preparation cannot consume a sampler draw, journal prefix or optimizer step.
No numerical/frontend library is imported until a row is actually prepared.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

from fastmss_clean_phase_reader_v1 import CleanPhaseOrderedAudioReader, KIND as PROTECTION_KIND, PHASE, row_contract, waveform_hash
from fastmss_feature_window_v1 import CACHE, CAPACITY, FRONTEND_ARGS, MAX_METADATA, record, read_bound, scoped
from fastmss_ordered_draw_binding_v1 import AugmentationDrawContext, file_binding
import fastmss_ordered_draw_binding_v1 as draw_helper
import fastmss_train_augmentation_v1 as adapter
from fastmss_train_augmentation_v1 import AugmentationPolicy, UPSTREAM, canonical, require

KIND = "FastMSS_clean_feature_update_request_v1"
PROOF_KIND = "FastMSS_independent_clean_feature_audio_references_v1"
POPULATION = 352768
FULL_TARGET_TOKENS = 11972886


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def target_hash(ids, language):
    ceiling = 8711 if language == "tr" else 8193
    require(type(ids) is list and ids and all(type(v) is int and 0 <= v < ceiling and v != 8192 for v in ids),
            "Every complete clean target ID must match the phase vocabulary; no cap or prefix")
    value = hashlib.sha256()
    for item in ids:
        value.update(struct.pack("<I", item))
    return value.hexdigest()


def metadata_clean_binding(identity, guard):
    """Verify frozen policy metadata without constructing or invoking an augmenter.

    The namespace satisfies the frozen clean wrapper's metadata-only interface.
    Numerical package provenance remains the producer's original identity.
    """
    expected_keys = {"kind", "adapter", "policy_file", "binding_helper", "policy_sha256", "registry_sha256",
                     "numpy_version", "scipy_version", "draw_owner", "commit_owner"}
    require(type(identity) is dict and set(identity) == expected_keys and
            identity["kind"] == "FastMSS_production_draw_binding_v1", "Complete frozen broad identity required")
    require(identity["adapter"] == file_binding(adapter.__file__) and
            identity["binding_helper"] == file_binding(draw_helper.__file__) and
            record(identity["policy_file"]["path"], guard) == identity["policy_file"],
            "Frozen adapter/helper/policy source changed")
    policy = AugmentationPolicy.from_json(identity["policy_file"]["path"])
    require(policy.phases == ("weak_label_continued_training",) and policy.sample_rate == 16000,
            "Cleaner must preserve the existing broad-only augmentation policy")
    registry_sha = None
    if policy.enabled and policy.registry_path is not None:
        registry = scoped(policy.registry_path)
        require(registry.stat().st_size <= 256 * 1024, "Bounded registry metadata only")
        registry_sha = record(registry, guard)["sha256"]
    require(registry_sha == identity["registry_sha256"] and
            digest({"adapter": 1, "policy": asdict(policy), "registry_sha256": registry_sha,
                    "upstream_revision": UPSTREAM}) == identity["policy_sha256"],
            "Exact frozen policy/registry digest differs")
    return SimpleNamespace(identity=json.loads(canonical(identity)), augmenter=SimpleNamespace(policy=policy))


def validate_frontend(frontend, guard):
    require(type(frontend) is dict and set(frontend) == {"Torch", "torch_threads", "arguments", "source",
            "processor_config", "base_path", "CPU_GPU_byte_equivalence_proof"} and
            frontend["Torch"] == "2.7.1+cpu" and frontend["torch_threads"] == 4 and
            frontend["arguments"] == FRONTEND_ARGS, "Exact original CPU frontend contract required")
    for name in ("source", "processor_config"):
        require(record(frontend[name]["path"], guard) == frontend[name], "Original frontend source/config differs")
    scoped(frontend["base_path"], existing_input=True)
    require(read_bound(frontend["CPU_GPU_byte_equivalence_proof"], guard)["passed"] is True,
            "Actual original complete-wave frontend byte-equivalence proof required")


def _verify_file(binding, guard):
    require(type(binding) is dict and set(binding) == {"path", "bytes", "sha256", "mtime_ns"} and
            type(binding["bytes"]) is int and binding["bytes"] > 0 and type(binding["mtime_ns"]) is int,
            "Exact independent original source/cache file identity required")
    path = scoped(binding["path"], existing_input=True)
    require(record(path, guard) == {k: binding[k] for k in ("path", "bytes", "sha256")} and
            path.stat().st_mtime_ns == binding["mtime_ns"], "Independent original source/cache changed")


def validate_request(request, guard):
    """Read only bounded metadata. No source audio, frontend, sampler or ML load."""
    fields = {"kind", "phase", "update_index", "batch_id", "rows", "contexts", "ordered_draws",
              "independent_clean_audio_proof", "independent_clean_audio_actual_closure", "cleaner_catalog_identity", "augmentation_identity", "frontend",
              "native_transaction_sha256", "before_sampler_sha256", "after_sampler_sha256",
              "prepared_is_committed", "sampler_journal_frontier_advanced", "feature_sink_source", "feature_cache_owner"}
    require(type(request) is dict and set(request) == fields and request["kind"] == KIND and request["phase"] == PHASE and
            type(request["update_index"]) is int and 0 <= request["update_index"] < 20 and
            isinstance(request["batch_id"], str) and 0 < len(request["batch_id"]) <= 200 and
            request["prepared_is_committed"] is False and request["sampler_journal_frontier_advanced"] is False,
            "Explicit provisional cleaner update required; no weak default or actual commit")
    require(len(canonical(request)) <= MAX_METADATA, "One bounded request, never a complete audio/feature pool")
    require(all(sha(request[k]) for k in ("native_transaction_sha256", "before_sampler_sha256", "after_sampler_sha256")),
            "Exact caller-owned transaction and before/after sampler hashes required")
    rows, contexts, draws = request["rows"], request["contexts"], request["ordered_draws"]
    require(type(rows) is list and type(contexts) is list and type(draws) is list and 3 <= len(rows) <= 16 and
            len(rows) == len(contexts) == len(draws), "One full1..14TR+2carried-replay update required")
    contexts = tuple(AugmentationDrawContext(**v) for v in contexts)
    require(len({(v.epoch, v.draw_index) for v in contexts}) == len(rows) and
            all(v.phase == PHASE and v.split == "train" for v in contexts), "Unique exact cleaner TRAIN contexts required")
    require(all(v["language"] == "tr" for v in rows[:-2]) and rows[-2]["language"] in ("en", "es", "it", "fr", "de") and
            rows[-1]["language"] == "pt", "All original clean TR rows plus original old-five/PT carried replay required")
    catalog = request["cleaner_catalog_identity"]
    require(catalog["kind"] == "ASR18_cleaner352768_BPE_catalog_binding_v1" and catalog["rows"] == POPULATION and
            catalog["full_target_tokens"] == FULL_TARGET_TOKENS and
            catalog["population_mean_U"] == FULL_TARGET_TOKENS / POPULATION and
            catalog["quality_or_language_selection_added"] is False and catalog["augmentation_phase"] == PHASE,
            "Exact existing full cleaner pool/token denominator required")
    proof = read_bound(request["independent_clean_audio_proof"], guard)
    require(proof["kind"] == PROOF_KIND and proof["passed"] is True and proof["phase"] == PHASE and
            proof["cleaner_catalog_identity"] == catalog and proof["actual_optimizer_updates"] == 0 and
            proof["sampler_journal_frontier_advanced"] is False and len(proof["references"]) == len(rows),
            "Complete independently produced clean PCM/full-target reference proof required")
    closure = read_bound(request["independent_clean_audio_actual_closure"], guard)
    require(closure["kind"] == "FastMSS_independent_clean_audio_reference_actual_closure_v1" and
            closure["report"] == request["independent_clean_audio_proof"] and closure["actual_exit_code"] == 0 and
            closure["parent_observed_wait"] is True and type(closure["native_pid"]) is int and closure["native_pid"] > 0,
            "Independent reference producer must actually close0 before feature preparation")
    monitor = read_bound(closure["guard_closure"], guard)
    require(monitor["pid"] == closure["native_pid"] and monitor["actual_exit_code"] == 0 and
            monitor["monitor_clean"] is True and monitor["monitor_joined"] is True and
            monitor["observed"]["samples"] >= 3 and monitor["observed"]["monitor_error"] is None,
            "Actual clean reference closure requires clean joined guarded producer proof")
    binding = metadata_clean_binding(request["augmentation_identity"], guard)
    validate_frontend(request["frontend"], guard)
    require(record(request["feature_sink_source"]["path"], guard) == request["feature_sink_source"], "Admitted bounded sink source differs")
    owner = read_bound(request["feature_cache_owner"], guard)
    folder = scoped(request["feature_cache_owner"]["path"]).parent
    require(folder.parent == CACHE.resolve() and Path(request["feature_cache_owner"]["path"]).name == "owner.json" and
            owner["kind"] == "FastMSS_owned_clean_provisional_feature_window_v1" and owner["phase"] == PHASE and
            owner["source"] == request["feature_sink_source"] and owner["producer_contract"] == file_binding(__file__) and
            owner["capacity_bytes"] == CAPACITY and type(owner["updates"]) is int and 1 <= owner["updates"] <= 20 and
            request["update_index"] < owner["updates"] and type(owner["owned_names"]) is list and
            0 < len(owner["owned_names"]) == len(set(owner["owned_names"])) <= 704 and
            all(type(v) is str and Path(v).name == v and v not in (".", "..") for v in owner["owned_names"]) and
            owner["prepared_is_committed"] is False and owner["sampler_journal_frontier_advanced"] is False,
            "Exact separate clean owner and shared global512MiB feature budget required")
    for row, context, draw, reference in zip(rows, contexts, draws, proof["references"]):
        require(reference["source_id"] == row["source_id"] == context.source_id == draw["source_id"] and
                reference["language"] == row["language"] == context.language == draw["language"] and
                reference["context"] == asdict(context) == row["audio_draw_context"] and
                all(draw[k] == getattr(context, k) for k in ("epoch", "draw_index", "phase")) and
                reference["source_row_sha256"] == digest(row), "Exact row/order/raw metadata/phase/draw binding differs")
        expected = reference["full_target_ids"]
        require(target_hash(expected, context.language) == reference["target_ids_u32_sha256"], "Full independent target IDs differ")
        row_contract(row, context, expected, reference["expected_audio"])
        require(scoped(reference["source_audio_file"]["path"], existing_input=True) == scoped(row["audio_filepath"], existing_input=True) and
                reference["full_clean_original_audio"] is True and reference["augmentation"] is False,
                "Direct original full clean source required")
        if context.language == "tr":
            require(type(row["cleaner_original_ordinal"]) is int and 1 <= row["cleaner_original_ordinal"] <= POPULATION and
                    draw["ordinal"] == row["cleaner_original_ordinal"] == reference["cleaner_original_ordinal"] and
                    row["augmentation_role"] == "student_tr" and reference["genuine_teacher_feature_cache"] is None,
                    "Exact full cleaner ordinal required; TR has no teacher cache")
        else:
            require(row["augmentation_role"] == "replay" and reference["cleaner_original_ordinal"] is None and
                    type(reference["genuine_teacher_feature_cache"]) is dict,
                    "Original genuine clean replay cache binding required")
    return binding, contexts, proof, owner, folder


class CleanFeatureUpdateProducer:
    """Serial row hook. Caller owns bounded files/actual-child-close/commit lifecycle.

    frontend(waveform) must return CPU NumPy input_features float32[1,T,128]
    and attention_mask bool[1,T] using the bound original frontend defaults.
    save_array(local_row,key,array) returns the existing feature file/shape/dtype/
    array_sha256 schema. capacity_guard(additional_bytes) must verify the actual
    whole-window512MiB budget, not just this update. No waveform is retained.
    """
    def __init__(self, request, *, clean_reader, frontend, save_array, capacity_guard, guard):
        require(len(canonical(request)) <= MAX_METADATA, "Bounded clean update request required")
        self._request = json.loads(canonical(request))
        binding, contexts, proof, self._owner, self._folder = validate_request(self._request, guard)
        require(all(callable(v) for v in (clean_reader, frontend, save_array, capacity_guard, guard)), "Explicit bounded CPU hooks required")
        self._contexts, self._proof = contexts, proof
        self._guard, self._frontend, self._save, self._capacity = guard, frontend, save_array, capacity_guard
        self._reader = CleanPhaseOrderedAudioReader(binding, self._request["rows"], contexts,
            batch_id=self._request["batch_id"], clean_reader=clean_reader,
            expected_target_ids=[v["full_target_ids"] for v in proof["references"]],
            expected_audio=[v["expected_audio"] for v in proof["references"]], guard=guard, record_prepared=lambda value: None)
        self._cursor, self._failed, self._entries, self._paths = 0, False, [], set()

    def prepare_next(self, row_index):
        import numpy as np
        try:
            require(not self._failed and type(row_index) is int and row_index == self._cursor < len(self._contexts),
                    "Clean feature row reused/out of order; no silent skip")
            self._guard()
            row, context, ref = self._request["rows"][row_index], self._contexts[row_index], self._proof["references"][row_index]
            _verify_file(ref["source_audio_file"], self._guard)
            if row["language"] != "tr":
                _verify_file(ref["genuine_teacher_feature_cache"], self._guard)
            wave = self._reader(row)  # Direct original reader, exact same object/bytes.
            raw_sha = waveform_hash(wave, self._guard)
            values = self._frontend(wave)
            require(type(values) is dict and set(values) == {"input_features", "attention_mask"}, "Original full frontend keys required")
            features, mask = values["input_features"], values["attention_mask"]
            require(isinstance(features, np.ndarray) and features.dtype == np.float32 and features.ndim == 3 and
                    features.shape[0] == 1 and features.shape[2] == 128 and features.flags.c_contiguous and np.isfinite(features).all() and
                    isinstance(mask, np.ndarray) and mask.dtype == np.bool_ and mask.shape == features.shape[:2] and mask.flags.c_contiguous and
                    mask.any() and np.array_equal(mask, np.arange(mask.shape[1])[None, :] < mask.sum(1)[:, None]) and
                    not features[~mask].any() and int((int(mask.sum()) + 7) // 8) == row["encoder_frames"] and
                    waveform_hash(wave, self._guard) == raw_sha, "Original clean buffer/features/mask/full encoder geometry changed")
            needed = features.nbytes + mask.nbytes + 2048
            require(self._capacity(needed) is True, "Actual entire-window bounded capacity admission required")
            entry = {"context": asdict(context), "source_id": row["source_id"], "language": row["language"],
                "role": row["augmentation_role"], "cleaner_original_ordinal": row.get("cleaner_original_ordinal"),
                "raw_text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
                "full_target_ids": list(ref["full_target_ids"]), "target_ids_u32_sha256": ref["target_ids_u32_sha256"],
                "target_tokens": len(ref["full_target_ids"]), "encoder_frames": row["encoder_frames"],
                "sample_rate": 16000, "samples": len(wave), "clean_PCM_sha256": raw_sha, "prepared_PCM_sha256": raw_sha}
            for name, array in (("input_features", features), ("attention_mask", mask)):
                saved = self._save(row_index, name, array)
                require(type(saved) is dict and set(saved) == {"file", "shape", "dtype", "array_sha256"}, "Exact owned feature sink result required")
                path = scoped(saved["file"]["path"])
                require(path.parent == self._folder and path.name in self._owner["owned_names"] and path.suffix == ".npy" and
                        str(path) not in self._paths and record(path, self._guard) == saved["file"] and
                        saved["shape"] == list(array.shape) and saved["dtype"] == str(array.dtype) and
                        saved["array_sha256"] == hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
                        "Bounded owned sink file/array identity differs")
                actual = np.load(path, allow_pickle=False)
                require(actual.flags.c_contiguous and actual.dtype == array.dtype and actual.shape == array.shape and
                        hashlib.sha256(memoryview(actual).cast("B")).hexdigest() == saved["array_sha256"],
                        "Actual saved NumPy feature bytes differ from original frontend output")
                del actual
                self._paths.add(str(path)); entry[name] = saved
            require(self._capacity(0) is True, "Whole-window capacity changed during clean feature write")
            self._entries.append(canonical(entry)); self._cursor += 1
            return json.loads(self._entries[-1])
        except BaseException:
            self._failed = True
            raise

    def prepared_update(self):
        require(not self._failed and self._cursor == len(self._contexts), "Incomplete/failed clean feature update cannot publish")
        payload = self._reader.prepared_payload()
        require(payload["protection_kind"] == PROTECTION_KIND and payload["committed_frontier_advanced"] is False and
                all(v["augmentation"]["applied"] is False and v["draw"]["phase"] == PHASE for v in payload["rows"]),
                "Every clean source/replay receipt must bypass augmentation")
        request = self._request
        result = {"kind": "FastMSS_provisional_clean_feature_update_v1", "phase": PHASE,
            "owner": request["feature_cache_owner"],
            "update_index": request["update_index"], "batch_id": request["batch_id"],
            "native_transaction_sha256": request["native_transaction_sha256"], "before_sampler_sha256": request["before_sampler_sha256"],
            "after_sampler_sha256": request["after_sampler_sha256"], "rows": [json.loads(v) for v in self._entries],
            "prepared_payload": payload, "frontend": request["frontend"], "augmentation_identity": request["augmentation_identity"],
            "input_evidence": {"independent_clean_audio_proof": request["independent_clean_audio_proof"],
                "independent_clean_audio_actual_closure": request["independent_clean_audio_actual_closure"],
                "independent_clean_audio_proof_contents": self._proof, "cleaner_catalog_identity": request["cleaner_catalog_identity"],
                "contract_source": file_binding(__file__), "feature_sink_source": request["feature_sink_source"],
                "augmentation_or_weak_PCM_transport_invoked": False, "original_teacher_features_copied": False},
            "prepared_is_committed": False, "sampler_journal_frontier_advanced": False,
            "actual_CPU_child_exit_observed": False, "native_consumer_must_require_independent_actual_child_closure": True}
        require(len(canonical(result)) <= MAX_METADATA and self._capacity(len(canonical(result))) is True,
                "Full clean metadata must fit the same actual window capacity")
        return json.loads(canonical(result))
