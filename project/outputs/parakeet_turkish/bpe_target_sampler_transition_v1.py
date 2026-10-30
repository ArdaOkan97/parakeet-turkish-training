"""Inactive declared target-metadata migration for the existing TSC sampler.

This does not admit a model/checkpoint, run training, or claim exact ASR resume.
The future acquisition worker must first admit an actually CLOSED source and
fully annotate its existing TR pool with the checked candidate consumer.
Only the target-dependent TR fingerprint changes. Pending orders, cursors,
quota clocks, coverage and both sampler RNG streams retain their source state.
Future bucket sorts use the new target lengths, so future old-recipe ordering
is deliberately not claimed.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / "TSC_BPE512_TRAIN_target_metadata_preparation_v1.json"
REPORT_SHA = "91e0d70894b55c552e3cbb2046c5e22afde97509b163b2496cf1b237067ddb06"
PARENT_RUN = "adaptive_20261011_hf_TSC_headlr02_01"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


def preparation_for_source(source_config):
    """Bind the prepared target replacement to the source's existing inputs."""
    raw = REPORT.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == REPORT_SHA, "Frozen TRAIN preparation changed")
    report = json.loads(raw)
    require(source_config.get("run_id") == PARENT_RUN, "Only the reviewed head-LR lineage is supported")
    require(report.get("passed") is True and report.get("CPU_only") is True
            and report.get("candidate_activated") is False and report.get("clips") == 253798
            and report.get("clips_dropped_or_truncated") == 0,
            "Complete inactive TRAIN target preparation required")
    identity = source_config["config_identity"]
    pools = identity["domain_sampling_identity"]["source_group_pool_counts"]
    require(pools == {"cv": 72846, "fleurs": 2037, "tsc": 178915}, "Original full TSC pool required")
    for key, artifact in (("tr_manifest", report["manifest"]),
                          ("tr_frame_metadata", report["parent_metadata"]),
                          ("tr_frame_metadata_proof", report["source_proof"])):
        path = str(Path(source_config[key]).resolve())
        require(path == str(Path(artifact["path"]).resolve())
                and identity["input_sha256"].get(path) == artifact["sha256"],
                "Source input differs from prepared TRAIN targets: " + key)
    require(report["geometry"]["baseline_vocab"] == 8199
            and report["geometry"]["candidate_vocab"] == 8711
            and report["geometry"]["blank_id_unchanged"] == 8192,
            "Candidate lexical geometry differs")
    return report


def _identity_only(source_state, source_identity, candidate_identity):
    """Pure state migration; caller separately verifies the complete pool."""
    require(source_state.get("schema") == 3
            and source_state.get("kind") == "TSC_FLEURS30_CV_TSC70_independent_replay_sampler_v3"
            and source_state.get("identity") == source_identity,
            "Whole source sampler state and identity required")
    require(set(candidate_identity) == set(source_identity), "Sampler identity structure changed")
    changed = {key for key in source_identity if source_identity[key] != candidate_identity[key]}
    require(changed == {"turkish_pool_fingerprint"}, "Only the TR target fingerprint may change")
    for value in (source_identity["turkish_pool_fingerprint"], candidate_identity["turkish_pool_fingerprint"]):
        require(isinstance(value, str) and len(value) == 64
                and all(char in "0123456789abcdef" for char in value), "Invalid pool fingerprint")
    require(source_identity.get("sampling_mode") == "domain30"
            and source_identity.get("sampler_family") == "TSC_domain30_v3"
            and source_identity.get("batch_size") == 8
            and source_identity.get("accumulation") == 2
            and source_identity.get("replay_per_update") == 8
            and source_identity.get("replay_rng_independent_of_turkish") is True,
            "Existing domain30/B8/independent-replay recipe required")
    require(type(source_state.get("updates")) is int and source_state["updates"] >= 0,
            "Invalid source update frontier")
    before = canonical({k: v for k, v in source_state.items() if k != "identity"})
    migrated = deepcopy(source_state)
    migrated["identity"] = deepcopy(candidate_identity)
    require(canonical({k: v for k, v in migrated.items() if k != "identity"}) == before,
            "Non-identity sampler state changed")
    return migrated, before


def prepare_sampler_state(source_state, source_config, candidate_identity, acceptance):
    """Call after the candidate consumer verifies/annotates every accepted row.

    Acceptance values must come from the consumer's completed full streaming
    verification, not a partial iterator. The normal sampler.load_state_dict
    must still validate all orders, cursors, quotas and coverage before updates.
    """
    report = preparation_for_source(source_config)
    require(acceptance.get("all_rows_verified") is True
            and acceptance.get("verified_clips") == report["clips"]
            and acceptance.get("metadata_sha256") == report["metadata_sha256"],
            "Candidate consumer must verify the complete pool before migration")
    source_identity = source_config["config_identity"]["domain_sampling_identity"]
    migrated, non_identity_sha = _identity_only(source_state, source_identity, candidate_identity)
    evidence = {"kind": "declared_TSC_BPE_target_sampler_metadata_transition_v1",
                "source_run_id": PARENT_RUN, "source_boundary_step": source_state["updates"],
                "preparation_sha256": REPORT_SHA, "candidate_metadata_sha256": report["metadata_sha256"],
                "old_TR_fingerprint": source_identity["turkish_pool_fingerprint"],
                "new_TR_fingerprint": candidate_identity["turkish_pool_fingerprint"],
                "non_identity_state_sha256_unchanged": non_identity_sha,
                "sampler_RNG_quota_cursors_orders_coverage_preserved": True,
                "only_sampler_identity_field_changed": "turkish_pool_fingerprint",
                "future_buckets_use_candidate_target_lengths": True,
                "old_recipe_future_order_or_exact_ASR_resume_claimed": False,
                "actual_CLOSED_source_admission_verified_here": False,
                "actual_CLOSED_source_admission_required_before_updates": True,
                "sampler_native_load_still_required": True}
    return migrated, evidence
