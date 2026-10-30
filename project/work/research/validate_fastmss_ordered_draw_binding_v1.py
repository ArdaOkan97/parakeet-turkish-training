"""Small CPU-only production draw-binding fixture; no decoder/model/GPU imports."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

from fastmss_cpu_guard_v1 import Guard, output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = output_path(args.output)
    guard = Guard(wall_seconds=90)
    source = Path(__file__).resolve().parents[2] / "outputs/parakeet_turkish"
    sys.path.insert(0, str(source))
    import numpy as np
    from fastmss_train_augmentation_v1 import AugmentationPolicy, TrainWaveformAugmenter, canonical
    from fastmss_ordered_draw_binding_v1 import AugmentationDrawContext as Draw, FastMSSOrderedDrawBinding
    checks = {}

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks[name] = True

    def rejects(name, function):
        try:
            function()
        except (ValueError, TypeError):
            checks[name] = True
        else:
            raise AssertionError(name)

    pilot = source / "fastmss_train_augmentation_v1.pilot.disabled.json"
    policy_value = json.loads(pilot.read_text())
    policy_value.update(enabled=True, apply_probability=1.0, noise_probability=1.0, rir_probability=1.0)
    policy_path = out / "enabled_all_on_fixture_policy.json"
    policy_path.write_text(json.dumps(policy_value, indent=2), encoding="utf-8")
    policy = AugmentationPolicy.from_json(policy_path)
    augmenter = TrainWaveformAugmenter(policy, guard=guard.check)
    binding = FastMSSOrderedDrawBinding(augmenter, policy_path=policy_path,
                                        adapter_path=source / "fastmss_train_augmentation_v1.py")
    original_identity = binding.identity
    identity_copy = binding.identity
    identity_copy["policy_sha256"] = "tamper"
    check("identity_copy_cannot_mutate_binding", binding.identity == original_identity)
    waveform = (0.2 * np.sin(np.arange(16000, dtype=np.float32) * (2 * np.pi * 220 / 16000))).astype(np.float32)
    waveform_before = waveform.tobytes()
    rows = [{"source_id": "duplicate_TR_source", "language": "tr"},
            {"source_id": "duplicate_TR_source", "language": "tr"},
            {"source_id": "PT_replay", "language": "pt"}, {"source_id": "EN_replay", "language": "en"}]
    contexts = tuple(Draw(row["source_id"], row["language"], epoch=1, draw_index=17 + i) for i, row in enumerate(rows))
    prepared = []
    reader = binding.prepare_batch(lambda row: waveform, contexts, batch_id="update64251/microbatch0",
                                   record_prepared=prepared.append)
    rejects("incomplete_batch_cannot_prepare_commit_payload", reader.prepared_payload)
    first_outputs = [reader(row) for row in rows]
    payload = reader.prepared_payload()
    check("all_receipts_are_prepared_not_committed", len(prepared) == 4 and
          all(v["state"] == "prepared_not_committed" for v in prepared) and
          payload["committed_frontier_advanced"] is False)
    check("duplicate_source_IDs_have_distinct_durable_draws", not np.array_equal(first_outputs[0], first_outputs[1]) and
          prepared[0]["draw"]["draw_index"] == 17 and prepared[1]["draw"]["draw_index"] == 18)
    check("PT_EN_replay_exact", np.array_equal(first_outputs[2], waveform) and np.array_equal(first_outputs[3], waveform))
    check("clean_source_unchanged", waveform.tobytes() == waveform_before)
    rejects("consumed_batch_cannot_read_extra_row", lambda: reader(rows[0]))
    prepared[0]["augmentation"]["applied"] = "external_mutation"
    check("external_callback_cannot_mutate_internal_receipts", reader.prepared_payload() == payload)
    fresh_augmenter = TrainWaveformAugmenter(policy, guard=guard.check)
    fresh = FastMSSOrderedDrawBinding(fresh_augmenter, policy_path=policy_path,
                                     adapter_path=source / "fastmss_train_augmentation_v1.py")
    reconstructed = fresh.prepare_batch(lambda row: waveform, contexts, batch_id="update64251/microbatch0",
                                        record_prepared=lambda value: None)
    new_outputs = [reconstructed(row) for row in rows]
    check("fresh_binding_reconstructs_draw_waveforms_and_receipts_exactly",
          all(np.array_equal(a, b) for a, b in zip(first_outputs, new_outputs)) and reconstructed.prepared_payload() == payload)
    wrong = binding.prepare_batch(lambda row: waveform, contexts, batch_id="wrong_order", record_prepared=lambda value: None)
    rejects("wrong_source_order_rejected", lambda: wrong(rows[2]))
    rejects("failed_batch_cannot_prepare_commit_payload", wrong.prepared_payload)
    rejects("failed_batch_cannot_silently_resume", lambda: wrong(rows[0]))
    rejects("duplicate_durable_position_rejected", lambda: binding.prepare_batch(lambda row: waveform,
        [contexts[0], contexts[0]], batch_id="duplicate_position", record_prepared=lambda value: None))
    rejects("DEV_cannot_construct_training_context", lambda: replace(contexts[0], split="dev"))
    rejects("TEST_cannot_construct_training_context", lambda: replace(contexts[0], split="test"))
    rejects("negative_draw_rejected", lambda: replace(contexts[0], draw_index=-1))
    rejects("bool_draw_not_integer_position", lambda: replace(contexts[0], draw_index=True))
    # Two prefetched batches are deliberately interleaved: contexts stay per batch.
    one = binding.prepare_batch(lambda row: waveform, contexts[:2], batch_id="prefetchA", record_prepared=lambda value: None)
    two_contexts = tuple(replace(v, draw_index=v.draw_index + 100) for v in contexts[:2])
    two = binding.prepare_batch(lambda row: waveform, two_contexts, batch_id="prefetchB", record_prepared=lambda value: None)
    a0 = one(rows[0]); b0 = two(rows[0]); a1 = one(rows[1]); b1 = two(rows[1])
    check("prefetched_batches_do_not_share_mutable_draw_pointer",
          np.array_equal(a0, first_outputs[0]) and np.array_equal(a1, first_outputs[1]) and
          not np.array_equal(a0, b0) and not np.array_equal(a1, b1))
    clean_context = replace(contexts[0], phase="cleaner_data_fine_tuning")
    clean_reader = binding.prepare_batch(lambda row: waveform, [clean_context], batch_id="clean_phase", record_prepared=lambda value: None)
    check("clean_finetune_default_preserved", clean_reader(rows[0]) is waveform)
    # The helper binds caller-owned journal prefix, not its uncommitted tail.
    prefix = canonical(payload) + b"\n"
    tail = b'{"state":"prepared_only_future_batch"}\n'
    journal_path = out / "caller_owned_journal.jsonl"
    journal_path.write_bytes(prefix + tail)
    journal = {"path": str(journal_path), "offset": len(prefix), "sha256": hashlib.sha256(prefix).hexdigest()}
    state = binding.checkpoint_binding(committed_journal=journal)

    def verify_prefix(value):
        with Path(value["path"]).open("rb") as stream:
            raw = stream.read(value["offset"])
        return hashlib.sha256(raw).hexdigest() == value["sha256"]

    restored_identity = fresh.verify_restore(state, verify_journal_prefix=verify_prefix)
    check("restore_binds_checkpoint_prefix_and_ignores_uncommitted_tail", restored_identity == original_identity)
    tampered = json.loads(canonical(state))
    tampered["identity"]["registry_sha256"] = "0" * 64
    rejects("registry_binding_mismatch_rejected", lambda: fresh.verify_restore(tampered, verify_journal_prefix=verify_prefix))
    tampered = json.loads(canonical(state))
    tampered["identity"]["policy_sha256"] = "0" * 64
    rejects("policy_binding_mismatch_rejected", lambda: fresh.verify_restore(tampered, verify_journal_prefix=verify_prefix))
    tampered = json.loads(canonical(state))
    tampered["identity"]["numpy_version"] = "wrong_version"
    rejects("environment_version_mismatch_rejected", lambda: fresh.verify_restore(tampered, verify_journal_prefix=verify_prefix))
    rejects("bad_committed_prefix_rejected", lambda: fresh.verify_restore(state, verify_journal_prefix=lambda _: False))
    rejects("nonboolean_prefix_verifier_rejected", lambda: fresh.verify_restore(state, verify_journal_prefix=lambda _: "yes"))
    rejects("future_journal_offset_rejected", lambda: fresh.checkpoint_binding(committed_journal={**journal, "offset": 99999999}))
    rejects("wrong_actual_policy_file_rejected", lambda: FastMSSOrderedDrawBinding(augmenter,
        policy_path=pilot, adapter_path=source / "fastmss_train_augmentation_v1.py"))
    rejects("wrong_actual_adapter_path_rejected", lambda: FastMSSOrderedDrawBinding(augmenter,
        policy_path=policy_path, adapter_path=__file__))
    check("source_default_configs_still_disabled", not json.loads(pilot.read_text())["enabled"] and
          not json.loads((source / "fastmss_train_augmentation_v1.disabled.json").read_text())["enabled"])
    check("Torch_never_imported", "torch" not in sys.modules)
    guard.check()
    report = {"status": "passed", "checks_passed": len(checks), "checks": checks,
              "binding_identity": binding.identity, "checkpoint_contract": state,
              "fixture_only": True, "native_new_production_caller_validation": False,
              "sampler_optimizer_RNG_frontier_advanced": False, "production_enabled": False,
              "resources": guard.report()}
    path = out / "ordered_draw_binding_CPU_validation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "checks": len(checks), "report": str(path), "resources": report["resources"]}))
    guard.close()


if __name__ == "__main__":
    main()
