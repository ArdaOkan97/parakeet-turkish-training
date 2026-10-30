"""Seal lightweight disabled design receipts; no archive or checkpoint writes."""
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OUT = Path(r'E:\gpu_training_scratchpad\analysis\FastMSS_integration_v1')


def pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def write(name, value):
    path = OUT / name
    assert not path.exists(), 'Never overwrite sealed evidence'
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    return pin(path)


def main():
    report = json.loads((OUT / 'packed_storage_sizing_actual_01/report.json').read_text())
    guard = json.loads((OUT / 'packed_storage_sizing_guard_01/closure.json').read_text())
    assert report['passed'] is True and report['check_count'] == 25
    assert guard['actual_exit_code'] == 0 and guard['monitor_clean'] is True and guard['monitor_joined'] is True
    assert guard['observed']['samples'] >= 3 and guard['observed']['monitor_error'] is None
    actual = write('BPE_packed_checkpoint_storage_sizing_actual_CLOSED0_v1.json', {
        'kind': 'ASR18_actual_owned_JSON_only_checkpoint_capacity_sizing_CLOSED0_v1',
        'actual_tool_chunk_id': 'b2c1bd', 'actual_tool_exit_code': 0,
        'native_pid': guard['pid'], 'native_exit_code': guard['actual_exit_code'],
        'private_peak_bytes': guard['observed']['peak_private_bytes'],
        'native_wall_seconds': guard['wall_seconds'], 'checks': report['check_count'],
        'guard': pin(OUT / 'packed_storage_sizing_guard_01/closure.json'),
        'report': pin(OUT / 'packed_storage_sizing_actual_01/report.json'),
        'tensor_payload_or_audio_read': False, 'archive_or_retirement_performed': False,
        'GPU_or_model_used': False, 'production_approval': False,
    })
    interface = write('BPE_packed_checkpoint_storage_capacity_interface_v1.json', {
        'kind': 'ASR18_future_owned_packed_checkpoint_storage_design_interface_v1',
        'production_enabled': False, 'implementation_status': 'capacity model and streaming sizing only; no catalog, publication or retirement backend',
        'proposal': pin(OUT / 'BPE_packed_checkpoint_capacity_PROPOSAL_NOT_LAUNCHABLE_v1.json'),
        'actual_small_metadata_measurement': actual,
        'capacity_arithmetic': report['disabled_capacity_arithmetic'],
        'actual_E_allocation_unit': report['actual_E_allocation_unit'],
        'measured_current_E_free_bytes': report['current_E_free_bytes'],
        'actual_small_JSON_pair_measurements': report['actual_small_JSON_pair_measurements'],
        'source_supported_caps_and_failure_rules': {
            'payload_pair_path': "metadataPath.with_suffix('.tensors.bin'); exact existing writer semantics retained",
            'maximum_publications': 401072,
            'derivation': '2*ceil(2631115/14)+ceil(352768/14), at most one payload publication per actual positive update; no minimum-window-cadence assumption',
            'earlier64KiB_pointer_cap_rejected_by_actual_sources': True,
            'candidate_caps_bytes': {'payload': 268435456, 'metadata': 2097152, 'pointer': 1048576},
            'complete_lossless_archive_record_caps_bytes': {'raw': 8388608, 'compressed': 524288},
            'complete_record_content': 'Exact original metadata and pointer bytes plus required immutable PRE-save/next-draw, source/history/DEV/winner/owner and retirement receipt bytes or independently retained immutable bindings. All per-publication records must fit the same declared compressed cap; no omitted fields, dangling retired-file links or unbudgeted sidecars.',
            'actual_sizing_scope': 'Two closed diagnostic metadata+pointer pairs only, not the complete future production archive record or all future checkpoint identities',
            'overflow_action': 'Stop before publication/retirement and report unresolved support; never drop fields, skip a checkpoint, raise a cap or silently change cadence.',
            'retained_live_pairs': 'Latest plus at most three separately designated owned qualified winners; at most one new in-flight pair. External scientific source and initial qualified winner stay immutable/external and cannot become owned by registration.',
            'maximum_candidate_publication_count_enforced_before_write': True,
            'allocation_rounding': 'Actual E allocation unit524288B included for bounded pack/index/history/hot/control files; file-count cap enforced by future backend, not one file per retired checkpoint.',
        },
        'proposed_bounded_catalog': {
            'status': 'NEW future-only backend design, not implemented by storage2 or this model',
            'hot_checkpoint_entries_maximum': 5,
            'open_pack_records_maximum': 1024,
            'open_pack_index_maximum_bytes': 4194304,
            'archive_pack_maximum_bytes': 536870912,
            'sealed_pack_root_entries_maximum': 392,
            'bounded_history': 'At most1024 events per append-only segment, at most2048B/event, at most8events/publication plus32fixed; event count and size overflow fails before append.',
            'atomicity': 'Append framed length/hash records into the one owned current pack, fsync and independently decode/hash the complete new record before publishing the catalog root; seal pack/index once without full-pack copies. Atomic hot pointers/control/index temporary storage must remain within16MiB aggregate scratch and declared live-pair/file counts.',
            'normal_publication_reads': 'Verify exact candidate, coherent current latest and all protected live winners/source/journal prefix; parse only hot state and bounded current-pack index. Never claim sealed historical bytes were reread when only their pinned digest was checked.',
            'historical_verification': 'Stream every sealed pack/index/history on owner startup, recovery, actual pass boundaries and final completion; audit a newly sealed pack once before treating it immutable. Bind cursor/root digests and fail closed on missing, changed, unknown or inconsistent entries.',
            'historical_corruption_detection_latency': 'Corruption of sealed historical bytes between full audit boundaries is detected at the next explicit audit, not necessarily at the next publication. This changed verification schedule requires separate ROOT source/policy review and explicit immutable-file ownership controls.',
            'RAM_claim_limit': 'Entry/index/raw-record byte counts bound application structures only. Exact Python parser/catalog RAM peak and long-horizon I/O/latency remain unmeasured and unapproved.',
        },
        'mandatory_retirement_and_resume_protections': [
            'Only a NEW future namespace with exact ROOT-reviewed owner/source/policy/protected-plan identity may use this backend; no adoption of existing checkpoints or diagnostic artifacts.',
            'Reuse actual streamed writer flush/fsync/rename return semantics; independently verify scalar saved typed identity, optimizer count, sampler digest and authoritative journal prefix plus complete payload hash before latest publication.',
            'Saved semantic tree and canonical identity digest plus exact typed equality must match the live expectation; stale-return checkpoint cannot be published with newer pointer counters.',
            'Require actual native-owner quiescent and mapping-release/no-pending-restore attestation for the exact obsolete pair. A resident idle acknowledgement or process exit alone does not prove all mappings released by every consumer.',
            'Archive exact original JSON/pointer/receipts before retiring uncompressed metadata; independently lossless-roundtrip and hash, fsync, durable catalog commit and verify restore reconstruction. Payload retirement requires the separate existing future-only protection rules.',
            'Never retire coherent latest, qualified winners, external original source, immutable source/history/DEV/winner receipts, or any checkpoint needed by a pending deferred exact restore. Active authoritative journal remains prefix-bound, not frozen as a whole file.',
            'Require exact E containment/resolved owned paths, no symlink/reparse escape, no unknown files, declared disk/file/count caps and unchanged100GB reserve before publication and retirement.',
            'Interrupted archive/publish/retire transactions must recover to the last coherent catalog/pointer without treating prepared or persisted as restored/optimizer-committed.',
        ],
        'remaining_production_gates': [
            'Packed append-only archive/catalog/history backend, atomic recovery and exact owned-only retirement implementation do not exist yet.',
            'Actual complete production metadata+pointer+PRE-save/next-draw receipt record must be measured under hard caps; two diagnostic JSON pairs are not a universal compressibility guarantee.',
            'Current source/initial winner preservation, immutable history/DEV links, active journal prefix and exact owner mapping release callbacks need concrete future source/config review.',
            'Actual streamed writer, archive restore, native optimizer/RNG/draw restore and deferred checkpoint verification integration remain untested by this sizing fixture.',
            'Other authoritative journal, DEV, telemetry, feature-cache, DATA, source and concurrent-use disk growth are excluded from322.045GB isolated namespace-plus-reserve arithmetic; no full-disk fit claimed.',
            'Worst declared payload write IO107661945208832B and strict candidate+four-live payload hash IO538309726044160B remain large; no long-run throughput or latency claim. Cache/immutable-handle optimization requires separate explicit verification policy review.',
            'Full-horizon archive audit bandwidth, hot-root/catalog performance, parser RAM, filesystem behavior and pause/recovery duration require bounded representative evidence before production enablement.',
            'Only at-most-one publication per actual positive update is modeled. Additional pre-update/intermediate/error snapshots must consume an explicit reviewed count and capacity budget or fail before write.',
        ],
        'existing_storage2_or_admitted_caller_changes': False,
        'actual_cleanup_or_general_deletion_authority_granted': False,
    })
    sources = [PROJECT / 'outputs/parakeet_turkish/bpe_checkpoint_storage_capacity_v1.py',
               PROJECT / 'work/research/validate_checkpoint_storage_capacity_CPU_v1.py', Path(__file__)]
    manifest = write('BPE_packed_checkpoint_storage_capacity_source_review_manifest_v1.json', {
        'kind': 'ASR18_inactive_checkpoint_storage_capacity_source_review_manifest_v1',
        'production_enabled': False, 'capacity_or_performance_approval': False,
        'source': [pin(x) for x in sources], 'interface': interface, 'actual_small_fixture': actual,
        'previous_storage2_unchanged': pin(PROJECT / 'outputs/parakeet_turkish/bpe_owned_checkpoint_storage_v2.py'),
        'no_catalog_or_retirement_backend_implemented': True,
    })
    print(json.dumps({'interface': interface, 'manifest': manifest, 'actual': actual}, sort_keys=True))


if __name__ == '__main__':
    main()
