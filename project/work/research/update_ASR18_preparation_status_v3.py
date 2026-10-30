"""Small current pointer, preserving immutable diagnostic history."""
import datetime, hashlib, json, sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(r'E:\gpu_training_scratchpad')
AREA = ROOT / 'analysis/ASR18_memory_diagnosis_v1'
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json

def bound(path):
    raw = path.read_bytes()
    value = json.loads(raw)
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'passed': value.get('passed', value.get('complete', False))}

def main():
    assert json.loads((ROOT / 'storage_config.json').read_text())['storage_root'] == str(ROOT)
    lease = json.loads((PROJECT / 'work/gpu_training_owner.json').read_text())
    proofs = {}
    for key, relative in {
        'all207_canonical_oracle': 'real_target_TDT_gradients_06/real_model_equivalence.json',
        'joint_canonical_oracle': 'actual_joint_precision_08/actual_joint_precision.json',
        'full89second_U419_native_update': 'real_target_TDT_large_02/full_large_row_validation.json',
        'heldout12273_PCM': 'heldout_PCM_binding_v1/report.json',
        'exact14_lowmemory_CPU': 'full14_FLAC_CPU_actual_01/report.json',
        'cleaner352768_metadata': 'cleaner_metadata_actual_01/report.json',
        'prior_integrated_partial': 'integrated_broad_caller_01/report.json',
        'current_integrated': 'integrated_broad_caller_02/report.json',
    }.items():
        path = AREA / relative
        if path.exists():
            proofs[key] = bound(path)
    value = {
        'updated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'owner': '/root/memory_diagnosis', 'phase': lease['status'],
        'actual_GPU_running': lease['actual_GPU_running'],
        'actual_owned_GPU_PID': lease['actual_owned_worker_pid'],
        'current_stage': lease.get('diagnostic_phase'), 'diagnostic_folder': lease.get('diagnostic_folder'),
        'proofs': proofs, 'production_ready': False, 'production_enabled': False,
        'long_production_started': False, 'diagnostic_checkpoints_unqualified': True,
        'complete_broad_corpus_passes': 0, 'full_corpus_final_mean_and_pass_denominator_admitted': False,
        'data_continues_in_parallel': True, 'all_raw_broad_labels_and_flags_retained': True,
        'heldout_reference_PCM_complete': True, 'full_TRAIN_PCM_runtime_reconciliation_pending': True,
        'historical_external_commit_attribution_unresolved': True,
        'new_CPU_remedy': {'exact14_original_private_peak_bytes': 780013568,
                           'exact14_changed_private_peak_bytes': 168316928,
                           'source_identical_original_compressed_FLAC': True,
                           'all_clean_PCM_and_prepared_augmentation_byte_exact': True,
                           'arbitrary_production_geometry_forecast_claimed': False},
        'production_remaining': [
            'final pinned source acquisition/catalog/labels and full mean with actual owned closures',
            'complete final geometry and compressed-FLAC carrier coverage',
            'integrated native checkpoint/restore/prepared input/cache lifecycle actual closure',
            'production loop, cleaner coordinator and serial350DEV selection gates',
            'final ROOT review of exact production sources/config',
        ],
        'original_memory_remedy_completion_preserved': str(AREA / 'completion_handoff_v1.json'),
    }
    write_json(AREA / 'production_preparation_status_v1.json', value)
    write_json(AREA / 'launch_readiness_v2.json', value)
    path = AREA / 'memory_diagnosis_status.json'
    memory = json.loads(path.read_text())
    memory.update(updated_at_utc=value['updated_at_utc'],
                  owned_GPU_child_running=lease['actual_GPU_running'],
                  actual_owned_GPU_PID=lease['actual_owned_worker_pid'],
                  production_preparation_receipt=str(AREA / 'production_preparation_status_v1.json'))
    write_json(path, memory)
    print(json.dumps({key: value[key] for key in ('phase', 'actual_owned_GPU_PID', 'current_stage', 'production_ready')}))

if __name__ == '__main__':
    main()
