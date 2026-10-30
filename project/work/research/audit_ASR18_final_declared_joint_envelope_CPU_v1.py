"""Run logical JOINT/DP audit only after genuine independent final DATA proof.

No audio/model/GPU work, native admission or production activation is supplied.
"""
import argparse, json, os, shutil, sys, time
from pathlib import Path
P = Path(__file__).resolve().parents[2]; S = P/'outputs/parakeet_turkish'; sys.path.insert(0, str(S))
from atomic_io_retry_v1 import read_json, write_json
from bpe_broad_audio_v1 import record, require
from bpe_declared_joint_budget_audit_v2 import audit_index_joint_budget
from fastmss_phase_aware_plan_bridge_v4 import validate_DATA
from bpe_final_declared_geometry_v2 import check4
from resource_watchdog_v2 import sample_memory
A = Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')


def resource_guard():
    value = sample_memory(os.getpid(), 'declared_joint_scalar_CPU_audit')
    worker, system = value['worker_private_memory'], value['system_memory']
    require(value['paired_sampling_within_one_second'] is True and worker['available'] is True
            and system['available'] is True and worker['private_bytes'] < 2_000_000_000,
            'Fresh paired CPU/private2GB observation required')
    require(system['commit_used_bytes'] < .90*system['commit_limit_bytes']
            and system['commit_available_bytes'] >= 2_000_000_000
            and shutil.disk_usage(A.parents[1]).free >= 100_000_000_000, 'Unchanged global/headroom/E100GB bounds')


def main(evidence_path, folder):
    config = read_json(A.parents[1]/'storage_config.json')
    require(config['storage_root'] == str(A.parents[1]) and config['allow_fallback_outside_root'] is False,
            'Authorized E storage policy required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
            and all(os.environ.get(key) == '1' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),
            'CPU masked/four numerical thread caps required')
    for key in ('TEMP','TMP','HF_HOME','TORCH_EXTENSIONS_DIR'):
        require(Path(os.environ[key]).resolve().is_relative_to(A.parents[1].resolve()), 'Process-local E environment required')
    resource_guard()
    require(folder.resolve().is_relative_to(A.resolve()) and not folder.exists(), 'New E-only final-audit namespace')
    require(evidence_path.resolve().is_relative_to(A.parent.resolve()) and evidence_path.stat().st_size < 128*1024,
            'Bounded independent final DATA evidence metadata')
    evidence_binding = record(evidence_path); evidence = read_json(evidence_path)
    data = validate_DATA(evidence, resource_guard)
    index = data['bindings']['declared_geometry_index']; check4(index, guard=resource_guard)
    began = time.monotonic()
    audit = audit_index_joint_budget(index['path'], expected_rows=data['retained_rows'],
        expected_tokens=data['total_target_tokens'], guard=resource_guard)
    require(record(evidence_path) == evidence_binding, 'Final DATA evidence changed during full audit')
    result = {**audit, 'DATA_evidence': evidence_binding, 'validated_DATA': evidence['validated'],
        'declared_geometry_index': index, 'audit_wall_seconds': time.monotonic()-began,
        'entrypoint_source': record(__file__), 'actual_owned_exit_not_inferred_in_child': True,
        'actual_GPU_or_ASR_updates': 0, 'production_enabled': False}
    folder.mkdir(exist_ok=False); write_json(folder/'report.json', result)
    print(json.dumps({'passed': audit['passed'], 'rows': audit['retained_rows'],
        'unsupported_rows': audit['unsupported_rows'], 'actual_GPU_updates': 0}), flush=True)
    require(audit['passed'], 'Unsupported retained declared JOINT/DP row; explicit reconciliation required, no exclusion')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--DATA-evidence', required=True); parser.add_argument('--folder', required=True)
    args = parser.parse_args(); main(Path(args.DATA_evidence), Path(args.folder))
