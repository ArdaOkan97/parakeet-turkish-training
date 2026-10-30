"""Execute only disabled caller gate; any child launch is intercepted as failure."""
import argparse, hashlib, json, runpy, subprocess, sys
from pathlib import Path
P = Path(__file__).resolve().parents[2]; S = P/'outputs/parakeet_turkish'; sys.path.insert(0, str(S))
from atomic_io_retry_v1 import read_json, write_json
from bpe_broad_audio_v1 import record, require
A = Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
CACHE = Path(r'E:\gpu_training_scratchpad\temp\ASR18_CPU_feature_windows_v1')


def main(folder):
    require(folder.resolve().is_relative_to(A.resolve()) and not folder.exists(), 'Fresh E-only gate control')
    folder.mkdir(); source = P/'work/research/run_ASR18_full_DATA_multiwindow_v5.py'
    manifest = read_json(A/'concrete_multiwindow_caller_source_review_manifest_v5.json')
    config = Path(manifest['configuration_template']['path'])
    require(record(source) in manifest['bindings'] and record(config) == manifest['configuration_template'], 'Frozen caller5/config only')
    before_source = record(source); lease = P/'work/gpu_training_owner.json'; before_lease = hashlib.sha256(lease.read_bytes()).hexdigest()
    before_cache = sorted(p.name for p in CACHE.iterdir()); target = folder/'must_not_be_created'
    checks = []; attempts = []; original = subprocess.Popen
    def forbidden_launch(*args, **kwargs):
        attempts.append(True); raise AssertionError('A disabled caller attempted a child launch')
    subprocess.Popen = forbidden_launch
    try:
        namespace = runpy.run_path(str(source), run_name='ASR18_disabled_caller_negative_control')
        try: namespace['controller'](str(config), str(folder/'review_must_not_be_read.json'), target)
        except ValueError as error:
            require(str(error) == 'Disabled preparation template cannot launch any child', 'Failure must be exact first disabled gate')
            checks.append('Executed frozen caller rejects disabled config at exact first gate')
        else: raise AssertionError('Disabled caller unexpectedly returned')
    finally: subprocess.Popen = original
    require(not attempts, 'No child launch attempted'); checks.append('Popen interception observed zero child attempts')
    require(not target.exists(), 'No output/cache preparation before disabled gate'); checks.append('Caller did not create its target namespace')
    require(not (folder/'review_must_not_be_read.json').exists(), 'No synthetic review created'); checks.append('No ROOT review fabricated or needed by negative control')
    require(before_lease == hashlib.sha256(lease.read_bytes()).hexdigest(), 'GPU lease changed'); checks.append('Actual exclusive idle GPU lease bytes unchanged')
    require(before_cache == sorted(p.name for p in CACHE.iterdir()), 'Shared feature cache changed'); checks.append('Shared real feature cache names unchanged')
    require(before_source == record(source), 'Frozen source changed'); checks.append('Frozen caller source unchanged')
    require(not any(key in sys.modules for key in ('torch','numpy','scipy','transformers','pyarrow')), 'Forbidden numerical/model runtime imported')
    checks.append('No Torch/NumPy/SciPy/frontend/model imported')
    report = {'passed': True, 'kind': 'ASR18_actual_disabled_caller5_gate_negative_control_v1',
        'check_count': len(checks), 'checks': checks, 'source': record(__file__), 'caller_source': before_source,
        'configuration_template': record(config), 'actual_child_launch_attempts': 0, 'actual_GPU_updates': 0,
        'actual_full_DATA_or_positive_CPU_or_native_path_tested': False, 'production_enabled': False}
    write_json(folder/'report.json', report)
    print(json.dumps({'passed': True, 'checks': len(checks), 'child_launch_attempts': len(attempts)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--folder', required=True)
    main(Path(parser.parse_args().folder))
