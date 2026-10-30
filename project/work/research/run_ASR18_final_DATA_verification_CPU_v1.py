"""Thin metadata entrypoint for the frozen independently reviewed DATA verifier.

No source generator, decoder, frontend, cache creation, GPU or sampler is used.
Outer ASR18_bounded_CPU_phase_v3 supplies the unchanged fail-closed watchdog.
"""
import argparse, json, os, shutil, sys, time
from pathlib import Path
P = Path(__file__).resolve().parents[2]; S = P/'outputs/parakeet_turkish'; sys.path.insert(0, str(S))
from atomic_io_retry_v1 import read_json, write_json
from bpe_broad_audio_v1 import record, require
from bpe_final_DATA_population_v1 import final_data_population
from bpe_windows_final_metadata_lock_v1 import FinalMetadataReadLocks
from fastmss_phase_aware_plan_bridge_v4 import actual0, read_bound
from resource_watchdog_v2 import ResourceGuard, sample_memory
ROOT = Path(r'E:\gpu_training_scratchpad'); A = ROOT/'analysis/ASR18_memory_diagnosis_v1'; B = ROOT/'analysis/BPE_weak_then_clean_v1'
VERIFIER_SHA = 'f8536cb791e1f2e66a5a4f7565c3b569bda067e01f77f296f908ed7d12e050ee'
PUBLISHER_SHA = '1ba198294761d0c7aa3c70dd610426d4c77e02613722ee6009f1d39416a2769f'
INCREMENTAL_PRIVATE_FORECAST = 512_000_000


class MetadataGuard:
    """Synchronous callbacks sample again after a0.1s throttle interval.

    Callback invocations between real samples reuse only the already validated
    healthy state. The independent outer watchdog still samples every0.2s.
    """
    def __init__(self):
        self.watch = ResourceGuard(); self.last = None; self.count = 0; self.peak = 0; self.start_private = None
    def __call__(self, *, force=False):
        if not force and self.last is not None and time.monotonic()-self.last < .1: return
        value = sample_memory(os.getpid(), 'final_DATA_metadata_verification')
        worker, system = value['worker_private_memory'], value['system_memory']
        require(value['paired_sampling_within_one_second'] is True and worker['available'] is True
                and system['available'] is True, 'Fresh available paired metadata observation required')
        stop, detail = self.watch.observe(system, worker)
        require(stop is None and worker['private_bytes'] < 2_000_000_000, 'Unchanged CPU/private/global guard: '+str(stop))
        require(shutil.disk_usage(ROOT).free >= 100_000_000_000, 'Unchanged E100GB reserve')
        if self.start_private is None: self.start_private = worker['private_bytes']
        require(worker['private_bytes'] <= self.start_private+INCREMENTAL_PRIVATE_FORECAST,
                'Metadata incremental private forecast exceeded; stop, no unchanged retry')
        self.peak = max(self.peak, worker['private_bytes']); self.last = time.monotonic(); self.count += 1; self.sample = value


def environment():
    config = read_json(ROOT/'storage_config.json')
    require(config['storage_root'] == str(ROOT) and config['allow_fallback_outside_root'] is False, 'Authorized E-only policy')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', 'CPU-only verifier')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        require(os.environ.get(key) == '1', 'Four unchanged numerical thread caps')
    for key in ('TEMP','TMP','HF_HOME','TORCH_EXTENSIONS_DIR'):
        require(Path(os.environ[key]).resolve().is_relative_to(ROOT.resolve()), 'Process-local E environment required')


def main(folder):
    environment(); guard = MetadataGuard()
    for _ in range(3): guard(force=True); time.sleep(.05)
    require(folder.resolve().is_relative_to(A.resolve()) and not folder.exists(), 'New E-only independent verification namespace')
    verifier_pin = record(S/'bpe_final_DATA_population_v1.py')
    require(verifier_pin['sha256'] == VERIFIER_SHA, 'Frozen reviewed verifier function changed')
    publisher_pin = record(P/'work/research/finalize_ASR18_DATA_population_for_bounded_preparation_v2.py')
    require(publisher_pin['sha256'] == PUBLISHER_SHA, 'Pinned prepared DATA publisher version changed')
    raw_path = B/'ASR18_final_data_population_for_bounded_preparation_v1.json'
    proof_path = B/'ASR18_final_data_population_for_bounded_preparation_owned_tool_actual_CLOSED0_v1.json'
    # No marker alone supplies permission or successful producer completion.
    raw_pin, proof_pin = record(raw_path), record(proof_path)
    proof = actual0(proof_pin, raw_pin, guard, owner='/root/cpu_data_preparation')
    require(Path(proof['source_path']).resolve() == Path(publisher_pin['path']).resolve()
            and proof['source_SHA256'] == publisher_pin['sha256'], 'Genuine closed exact DATA publisher source required')
    raw = read_bound(raw_pin, guard)
    lock_paths = [raw['bindings'][key]['path'] for key in ('final_catalog','final_labels','declared_geometry_index')]
    expected = [str((ROOT/'indexes'/name).resolve()) for name in
        ('BPE_broad_weak_labels_v1.sqlite','BPE_broad_labels_8711_v1.sqlite','ASR18_declared_training_geometry_v1.sqlite')]
    require([str(Path(path).resolve()) for path in lock_paths] == expected, 'Only canonical genuinely finalized DBs may be read-locked')
    guard(force=True); system = guard.sample['system_memory']
    admission = {'passed': system['commit_used_bytes']+INCREMENTAL_PRIVATE_FORECAST < .90*system['commit_limit_bytes']
        and system['commit_available_bytes']-INCREMENTAL_PRIVATE_FORECAST > 2_000_000_000,
        'incremental_private_forecast_bytes': INCREMENTAL_PRIVATE_FORECAST, 'fresh_sample': guard.sample,
        'private_hard_cap_bytes': 2_000_000_000, 'native_or_CPU_frontend_admission_claimed': False}
    require(admission['passed'], 'Fresh bounded metadata forecast does not fit; no unchanged resource retry')
    began = time.monotonic()
    with FinalMetadataReadLocks(lock_paths):
        validated = final_data_population(raw, guard=guard)
        require(record(raw_path) == raw_pin and record(proof_path) == proof_pin, 'Genuine raw DATA proof changed during independent verification')
    guard(force=True)
    require(guard.count >= 3 and guard.last is not None and time.monotonic()-guard.last < 2, 'Available fresh callback samples at closure')
    require(not any(key in sys.modules for key in ('torch','numpy','scipy','av','soundfile','pyarrow','transformers')),
            'No numerical/audio/frontend/model import during DATA-only verification')
    validated.update(independent_verification_entrypoint=record(__file__), raw_population=raw_pin, raw_population_actual0=proof_pin,
        prospective_metadata_admission=admission, metadata_wall_seconds=time.monotonic()-began,
        metadata_callback_observations={'samples':guard.count,'peak_private_bytes':guard.peak,
            'last_sample_monotonic':guard.last,'callback_sampling_throttle_seconds':.1,'sampled_incremental_forecast_covered':True},
        actual_owned_tool_exit_not_inferred_in_child=True)
    folder.mkdir(exist_ok=False); write_json(folder/'report.json', validated)
    print(json.dumps({'passed':True,'retained_rows':validated['retained_rows'],'full_target_SUM':validated['total_target_tokens'],
        'mean_U':validated['population_mean_U'],'actual_audio_or_GPU_updates':0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--folder', required=True)
    main(Path(parser.parse_args().folder))
