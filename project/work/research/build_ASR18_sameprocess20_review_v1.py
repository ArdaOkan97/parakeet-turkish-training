"""Pin new bounded20 sources plus the owned transitive import closure."""
import datetime,json,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];ROOT=Path(r'E:\gpu_training_scratchpad');AREA=ROOT/'analysis/ASR18_memory_diagnosis_v1'
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import write_json
from asr18_owned_source_closure_v1 import owned_closure

def main():
    old=json.loads((AREA/'production_next_source_review_manifest_v9.json').read_text())
    entries=[Path(item['path']) for item in old['source_bindings']]
    entries.extend(PROJECT/name for name in ('work/research/run_ASR18_stream_checkpoint20_v1.py',
        'work/research/asr18_owned_source_closure_v1.py','outputs/parakeet_turkish/asr18_broad_stream_checkpoint20_probe_v1.py',
        'outputs/parakeet_turkish/asr18_repeat_fixture_cursor_v1.py','outputs/parakeet_turkish/bpe_broad_stream_checkpoint20_v1.disabled.json'))
    entries.extend((AREA/'legacy_CPU_step_retention_01/report.json',AREA/'ROOT_memory_primitives_read_review_v9.json'))
    entries.extend(owned_closure(entries));entries=sorted(set(entries))
    bindings=[record(path) for path in entries]
    root_review=json.loads((AREA/'ROOT_memory_primitives_read_review_v9.json').read_text())
    require(root_review['accepted'] is True,'Actual bounded primitive review required')
    manifest={'scope':'bounded_production_caller_preparation','accepted':False,'production_approval_claimed':False,
        'change':'NEW SAMEprocess20 diagnostic: streamed4MiBCPU checkpoint/READ_ONLYrestore with ownedCPUAdamstepclones removes measured245.9MB legacyCOWmappingretention. Fixed14originalsources repeated20cycles/0corpuspasses, nativecanonicalv1/oldbounds unchanged; restore1+10; fullRNG/replay clocks carried.',
        'native_updates_bound':20,'source_bindings':bindings,'owned_transitive_python_sources':len([p for p in entries if p.suffix=='.py']),
        'dynamic_owned_runtime_overrides_explicitly_pinned':True,'production_forecast_approval_claimed':False}
    path=AREA/'production_next_source_review_manifest_v10.json';require(not path.exists(),'Fresh manifest required');write_json(path,manifest)
    review={'kind':'ASR18_local_sameprocess20_streaming_memory_admission_v10','scope':'bounded_production_caller_preparation',
        'accepted':True,'production_approval_claimed':False,'reviewer':'/root/memory_diagnosis',
        'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'native_updates_bound':20,
        'full_broad_corpus_passes':0,'physical_batch':1,'TR_replay_per_update':[14,2],
        'production_lifecycle_or_steady_forecast_approval_claimed':False,
        'guards_unchanged':True,'new_numerical_policy_introduced_by_this_probe':False,
        'expected_changed_allocation':'No245.9MB legacyCOWmapping kept alive by848B CPUAdamsteps; no fullCPUweights snapshot.',
        'memory_write_forecast':{'incremental_bytes':64000000,'actual_GPU_prior_increment_bytes':4251648},
        'native_repeated_geometry_forecast':'previous clean>=3sample native peak minus current private plus250MB; no unavailable fallback',
        'history':'All failed prior reports preserved; no tolerances widened; actual source/RNG/3Adam histories remain original.',
        'bindings':bindings+[record(path)]}
    review_path=AREA/'sameprocess20_bounded_admission_review_v10.json';require(not review_path.exists(),'Fresh review required');write_json(review_path,review)
    print(json.dumps({'manifest':record(path),'review':record(review_path),'owned_python':manifest['owned_transitive_python_sources']}))

if __name__=='__main__':main()
