"""Bounded CPU extraction of original schedule scalars, with no model creation."""
import argparse, copy, gc, json, os, sys
from pathlib import Path
P=Path(__file__).resolve().parents[2]; S=P/'outputs/parakeet_turkish'
sys.path.insert(0,str(S))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record, require
from bpe_broad_production_state_v1 import digest
from bpe_original_schedule_snapshot_v1 import KIND, RNG_KEYS, load_original_snapshot
from resource_watchdog_v2 import sample_memory
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args()
    folder=Path(a.folder).resolve();require(folder.is_relative_to(A.resolve()),'E-only source scalar evidence')
    folder.mkdir(exist_ok=False)
    sample=sample_memory(os.getpid(),'before_original_scalar_extraction')
    system=sample['system_memory'];require(system['available'] and sample['paired_sampling_within_one_second'], 'Fresh paired CPU preparation admission')
    forecast=700_000_000
    require(system['commit_used_bytes']+forecast < .9*system['commit_limit_bytes']
            and system['commit_limit_bytes']-system['commit_used_bytes']-forecast >= 2_000_000_000,
            'Full700MB CPU extraction forecast must fit unchanged system/headroom guards')
    write_json(folder/'admission.json',{'paired_sample':sample,'incremental_forecast_bytes':forecast,'passed':True})
    training_site=P/'work/training-env/Lib/site-packages'
    sys.path.insert(0,str(training_site));import numpy;sys.path.remove(str(training_site))
    import torch
    require(str(torch.__version__)=='2.7.1+cpu' and torch.version.cuda is None, 'Isolated CPU-only Torch required')
    torch.set_num_threads(1)
    projection_path=A/'source_projection_v4/report.json';projection=json.loads(projection_path.read_text(encoding='utf-8'))
    require(projection['passed'] is True and projection['parents_unchanged'] is True, 'Original reviewed projection required')
    source=projection['projections']['bpe']; original=projection['projections']['original']
    require(source['step']==64250 and original['step']==62250
            and source['parent']['sha256']=='71ca1897d95622c5556df6a3b568d26136004e25c4bcca515f7a242df9e9b63a'
            and original['parent']['sha256']=='b0f5d5e207caeda3c030a5f11eda4ae9004e4826a7fd050a8ac8de78333e3e38'
            and record(source['projected']['path'])==source['projected'], 'Exact original BPE/parked lineage and compact projection required')
    saved=torch.load(source['projected']['path'],map_location='cpu',weights_only=True,mmap=True)
    carried=copy.deepcopy(saved['sampler']['continued_parent_sampler'])
    snapshot={'source_step':64250,'source_identity':source['parent'], **{k:carried[k] for k in ('replay','replay_language_counts',*RNG_KEYS)}}
    def scalars_only(value):
        if isinstance(value,torch.Tensor):return False
        if type(value) is dict:return all(type(k) is str and scalars_only(v) for k,v in value.items())
        if type(value) in (list,tuple):return all(scalars_only(v) for v in value)
        return value is None or type(value) in (int,float,bool,str)
    require(scalars_only(snapshot),'Do not retain mapped tensor/CPU Adam-step references in the scalar snapshot')
    del saved,carried;gc.collect()
    records={'bpe_parent':source['parent'],'bpe_projection':source['projected'], 'parked_parent':original['parent'],
             'projection_proof':record(projection_path)}
    result={'kind':KIND,'original_clocks':[64250,2000,62250],'source_records':records,'snapshot':snapshot,
            'snapshot_sha256':digest(snapshot),'authoritative_draws_advanced':False,'optimizer_or_RNG_reset':False,
            'extraction_source':record(__file__)}
    path=folder/'snapshot.json';write_json(path,result)
    restored=load_original_snapshot(record(path),expected_sources=records)
    require(restored==snapshot and digest(restored)==digest(snapshot),'Original typed schedule RNG/queue snapshot roundtrip exact')
    # The previous fixed14 control is independent evidence of the same initial
    # carried replay state. Its exhausted cursor is never loaded or extended.
    initial_plan_path=A/'feature_window20_plan_01/plan.json'
    plan=json.loads(initial_plan_path.read_text(encoding='utf-8'))
    initial=plan['initial_private_planning_sampler'];actual=initial['carried_replay']
    require(actual['updates']==0 and actual['source_identity']==snapshot['source_identity']
            and actual['source_counts']==snapshot['replay_language_counts'] and actual['replay']==snapshot['replay']
            and digest(actual['replay_rng'])==digest(snapshot['replay_rng'])
            and digest(actual['Portuguese_rng'])==digest(snapshot['Portuguese_rng'])
            and initial['broad_cursor']['seed']==digest(snapshot['turkish_rng']),
            'Extracted original state differs from independently recorded initial64250 diagnostic state')
    write_json(folder/'report.json',{'passed':True,'snapshot':record(path),'source':record(__file__),
        'loader':record(S/'bpe_original_schedule_snapshot_v1.py'),'source_records':records,
        'independent_original_initial_plan':record(initial_plan_path), 'all_original_RNG_queues_counts_exact':True,
        'retained_snapshot_contains_no_tensor':True,'mapped_archive_references_dropped':True,
        'new_model_created':False,'GPU_used':False,'actual_optimizer_updates':0,'draws_consumed':0,
        'full_population_schedule_constructed':False,'production_enabled':False})
    print(json.dumps({'passed':True,'snapshot':record(path),'actual_optimizer_updates':0}))
if __name__=='__main__':main()
