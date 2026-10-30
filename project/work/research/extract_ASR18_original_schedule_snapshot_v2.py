"""Extract original schedule scalars with restricted metadata, no Torch import."""
import argparse,copy,gc,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from bpe_original_schedule_snapshot_v1 import KIND,RNG_KEYS,load_original_snapshot
from bpe_checkpoint_scalar_metadata_v1 import read_scalar_metadata
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();folder=Path(a.folder).resolve()
    require(folder.is_relative_to(A.resolve()),'E-only scalar evidence');folder.mkdir(exist_ok=False)
    projection_path=A/'source_projection_v4/report.json';projection=json.loads(projection_path.read_text(encoding='utf-8'))
    source=projection['projections']['bpe'];original=projection['projections']['original']
    require(projection['passed'] is True and projection['parents_unchanged'] is True and source['step']==64250 and original['step']==62250
        and source['parent']['sha256']=='71ca1897d95622c5556df6a3b568d26136004e25c4bcca515f7a242df9e9b63a'
        and original['parent']['sha256']=='b0f5d5e207caeda3c030a5f11eda4ae9004e4826a7fd050a8ac8de78333e3e38','Exact original lineage required')
    saved,read_proof=read_scalar_metadata(source['projected'],expected_pickle_bytes=66_539_150)
    carried=saved['sampler']['continued_parent_sampler']
    snapshot=copy.deepcopy({'source_step':64250,'source_identity':source['parent'],
        **{key:carried[key] for key in ('replay','replay_language_counts',*RNG_KEYS)}})
    def scalar(value):
        if type(value) is dict:return all(type(k) is str and scalar(v) for k,v in value.items())
        if type(value) in (list,tuple):return all(scalar(v) for v in value)
        return value is None or type(value) in (int,float,bool,str)
    require(scalar(snapshot),'No inert storage/tensor or executable object may enter schedule snapshot')
    del saved,carried;gc.collect()
    records={'bpe_parent':source['parent'],'bpe_projection':source['projected'],'parked_parent':original['parent'],'projection_proof':record(projection_path)}
    value={'kind':KIND,'original_clocks':[64250,2000,62250],'source_records':records,'snapshot':snapshot,'snapshot_sha256':digest(snapshot),
        'authoritative_draws_advanced':False,'optimizer_or_RNG_reset':False,'extraction_source':record(__file__)}
    path=folder/'snapshot.json';write_json(path,value);loaded=load_original_snapshot(record(path),expected_sources=records)
    require(loaded==snapshot and digest(loaded)==digest(snapshot),'Original typed sampler snapshot roundtrip exact')
    plan_path=A/'feature_window20_plan_01/plan.json';plan=json.loads(plan_path.read_text(encoding='utf-8'))
    initial=plan['initial_private_planning_sampler'];actual=initial['carried_replay']
    require(actual['updates']==0 and actual['source_identity']==snapshot['source_identity']
        and actual['source_counts']==snapshot['replay_language_counts'] and actual['replay']==snapshot['replay']
        and digest(actual['replay_rng'])==digest(snapshot['replay_rng']) and digest(actual['Portuguese_rng'])==digest(snapshot['Portuguese_rng'])
        and initial['broad_cursor']['seed']==digest(snapshot['turkish_rng']),'Independent initial64250 queues/seed/RNG must match')
    require(not any(k in sys.modules for k in ('torch','numpy','scipy','transformers','pyarrow')),'Scalar-only extraction imports no numeric/model runtime')
    write_json(folder/'report.json',{'passed':True,'snapshot':record(path),'source':record(__file__),'loader':record(S/'bpe_original_schedule_snapshot_v1.py'),
        'restricted_metadata_reader':read_proof,'source_records':records,'independent_initial_plan':record(plan_path),
        'all_original_RNG_queues_counts_exact':True,'retained_snapshot_contains_no_tensor':True,'model_or_GPU_created':False,
        'actual_optimizer_updates':0,'draws_consumed':0,'full_population_schedule_constructed':False,'production_enabled':False})
    print(json.dumps({'passed':True,'snapshot':record(path),'inert_tensor_descriptors':read_proof['inert_tensor_descriptors'],'actual_optimizer_updates':0}))
if __name__=='__main__':main()
