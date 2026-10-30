"""Hypothetical fixed-fixture inputs only; no actual sampler/Adam frontier moves."""
import argparse,copy,gc,json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';T=P/'work/training-env/Lib/site-packages'
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
sys.path.insert(0,str(T))
import numpy as np
sys.path.remove(str(T))
import torch
assert torch.__version__=='2.7.1+cpu' and torch.version.cuda is None
sys.path.extend([str(S)])
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import CarriedReplay,digest
from bpe_broad_transaction_v1 import DrawTransaction
from asr18_repeat_fixture_cursor_v1 import RepeatedFixtureCursor
from asr18_narrow_training_primitives_v1 import BucketCycler

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();folder=Path(a.folder)
    require(folder.is_relative_to(A),'E-only provisional window');folder.mkdir(exist_ok=False);torch.set_num_threads(1)
    projection=json.loads((A/'source_projection_v4/report.json').read_text());source=projection['projections']['bpe']
    require(record(source['projected']['path'])==source['projected'],'Original source projection changed')
    saved=torch.load(source['projected']['path'],map_location='cpu',weights_only=True,mmap=True)
    sampler=copy.deepcopy(saved['sampler']['continued_parent_sampler']);del saved;gc.collect()
    snapshot={'source_step':64250,'source_identity':source['parent'],**{key:sampler[key] for key in ('replay','replay_language_counts','turkish_rng','replay_rng','Portuguese_rng')}}
    pool=P/'outputs/parakeet_turkish/runs/adaptive_20261014_TR_WorldSpeech_expanded_01/training_pool.json'
    require(record(pool)['sha256']=='00b839c717762aba74326e9867969d85b7ffaaabd7ead7ddbe732fe6be8c6cf4','Complete original replay pool')
    replay=CarriedReplay(json.loads(pool.read_text())['replay'],snapshot,BucketCycler)
    manifest=Path(r'E:\gpu_training_scratchpad\analysis\BPE_weak_then_clean_v1\production_reader_contract_v1\manifest.jsonl')
    rows=[json.loads(v) for v in manifest.read_text().splitlines()];require(len(rows)==14,'Exact fixed14 fixture')
    by_ordinal={r['ordinal']:r for r in rows};identity={'fixture_manifest':record(manifest),'is_final_corpus':False}
    cursor=RepeatedFixtureCursor(ordinals=sorted(by_ordinal),identity=identity,seed=digest(snapshot['turkish_rng']),repeats=20)
    def resolve(ordinal):return {'ordinal':ordinal,'source_id':by_ordinal[ordinal]['unique_id'],'language':'tr'}
    initial={'broad_cursor':cursor.state_dict(),'carried_replay':replay.state_dict()};updates=[];phase='weak_label_continued_training'
    for update in range(1,21):
        before={'broad_cursor':cursor.state_dict(),'carried_replay':replay.state_dict()}
        transaction=DrawTransaction(cursor,replay,optimizer_updates=update-1,resolve_TR=resolve)
        draws=[{**row,**draw,'phase':phase} for row,draw in zip(transaction.TR_identity,transaction.TR_draws)]
        replay_rows=[{**r,'augmentation_role':'replay'} for r in transaction.replay_rows]
        for i,row in enumerate(replay_rows):draws.append({'source_id':row['source_id'],'language':row['language'],'ordinal':None,'phase':phase,
            'epoch':transaction.TR_draws[-1]['epoch'],'draw_index':transaction.TR_draws[-1]['draw_index']+1+i})
        contexts=[{'source_id':d['source_id'],'language':d['language'],'epoch':d['epoch'],'draw_index':d['draw_index'],'phase':phase,'split':'train'} for d in draws]
        entry={'update':update,'batch_id':f'update_{update}','phase':phase,'TR_draws':transaction.TR_draws,'TR_identity':transaction.TR_identity,
            'replay_rows':replay_rows,'draws':draws,'contexts':contexts,'expected_native_transaction_sha256':transaction.identity_digest,
            'expected_before_sampler_sha256':digest(before),'virtual_only':True}
        next_replay=copy.deepcopy(transaction.after_replay);TR_draws=copy.deepcopy(transaction.TR_draws)
        transaction.abandon() # Verify the actual proposal left every queue/RNG untouched.
        # These are private planning copies. No production sampler or Adam exists.
        cursor.commit(TR_draws);replay.load_state_dict(next_replay)
        entry['expected_after_sampler_sha256']=digest({'broad_cursor':cursor.state_dict(),'carried_replay':replay.state_dict()})
        updates.append(entry)
    require(cursor.draws==280 and replay.updates==20,'Finite hypothetical window exactly20 fixtures')
    final={'broad_cursor':cursor.state_dict(),'carried_replay':replay.state_dict()}
    result={'kind':'ASR18_provisional_fixed14_feature_window_plan_v1','passed':True,'diagnostic_only':True,'production_enabled':False,
        'maximum_updates':20,'maximum_feature_window_bytes':512*1024**2,'per_update_TR':14,'per_update_carried_replay':2,'phase':phase,
        'actual_optimizer_updates':0,'actual_broad_draws_consumed':0,'actual_replay_frontiers_advanced':0,'complete_corpus_passes':0,
        'source_records':{'bpe':source['parent'],'projected':source['projected'],'replay_pool':record(pool)},'original_clocks':[64250,2000,62250],
        'TR_fixture_manifest':record(manifest),'TR_rows':rows,'fixture_mean_U':sum(r['BPE']['target_tokens'] for r in rows)/14,
        'initial_private_planning_sampler':initial,'final_hypothetical_sampler':final,'updates':updates,
        'augmentation_policy':record(S/'fastmss_broad_optin_v1.json'),'frozen_decoder':record(P/'work/research/decode_ASR18_broad_batch_v6.py'),
        'frozen_guard':record(P/'work/research/ASR18_bounded_CPU_phase_v3.py'),'source':record(__file__),
        'no_actual_Adam_or_authoritative_cursor_object_created':True,'provisional_preparation_never_commits_draw':True,
        'source_RNG_reset_or_reseeded':False,'frozen_PCM_cache_guards_unchanged':True,'full_PCM_pool_created':False}
    write_json(folder/'plan.json',result);print(json.dumps(record(folder/'plan.json')))

if __name__=='__main__':main()
