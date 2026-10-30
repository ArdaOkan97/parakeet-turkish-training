"""Tiny actual cleaner hook plan from coherent20 state; not two completed broad passes."""
import argparse,copy,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import CarriedReplay,digest
from bpe_broad_transaction_v1 import DrawTransaction
from bpe_replay_bucket_no_runtime_v1 import BucketCycler
from asr18_cleaner_fixture_cursor_v1 import DiagnosticCleanerCursor
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');F=A.parent/'FastMSS_integration_v1'
def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();folder=Path(a.folder).resolve();require(folder.is_relative_to(A),'E-only plan');folder.mkdir(exist_ok=False)
    replay_path=A/'post20_replay_metadata_actual_03/report.json';r=json.loads(replay_path.read_text());require(r['passed'] and r['actual20_carried_replay_matches_source_20_choices'],'Actual original carried20 proof')
    clean_path=F/'actual_cleaner_TR_references_01/report.json';clean=json.loads(clean_path.read_text());gpath=F/'actual_cleaner_TR_references_01_guard/closure.json';guard=json.loads(gpath.read_text())
    require(clean['passed'] and clean['selected_exact_ordinals']==[1,352768] and guard['pid']==29052 and guard['actual_exit_code']==0
        and guard['monitor_clean'] and guard['monitor_joined'],'Actual closed original cleaner reference proof')
    refs={v['cleaner_original_ordinal']:v for v in clean['references']};identity={'references':record(clean_path),'is_production_cleaner_pass':False}
    cursor=DiagnosticCleanerCursor(ordinals=[1,352768],identity=identity,seed=digest(r['checkpoint20_scalar_sampler']))
    original=json.loads(Path(r['initial_original_plan']['path']).read_text());initial=original['initial_private_planning_sampler']['carried_replay']
    def rng(value):return (value[0],tuple(value[1]),value[2])
    snapshot={'source_step':64250,'source_identity':initial['source_identity'],'replay_language_counts':initial['source_counts'],
        'replay':initial['replay'],'replay_rng':rng(initial['replay_rng']),'Portuguese_rng':rng(initial['Portuguese_rng'])}
    pool=r['original_replay_pool'];require(record(pool['path'])==pool,'Original replay source binding')
    replay=CarriedReplay(json.loads(Path(pool['path']).read_text())['replay'],snapshot,BucketCycler)
    # This JSON report preserves Random.getstate tuples as JSON lists.
    current=copy.deepcopy(r['before_carried_replay']);current['replay_rng']=rng(current['replay_rng']);current['Portuguese_rng']=rng(current['Portuguese_rng'])
    replay.load_state_dict(current);require(replay.updates==20,'Global carried sourceplus20 frontier')
    def resolve(ordinal):return {'ordinal':ordinal,'source_id':refs[ordinal]['source_id'],'language':'tr'}
    def state_now():return {'continued_broad_diagnostic_cursor':r['checkpoint20_scalar_sampler']['broad_cursor'],
        'diagnostic_cleaner_cursor':cursor.state_dict(),'carried_replay':replay.state_dict()}
    before=copy.deepcopy(state_now());tx=DrawTransaction(cursor,replay,optimizer_updates=20,resolve_TR=resolve)
    require(list(tx.replay_rows)==r['next_carried_replay_rows'],'Exact source-derived update21 replay pair')
    phase='cleaner_data_fine_tuning';draws=[{**row,**draw,'phase':phase} for row,draw in zip(tx.TR_identity,tx.TR_draws)]
    for i,row in enumerate(tx.replay_rows):draws.append({'source_id':row['source_id'],'language':row['language'],'ordinal':None,'phase':phase,
        'epoch':tx.TR_draws[-1]['epoch'],'draw_index':tx.TR_draws[-1]['draw_index']+1+i})
    contexts=[{key:d[key] for key in ('source_id','language','epoch','draw_index','phase')}|{'split':'train'} for d in draws]
    entry={'update':1,'global_update':21,'batch_id':'diagnostic_cleaner_after20_update21','phase':phase,'virtual_only':True,
        'TR_draws':copy.deepcopy(tx.TR_draws),'TR_identity':copy.deepcopy(tx.TR_identity),'replay_rows':copy.deepcopy(list(tx.replay_rows)),
        'TR_rows':[refs[d['ordinal']]['source_row'] for d in tx.TR_draws],'draws':draws,'contexts':contexts,
        'expected_native_transaction_sha256':tx.identity_digest,'expected_before_sampler_sha256':digest(before)}
    next_replay=copy.deepcopy(tx.after_replay);tr_draws=copy.deepcopy(tx.TR_draws);tx.abandon()
    require(state_now()==before,'No source-derived planner frontier consumed')
    cursor.commit(tr_draws);replay.load_state_dict(next_replay);after=state_now();entry['expected_after_sampler_sha256']=digest(after)
    cp=r['actual_native_checkpoint'];require(record(cp['path'])==cp,'Actual20 parent checkpoint binding')
    document=json.loads(Path(cp['path']).read_text())
    def scalar(node):
        kind,value=node
        if kind=='scalar':return value
        if kind in ('list','tuple'):
            values=[scalar(v) for v in value];return tuple(values) if kind=='tuple' else values
        require(kind=='dict','No tensor load in parent identity extraction');return {scalar(k):scalar(v) for k,v in value}
    parent_identity=scalar(next(v for k,v in document['tree'][1] if k==['scalar','identity']))
    plan={'kind':'FastMSS_provisional_clean_feature_window_plan_v1','passed':True,'phase':phase,'diagnostic_only':True,'production_enabled':False,
        'actual_optimizer_updates':0,'actual_cleaner_draws_consumed':0,'actual_replay_frontiers_advanced':False,'maximum_feature_window_bytes':512*1024**2,
        'cleaner_catalog_identity':clean['cleaner_catalog_identity'],'cleaner_actual_references':record(clean_path),'cleaner_actual_guard':record(gpath),
        'source_parent_checkpoint':cp,'source_parent_identity':parent_identity,'source_parent_actual_exit':r['actual_owned_exit'],
        'source_parent_payload':document['payload'],'source_carried_metadata_proof':record(replay_path),'original_replay_pool':pool,
        'initial_diagnostic_sampler':before,'final_private_planning_sampler':after,'updates':[entry],
        'cleaner_fixture_cursor_source':record(S/'asr18_cleaner_fixture_cursor_v1.py'),'runtime_free_bucket':record(S/'bpe_replay_bucket_no_runtime_v1.py'),
        'source_optimizer_or_RNG_reset':False,'initial_source_optimizer_updates':20,'expected_source_clocks':[64270,2020,62270],
        'expected_after_one_native_clocks':[64271,2021,62271],'prior_full_broad_corpus_passes':0,'production_phase_transition_claimed':False,
        'full_cleaner_mean_U':11972886/352768,'source':record(__file__)}
    require(not any(n in sys.modules for n in ('torch','numpy','scipy','transformers','pyarrow')),'Scalar planner imports no numerical/model runtime')
    write_json(folder/'plan.json',plan);print(json.dumps({'passed':True,'plan':record(folder/'plan.json'),'ordered_languages':[d['language'] for d in draws],
        'TR_ordinals':[d['ordinal'] for d in tr_draws],'actual_optimizer_updates':0,'production_phase_transition_claimed':False}))
if __name__=='__main__':main()
