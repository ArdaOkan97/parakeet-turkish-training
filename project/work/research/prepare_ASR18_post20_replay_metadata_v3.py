"""Exact scalar sampler extraction and carried replay control without Torch."""
import argparse,ast,copy,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import write_json
from bpe_broad_production_state_v1 import CarriedReplay,digest
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();f=Path(a.folder).resolve();require(f.is_relative_to(A),'E-only scalar proof');f.mkdir(exist_ok=False)
    source=S/'asr18_narrow_training_primitives_v1.py';original=source.read_text();nodes=ast.parse(original).body
    bodies=[ast.get_source_segment(original,next(v for v in nodes if getattr(v,'name',None)==name)) for name in ('ids_sha','BucketCycler')]
    target=S/'bpe_replay_bucket_no_runtime_v1.py'
    new='"""Exact retained BucketCycler/ids_sha bodies, without numerical imports."""\nimport hashlib,math\n\n'+'\n\n'.join(bodies)+'\n'
    if target.exists():require(target.read_text()==new,'Existing no-runtime bucket source differs')
    else:target.write_text(new,encoding='utf-8',newline='\n')
    derived=ast.parse(target.read_text()).body
    require(all(ast.dump(next(v for v in nodes if getattr(v,'name',None)==name),include_attributes=False)==
        ast.dump(next(v for v in derived if getattr(v,'name',None)==name),include_attributes=False) for name in ('ids_sha','BucketCycler')),
        'Exact retained sampler AST bodies required')
    from bpe_replay_bucket_no_runtime_v1 import BucketCycler
    native=A/'feature_window_native20_01';actual=json.loads((native/'actual_owned_CLOSED0.json').read_text())
    require(actual['actual_tool_exit_code']==actual['actual_native_exit_code']==0 and actual['native_updates']==20 and actual['report']==record(native/'report.json'), 'Actual owned20 scalar source')
    checkpoint=actual['checkpoint'];require(record(checkpoint['path'])==checkpoint,'Exact metadata binding')
    document=json.loads(Path(checkpoint['path']).read_text());require(document['source']==record(S/'bpe_streaming_checkpoint_v1.py') and document['kind']=='ASR18_streaming_named_tensor_checkpoint_v1','Pinned scalar-tree format')
    def decode(node):
        require(type(node) is list and len(node)==2,'Scalar subtree node');kind,value=node
        if kind=='scalar':require(value is None or type(value) in (str,int,float,bool),'Scalar type');return value
        if kind in ('tuple','list'):
            values=[decode(v) for v in value];return tuple(values) if kind=='tuple' else values
        require(kind=='dict' and type(value) is list,'No tensor data may be loaded in scalar sampler extraction')
        pairs=[(decode(k),decode(v)) for k,v in value];require(len({k for k,v in pairs})==len(pairs),'Unique scalar keys');return dict(pairs)
    root=document['tree'];require(root[0]=='dict','Top dictionary required');top={k[1]:v for k,v in root[1] if k[0]=='scalar'}
    state=decode(top['sampler']);post=state['carried_replay'];require(post['updates']==20,'Actual sourceplus20 replay frontier')
    plan_path=A/'feature_window20_plan_01/plan_v2.json';plan=json.loads(plan_path.read_text());initial=plan['initial_private_planning_sampler']['carried_replay']
    snapshot={'source_step':64250,'source_identity':initial['source_identity'],'replay_language_counts':initial['source_counts'],'replay':initial['replay'],
        'replay_rng':(initial['replay_rng'][0],tuple(initial['replay_rng'][1]),initial['replay_rng'][2]),'Portuguese_rng':(initial['Portuguese_rng'][0],tuple(initial['Portuguese_rng'][1]),initial['Portuguese_rng'][2])}
    pool=plan['source_records']['replay_pool'];require(record(pool['path'])==pool,'Original1914 replay pool binding')
    rows=json.loads(Path(pool['path']).read_text())['replay'];replay=CarriedReplay(rows,snapshot,BucketCycler)
    for update in plan['updates']:
        chosen=replay.next();expected=[{k:v for k,v in r.items() if k!='augmentation_role'} for r in update['replay_rows']]
        require(chosen==expected,'Runtime-free exact20 original carried replay choices differ')
    require(replay.state_dict()==post,'Actual checkpoint20 allqueues/selector RNG must match all20 source-derived choices')
    before=copy.deepcopy(replay.state_dict());next_rows=replay.next();after=copy.deepcopy(replay.state_dict());replay.load_state_dict(before)
    require(replay.state_dict()==before and replay.updates==20,'Provisional next21 leaves actual carried state unchanged')
    require(not any(n in sys.modules for n in ('torch','numpy','scipy','transformers','pyarrow')),'Scalar metadata no numerical/model imports')
    write_json(f/'report.json',{'kind':'ASR18_actual20_scalar_carried_replay_extraction_v1','passed':True,'actual_native_checkpoint':checkpoint,
        'actual_owned_exit':record(native/'actual_owned_CLOSED0.json'),'initial_original_plan':record(plan_path),'original_replay_pool':pool,
        'runtime_free_bucket_source':record(target),'derived_from':record(source),'sampler_class_and_ids_body_AST_exact':True,
        'checkpoint20_scalar_sampler':state,'actual20_carried_replay_matches_source_20_choices':True,
        'next_diagnostic_update':21,'next_carried_replay_rows':next_rows,'before_carried_replay':before,'after_private_provisional_carried_replay':after,
        'before_carried_sha256':digest(before),'after_private_carried_sha256':digest(after),'actual_Adam_or_authoritative_frontier_advanced':False,
        'production_phase_transition_proved':False,'source':record(__file__)})
    print(json.dumps({'passed':True,'next_replay':[{'source_id':r['source_id'],'language':r['language']} for r in next_rows],
        'source_20_choices_and_checkpoint_all_queues_RNG_exact':True,'report':record(f/'report.json')}))
if __name__=='__main__':main()
