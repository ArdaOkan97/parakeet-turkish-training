"""Original source scalar and replay equivalence; no model/audio/numeric runtime."""
import argparse,ast,copy,io,json,pickle,random,sys
from collections import OrderedDict
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import CarriedReplay,digest
from bpe_original_schedule_snapshot_v1 import load_original_snapshot,python_random_state
from bpe_checkpoint_scalar_metadata_v1 import ScalarMetadataUnpickler,StorageDescriptor
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();f=Path(a.folder)
    require(f.is_relative_to(A),'E-only metadata controls');f.mkdir(exist_ok=False)
    checks=[]
    def reject(name,call):
        try:call()
        except (ValueError,KeyError,pickle.UnpicklingError):checks.append(name);return
        raise ValueError('Expected rejection: '+name)
    # Copy only exact retained pure bodies; do not import their Torch module.
    parent=S/'asr18_narrow_training_primitives_v1.py';text=parent.read_text(encoding='utf-8');tree=ast.parse(text)
    nodes=[next(n for n in tree.body if getattr(n,'name',None)==name) for name in ('ids_sha','BucketCycler')]
    target=S/'bpe_replay_bucket_stdlib_v1.py'
    payload='"""Exact retained replay bucket bodies, with stdlib-only imports."""\nimport hashlib,math\n\n'+'\n\n'.join(ast.get_source_segment(text,n) for n in nodes)+'\n'
    if target.exists():require(target.read_text(encoding='utf-8')==payload,'Preserve frozen derived source')
    else:target.write_text(payload,encoding='utf-8',newline='\n')
    copied=ast.parse(target.read_text(encoding='utf-8'))
    for original in nodes:
        candidate=next(n for n in copied.body if getattr(n,'name',None)==original.name)
        require(ast.dump(original,include_attributes=False)==ast.dump(candidate,include_attributes=False),'Derived replay body AST differs')
    checks.append('Exact ids_sha and complete BucketCycler AST retained without Torch imports')
    from bpe_replay_bucket_stdlib_v1 import BucketCycler
    snapshot_path=A/'original_schedule_snapshot_CPU_actual_02/snapshot.json';raw=json.loads(snapshot_path.read_text(encoding='utf-8'))
    saved=record(snapshot_path);before_global=random.getstate();snapshot=load_original_snapshot(saved,expected_sources=raw['source_records'])
    require(digest(snapshot)==raw['snapshot_sha256'] and random.getstate()==before_global,'Typed original RNG load must not consume global random state')
    checks.append('Actual original64250 scalar snapshot and all three RNG states restore without global RNG consumption')
    reject('Boolean Python RNG version rejected',lambda:python_random_state([True,snapshot['turkish_rng'][1],None]))
    reject('Float Python RNG integer rejected',lambda:python_random_state([3,[1.0,*snapshot['turkish_rng'][1][1:]],None]))
    reject('Wrong625 RNG geometry rejected',lambda:python_random_state([3,[1,2],None]))
    reject('Nonfinite Gaussian cache rejected',lambda:python_random_state([3,snapshot['turkish_rng'][1],float('inf')]))
    for name,mutate in (
        ('Changed original clocks rejected',lambda v:v['original_clocks'].__setitem__(0,64251)),
        ('Changed source lineage rejected',lambda v:v['source_records'].__setitem__('bpe_parent',{})),
        ('Advanced source frontier declaration rejected',lambda v:v.__setitem__('authoritative_draws_advanced',True)),
        ('Missing source RNG key rejected',lambda v:v['snapshot'].pop('turkish_rng')),
        ('False source snapshot digest rejected',lambda v:v.__setitem__('snapshot_sha256','0'*64)),
        ('Boolean replay clock rejected',lambda v:v['snapshot']['replay_language_counts'].__setitem__('en',True))):
        value=copy.deepcopy(raw);mutate(value)
        if name=='Boolean replay clock rejected':value['snapshot_sha256']=digest(value['snapshot'])
        path=f/(str(len(checks))+'_negative_snapshot.json');write_json(path,value)
        reject(name,lambda:load_original_snapshot(record(path),expected_sources=raw['source_records']))
    pool_binding=json.loads((A/'cleaner_after20_plan_01/plan.json').read_text(encoding='utf-8'))['original_replay_pool']
    require(record(pool_binding['path'])==pool_binding,'Original whole replay pool changed')
    pool=json.loads(Path(pool_binding['path']).read_text(encoding='utf-8'))['replay'];replay=CarriedReplay(pool,snapshot,BucketCycler)
    state=copy.deepcopy(replay.state_dict());plan=json.loads((A/'feature_window20_plan_01/plan.json').read_text(encoding='utf-8'))
    for entry in plan['updates']:
        actual=replay.next();expected=[{k:v for k,v in row.items() if k!='augmentation_role'} for row in entry['replay_rows']]
        require(actual==expected,'Pure retained bucket changed original ordered replay source rows')
    require(replay.updates==20,'Exactly20 private metadata selections');checks.append('All40 original replay rows byte-semantic equal to independent initial20 plan')
    replay.load_state_dict(state);require(replay.state_dict()==state,'All six original queue and selector frontiers restored')
    checks.append('Private metadata selection restores all original six queues/counts/RNG with no authoritative commit')
    require(ScalarMetadataUnpickler(io.BytesIO(pickle.dumps({'safe':[1,'x',None]},protocol=2))).load()=={'safe':[1,'x',None]},'Plain scalar pickle positive')
    checks.append('Restricted metadata reader accepts inert scalar tree')
    reject('Unsupported executable GLOBAL rejected before execution',lambda:ScalarMetadataUnpickler(io.BytesIO(b'cos\nsystem\n.')).load())
    reader=ScalarMetadataUnpickler(io.BytesIO(b''))
    reject('Unknown tensor storage class rejected',lambda:reader.persistent_load(('storage','LongStorage','1','cpu',1)))
    reject('GPU storage descriptor rejected',lambda:reader.persistent_load(('storage','FloatStorage','1','cuda:0',1)))
    reject('Boolean storage length rejected',lambda:reader.persistent_load(('storage','FloatStorage','1','cpu',True)))
    descriptor=reader.persistent_load(('storage','FloatStorage','1','cpu',10))
    require(type(descriptor) is StorageDescriptor,'Inert storage descriptor expected')
    tensor=reader.rebuild_tensor(descriptor,0,(2,),(1,),False,OrderedDict());require(tensor.storage is descriptor,'Inert tensor metadata positive')
    checks.append('Allowed CPU FloatStorage and tensor descriptors never open tensor storage payload')
    reject('Requires-grad tensor metadata rejected',lambda:reader.rebuild_tensor(descriptor,0,(2,),(1,),True,OrderedDict()))
    reject('Boolean tensor dimension rejected',lambda:reader.rebuild_tensor(descriptor,0,(True,),(1,),False,OrderedDict()))
    require(not any(k in sys.modules for k in ('torch','numpy','scipy','transformers','pyarrow')),'No numeric/model/audio runtime')
    checks.append('No numeric/model/audio runtime or actual optimizer update')
    write_json(f/'report.json',{'passed':True,'checks':checks,'check_count':len(checks),'snapshot':saved,
        'source':record(__file__),'pure_replay_source':record(target),'retained_parent_source':record(parent),'retained_ids_and_bucket_AST_exact':True,
        'original_pool':pool_binding,'metadata_only_original40_replay_row_equivalence':True,
        'actual_authoritative_draws_consumed':0,'actual_optimizer_updates':0,'full_final_DATA_positive_exercised':False,'production_enabled':False})
    print(json.dumps({'passed':True,'checks':len(checks),'actual_optimizer_updates':0}))
if __name__=='__main__':main()
