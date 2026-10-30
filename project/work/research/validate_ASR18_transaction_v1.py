"""Finite sampler/journal crash transaction and strict final-schema fixture."""
import argparse,ast,hashlib,json,math,pickle,random,sys
from copy import deepcopy
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from bpe_broad_production_state_v1 import CompletePassCursor,CarriedReplay,CommitJournal,require
from bpe_broad_transaction_v1 import DrawTransaction
from bpe_broad_final_population_v2 import final_population,REQUIRED
from atomic_io_retry_v1 import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();folder=Path(a.folder);folder.mkdir(exist_ok=False)
    # Execute only the exact two pure sampler AST nodes, avoiding Torch imports.
    source=PROJECT/'outputs/parakeet_turkish/train_lora_batched_data_v11.py';tree=ast.parse(source.read_text())
    nodes=[v for v in tree.body if isinstance(v,(ast.ClassDef,ast.FunctionDef)) and v.name in ('ids_sha','BucketCycler')]
    require(len(nodes)==2,'Actual pinned sampler nodes');space={'math':math,'hashlib':hashlib};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),space)
    cycler=space['BucketCycler'];languages=('en','es','it','fr','de','pt')
    rows=[{'source_id':l+str(i),'language':l,'encoder_frames':i+1,'target_tokens':i+2} for l in languages for i in range(3)]
    oldrng=random.Random(17);ptrng=random.Random(23)
    snapshot={'source_step':64250,'source_identity':{'fixture':True},'replay_language_counts':{l:0 for l in languages},
        'replay_rng':oldrng.getstate(),'Portuguese_rng':ptrng.getstate(),'replay':{l:cycler([r for r in rows if r['language']==l],ptrng if l=='pt' else oldrng).state_dict() for l in languages}}
    cursor=CompletePassCursor(population=15,identity={'fixture':True},seed=37);replay=CarriedReplay(rows,snapshot,cycler)
    resolve=lambda n:{'ordinal':n,'source_id':'tr'+str(n),'language':'tr'}
    class Optimizer:updates=0;poisoned=False
    optimizer=Optimizer();journal=CommitJournal(folder/'committed.jsonl');checks=0;seen={0:[],1:[]};sizes=[]
    while cursor.remaining:
        before=(cursor.state_dict(),deepcopy(replay.state_dict()))
        first=DrawTransaction(cursor,replay,optimizer_updates=optimizer.updates,resolve_TR=resolve)
        require((cursor.state_dict(),replay.state_dict())==before,'Preparation must leave BOTH durable frontiers/RNG untouched')
        identity=first.identity_digest;first.abandon();checks+=2
        tx=DrawTransaction(cursor,replay,optimizer_updates=optimizer.updates,resolve_TR=resolve);require(tx.identity_digest==identity,'Reprepared source/replay draws differ');checks+=1
        for draw in tx.TR_draws:seen[draw['epoch']].append(draw['ordinal'])
        sizes.append(len(tx.TR_draws));optimizer.updates+=1
        expected=tx.TR_identity+list(tx.replay_rows);receipt={'update':optimizer.updates,'source_ids':[r['source_id'] for r in expected],'languages':[r['language'] for r in expected]}
        state=tx.commit(optimizer=optimizer,journal=journal,native_receipt=receipt);checks+=1
        if optimizer.updates==1:
            saved=folder/'sampler_fixture.pkl';saved.write_bytes(pickle.dumps(state));restored=pickle.loads(saved.read_bytes())
            other=CompletePassCursor(population=15,identity={'fixture':True},seed=37);other.load_state_dict(restored['broad_cursor'])
            carry2=CarriedReplay(rows,snapshot,cycler);carry2.load_state_dict(restored['carried_replay'])
            a1=DrawTransaction(cursor,replay,optimizer_updates=optimizer.updates,resolve_TR=resolve);b1=DrawTransaction(other,carry2,optimizer_updates=optimizer.updates,resolve_TR=resolve)
            require(a1.identity_digest==b1.identity_digest,'Actual saved/restored BOTH sampler next draws/RNG differ');a1.abandon();b1.abandon();checks+=1
    require(sizes==[14,1,14,1] and all(sorted(v)==list(range(1,16)) for v in seen.values()),'No pass-boundary crossing/tail loss/duplicate');checks+=1
    require(replay.updates==4 and replay.streams['pt'].draws==4 and sum(replay.streams[l].draws for l in languages[:-1])==4,'Carried exact +1PT/+1oldfive counters');checks+=1
    for binding in ({},{'extra':{}}, {k:{} for k in REQUIRED-{'geometry_admission'}}, {**{k:{} for k in REQUIRED},'extra':{}}):
        try:final_population({'bindings':binding})
        except ValueError:checks+=1
        else:raise AssertionError('Missing/extra final binding admitted')
    c=CompletePassCursor(population=3,identity={'fixture':True},seed=1);r=CarriedReplay(rows,snapshot,cycler);o=Optimizer();o.updates=0
    t=DrawTransaction(c,r,optimizer_updates=0,resolve_TR=resolve);o.updates=1
    class FailedJournal:
        def append(self,event):raise OSError('intentional fixture write failure after Adam')
    allrows=t.TR_identity+list(t.replay_rows)
    try:t.commit(optimizer=o,journal=FailedJournal(),native_receipt={'update':1,'source_ids':[v['source_id'] for v in allrows],'languages':[v['language'] for v in allrows]})
    except OSError:require(o.poisoned and t.poisoned,'After-Adam journal failure must quarantine live state');checks+=1
    else:raise AssertionError('Expected journal failure')
    require('torch' not in sys.modules and 'numpy' not in sys.modules,'No model/audio imports')
    write_json(folder/'report.json',{'passed':True,'meaningful_checks':checks,'sizes':sizes,'source_sampler_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_actual_BucketCycler_AST_used':True,'both_sampler_transactions_and_actual_saved_restore':True,'afterAdam_failure_poisoned_not_retried':True,
        'empty_omitted_extra_final_bindings_rejected':True,'fixture_only':True})

if __name__=='__main__':main()
