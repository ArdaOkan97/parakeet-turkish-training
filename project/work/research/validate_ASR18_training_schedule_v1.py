"""Finite pure-state phase, exact tails, rollback and post-Adam failure proof."""
import argparse, copy, json, os, random, sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];ROOT=Path(r'E:\gpu_training_scratchpad')
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
from bpe_training_schedule_v1 import TrainingSchedule,WEAK,CLEANER,COMPLETE
from bpe_broad_production_state_v1 import CommitJournal,digest,require
from bpe_broad_audio_v1 import record

class ReplayFixture:
    def __init__(self):self.rng=random.Random(741);self.updates=0
    def next(self):
        language=self.rng.choice(('en','es','it','fr','de'));self.updates+=1
        return [{'source_id':language+':'+str(self.updates),'language':language}, {'source_id':'pt:'+str(self.updates),'language':'pt'}]
    def state_dict(self):return {'updates':self.updates,'rng':self.rng.getstate()}
    def load_state_dict(self,value):self.updates=value['updates'];self.rng.setstate(value['rng'])

class OptimizerFixture:
    def __init__(self):self.updates=0;self.poisoned=False

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    folder=Path(a.output).resolve();require(folder.is_relative_to(ROOT.resolve()),'E-only schedule fixture');folder.mkdir(exist_ok=False)
    checks=0
    def check(value,message):
        nonlocal checks
        require(value,message);checks+=1
    def make():
        return TrainingSchedule(broad_population=15,broad_identity={'fixture_only':15},broad_mean_U=31.5,
            cleaner_population=352768,cleaner_identity={'existing_pool':352768},cleaner_mean_U=11972886/352768,
            source_turkish_rng=('fixture_rng',481),replay=ReplayFixture())
    def resolve(phase,ordinal):return {'ordinal':ordinal,'source_id':phase+':'+str(ordinal),'language':'tr'}
    evidence=folder/'preparation.json';write_json(evidence,{'fixture_only':True,'prepared_not_committed':True});binding=record(evidence)
    schedule=make();optimizer=OptimizerFixture();journal=CommitJournal(folder/'journal.jsonl')
    before=schedule.state_dict();proposal=schedule.propose(optimizer_updates=0,resolve_TR=resolve)
    first_id=proposal.identity_digest;first_draw=copy.deepcopy(proposal.TR_draws)
    check(schedule.state_dict()==before,'Preparation preserved both cursors and replay/RNG')
    proposal.abandon();check(schedule.state_dict()==before,'Abandon preserved exact state')
    proposal=schedule.propose(optimizer_updates=0,resolve_TR=resolve)
    check(proposal.identity_digest==first_id and proposal.TR_draws==first_draw,'Repeated preparation identity exact')
    proposal.abandon();actual_draws=[];sizes=[];phases=[]
    for update in range(1,6):
        transaction=schedule.propose(optimizer_updates=optimizer.updates,resolve_TR=resolve)
        sizes.append(len(transaction.TR_draws));phases.append(transaction.phase)
        actual_draws.extend((transaction.phase,d['epoch'],d['ordinal']) for d in transaction.TR_draws)
        expected=transaction.TR_identity+list(transaction.replay_rows)
        optimizer.updates+=1
        native={'update':optimizer.updates,'phase':transaction.phase,'source_ids':[r['source_id'] for r in expected],
                'languages':[r['language'] for r in expected]}
        state=transaction.commit(optimizer=optimizer,journal=journal,native_receipt=native,prepared_evidence_binding=binding)
        check(schedule.replay.updates==optimizer.updates==schedule.committed_updates(),'Exact phase/replay/global Adam counts')
        check(state==schedule.state_dict(),'Checkpoint state includes both stages and carried RNG')
    check(sizes==[14,1,14,1,14] and phases==[WEAK]*4+[CLEANER],'Two full broad passes with1-row tails before cleaner')
    broad=[v for v in actual_draws if v[0]==WEAK]
    check(len(broad)==30 and all(sorted(v[2] for v in broad if v[1]==epoch)==list(range(1,16)) for epoch in (0,1)),
          'Every fixture ordinal exactly once in each complete broad pass')
    saved=copy.deepcopy(schedule.state_dict());next_original=schedule.propose(optimizer_updates=optimizer.updates,resolve_TR=resolve)
    original_id=next_original.identity_digest;next_original.abandon()
    fresh=make();fresh.load_state_dict(saved)
    restored=fresh.propose(optimizer_updates=optimizer.updates,resolve_TR=resolve)
    check(restored.identity_digest==original_id,'Actual fresh-instance phase/cursor/replay-RNG next draw exact');restored.abandon()
    # Synthetic already-committed tail state tests one final10-row cleaner
    # boundary. This is a state fixture, not actual corpus processing.
    tail=copy.deepcopy(saved);tail['cleaner_cursor']['draws']=352758
    tail['carried_replay']['updates']=4+352758//14
    fresh=make();fresh.load_state_dict(tail);tail_optimizer=OptimizerFixture();tail_optimizer.updates=tail['carried_replay']['updates']
    transaction=fresh.propose(optimizer_updates=tail_optimizer.updates,resolve_TR=resolve)
    check(len(transaction.TR_draws)==10 and transaction.phase==CLEANER,'Exact final cleaner10-row tail, no wrap/discard')
    tail_optimizer.updates+=1;expected=transaction.TR_identity+list(transaction.replay_rows)
    transaction.commit(optimizer=tail_optimizer,journal=journal,native_receipt={'phase':CLEANER,'update':tail_optimizer.updates,
        'source_ids':[r['source_id'] for r in expected],'languages':[r['language'] for r in expected]},prepared_evidence_binding=binding)
    check(fresh.phase==COMPLETE and fresh.broad.draws==30 and fresh.cleaner.draws==352768,'Exact declared phase completion')
    try:fresh.propose(optimizer_updates=tail_optimizer.updates,resolve_TR=resolve)
    except ValueError:check(True,'No draw after all required passes')
    else:raise ValueError('Extra pass accepted')
    bad=copy.deepcopy(saved);bad['phase']=WEAK;fresh=make()
    try:fresh.load_state_dict(bad)
    except ValueError:check(fresh.poisoned,'Invalid restored phase poisons live state')
    else:raise ValueError('Invalid phase accepted')
    fresh=make();transaction=fresh.propose(optimizer_updates=0,resolve_TR=resolve);opt=OptimizerFixture();opt.updates=1
    class FailingJournal:
        def append(self,event):raise OSError('intentional post-Adam journal failure')
    expected=transaction.TR_identity+list(transaction.replay_rows)
    try:transaction.commit(optimizer=opt,journal=FailingJournal(),native_receipt={'phase':WEAK,'update':1,
        'source_ids':[r['source_id'] for r in expected],'languages':[r['language'] for r in expected]},prepared_evidence_binding=binding)
    except OSError:check(opt.poisoned and fresh.poisoned and transaction.poisoned,'Post-Adam journal failure forbids changed-state retry')
    else:raise ValueError('Intentional journal failure missing')
    events=[json.loads(line) for line in Path(journal.binding()['path']).read_text().splitlines()]
    check(all('sampler_state_sha256'in event and 'sampler'not in event for event in events),'Journal binds compact digest instead of copying all replay queues')
    check('torch'not in sys.modules and 'numpy'not in sys.modules,'Pure state fixture imports no runtime')
    write_json(folder/'report.json',{'passed':True,'checks':checks,'PID':os.getpid(),'CPU_only':True,
        'source':record(__file__),'schedule_source':record(PROJECT/'outputs/parakeet_turkish/bpe_training_schedule_v1.py'),
        'fixture_broad_population':15,'actual_training_draws_consumed':0,'synthetic_tail_fixture_not_corpus_processing':True,
        'two_broad_then_one_cleaner_state_order':True,'all_rng_and_both_cursors_exact_next_restore':True,
        'post_Adam_journal_failure_poisoned':True,'production_ready':False})
    print(json.dumps({'passed':True,'checks':checks}))

if __name__=='__main__':main()
