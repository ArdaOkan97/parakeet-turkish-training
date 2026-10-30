"""Pure-state production-window rollback/tail/phase/checkpoint controls; no training."""
import argparse,copy,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path[:0]=[str(S),str(Path(__file__).resolve().parent)]
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from bpe_training_schedule_v1 import TrainingSchedule,WEAK,CLEANER
from bpe_broad_production_state_v1 import CommitJournal,digest
from bpe_production_window_coordinator_v2 import make_plan,WindowCoordinator,CAPACITY
from validate_ASR18_training_schedule_v1 import ReplayFixture,OptimizerFixture
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();f=Path(a.folder).resolve()
    require(f.is_relative_to(A),'E-only metadata fixture');f.mkdir(exist_ok=False)
    checks=[]
    def check(value,name):require(value,name);checks.append(name)
    def reject(call,name):
        try:call()
        except (ValueError,OSError,RuntimeError):checks.append(name);return
        raise ValueError('Expected fail-closed: '+name)
    def schedule():return TrainingSchedule(broad_population=15,broad_identity={'FIXTURE_NOT_FINAL':True},broad_mean_U=31.5,
        cleaner_population=352768,cleaner_identity={'FIXTURE_EXISTING_POOL_CONTRACT_ONLY':True},cleaner_mean_U=11972886/352768,
        source_turkish_rng=('STATE_FIXTURE',481),replay=ReplayFixture())
    def identity(phase,ordinal):return {'ordinal':ordinal,'source_id':phase+':'+str(ordinal),'language':'tr'}
    def rows(phase,draws,replay):return [identity(phase,d['ordinal'])|{'full_fixture_targets':[5,6,8710]} for d in draws]+list(replay)
    pop=f/'NOT_actual_final_population.json';write_json(pop,{'FIXTURE_ONLY':True,'production_ready':False});population=record(pop)
    def plan(s,**kw):return make_plan(s,clone_schedule=schedule,resolve_identity=identity,resolve_rows=rows,
        feature_bytes_upper_bound=kw.pop('feature_bytes_upper_bound',lambda phase,r:1000),finalized_population=population,
        verify_finalized_population=lambda bound:bound==population,**kw)
    s=schedule();before=copy.deepcopy(s.state_dict());w=plan(s)
    check(s.state_dict()==before and s.replay.updates==0,'Planning preserves both actual cursors/replay selector RNG')
    check(len(w['updates'])==4 and [len(v['TR_draws']) for v in w['updates']]==[14,1,14,1], 'Two broad passes retain both1-row tails')
    check(all(v['phase']==WEAK for v in w['updates']),'Window stops exactly at phase transition')
    check(w==plan(s),'Fresh private planning has exact ordered draws/targets/contexts/replay identity')
    short=plan(s,feature_bytes_upper_bound=lambda phase,r:260*1024**2)
    check(len(short['updates'])==1 and s.state_dict()==before,'Capacity makes shorter full update window without consuming a draw')
    reject(lambda:plan(s,feature_bytes_upper_bound=lambda phase,r:CAPACITY),'Oversized first full raw update stops before draw')
    reject(lambda:plan(s,maximum_updates=True),'Boolean maximum update rejected')
    reject(lambda:make_plan(s,clone_schedule=schedule,resolve_identity=identity,resolve_rows=rows,feature_bytes_upper_bound=lambda p,r:1,
        finalized_population=population,verify_finalized_population=lambda b:False),'Incomplete population verifier rejects partial mean')
    e=f/'durable_prepared_fixture.json';write_json(e,{'prepared_not_committed':True,'FIXTURE_ONLY':True});eb=record(e)
    def evidence(planned):return eb
    def verify(planned,bound):return bound==eb
    opt=OptimizerFixture();journal=CommitJournal(f/'committed.jsonl');co=WindowCoordinator(s,opt,journal,w,resolve_identity=identity)
    def native(planned,tx):
        opt.updates+=1
        expected=tx.TR_identity+list(tx.replay_rows)
        return {'phase':tx.phase,'update':opt.updates,'source_ids':[r['source_id'] for r in expected],'languages':[r['language'] for r in expected]}
    def prep_failure(_):raise OSError('intentional failed preparation')
    reject(lambda:co.consume(prepare_evidence=prep_failure,native_call=native,verify_preparation=verify),'Preparation failure propagates')
    check(not co.poisoned and not opt.poisoned and s.state_dict()==before and co.index==0,'Failed preparation leaves authoritative state reusable')
    first=co.consume(prepare_evidence=evidence,native_call=native,verify_preparation=verify)
    check(first['committed_window_updates']==1 and s.broad.draws==14 and opt.updates==1,'One positive simulated Adam commits one exact prefix')
    saved=copy.deepcopy(s.state_dict());saved_prefix=journal.binding()
    def writer(state,identity_value):
        payload=f/('state_'+str(opt.updates)+'.bin');payload.write_bytes(b'FIXTURE_NOT_NATIVE_CHECKPOINT')
        metadata=f/('state_'+str(opt.updates)+'.json')
        write_json(metadata,{'kind':'ASR18_streaming_named_tensor_checkpoint_v1','schema_version':1,'dynamic_module_serialization':False,
            'maximum_CPU_tensor_slice_bytes':4194304,'payload':record(payload),'FIXTURE_ONLY':True,'state_digest':digest(state)})
        return record(metadata)
    pointer=f/'latest.json';published=co.publish_checkpoint(write_coherent_checkpoint=writer,pointer_path=pointer,checkpoint_identity={'FIXTURE_ONLY':True})
    check(published['committed_updates']==1 and not published['all_window_updates_committed'],'Intermediate checkpoint keeps future window live')
    fresh=schedule();fresh.load_state_dict(saved);fresh_opt=OptimizerFixture();fresh_opt.updates=1
    resumed_journal=CommitJournal(f/'branched.jsonl');resumed_journal.path.write_bytes(Path(saved_prefix['path']).read_bytes());resumed_journal=CommitJournal(resumed_journal.path)
    resume={'committed_window_updates':1,'checkpoint':published['checkpoint'],'plan_sha256':digest(w),
        'last_transaction_sha256':w['updates'][0]['native_transaction_sha256'],'sampler_state_sha256':digest(saved),'journal_prefix':saved_prefix}
    reject(lambda:WindowCoordinator(fresh,fresh_opt,resumed_journal,w,resolve_identity=identity,resume=resume),'Bare restore index without actual verifier rejected')
    restored=WindowCoordinator(fresh,fresh_opt,resumed_journal,w,resolve_identity=identity,resume=resume,
        verify_resume=lambda value,state,n,prefix:state==saved and n==1 and prefix['sha256']==saved_prefix['sha256'])
    check(restored.index==1 and digest(fresh.state_dict())==w['updates'][1]['before_sampler_sha256'],'Checkpoint prefix verified fresh coordinator resumes exact next update')
    for _ in range(3):co.consume(prepare_evidence=evidence,native_call=native,verify_preparation=verify)
    check(s.phase==CLEANER and s.broad.draws==30 and s.cleaner.draws==0 and opt.updates==4,'No cleaner preparation before both exact broad passes')
    final=co.publish_checkpoint(write_coherent_checkpoint=writer,pointer_path=pointer,checkpoint_identity={'FIXTURE_ONLY':True})
    check(final['all_window_updates_committed'] and final['committed_window_updates']==4,'Final coherent pointer binds entire window')
    clean=plan(s,maximum_updates=2)
    check(clean['phase']==CLEANER and all(v['phase']==CLEANER for v in clean['updates']) and clean['initial_committed_updates']==4,
          'Cleaner planning carries global Adam/replay clocks and explicit clean contexts')
    require(s.state_dict()!=before,'Fixture should actually exercise hypothetical state commits')
    # Late write failure must never replace an earlier completed pointer.
    preserved=pointer.read_bytes()
    def failed_writer(state,identity_value):
        write_json(f/'incomplete_exists.json',{'writer_has_not_returned':True});raise OSError('late fsync/write failure')
    reject(lambda:co.publish_checkpoint(write_coherent_checkpoint=failed_writer,pointer_path=pointer,checkpoint_identity={}), 'Metadata existence alone cannot publish latest')
    check(pointer.read_bytes()==preserved,'Failed checkpoint leaves previous latest pointer intact')
    # A native error after a clock advance poisons live state; no replay-on-top.
    broken=schedule();badopt=OptimizerFixture();badco=WindowCoordinator(broken,badopt,CommitJournal(f/'failed_native.jsonl'),w,resolve_identity=identity)
    def native_failure(planned,tx):badopt.updates+=1;raise RuntimeError('intentional after Adam')
    reject(lambda:badco.consume(prepare_evidence=evidence,native_call=native_failure,verify_preparation=verify),'Native late failure propagates')
    check(badco.poisoned and badopt.poisoned and broken.poisoned and broken.broad.draws==0,'Native failure poisons weights while unconsumed sampler prefix stays original')
    reject(lambda:badco.consume(prepare_evidence=evidence,native_call=native_failure,verify_preparation=verify),'Changed live native retry rejected')
    def shared_clone():
        private=schedule();private.broad=s.broad;return private
    reject(lambda:make_plan(s,clone_schedule=shared_clone,resolve_identity=identity,resolve_rows=rows,feature_bytes_upper_bound=lambda p,r:1,
        finalized_population=population,verify_finalized_population=lambda b:True),'Shared phase cursor clone rejected')
    # A failed journal may follow a real Adam and both provisional cursor moves.
    # Restore bookkeeping to the prior authoritative prefix, then poison live
    # optimizer/schedule so only coherent checkpoint recovery can continue.
    jstate=schedule();jbefore=copy.deepcopy(jstate.state_dict());jopt=OptimizerFixture()
    class FailedJournal:
        def append(self,event):raise OSError('intentional post Adam journal failure')
    jco=WindowCoordinator(jstate,jopt,FailedJournal(),w,resolve_identity=identity)
    def jnative(planned,tx):
        jopt.updates+=1;expected=tx.TR_identity+list(tx.replay_rows)
        return {'phase':tx.phase,'update':jopt.updates,'source_ids':[r['source_id'] for r in expected],'languages':[r['language'] for r in expected]}
    reject(lambda:jco.consume(prepare_evidence=evidence,native_call=jnative,verify_preparation=verify),'Late journal failure propagates')
    check(jstate.state_dict()==jbefore and jstate.poisoned and jco.poisoned and jopt.poisoned,
          'Late journal failure restores BOTH cursors and replay RNG then prohibits changed live retry')
    check(not any(n in sys.modules for n in ('torch','numpy','scipy','transformers','pyarrow')),'Metadata fixture imports no numerical/model runtime')
    write_json(f/'report.json',{'passed':True,'checks':checks,'check_count':len(checks),'source':record(__file__),
        'coordinator_source':record(S/'bpe_production_window_coordinator_v2.py'),'actual_native_updates':0,'actual_production_draws_consumed':0,
        'synthetic_optimizer_and_population_only':True,'actual_cleaner_audio_or_frontend_proved':False,'production_ready':False})
    print(json.dumps({'passed':True,'checks':len(checks),'actual_native_updates':0}))
if __name__=='__main__':main()
