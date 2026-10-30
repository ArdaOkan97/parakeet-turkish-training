"""New unadmitted rollback hardening; preserve the actual v1 sources/reports."""
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish'
p=S/'bpe_production_window_coordinator_v1.py';q=p.with_name('bpe_production_window_coordinator_v2.py')
t=p.read_text()
needle='def make_plan(schedule, *,'
helper="""def restore_frontiers(schedule,before):
    # These source-bound objects validate identities before restoring counters,
    # queue order and selector RNG. Native weights/RNG require coherent reload.
    schedule.broad.load_state_dict(before['broad_cursor'])
    schedule.cleaner.load_state_dict(before['cleaner_cursor'])
    schedule.replay.load_state_dict(before['carried_replay'])
    require(schedule.state_dict()==before,'Both phase cursors and replay RNG restoration failed; stop')

"""
t=t.replace(needle,helper+needle)
t=t.replace("private is not schedule and private.replay is not schedule.replay", "private is not schedule and private.replay is not schedule.replay and private.broad is not schedule.broad and private.cleaner is not schedule.cleaner")
t=t.replace("        require(schedule.state_dict()==before,'Planning touched authoritative cursors/replay/RNG; do not arm')", "        unchanged=schedule.state_dict()==before\n        if not unchanged:restore_frontiers(schedule,before)\n        require(unchanged,'Planning touched authoritative cursors/replay/RNG; restored original snapshot and did not arm')")
t=t.replace("                self.poisoned=True;self.optimizer.poisoned=True;self.schedule.poisoned=True", "                restore_frontiers(self.schedule,before)\n                self.poisoned=True;self.optimizer.poisoned=True;self.schedule.poisoned=True")
t=t.replace("                require(self.schedule.state_dict()==before,'Failed preparation advanced schedule; stop')", "                if self.schedule.state_dict()!=before:restore_frontiers(self.schedule,before)")
assert not q.exists() and t!=p.read_text();q.write_text(t,encoding='utf-8',newline='\n')
p=Path(__file__).with_name('validate_ASR18_production_window_v1.py');q=p.with_name('validate_ASR18_production_window_v2.py');t=p.read_text().replace('bpe_production_window_coordinator_v1','bpe_production_window_coordinator_v2')
needle="    check(not any(n in sys.modules for n in ('torch','numpy','scipy','transformers','pyarrow'))"
extra="""    def shared_clone():
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
"""
assert needle in t;t=t.replace(needle,extra+needle);assert not q.exists();q.write_text(t,encoding='utf-8',newline='\n')
