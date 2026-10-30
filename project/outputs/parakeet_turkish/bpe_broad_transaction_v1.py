"""Provisional complete-pass/replay draws, committed only after positive Adam.

Preparing rows never advances either durable sampler. A failure after Adam or
journal append poisons this live transaction; recover the prior checkpoint.
"""
from copy import deepcopy
from bpe_broad_production_state_v1 import require,digest

class DrawTransaction:
    def __init__(self,cursor,replay,*,optimizer_updates,resolve_TR):
        require(cursor.remaining>0,'Completed broad phase has no further draw')
        self.cursor=cursor;self.replay=replay;self.before_cursor=cursor.state_dict();self.before_replay=deepcopy(replay.state_dict())
        self.optimizer_updates=optimizer_updates;self.closed=False;self.poisoned=False
        to_pass_boundary=cursor.population-(cursor.draws%cursor.population)
        self.TR_draws=cursor.preview(min(14,cursor.remaining,to_pass_boundary))
        try:
            self.replay_rows=tuple(replay.next());self.after_replay=deepcopy(replay.state_dict())
        finally:replay.load_state_dict(self.before_replay)
        require(cursor.state_dict()==self.before_cursor and replay.state_dict()==self.before_replay,'Preparation must preserve both durable sampler states')
        self.TR_identity=[resolve_TR(draw['ordinal']) for draw in self.TR_draws]
        require(len(self.TR_identity)==len(self.TR_draws) and all(v['language']=='tr' and v['ordinal']==d['ordinal'] and v['source_id'] for v,d in zip(self.TR_identity,self.TR_draws)),
                'Independent ordinal/source identity required before decode')
        self.identity_digest=digest({'TR':self.TR_identity,'draws':self.TR_draws,
            'replay':[{'source_id':r['source_id'],'language':r['language']} for r in self.replay_rows],
            'before_cursor':self.before_cursor,'before_replay':self.before_replay,'after_replay':self.after_replay})

    def _unchanged(self):
        require(not self.closed and not self.poisoned,'Consumed/poisoned transaction cannot be reused')
        require(self.cursor.state_dict()==self.before_cursor and self.replay.state_dict()==self.before_replay,'Concurrent sampler mutation prohibited')

    def abandon(self):
        self._unchanged();self.closed=True
        return {'draws_consumed':0,'cursor_restored':True,'replay_and_selector_RNG_restored':True}

    def commit(self,*,optimizer,journal,native_receipt):
        self._unchanged()
        require(not optimizer.poisoned and optimizer.updates==self.optimizer_updates+1,'Actual exactly-one positive Adam commit required')
        try:
            require(native_receipt['update']==optimizer.updates,'Native receipt must identify the actual optimizer clock')
            expected=self.TR_identity+list(self.replay_rows)
            require(native_receipt['source_ids']==[r['source_id'] for r in expected] and native_receipt['languages']==[r['language'] for r in expected],
                    'Actual native rows differ from independently prepared source/draw identities')
            self.cursor.commit(self.TR_draws);self.replay.load_state_dict(self.after_replay)
            state={'broad_cursor':self.cursor.state_dict(),'carried_replay':self.replay.state_dict()}
            event={'kind':'ASR18_committed_broad_update_v1','transaction_identity_sha256':self.identity_digest,
                'optimizer_updates':optimizer.updates,'sampler':state,'native':native_receipt}
            journal.append(event);self.closed=True;return state
        except BaseException:
            # Do not retry changed live weights against an unadvanced frontier.
            self.poisoned=True;optimizer.poisoned=True;raise
