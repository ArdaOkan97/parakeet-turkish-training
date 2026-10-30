"""Source-resolved provisional next draw; no sampler commit or audio work."""
from copy import deepcopy
from bpe_broad_production_state_v1 import digest, require


def next_draw_packet(schedule, context, *, optimizer_updates, guard):
    guard(); before = deepcopy(schedule.state_dict())
    require(type(optimizer_updates) is int and optimizer_updates == schedule.committed_updates()
            and not schedule.poisoned, 'Actual positive source schedule frontier required')
    tx = schedule.propose(optimizer_updates=optimizer_updates, resolve_TR=context.resolve_identity)
    try:
        rows = context.resolve_rows(tx.phase, tx.TR_draws, tx.replay_rows)
        require(len(rows) == len(tx.TR_draws)+2, 'Complete provisional TR tail plus carried replay pair')
        draws = [{**identity, **draw, 'phase':tx.phase}
                 for identity,draw in zip(tx.TR_identity,tx.TR_draws)]
        last = tx.TR_draws[-1]
        draws.extend({'source_id':r['source_id'],'language':r['language'],'ordinal':None,'phase':tx.phase,
                      'epoch':last['epoch'],'draw_index':last['draw_index']+1+j}
                     for j,r in enumerate(tx.replay_rows))
        contexts = [{key:draw[key] for key in ('source_id','language','epoch','draw_index','phase')}
                    | {'split':'train'} for draw in draws]
        value = {'kind':'ASR18_actual_checkpoint_provisional_next_draw_packet_v1',
            'optimizer_updates':optimizer_updates,'phase':tx.phase,
            'before_sampler_sha256':digest(before),'native_transaction_sha256':tx.identity_digest,
            'TR_draws':deepcopy(tx.TR_draws),'TR_identity':deepcopy(tx.TR_identity),
            'replay_rows':deepcopy(list(tx.replay_rows)),'after_replay_sha256':digest(tx.after_replay),
            'source_resolved_full_rows_sha256':digest(rows),'draws':draws,'contexts':contexts,
            'draws_consumed':0,'replay_frontier_advanced':False,'audio_or_augmentation_applied':False}
    finally:
        tx.abandon()
    require(digest(schedule.state_dict()) == digest(before),
            'Provisional packet changed actual sampler/frontiers/replay RNG')
    return value
