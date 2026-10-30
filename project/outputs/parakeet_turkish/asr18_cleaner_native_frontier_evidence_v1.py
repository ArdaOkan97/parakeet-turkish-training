"""Truthful own-optimizer frontier evidence even when native monitoring fails."""
from atomic_io_retry_v1 import write_json
from bpe_broad_production_state_v1 import require
def publish(folder,optimizer,*,native=None,committed=False,error=None):
    require(type(optimizer.updates) is int and optimizer.updates in (20,21) and type(committed) is bool,'Bounded original20 plus at most one actual native update')
    require(not committed or (optimizer.updates==21 and native is not None),'Only actual positive native receipt can commit a sampler')
    value={'completed_updates':optimizer.updates,'native_updates_this_child':optimizer.updates-20,
        'actual_positive_Adam_observed_in_owned_optimizer':optimizer.updates==21,
        'clocks':[64250+optimizer.updates,2000+optimizer.updates,62250+optimizer.updates],
        'journal_sampler_committed':committed,'native_return_complete':native is not None,
        'history':[] if native is None else [native],'diagnostic_only':True,'phase_failure':error}
    write_json(folder/'native_progress.json',value);return value
