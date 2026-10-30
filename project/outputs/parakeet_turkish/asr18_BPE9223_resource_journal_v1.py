"""Explicit abnormal-stop journal handoff; never calls normal resume as fake0.

Only the entire exact closed checkpoint journal can be reused. No branch,
truncation, suffix replay, registry rewrite or authoritative draw occurs here.
The unchanged namespace/writer retains its quotas, owner and append checks.
"""
from copy import deepcopy
from pathlib import Path
from bpe_broad_audio_v1 import record, require
from asr18_BPE9223_resource_operational_compatibility_v1 import checked_resource_stop

def resume_after_resource_stop(namespace, saved, *, resource_closure,
                               operational_compatibility, guard):
    value = checked_resource_stop(resource_closure, identity=namespace.identity,
        compatibility=operational_compatibility, guard=guard)
    require(type(saved) is dict and set(saved) == {'path','offset','sha256'}
            and type(saved['offset']) is int and saved['offset'] >= 0,
            'Exact typed source checkpoint prefix')
    closed = value['closed_registered_journal']
    require(saved == {'path':closed['path'],'offset':closed['bytes'],'sha256':closed['sha256']}
            and Path(saved['path']) in namespace.paths,
            'Whole registered closed p1 journal only; no suffix or rewind')
    namespace.check(); journal = namespace.writer(Path(saved['path']))
    require(journal.binding() == saved, 'Unmodified exact authoritative journal before any new draw')
    return journal, {'kind':'ASR18_resource_stop_bounded_journal_resume_v1',
        'source':record(__file__),'identity':deepcopy(namespace.identity),
        'source_checkpoint_prefix':deepcopy(saved),'current_native_resource_closure':resource_closure,
        'operational_compatibility':operational_compatibility,
        'actual_resource_quiescence_proven':True,'actual_native_normal_exit0_claimed':False,
        'same_whole_closed_journal_reused':True,'copied_bytes':0,'unknown_tail_preserved_bytes':0,
        'original_journal_modified':False,'tail_replayed_or_sampler_advanced':False,
        'journal':journal.binding(),'journal_quota_bytes':16*1024**3}
