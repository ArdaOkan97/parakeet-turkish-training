"""Inactive performance candidate: bounded caches on existing immutable metadata.

No row, SQL, geometry, sampler or model semantics change. The caller supplies
the existing current-resource admission and lifetime guard, before any draw.
"""
from pathlib import Path
from bpe_broad_audio_v1 import record,require

PER_CONNECTION_BYTES=64*1024**2
CACHE_PAYLOAD_BYTES=3*PER_CONNECTION_BYTES
ALLOCATOR_MARGIN_BYTES=16*1024**2
ALLOCATION_ADMISSION_BYTES=CACHE_PAYLOAD_BYTES+ALLOCATOR_MARGIN_BYTES


def install(context,*,guard,admit):
    from asr18_BPE9223_source_context_v2 import FullRunSourceContext9223,FinalBroadRows9223
    from bpe_declared_geometry_index_v1 import signature
    require(type(context) is FullRunSourceContext9223 and not context.closed
        and type(context.resolver) is FinalBroadRows9223,'Existing exact full9223 metadata resolver')
    resolver=context.resolver;b=context.validated['bindings']
    names=('final_catalog','final_labels','declared_geometry_index')
    expected=[Path(b[k]['path']).resolve()for k in names]
    locks=resolver.metadata_locks
    require(len(locks.handles)==3 and locks.paths==expected,'All three existing deny-write metadata handles stay live')
    connections=[*resolver.connections,resolver.geometry.connection]
    require(len(connections)==3 and not resolver.poisoned and not resolver.geometry.poisoned,'Healthy unchanged three readonly DBs')
    before=[]
    for name,path,connection in zip(names,expected,connections):
        guard();require(signature(path)=={k:b[name][k]for k in ('path','bytes','mtime_ns')},'Exact admitted immutable DB signature')
        rows=connection.execute('PRAGMA database_list').fetchall()
        require(len(rows)==1 and Path(rows[0][2]).resolve()==path
            and connection.execute('PRAGMA query_only').fetchone()[0]==1,'Same read-only database, no temporary DB or writer')
        current=connection.execute('PRAGMA cache_size').fetchone()[0]
        require(current==-4096,'Original4MiB pagecache only; no cumulative/repeated reconfiguration')
        before.append(current)
    admission=admit('three_immutable_metadata_SQLite_pagecaches',ALLOCATION_ADMISSION_BYTES)
    require(admission['passed'] is True,'Full new192MiB caches plus16MiB allocator allowance; no old-cache credit')
    changed=[]
    try:
        for connection in connections:
            guard();connection.execute('PRAGMA cache_size=-65536');changed.append(connection)
            require(connection.execute('PRAGMA cache_size').fetchone()[0]==-65536,'Actual64MiB SQLite payload target')
    except BaseException:
        for connection,previous in zip(changed,before):connection.execute('PRAGMA cache_size='+str(previous))
        raise
    guard()
    return {'kind':'ASR18_actual_readonly_metadata_pagecache_configuration_v1','source':record(__file__),
        'bindings':{k:b[k]for k in names},'previous_cache_size_KiB':before,
        'current_cache_size_KiB':[-65536]*3,'per_connection_cache_payload_bytes':PER_CONNECTION_BYTES,
        'total_cache_payload_bytes':CACHE_PAYLOAD_BYTES,'allocator_margin_bytes':ALLOCATOR_MARGIN_BYTES,
        'full_allocation_admission_bytes':ALLOCATION_ADMISSION_BYTES,'old_cache_savings_credited_bytes':0,
        'SQLite_payload_target_is_not_a_process_private_cap':True,'query_only_and_deny_write_handles_preserved':True,
        'rowcache_SQL_geometry_source_and_sampler_semantics_changed':False,'admission':admission}
