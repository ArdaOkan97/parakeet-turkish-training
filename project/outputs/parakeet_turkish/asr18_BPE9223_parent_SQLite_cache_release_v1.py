"""Release quiescent immutable metadata page caches, without changing limits.

Connections, locks, exact file signatures and their original 4 MiB cache
settings survive. This helper cannot draw, close/reopen a reader, write a DB,
change a PRAGMA setting or infer that the supervisor now fits its CPU cap.
"""
from pathlib import Path
import sqlite3

from bpe_broad_audio_v1 import record, require
from bpe_declared_geometry_index_v1 import signature
from asr18_BPE9223_source_context_v2 import FullRunSourceContext9223


def _targets(context, guard):
    require(type(context) is FullRunSourceContext9223 and not context.closed,
            'Exact live original scientific metadata context')
    bindings = context.validated['bindings']
    require(len(context.resolver.connections) == 2,
            'Only the two original broad row connections')
    values = [
        ('broad_catalog', context.resolver.connections[0], bindings['final_catalog'], context.resolver.signatures[0]),
        ('broad_labels', context.resolver.connections[1], bindings['final_labels'], context.resolver.signatures[1]),
        ('broad_geometry', context.resolver.geometry.connection, bindings['declared_geometry_index'], context.resolver.geometry.signature),
        ('cleaner_labels', context.cleaner_rows.connection, context.cleaner_rows.database, context.cleaner_rows.signature),
    ]
    result = []
    for name, connection, binding, established_signature in values:
        guard()
        require(type(connection) is sqlite3.Connection and not connection.in_transaction,
                'Quiescent exact SQLite connection, no outstanding transaction')
        path = Path(binding['path']).resolve()
        tables = list(connection.execute('PRAGMA database_list'))
        main = [row for row in tables if row[1] == 'main']
        require(len(main) == 1 and Path(main[0][2]).resolve() == path
                and all(row[1] in ('main', 'temp') for row in tables),
                'Only the exact source-bound immutable metadata DB, no attached DB')
        settings = tuple(connection.execute('PRAGMA ' + key).fetchone()[0]
                         for key in ('query_only', 'temp_store', 'cache_size'))
        require(settings == (1, 2, -4096),
                'Original query-only, memory temp and 4 MiB cache settings unchanged')
        before = signature(path)
        require(before == established_signature and before['bytes'] == binding['bytes'],
                'Same exact pinned DB signature before cache release')
        result.append((name, connection, binding, settings, before))
    return result


def release_owned_readonly_caches(context, *, guard, measure):
    """Return actual per-connection private observations, never memory credit."""
    targets = _targets(context, guard)
    rows = []
    initial = measure()
    require(type(initial) is int and initial > 0, 'Actual current private telemetry')
    for name, connection, binding, settings, before in targets:
        guard()
        start = measure()
        require(list(connection.execute('PRAGMA shrink_memory')) == [],
                'The original SQLite cache release operation returned normally')
        after_settings = tuple(connection.execute('PRAGMA ' + key).fetchone()[0]
                               for key in ('query_only', 'temp_store', 'cache_size'))
        require(not connection.in_transaction and after_settings == settings
                and signature(binding['path']) == before,
                'No DB signature, transaction, cache limit or query policy changed')
        end = measure()
        require(type(start) is int and type(end) is int and start > 0 and end > 0,
                'Actual per-connection current private observations')
        rows.append({'connection': name, 'binding': dict(binding),
                     'query_only': 1, 'temp_store': 2, 'cache_size': -4096,
                     'actual_before_private_bytes': start,
                     'actual_after_private_bytes': end})
    guard()
    return {'kind': 'ASR18_actual_quiescent_readonly_metadata_SQLite_cache_release_v1',
            'source': record(__file__), 'connections': rows,
            'actual_before_private_bytes': initial,
            'actual_after_private_bytes': measure(),
            'connections_closed_or_reopened': False,
            'cache_limits_or_resource_guards_changed': False,
            'sampler_or_training_work_performed': False,
            'release_or_resource_credit_assumed': False}
