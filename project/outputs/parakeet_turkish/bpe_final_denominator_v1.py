"""Read-only final contiguous-ordinal and target-token denominator audit."""
from pathlib import Path
import sqlite3
from bpe_broad_audio_v1 import scoped,require

def audit(catalog_path,labels_path,*,rows,tokens,guard=lambda:None):
    require(type(rows) is int and rows>0 and type(tokens) is int and tokens>=rows,'Positive exact final denominator required')
    paths=[scoped(catalog_path),scoped(labels_path)]
    signatures={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths}
    result={}
    for path,table in zip(paths,('clips','labels')):
        for suffix in ('-wal','-journal'):
            side=Path(str(path)+suffix)
            require(not side.exists() or side.stat().st_size==0,'Final immutable DB must have no outstanding write journal')
        guard();connection=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)
        try:
            connection.execute('PRAGMA query_only=ON');connection.execute('PRAGMA temp_store=MEMORY');connection.execute('PRAGMA cache_size=-4096')
            connection.set_progress_handler(lambda:(guard() or 0),10000)
            columns={v[1]:v for v in connection.execute(f'PRAGMA table_info({table})')}
            require(columns['ordinal'][2].upper()=='INTEGER' and columns['ordinal'][5]==1,'Unique INTEGER PRIMARY KEY ordinals required')
            n,minimum,maximum=connection.execute(f'SELECT count(*),min(ordinal),max(ordinal) FROM {table}').fetchone()
            require((n,minimum,maximum)==(rows,1,rows),'Final ordinals must be exactly contiguous1..N without holes/extra rows')
            result[table]={'rows':n,'minimum_ordinal':minimum,'maximum_ordinal':maximum,'contiguous_unique_integer_primary_key':True}
            if table=='labels':
                total,invalid=connection.execute('SELECT sum(target_tokens),sum(CASE WHEN typeof(target_ids_u32)!=\'blob\' OR typeof(target_tokens)!=\'integer\' OR target_tokens<=0 OR length(target_ids_u32)!=4*target_tokens THEN 1 ELSE 0 END) FROM labels').fetchone()
                require(total==tokens and invalid==0,'Full target-token SUM/BLOB lengths differ from final denominator')
                result[table].update(total_target_tokens=total,invalid_target_BLOB_rows=invalid,population_mean_U=total/n)
        finally:connection.close()
    guard();require(signatures=={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths},'Final DB changed during denominator validation')
    return {'passed':True,'kind':'ASR18_final_contiguous_denominator_audit_v1','catalog':result['clips'],'labels':result['labels'],
        'actual_optimizer_updates':0,'source_text_rewritten':False,'quality_or_language_filters':[]}
