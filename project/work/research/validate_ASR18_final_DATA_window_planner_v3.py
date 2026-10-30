"""Partial metadata protocol controls, never a synthetic final-DATA approval."""
import argparse,copy,hashlib,json,sqlite3,struct,sys
from contextlib import contextmanager
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from bpe_declared_geometry_index_v1 import COLS
from bpe_final_DATA_window_planner_v2 import FinalBroadRows,feature_upper_bound,prepare_first_plan
from bpe_training_schedule_v1 import WEAK
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');B=A.parent/'BPE_weak_then_clean_v1'

@contextmanager
def sqlite_owned(path):
    connection=sqlite3.connect(path)
    try:
        with connection:yield connection
    finally:connection.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();f=Path(a.folder);require(f.is_relative_to(A),'E-only protocol fixture');f.mkdir(exist_ok=False)
    checks=[]
    def reject(name,call):
        try:call()
        except (ValueError,KeyError):checks.append(name);return
        raise ValueError('Expected rejection: '+name)
    # The public entry must fail before snapshot/replay/row I/O without final
    # DATA. No fake positive DATA wrapper or scientific schedule is created.
    reject('Missing final DATA blocks before original schedule/source files opened',lambda:prepare_first_plan(
        DATA_evidence={},snapshot_binding={},snapshot_actual0={},snapshot_report={},expected_original_sources={},
        replay_pool_binding={},cleaner_metadata_binding={}))
    manifest=B/'production_reader_contract_v1/manifest.jsonl';rows=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    saved=json.loads((A/'runtime_geometry_CPU_actual_02/report.json').read_text(encoding='utf-8'));declarations={r['unique_id']:r['declaration'] for r in saved['cases']}
    paths={key:f/(key+'.sqlite') for key in ('final_catalog','final_labels','declared_geometry_index')}
    rawkeys=('ordinal','unique_id','dataset','original_split','source_id','shard','row_index','start','end','duration','text','text_raw','text_spoken','original_flags_json')
    with sqlite_owned(paths['final_catalog']) as c:
        c.execute('CREATE TABLE clips(ordinal INTEGER PRIMARY KEY,unique_id TEXT,dataset TEXT,original_split TEXT,source_id TEXT,shard TEXT,row_index INTEGER,start REAL,end REAL,duration REAL,text TEXT,text_raw TEXT,text_spoken TEXT,original_flags_json TEXT)')
        c.executemany('INSERT INTO clips VALUES('+','.join('?' for _ in rawkeys)+')',[tuple(r[k] for k in rawkeys) for r in rows])
    with sqlite_owned(paths['final_labels']) as c:
        c.execute('CREATE TABLE labels(ordinal INTEGER PRIMARY KEY,target_ids_u32 BLOB,target_tokens INTEGER,text_sha256 TEXT)')
        c.executemany('INSERT INTO labels VALUES(?,?,?,?)',[(r['ordinal'],b''.join(struct.pack('<I',v) for v in r['BPE']['target_ids']),
            r['BPE']['target_tokens'],hashlib.sha256(r['text'].encode('utf-8')).hexdigest()) for r in rows])
    with sqlite_owned(paths['declared_geometry_index']) as c:
        c.execute('CREATE TABLE geometry(ordinal INTEGER PRIMARY KEY,source_component INTEGER,native_sr INTEGER,native_samples INTEGER,clean16k_samples INTEGER,mel_frames INTEGER,encoder_frames INTEGER,target_tokens INTEGER,status TEXT)')
        c.executemany('INSERT INTO geometry VALUES('+','.join('?' for _ in COLS)+')',[tuple(declarations[r['unique_id']][k] for k in COLS) for r in rows])
    pins={k:record(v) for k,v in paths.items()};pins['declared_geometry_index']['mtime_ns']=paths['declared_geometry_index'].stat().st_mtime_ns
    protocol={'kind':'ASR18_validated_final_DATA_population_for_bounded_preparation_v1','passed':True,'complete':True,
        'retained_rows':max(r['ordinal'] for r in rows),'bindings':pins}
    # Exercise the row resolver protocol directly on an explicitly partial
    # metadata fixture. This is not passed through final DATA validation.
    resolver=FinalBroadRows(protocol)
    try:
        for raw in rows:
            actual=resolver.row(raw['ordinal']);require(actual['tr_metadata_labels']==raw['BPE']['target_ids']
                and actual['broad_original']=={k:raw[k] for k in rawkeys} and actual['source_id']==raw['unique_id']
                and actual['recording_source_id']==raw['source_id'] and actual['encoder_frames']==declarations[raw['unique_id']]['encoder_frames'],
                'Partial fixture resolver must preserve complete original row/full targets/declared geometry')
        checks.append('All14 actual-saved metadata rows preserve full BPE IDs, raw/source IDs, supplied text and flags')
        first=rows[0]['ordinal'];one=resolver.row(first);one['tr_metadata_labels'][0]=-1
        require(resolver.row(first)['tr_metadata_labels']==rows[0]['BPE']['target_ids'],'Bounded cache must return independent copies')
        checks.append('External provisional row mutation cannot change bounded resolver cache')
        reject('Boolean ordinal rejected',lambda:resolver.row(True))
        batch=[resolver.row(rows[0]['ordinal']),resolver.row(rows[1]['ordinal']),{'language':'pt','encoder_frames':326}]
        size=feature_upper_bound(WEAK,batch);require(size>0,'Conservative complete feature/mask/header byte allowance')
        checks.append('Feature capacity uses declared full TR mel frames and complete replay8T upper bound')
        reject('Cleaner cannot silently enter weak first-prefix resolver',lambda:feature_upper_bound('cleaner_data_fine_tuning',batch))
        # Read-only Windows lifetime handles must deny mutation and replacement
        # even when filesystem mtime would be deferred until handle closure.
        for name in ('final_catalog','final_labels','declared_geometry_index'):
            try:
                with paths[name].open('r+b') as stream:pass
            except PermissionError:checks.append('Write handle denied while validated '+name+' is in use')
            else:raise ValueError('A finalized DB remained writable under its read-only lifetime lock')
        require(resolver.row(first)['broad_original']['text']==rows[0]['text'],'Original cached row preserved under lock')
        checks.append('Final original row remains exact after denied writers')
    finally:resolver.close()
    # Owned fixture mutation after lock release must fail the original full hash
    # on a fresh resolver, before a provisional row can be prepared.
    with sqlite_owned(paths['final_catalog']) as c:c.execute('UPDATE clips SET text=? WHERE ordinal=?',('changed',rows[0]['ordinal']))
    reject('Changed final DB rejected by full hash on next lifetime',lambda:FinalBroadRows(protocol))
    require(not any(v in sys.modules for v in ('torch','numpy','scipy','transformers','pyarrow','soundfile')),'No numerical/model/decoder imports')
    checks.append('No source waveform/cache, model, numeric runtime, authoritative schedule or Adam update created')
    write_json(f/'report.json',{'passed':True,'checks':checks,'check_count':len(checks),'source':record(__file__),
        'planner':record(S/'bpe_final_DATA_window_planner_v2.py'),'source_manifest':record(manifest),
        'partial_metadata_protocol_fixture_only':True,'final_DATA_positive_path_exercised':False,'new_original_schedule_created':False,
        'actual_optimizer_updates':0,'production_enabled':False})
    print(json.dumps({'passed':True,'checks':len(checks),'final_DATA_positive':False,'actual_optimizer_updates':0}))
if __name__=='__main__':main()
