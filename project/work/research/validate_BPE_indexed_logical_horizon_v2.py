"""PRIVATE logical95742-row SQLite/framed-history horizon, no native or payload IO.

Only file-proof boundaries are virtual. Actual index rows, immutable range/hash
checks, complete startup history/catalog accounting and normal hot-load code
are used. Bulk creation is NOT95742 actual public publication operations.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'outputs/parakeet_turkish'));sys.path.insert(0,str(Path(__file__).parent))
import bpe_indexed_owned_checkpoint_storage_v2 as m
import validate_BPE_indexed_future_storage_v2 as tiny
from fastmss_row_serial_cpu_v6 import environment,resource_guard,admit
from bpe_broad_production_state_v1 import canonical,digest,require
from bpe_broad_audio_v1 import record
from atomic_io_retry_v1 import write_json

N=95742
MAX_BYTES=1024**3
MAX_SECONDS=300
PRIVATE=160_000_000
began=time.monotonic();last_guard=None;last_checked=0.;peak=0

def guard():
    global last_checked,last_guard,peak
    now=time.monotonic();require(now-began<MAX_SECONDS,'Logical metadata horizon300s hard fixture cap')
    # All loops check this deadline; expensive paired telemetry at most100ms
    # old, with an independent200ms fail-closed outer watchdog throughout.
    if last_guard is None or now-last_checked>=.1:
        last_guard=resource_guard();last_checked=now
        peak=max(peak,last_guard['worker_private_memory']['private_bytes'])
        require(peak<PRIVATE,'Logical metadata fixture160MB private cap')
    return last_guard

class LogicalIO(m._IndexedProtocol):
    def _verify_entry_files(self,item,stats,*,verify_files):
        require(item['LOGICAL_ONLY_NO_PHYSICAL_PAIR'] is True and stats=={'LOGICAL_ONLY':True},'Explicit virtual file proof IO')
        self.logical_file_accesses+=1
    def _verify_auxiliary_files(self,item,*,verify_archive):
        require(item['LOGICAL_ONLY_NO_PHYSICAL_ARCHIVE'] is True and item['status']=='retired','Explicit virtual retired auxiliary IO')
        self.logical_auxiliary_accesses+=1
    def _verify_auxiliary_archive(self,item):
        require(item['LOGICAL_ONLY_NO_PHYSICAL_ARCHIVE'] is True,'No fake physical archive proof')
        return {'LOGICAL_ONLY':True}
    def _namespace_stat_scan(self,state):
        self.full_namespace_scans+=1
        paths=list(self.folder.iterdir())
        require({p.name for p in paths}=={'ownership.json','state.json','catalog.sqlite','history.jsonl','checkpoints','archive','auxiliary','transient'},
                'Logical fixture actual files only; virtual files were never made')
        return sum(p.is_file() for p in paths),sum(p.stat().st_size for p in paths if p.is_file())
    def _history_frame(self,pin):
        self.history_range_reads+=1;self.history_range_bytes+=pin['bytes']
        return super()._history_frame(pin)

def logical_pin(path,size):
    return {'path':str(path),'bytes':size,'sha256':'0'*64}

def build(engine):
    folder=engine.folder;policy=deepcopy(engine.policy)
    policy.update(normal_window_publications=95230,emergency_publications=512,maximum_publications=N,planned_native_updates=401072,
        maximum_archive_bytes=(N+3)*65536,maximum_retained_metadata_bytes=N*(65536+16384),
        maximum_control_bytes=8*1024**3,maximum_catalog_bytes=2*1024**3,
        maximum_total_auxiliary_archive_bytes=N*65536,maximum_total_owned_allocation_bytes=MAX_BYTES,
        maximum_namespace_audit_gap_operations=16384)
    m.policy_values(policy);engine.policy=policy
    # Private fixture policy artifact is unaccepted and honestly describes the
    # logical file-proof boundary. It cannot pass a public constructor.
    engine.owner['policy']=m.frozen.new_json(folder.parent/'LOGICAL_POLICY_UNACCEPTED.json',policy)
    engine.marker_pin=m.frozen.replace_json(folder/'ownership.json',engine.owner)
    # replace_json returns None in the frozen helper; capture the actual pin.
    engine.marker_pin=record(folder/'ownership.json')
    db=engine._db();db.execute('PRAGMA cache_size=-1024')
    offset=0;root=hashlib.sha256(b'').hexdigest();accounting=deepcopy(m.ZERO_ACCOUNTING);last=None
    with (folder/'history.jsonl').open('wb') as stream:
        for number in range(1,N+1):
            guard();key=f'p{number:09d}';retired=number<N
            item={'id':key,'seq':number,'metadata':logical_pin(folder/'checkpoints'/(key+'.json'),1000),
                'pointer':logical_pin(folder/'checkpoints'/(key+'.pointer.json'),200),
                'payload':logical_pin(folder/'checkpoints'/(key+'.tensors.bin'),1024),
                'archive':logical_pin(folder/'archive'/(key+'.json.gz'),300),'optimizer_updates':number,
                'journal_prefix':{'path':str(folder.parent/'NO_REAL_JOURNAL'),'offset':0,'sha256':'0'*64},
                'run_identity_sha256':digest(engine.owner['run_identity']),'retired':retired,'LOGICAL_ONLY_NO_PHYSICAL_PAIR':True}
            aux={'id':key,'status':'retired','workspace':str(folder/'transient'/key),
                'archive':logical_pin(folder/'auxiliary'/(key+'.auxiliary.gz'),400),'LOGICAL_ONLY_NO_PHYSICAL_ARCHIVE':True}
            accounting['retained_metadata_bytes']+=1200;accounting['archive_bytes']+=300
            accounting['auxiliary_archives']+=1;accounting['auxiliary_bytes']+=400
            accounting['live_pairs']=int(not retired);accounting['live_payload_bytes']=1024 if not retired else 0
            frame={'kind':'ASR18_indexed_owned_storage_history_record_v1','revision':number,'previous_offset':offset,'previous_sha256':root,
                'event':{'event':'PRIVATE_BULK_LOGICAL_ROW_NOT_PUBLIC_PUBLICATION'},'changed_checkpoints':{key:item},
                'file_stats':{key:{'LOGICAL_ONLY':True}},'changed_auxiliary':{key:aux},'accounting_after':deepcopy(accounting)}
            raw=canonical(frame)+b'\n';require(len(raw)<=policy['maximum_history_record_bytes'],'Bounded logical complete frame')
            require(len(canonical(item))<=policy['maximum_catalog_entry_bytes'] and len(canonical(aux))<=policy['maximum_auxiliary_entry_bytes'],'Bounded logical row')
            root=hashlib.sha256(raw).hexdigest();last={'offset':offset,'bytes':len(raw),'sha256':root};stream.write(raw);offset+=len(raw)
            db.execute('INSERT INTO checkpoints VALUES (?,?,?,?)',(key,number,canonical(item).decode(),int(retired)))
            db.execute('INSERT INTO row_integrity VALUES (?,?,?,?,?)',(key,last['offset'],last['bytes'],root,canonical({'LOGICAL_ONLY':True}).decode()))
            db.execute('INSERT INTO auxiliary VALUES (?,?,?,?,?,?)',(key,canonical(aux).decode(),last['offset'],last['bytes'],root,'retired'))
            if number%512==0:
                db.commit();require(offset+(folder/'catalog.sqlite').stat().st_size<=MAX_BYTES,'Logical metadata1GiB predeclared byte cap')
        stream.flush();os.fsync(stream.fileno())
    history={'kind':'ASR18_indexed_chained_history_prefix_v1','offset':offset,'sha256':root}
    catalog={'kind':'ASR18_indexed_catalog_commit_root_v1','revision':N,'history':history,'publications':N,'accounting':accounting,'last_frame':last}
    db.execute('UPDATE control SET value=? WHERE key=?',(canonical(catalog).decode(),'committed_root'));db.commit();db.close()
    m.frozen.replace_json(folder/'state.json',{'owner':engine.marker_pin,'revision':N,'publications':N,'normal_publications':95230,'emergency_publications':512,
        'latest':f'p{N:09d}','winners':[],'history':history,'catalog':catalog,'hot_integrity':{f'p{N:09d}':last},
        'pending_publication':None,'pending_retirement':None,'historical_audit_cursor':0,'auxiliary_audit_cursor':'','accounting':accounting})
    return policy

def main(output):
    environment();admission=admit(PRIVATE,guard=guard);output=Path(output).resolve()
    require(output.is_relative_to(m.ROOT) and not output.exists(),'Fresh E logical fixture only');output.mkdir()
    base=tiny.initialize_private(output/'logical_engine');policy=build(base);base.close()
    engine=object.__new__(LogicalIO);engine.__dict__.update(base.__dict__)
    engine.guard=guard;engine.policy=policy;engine._catalog_startup_complete=False;engine._namespace_iterator=None
    engine._state=None;engine.full_namespace_scans=0;engine.logical_file_accesses=0;engine.logical_auxiliary_accesses=0
    engine.history_range_reads=0;engine.history_range_bytes=0
    startup=time.monotonic();engine._startup_history_audit();state,_,usage=engine._load();startup_wall=time.monotonic()-startup
    startup_reads=engine.history_range_reads;startup_range_bytes=engine.history_range_bytes
    require(state['publications']==N and usage['auxiliary_archives']==N and engine.full_namespace_scans==1,'Full logical index/accounting startup')
    hot=time.monotonic();before=engine.history_range_reads;before_bytes=engine.history_range_bytes;before_access=engine.logical_file_accesses
    for _ in range(100):
        guard();state,rows,_=engine._load();rows[f'p{N:09d}']
    hot_wall=time.monotonic()-hot;hot_reads=engine.history_range_reads-before
    require(engine.full_namespace_scans==1 and hot_reads<=1000 and engine.logical_file_accesses-before_access<=400,
            'Hot lookups stay bounded, do not reread/materialize95742 rows per operation')
    db=engine._db()
    try:plan=db.execute('EXPLAIN QUERY PLAN SELECT id FROM checkpoints WHERE retired=0 ORDER BY seq LIMIT 6').fetchall()
    finally:db.close()
    require('checkpoint_live' in plan[0][3],'Live-set query uses actual SQLite retired/seq index')
    engine.close();guard();total=sum(p.stat().st_size for p in output.rglob('*') if p.is_file());require(total<MAX_BYTES,'Actual1GiB logical metadata cap')
    result={'kind':'ASR18_PRIVATE_logical_indexed95742_horizon_actual_v1','passed':True,'source':record(__file__),'backend_source':record(m.__file__),
        'producer':{'pid':os.getpid(),'ppid':os.getppid()},'logical_checkpoint_rows':N,'logical_retired_auxiliary_rows':N,
        'actual_regular_files':sum(p.is_file() for p in output.rglob('*')),'actual_raw_bytes':total,'maximum_actual_raw_bytes':MAX_BYTES,
        'startup_seconds':startup_wall,'startup_history_range_reads':startup_reads,'startup_history_range_bytes':startup_range_bytes,
        'hot_operations':100,'hot_seconds':hot_wall,'hot_history_range_reads':hot_reads,'hot_history_range_bytes':engine.history_range_bytes-before_bytes,
        'full_namespace_scans':engine.full_namespace_scans,'observed_private_peak_bytes':peak,'wall_seconds':time.monotonic()-began,'CPU_admission':admission,
        'virtual_file_proof_IO':True,'public_native_publication_or_payload_hashing_proved':False,'physical95742_file_layout_proved':False,
        'all_history_per_hot_operation':False,'full_production_capacity_or_emergency_recovery_proved':False,'production_enabled':False,
        'limits':['Bulk logical metadata creation skips public writer/owner/OS-fsync cadence; it is not95742 real native publications.',
            'Only physical pin/stat/archive IO is virtual; real SQLite/hash-chain/range/accounting and hot code run.',
            'No audio/model/tensor payload exists or is read; no old/current/original path is mutated.']}
    write_json(output/'report.json',result);print(json.dumps({'passed':True,'report':record(output/'report.json'),'private':peak}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)
