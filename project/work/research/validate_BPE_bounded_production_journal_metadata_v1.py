"""NEW owned small journal/prefix/resume controls; explicitly synthetic exit IO.

Private toy quota2MiB/count4 exercises the exact rounded arithmetic without
allocating16GiB or513 files. No real native exit, optimizer/sampler or old file.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
import bpe_bounded_production_journal_v4 as j
from fastmss_row_serial_cpu_v6 import environment,resource_guard,admit
from bpe_broad_production_state_v1 import require
from bpe_broad_audio_v1 import record
from atomic_io_retry_v1 import write_json

def guard():
    value=resource_guard();require(value['worker_private_memory']['private_bytes']<100_000_000,'Journal metadata100MB')
    return value

def synthetic_exit_verifier(value,saved):
    require(value=={'kind':'PRIVATE_SYNTHETIC_EXIT_IO_NO_NATIVE_PROCESS','passed':True},'Never a real process proof')
    require(type(saved['offset']) is int and saved['offset']>=0,'Explicit checkpoint prefix input')
    guard();return True

def main(output,namespace):
    environment();admission=admit(100_000_000,guard=guard);began=time.monotonic();checks=[];samples=[guard()]
    output=Path(output).resolve();namespace=Path(namespace).resolve()
    require(output.is_relative_to(j.ROOT) and not output.exists(),'Fresh E metadata reports')
    require(namespace==j.ROOT/'runs/_owned_validation/journal_metadata_fixture_01' and not namespace.exists(),'Exact fresh disposable fixture namespace')
    output.mkdir();source=record(j.__file__);original_quota,original_max=j.QUOTA,j.MAX_JOURNALS
    # PRIVATE parameter substitution only; saved owner marker explicitly records
    # this toy quota/count and private identity. Not a production capacity test.
    j.QUOTA=2*1024**2;j.MAX_JOURNALS=4
    identity={'kind':'PRIVATE_JOURNAL_METADATA_FIXTURE_ONLY','production_enabled':False,'actual_native_authority':False}
    ns=j.JournalNamespace(namespace,identity=identity,guard=guard,create=True)
    def check(name,condition):require(condition,name);checks.append(name);samples.append(guard())
    def reject(name,operation):
        try:operation()
        except (ValueError,KeyError,TypeError,OSError):checks.append(name);samples.append(guard());return
        raise ValueError('Expected journal rejection: '+name)
    exit_io={'kind':'PRIVATE_SYNTHETIC_EXIT_IO_NO_NATIVE_PROCESS','passed':True}
    try:
        a=ns.register('commits_A.jsonl');b=ns.register('commits_B.jsonl')
        wa=ns.writer(a);first=wa.append({'kind':'PRIVATE_EVENT_A','target_ids':[1,300,8710]})
        check('Full actual prefix hash and exact binding',wa.verify(first) and first['sha256']==record(a)['sha256'])
        wb=ns.writer(b)
        check('Switch captures actual previously grown A, not startup0',wb.other_bytes==j.allocated(a.stat().st_size)==j.ALLOCATION_UNIT)
        saved_b=wb.append({'kind':'PRIVATE_EVENT_B','event':1})
        reject('Old A writer cannot append after selecting B',lambda:wa.append({'PRIVATE_STALE':True}))
        wa2=ns.writer(a);wa2.append({'kind':'PRIVATE_EVENT_A','event':2})
        count_before=len(ns.paths)
        reused,receipt=ns.resume(saved_b,destination_name='commits_unused.jsonl',actual_exit=exit_io,verify_quiescence=synthetic_exit_verifier)
        check('Whole exact closed B reuses same path without copy/extra registry row',receipt['same_whole_closed_journal_reused'] and receipt['copied_bytes']==0 and reused.path==b and len(ns.paths)==count_before)
        check('Reuse existing B captures grown A actual allocated size',reused.other_bytes==j.allocated(a.stat().st_size))
        # Unknown tail is deliberately NEW fixture data; preserve it exactly.
        with b.open('ab') as stream:stream.write(b'PRIVATE_UNKNOWN_UNACKNOWLEDGED_TAIL\n');stream.flush();os.fsync(stream.fileno())
        before_b=record(b)
        branch,receipt=ns.resume(saved_b,destination_name='commits_branch.jsonl',actual_exit=exit_io,verify_quiescence=synthetic_exit_verifier)
        check('Unknown source tail remains byte exact and is never replayed',record(b)==before_b and receipt['unknown_tail_preserved_bytes']>0 and receipt['tail_replayed_or_sampler_advanced'] is False)
        check('Branch contains exact acknowledged prefix only',not receipt['same_whole_closed_journal_reused'] and receipt['copied_bytes']==saved_b['offset'] and branch.binding()['sha256']==saved_b['sha256'])
        reject('Old source binding notices unknown appended suffix',wb.binding)
        reject('No quiescence means no resume authority',lambda:ns.resume(saved_b,destination_name='commits_no_exit.jsonl',actual_exit={},verify_quiescence=lambda *_:False))
        bad=deepcopy(saved_b);bad['sha256']='f'*64
        reject('Wrong checkpoint prefix hash rejected',lambda:ns.resume(bad,destination_name='commits_bad_sha.jsonl',actual_exit=exit_io,verify_quiescence=synthetic_exit_verifier))
        bad=deepcopy(saved_b);bad['offset']=True
        reject('Boolean prefix length rejected',lambda:ns.resume(bad,destination_name='commits_bool.jsonl',actual_exit=exit_io,verify_quiescence=synthetic_exit_verifier))
        bad=deepcopy(saved_b);bad['offset']=b.stat().st_size+1
        reject('Truncated checkpoint prefix rejected',lambda:ns.resume(bad,destination_name='commits_too_long.jsonl',actual_exit=exit_io,verify_quiescence=synthetic_exit_verifier))
        prospective=namespace/'commits_excess.jsonl'
        reject('Prospective branch reservation counts rounded aggregate before creation',lambda:ns.register(prospective.name,prospective_bytes=j.QUOTA))
        check('Denied reservation leaves no fresh file',not prospective.exists())
        c=ns.register('commits_C.jsonl');wc=ns.writer(c)
        for index in range(10):
            previous=record(c);offset=wc.binding()['offset']
            try:wc.append({'kind':'PRIVATE_QUOTA_EVENT','i':index,'data':'x'*100000})
            except ValueError:
                check('Rounded aggregate quota stops BEFORE crossing file write',record(c)==previous and wc.binding()['offset']==offset)
                break
        else:raise ValueError('Expected2MiB rounded toy quota stop')
        check('Actual rounded journal aggregate never exceeded toy quota',sum(j.allocated(path.stat().st_size) for path in ns.paths)<=j.QUOTA)
        reject('Bounded registered journal count rejected before creation',lambda:ns.register('commits_fifth.jsonl'))
        reject('Source namespace cannot escape through parent traversal',lambda:ns.register('../commits_escape.jsonl'))
        reject('Unregistered path cannot obtain a writer',lambda:ns.writer(namespace/'commits_unknown.jsonl'))
        with patch.object(Path,'lstat',return_value=SimpleNamespace(st_file_attributes=1024)):
            reject('Synthetic reparse attribute rejected without creating a link',lambda:j.safe(a,namespace))
        # Genuine tiny orphan models file-before-registry crash; never deleted.
        orphan=namespace/'commits_unregistered_orphan.jsonl';orphan.write_bytes(b'PRIVATE_ORPHAN')
        reject('Startup rejects real unregistered orphan and preserves it',lambda:j.JournalNamespace(namespace,identity=identity,guard=guard))
        check('Unknown orphan still exists with original bytes',orphan.read_bytes()==b'PRIVATE_ORPHAN')
        reject('Wrong namespace owner identity rejected',lambda:j.JournalNamespace(namespace,identity={'other':True},guard=guard))
        # Current writer identity check cannot accept even semantically unchanged
        # registry byte drift. Preserve the tail; no rollback or auto retry.
        with ns.registry.open('ab') as stream:stream.write(b'\n');stream.flush();os.fsync(stream.fileno())
        reject('Current owner registry byte drift rejected',ns.check)
        check('Source component remained byte exact',record(j.__file__)==source)
        require(time.monotonic()-began<60,'Journal fixture60s')
        allocated=sum(j.allocated(p.stat().st_size) for p in namespace.rglob('*') if p.is_file())
        require(allocated<16*1024**2,'Small fixture<16MiB actual regular-file allocation')
        result={'kind':'ASR18_PRIVATE_bounded_journal4_small_metadata_controls_v1','passed':True,'source':record(__file__),
            'journal_source':source,'producer':{'pid':os.getpid(),'ppid':os.getppid()},'checks':checks,'check_count':len(checks),
            'CPU_admission':admission,'observed_private_peak_bytes':max(s['worker_private_memory']['private_bytes'] for s in samples),
            'wall_seconds':time.monotonic()-began,'namespace':str(namespace),'namespace_owner':record(ns.owner),
            'toy_quota_bytes':j.QUOTA,'toy_maximum_journals':j.MAX_JOURNALS,'official_quota_not_bulk_allocated':original_quota,
            'official_count_not_bulk_created':original_max,'actual_regular_file_allocation_bytes':allocated,
            'actual_rounding_unit_bytes':j.ALLOCATION_UNIT,'native_exit_verifier_IO_explicitly_synthetic':True,
            'reparse_negative_used_synthetic_attribute_injection':True,'unknown_orphan_and_registry_tail_preserved':True,
            'actual_native_or_fullrun_capacity_proved':False,'old_current_original_or_diagnostic_files_changed':False,'production_enabled':False}
        write_json(output/'report.json',result);print(json.dumps({'passed':True,'checks':len(checks),'report':record(output/'report.json')}),flush=True)
    finally:j.QUOTA=original_quota;j.MAX_JOURNALS=original_max

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--namespace',required=True)
    a=p.parse_args();main(a.output,a.namespace)
