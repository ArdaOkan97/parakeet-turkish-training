"""Synthetic internal durability controls; NOT public policy/native proof.

Does not create accepted ROOT reviews, load checkpoints or call a model. The
public constructor is deliberately not bypassed under a fake approval: private
engine objects are visibly synthetic, and that limited scope is in the report.
"""
import argparse
from copy import deepcopy
import ctypes
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

P = Path(__file__).resolve().parents[2]; S = P/'outputs/parakeet_turkish'
sys.path.insert(0, str(S)); sys.path.insert(0, str(Path(__file__).parent))
import bpe_full_run_durable_owner_adapter_v2 as module
from bpe_full_run_lifecycle_v2 import _LifecycleProtocol
from validate_BPE_full_run_lifecycle_CPU_v2 import metadata_state, metadata_plan, pointer
from fastmss_row_serial_cpu_v6 import environment, resource_guard, admit
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record, require
from bpe_broad_production_state_v1 import digest


class SyntheticSchedule:
    def __init__(self, state): self.value = deepcopy(state)
    def state_dict(self): return deepcopy(self.value)


class SyntheticLifecycle:
    def __init__(self, state, identity, restored=None):
        self.identity = identity; self.schedule = SyntheticSchedule(state)
        self.protocol = _LifecycleProtocol(state, identity, None, restored=restored)
    def state_dict(self): return deepcopy(self.protocol.state)


class SyntheticProofIO:
    """No actual source/lease/native proof is produced by this test adapter."""
    def __init__(self): self.calls = []
    def verify(self, name, request):
        guard(); self.calls.append({'name': name, 'synthetic_in_memory_IO': True})


def guard():
    value = resource_guard()
    require(value['worker_private_memory']['private_bytes'] < 100_000_000, 'Fixture private100MB cap')
    return value


def main(output):
    began = time.monotonic(); environment(); admission = admit(100_000_000, guard=guard)
    output = Path(output).resolve()
    require(output.is_relative_to(module.E) and not output.exists(), 'Fresh E-only synthetic fixture')
    output.mkdir(); checks = []; engines = []; samples = [guard()]
    frozen = record(module.__file__)
    def check(name, value):
        require(value, name); checks.append(name); samples.append(guard())
    def reject(name, operation):
        try: operation()
        except (ValueError, KeyError, TypeError, OSError, sqlite3.Error):
            checks.append(name); samples.append(guard()); return
        raise ValueError('Expected rejection: '+name)
    def engine(name):
        path = output/name; path.mkdir()
        x = object.__new__(module.PackedDurableLifecycleAdapter)
        x.folder = path; x.identity = {'synthetic_only': True, 'not_final_DATA': True, 'maximum_pass_window_publications': 4}
        x.lifecycle = SyntheticLifecycle(metadata_state(), x.identity); x.proofs = SyntheticProofIO()
        x.policy_pin = {'synthetic_private_engine_policy': True}
        x.policy = {'maximum_records':64, 'maximum_stream_bytes':4*1024**2,
            'maximum_index_bytes':4*1024**2, 'maximum_record_bytes':65536,
            'maximum_pending_windows':8, 'maximum_pending_publications':8}
        x.creator_pid = os.getpid(); x.poisoned = False; x.lock = None
        x.head = None; x.head_pin = None; x.latest_pin = None; x._reconcile_tail = False
        write_json(path/'owner.json', {'kind':'SYNTHETIC_PRIVATE_ENGINE_NOT_ACTUAL_OWNER', 'synthetic_only':True})
        x.owner = record(path/'owner.json'); (path/'records.bin').touch(); x._acquire_lock()
        db = x._db()
        try: db.execute('CREATE TABLE heads (seq INTEGER PRIMARY KEY, offset INTEGER UNIQUE NOT NULL, framed_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL)'); db.commit()
        finally: db.close()
        x._append('genesis', x.lifecycle.state_dict(), x._empty_pending(), {'synthetic_only':True}, None)
        engines.append(x); return x
    def packet(x):
        before = metadata_state(); plan = metadata_plan(before, 2); after = plan['final_private_planning_sampler']
        table = {}
        def pin(name, value=None):
            p = {'synthetic_in_memory_reference': name}
            if value is not None: table[name] = value
            return p
        pointer_pin = pin('pointer', pointer(after, diagnostic=False))
        capture = {'pointer':pointer_pin, 'checkpoint':pin('checkpoint'), 'checkpoint_persistence_proof':pin('persistence'),
            'pre_save_fingerprint':pin('PRE_save'), 'next_draw_packet':pin('next_draw'),
            'publication_role':'completed_window_end', 'checkpoint_persistence_verified':True,
            'exact_GPU_restore_performed':False, 'current_latest_recovery_required_before_phase_DEV_winner_advance':True}
        completion = {'kind':'SYNTHETIC_IN_MEMORY_NOT_ACTUAL_NATIVE', 'passed':True, 'diagnostic_only':False,
            'run_identity_sha256':digest(x.identity), 'lifecycle_parent_head_sha256':x.head_pin['sha256'],
            'actual_new_updates':2, 'global_new_updates':2, 'latest_pointer':pointer_pin,
            'all_checkpoint_persistence_verified':True, 'exact_GPU_restores_deferred':True, 'checkpoints':[capture]}
        before_pin = pin('before', before); plan_pin = pin('plan', plan); completion_pin = pin('completion', completion)
        p = {'kind':'ASR18_actual_orphan_committed_window_for_durable_reconciliation_v2', 'diagnostic_only':False,
            'prior_head':deepcopy(x.head_pin), 'run_identity_sha256':digest(x.identity),
            'before_sampler_state':before_pin, 'coordinator_plan':plan_pin, 'actual_completion':completion_pin}
        packet_pin = pin('packet', p)
        candidate = module.derive_orphan_protocol_state(durable_state=x.head['lifecycle'], before_state=before,
            plan=plan, pointer_binding=pointer_pin, pointer=table['pointer'], after_state=after, identity=x.identity)
        return table, packet_pin, before, after, candidate
    real_read, real_bound = module.read_bound, module.bound
    def memory_IO(table):
        module.read_bound = lambda pin: deepcopy(table[pin['synthetic_in_memory_reference']]) if 'synthetic_in_memory_reference' in pin else real_read(pin)
        module.bound = lambda pin: pin if 'synthetic_in_memory_reference' in pin else real_bound(pin)
    try:
        check('No numerical/ML/audio imports', not any(k=='torch' or k.startswith(('torch.','numpy','scipy','transformers','pyarrow')) for k in sys.modules))
        x = engine('synthetic_native_orphan')
        old_pin = deepcopy(x.head_pin); old_bytes = (x.folder/'records.bin').read_bytes()
        table, packet_pin, before, after, candidate = packet(x)
        check('Pure orphan derivation does not mutate before/after inputs', before == metadata_state() and after == metadata_plan(before,2)['final_private_planning_sampler'])
        x.lifecycle = SyntheticLifecycle(after, x.identity, restored=candidate)
        memory_IO(table); x._reconcile_orphan_native(packet_pin)
        check('Actual-current metadata candidate reconciles without old schedule rollback', x.lifecycle.schedule.state_dict()==after and x.head['lifecycle']==candidate)
        check('Committed orphan persists explicit pending prefix, not restore success', x.head['pending']['windows']==1 and x.head['pending']['publications']==1)
        check('Old logical descriptor bytes survive a later append', (x.folder/'records.bin').read_bytes().startswith(old_bytes) and x._read_descriptor(old_pin)['head']['sequence']==1)
        check('Pass tail blocks DEV until current recovery', x.next_action()['kind']=='actual_native_exit_then_current_latest_recovery_and_persistence_prefix_required')
        x._audit_chain(); checks.append('Complete packed prefix/index audit passed')
        bad = deepcopy(x.head_pin); bad['offset'] = True
        reject('Boolean range offset rejected', lambda:x._read_descriptor(bad))
        bad = deepcopy(x.head_pin); bad['sha256'] = '0'*64
        reject('Wrong raw frame hash rejected', lambda:x._read_descriptor(bad))
        bad = deepcopy(x.head_pin); bad['framed_bytes'] += 1
        reject('Wrong frame length rejected', lambda:x._read_descriptor(bad))
        before_size=(x.folder/'records.bin').stat().st_size
        reject('Oversize metadata rejects before append', lambda:x._append('synthetic_oversize',x.lifecycle.state_dict(),x.head['pending'],{'blob':'x'*65536},None))
        check('Oversize did not grow stream or move latest', (x.folder/'records.bin').stat().st_size==before_size and x.head['sequence']==2)
        # Genuine file/index writes, with instance-local controlled commit error.
        y=engine('synthetic_frame_before_index'); saved_db=y._db
        class FailCommit:
            def __init__(self, db):self.db=db
            def execute(self,*a,**k):return self.db.execute(*a,**k)
            def commit(self):raise OSError('SYNTHETIC crash before index commit')
            def close(self):self.db.close()
        y._db=lambda:FailCommit(saved_db()); old_latest=record(y.folder/'latest.json')
        reject('Frame-before-index failure preserves old latest', lambda:y._append('synthetic_frame_crash',y.lifecycle.state_dict(),y._empty_pending(),{'synthetic_only':True},None))
        check('Unknown frame tail retained after rollback', y.poisoned and record(y.folder/'latest.json')==old_latest and (y.folder/'records.bin').stat().st_size>y.head['total_event_bytes'])
        y._db=saved_db; y.poisoned=False; y._reconcile_tail=True
        y._append('synthetic_frame_crash',y.lifecycle.state_dict(),y._empty_pending(),{'synthetic_only':True},None); y._reconcile_tail=False
        check('Exact independently expected internal tail can be adopted without duplicate frame', y.head['sequence']==2 and (y.folder/'records.bin').stat().st_size==y.head['total_event_bytes'])
        # Metadata acknowledgement tails have no supported actual-native packet.
        for name,operation in (('recovery_ack','current_latest_recovery_and_persistence_prefix_acknowledged'),('DEV_ack','complete_DEV_and_winner_acknowledged')):
            z=engine('synthetic_'+name+'_tail'); published=record(z.folder/'latest.json'); old_write=module.write_json
            def fail_latest(path,value):
                if Path(path).name=='latest.json':raise OSError('SYNTHETIC crash after index, before latest')
                return old_write(path,value)
            module.write_json=fail_latest
            try: reject(name+' index-before-latest crash is fail closed',lambda:z._append(operation,z.lifecycle.state_dict(),z._empty_pending(),{'synthetic_ack_not_actual':True},None))
            finally:module.write_json=old_write
            tail=(z.folder/'records.bin').read_bytes(); z.poisoned=False
            check(name+' unacknowledged tail and old authority preserved',record(z.folder/'latest.json')==published and len(tail)>z.head['total_event_bytes'])
            table2,pin2,b2,a2,c2=packet(z);z.lifecycle=SyntheticLifecycle(a2,z.identity,restored=c2);memory_IO(table2)
            reject(name+' tail cannot masquerade as a native orphan',lambda:z._reconcile_orphan_native(pin2))
            check(name+' rejection did not truncate or replace tail',(z.folder/'records.bin').read_bytes()==tail and record(z.folder/'latest.json')==published)
        module.read_bound, module.bound=real_read,real_bound
        check('Frozen adapter source unchanged',record(module.__file__)==frozen)
        counts=[]
        for item in engines:
            db=item._db()
            try: largest=db.execute('SELECT max(framed_bytes) FROM heads').fetchone()[0]
            finally:db.close()
            counts.append({'fixture':item.folder.name,'regular_files':sum(1 for p in item.folder.iterdir() if p.is_file()),
                'stream_bytes':(item.folder/'records.bin').stat().st_size,'index_bytes':(item.folder/'heads.sqlite').stat().st_size,
                'acknowledged_records':item.head['sequence'],'largest_indexed_frame_bytes':largest})
        check('Fixed files avoid per-window event/head allocation',all(v['regular_files']==5 for v in counts))
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        sectors=ctypes.c_ulong();bytes_sector=ctypes.c_ulong();free=ctypes.c_ulong();total=ctypes.c_ulong()
        require(kernel.GetDiskFreeSpaceW('E:\\',ctypes.byref(sectors),ctypes.byref(bytes_sector),ctypes.byref(free),ctypes.byref(total)), 'Actual E allocation geometry')
        cluster=sectors.value*bytes_sector.value
        rounded=sum(((p.stat().st_size+cluster-1)//cluster)*cluster for p in output.rglob('*') if p.is_file())
        require(rounded<128*1024**2,'Bounded synthetic regular-file allocation128MiB')
        samples.append(guard())
        result={'kind':'ASR18_synthetic_packed_durable_adapter_metadata_controls_v1','passed':True,'source':frozen,
            'producer':{'pid':os.getpid(),'ppid':os.getppid()},'checks':checks,'check_count':len(checks),'CPU_admission':admission,
            'observed_private_peak_bytes':max(v['worker_private_memory']['private_bytes'] for v in samples),
            'E_allocation_unit_bytes':cluster,'fixture_regular_file_allocation_rounded_bytes':rounded,'physical_fixtures':counts,
            'maximum_observed_frame_bytes':max(item['largest_indexed_frame_bytes'] for item in counts),'hard_frame_cap_including_header':65536,
            'allocation_measurement_excludes_final_report_and_guard_receipts':True,
            'startup_and_hot_memory_scope':'Observed after stdlib/protocol imports and during bounded internal synthetic operations; continuous outer guard closure is separate',
            'wall_seconds':time.monotonic()-began,'public_constructor_or_ROOT_policy_proved':False,
            'actual_native_or_current_checkpoint_or_DEV_or_winner_proved':False,'actual_owner_source_verifiers_proved':False,
            'synthetic_in_memory_proof_IO_only':True,'unsupported_metadata_ack_tails_preserved_and_rejected':True,
            'metadata_tail_recovery_implemented':False,'emergency_support_proved':False,'no_original_or_current_artifact_opened':True}
        write_json(output/'report.json',result);print(json.dumps({'passed':True,'checks':len(checks),'report':record(output/'report.json')}),flush=True)
    finally:
        module.read_bound,module.bound=real_read,real_bound
        for item in engines:item.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    main(parser.parse_args().output)
