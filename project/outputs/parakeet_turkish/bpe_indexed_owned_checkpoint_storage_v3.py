"""Inactive future-only indexed storage over frozen storage2 operations.

Retains the original semantic/payload verification, mapping-release owner proof,
journal-prefix checks and exact obsolete-payload retirement. Only catalog/history
loading and bookkeeping change. No current/original namespace adoption or CLI.
Historical stat checks are immediate; a bounded rotating hash audit has explicit
latency. Live latest/winners and every operation target are hashed on access.
"""
from collections.abc import Mapping
from copy import deepcopy
import gzip
import hashlib
import inspect
import json
import lzma
import os
from pathlib import Path
import re
import shutil
import sqlite3
import struct

import bpe_owned_checkpoint_storage_v2 as frozen
from bpe_broad_audio_v1 import record
from bpe_broad_production_state_v1 import canonical, digest, require
from bpe_training_schedule_v1 import WEAK, CLEANER

ROOT, RESERVE = frozen.ROOT, frozen.RESERVE
safe, bound, read, integer = frozen.safe, frozen.bound, frozen.read, frozen.integer
POLICY = 'ASR18_reviewed_indexed_window_end_checkpoint_storage_policy_v1'
OWNER = 'ASR18_future_indexed_checkpoint_namespace_v1'
ZERO_ACCOUNTING={'live_payload_bytes':0,'retained_metadata_bytes':0,'archive_bytes':0,
    'live_pairs':0,'auxiliary_archives':0,'auxiliary_bytes':0,'active_auxiliary_workspaces':0}
AUX_MAGIC = b'ASR18_FUTURE_OWNED_AUXILIARY_1\n'
AUXILIARY_CODEC={'container':'XZ','algorithm':'LZMA2','preset':6,'check':'CRC64',
    'decoder_memory_limit_bytes':32*1024**2,'maximum_decoder_chunk_bytes':1024**2}
AUX_ROLES = frozenset(('pre_save','next_draw','persistence','window_plan','CPU_provenance',
    'native_completion','source_manifest','DEV_receipt','durable_receipt','guard_closure','checkpoint_pointer'))


def _read_exact(stream, count):
    result=bytearray()
    while len(result)<count:
        block=stream.read(min(1024**2,count-len(result)))
        require(block,'Complete bounded auxiliary archive; no truncated member')
        result.extend(block)
    return bytes(result)


class _CappedWriter:
    def __init__(self,stream,limit):self.stream=stream;self.limit=limit;self.count=0
    def write(self,value):
        require(self.count+len(value)<=self.limit,'Compressed complete auxiliary archive cap; partial file remains unresolved')
        written=self.stream.write(value);require(written==len(value),'Complete archive write');self.count+=written;return written
    def flush(self):return self.stream.flush()


def codec_admission(increment,guard):
    """DISTINCT prospective parent metadata phase, not the old64MB write stage."""
    from fastmss_row_serial_cpu_v6 import resource_guard,admit
    guard();value=resource_guard()
    require(value['worker_private_memory']['private_bytes']+increment<2_000_000_000,
            'Codec prospective private stays under unchanged parent2GB')
    return admit(increment,guard=resource_guard)


class _BoundedAuxiliaryXZReader:
    """One XZ stream; bounded dictionary, compressed input and total raw output."""
    def __init__(self,path,*,guard,compressed_limit,output_limit,memory_limit=32*1024**2):
        self.path=Path(path);self.guard=guard;self.limit=output_limit;self.total=0;self.finished=False
        require(type(output_limit) is int and 0<output_limit<=34*1024**2,'Bounded complete raw+header+framing output domain')
        require(0<compressed_limit<=2*1024**2 and self.path.stat().st_size<=compressed_limit,
                'Actual compressed auxiliary cap BEFORE decoder allocation')
        require(type(memory_limit) is int and 0<memory_limit<=32*1024**2,'Bounded XZ decoder memory')
        self.decoder=lzma.LZMADecompressor(format=lzma.FORMAT_XZ,memlimit=memory_limit)
        self.stream=self.path.open('rb');self.compressed_limit=compressed_limit;self.compressed_read=0
    def __enter__(self):return self
    def __exit__(self,*exc):self.stream.close()
    def read(self,count):
        require(type(count) is int and 0<=count<=1024**2,'Bounded<=1MiB explicit decoder reads; no read-all')
        result=bytearray()
        while len(result)<count and not self.finished:
            self.guard()
            if self.decoder.needs_input:
                block=self.stream.read(65536);require(block,'Truncated/unresolved XZ stream')
                self.compressed_read+=len(block);require(self.compressed_read<=self.compressed_limit,'Compressed byte cap')
            else:block=b''
            value=self.decoder.decompress(block,max_length=count-len(result))
            self.total+=len(value);require(self.total<=self.limit,'Cumulative decompressed auxiliary raw cap')
            result.extend(value)
            if self.decoder.eof:
                require(self.decoder.check==lzma.CHECK_CRC64 and not self.decoder.unused_data and self.stream.read(1)==b'',
                        'Exactly one complete CRC64 XZ stream; no concatenated/trailing data')
                self.finished=True
        return bytes(result)


def policy_values(value):
    require(value['kind'] == POLICY and value['production_enabled'] is False
            and value['retire_existing_or_original_artifacts'] is False
            and value['retire_owned_metadata_or_pointer_files'] is False
            and value['publication_cadence'] == 'completed_window_end_plus_explicit_bounded_emergency_v1',
            'NEW exact window-end policy; no old maxpubs>=updates assumption or old-file authority')
    for key in ('maximum_payload_bytes','maximum_metadata_bytes','maximum_pointer_bytes','maximum_archive_record_bytes',
                'maximum_publications','maximum_archive_bytes','maximum_retained_metadata_bytes','maximum_control_bytes',
                'maximum_history_record_bytes','maximum_catalog_bytes','maximum_catalog_entry_bytes',
                'maximum_history_fixed_overhead_bytes','maximum_history_file_stats_bytes',
                'maximum_history_rows_per_operation','normal_window_publications','planned_native_updates',
                'maximum_auxiliary_archive_bytes','maximum_auxiliary_raw_bytes','maximum_auxiliary_entry_bytes','maximum_auxiliary_header_bytes',
                'maximum_auxiliary_files','maximum_auxiliary_directories','maximum_total_auxiliary_archive_bytes',
                'maximum_total_owned_allocation_bytes','maximum_retained_auxiliary_workspaces',
                'historical_hash_rows_per_operation','E_allocation_unit_bytes','namespace_entries_per_operation',
                'maximum_namespace_audit_gap_operations'):
        integer(value[key],1)
    require(integer(value['emergency_publications'])+value['normal_window_publications']==value['maximum_publications']
            and value['steady_live_pairs']==4 and value['maximum_inflight_publications']==1 and value['hard_live_pair_peak']==5
            and value['maximum_catalog_entry_bytes']<=4096 and value['maximum_auxiliary_entry_bytes']<=4096
            and value['maximum_history_fixed_overhead_bytes']<=2048 and value['maximum_history_file_stats_bytes']<=1024
            and value['maximum_auxiliary_header_bytes']<=512*1024
            and value['maximum_history_record_bytes']<=65536 and value['maximum_history_rows_per_operation']<=5
            and value['maximum_auxiliary_files']<=512 and value['maximum_auxiliary_raw_bytes']<=32*1024**2
            and value['maximum_auxiliary_archive_bytes']<=2*1024**2,
            'Explicit bounded normal/emergency counts, row/frame/auxiliary geometry and live set')
    require(value['maximum_retained_auxiliary_workspaces']==5
            and value['maximum_auxiliary_directories']<=128,
            'Only latest/three qualified and one in-flight future workspaces; exact bounded directory inventory')
    require(value['maximum_archive_bytes']>=(value['maximum_publications']+3)*value['maximum_archive_record_bytes']
            and value['maximum_retained_metadata_bytes']>=value['maximum_publications']*(value['maximum_metadata_bytes']+value['maximum_pointer_bytes'])
            and value['maximum_total_auxiliary_archive_bytes']>=value['maximum_publications']*value['maximum_auxiliary_archive_bytes'],
            'Full declared metadata/archive/auxiliary horizon, not minimum window count')
    # Each publication permits reserve+publish, one later payload retirement,
    # and auxiliary reserve+archive+retire; at most three winner protections.
    # This bounds byte volume, independently of how rows are grouped per event.
    n=value['maximum_publications']
    maximum_history=(2*n+3)*(value['maximum_catalog_entry_bytes']+value['maximum_history_file_stats_bytes']) \
        +3*n*value['maximum_auxiliary_entry_bytes']+(6*n+3)*value['maximum_history_fixed_overhead_bytes']
    require(maximum_history+value['maximum_catalog_bytes']+1024**2<=value['maximum_control_bytes'],
            'Complete operation/changed-row/stat/framing horizon fits control suballocation')
    return value


def verify_capacity_evidence(evidence,policy,review):
    """Bind ROOT's conditional arithmetic without treating it as implementation proof."""
    require(set(evidence)=={'capacity','actual0','source','base_source'},'Exact capacity2 evidence set')
    value=read(evidence['capacity'],maximum=4*1024**2);proof=read(evidence['actual0'])
    bound(evidence['source']);bound(evidence['base_source'])
    require(value['kind']=='ASR18_conditional_full_run_cadence_capacity_arithmetic_v2'
            and value['production_enabled'] is False and value['launch_ready'] is False
            and value['conditional_arithmetic_fits'] is True
            and proof['kind']=='ROOT_actual_small_metadata_directory_growth_delta_TOOL0_v2'
            and proof['actual_tool_exit_code']==0 and proof['actual_owned_tool_completion_observed'] is True
            and type(proof['actual_tool_chunk_id']) is str and proof['actual_tool_chunk_id']
            and proof['capacity']==evidence['capacity'] and proof['source']==evidence['source']
            and proof['base_source']==evidence['base_source'] and proof['guard_clean_joined'] is True,
            'Genuine bounded arithmetic TOOL0, not native/layout/resource approval')
    bound(proof['report']);bound(proof['guard']);bound(proof['prior_actual0'])
    limits=value['limits'];caps=limits['artifact_caps_bytes'];unit=integer(policy['E_allocation_unit_bytes'],1)
    require(value['maximum_normal_publications']==policy['normal_window_publications']
            and value['maximum_emergency_publications']==policy['emergency_publications']
            and value['maximum_publications']==policy['maximum_publications']
            and value['total_native_updates']==policy['planned_native_updates']
            and value['disk_observation']['allocation_unit_bytes']==unit
            and caps=={'payload':policy['maximum_payload_bytes'],'metadata':policy['maximum_metadata_bytes'],
                'pointer':policy['maximum_pointer_bytes'],'archive':policy['maximum_archive_record_bytes'],
                'window_auxiliary':policy['maximum_auxiliary_archive_bytes']}
            and policy['maximum_total_owned_allocation_bytes']==value['whole_required_bytes_including_reserve']-RESERVE,
            'Exact final population/cadence/artifact caps/allocation horizon; no alternate minimum-window assumption')
    require(policy['maximum_catalog_bytes']<=value['owned_index_bytes']
            and policy['maximum_control_bytes']<=value['other_journal_log_control_growth_bytes']-2*1024**3
            and 5+policy['maximum_retained_auxiliary_workspaces']*policy['maximum_auxiliary_directories']
                <=limits['maximum_retained_owned_directories']+limits['maximum_inflight_owned_directories']
            and review['capacity_evidence']==evidence
            and review['transient_feature_raw_media_scratch_suballocation_bytes']>=4*1024**3
            and integer(review['native_journal_log_control_suballocation_bytes'],1)
            and review['transient_feature_raw_media_scratch_suballocation_bytes']
                +review['native_journal_log_control_suballocation_bytes']+policy['maximum_control_bytes']
                <=value['other_journal_log_control_growth_bytes'],
            'Explicit index/control/transient and directory suballocations inside the conditional whole budget')
    # The separate accepted ROOT review must assess the actual packing/cadence,
    # emergency handling, archive sizes and source/native integrations. This
    # conditional arithmetic never grants those approvals itself.
    return value


class _Rows(Mapping):
    """Exact primary-key lookups; never materialize the full historical catalog."""
    def __init__(self, store):self.store=store
    def __getitem__(self,key):return self.store._entry(key,verify_files=True)
    def __iter__(self):raise ValueError('Full catalog materialization is unsupported; use streaming audit')
    def __len__(self):return self.store._state['publications']
    def __contains__(self,key):
        db=self.store._db()
        try:return db.execute('SELECT 1 FROM checkpoints WHERE id=?',(key,)).fetchone() is not None
        finally:db.close()


class _IndexedProtocol(frozen._StorageProtocol):
    def __init__(self,marker_pin,*,checkpoint_verifier,guard):
        self.guard=guard;self.verifier=checkpoint_verifier;self.synthetic=False
        self.marker_pin=deepcopy(marker_pin);self.owner=read(marker_pin);self.folder=safe(self.owner['namespace'])
        require(self.owner['kind']==OWNER and self.owner['owner']==frozen.OWNER
                and Path(marker_pin['path']).resolve()==self.folder/'ownership.json'
                and self.owner['component_source']==record(__file__)
                and self.owner['frozen_operation_source']==record(frozen.__file__), 'Exact indexed future owner/source')
        self.policy=policy_values(read(self.owner['policy']));self._state=None
        self._catalog_startup_complete=False;self._namespace_iterator=None;self._namespace_audit_age=0
        self._journal_cache=None;self._journal_authority=None;self._journal_bytes_read=0
        self._requested_role=None;self._startup_history_audit()

    def _verify_owner(self,proof_pin,operation,state,verify_current_owner):
        proof=super()._verify_owner(proof_pin,operation,state,verify_current_owner)
        if operation in ('complete_publication','publish_latest_after_fsynced_archive'):
            require(proof['publication_cadence_evidence']==self._expected_cadence
                    and digest(proof['publication_cadence_evidence'])==digest(self._expected_cadence),
                    'Current native owner must verify exact normal/tail/emergency packing evidence')
        # Optional incremental credit is available ONLY under this additional
        # source-reviewed current-owner witness. Without it, retain the frozen
        # full-prefix byte verification (no metadata boolean shortcut).
        self._journal_authority=None
        witness=proof.get('incremental_authoritative_journal')
        if witness is not None:
            epoch=witness['writer_epoch']
            require(witness['kind']=='ASR18_current_sole_append_journal_writer_identity_v1'
                    and witness['writer_source']==self.owner['journal_writer_source']
                    and witness['sole_current_writer'] is True
                    and witness['append_operation']=='open_ab_write_flush_fsync_only'
                    and witness['prior_bytes_never_rewritten_or_truncated'] is True
                    and epoch['kind']=='ASR18_verified_continuous_append_writer_epoch_v1'
                    and epoch['writer_source']==self.owner['journal_writer_source']
                    and type(epoch['producer_PID']) is int and epoch['producer_PID']>0
                    and type(epoch['producer_create_time']) is float and epoch['producer_create_time']>0
                    and type(epoch['session_owner_nonce']) is str and len(epoch['session_owner_nonce'])>=32,
                    'Exact sole append-only implementation/owner; idle or saved metadata alone cannot grant credit')
            bound(witness['writer_source']);bound(epoch['configuration']);safe(witness['path'])
            self._journal_authority=deepcopy(witness)
        else:self._journal_cache=None
        return proof

    @staticmethod
    def _creation_identity(stream):
        # Exact live handle, supported by the pinned Python3.11 runtime. Never
        # use st_ctime as an implicit creation-time substitute or reopen a path.
        import ctypes
        from ctypes import wintypes
        import msvcrt
        require(os.name=='nt','Actual Windows file identity required')
        class Information(ctypes.Structure):
            _fields_=[('attributes',wintypes.DWORD),('creation',wintypes.FILETIME),
                ('access',wintypes.FILETIME),('write',wintypes.FILETIME),('volume',wintypes.DWORD),
                ('size_high',wintypes.DWORD),('size_low',wintypes.DWORD),('links',wintypes.DWORD),
                ('index_high',wintypes.DWORD),('index_low',wintypes.DWORD)]
        api=ctypes.WinDLL('kernel32',use_last_error=True).GetFileInformationByHandle
        api.argtypes=[wintypes.HANDLE,ctypes.POINTER(Information)];api.restype=wintypes.BOOL
        value=Information();handle=msvcrt.get_osfhandle(stream.fileno())
        require(handle not in (0,-1) and api(handle,ctypes.byref(value))!=0,'Live file handle identity unavailable')
        index=(value.index_high<<32)|value.index_low
        created=((value.creation.dwHighDateTime<<32)|value.creation.dwLowDateTime)*100
        require(index>0 and created>0,'Exact positive file index and creationFILETIME')
        return {'device':value.volume,'file_index':index,'creation_ns':created}

    def _journal(self,prefix):
        path=safe(prefix['path']);offset=integer(prefix['offset']);witness=self._journal_authority
        if witness is None or safe(witness['path'])!=path:
            self._journal_cache=None
            result=super()._journal(prefix);self._journal_bytes_read+=offset;return result
        require(not path.is_relative_to(self.folder/'checkpoints'),'Authoritative journal is not a tensor payload')
        with path.open('rb') as stream:
            identity=self._creation_identity(stream)
            require(identity==witness['creation_identity'] and digest(identity)==digest(witness['creation_identity'])
                    and os.fstat(stream.fileno()).st_size>=offset,'Actual source/file creation identity, no truncate or unknown branch')
            targets={offset:prefix['sha256']};db=self._db()
            try:
                for raw, in db.execute('SELECT value FROM checkpoints WHERE retired=0 LIMIT 6'):
                    item=json.loads(raw);candidate=item['journal_prefix']
                    if safe(candidate['path'])==path:
                        target=integer(candidate['offset']);sha=candidate['sha256']
                        require(target not in targets or targets[target]==sha,'Conflicting live-prefix digest')
                        targets[target]=sha
            finally:db.close()
            require(len(targets)<=5,'At most latest/three winners/one in-flight prefix hash states')
            cache=self._journal_cache
            epoch_sha=digest(witness['writer_epoch'])
            if cache is None or cache['path']!=str(path) or cache['identity']!=identity or cache['writer_epoch_sha256']!=epoch_sha:
                cache={'path':str(path),'identity':identity,'writer_epoch_sha256':epoch_sha,'offset':0,'hasher':hashlib.sha256(),'prefixes':{}}
            else:cache={**cache,'hasher':cache['hasher'].copy(),'prefixes':dict(cache['prefixes'])}
            # If a requested historical prefix was not among the bounded live
            # snapshots retained in this resident process, perform ONE new
            # sequential startup pass collecting all current live boundaries.
            if any(target<cache['offset'] and target not in cache['prefixes'] for target in targets):
                cache={'path':str(path),'identity':identity,'writer_epoch_sha256':epoch_sha,'offset':0,'hasher':hashlib.sha256(),'prefixes':{}}
            for target in sorted(targets):
                if target<cache['offset']:
                    require(cache['prefixes'][target]==targets[target],'Exact previously verified immutable live prefix')
                    continue
                stream.seek(cache['offset']);remaining=target-cache['offset']
                while remaining:
                    self.guard();block=stream.read(min(1024**2,remaining));require(block,'Full appended range bytes required')
                    self._journal_bytes_read+=len(block)
                    cache['hasher'].update(block);remaining-=len(block);cache['offset']+=len(block)
                sha=cache['hasher'].hexdigest();require(sha==targets[target],'Actual append/full-startup bytes differ from authoritative prefix')
                cache['prefixes'][target]=sha
            cache['prefixes']={key:cache['prefixes'][key] for key in targets}
            require(self._creation_identity(stream)==identity and os.fstat(stream.fileno()).st_size>=max(targets),
                    'Actual file identity/size changed during prefix verification')
            self._journal_cache=cache

    def _db(self, *, setup=False):
        db=sqlite3.connect(self.folder/'catalog.sqlite')
        # Read connections do not repeatedly change journal mode while a bounded
        # streaming cursor is open. The inherited writer remains unchanged.
        if setup:db.execute('PRAGMA journal_mode=DELETE')
        db.execute('PRAGMA synchronous=FULL')
        db.execute('PRAGMA cache_size=-1024');db.execute('PRAGMA temp_store=MEMORY')
        # Size is checked at every operation and after the unchanged inherited
        # SQL mutation. No claim that a per-connection PRAGMA constrains the
        # inherited writer's separate connection or every filesystem allocation.
        return db

    def _history_frame(self,pin):
        offset=integer(pin['offset']);size=integer(pin['bytes'],1)
        require(size<=self.policy['maximum_history_record_bytes'],'Bounded history range before read')
        with (self.folder/'history.jsonl').open('rb') as stream:stream.seek(offset);raw=stream.read(size)
        require(len(raw)==size and hashlib.sha256(raw).hexdigest()==pin['sha256'],'Exact indexed append-only history range')
        value=json.loads(raw);require(value['kind']=='ASR18_indexed_owned_storage_history_record_v1','Typed history hash-chain record')
        return value

    def _startup_history_audit(self):
        state=json.loads((self.folder/'state.json').read_text());history=state['history']
        offset=0;root=hashlib.sha256(b'').hexdigest();revision=0
        with (self.folder/'history.jsonl').open('rb') as stream:
            while offset<history['offset']:
                self.guard();raw=stream.readline(self.policy['maximum_history_record_bytes']+1)
                require(raw and raw.endswith(b'\n') and len(raw)<=self.policy['maximum_history_record_bytes'],'Complete bounded history frame')
                value=json.loads(raw);revision+=1
                require(value['kind']=='ASR18_indexed_owned_storage_history_record_v1'
                        and value['revision']==revision and value['previous_sha256']==root
                        and value['previous_offset']==offset,'Exact complete catalog/history integrity chain')
                root=hashlib.sha256(raw).hexdigest();offset+=len(raw)
            require(stream.read(1)==b'','Unknown history tail preserved; explicit owner reconciliation required')
        require(offset==history['offset'] and root==history['sha256'] and revision==state['revision'],
                'Startup full history prefix differs')
        require(state['accounting']==(value['accounting_after'] if revision else ZERO_ACCOUNTING),
                'Final history frame must bind exact catalog accounting')

    def _entry(self,key,*,verify_files):
        db=self._db()
        try:
            row=db.execute('SELECT value,retired FROM checkpoints WHERE id=?',(key,)).fetchone()
            proof=db.execute('SELECT offset,bytes,sha256,stats FROM row_integrity WHERE id=?',(key,)).fetchone()
        finally:db.close()
        require(row is not None and proof is not None,'Exact registered catalog target')
        require(len(row[0].encode())<=self.policy['maximum_catalog_entry_bytes'],'Bounded individual catalog row')
        item=json.loads(row[0]);item['retired']=bool(row[1])
        pin={'offset':proof[0],'bytes':proof[1],'sha256':proof[2]};history=self._history_frame(pin)
        require(history['changed_checkpoints'][key]==item,'Catalog target differs from immutable indexed history bytes')
        if self._state and key in self._state['hot_integrity']:
            require(self._state['hot_integrity'][key]==pin,'Live latest/winner/catalog version drift')
        stats=json.loads(proof[3])
        self._verify_entry_files(item,stats,verify_files=verify_files)
        return item

    def _verify_entry_files(self,item,stats,*,verify_files):
        """Actual-file boundary; the private logical-index fixture substitutes only this IO."""
        for field in ('metadata','pointer','archive',*(['qualification'] if 'qualification' in item else [])):
            p=safe(item[field]['path'],self.folder);s=p.stat()
            require(s.st_size==item[field]['bytes'] and [s.st_size,s.st_mtime_ns]==stats[field], 'Immediate retained-file stat identity drift')
            if verify_files:bound(item[field])
        payload=safe(item['payload']['path'],self.folder)
        if item['retired']:require(not payload.exists(),'Retired payload reappearance/restore conflict')
        else:
            s=payload.stat();require([s.st_size,s.st_mtime_ns]==stats['payload'],'Live payload stat identity drift')
            if verify_files:bound(item['payload'])

    def _auxiliary(self,key,*,verify_archive=False):
        db=self._db()
        try:row=db.execute('SELECT value,offset,frame_bytes,sha256,status FROM auxiliary WHERE id=?',(key,)).fetchone()
        finally:db.close()
        require(row is not None and len(row[0].encode())<=self.policy['maximum_auxiliary_entry_bytes'],'Bounded registered auxiliary target')
        item=json.loads(row[0]);frame=self._history_frame({'offset':row[1],'bytes':row[2],'sha256':row[3]})
        require(item['id']==key and item['status']==row[4] and frame['changed_auxiliary'][key]==item,'Auxiliary catalog/status differs from immutable history range')
        self._verify_auxiliary_files(item,verify_archive=verify_archive)
        return item

    def _verify_auxiliary_files(self,item,*,verify_archive):
        root=safe(item['workspace'],self.folder/'transient')
        if item['status']=='retired':require(not root.exists(),'Retired auxiliary workspace reappeared')
        else:require(root.is_dir() and read(item['workspace_marker'])==item['workspace_ownership'],'Exact future-owned auxiliary workspace')
        if 'archive' in item:
            p=safe(item['archive']['path'],self.folder/'auxiliary');require(p.stat().st_size==item['archive']['bytes'],'Immediate retained auxiliary archive stat')
            if verify_archive:bound(item['archive']);self._verify_auxiliary_archive(item)

    def _workspace_inventory(self,item,*,manifest=None):
        """Bound actual simultaneously owned raw files AND directories, not just JSON sizes."""
        root=safe(item['workspace'],self.folder/'transient');files={};directories=['.'];total=0
        def walk(directory):
            nonlocal total
            with os.scandir(directory) as scan:
                for entry in scan:
                    self.guard();path=safe(entry.path,root)
                    if entry.is_dir(follow_symlinks=False):
                        directories.append(path.relative_to(root).as_posix())
                        require(len(directories)<=self.policy['maximum_auxiliary_directories'],'Predeclared transient directory allocation cap')
                        walk(path)
                    else:
                        require(entry.is_file(follow_symlinks=False),'Only exact regular transient files')
                        total+=path.stat().st_size;files[str(path)]=path
                        require(len(files)<=self.policy['maximum_auxiliary_files']+1
                                and total<=self.policy['maximum_auxiliary_raw_bytes'],'Bounded raw auxiliary files/bytes before archive')
        walk(root)
        require(str(Path(item['workspace_marker']['path']).resolve()) in files,'Owned workspace marker must remain live')
        if manifest is not None:
            require(manifest['kind']=='ASR18_future_owned_window_auxiliary_manifest_v1'
                    and manifest['workspace_marker']==item['workspace_marker']
                    and manifest['coordinator_plan']==item['plan']
                    and manifest['run_identity_sha256']==digest(self.owner['run_identity'])
                    and manifest['tensor_payloads_audio_features_and_models_included'] is False,
                    'Exact auxiliary-only manifest/source/plan; no model/audio/feature corpus archive')
            rows=manifest['files'];require(type(rows) is list and 1<=len(rows)<=self.policy['maximum_auxiliary_files'],'Bounded explicit metadata rows')
            expected={};relative=[]
            for row in rows:
                require(set(row)=={'role','file'} and row['role'] in AUX_ROLES,'Explicit protected metadata role')
                pin=row['file'];p=safe(pin['path'],root)
                require(p!=Path(item['workspace_marker']['path']).resolve() and p.suffix.lower() in ('.json','.jsonl','.log','.txt','.sha256'),
                        'Only new owned metadata, never raw tensor/audio or workspace marker as arbitrary member')
                require(str(p) not in expected,'No duplicate/aliased archive member')
                bound(pin,maximum=self.policy['maximum_auxiliary_raw_bytes']);expected[str(p)]=p;relative.append(p.relative_to(root).as_posix())
            require(relative==sorted(relative) and set(files)==set(expected)|{item['workspace_marker']['path']}
                    and manifest['directories']==sorted(directories),'Exact sorted full owned file/directory inventory; unknown files stop retirement')
            self._journal(manifest['journal_prefix'])
        return {'files':files,'directories':sorted(directories),'raw_bytes':total}

    def _verify_auxiliary_archive(self,item):
        """Lossless streaming validation. No old raw file is required for later historical verification."""
        header_limit=self.policy['maximum_auxiliary_header_bytes']
        with self._auxiliary_open(item['archive']['path']) as zipped:
            require(_read_exact(zipped,len(AUX_MAGIC))==AUX_MAGIC,'Typed future auxiliary stream')
            length=struct.unpack('<Q',_read_exact(zipped,8))[0]
            require(1<=length<=header_limit,'Complete bounded archive header before allocation')
            header=json.loads(_read_exact(zipped,length));manifest=header['manifest']
            require(header['kind']=='ASR18_lossless_future_auxiliary_archive_v1'
                    and header['ownership']==self.marker_pin and header['workspace_ownership']==item['workspace_ownership']
                    and header['auxiliary_codec']==AUXILIARY_CODEC
                    and digest(manifest)==item['manifest_sha256'],'Exact archive manifest/owner identity')
            require(type(manifest['files']) is list and 1<=len(manifest['files'])<=self.policy['maximum_auxiliary_files'],
                    'Bounded original member count before decompression')
            marker_bytes=canonical(item['workspace_ownership'])
            require(len(marker_bytes)==item['workspace_marker']['bytes']
                    and hashlib.sha256(marker_bytes).hexdigest()==item['workspace_marker']['sha256'],
                    'Original exclusive canonical workspace marker remains reconstructible byte exact')
            total_raw=0
            for row in manifest['files']:
                pin=row['file'];size=struct.unpack('<Q',_read_exact(zipped,8))[0]
                require(size==integer(pin['bytes'],1) and size<=self.policy['maximum_auxiliary_raw_bytes'],'Bounded exact member length')
                total_raw+=size;require(total_raw<=self.policy['maximum_auxiliary_raw_bytes'],'Cumulative raw member cap')
                remaining=size;sha=hashlib.sha256()
                while remaining:
                    self.guard();block=_read_exact(zipped,min(1024**2,remaining));sha.update(block);remaining-=len(block)
                require(sha.hexdigest()==pin['sha256'],'Lossless complete original metadata bytes')
            require(zipped.read(1)==b'','No additional/trailing archive member')
        return manifest

    def _auxiliary_open(self,path):
        codec_admission(64_000_000,self.guard)
        return _BoundedAuxiliaryXZReader(safe(path,self.folder/'auxiliary'),guard=self.guard,
            compressed_limit=self.policy['maximum_auxiliary_archive_bytes'],
            output_limit=self.policy['maximum_auxiliary_raw_bytes']+self.policy['maximum_auxiliary_header_bytes']
                +8*self.policy['maximum_auxiliary_files']+len(AUX_MAGIC)+8)

    def stream_archived_auxiliary_member(self,key,member_pin,on_chunk):
        """Resolve an original logical file identity without recreating its old path.

        The owner supplies its own reviewed metadata consumer. This component
        allocates at most a bounded header and one1MiB member block, and never
        writes a destination or gives old files new ownership.
        """
        require(callable(on_chunk),'Explicit bounded owner metadata consumer')
        with self.locked():
            self._load();item=self._auxiliary(key,verify_archive=True);manifest=self._verify_auxiliary_archive(item)
            require(any(row['file']==member_pin and digest(row['file'])==digest(member_pin) for row in manifest['files']),
                    'Exact original logical member identity in immutable archive')
            delivered=0;selected=0
            with self._auxiliary_open(item['archive']['path']) as zipped:
                require(_read_exact(zipped,len(AUX_MAGIC))==AUX_MAGIC,'Typed archive before resolution')
                size=struct.unpack('<Q',_read_exact(zipped,8))[0]
                require(size<=self.policy['maximum_auxiliary_header_bytes'],'Bounded original header')
                header=json.loads(_read_exact(zipped,size));require(digest(header['manifest'])==item['manifest_sha256'],'Exact manifest on resolution')
                for row in header['manifest']['files']:
                    pin=row['file'];remaining=struct.unpack('<Q',_read_exact(zipped,8))[0]
                    require(remaining==pin['bytes'] and remaining<=self.policy['maximum_auxiliary_raw_bytes'],'Exact member geometry')
                    selected+=int(pin==member_pin);sha=hashlib.sha256()
                    while remaining:
                        self.guard();block=_read_exact(zipped,min(1024**2,remaining));sha.update(block);remaining-=len(block)
                        if pin==member_pin:on_chunk(block);delivered+=len(block)
                    require(sha.hexdigest()==pin['sha256'],'Full exact archived original bytes')
                require(zipped.read(1)==b'' and selected==1 and delivered==member_pin['bytes'],'One complete uniquely bound logical member')
            return {'kind':'ASR18_original_auxiliary_logical_identity_resolved_from_lossless_archive_v1','passed':True,
                'archive':item['archive'],'original_logical_file':deepcopy(member_pin),'bytes_delivered':delivered,
                'original_path_recreated':False,'any_file_deleted_or_adopted':False}

    def reserve_auxiliary_workspace(self,intent,plan_pin):
        """Must run BEFORE future CPU producers. It never adopts a preexisting directory."""
        with self.locked():
            state,_,_=self._load();require(state['pending_publication']==intent,'Bind the exact future window-end/emergency reservation')
            bound(plan_pin);key=intent['id'];db=self._db()
            try:
                require(db.execute('SELECT 1 FROM auxiliary WHERE id=?',(key,)).fetchone() is None,'No workspace reuse/adoption')
                active=len(db.execute("SELECT id FROM auxiliary WHERE status IN ('preparing','archived') LIMIT 6").fetchall())
                require(active<self.policy['maximum_retained_auxiliary_workspaces'],'Retire exact obsolete auxiliary trees before another reservation')
            finally:db.close()
            root=safe(self.folder/'transient'/key,self.folder);require(not root.exists(),'Fresh future workspace only')
            root.mkdir();ownership={'kind':'ASR18_fresh_future_owned_auxiliary_workspace_v1','ownership':self.marker_pin,
                'publication_id':key,'plan':plan_pin,'workspace':str(root),'nonce':os.urandom(24).hex(),
                'created_before_any_producer':True,'preparation_is_optimizer_commit':False,'component_source':record(__file__)}
            marker=frozen.new_json(root/'ownership.json',ownership)
            item={'id':key,'status':'preparing','workspace':str(root),'workspace_marker':marker,
                  'workspace_ownership':ownership,'plan':plan_pin}
            db=self._db()
            try:db.execute('INSERT INTO auxiliary VALUES (?,?,?,?,?,?)',(key,canonical(item).decode(),0,0,'','preparing'));db.commit()
            finally:db.close()
            self._save(state,{'event':'future_auxiliary_workspace_reserved_before_producers','auxiliary_ids':[key]})
            return deepcopy(item)

    def archive_auxiliary(self,key,manifest,proof_pin,verify_current_owner):
        """Archive current small receipts without deleting active PRE/next-draw files."""
        with self.locked():
            state,_,usage=self._load(allow_preparing_auxiliary=True);item=self._auxiliary(key)
            require(item['status']=='preparing' and state['pending_publication']['id']==key,'Exact current predeclared workspace')
            proof=self._verify_owner(proof_pin,'archive_future_auxiliary',state,verify_current_owner)
            require(proof['auxiliary_manifest_sha256']==digest(manifest)
                    and proof['all_CPU_auxiliary_producers_actually_closed0'] is True
                    and proof['native_auxiliary_writer_contexts_returned'] is True
                    and proof['all_auxiliary_handles_released'] is True,'Actual source-bound producer closure and raw handle-release proof')
            self._workspace_inventory(item,manifest=manifest)
            header={'kind':'ASR18_lossless_future_auxiliary_archive_v1','ownership':self.marker_pin,
                'workspace_ownership':item['workspace_ownership'],'manifest':deepcopy(manifest),
                'auxiliary_codec':deepcopy(AUXILIARY_CODEC),
                'actual_quiescence_and_producer_closure':proof,'actual_quiescence_binding':proof_pin,
                'protected_source_history_DEV_winner':self.owner['protected_external']}
            encoded=canonical(header);require(len(encoded)<=self.policy['maximum_auxiliary_header_bytes'],'Complete auxiliary header fits declared bound before archive allocation')
            require(usage['auxiliary_bytes']+self.policy['maximum_auxiliary_archive_bytes']<=self.policy['maximum_total_auxiliary_archive_bytes']
                    and shutil.disk_usage(ROOT).free-self.policy['maximum_auxiliary_archive_bytes']>=RESERVE,'Prospective auxiliary horizon/E100GB reserve')
            target=safe(self.folder/'auxiliary'/(key+'.auxiliary.xz'),self.folder)
            codec_admission(256_000_000,self.guard)
            with target.open('xb') as raw:
                capped=_CappedWriter(raw,self.policy['maximum_auxiliary_archive_bytes'])
                with lzma.LZMAFile(capped,mode='wb',format=lzma.FORMAT_XZ,check=lzma.CHECK_CRC64,preset=6) as zipped:
                    zipped.write(AUX_MAGIC);zipped.write(struct.pack('<Q',len(encoded)));zipped.write(encoded)
                    for row in manifest['files']:
                        pin=row['file'];bound(pin);zipped.write(struct.pack('<Q',pin['bytes']));sha=hashlib.sha256();remaining=pin['bytes']
                        with safe(pin['path'],item['workspace']).open('rb') as source:
                            while remaining:
                                self.guard();block=source.read(min(1024**2,remaining));require(block,'Original auxiliary file truncated during archive')
                                sha.update(block);zipped.write(block);remaining-=len(block)
                            require(source.read(1)==b'' and sha.hexdigest()==pin['sha256'],'Actual original metadata changed during archival')
                raw.flush();os.fsync(raw.fileno())
            item.update(status='archived',archive=record(target),manifest_sha256=digest(manifest))
            self._verify_auxiliary_archive(item);self._verify_owner(proof_pin,'register_lossless_auxiliary_archive',state,verify_current_owner)
            db=self._db()
            try:db.execute('UPDATE auxiliary SET value=?,status=? WHERE id=?',(canonical(item).decode(),'archived',key));db.commit()
            finally:db.close()
            self._save(state,{'event':'lossless_future_auxiliary_archived_raw_files_retained','auxiliary_ids':[key]})
            return deepcopy(item)

    def retire_obsolete_auxiliary(self,keys,proof_pin,verify_current_owner):
        """Only explicit obsolete FUTURE metadata and then exact empty owned dirs."""
        require(type(keys) is list and 1<=len(keys)<=5 and len(set(keys))==len(keys),'Bounded explicit auxiliary retirement targets')
        with self.locked():
            state,_,_=self._load();require(state['pending_publication'] is None,'No retirement during checkpoint write/preparation')
            proof=self._verify_owner(proof_pin,'retire_future_auxiliary',state,verify_current_owner)
            protected=sorted(set(state['winners']+([state['latest']] if state['latest'] else [])))
            require(proof['retire_auxiliary_ids']==keys and proof['protected_active_auxiliary_ids']==protected
                    and proof['all_auxiliary_handles_released'] is True,'Exact active PRE/next-draw/latest/winner protection + current handle release')
            candidates=[]
            for key in keys:
                require(key not in protected,'Active PRE/next-draw/latest/qualified winner auxiliary files remain raw/live')
                item=self._auxiliary(key,verify_archive=True);require(item['status']=='archived','Actual lossless archive required before future metadata retirement')
                manifest=self._verify_auxiliary_archive(item);inventory=self._workspace_inventory(item,manifest=manifest)
                candidates.append((item,manifest,inventory))
            state['pending_retirement']={'auxiliary_ids':keys,'proof':proof_pin,'archives':[v[0]['archive'] for v in candidates]}
            frozen.replace_json(self.folder/'state.json',state)
            for item,manifest,inventory in candidates:
                for row in manifest['files']:
                    self.guard();self._verify_owner(proof_pin,'unlink_exact_archived_future_auxiliary',state,verify_current_owner)
                    pin=row['file'];bound(pin);safe(pin['path'],item['workspace']).unlink()
                bound(item['workspace_marker']);safe(item['workspace_marker']['path'],item['workspace']).unlink()
                root=Path(item['workspace'])
                for relative in sorted(inventory['directories'],key=lambda v:(v.count('/'),v),reverse=True):
                    self.guard();self._verify_owner(proof_pin,'remove_exact_empty_future_owned_directory',state,verify_current_owner)
                    path=root if relative=='.' else safe(root/relative,root)
                    path.rmdir();require(not path.exists(),'Exact empty-directory retirement was not observed')
                item['status']='retired';db=self._db()
                try:db.execute('UPDATE auxiliary SET value=?,status=? WHERE id=?',(canonical(item).decode(),'retired',item['id']));db.commit()
                finally:db.close()
            state['pending_retirement']=None
            self._save(state,{'event':'exact_archived_future_auxiliary_and_empty_directories_retired','auxiliary_ids':keys,'proof':proof_pin})
            return {'passed':True,'retired_future_auxiliary_ids':keys,'original_current_or_diagnostic_artifacts_changed':False,
                    'lossless_flat_archives_preserved':[v[0]['archive'] for v in candidates],'active_raw_auxiliary_ids_preserved':protected}

    def _namespace_stat_scan(self,state):
        fixed={'ownership.json','state.json','catalog.sqlite','history.jsonl','operation.lock'}
        pending=state['pending_publication'];pending_paths=set()
        if pending:pending_paths={str(safe(pending[k],self.folder)) for k in ('metadata_path','payload_path','pointer_path','temporary_path')}
        def walk(directory):
            with os.scandir(directory) as scan:
                for entry in scan:
                    path=safe(entry.path,self.folder)
                    if entry.is_dir(follow_symlinks=False):
                        if path.parent==self.folder:
                            require(path.name in ('checkpoints','archive','auxiliary','transient'),'Unknown owned directory')
                        else:
                            require(path.is_relative_to(self.folder/'transient'),'Only bounded registered transient subdirectories')
                            key=path.relative_to(self.folder/'transient').parts[0]
                            require(self._auxiliary(key)['status']!='retired','No unknown/retired empty transient directory')
                        yield from walk(path)
                    else:yield path
        actual_files=0;actual_bytes=0
        for path in walk(self.folder):
            self.guard();actual_files+=1;actual_bytes+=path.stat().st_size
            if not path.is_relative_to(self.folder/'transient'):self._check_namespace_entry(path,False,state)
            if path.parent==self.folder:require(path.name in fixed,'Unknown control file/tail; no cleanup authority')
            elif str(path) in pending_paths:continue
            elif path.is_relative_to(self.folder/'transient'):
                relative=path.relative_to(self.folder/'transient');require(len(relative.parts)>=2,'No unregistered transient root files')
                key=relative.parts[0];item=self._auxiliary(key)
                require(item['status']!='retired','No retired transient files')
                # _load already verifies the complete bounded workspace inventory.
                # Do not decompress its archive once for every member here.
            elif path.parent==self.folder/'auxiliary':
                require(re.fullmatch(r'p[0-9]{9}\.auxiliary\.xz',path.name),'Only registered future auxiliary archives')
                db=self._db()
                try:row=db.execute('SELECT value FROM auxiliary WHERE id=?',(path.name[:10],)).fetchone()
                finally:db.close()
                require(row is not None and json.loads(row[0])['archive']['path']==str(path),'Unknown auxiliary archive')
            else:
                match=re.fullmatch(r'(p[0-9]{9})(\.json|\.pointer\.json|\.tensors\.bin|\.json\.gz|\.qualified\.json\.gz)',path.name)
                require(match is not None and 1<=int(match[1][1:])<=state['publications'],'Unknown historical file or namespace escape')
        return actual_files,actual_bytes

    def _check_namespace_entry(self,path,is_directory,state):
        """Exact paths on observation; no sorting/materializing all historical names."""
        path=safe(path,self.folder)
        if is_directory:
            require(path.parent==self.folder and path.name in ('checkpoints','archive','auxiliary','transient')
                    or path.parent==self.folder/'transient','Only fixed flat archives and registered bounded transient roots')
            if path.parent==self.folder/'transient':self._auxiliary(path.name)
            return
        if path.parent==self.folder:
            require(path.name in ('ownership.json','state.json','catalog.sqlite','history.jsonl','operation.lock'),'Unknown control/tail')
            return
        pending=state['pending_publication']
        if pending and str(path) in {pending[k] for k in ('metadata_path','payload_path','pointer_path','temporary_path')}:
            return
        if path.parent==self.folder/'transient':raise ValueError('No unregistered files in transient root')
        if path.parent==self.folder/'auxiliary':
            require(re.fullmatch(r'p[0-9]{9}\.auxiliary\.xz',path.name),'Exact flat auxiliary archive filename')
            item=self._auxiliary(path.name[:10]);require(item.get('archive',{}).get('path')==str(path),'Unknown auxiliary archive')
            return
        match=re.fullmatch(r'(p[0-9]{9})(\.json|\.pointer\.json|\.tensors\.bin|\.json\.gz|\.qualified\.json\.gz)',path.name)
        require(match is not None and 1<=int(match[1][1:])<=state['publications'],'Unknown historical name')
        field={'.json':'metadata','.pointer.json':'pointer','.tensors.bin':'payload','.json.gz':'archive','.qualified.json.gz':'qualification'}[match[2]]
        item=self._entry(match[1],verify_files=False)
        require(field in item and item[field]['path']==str(path),'Known ID cannot authorize an extra/wrong-directory file')
        if field=='payload' and item['retired']:require(not path.exists(),'Retired payload reappearance')

    def _namespace_observation_iterator(self):
        # Historical directories are flat and never retired. Active transient
        # trees are small and exhaustively checked separately on every _load.
        for directory in (self.folder,*(self.folder/v for v in ('checkpoints','archive','auxiliary','transient'))):
            with os.scandir(directory) as scan:
                for entry in scan:yield Path(entry.path),entry.is_dir(follow_symlinks=False)

    def _bounded_namespace_audit(self,state):
        if self._namespace_iterator is None:self._namespace_iterator=iter(self._namespace_observation_iterator())
        self._namespace_audit_age+=1
        require(self._namespace_audit_age<=self.policy['maximum_namespace_audit_gap_operations'],
                'Bounded namespace audit cycle overdue; stop before any new writer, no silent deferred scan')
        checked=0;completed=False
        for _ in range(self.policy['namespace_entries_per_operation']):
            try:path,is_directory=next(self._namespace_iterator)
            except StopIteration:
                self._namespace_iterator=None;self._namespace_audit_age=0;completed=True;break
            self.guard();self._check_namespace_entry(path,is_directory,state);checked+=1
        return {'entries_observed':checked,'cycle_completed':completed,'operations_in_current_cycle':self._namespace_audit_age,
            'scope':'full startup scan, bounded resident observations; live/target identity immediate; cold unknown changes detected on observation'}

    def _rotating_hash_audit(self,state,count,aux_count):
        cursor=integer(state['historical_audit_cursor'])
        for _ in range(min(count,self.policy['historical_hash_rows_per_operation'])):
            cursor=(cursor%count)+1;self._entry(f'p{cursor:09d}',verify_files=True)
        state['historical_audit_cursor']=cursor
        cursor=state['auxiliary_audit_cursor'];require(type(cursor) is str,'Exact auxiliary B-tree seek cursor')
        for _ in range(min(aux_count,self.policy['historical_hash_rows_per_operation'])):
            db=self._db()
            try:
                row=db.execute('SELECT id FROM auxiliary WHERE id>? ORDER BY id LIMIT 1',(cursor,)).fetchone()
                if row is None:row=db.execute('SELECT id FROM auxiliary ORDER BY id LIMIT 1').fetchone()
            finally:db.close()
            require(row is not None,'History-bound auxiliary count differs from index');cursor=row[0]
            self._auxiliary(cursor,verify_archive=True)
        state['auxiliary_audit_cursor']=cursor

    def _hot_load(self,state,*,allow_preparing_auxiliary):
        """B-tree live/target operations; never stream the whole catalog per publication."""
        accounting=deepcopy(state['accounting']);db=self._db()
        try:
            root=db.execute('SELECT value FROM control WHERE key=?',('committed_root',)).fetchone()
            require(root is not None and json.loads(root[0])==state['catalog'],'Exact committed catalog root')
            live=[v[0] for v in db.execute('SELECT id FROM checkpoints WHERE retired=0 ORDER BY seq LIMIT 6')]
            active_aux=[v[0] for v in db.execute("SELECT id FROM auxiliary WHERE status IN ('preparing','archived') ORDER BY id LIMIT 6")]
        finally:db.close()
        require(len(live)<=5 and len(active_aux)<=self.policy['maximum_retained_auxiliary_workspaces']
                and set(live)==set(state['hot_integrity']) and len(live)==accounting['live_pairs']
                and len(active_aux)==accounting['active_auxiliary_workspaces'],'Bounded indexed live sets/history accounting')
        live_bytes=0
        for key in live:
            item=self._entry(key,verify_files=True);live_bytes+=item['payload']['bytes']
        require(live_bytes==accounting['live_payload_bytes'] and set(state['winners'])<=set(live)
                and state['latest']==(f'p{state["publications"]:09d}' if state['publications'] else None),'Current/latest/winner accounting')
        for key in active_aux:
            item=self._auxiliary(key,verify_archive=True)
            require(allow_preparing_auxiliary or item['status']!='preparing','Finish declared auxiliary preparation before publication')
            manifest=self._verify_auxiliary_archive(item) if item['status']=='archived' else None
            self._workspace_inventory(item,manifest=manifest)
        self._rotating_hash_audit(state,state['publications'],accounting['auxiliary_archives'])
        for pin in self.owner['protected_external']:bound(pin)
        namespace=self._bounded_namespace_audit(state)
        require(accounting['retained_metadata_bytes']<=self.policy['maximum_retained_metadata_bytes']
                and accounting['archive_bytes']<=self.policy['maximum_archive_bytes']
                and accounting['auxiliary_bytes']<=self.policy['maximum_total_auxiliary_archive_bytes']
                and accounting['auxiliary_archives']<=self.policy['maximum_publications']
                and (self.folder/'catalog.sqlite').stat().st_size<=self.policy['maximum_catalog_bytes']
                and (self.folder/'history.jsonl').stat().st_size+(self.folder/'catalog.sqlite').stat().st_size<=self.policy['maximum_control_bytes'],
                'History-bound full-horizon counters and actual control/index caps')
        return state,_Rows(self),accounting|{'namespace_audit':namespace,'namespace_files':None,
            'control_bytes':(self.folder/'history.jsonl').stat().st_size+(self.folder/'catalog.sqlite').stat().st_size}

    def _load(self, *, allow_preparing_auxiliary=False):
        self.guard();require(shutil.disk_usage(ROOT).free>=RESERVE and read(self.marker_pin)==self.owner,'Owner/E100GB guard')
        frozen.protection_plan(self.owner['protection_plan'],self.owner['protected_external'])
        state=json.loads((self.folder/'state.json').read_text());self._state=state
        require(state['owner']==self.marker_pin and state['pending_retirement'] is None,'Exact owner; incomplete retirement requires explicit reconciliation')
        require(integer(state['normal_publications'])+integer(state['emergency_publications'])==integer(state['publications'])
                and state['normal_publications']<=self.policy['normal_window_publications']
                and state['emergency_publications']<=self.policy['emergency_publications'],
                'Exact typed cadence counters; emergency allowance is finite, not implicit retry authority')
        require((self.folder/'history.jsonl').stat().st_size==state['history']['offset'],'Unknown catalog/history tail')
        require(state['catalog']['kind']=='ASR18_indexed_catalog_commit_root_v1'
                and state['catalog']['revision']==state['revision'] and state['catalog']['history']==state['history']
                and state['catalog']['accounting']==state['accounting'], 'Typed catalog/history/accounting root, not fake whole DB hash')
        require(set(state['accounting'])==set(ZERO_ACCOUNTING),'Exact bounded accounting fields')
        for amount in state['accounting'].values():integer(amount)
        if state['revision']:
            last=state['catalog']['last_frame'];frame=self._history_frame(last)
            require(last['offset']+last['bytes']==state['history']['offset'] and last['sha256']==state['history']['sha256']
                    and frame['accounting_after']==state['accounting'],'Latest exact history range binds accounting')
        else:require(state['accounting']==ZERO_ACCOUNTING and state['catalog']['last_frame'] is None,'Exact empty accounting')
        if self._catalog_startup_complete:return self._hot_load(state,allow_preparing_auxiliary=allow_preparing_auxiliary)
        db=self._db();metadata_bytes=archive_bytes=live_bytes=0;live=[];qualified=[];count=0
        try:
            tables={v[0] for v in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            require(tables=={'checkpoints','row_integrity','control','auxiliary'}
                    and db.execute('SELECT count(*) FROM control').fetchone()[0]==1,
                    'No unknown mutable catalog tables/control rows')
            meta=db.execute('SELECT value FROM control WHERE key=?',('committed_root',)).fetchone()
            require(meta is not None and json.loads(meta[0])==state['catalog'],'Mutable index committed root differs')
            for key,raw,retired in db.execute('SELECT id,value,retired FROM checkpoints ORDER BY seq'):
                self.guard();count+=1;require(len(raw.encode())<=self.policy['maximum_catalog_entry_bytes'],'Bounded streaming catalog row')
                item=json.loads(raw)
                require(item['id']==key==f'p{count:09d}' and item['seq']==count,'Contiguous exact publication IDs')
                self._entry(key,verify_files=False)
                metadata_bytes+=item['metadata']['bytes']+item['pointer']['bytes'];archive_bytes+=item['archive']['bytes']
                if 'qualification' in item:qualified.append(key);archive_bytes+=item['qualification']['bytes'];require(len(qualified)<=3,'At most3 protected phase winners')
                if not retired:live.append(key);live_bytes+=item['payload']['bytes'];require(len(live)<=5,'Bounded live set before materialization')
            require(count==state['publications'] and state['latest']==(f'p{count:09d}' if count else None)
                    and set(qualified)==set(state['winners']) and set(live)==set(state['hot_integrity']), 'No latest rollback/winner/live-set drift')
            require(db.execute('SELECT count(*) FROM row_integrity').fetchone()[0]==count,'No orphan mutable row-integrity entries')
            aux_count=aux_bytes=0;active_aux=[]
            for key,raw in db.execute('SELECT id,value FROM auxiliary ORDER BY id'):
                self.guard();aux_count+=1;require(len(raw.encode())<=self.policy['maximum_auxiliary_entry_bytes'],'Bounded auxiliary streaming row')
                item=self._auxiliary(key)
                require(item['status'] in ('preparing','archived','retired'),'Exact auxiliary state')
                require(allow_preparing_auxiliary or item['status']!='preparing','Complete current auxiliary preparation before publication/retirement')
                if 'archive' in item:aux_bytes+=item['archive']['bytes']
                if item['status']!='retired':
                    active_aux.append(key);require(len(active_aux)<=self.policy['maximum_retained_auxiliary_workspaces'],'Bounded active auxiliary trees')
                    manifest=self._verify_auxiliary_archive(item) if item['status']=='archived' else None
                    self._workspace_inventory(item,manifest=manifest)
        finally:db.close()
        for key in live:self._entry(key,verify_files=True)
        # Active archive contents were fully checked above; hashes are rechecked
        # on access, not for all retired files on every new publication.
        for key in active_aux:
            item=self._auxiliary(key)
            if 'archive' in item:bound(item['archive'])
        self._rotating_hash_audit(state,count,aux_count)
        for pin in self.owner['protected_external']:bound(pin)
        files,raw_bytes=self._namespace_stat_scan(state)
        require(metadata_bytes<=self.policy['maximum_retained_metadata_bytes'] and archive_bytes<=self.policy['maximum_archive_bytes']
                and aux_bytes<=self.policy['maximum_total_auxiliary_archive_bytes']
                and (self.folder/'catalog.sqlite').stat().st_size<=self.policy['maximum_catalog_bytes']
                and (self.folder/'history.jsonl').stat().st_size+(self.folder/'catalog.sqlite').stat().st_size<=self.policy['maximum_control_bytes'], 'Explicit metadata/archive/control/aux capacity')
        require(aux_count<=self.policy['maximum_publications'],'At most one auxiliary archive/workspace reservation per allowed publication')
        measured={'live_payload_bytes':live_bytes,'retained_metadata_bytes':metadata_bytes,'archive_bytes':archive_bytes,
            'live_pairs':len(live),'auxiliary_archives':aux_count,'auxiliary_bytes':aux_bytes,'active_auxiliary_workspaces':len(active_aux)}
        require(state['accounting']==measured,'One startup stream verifies complete catalog counters, not metadata trust')
        self._catalog_startup_complete=True
        return state,_Rows(self),{'live_payload_bytes':live_bytes,'retained_metadata_bytes':metadata_bytes,'archive_bytes':archive_bytes,
            'control_bytes':raw_bytes-metadata_bytes-archive_bytes-live_bytes-aux_bytes,'live_pairs':len(live),
            'namespace_files':files,'auxiliary_archives':aux_count,'auxiliary_bytes':aux_bytes}

    @staticmethod
    def _contribution(item,*,auxiliary=False):
        value=deepcopy(ZERO_ACCOUNTING)
        if item is None:return value
        if auxiliary:
            value['auxiliary_archives']=1
            value['auxiliary_bytes']=item.get('archive',{}).get('bytes',0)
            value['active_auxiliary_workspaces']=int(item['status']!='retired')
        else:
            value['retained_metadata_bytes']=item['metadata']['bytes']+item['pointer']['bytes']
            value['archive_bytes']=item['archive']['bytes']+item.get('qualification',{}).get('bytes',0)
            if not item['retired']:value['live_payload_bytes']=item['payload']['bytes'];value['live_pairs']=1
        return value

    def _save(self,state,event):
        ids=event.get('ids',[event['id']] if 'id' in event else [])
        require(len(ids)<=self.policy['maximum_history_rows_per_operation'],'Bounded target rows per operation')
        if event['event']=='publication_reserved_before_write':
            role=self._requested_role;require(role in ('completed_window_end','emergency'),'Explicit publication role before writer')
            counter='normal_publications' if role=='completed_window_end' else 'emergency_publications'
            limit='normal_window_publications' if role=='completed_window_end' else 'emergency_publications'
            require(state[counter]<self.policy[limit],'Declared normal/emergency allowance exhausted before writer')
            state['pending_publication']['publication_role']=role
        if event['event']=='verified_owned_checkpoint_published':
            require(self._requested_role in ('completed_window_end','emergency'),'Exact reserved role must survive completion')
            state['normal_publications' if self._requested_role=='completed_window_end' else 'emergency_publications']+=1
        aux_ids=event.get('auxiliary_ids',[])
        require(type(aux_ids) is list and len(aux_ids)<=5 and len(set(aux_ids))==len(aux_ids),'Bounded exact auxiliary targets')
        db=self._db();changed={};stats={};changed_aux={};accounting=deepcopy(state['accounting'])
        def account(old,new,*,auxiliary=False):
            before=self._contribution(old,auxiliary=auxiliary);after=self._contribution(new,auxiliary=auxiliary)
            for key in accounting:accounting[key]=integer(accounting[key]+after[key]-before[key])
        try:
            for key in ids:
                row=db.execute('SELECT value,retired FROM checkpoints WHERE id=?',(key,)).fetchone()
                if row is None:continue
                require(len(row[0].encode())<=self.policy['maximum_catalog_entry_bytes'],'Bounded changed row')
                item=json.loads(row[0]);item['retired']=bool(row[1]);changed[key]=item;stats[key]={}
                old=db.execute('SELECT offset,bytes,sha256 FROM row_integrity WHERE id=?',(key,)).fetchone()
                if old is None:
                    require(event['event']=='verified_owned_checkpoint_published' and item['seq']==state['publications'],
                            'Only actual new publication may create an unindexed row')
                    previous=None
                else:previous=self._history_frame({'offset':old[0],'bytes':old[1],'sha256':old[2]})['changed_checkpoints'][key]
                account(previous,item)
                for field in ('metadata','pointer','archive',*(['qualification'] if 'qualification' in item else []),*([] if item['retired'] else ['payload'])):
                    p=safe(item[field]['path'],self.folder);s=p.stat();require(s.st_size==item[field]['bytes'],'Actual file size before catalog commit')
                    stats[key][field]=[s.st_size,s.st_mtime_ns]
            for key in aux_ids:
                row=db.execute('SELECT value,offset,frame_bytes,sha256 FROM auxiliary WHERE id=?',(key,)).fetchone()
                require(row is not None and len(row[0].encode())<=self.policy['maximum_auxiliary_entry_bytes'],'Bounded changed auxiliary row')
                changed_aux[key]=json.loads(row[0])
                if row[2]==0:
                    require(event['event']=='future_auxiliary_workspace_reserved_before_producers','Only a new pre-producer reservation may create auxiliary row')
                    previous=None
                else:previous=self._history_frame({'offset':row[1],'bytes':row[2],'sha256':row[3]})['changed_auxiliary'][key]
                account(previous,changed_aux[key],auxiliary=True)
            value={'kind':'ASR18_indexed_owned_storage_history_record_v1','revision':state['revision']+1,
                'previous_offset':state['history']['offset'],'previous_sha256':state['history']['sha256'],
                'event':event,'changed_checkpoints':changed,'file_stats':stats,'changed_auxiliary':changed_aux,
                'accounting_after':accounting}
            raw=canonical(value)+b'\n';require(len(raw)<=self.policy['maximum_history_record_bytes'],'Bounded complete history frame before append')
            row_bytes=sum(len(canonical(v)) for v in changed.values())+sum(len(canonical(v)) for v in changed_aux.values())
            stats_bytes=sum(len(canonical(v)) for v in stats.values())
            require(all(len(canonical(v))<=self.policy['maximum_history_file_stats_bytes'] for v in stats.values())
                    and len(raw)-row_bytes-stats_bytes<=self.policy['maximum_history_fixed_overhead_bytes'],
                    'Complete framing/event/accounting overhead inside declared full-horizon cap')
            require((self.folder/'history.jsonl').stat().st_size+len(raw)+(self.folder/'catalog.sqlite').stat().st_size+4*1024**2
                    <=self.policy['maximum_control_bytes'],'Prospective control/index/history capacity')
            pin={'offset':state['history']['offset'],'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
            with (self.folder/'history.jsonl').open('ab') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
            for key in changed:db.execute('INSERT OR REPLACE INTO row_integrity VALUES (?,?,?,?,?)',
                (key,pin['offset'],pin['bytes'],pin['sha256'],canonical(stats[key]).decode()))
            for key in changed_aux:db.execute('UPDATE auxiliary SET offset=?,frame_bytes=?,sha256=? WHERE id=?',
                (pin['offset'],pin['bytes'],pin['sha256'],key))
            state['revision']+=1;state['history']={'kind':'ASR18_indexed_chained_history_prefix_v1','offset':pin['offset']+pin['bytes'],'sha256':pin['sha256']}
            state['accounting']=accounting
            state['catalog']={'kind':'ASR18_indexed_catalog_commit_root_v1','revision':state['revision'],'history':state['history'],
                'publications':state['publications'],'accounting':deepcopy(accounting),'last_frame':pin}
            state['hot_integrity']={}
            for key, in db.execute('SELECT id FROM checkpoints WHERE retired=0 LIMIT 6'):
                require(len(state['hot_integrity'])<5,'Only bounded live rows in hot state')
                r=db.execute('SELECT offset,bytes,sha256 FROM row_integrity WHERE id=?',(key,)).fetchone()
                require(r is not None,'Every live row must have immutable history identity')
                state['hot_integrity'][key]={'offset':r[0],'bytes':r[1],'sha256':r[2]}
            db.execute('INSERT OR REPLACE INTO control VALUES (?,?)',('committed_root',canonical(state['catalog']).decode()));db.commit()
        finally:db.close()
        frozen.replace_json(self.folder/'state.json',state);self._state=state

    def reserve_publication(self,*,publication_role):
        require(publication_role in ('completed_window_end','emergency'),'Explicit bounded role, no implicit diagnostic captures')
        self._requested_role=publication_role
        try:return super().reserve_publication()
        finally:self._requested_role=None

    def complete_publication(self,intent,pointer_pin,expectation,proof_pin,verify_current_owner):
        require(expectation['publication_role']==intent['publication_role'],'Exact role/cadence binding')
        cadence=expectation['publication_cadence_evidence']
        require(cadence['kind']=='ASR18_source_bound_actual_completed_window_cadence_v1'
                and cadence['phase'] in (WEAK,CLEANER)
                and 1<=integer(cadence['committed_window_updates'],1)<=20
                and type(cadence['exact_pass_tail']) is bool,'Explicit actual phase/window/tail source evidence')
        bound(cadence['coordinator_plan'])
        if intent['publication_role']=='completed_window_end':
            require(cadence['committed_window_updates']>=(4 if cadence['phase']==WEAK else 20) or cadence['exact_pass_tail'],
                    'Normal horizon assumes actual greedy4weak/20cleaner or exact pass tail; earlier stop must use emergency quota')
        else:
            require(cadence['earlier_stop_reason'] in ('resource','time','budget','native_failure'),
                    'Bounded emergency publication is explicit, never unlimited retry authority')
        self._expected_cadence=deepcopy(cadence)
        self._requested_role=intent['publication_role']
        try:return super().complete_publication(intent,pointer_pin,expectation,proof_pin,verify_current_owner)
        finally:self._requested_role=None;self._expected_cadence=None

    def retire_obsolete_payloads(self,checkpoint_ids,proof_pin,verify_current_owner):
        require(type(checkpoint_ids) is list and 1<=len(checkpoint_ids)<=5,'Bounded explicit obsolete live targets')
        return super().retire_obsolete_payloads(checkpoint_ids,proof_pin,verify_current_owner)

    def close(self):
        iterator=self._namespace_iterator;self._namespace_iterator=None
        if iterator is not None:iterator.close()
        self._journal_cache=None;self._journal_authority=None


class IndexedOwnedFutureCheckpointStore(_IndexedProtocol):
    """Future-only public gate. No accepted review/policy or native verifier ships here."""
    def __init__(self,marker_pin,*,guard):
        super().__init__(marker_pin,checkpoint_verifier=frozen.verify_streamed_pointer,guard=guard)
        review=read(self.owner['ROOT_review'])
        require(review['kind']=='ROOT_indexed_future_window_end_checkpoint_storage_review_v1'
                and review['accepted'] is True and review['reviewer']=='/root'
                and review['policy']==self.owner['policy'] and review['component_source']==record(__file__)
                and review['frozen_operation_source']==record(frozen.__file__)
                and review['future_namespace']==str(self.folder)
                and review['run_identity_sha256']==digest(self.owner['run_identity'])
                and review['owner_verifier_source']==self.owner['verifier_source']
                and review['protected_external']==self.owner['protected_external']
                and review['protection_plan']==self.owner['protection_plan']
                and review['capacity_evidence']==self.owner['capacity_evidence']
                and review['auxiliary_codec']==AUXILIARY_CODEC
                and review['auxiliary_compress_stage_increment_bytes']==256_000_000
                and review['auxiliary_read_index_stage_increment_bytes']==64_000_000
                and review['journal_writer_source']==self.owner['journal_writer_source'],'Exact new future namespace/cadence/source/protection/capacity/journal review')
        verify_capacity_evidence(self.owner['capacity_evidence'],self.policy,review)

    @classmethod
    def create(cls,folder,policy_pin,*,ROOT_review,run_identity,verifier_source,protected,guard):
        guard();folder=safe(folder);policy=policy_values(read(policy_pin));review=read(ROOT_review)
        require(not folder.exists() and folder.is_relative_to(ROOT/'runs'),'Fresh reviewed E/runs namespace only')
        require(policy['accepted'] is True and policy['full_horizon_accounting_reviewed'] is True
                and review['kind']=='ROOT_indexed_future_window_end_checkpoint_storage_review_v1'
                and review['accepted'] is True and review['reviewer']=='/root'
                and review['policy']==policy_pin and review['component_source']==record(__file__)
                and review['frozen_operation_source']==record(frozen.__file__)
                and review['future_namespace']==str(folder) and review['run_identity_sha256']==digest(run_identity)
                and review['auxiliary_codec']==AUXILIARY_CODEC
                and review['auxiliary_compress_stage_increment_bytes']==256_000_000
                and review['auxiliary_read_index_stage_increment_bytes']==64_000_000
                and review['owner_verifier_source']==verifier_source,'Exact separately reviewed indexed policy/namespace/source')
        flattened=frozen.protection_plan(review['protection_plan'],protected)
        require(review['protected_external']==flattened,'Exact required source/history/DEV/winner protection plan')
        capacity=verify_capacity_evidence(review['capacity_evidence'],policy,review)
        import ctypes
        require(os.name=='nt','Reviewed storage target is Windows E only')
        sectors=ctypes.c_ulong();sector_bytes=ctypes.c_ulong();unused=ctypes.c_ulong();total=ctypes.c_ulong()
        ok=ctypes.windll.kernel32.GetDiskFreeSpaceW(str(ROOT.anchor),ctypes.byref(sectors),ctypes.byref(sector_bytes),ctypes.byref(unused),ctypes.byref(total))
        require(ok!=0 and sectors.value*sector_bytes.value==policy['E_allocation_unit_bytes'],'Actual E cluster changed; no stale allocation assumption')
        require(shutil.disk_usage(ROOT).free-policy['maximum_total_owned_allocation_bytes']>=RESERVE,'Whole allocation-rounded horizon plus E100GB reserve')
        bound(verifier_source);bound(review['journal_writer_source'])
        folder.mkdir();(folder/'checkpoints').mkdir();(folder/'archive').mkdir();(folder/'auxiliary').mkdir();(folder/'transient').mkdir()
        owner={'kind':OWNER,'owner':frozen.OWNER,'namespace':str(folder),'nonce':os.urandom(24).hex(),
            'policy':policy_pin,'run_identity':deepcopy(run_identity),'component_source':record(__file__),
            'frozen_operation_source':record(frozen.__file__),'verifier_source':verifier_source,'protected_external':flattened,
            'protected_external_sha256':digest(flattened),'ROOT_review':ROOT_review,'protection_plan':review['protection_plan'],
            'capacity_evidence':deepcopy(review['capacity_evidence']),
            'journal_writer_source':deepcopy(review['journal_writer_source']),
            'existing_source_checkpoints_adopted_as_owned':False}
        marker=frozen.new_json(folder/'ownership.json',owner)
        with (folder/'history.jsonl').open('xb') as stream:stream.flush();os.fsync(stream.fileno())
        db=sqlite3.connect(folder/'catalog.sqlite')
        try:
            db.execute('CREATE TABLE checkpoints (id TEXT PRIMARY KEY, seq INTEGER UNIQUE, value TEXT NOT NULL, retired INTEGER NOT NULL)')
            db.execute('CREATE TABLE row_integrity (id TEXT PRIMARY KEY, offset INTEGER NOT NULL, bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, stats TEXT NOT NULL)')
            db.execute('CREATE TABLE control (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            db.execute('CREATE TABLE auxiliary (id TEXT PRIMARY KEY, value TEXT NOT NULL, offset INTEGER NOT NULL, frame_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, status TEXT NOT NULL)')
            db.execute('CREATE INDEX checkpoint_live ON checkpoints(retired,seq)')
            db.execute('CREATE INDEX auxiliary_live ON auxiliary(status,id)')
            history={'kind':'ASR18_indexed_chained_history_prefix_v1','offset':0,'sha256':hashlib.sha256(b'').hexdigest()}
            catalog={'kind':'ASR18_indexed_catalog_commit_root_v1','revision':0,'history':history,'publications':0,
                'accounting':deepcopy(ZERO_ACCOUNTING),'last_frame':None}
            db.execute('INSERT INTO control VALUES (?,?)',('committed_root',canonical(catalog).decode()));db.commit()
        finally:db.close()
        frozen.new_json(folder/'state.json',{'owner':marker,'revision':0,'publications':0,'normal_publications':0,'emergency_publications':0,
            'latest':None,'winners':[],'history':history,'catalog':catalog,'hot_integrity':{},'pending_publication':None,'pending_retirement':None,
            'historical_audit_cursor':0,'auxiliary_audit_cursor':'','accounting':deepcopy(ZERO_ACCOUNTING)})
        return cls(marker,guard=guard)
