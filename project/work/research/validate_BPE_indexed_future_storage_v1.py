"""Small PRIVATE synthetic indexed-storage controls + closed-metadata sizing.

No public constructor, accepted ROOT policy, model, tensor payload, audio, native
lease/proof or real checkpoint retirement is exercised. The optional reference
is read-only already-closed JSON/log/text, never its binary payloads. Every file
removed by the private protocol is NEW explicitly synthetic fixture data.
"""
import argparse
from copy import deepcopy
import ctypes
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct
import sys
import time

P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish'
sys.path.insert(0,str(S));sys.path.insert(0,str(Path(__file__).parent))
import bpe_indexed_owned_checkpoint_storage_v1 as module
from fastmss_row_serial_cpu_v6 import environment,resource_guard,admit
from bpe_broad_audio_v1 import record
from bpe_broad_production_state_v1 import canonical,digest,require
from atomic_io_retry_v1 import write_json


def guard():
    value=resource_guard();require(value['worker_private_memory']['private_bytes']<100_000_000,'Private fixture100MB')
    return value


def synthetic_owner_callback(request):
    proof=module.read(request['proof'])
    require(proof['PRIVATE_SYNTHETIC_PROTOCOL_ONLY'] is True and proof['actual_native_proof'] is False,
            'This callback must never validate actual native/lease/mapping evidence')
    guard();return True


def synthetic_checkpoint_verifier(pointer_pin,expectation):
    pointer=module.read(pointer_pin);metadata=module.read(pointer['checkpoint'])
    require(pointer['kind']=='SYNTHETIC_BYTE_PAIR_NOT_A_NATIVE_CHECKPOINT'
            and metadata['kind']=='SYNTHETIC_SCALAR_METADATA_NO_TENSORS'
            and pointer['identity']==expectation['identity'],'Explicit private byte fixture, not streamed native semantics')
    module.bound(pointer['payload'])
    return {'pointer':pointer,'metadata':metadata}


def initialize_private(folder):
    folder.mkdir();[(folder/name).mkdir() for name in ('checkpoints','archive','auxiliary','transient')]
    protected=module.frozen.new_json(folder.parent/'synthetic_protected_metadata.json',{'kind':'SYNTHETIC_NO_ACTUAL_QUALIFICATION_OR_SOURCE'})
    plan=module.frozen.new_json(folder.parent/'synthetic_protection_plan.json',{
        'kind':'ASR18_future_owned_checkpoint_protection_plan_v1','adopt_external_sources_as_owned':False,
        'required_roles':{role:[protected] for role in ('source','history','DEV','qualified_winner')}})
    policy={'kind':module.POLICY,'accepted':False,'production_enabled':False,'retire_existing_or_original_artifacts':False,
        'retire_owned_metadata_or_pointer_files':False,'publication_cadence':'completed_window_end_plus_explicit_bounded_emergency_v1',
        'normal_window_publications':3,'emergency_publications':1,'maximum_publications':4,'planned_native_updates':4,
        'maximum_payload_bytes':1024**2,'maximum_metadata_bytes':65536,'maximum_pointer_bytes':16384,'maximum_archive_record_bytes':65536,
        'maximum_archive_bytes':7*65536,'maximum_retained_metadata_bytes':4*(65536+16384),
        'maximum_control_bytes':8*1024**2,'maximum_catalog_bytes':4*1024**2,'maximum_history_record_bytes':65536,
        'maximum_catalog_entry_bytes':4096,'maximum_history_rows_per_operation':5,
        'maximum_history_fixed_overhead_bytes':2048,'maximum_history_file_stats_bytes':1024,
        'steady_live_pairs':4,'maximum_inflight_publications':1,'hard_live_pair_peak':5,
        'maximum_auxiliary_archive_bytes':65536,'maximum_auxiliary_raw_bytes':1024**2,'maximum_auxiliary_entry_bytes':4096,
        'maximum_auxiliary_header_bytes':65536,'maximum_auxiliary_files':16,'maximum_auxiliary_directories':16,
        'maximum_retained_auxiliary_workspaces':5,'maximum_total_auxiliary_archive_bytes':4*65536,
        'maximum_total_owned_allocation_bytes':32*1024**2,'historical_hash_rows_per_operation':1,
        'namespace_entries_per_operation':32,'maximum_namespace_audit_gap_operations':128,'E_allocation_unit_bytes':524288}
    module.policy_values(policy)
    policy_pin=module.frozen.new_json(folder.parent/'PRIVATE_SYNTHETIC_POLICY_UNACCEPTED.json',policy)
    owner={'kind':'PRIVATE_SYNTHETIC_INDEXED_ENGINE_NOT_PUBLIC_OWNER','owner':module.frozen.OWNER,
        'namespace':str(folder),'component_source':record(module.__file__),'frozen_operation_source':record(module.frozen.__file__),
        'policy':policy_pin,'run_identity':{'synthetic_only':True},'verifier_source':record(__file__),
        'protected_external':[protected],'protected_external_sha256':digest([protected]),'protection_plan':plan,
        'ROOT_review':None,'journal_writer_source':record(__file__)}
    marker=module.frozen.new_json(folder/'ownership.json',owner)
    (folder/'history.jsonl').touch();db=sqlite3.connect(folder/'catalog.sqlite')
    try:
        db.execute('CREATE TABLE checkpoints (id TEXT PRIMARY KEY, seq INTEGER UNIQUE, value TEXT NOT NULL, retired INTEGER NOT NULL)')
        db.execute('CREATE TABLE row_integrity (id TEXT PRIMARY KEY, offset INTEGER NOT NULL, bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, stats TEXT NOT NULL)')
        db.execute('CREATE TABLE control (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        db.execute('CREATE TABLE auxiliary (id TEXT PRIMARY KEY,value TEXT NOT NULL,offset INTEGER NOT NULL,frame_bytes INTEGER NOT NULL,sha256 TEXT NOT NULL,status TEXT NOT NULL)')
        db.execute('CREATE INDEX checkpoint_live ON checkpoints(retired,seq)');db.execute('CREATE INDEX auxiliary_live ON auxiliary(status,id)')
        history={'kind':'ASR18_indexed_chained_history_prefix_v1','offset':0,'sha256':hashlib.sha256(b'').hexdigest()}
        catalog={'kind':'ASR18_indexed_catalog_commit_root_v1','revision':0,'history':history,'publications':0,
            'accounting':deepcopy(module.ZERO_ACCOUNTING),'last_frame':None}
        db.execute('INSERT INTO control VALUES (?,?)',('committed_root',canonical(catalog).decode()));db.commit()
    finally:db.close()
    module.frozen.new_json(folder/'state.json',{'owner':marker,'revision':0,'publications':0,'normal_publications':0,'emergency_publications':0,
        'latest':None,'winners':[],'history':history,'catalog':catalog,'hot_integrity':{},'pending_publication':None,'pending_retirement':None,
        'historical_audit_cursor':0,'auxiliary_audit_cursor':'','accounting':deepcopy(module.ZERO_ACCOUNTING)})
    return reopen_private(marker,owner,policy)


def reopen_private(marker,owner,policy):
    # Explicit private internal engine. It does NOT call/bypass a public
    # constructor under fake accepted policy and does not produce ROOT review.
    engine=object.__new__(module._IndexedProtocol);engine.guard=guard;engine.verifier=synthetic_checkpoint_verifier
    engine.synthetic=True;engine.marker_pin=marker;engine.owner=owner;engine.folder=Path(owner['namespace']);engine.policy=policy
    engine._state=None;engine._requested_role=None;engine._catalog_startup_complete=False
    engine._namespace_iterator=None;engine._namespace_audit_age=0;engine._journal_cache=None;engine._journal_authority=None;engine._journal_bytes_read=0
    engine._startup_history_audit();return engine


def owner_proof(engine,folder,name,**extension):
    state,entries,_=engine._load(allow_preparing_auxiliary=True)
    proof={'kind':'ASR18_owned_checkpoint_mapping_release_quiescence_v1','owner':module.frozen.OWNER,
        'ownership':engine.marker_pin,'state_revision':state['revision'],'catalog':state['catalog'],
        'coherent_latest_pointer':entries[state['latest']]['pointer'] if state['latest'] else None,'protected_owned_winners':[],
        'nonce':'synthetic_only_not_a_native_nonce_0000','native_work_quiescent':True,'checkpoint_mappings_released':True,
        'candidate_payload_handles_released':True,'checkpoint_write_inflight':False,'restore_in_progress':False,
        'PRIVATE_SYNTHETIC_PROTOCOL_ONLY':True,'actual_native_proof':False,**extension}
    return module.frozen.new_json(folder/(name+'.json'),proof)


def synthetic_cadence(output):
    path=output/'PRIVATE_synthetic_cadence_plan.json'
    if not path.exists():module.frozen.new_json(path,{'kind':'SYNTHETIC_WINDOW_NO_NATIVE_DRAWS'})
    return {'kind':'ASR18_source_bound_actual_completed_window_cadence_v1','phase':module.WEAK,
        'committed_window_updates':1,'exact_pass_tail':True,'coordinator_plan':record(path),
        'PRIVATE_SYNTHETIC_PROTOCOL_ONLY':True,'actual_native_proof':False}


def publish(engine,output,number,journal):
    intent=engine.reserve_publication(publication_role='completed_window_end')
    payload=Path(intent['payload_path']);payload.write_bytes(bytes([number])*1024)
    metadata=module.frozen.new_json(intent['metadata_path'],{'kind':'SYNTHETIC_SCALAR_METADATA_NO_TENSORS','number':number,'payload':record(payload)})
    identity={'synthetic_only':True,'number':number};sampler={'synthetic_draw':number}
    pointer={'kind':'SYNTHETIC_BYTE_PAIR_NOT_A_NATIVE_CHECKPOINT','identity':identity,'checkpoint':metadata,'payload':record(payload)}
    pin=module.frozen.new_json(intent['pointer_path'],pointer)
    expectation={'identity':identity,'optimizer_updates':number,'sampler_state':sampler,'journal_prefix':journal,
        'run_identity':engine.owner['run_identity'],'publication_role':'completed_window_end','publication_cadence_evidence':synthetic_cadence(output)}
    proof=owner_proof(engine,output,'publication_'+str(number)+'_PRIVATE_proof',publication_cadence_evidence=expectation['publication_cadence_evidence'])
    return engine.complete_publication(intent,pin,expectation,proof,synthetic_owner_callback)


def size_closed_metadata(reference,output):
    require(reference.name=='fullpopulation_firstwindow20_conv_actual_01' and reference.is_relative_to(module.ROOT),'Exact closed reference directory')
    files=[];directories=0;raw_bytes=0
    for path in reference.rglob('*'):
        guard()
        if path.is_dir():directories+=1;continue
        if path.suffix.lower() not in ('.json','.jsonl','.log','.txt'):continue
        require(not path.is_symlink() and path.stat().st_size<=4*1024**2,'Only bounded closed metadata, no binary payload/audio/model')
        raw_bytes+=path.stat().st_size;files.append(path)
        require(len(files)<=512 and raw_bytes<=32*1024**2 and directories<=128,'Actual future auxiliary census bound')
    require(directories<=128,'Bound final closed reference directory census')
    files.sort(key=str);members=[record(path) for path in files]
    header={'kind':'READ_ONLY_CLOSED_METADATA_SIZING_NOT_OWNERSHIP_OR_NATIVE_PROOF','members':members}
    target=output/'READ_ONLY_closed_metadata_sizing.gz'
    with target.open('xb') as raw:
        writer=module._CappedWriter(raw,2*1024**2)
        with gzip.GzipFile(fileobj=writer,mode='wb',mtime=0) as zipped:
            encoded=canonical(header);require(len(encoded)<=512*1024,'Bounded complete header')
            zipped.write(struct.pack('<Q',len(encoded)));zipped.write(encoded)
            for pin in members:
                zipped.write(struct.pack('<Q',pin['bytes']));sha=hashlib.sha256();count=0
                with Path(pin['path']).open('rb') as source:
                    while block:=source.read(1024**2):guard();sha.update(block);count+=len(block);zipped.write(block)
                require(count==pin['bytes'] and sha.hexdigest()==pin['sha256'],'Closed reference metadata must remain byte exact')
        raw.flush();os.fsync(raw.fileno())
    with gzip.open(target,'rb') as zipped:
        n=struct.unpack('<Q',module._read_exact(zipped,8))[0];require(json.loads(module._read_exact(zipped,n))==header,'Exact sizing header')
        for pin in members:
            n=struct.unpack('<Q',module._read_exact(zipped,8))[0];require(n==pin['bytes'],'Exact metadata member bytes')
            sha=hashlib.sha256()
            while n:guard();block=module._read_exact(zipped,min(1024**2,n));sha.update(block);n-=len(block)
            require(sha.hexdigest()==pin['sha256'],'Lossless redecoded original metadata')
        require(zipped.read(1)==b'','Complete sized archive')
    return {'reference':str(reference),'files':len(files),'directories':directories,'raw_bytes':raw_bytes,
        'gzip':record(target),'complete_header_bytes':len(canonical(header)),'all_original_files_unchanged':True,
        'scope':'One actual CLOSED20 metadata directory only, NOT arbitrary fullpopulation window or future archive proof'}


def main(output,reference):
    environment();admission=admit(100_000_000,guard=guard);began=time.monotonic();source=record(module.__file__)
    output=Path(output).resolve();require(output.is_relative_to(module.ROOT) and not output.exists(),'Fresh E-only fixture')
    output.mkdir();checks=[];samples=[guard()];engine=None
    def check(name,condition):require(condition,name);checks.append(name);samples.append(guard())
    def reject(name,operation):
        try:operation()
        except (ValueError,KeyError,TypeError,OSError,sqlite3.Error):checks.append(name);samples.append(guard());return
        raise ValueError('Expected fail-closed rejection: '+name)
    try:
        engine=initialize_private(output/'private_engine');full_scans=[];original_scan=engine._namespace_stat_scan
        def measured_scan(state):full_scans.append(1);return original_scan(state)
        engine._namespace_stat_scan=measured_scan
        state,_,usage=engine._load();check('Explicitly synthetic empty catalog starts with zero accounting',state['accounting']==module.ZERO_ACCOUNTING)
        journal_path=output/'PRIVATE_synthetic_journal.jsonl';journal_path.write_bytes(b'{"synthetic":1}\n')
        journal={'path':str(journal_path),'offset':journal_path.stat().st_size,'sha256':record(journal_path)['sha256']}
        first=publish(engine,output,1,journal);check('Actual new synthetic bytes indexed and archived',first['id']=='p000000001')
        intent=engine.reserve_publication(publication_role='completed_window_end')
        plan=module.frozen.new_json(output/'PRIVATE_synthetic_plan.json',{'kind':'SYNTHETIC_NOT_FINAL_DATA'})
        auxiliary=engine.reserve_auxiliary_workspace(intent,plan);root=Path(auxiliary['workspace']);(root/'nested').mkdir()
        a=module.frozen.new_json(root/'a.json',{'original_raw':1});b=module.frozen.new_json(root/'nested'/'b.json',{'full_targets':[1,2,3]})
        manifest={'kind':'ASR18_future_owned_window_auxiliary_manifest_v1','workspace_marker':auxiliary['workspace_marker'],
            'coordinator_plan':plan,'run_identity_sha256':digest(engine.owner['run_identity']),
            'tensor_payloads_audio_features_and_models_included':False,
            'files':[{'role':'pre_save','file':a},{'role':'next_draw','file':b}],'directories':['.','nested'],'journal_prefix':journal}
        bad=deepcopy(manifest);bad['files'].append(deepcopy(bad['files'][0]))
        reject('Duplicate auxiliary raw member cannot be archived',lambda:engine._workspace_inventory(auxiliary,manifest=bad))
        bad=deepcopy(manifest);bad['directories']=['.']
        reject('Missing owned directory cannot be silently omitted',lambda:engine._workspace_inventory(auxiliary,manifest=bad))
        proof=owner_proof(engine,output,'PRIVATE_aux_archive_proof',auxiliary_manifest_sha256=digest(manifest),
            all_CPU_auxiliary_producers_actually_closed0=True,native_auxiliary_writer_contexts_returned=True,all_auxiliary_handles_released=True)
        archived=engine.archive_auxiliary(intent['id'],manifest,proof,synthetic_owner_callback)
        check('Lossless archive retains both original live files',Path(a['path']).exists() and Path(b['path']).exists())
        payload=Path(intent['payload_path']);payload.write_bytes(b'2'*1024)
        metadata=module.frozen.new_json(intent['metadata_path'],{'kind':'SYNTHETIC_SCALAR_METADATA_NO_TENSORS','payload':record(payload)})
        identity={'synthetic_only':True,'number':2};pointer=module.frozen.new_json(intent['pointer_path'],{'kind':'SYNTHETIC_BYTE_PAIR_NOT_A_NATIVE_CHECKPOINT',
            'identity':identity,'checkpoint':metadata,'payload':record(payload)})
        expectation={'identity':identity,'optimizer_updates':2,'sampler_state':{'synthetic_draw':2},'journal_prefix':journal,
            'run_identity':engine.owner['run_identity'],'publication_role':'completed_window_end','publication_cadence_evidence':synthetic_cadence(output)}
        proof=owner_proof(engine,output,'PRIVATE_publication2_proof',publication_cadence_evidence=expectation['publication_cadence_evidence'])
        bad=deepcopy(expectation);bad['publication_cadence_evidence']['exact_pass_tail']=False
        reject('Short normal weak window must consume emergency quota instead',lambda:engine.complete_publication(intent,pointer,bad,proof,synthetic_owner_callback))
        bad=deepcopy(expectation);bad['publication_cadence_evidence']['committed_window_updates']=True
        reject('Boolean is not a committed update count',lambda:engine.complete_publication(intent,pointer,bad,proof,synthetic_owner_callback))
        second=engine.complete_publication(intent,pointer,expectation,proof,synthetic_owner_callback)
        check('Second publication advances exact history-bound accounting',engine._load()[0]['accounting']['live_pairs']==2)
        proof=owner_proof(engine,output,'PRIVATE_protected_aux_negative',retire_auxiliary_ids=[second['id']],protected_active_auxiliary_ids=[second['id']],all_auxiliary_handles_released=True)
        reject('Latest raw PRE/next-draw workspace cannot retire',lambda:engine.retire_obsolete_auxiliary([second['id']],proof,synthetic_owner_callback))
        third=publish(engine,output,3,journal)
        proof=owner_proof(engine,output,'PRIVATE_aux_retirement_proof',retire_auxiliary_ids=[second['id']],protected_active_auxiliary_ids=[third['id']],all_auxiliary_handles_released=True)
        retired=engine.retire_obsolete_auxiliary([second['id']],proof,synthetic_owner_callback)
        check('Only archived new synthetic files and exact empty directories removed',retired['passed'] and not root.exists() and Path(archived['archive']['path']).exists())
        recovered=[];resolution=engine.stream_archived_auxiliary_member(second['id'],b,recovered.append)
        check('Original logical hash/bytes resolve after raw directory retirement',resolution['passed'] and hashlib.sha256(b''.join(recovered)).hexdigest()==b['sha256'])
        proof=owner_proof(engine,output,'PRIVATE_payload_retirement_proof',retire_checkpoint_ids=[first['id']])
        engine.retire_obsolete_payloads([first['id']],proof,synthetic_owner_callback)
        check('Only exact obsolete NEW synthetic payload removed; metadata retained',not Path(first['payload']['path']).exists() and Path(first['metadata']['path']).exists())
        check('Resident hot operations do not repeat startup full catalog/namespace scan',len(full_scans)==1)
        before=engine._load()[0]['accounting'];marker,owner,policy=engine.marker_pin,engine.owner,engine.policy
        engine.close();engine=reopen_private(marker,owner,policy)
        check('Fresh open fully verifies history/index/accounting and retired paths',engine._load()[0]['accounting']==before)
        check('Frozen source not mutated',record(module.__file__)==source)
        bad=deepcopy(policy);bad['maximum_control_bytes']=1024
        reject('Full-horizon changed-row/framing bytes cannot fit a falsely small control cap',lambda:module.policy_values(bad))
        with journal_path.open('rb') as stream:creation=engine._creation_identity(stream)
        import psutil
        epoch={'kind':'ASR18_verified_continuous_append_writer_epoch_v1','writer_source':record(__file__),
            'producer_PID':os.getpid(),'producer_create_time':psutil.Process().create_time(),
            'session_owner_nonce':'PRIVATE_SYNTHETIC_APPEND_EPOCH_0000000000','configuration':engine.owner['policy']}
        authority={'kind':'ASR18_current_sole_append_journal_writer_identity_v1','writer_source':record(__file__),
            'path':str(journal_path),'creation_identity':creation,'sole_current_writer':True,
            'append_operation':'open_ab_write_flush_fsync_only','prior_bytes_never_rewritten_or_truncated':True,
            'writer_epoch':epoch,'PRIVATE_SYNTHETIC_PROTOCOL_ONLY':True,'actual_native_proof':False}
        engine._journal_authority=authority;reads=engine._journal_bytes_read;engine._journal(journal)
        check('First private journal verification reads actual complete prefix',engine._journal_bytes_read-reads==journal['offset'])
        extension=b'{"synthetic":2}\n'
        with journal_path.open('ab') as stream:stream.write(extension);stream.flush();os.fsync(stream.fileno())
        updated={'path':str(journal_path),'offset':journal_path.stat().st_size,'sha256':record(journal_path)['sha256']}
        reads=engine._journal_bytes_read;engine._journal(updated)
        check('Same source/file/epoch verifies ONLY actual appended bytes',engine._journal_bytes_read-reads==len(extension))
        reads=engine._journal_bytes_read;engine._journal(journal)
        check('Retained old live prefix uses prior byte-verified digest without reread',engine._journal_bytes_read==reads)
        bad=deepcopy(authority);bad['creation_identity']['file_index']+=1;engine._journal_authority=bad
        reject('Actual journal creation identity mismatch stops',lambda:engine._journal(updated))
        engine._journal_authority=deepcopy(authority);engine._journal_authority['writer_epoch']['session_owner_nonce']+='new_epoch'
        reads=engine._journal_bytes_read;engine._journal(updated)
        check('Changed owner epoch forces fresh complete byte verification',engine._journal_bytes_read-reads==updated['offset'])
        sizing=size_closed_metadata(Path(reference).resolve(),output) if reference else None
        if sizing:check('Closed actual metadata losslessly fits declared2MiB preview cap',sizing['gzip']['bytes']<=2*1024**2)
        state=engine._load()[0];history=engine.folder/'history.jsonl';original_bytes=history.stat().st_size
        with history.open('ab') as stream:stream.write(b'UNACKNOWLEDGED_SYNTHETIC_TAIL');stream.flush();os.fsync(stream.fileno())
        reject('Unknown append tail is preserved and rejected',engine._load)
        check('Rejection does not truncate history or roll back authority',history.stat().st_size>original_bytes and json.loads((engine.folder/'state.json').read_text())['revision']==state['revision'])
        samples.append(guard());sectors=ctypes.c_ulong();width=ctypes.c_ulong();free=ctypes.c_ulong();total=ctypes.c_ulong()
        require(ctypes.windll.kernel32.GetDiskFreeSpaceW('E:\\',ctypes.byref(sectors),ctypes.byref(width),ctypes.byref(free),ctypes.byref(total)),
                'Actual E allocation geometry')
        cluster=sectors.value*width.value;require(cluster==524288,'Reviewed E allocation unit changed')
        rounded=sum(((p.stat().st_size+cluster-1)//cluster)*cluster for p in output.rglob('*') if p.is_file())
        require(rounded<128*1024**2,'Bounded NEW synthetic/sizing regular-file allocation128MiB')
        result={'kind':'ASR18_PRIVATE_synthetic_indexed_storage_controls_and_closed_metadata_sizing_v1','passed':True,'source':source,
            'producer':{'pid':os.getpid(),'ppid':os.getppid()},'checks':checks,'check_count':len(checks),'CPU_admission':admission,
            'observed_private_peak_bytes':max(v['worker_private_memory']['private_bytes'] for v in samples),
            'wall_seconds':time.monotonic()-began,'E_allocation_unit_bytes':cluster,'new_regular_file_allocation_rounded_bytes':rounded,
            'closed_reference_sizing':sizing,'new_synthetic_payload_unlinks_only':True,'old_or_current_artifact_changed':False,
            'private_synthetic_owner_IO_only':True,'public_constructor_or_policy_proved':False,'actual_native_or_checkpoint_semantics_proved':False,
            'full_horizon_performance_or_capacity_proved':False,'emergency_recovery_proved':False,'production_enabled':False,
            'scope_limits':['Only THREE tiny synthetic pairs, not the full95742-publication horizon.','Hot scan-count behavior is scoped to these synthetic operations.',
                'Actual closed20 sizing is one readonly metadata example; no arbitrary future auxiliary fit is inferred.','Crash tail rejected/preserved, not recovered.']}
        write_json(output/'report.json',result);print(json.dumps({'passed':True,'checks':len(checks),'report':record(output/'report.json')}),flush=True)
    finally:
        if engine is not None:engine.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--reference')
    args=parser.parse_args();main(args.output,args.reference)
