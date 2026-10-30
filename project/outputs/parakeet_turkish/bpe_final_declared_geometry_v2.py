"""Inactive final declared-envelope validation, separate from native fit.

An actual source-bound DATA producer exit is mandatory. This validator checks
its exact source/metadata chain and independently streams the final retained
scalar index. It does not convert declarations into decoded-audio validation.
"""
import hashlib,json
from pathlib import Path
from bpe_broad_production_state_v1 import require
from bpe_declared_geometry_index_v1 import audit_index,signature

PROJECT=Path(__file__).resolve().parents[2]
ROOT=Path(r'E:\gpu_training_scratchpad');AREA=ROOT/'analysis/BPE_weak_then_clean_v1'
CONTRACT_SHA='75c28129882d076ad6abc7789b3fed61d3a7f26ccefc1254d7fcf451e6823a7c'
COMPANION_SHA='7996b5da7e76dde5f483c1a57afcea323a94d688c03de62979eb199000897409'
PIN_FIELDS={'path','bytes','sha256','mtime_ns'}

def base(value):return {k:value[k] for k in ('path','bytes','sha256')}

def check4(pin,*,guard=lambda:None,metadata=False,hash_file=True):
    require(type(pin) is dict and set(pin)==PIN_FIELDS and type(pin['bytes']) is int and pin['bytes']>0 and type(pin['mtime_ns']) is int and pin['mtime_ns']>0
        and type(pin['sha256']) is str and len(pin['sha256'])==64 and set(pin['sha256'])<=set('0123456789abcdef'),'Exact signed four-field binding required; no bool integers')
    path=Path(pin['path']).resolve();require(path.is_relative_to(ROOT.resolve()) or path.is_relative_to(PROJECT),'Only E or existing project source access permitted')
    expected={k:pin[k] for k in ('path','bytes','mtime_ns')};require(signature(path)==expected,'Final declared geometry binding signature changed')
    if hash_file:
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            while chunk:=stream.read(1024*1024):guard();digest.update(chunk)
        require(digest.hexdigest()==pin['sha256'] and signature(path)==expected,'Final declared geometry binding hash/signature changed')
    guard()
    if metadata:
        require(pin['bytes']<=32_000_000,'Bounded final declared geometry JSON required')
        return json.loads(path.read_text(encoding='utf-8-sig'))
    return path

def actual_exit(value,*,marker,positive_native=True):
    require(value['kind']=='ASR18_source_bound_actual_tool_exit_v1' and value['owner']=='/root/cpu_data_preparation'
        and type(value['actual_exit_code']) is int and value['actual_exit_code']==0 and type(value['actual_tool_chunk_id']) is str and value['actual_tool_chunk_id'],
        'Genuine source-bound observed producer/tool exit0 required')
    require(value['published_marker']==marker,'Actual producer exit binds another published envelope')
    if positive_native:require(type(value['native_PID']) is int and value['native_PID']>0 and type(value.get('owned_session')) is int and value['owned_session']>0,
        'Actual native PID and owned session required; publication is not actual completion')

def validate(envelope,contract,companion,actual,*,bindings,inventory,carrier,guard=lambda:None):
    """Called only after the separate final inventory/carrier validators pass."""
    require(bindings['declared_geometry_contract']['sha256']==CONTRACT_SHA and bindings['declared_geometry_companion']['sha256']==COMPANION_SHA,'Pinned declared contract/interface companion required')
    require(contract['kind']=='ASR18_declared_training_geometry_envelope_receipt_contract_v2' and companion['kind']=='ASR18_declared_geometry_envelope_concrete_interface_companion_v1',
        'Expected declared contract versions')
    require(base(companion['receipt_contract'])==bindings['declared_geometry_contract'],'Concrete companion binds another declared contract')
    require(type(envelope) is dict and set(envelope)==set(contract['final_receipt_exact_keys']) and envelope['kind']==contract['final_receipt_kind']
        and type(envelope['schema_version']) is int and envelope['schema_version']==1 and envelope['complete'] is True
        and envelope['owner']=='/root/cpu_data_preparation' and type(envelope['PID']) is int and envelope['PID']>0 and envelope['TRAIN_admitted'] is False,
        'Exact complete declared envelope required; it cannot itself admit TRAIN')
    marker={**bindings['declared_geometry_envelope'],'mtime_ns':Path(bindings['declared_geometry_envelope']['path']).stat().st_mtime_ns}
    actual_exit(actual,marker=marker)
    require(actual['native_PID']==envelope['PID'],'Published envelope and actual owned native PID differ')
    require(envelope['geometry_basis']==companion['geometry_basis_exact_value'] and envelope['runtime_contract']==contract['runtime_contract']
        and envelope['scope']==contract['scope_constants'] and envelope['Alania_interval_declaration']==contract['Alania_interval_declaration'],
        'Exact declared-only basis and mandatory runtime geometry/heldout enforcement required')
    pins=envelope['bindings'];require(type(pins) is dict and set(pins)==set(contract['required_binding_paths']),'All16 declared source/closure/contract bindings required')
    for key,path in contract['required_binding_paths'].items():
        require(Path(pins[key]['path']).resolve()==Path(path).resolve(),'Wrong final declared binding path: '+key)
        check4(pins[key],guard=guard,hash_file=key not in ('final_catalog','final_labels'))
    require(pins['envelope_receipt_contract']==companion['receipt_contract'] and pins['producer_source']==companion['producer_source'],
        'Exact finalized declared producer/contract pins required')
    require(actual['source']==pins['producer_source'],'Observed actual producer source must equal the exact finalized source pin')
    require(type(envelope['geometry_index']) is dict and envelope['geometry_index']==pins['declared_geometry_index']
        and envelope['geometry_index']==bindings['declared_geometry_index'],'Exact same four-field final geometry index required')
    outer={'preparation_plan':'preparation_plan','finalization_contract':'finalization_contract','completed_source_inventory':'completed_source_inventory',
        'original_FLAC_carrier_coverage':'carrier_coverage','original_FLAC_carrier_coverage_actual_closed':'carrier_coverage_actual_closed',
        'final_catalog':'final_catalog','catalog_actual_closed':'catalog_actual_closed','final_labels':'final_labels',
        'labels_actual_closed':'labels_actual_closed','heldout_metadata':'heldout_metadata'}
    for key,parent in outer.items():require(base(pins[key])==bindings[parent],'Declared envelope differs from finalized population: '+key)
    # Final catalog/labels have already had their whole hashes checked by the
    # population validator; their four-field signatures above bind that same
    # read-only state without an unnecessary duplicate whole-file hash here.
    sidecars=check4(pins['KIRAAT_declared_sidecar_inventory'],guard=guard,metadata=True)
    require(type(sidecars) is dict and set(sidecars)==set(contract['KIRAAT_sidecar_inventory_exact_keys']) and sidecars['kind']=='ASR18_complete_KIRAAT_declared_geometry_sidecar_inventory_v1' and sidecars['complete'] is True
        and type(sidecars['schema_version']) is int and sidecars['schema_version']==1 and sidecars['owner']=='/root/cpu_data_preparation' and sidecars['TRAIN_admitted'] is False
        and type(sidecars['sources_count']) is int and sidecars['sources_count']==len(sidecars['sources'])==729,
        'Exact complete729 declared sidecar inventory required')
    for key in ('preparation_plan','completed_source_inventory','original_FLAC_carrier_coverage'):
        require(sidecars[key]==pins[key],'Sidecar inventory binds another finalized source population')
    by_sha={row['source_sha256']:row for row in carrier['sources']};require(len(by_sha)==729,'Independent final carrier proof must cover729 unique sources')
    seen=set();physical=original_bytes=sidecar_bytes=0
    for row in sidecars['sources']:
        guard();require(type(row) is dict and set(row)==set(contract['KIRAAT_source_entry_exact_keys']),'Exact declared source sidecar keys required')
        sha=row['source_SHA256'];require(sha in by_sha and sha not in seen,'Unknown/duplicate/missing declared source');seen.add(sha);old=by_sha[sha]
        for key in ('source_signature','source_acquisition_receipt','carrier_receipt','payload','row_index'):
            require(row[key]==old[key],'Declared source sidecar carrier identity differs: '+key)
        for key in ('physical_rows','original_FLAC_bytes','unresolved_physical_rows'):
            require(type(row[key]) is int and row[key]>=0,'Exact physical counts required; no bool integers')
        require(row['physical_rows']==old['physical_rows'] and row['original_FLAC_bytes']==old['original_FLAC_bytes'] and row['physical_rows']>0,
            'Physical sidecar and independently audited original carrier counts differ')
        require(row['collection_mode'] in contract['collection_modes'] and row['original_normalized_independent_allbyte_actual_exit']==old['independent_audit_actual_closed']['actual_exit_proof'],
            'Preserved original independently observed all-byte audit proof required')
        for key in ('geometry_sidecar','geometry_parser','geometry_writer','declared_geometry_actual_child_exit',
                    'declared_geometry_native_guard_footer','declared_geometry_actual_parent_tool_closure','original_normalized_independent_allbyte_actual_exit'):
            check4(row[key],guard=guard,metadata=key in ('declared_geometry_actual_child_exit','declared_geometry_native_guard_footer','declared_geometry_actual_parent_tool_closure','original_normalized_independent_allbyte_actual_exit'))
        child=check4(row['declared_geometry_actual_child_exit'],guard=guard,metadata=True);footer=check4(row['declared_geometry_native_guard_footer'],guard=guard,metadata=True)
        parent=check4(row['declared_geometry_actual_parent_tool_closure'],guard=guard,metadata=True)
        require(child['kind']=='ASR18_actual_owned_FLAC_child_exit_v1' and type(child['actual_exit_code']) is int and child['actual_exit_code']==0
            and child['parent_observed_wait'] is True and child['source_sha256']==sha and type(child['native_PID']) is int and child['native_PID']>0
            and child['mode']==('audit' if row['collection_mode']=='during_original_independent_allbyte_audit' else 'declared_geometry_completion'),
            'Actual closed source-specific declared geometry child required')
        require(base(child['native_closure'])==base(row['declared_geometry_native_guard_footer']),'Observed child and independently bound native guard footer differ')
        require(type(footer['PID']) is int and footer['PID']==child['native_PID'] and type(footer['native_return_code']) is int and footer['native_return_code']==0
            and footer['watchdog_clean'] is True and type(footer['samples']) is int and footer['samples']>=3
            and type(footer['peak_private_bytes']) is int and 0<footer['peak_private_bytes']<2_000_000_000
            and footer['CPU_private_cap_bytes']==2_000_000_000 and footer['CUDA_masked'] is True,'Clean actual declared geometry watchdog closure required')
        require(parent['owner']=='/root/cpu_data_preparation' and type(parent['actual_exit_code']) is int and parent['actual_exit_code']==0
            and (parent.get('actual_tool_chunk_id') or parent.get('actual_tool_chunk')) and parent['no_owned_carrier_child_running'] is True,
            'Genuine actual parent tool closure required; worker status alone is insufficient')
        result_path=Path(row['declared_geometry_actual_child_exit']['path']).parent/'result.json'
        result_signature=signature(result_path);require(result_signature['bytes']<=32_000_000,'Bounded observed child result metadata required')
        result_pin={**result_signature,'sha256':hashlib.sha256(result_path.read_bytes()).hexdigest()}
        result=check4(result_pin,guard=guard,metadata=True);declared=result['declared_STREAMINFO_geometry']
        require(result['PID']==child['native_PID'] and result['mode']==child['mode'] and declared['source_SHA256']==sha
            and declared['sidecar']==row['geometry_sidecar'] and declared['parser']==row['geometry_parser'] and declared['writer']==row['geometry_writer']
            and declared['physical_rows']==row['physical_rows'] and declared['unresolved_declared_geometry_rows']==row['unresolved_physical_rows'],
            'Observed declared geometry result must bind the exact sidecar/source/row/parser/writer metadata')
        physical+=row['physical_rows'];original_bytes+=row['original_FLAC_bytes'];sidecar_bytes+=row['geometry_sidecar']['bytes']
    require(seen==set(by_sha) and (physical,original_bytes,sidecar_bytes)==(sidecars['all_physical_rows'],sidecars['all_original_FLAC_bytes'],sidecars['all_sidecar_bytes']),
        'All physical sidecar totals must reconcile independently of retained TRAIN count')
    require(envelope['KIRAAT_source_sidecars']=={'inventory':pins['KIRAAT_declared_sidecar_inventory'],'sources_count':729,'all_physical_rows':physical,
        'all_original_FLAC_bytes':original_bytes,'all_sidecar_bytes':sidecar_bytes},'Envelope physical inventory summary differs')
    unknown=envelope['unknown_invalid_counts'];require(type(unknown) is dict and set(unknown)==set(contract['unknown_invalid_count_keys_exact'])
        and all(type(v) is int and v>=0 for v in unknown.values()) and all(value==0 for key,value in unknown.items() if key.startswith('retained_')),
        'Every retained unknown/invalid/missing declaration blocks final publication; never exclusion')
    require(type(envelope['population']) is dict and set(envelope['population'])==set(contract['population_exact_keys'])
        and all(type(v) is int and v>=0 for v in envelope['population'].values()),'Exact scalar retained envelope integers required; no bool counts')
    require(type(envelope['groups']) is dict and set(envelope['groups'])==set(contract['group_keys_exact'])
        and all(type(g) is dict and set(g)==set(contract['group_scalar_keys_exact']) and all(type(v) is int and v>=0 for v in g.values()) for g in envelope['groups'].values()),
        'Exact per-source scalar group integer fields required')
    rows=envelope['population']['retained_rows'];tokens=envelope['population']['total_target_tokens']
    audited=audit_index(pins['final_catalog']['path'],pins['final_labels']['path'],pins['declared_geometry_index']['path'],rows=rows,tokens=tokens,guard=guard)
    require(envelope['population']==audited['population'] and envelope['groups']==audited['groups'],'Exact independently audited retained envelope/counts/targets required')
    return {'passed':True,'kind':'ASR18_final_declared_geometry_validation_v1','scalar_audit':audited,'runtime_contract':envelope['runtime_contract'],
        'source_sidecar_files':729,'actual_full_corpus_decode_validation_claimed':False,'native_allocation_or_encoder_fit_approved':False,'TRAIN_admitted':False}
