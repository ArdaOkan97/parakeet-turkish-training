"""Fail-closed final compressed carrier coverage/provenance; no audio decoder."""
import json
from pathlib import Path
from bpe_broad_audio_v1 import scoped,record,require

PROJECT=Path(__file__).resolve().parents[2]

def base(binding):return {key:binding[key] for key in ('path','bytes','sha256')}

def check(binding,*,guard,metadata=False,verify_hash=False):
    expected={'path','bytes','mtime_ns'} if 'sha256' not in binding and not(metadata or verify_hash) else {'path','bytes','sha256','mtime_ns'}
    require(set(binding)==expected,'Exact signed carrier binding required')
    path=Path(binding['path']).resolve()
    require(path.is_relative_to(scoped(r'E:\gpu_training_scratchpad')) or path.is_relative_to(PROJECT),'Carrier input outside permitted E/project storage')
    stat=path.stat();require((stat.st_size,stat.st_mtime_ns)==(binding['bytes'],binding['mtime_ns']),'Final carrier source/receipt/index signature changed')
    guard()
    if metadata or verify_hash:
        require(record(path)==base(binding),'Final carrier metadata/source hash changed')
    if metadata:
        require(stat.st_size<=8_000_000,'Bounded carrier JSON metadata required')
        return json.loads(path.read_text(encoding='utf-8-sig'))
    return path

def validate(coverage,coverage_contract,bindings,inventory,stage_contract,*,guard):
    from asr18_closed_contract_schema_v1 import ClosedSchema
    require(stage_contract['kind']=='ASR18_normalized_FLAC_stage_provenance_contract_v1','Pinned normalized actual-stage schema required')
    proof_schema=ClosedSchema(stage_contract['stage_proof_schema']);footer_schema=ClosedSchema(stage_contract['native_guard_schema'])
    check(stage_contract['normalizer_source'],guard=guard,verify_hash=True)
    require(coverage_contract['kind']=='ASR18_compressed_carrier_coverage_contract_v1','Pinned carrier schema companion required')
    ClosedSchema(coverage_contract['final_receipt_schema']).validate(coverage)
    for name in ('preparation_plan','finalization_contract','completed_source_inventory'):
        require(base(coverage[name])==bindings[name],'Final carrier binds another fixed plan/contract/inventory')
        check(coverage[name],guard=guard,metadata=True)
    require(base(coverage['coverage_contract'])==bindings['carrier_coverage_contract'],'Carrier schema binding differs')
    check(coverage['coverage_contract'],guard=guard,metadata=True)
    require(coverage['source_files']==len(coverage['sources'])==729,'Exact729 train/rest physical sources required')
    expected={str(Path(v['destination']).resolve()):v for v in inventory['sources'] if str(v['destination']).lower().endswith('.parquet')}
    # Only train/rest KIRAAT media belong in this carrier. Reference manifests
    # are independently bound and must not silently become training coverage.
    expected={path:v for path,v in expected.items() if Path(path).name.startswith(('train-','rest-'))}
    require(len(expected)==729,'Completed source inventory does not contain exact729 training Parquets')
    seen=set();physical_rows=payload_bytes=cache_bytes=0
    for item in coverage['sources']:
        guard();source=check(item['source_signature'],guard=guard);key=str(source)
        require(key in expected and key not in seen and item['source_sha256']==expected[key]['source_SHA256'],'Missing/duplicate/different pinned carrier source')
        seen.add(key);check(item['source_acquisition_receipt'],guard=guard,metadata=True)
        for name in ('carrier_receipt','audit_receipt'):check(item[name],guard=guard,metadata=True)
        check(item['row_index'],guard=guard,verify_hash=True)
        check(item['payload'],guard=guard) # Full bytes already independently audited; signature is revalidated, not claimed rehashed.
        first=item['preparation_or_completion_actual_closed'];second=item['independent_audit_actual_closed']
        require(first['mode'] in ('prepare','completion_verify') and second['mode']=='audit' and first['native_PID']!=second['native_PID'],
            'Separate actual preparation/completion verifier and independent audit required')
        for stage in (first,second):
            proof=check(stage['actual_exit_proof'],guard=guard,metadata=True)
            footer=check(stage['native_guard_closure'],guard=guard,metadata=True)
            proof_schema.validate(proof);footer_schema.validate(footer)
            require(proof['normalizer_source']==stage_contract['normalizer_source'],'Exact reviewed normalization source required')
            for optional in ('parent_actual_tool_closure','PID_correction_provenance'):
                if proof[optional] is not None:check(proof[optional],guard=guard,metadata=True)
            check(stage['result_receipt'],guard=guard,metadata=True);check(stage['worker_source'],guard=guard,verify_hash=True)
            require(proof['kind']=='ASR18_actual_owned_FLAC_stage_completion_v1' and proof['schema_version']==1 and
                proof['owner']==stage['owner'] and proof['source_sha256']==item['source_sha256'] and proof['mode']==stage['mode'] and
                proof['actual_exit_code']==0 and proof['native_PID']==stage['native_PID'] and proof['parent_observed_wait'] is True,
                'Normalized stage must preserve actual parent wait/native source identity')
            require(footer['kind']=='ASR18_normalized_FLAC_native_guard_completion_v1' and footer['native_PID']==stage['native_PID'] and
                footer['actual_exit_code']==0 and footer['watchdog_clean'] is True and footer['monitor_joined'] is True and
                footer['samples']>=3 and footer['peak_private_bytes']<2_000_000_000 and footer['CPU_private_cap_bytes']==2_000_000_000 and footer['CUDA_masked'] is True,
                'Actual normalized native watchdog completion required; no inferred0')
            require(base(proof['native_guard_closure'])==base(stage['native_guard_closure']) and base(proof['result_receipt'])==base(stage['result_receipt']) and
                base(proof['worker_source'])==base(stage['worker_source']),'Actual stage proof/coverage provenance bindings differ')
            for name in ('original_parent_wait_proof','original_guard_footer'):check(proof[name],guard=guard,metadata=True)
            check(proof['normalizer_source'],guard=guard,verify_hash=True)
            require(base(footer['original_guard_footer'])==base(proof['original_guard_footer']),'Normalized footer/proof original provenance differs')
        physical_rows+=item['physical_rows'];payload_bytes+=item['original_FLAC_bytes'];cache_bytes+=item['payload']['bytes']+item['row_index']['bytes']+item['carrier_receipt']['bytes']
    require(seen==set(expected) and physical_rows==coverage['total_physical_rows'] and payload_bytes==coverage['total_original_FLAC_bytes'] and
        cache_bytes<=coverage['total_cache_bytes']<=400_000_000_000,'Final carrier source/row/byte/cache totals differ')
    return {'passed':True,'source_files':729,'physical_rows':physical_rows,'original_FLAC_bytes':payload_bytes,'full_payload_rehash_claimed':False}
