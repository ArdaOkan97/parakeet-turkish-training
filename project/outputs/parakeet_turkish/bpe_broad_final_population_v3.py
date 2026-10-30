"""Exact final-population binding schema; incremental statuses cannot arm it."""
import json
from pathlib import Path
from bpe_broad_audio_v1 import record,require,scoped

AREA=Path(r'E:\gpu_training_scratchpad\analysis\BPE_weak_then_clean_v1')
REQUIRED=frozenset(('preparation_plan','finalization_contract','completed_source_inventory','source_identity_inventory',
    'acquisition_actual_closed','header_actual_closed','catalog_actual_closed','labels_actual_closed',
    'final_catalog','final_labels','heldout_metadata','heldout_PCM','geometry_admission','carrier_coverage','carrier_coverage_contract'))
CONTRACT_SHA='2e46ddc915b166e1195f14bb1d5391b37b51a99b25f35bce1d003ed4b9c222c0'

def load_bound(item,*,guard):
    require(set(item)=={'path','bytes','sha256'} and type(item['bytes']) is int and item['bytes']>0,'Exact nonempty binding record required')
    path=scoped(item['path']);guard();require(record(path)==item,'Final binding hash/size/path changed')
    require(path.stat().st_size<=8_000_000,'Only bounded finalization metadata can be parsed')
    return json.loads(path.read_text(encoding='utf-8-sig'))

def final_population(receipt,*,guard=lambda:None):
    bindings=receipt.get('bindings',{})
    require(type(bindings) is dict and set(bindings)==REQUIRED,'Every exact final binding required; missing/extra/empty bindings rejected')
    require(receipt.get('kind')=='ASR18_final_production_population_v3' and receipt.get('complete') is True and receipt.get('pass_count')==2,
            'Versioned complete two-pass population required')
    n=receipt['retained_rows'];tokens=receipt['total_target_tokens']
    require(type(n) is int and n>0 and type(tokens) is int and tokens>=n and receipt['label_rows']==n and receipt['frontier']==n
            and receipt['population_mean_U']==tokens/n,'Exact reconciled final row/token/frontier denominator required')
    require(receipt['quality_filters']==[] and receipt['language_filters']==[] and receipt['unresolved_geometry_rows']==0
            and receipt['runtime_heldout_PCM_binding_complete'] is True,'Unfiltered supported geometry and exact reference binding required')
    expected_paths={'preparation_plan':AREA/'preparation_plan_v1.json','finalization_contract':AREA/'production_finalization_contract_v5.json',
        'completed_source_inventory':AREA/'completed_source_inventory_v1.json','source_identity_inventory':AREA/'final_source_identity_inventory_v1.json',
        'final_catalog':AREA.parents[1]/'indexes/BPE_broad_weak_labels_v1.sqlite','final_labels':AREA.parents[1]/'indexes/BPE_broad_labels_8711_v1.sqlite',
        'heldout_metadata':AREA/'heldout_metadata_bindings_v1.json',
        'carrier_coverage':AREA/'KIRAAT_original_FLAC_carrier_coverage_v1.json',
        'carrier_coverage_contract':AREA/'KIRAAT_FLAC_carrier_coverage_contract_v1.json'}
    for key,path in expected_paths.items():require(Path(bindings[key]['path']).resolve()==path.resolve(),'Unexpected final binding path: '+key)
    require(bindings['finalization_contract']['sha256']==CONTRACT_SHA,'Pinned DATA finalization contract version changed')
    docs={}
    for key,item in bindings.items():
        if key in ('final_catalog','final_labels'):
            require(set(item)=={'path','bytes','sha256'} and record(scoped(item['path']))==item,'Final whole DB hash differs')
        else:docs[key]=load_bound(item,guard=guard)
    contract=docs['finalization_contract'];require(contract['kind']=='ASR18_production_finalization_contract_v5','Expected contract kind')
    plan=docs['preparation_plan'];require(len(plan['download_jobs'])==773 and len(plan['reused_archives'])==11,'Complete fixed plan required')
    inventory=docs['completed_source_inventory']
    require(inventory['kind']=='ASR18_completed_source_inventory_v1' and inventory['schema_version']==1 and inventory['sources_count']==784
            and inventory['new_sources_count']==773 and inventory['reused_sources_count']==11 and inventory['total_source_bytes']==718_862_913_959
            and inventory['preparation_plan']==bindings['preparation_plan'] and inventory['finalization_contract']==bindings['finalization_contract'],
            'Completed inventory does not bind the full fixed source population')
    identity=docs['source_identity_inventory']
    require(identity['kind']=='ASR18_final_source_identity_inventory_v1' and identity['complete'] is True and identity['plan']==bindings['preparation_plan']
            and identity['completed_source_inventory']==bindings['completed_source_inventory'] and len(identity['sources'])==784,'Final source signatures missing or from another inventory')
    marker_paths={'acquisition_actual_closed':contract['acquisition_marker_path'],'header_actual_closed':contract['header_marker_path'],
        'catalog_actual_closed':contract['finalization_requirements']['catalog_marker'],'labels_actual_closed':contract['finalization_requirements']['BPE_marker']}
    markers={}
    for key,path in marker_paths.items():
        proof=docs[key]
        require(proof['kind']=='ASR18_source_bound_actual_tool_exit_v1' and proof['owner']=='/root/cpu_data_preparation'
                and proof['actual_exit_code']==0 and bool(proof['actual_tool_chunk_id']),'Actual owned final worker exit required: '+key)
        require(Path(proof['published_marker']['path']).resolve()==Path(path).resolve(),'Tool exit bound to wrong worker marker')
        markers[key]=load_bound(proof['published_marker'],guard=guard)
    acquisition=markers['acquisition_actual_closed'];headers=markers['header_actual_closed'];catalog=markers['catalog_actual_closed'];labels=markers['labels_actual_closed']
    require(acquisition['actual_exit_code']==0 and acquisition['status']=='completed' and acquisition['completed_files']==773,'All new acquisitions must actually finish')
    require(headers['actual_exit_code']==0 and headers['status']=='completed' and headers['completed_archives']==len(headers['indexes'])==51,'All archive indexes must actually finish')
    require(catalog['metadata_complete'] is True and catalog['status']=='completed_metadata_only' and catalog['committed_metadata_files']==730
            and catalog['metadata_retained_rows']==n and catalog['catalog_sha256']==bindings['final_catalog']['sha256']
            and Path(catalog['catalog_path']).resolve()==Path(bindings['final_catalog']['path']).resolve(),'Final catalog count/hash/path differs')
    require(labels['complete'] is True and labels['status']=='completed' and labels['frontier']==labels['label_rows']==n
            and labels['total_target_tokens']==tokens and labels['population_mean_U']==tokens/n
            and labels['catalog_sha256']==bindings['final_labels']['sha256']
            and Path(labels['catalog_path']).resolve()==Path(bindings['final_labels']['path']).resolve(),'Final BPE count/tokens/hash/path differs')
    reference=docs['heldout_PCM'];refcontract=contract['heldout_reference']
    require(reference['kind']=='ASR18_all_reference_PCM_binding_v1' and reference['passed'] is True and reference['complete'] is True
            and len(reference['PCM_i16_sha256'])==reference['reference_clips']==12273 and reference['metadata_binding']==bindings['heldout_metadata']
            and bindings['heldout_PCM']['sha256']==refcontract['report_SHA256']
            and Path(bindings['heldout_PCM']['path']).resolve()==Path(refcontract['report_path']).resolve(),'Complete pinned reference PCM binding required')
    geometry=docs['geometry_admission']
    require(geometry['kind']=='ASR18_final_geometry_admission_v1' and geometry['complete'] is True and geometry['catalog']==bindings['final_catalog']
            and geometry['labels']==bindings['final_labels'] and geometry['rows']==n and geometry['unresolved_rows']==0
            and geometry['quality_filters']==[] and geometry['language_filters']==[] and geometry['supported_processing_proofs'],
            'Every finalized row requires declared supported handling; no quality exclusion')
    for proof in geometry['supported_processing_proofs']:
        require(load_bound(proof,guard=guard)['passed'] is True,'A failed numerical/shape proof cannot support final admission')
    require(bindings['carrier_coverage_contract']==contract['compressed_carrier_coverage_contract'],'Exact DATA carrier contract pin required')
    from bpe_final_carrier_validation_v1 import validate as validate_carriers
    validate_carriers(docs['carrier_coverage'],docs['carrier_coverage_contract'],bindings,inventory,guard=guard)
    require(geometry.get('actual_full_audio_frame_geometry_complete') is True and geometry.get('full_native_allocation_budget_source')==record(Path(__file__).with_name('tdt_full_native_allocation_budget_v1.py')),
            'Final geometry requires actual full-audio frames and complete conservative allocation policy;48*cells/estimated shapes do not suffice')
    from bpe_final_denominator_v1 import audit
    audit(bindings['final_catalog']['path'],bindings['final_labels']['path'],rows=n,tokens=tokens,guard=guard)
    return n,tokens/n
