"""Separate DATA completeness from final native admission without bypasses."""
import sys,json
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import require,record
from atomic_io_retry_v1 import write_json
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def main():
    old=S/'bpe_broad_final_population_v8.py';new=S/'bpe_final_DATA_population_v1.py';require(not new.exists(),'Fresh inactive DATA verifier')
    text=old.read_text();text=text.replace('Exact final-population binding schema; incremental statuses cannot arm it.','Final DATA-only completeness for bounded preparation; never a production arm.')
    require(text.count("'heldout_PCM','geometry_admission','carrier_coverage'")==1,'Exact final/native schema split')
    text=text.replace("'heldout_PCM','geometry_admission','carrier_coverage'","'heldout_PCM','carrier_coverage'")
    text=text.replace('def final_population(receipt,','def final_data_population(receipt,').replace('ASR18_final_production_population_v8','ASR18_final_data_population_for_bounded_preparation_v1')
    text=text.replace("receipt['runtime_heldout_PCM_binding_complete']","receipt['heldout_reference_PCM_binding_complete']")
    begin=text.index("    geometry=docs['geometry_admission']");end=text.index("    require(bindings['carrier_coverage_contract']",begin)
    text=text[:begin]+text[end:]
    begin=text.index("    require(geometry.get('declared_geometry_envelope')");end=text.index('    from bpe_final_denominator_v1 import audit',begin)
    text=text[:begin]+text[end:]
    require(text.count('    return n,tokens/n')==1,'Exact return delta')
    text=text.replace('    return n,tokens/n',"""    return {'kind':'ASR18_validated_final_DATA_population_for_bounded_preparation_v1','schema_version':1,'passed':True,'complete':True,
        'retained_rows':n,'label_rows':n,'frontier':n,'total_target_tokens':tokens,'population_mean_U':tokens/n,'pass_count':2,
        'ordinal_domain':{'min':1,'max':n,'contiguous_unique':True},'quality_filters':[],'language_filters':[],
        'bindings':bindings,'declared_geometry':declared,'heldout_reference_PCM_binding_complete':True,
        'runtime_heldout_PCM_and_actual_geometry_checks_before_consumed_draw_still_required':True,
        'source_verifier':record(__file__),'native_envelope_or_forecast_approved':False,'full_corpus_PCM_decode_validation_claimed':False,
        'sampler_seed_or_frontier_chosen_by_verifier':False,'actual_optimizer_updates':0,'production_enabled':False,'TRAIN_admitted':False}""")
    require('geometry_admission' not in text and 'native_envelope_admission' not in text,'No circular native admission in DATA-only preparation')
    new.write_text(text)
    write_json(A/'final_DATA_population_source_interface_v1.json',{'kind':'ASR18_final_DATA_population_source_interface_v1','source':record(new),
        'preserved_final_production_validator':record(old),'call':'final_data_population(receipt,guard=...)',
        'input_receipt_kind':'ASR18_final_data_population_for_bounded_preparation_v1','output_kind':'ASR18_validated_final_DATA_population_for_bounded_preparation_v1',
        'all_final_DATA_assets_inventory_carriers_catalog_labels_declared_geometry_and_actual_exits_required':True,
        'full_target_sum_contiguous_ordinals_and_reference_heldout_binding_required':True,'no_partial_mean_or_geometry_status_admission':True,
        'no_upfront_full_PCM_requirement':True,'no_circular_native_forecast_requirement':True,'runtime_actual_geometry_and_heldout_match_before_draw_required':True,
        'bounded_preparation_only':True,'final_positive_path_exercised':False,'ROOT_read_review_pending':True,'production_enabled':False})
    print(json.dumps({'DATA_verifier':record(new),'production_enabled':False}))
if __name__=='__main__':main()
