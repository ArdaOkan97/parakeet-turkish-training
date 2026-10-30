"""Small oracle and numeric resource proposal; never builds the corpus index."""
import importlib,json,sys
from pathlib import Path
D=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1')
sys.path.insert(0,str(D));m=importlib.import_module('audit_and_fit_append_only_v5')
m.AREA=D/'compact_fit_v6'
from compact_pair_trainer_v1 import oracle_fixture

def main():
 g=m.Guard();g.phase='cheap_compact_trainer_oracle_and_resource_proposal'
 f=oracle_fixture();m.write('compact_trainer_independent_fixture.json',f)
 shape=json.loads((m.AREA/'readonly_shape_preflight.json').read_text())
 W=shape['word_types'];N=shape['initial_sequence_cells_N'];K=600000
 P=shape['initial_adjacencies_upper_member_count']+2*N
 terms={
  'per_word_sequence_array_objects_96B_each_plus_1point5_capacity_uint16':W*96+N*3,
  'sequence_pointer_list_12B_per_word':W*12,
  'uint64_frequency_array_1point5_capacity':W*12,
  'three_pair_dict_tables_128B_each_per_key_including_resize_overlap':3*K*128,
  'worst_three_independent_packed_int_key_objects_40B_each':3*K*40,
  'uint64_checked_Python_pair_count_objects_40B_each':K*40,
  'combined_piece_unicode_objects_max24letters_and_metaspace_160B':K*160,
  'posting_array_objects_96B_each':K*96,
  'uint32_cumulative_postings_1point5_capacity_including_retired_iteration_array':P*6,
  'guard_SQLite_Rust_json_runtime_baseline_and_validation_overlap':384000000,
  'bounded_serialization_counter_temporaries_and_canonical_tie_strings':64*1024*1024,
 }
 total=sum(terms.values());assert total<1800000000 and total+200000000<2000000000
 proposal={'kind':'Unaccepted_compact_BPE_RAM_resource_proposal_v1','at_utc':m.now(),
  'accepted':False,'allocation_executed':False,'shape':m.record(m.AREA/'readonly_shape_preflight.json'),
  'actual_v5_closure':m.record(D/'all_TRAIN_extension_v5/actual_owned_CLOSED1.json'),
  'fixture':m.record(m.AREA/'compact_trainer_independent_fixture.json'),
  'source':m.record(D/'compact_pair_trainer_v1.py'),'guard_source':m.record(D/'audit_and_fit_append_only_v5.py'),
  'reuse_original_immutable_words_only':True,'ignore_partial_mutable_working_pairs_members':True,
  'sealed256_prefix_replay_required_exact_frequency_piece_rank_ID_remap_and_affected_types':True,
  'final_candidate_base8711_no_manual_Ucircumflex':True,'new_merges':512,
  'initial_words':W,'initial_sequence_cells_N':N,'initial_pair_K_upper':shape['current_all_created_pair_rows_K_upper_initial'],
  'live_pair_key_cap':K,'cumulative_posting_cell_cap':P,
  'posting_bound_proof':'Each actual token replacement reduces sequence cells by one and creates at most two boundary pairs. Initial members<=N-W; all later new memberships<=2*replacements<2N. A removed old-ID adjacency cannot reappear because replacement creates a fresh token ID and never splits. Stale postings retire only when total count zero or pair becomes ineligible; cumulative bound includes them.',
  'heap_entries':0,'selection':'max integer weighted count, then exact canonical JSON pair string order; no heap and no tolerance/pruning',
  'uint16_token_ids_max':9222,'uint32_word_index_max':W-1,'weighted_count_uint64_no_overflow_checks':True,
  'union_budget_terms_bytes':terms,'forecast_leaf_private_bytes':total,
  'forecast_whole_with_parent_allowance_bytes':total+200000000,
  'maximum_actual_leaf_private_bytes':1800000000,'maximum_actual_whole_private_bytes':2000000000,
  'runtime_baseline_cap_bytes':384000000,'parent_actual_private_cap_bytes':200000000,
  'fresh_current_private_plus_forecast_increment_system_headroom_check_before_index':True,
  'system90yield_95critical_2GBheadroom_E100GBreserve':True,'maximum_actual_and_logical_IO_bytes':16000000000,
  'time_cap_seconds':3600,'CPU_threads':1,'priority':'BelowNormal','no_GPU_active_asset_mutation':True}
 m.write('RAM_resource_proposal_v1.json',proposal)
 g.phase='proposal_and_small_oracle_complete';g.publish(proposal=str(m.AREA/'RAM_resource_proposal_v1.json'),oracle=f)
 print(json.dumps({'proposal':m.record(m.AREA/'RAM_resource_proposal_v1.json'),'leaf_budget':total,'whole_budget':total+200000000,'fixture':f}))
if __name__=='__main__':main()
