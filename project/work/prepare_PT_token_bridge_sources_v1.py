from pathlib import Path
import ast

root=Path(__file__).resolve().parent.parent
base=root/'outputs/parakeet_turkish'
source=(base/'record_TSC_uniform_retention_constraint_source_v1.py').read_text(encoding='utf-8')
source=source.replace('Measured feature', 'Measured PT token')
source=source.replace('tsc_uniform_retention_constraint_transition_v1','tsc_uniform_pttoken_constraint_transition_v1')
source=source.replace('"""Byte archive for one measured feature-only fork from coherent qualified BEST.\n\nActual source/diagnostic CLOSED evidence precedes .pt hashing; failed latest\nand full8750 history remain separate from chosen8250. No ML/tensor load.\n"""',
'''"""Byte archive for PT-token-only continuation from coherent reviewed latest8750.

Actual source CLOSED evidence precedes .pt hashing. Separate qualified BEST8250
and original numeric/feature lineage remain archived without any rewind.
"""''')
source=source.replace("review.get('chosen_step')==8250", "review.get('chosen_step')==8750")
source=source.replace("review.get('chosen_checkpoint_role')=='best'", "review.get('chosen_checkpoint_role')=='latest'")
source=source.replace('allow_declared_feature_distillation_constraint_fork','allow_declared_PT_token_constraint_fork')
source=source.replace("review.get('source_counter_preserved')==0", "review.get('source_counter_preserved')==2")
source=source.replace('Actual delegated measured qualified8250 feature-only review required','Actual delegated reviewed latest8750 PT-token-only review required')
source=source.replace("closed=read(checked_record(review['actual_closed_source_review']));diagnostic=read(checked_record(review['actual_diagnostic']))", "closed=read(checked_record(review['actual_closed_source_review']));checked_record(review['actual_diagnostic'])")
source=source.replace("and review['actual_diagnostic_owned_exit']['exit_code']==0", "")
source=source.replace("and closed['completed_steps']==8750 and diagnostic.get('passed') is True,'Actual source and diagnostic CLOSED0 required'", "and closed['completed_steps']==8750 and closed['run_id']==review['source_run_id'],'Actual source CLOSED0 required'")
source=source.replace("and scoped(selection['weights'])==scoped(a['chosen_checkpoint']['path'])", "and selection['consecutive_retention_failures']==2\n        and scoped(selection['weights'])==scoped(a['source_best_checkpoint']['path'])")
source=source.replace("and scoped(selection['latest_weights'])==scoped(a['latest_checkpoint']['path'])", "and scoped(selection['latest_weights'])==scoped(a['chosen_checkpoint']['path'])\n        and a['chosen_checkpoint']==a['latest_checkpoint']")
source=source.replace('Actual coherent selected BEST and separate unqualified latest required before .pt bytes','Actual coherent latest8750/counter2 and separate BEST8250 required before .pt bytes')
source=source.replace('New explicit measured feature-only scientific namespace required','New explicit PT-token-only scientific namespace required')
source=source.replace("_history_frontier(a['training_history']['path'],8250,8750)", "_history_frontier(a['training_history']['path'],8750,8750)")
source=source.replace('actual_CLOSED_qualified_BEST_measured_feature_constraint','actual_CLOSED_reviewed_latest_PT_token_constraint')
source=source.replace("'chosen_checkpoint_role':'best','chosen_step':8250", "'chosen_checkpoint_role':'latest','chosen_step':8750")
source=source.replace("'explicit_coverage_rewind_clips':7000", "'explicit_coverage_rewind_clips':0")
source=source.replace('original_integer_source_receipt','original_feature_source_receipt')
source=source.replace("'source_chosen_is_qualified':True,'source_counter_preserved':0", "'source_chosen_is_qualified':False,'source_counter_preserved':2")
source=source.replace("'source_step':8250,'failed_latest_step':8750", "'source_step':8750,'separate_selected_BEST_step':8250")
source=source.replace("'explicit_rewind_TR_clips':7000,'active_feature_coefficient':20", "'explicit_rewind_TR_clips':0,'active_PT_token_multiplier':5")
source=source.replace("'source_step':8250,'GPU_launched'", "'source_step':8750,'GPU_launched'")
target=base/'record_TSC_uniform_pttoken_constraint_source_v1.py'
assert not target.exists()
ast.parse(source)
target.write_text(source,encoding='utf-8')
