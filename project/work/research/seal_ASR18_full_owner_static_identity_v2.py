"""Source/config-only successor: exact role-map and future-owner serializer order."""
from pathlib import Path
import ast
P=Path(__file__).resolve().parents[2];R=P/'work/research'
source=R/'prepare_ASR18_full_owner_configuration_v1.py';target=R/'prepare_ASR18_full_owner_configuration_v2.py'
assert not target.exists();text=source.read_text(encoding='utf-8-sig')
text=text.replace("controller=template['full_owner_controller_source'];prefix=template['full_run_identity']['persistent_prefix_sources']['prefix_verifier']",
"controller=record(template['full_owner_controller_source']['path']);prefix=record(template['full_run_identity']['persistent_prefix_sources']['prefix_verifier']['path'])")
text=text.replace("'transition_source':record(S/'bpe_full_run_durable_owner_adapter_v1.py'),",
"'transition_source':read_bound(template['durable_policy'],guard)['transition_source'],")
ast.parse(text);target.write_text(text,encoding='utf-8',newline='\n')
source=R/'seal_ASR18_full_owner_static_identity_v1.py';text=source.read_text(encoding='utf-8-sig')
text=text.replace('prepare_ASR18_full_owner_configuration_v1.py','prepare_ASR18_full_owner_configuration_v2.py')
for name in ('full_owner_source_contract','full_owner_validation_identity','full_owner_external_protection_plan','full_owner_exact_ROOT_review_inputs'):
    text=text.replace(name+'_v1.json',name+'_v2.json')
before=""" 'required_roles':['source','history','DEV','qualified_winner'],
 'source':[hard['planning_arguments']['snapshot_binding']],
 'history':[pin(A/'hard_effective16_native_actual_01/actual_hard_effective16_end_to_end_summary_v1.json')],
 'DEV':[pin(A/'ROOT_fixed_serial350DEV_manifest_membership_review_v1.json')],
 'qualified_winner':[qualified_summary],'adopt_external_sources_as_owned':False}"""
after=""" 'required_roles':{
 'source':[hard['planning_arguments']['snapshot_binding']],
 'history':[pin(A/'hard_effective16_native_actual_01/actual_hard_effective16_end_to_end_summary_v1.json')],
 'DEV':[pin(A/'ROOT_fixed_serial350DEV_manifest_membership_review_v1.json')],
 'qualified_winner':[qualified_summary]},'adopt_external_sources_as_owned':False}"""
assert text.count(before)==1;text=text.replace(before,after)
ast.parse(text);exec(compile(text,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
