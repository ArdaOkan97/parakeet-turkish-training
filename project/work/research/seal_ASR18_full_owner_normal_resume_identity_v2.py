"""Frozen head-field successor source/identity join; no target imports/jobs."""
from pathlib import Path
import ast
P=Path(__file__).resolve().parents[2]
source=P/'work/research/seal_ASR18_full_owner_normal_resume_identity_v1.py'
text=source.read_text(encoding='utf-8-sig')
text=text.replace('asr18_full_run_parent_controller_v12.py','asr18_full_run_parent_controller_v13.py')
text=text.replace('run_ASR18_full_run_owner_v14.py','run_ASR18_full_run_owner_v15.py')
text=text.replace('prepare_ASR18_full_owner_configuration_v9.py','prepare_ASR18_full_owner_configuration_v10.py')
text=text.replace("write(A/'full_owner_source_contract_v10.json'","write(A/'full_owner_source_contract_v11.json'")
text=text.replace("write(A/'full_owner_validation_identity_v10.json'","write(A/'full_owner_validation_identity_v11.json'")
text=text.replace("write(A/'full_owner_exact_ROOT_review_inputs_v11.json'","write(A/'full_owner_exact_ROOT_review_inputs_v12.json'")
text=text.replace("inputs_pin=write(","inputs.update(worker_control_callback_sources={'verify_worker_control_protection':pin(parent),'verify_closed_worker_control':pin(parent)},worker_control_CPU_fixture=pin(A.parent/'FastMSS_integration_v1/worker_control_pool_actual_CPU_handoff_v1.json'))\ninputs_pin=write(")
ast.parse(text);exec(compile(text,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
