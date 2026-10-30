"""Stdlib-only immutable successor bindings; no model/data imports or jobs."""
from pathlib import Path
import ast,copy,hashlib,json
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def write(path,value):
    assert not path.exists();path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n');return pin(path)
controller=S/'asr18_full_run_parent_controller_v12.py';parent=R/'run_ASR18_full_run_owner_v14.py'
builder=R/'prepare_ASR18_full_owner_configuration_v9.py';helper=S/'asr18_worker_control_pool_v1.py'
for path in (controller,parent,builder,helper):ast.parse(path.read_text(encoding='utf-8-sig'))
old=read(A/'full_owner_exact_ROOT_review_inputs_v10.json')
source=copy.deepcopy(read(A/'full_owner_source_contract_v9.json'))
bindings={item['path']:item for item in source['bindings']}
for path in (controller,parent,builder,helper):bindings[str(path)]=pin(path)
source['bindings']=sorted(bindings.values(),key=lambda item:item['path'])
source['direct_sources']={**source['direct_sources'],**{path.name:pin(path) for path in (controller,parent,builder,helper)}}
source.update(normal_boundary_resume='Same immutable genesis identity; actual owner0/dead and native0/CURRENT recovery0/prefixACK/pending0/whole journal required; no crash-tail/adoption',
    control_scope='unchanged fixed1GiB+48 LIVE worker slots64MiB with lossless source-owned CLOSED0 archive; cumulative513 and retained owner-root capacity conditional',
    finite_validation='same production body first4 normal pause/owner0, fresh owner next4; exact uninterrupted8 scientific oracle, no pass/DEV fabrication')
source_pin=write(A/'full_owner_source_contract_v10.json',source)
identity=copy.deepcopy(old['identity']);identity['full_run_source_contract']=source_pin
identity.update(permanent_namespace_scope='original_source_normal_boundary_resume_split4_4_validation_no_production_ancestry',
    normal_boundary_resume_enabled=True,crash_tail_rollback_supported=False)
identity['persistent_prefix_sources'].update(metadata_resolver=pin(controller),native_parent=pin(parent),recovery_parent=pin(parent))
identity_pin=write(A/'full_owner_validation_identity_v10.json',identity)
inputs=copy.deepcopy(old);root=Path(r'E:\gpu_training_scratchpad\runs\ASR18_full_owner_normal_resume_validation_08_01')
inputs.update(source_manifest=source_pin,identity=identity,identity_binding=identity_pin,run_identity_sha256=digest(identity),
    run_namespace=str(root),storage_namespace=str(root/'checkpoint_storage'),durable_namespace=str(root/'durable'),
    actual_folder=str(A/'full_owner_normal_resume_actual_01'),configuration_output=str(A/'full_owner_normal_resume_configuration_01'),
    owner_verifier_source=pin(controller),parent_source=pin(parent),metadata_builder_source=pin(builder),
    finite_scope='finite_full_owner_first4_normal_pause_resume_next4_equivalence_v1',
    normal_boundary_resume_enabled=True,owner_validation_pause_after_updates=4,
    uninterrupted_eight_update_oracle=pin(A/'full_owner_eight_update_actual_02/actual_full_owner_eight_update_pipeline_summary_v1.json'),
    previous_actual_scope_is_immutable_external_oracle_not_restore_ancestry=True,
    worker_control_pool_source=pin(helper))
inputs['lifecycle_callback_sources']={key:pin(controller) for key in old['lifecycle_callback_sources']}
inputs['proof_callback_sources']={key:(pin(controller) if key!='verify_current_recovery_and_persistence_prefix' else value) for key,value in old['proof_callback_sources'].items()}
inputs['DEV_execution_sources'].update(parent_source=pin(parent),guard_source=pin(parent))
inputs['DEV_native_boundary_sources'].update(native_parent=pin(parent),recovery_parent=pin(parent))
# Only source map identity changes; fixed actual evaluator and all assets remain.
inputs['DEV_source_bindings'].update({path.name:pin(path) for path in (controller,helper)})
inputs_pin=write(A/'full_owner_exact_ROOT_review_inputs_v11.json',inputs)
print(json.dumps({'source_contract':source_pin,'identity':identity_pin,'ROOT_inputs':inputs_pin,'run_identity_sha256':digest(identity),'bindings':len(bindings)}))
