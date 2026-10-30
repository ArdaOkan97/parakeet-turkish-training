"""NEW builder raw-state assertion fix and preserved actual metadata01 failure."""
from pathlib import Path
import ast,hashlib,json
P=Path(__file__).resolve().parents[2];R=P/'work/research'
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    path=Path(path).resolve();raw=path.read_bytes();return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(path,value):
    assert not path.exists();path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n');return pin(path)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
closure_pin=pin(A/'full_owner_eight_update_configuration_guard_01/closure.json');closure=read(closure_pin['path'])
assert closure['pid']==33216 and closure['actual_exit_code']==1 and closure['monitor_clean'] and closure['monitor_joined']
assert not (A/'full_owner_eight_update_configuration_01').exists()
failure=write(A/'full_owner_metadata01_actual_TOOL1_v1.json',{
 'kind':'ASR18_operator_observed_full_owner_metadata_TOOL1_v1','owner':'/root/memory_diagnosis',
 'actual_tool_chunk_id':'20672f','actual_tool_exit_code':1,'actual_native_exit_code':1,'native_PID':33216,
 'worker_source':pin(R/'prepare_ASR18_full_owner_configuration_v5.py'),'guard_closure':closure_pin,
 'ROOT_metadata_review':pin(A/'ROOT_full_owner_template2_metadata_source_read_review_v1.json'),
 'failure':{'type':'KeyError','key':'ends_population_pass','stage':'raw coordinator plan assertion'},
 'actual_configuration_output_created':False,'actual_new_audio_frontend_GPU_or_Adam_jobs':0,
 'production_enabled':False,'automatic_unchanged_retry_performed':False})
old=R/'prepare_ASR18_full_owner_configuration_v5.py';new=R/'prepare_ASR18_full_owner_configuration_v6.py'
assert not new.exists();text=old.read_text(encoding='utf-8-sig')
before="""        require(schedule.committed_updates()==0 and digest(initial)==digest(schedule.state_dict())==plan['initial_sampler_sha256']
            and len(plan['updates'])==4 and all(not row['ends_population_pass'] for row in plan['updates']),
            'Actual original first4, no counter reset/consumed draw/fake pass or DEV')"""
after="""        from bpe_full_run_lifecycle_v2 import progress,next_boundary_request
        final=plan['final_private_planning_sampler'];frontier=progress(final)
        require(schedule.committed_updates()==0 and digest(initial)==digest(schedule.state_dict())==plan['initial_sampler_sha256']
            and len(plan['updates'])==4 and frontier['optimizer_updates']==4 and frontier['broad_draws']==56
            and frontier['cleaner_draws']==0 and next_boundary_request(final,0)['kind']=='bounded_training_window',
            'Actual original first4 private preview; no consumed draw or fabricated pass/DEV')"""
assert text.count(before)==1;text=text.replace(before,after);ast.parse(text);compile(text,str(new),'exec');new.write_text(text,encoding='utf-8',newline='\n')
manifest=read(A/'full_owner_source_contract_v6.json');new_pin=pin(new);old_path=str(old)
manifest['bindings']=[new_pin if item['path']==old_path else item for item in manifest['bindings']]
manifest['direct_sources'].pop(old.name);manifest['direct_sources'][new.name]=new_pin
manifest_pin=write(A/'full_owner_source_contract_v7.json',manifest)
identity=read(A/'full_owner_validation_identity_v6.json');identity['full_run_source_contract']=manifest_pin
identity_pin=write(A/'full_owner_validation_identity_v7.json',identity)
inputs=read(A/'full_owner_exact_ROOT_review_inputs_v7.json')
inputs.update(source_manifest=manifest_pin,identity=identity,identity_binding=identity_pin,run_identity_sha256=digest(identity),
 metadata_builder_source=new_pin,configuration_output=str(A/'full_owner_eight_update_configuration_02'),
 preserved_metadata01_actual1=failure)
inputs_pin=write(A/'full_owner_exact_ROOT_review_inputs_v8.json',inputs)
print(json.dumps({'builder':new_pin,'source_contract':manifest_pin,'identity':identity_pin,'ROOT_inputs':inputs_pin,
 'preserved_actual1':failure,'scope':'AST_and_small_metadata_only_no_target_import_or_model_job'}))
