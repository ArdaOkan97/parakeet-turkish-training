"""Seal the explicit diagnostic bridge/CPU7 caller closure; no ML imports."""
import ast,hashlib,json,os
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');F=A.parent/'FastMSS_integration_v1'
def pin(path):
    path=Path(path).resolve();return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def new(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return pin(path)
prior_path=A/'targeted_cleaner23_full_caller_source_review_manifest_v1.json';prior=read(prior_path)
bundle_path=F/'FastMSS_targeted_cleaner_CPU7_source_review_manifest_v1.json';bundle=read(bundle_path)
for item in bundle['source']+bundle['diagnostic_dependencies']+[bundle['template'],bundle['interface']]:
    assert pin(item['path'])==item
template=read(prior['configuration_template']['path'])
template.update(kind='ASR18_targeted_cleaner23_config_TEMPLATE_NOT_LAUNCHABLE_v2',
    parent_driver_source=pin(R/'run_ASR18_targeted_cleaner23_v2.py'),
    worker_driver_source=str(R/'run_ASR18_targeted_cleaner23_v2.py'),
    CPU_controller_source=pin(R/'prepare_FastMSS_phase_window_CPU_v7.py'),
    cleaner_conditional_CPU_resource_review=pin(A/'ROOT_cleaner_CPU7_conditional_resource_component_review_v1.json'),
    targeted_cleaner_diagnostic_context=None,actual_metadata_bridged_cleaner_plan=None)
template['launch_requires']=[value.replace('CPU6','CPU7') for value in template['launch_requires']]
template['launch_requires']+=['Explicit diagnostic bridge context must bind original actual22 ancestry and exact unchanged initial diagnostic/replay state',
    'Actual metadata-only bridge_plan call on the real four-row plan must pass before any audio/controller/GPU launch']
tp=new(S/'asr18_targeted_cleaner23.disabled.v2.json',template)
paths=[R/'run_ASR18_targeted_cleaner23_v2.py',R/'prepare_ASR18_targeted_cleaner23_configuration_v2.py',
    S/'asr18_targeted_cleaner23_native_v2.py',S/'asr18_targeted_cleaner23_recovery_v2.py',S/'asr18_targeted_cleaner23_restore_runtime_v2.py']
for path in paths:ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
direct=[pin(path) for path in paths]
bindings=[];seen=set()
for part in prior['bindings']+direct+bundle['source']+bundle['diagnostic_dependencies']+[pin(bundle_path),bundle['template'],bundle['interface'],tp]:
    if part['path'] not in seen:bindings.append(part);seen.add(part['path'])
manifest={**prior,'kind':'ASR18_targeted_cleaner23_CPU7_explicit_diagnostic_bridge_source_review_manifest_v2',
    'later_execution_review_scope_required':'finite_targeted_cleaner23_CPU7_native_and_fresh_recovery_diagnostic_v2',
    'direct_sources':direct,'bindings':bindings,'configuration_template':tp,'CPU_owner_bundle':pin(bundle_path),
    'previous_source_manifest':pin(prior_path),
    'previous_source_read_review':pin(A/'ROOT_targeted_cleaner23_source_read_review_v1.json'),
    'cleaner_conditional_CPU_resource_review':template['cleaner_conditional_CPU_resource_review'],
    'preserved_actual_failure':pin(A/'targeted_cleaner23_native_actual_01/actual_owned_CLOSED1_v1.json'),
    'diagnostic_schedule_adapter_unchanged':pin(S/'asr18_targeted_cleaner_schedule_context_v1.py'),
    'source_delta':'CPU7/bridge6/sink5 source binding plus explicit diagnostic context in plan/session and actual metadata bridge call; unchanged schedule/native/RNG/target math',
    'actual_new_audio_frontend_or_GPU_jobs':0}
mp=new(A/'targeted_cleaner23_full_caller_source_review_manifest_v2.json',manifest)
print(json.dumps({'manifest':mp,'template':tp,'CPU_owner_bundle':pin(bundle_path),'direct_sources':direct,'binding_count':len(bindings)}),flush=True)
