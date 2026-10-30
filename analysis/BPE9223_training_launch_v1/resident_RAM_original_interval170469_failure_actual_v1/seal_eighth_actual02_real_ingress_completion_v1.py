"""Actual02 metadata/known-process closure only; no source/runtime/audio import."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import os
import psutil

F=Path(__file__).parent;O=F/'eighth_ingress_frontend_actual_02'
def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    p=Path(path).resolve();data=p.read_bytes();require(len(data)<=8*1024**2,'Compact/source only')
    return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def identity(process):return {'pid':process.pid,'create_time':process.create_time(),'ppid':process.ppid()}
def dead(who):
    try:return abs(psutil.Process(who['pid']).create_time()-who['create_time'])>=.001
    except psutil.NoSuchProcess:return True
r=read(O/'report.json');e=read(F/'eighth_ingress_frontend_external_result_v2.json');tool=read(F/'eighth_ingress_frontend_actual02_TOOL_v1.json')
fixed=read(F/'eighth_ingress_frontend_frozen_inputs_v2.json');certificate=read(fixed['certificate']['path'])
require(tool['actual_TOOL_exit_code']==0 and e['Popen_returncode']==0 and e['passed']is True
    and e['external_guard_error']is None and e['required_external_cleanup']==[] and e['inputs_before_after_exact']is True
    and r['passed']is True and r['guard_error']is None and r['input_pins_before_after_exact']is True,
    'Actual TOOL0/Popen0/inner+outer guardNone/no cleanup/all90pins exact')
require(r['actual_geometry']=={'clean16k_samples':273624,'valid_mel_frames':1710,'physical_feature_frames':1711,'encoder_frames':214}
    and r['arrays']['input_features']['shape']==[1,1711,128]and r['arrays']['input_features']['dtype']=='float32'
    and r['arrays']['attention_mask']['shape']==[1,1711]and r['arrays']['attention_mask']['dtype']=='bool'
    and r['reference']['full_target_ids']==certificate['full_target_ids']and len(r['reference']['full_target_ids'])==99
    and r['reference']['expected_audio']['waveform_sha256']==certificate['clean_f32_sha256']
    and r['actual_native_updates']==0 and r['draws_consumed']==0,'Actual full waveform/frontend/99labels identity')
require(len(r['negative_controls'])==8 and all(v['rejected']for v in r['negative_controls'])
    and sum(bool(v['METHOD_STUB_metadata_loader'])for v in r['negative_controls'])==1
    and r['all_stubs_restored']is True and r['raw_buffers_released_and_metadata_handles_closed']is True
    and r['successful_raw_ledger']['passed']is True and r['previous_policy_delegation']['prior7_policy_exact']is True
    and r['previous_policy_delegation']['parent_method_restored']is True
    and r['previous_policy_delegation']['METHOD_STUB_call_route_control']is True
    and len(r['previous_policy_delegation']['all_seven_prior_row_routes'])==7,'Eight actual rejections/explicit doubles/restoration/raw+DBclosure')
ids=e['recorded_identities']+[e['launcher_identity']];unique={(x['pid'],x['create_time']):x for x in ids}
deaths=[{'identity':x,'exact_dead':dead(x)}for x in unique.values()]
require(len(deaths)==4 and all(v['exact_dead']for v in deaths),'All four exact proof/external/launcher identities dead')
evidence=[F/'eighth_ingress_frontend_actual02_TOOL_v1.json',F/'eighth_ingress_frontend_external_result_v2.json',
    O/'report.json',O/'CPU_completion.json',O/'producer_identity.json',O/'before_real_ingress.json',O/'configuration.json',
    F/'eighth_ingress_frontend_stdout02.log',F/'eighth_ingress_frontend_stderr02.log',
    F/'ROOT_eighth_alias9_CPU13_exact_original_ingress_frontend_CPU02_execution_scope_v1.json',
    F/'verify_eighth_exact_ingress_frontend_v2.py',F/'launch_eighth_exact_ingress_frontend_external_v2.py',
    F/'eighth_ingress_frontend_frozen_inputs_v2.json',Path(fixed['overlay_source']['path']),
    Path(fixed['CPU13_source']['path']),Path(fixed['certificate']['path'])]
me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
value={'kind':'ASR18_eighth_alias9_CPU13_actual02_real_ingress_frontend_completion_v1',
    'observed_UTC':datetime.now(timezone.utc).isoformat(),'actual_TOOL_exit_code':0,'actual_Popen_returncode':0,
    'actual_inner_guard_error':None,'actual_external_guard_error':None,'required_external_cleanup':[],
    'all90input_pins_before_after_exact':True,'all_four_exact_observed_job_identities_dead':True,'exact_job_deaths':deaths,
    'evidence_pins':[pin(p)for p in evidence],'overlay_source':fixed['overlay_source'],'CPU_source':fixed['CPU13_source'],
    'certificate':fixed['certificate'],'actual_CPU_control_report':pin(O/'report.json'),
    'actual_geometry':r['actual_geometry'],'arrays':r['arrays'],'full_target_ids':r['reference']['full_target_ids'],
    'full_waveform_sha256':r['reference']['expected_audio']['waveform_sha256'],
    'historical820896_native_273632_clean_reservations_retained':True,
    'actual_raw_buffers_weakref_released_and_original3DBconnections_locks_closed':True,
    'negative_controls':r['negative_controls'],'previous_policy_delegation':r['previous_policy_delegation'],
    'all_stubs_restored':True,'no_old7audio_baseline_physical_transport_or_proof_rerun':True,
    'no_GPU_model_native_authoritative_sampler_RNG_or_journal_operation':True,
    'source32_or_production_launch_admitted':False,'speed_fit_saving_prevention_or_recovery_claim':False,
    'failed_actual01_stays_failed_and_immutable':pin(F/'eighth_ingress_frontend_actual01_failed_completion_v1.json'),
    'completion_sealer_identity':identity(me),'completion_launcher_identity':identity(me.parent()),
    'completion_self_launcher_alive_at_write_require_external_exact_death_verification':True}
out=F/'eighth_ingress_frontend_actual02_completion_v1.json'
with out.open('x',encoding='utf-8',newline='\n')as file:
    json.dump(value,file,indent=2,ensure_ascii=False,allow_nan=False);file.write('\n');file.flush();os.fsync(file.fileno())
print(json.dumps({'completion':pin(out),'all_four_job_exact_dead':True,'sealer':value['completion_sealer_identity'],
    'launcher':value['completion_launcher_identity'],'no_further_control_or_source32_preparation':True}))
