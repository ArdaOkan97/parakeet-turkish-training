"""Actual failed import proof metadata/death seal only. No corrected source/job."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import os
import psutil

F=Path(__file__).parent;OUTPUT=F/'eighth_ingress_frontend_actual_01'
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
e=read(F/'eighth_ingress_frontend_external_result_v1.json');tool=read(F/'eighth_ingress_frontend_actual01_failed_TOOL_v1.json')
stderr=(F/'eighth_ingress_frontend_stderr.log').read_text(encoding='utf-8-sig')
require(tool['actual_TOOL_exit_code']==1 and e['Popen_returncode']==1 and e['passed']is False
    and e['external_guard_error']is None and e['required_external_cleanup']==[]
    and e['inputs_before_after_exact']is True and e.get('report')is None and not OUTPUT.exists(),
    'Preserve actual1/no guard-cleanup/all75pins exact/no report or output directory')
require("AttributeError: module 'fastmss_original_clip_geometry_overlay_v6' has no attribute 'decode_full_member'"in stderr
    and 'decode_full_member=prior.prior.decode_full_member'in stderr,'Exact original propagated import failure traceback')
ids=e['recorded_identities']+[e['launcher_identity']];unique={(x['pid'],x['create_time']):x for x in ids}
deaths=[{'identity':x,'exact_dead':dead(x)}for x in unique.values()]
require(len(deaths)==4 and all(v['exact_dead']for v in deaths),'All four exact failed-job native/launcher/external identities dead')
evidence=[F/'eighth_ingress_frontend_actual01_failed_TOOL_v1.json',F/'eighth_ingress_frontend_external_result_v1.json',
    F/'eighth_ingress_frontend_stdout.log',F/'eighth_ingress_frontend_stderr.log',
    F/'ROOT_eighth_exact_original_ingress_frontend_CPU_execution_scope_v1.json',
    F/'verify_eighth_exact_ingress_frontend_v1.py',F/'launch_eighth_exact_ingress_frontend_external_v1.py',
    F/'eighth_ingress_frontend_frozen_inputs_v1.json',F/'configuration_source32_overlay8_scoped_source_candidate_v2.json']
fixed=read(F/'eighth_ingress_frontend_frozen_inputs_v1.json')
evidence.extend((Path(fixed['overlay_source']['path']),Path(fixed['CPU12_source']['path']),Path(fixed['certificate']['path'])))
me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
out=F/'eighth_ingress_frontend_actual01_failed_completion_v1.json'
value={'kind':'ASR18_eighth_ingress_frontend_actual01_import_failure_completion_v1',
    'observed_UTC':datetime.now(timezone.utc).isoformat(),'actual_TOOL_exit_code':1,'actual_Popen_returncode':1,
    'external_guard_error':None,'required_external_cleanup':[],'all75input_pins_before_after_exact':True,
    'all_four_exact_failed_job_identities_dead':True,'exact_job_deaths':deaths,
    'evidence_pins':[pin(p)for p in evidence],'actual_stderr_traceback':stderr,
    'supported_cause':'New overlay8 copied prior.prior alias from overlay7, but its prior is now overlay7; prior.prior names overlay6, which lacks decode_full_member. Import fails at module definition before ingress.',
    'no_actual_CPU_startup_ingress_audio_frontend_negativecase_or_transport_execution':True,
    'no_report_or_actual01_output_directory_created':True,
    'physical_certificate_and_all_closed_localization_fullmember_proofs_unchanged':True,
    'preserved_metadata_sealer_v1_failure':pin(F/'eighth_ingress_actual01_metadata_sealer_v1_failure_history.json'),
    'historical_failed_metadata_sealer_pid_ctime':'UNRECORDED; not invented',
    'all_source_config_history_preserved_and_no_retry':True,
    'proposed_but_NOT_implemented':'Separate immutable successor: change only the helper alias to direct prior.decode_full_member, preserve all numerical/guard/certificate/prior7 code, then new CPU overlay import and genuine ROOT gates/freshactual02. No old physical/transport reruns.',
    'all_actual_training_still_stopped_saved170400_69historical_tail_preserved':True,
    'completion_sealer_identity':identity(me),'completion_launcher_identity':identity(me.parent()),
    'completion_sealer_launcher_alive_at_write_need_external_exact_death_check':True}
with out.open('x',encoding='utf-8',newline='\n')as file:
    json.dump(value,file,indent=2,ensure_ascii=False,allow_nan=False);file.write('\n');file.flush();os.fsync(file.fileno())
print(json.dumps({'completion':pin(out),'all_four_failed_job_exact_dead':True,
    'sealer':value['completion_sealer_identity'],'launcher':value['completion_launcher_identity'],'no_retry_or_source_fix_executed':True}))
