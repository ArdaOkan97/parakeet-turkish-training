"""Lightweight actual decoder metadata/death closure; NO audio or control run."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import os
import psutil

F=Path(__file__).parent;O=F/'ordered14_decoder_actual_01'
def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    p=Path(path).resolve();data=p.read_bytes();require(len(data)<=8*1024**2,'Compact/source only')
    return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def identity(process):return {'pid':process.pid,'create_time':process.create_time(),'ppid':process.ppid()}
def exact_dead(who):
    try:return abs(psutil.Process(who['pid']).create_time()-who['create_time'])>=.001
    except psutil.NoSuchProcess:return True
report=read(O/'report.json');external=read(F/'ordered14_decoder_external_result_v1.json')
tool=read(F/'ordered14_decoder_actual_TOOL_v1.json')
require(tool['actual_TOOL_exit_code']==0 and external['Popen_returncode']==0 and external['passed']is True
    and external['external_guard_error']is None and external['required_external_cleanup']==[]
    and external['inputs_before_after_exact']is True and report['input_pins_before_after_exact']is True
    and report['guard_error']is None,'Actual outcome, guard, unchanged inputs and cleanup')
failure=report['first_actual_exception'];who=failure['identity'];row=who['candidate']['full_original_plan_row'];raw=who['candidate']['raw_original_annotation']
require(who['global_update']==170470 and who['local_update_index']==9 and who['original_update_row_1based']==6
    and row['ordinal']==332560 and row['source_id']=='alania:aedf14e56cf50416170171503e0dee9a_00000'
    and raw['start']==0 and raw['end']==17.102 and len(who['full_target_ids'])==99
    and failure['exception_type']=='ValueError'and failure['exception_message']=='Original interval beyond source unresolved'
    and [x['identity']['original_update_row_1based']for x in report['results']]==list(range(1,6)),
    'Actual first propagated failure and exactly five preceding ordered successes')
expected={'written':820872,'count':820896,'decoded':856,'begin':0,'finish':820896,'sr':48000,'channels':2,
    'sample_cursor':820872,'name':'aedf14e56cf50416170171503e0dee9a.webm','offset':6190319616,'size':261223,
    'origin':'0','start':'0','end':'8551/500','first_selected_pts':'0'}
require(failure['decoder_scalar_locals']==expected,'Actual saved scalar coverage evidence')
ids=external['recorded_identities']+[external['launcher_identity']]
unique={(x['pid'],x['create_time']):x for x in ids}
deaths=[{'identity':x,'exact_dead':exact_dead(x)}for x in unique.values()]
require(all(v['exact_dead']for v in deaths),'Every exact fixture/external/launcher identity dead')
evidence=[F/'ordered14_decoder_actual_TOOL_v1.json',F/'ordered14_decoder_external_result_v1.json',
    F/'ROOT_recorded170470_ordered14_original_decoder_exact_CPU_execution_scope_v1.json',
    F/'ordered14_decoder_frozen_inputs_v1.json',F/'observe_recorded170470_ordered14_decoder_v1.py',
    F/'launch_recorded170470_ordered14_decoder_external_v1.py',O/'report.json',O/'CPU_completion.json',
    O/'producer_identity.json',O/'before_row_06.json',O/'first_actual_exception.json',O/'first_decoder_exception_immediate.json',
    F/'ordered14_decoder_stdout.log',F/'ordered14_decoder_stderr.log']
for index in range(1,6):evidence.extend((O/f'before_row_{index:02}.json',O/f'success_row_{index:02}.json'))
canonical=lambda value:json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
completion={'kind':'ASR18_ordered14_decoder_actual_CPU_result_and_exact_death_closure_v1',
    'observed_UTC':datetime.now(timezone.utc).isoformat(),'actual_TOOL_exit_code':0,'actual_Popen_returncode':0,
    'external_guard_error':None,'required_external_cleanup':[],
    'diagnostic_complete_NOT_audio_pass_physical_cause_or_recovery':True,
    'all49_input_pins_before_after_exact':True,'all_exact_observed_job_identities_dead':True,'exact_job_deaths':deaths,
    'evidence_pins':[pin(path)for path in evidence],
    'global_update':170470,'recorded_plan154_index':9,'first_failing_original_row_1based':6,
    'failing_ordinal':row['ordinal'],'failing_source_id':row['source_id'],'original_annotation':raw,
    'original_declaration':row['declared_geometry'],'full_target_ids':who['full_target_ids'],
    'full_target_ids_canonical_sha256':hashlib.sha256(canonical(who['full_target_ids'])).hexdigest(),
    'full_plan_row_canonical_sha256':hashlib.sha256(canonical(row)).hexdigest(),
    'raw_annotation_canonical_sha256':hashlib.sha256(canonical(raw)).hexdigest(),
    'actual_propagated_exception':{k:failure[k]for k in('exception_type','exception_module','exception_message','exception_repr','traceback','decoder_scalar_locals')},
    'immediate_trace_event_is_observational_not_authoritative':True,
    'five_preceding_successful_rows':[{'row_1based':v['identity']['original_update_row_1based'],
        'ordinal':v['identity']['candidate']['ordinal'],'source_id':v['identity']['candidate']['source_id'],
        'full_declared_decoder_PCM_and_heldout_membership_checks_passed':v['observation']['passed']}for v in report['results']],
    'rows7_through14_not_decoded':True,'baseline_rerun_executed':False,'fullmember_read_executed':False,
    'actual_native_physical_PCM_hash':'UNKNOWN; failed ordinary decoder returned no full PCM',
    'actual_clean_PCM_hash_and_geometry':'UNKNOWN; no returned clean buffer or frontend execution',
    'decoder_coverage_deficit_samples':24,'deficit_is_not_independent_physical_count_or_discard_padding_cause_proof':True,
    'no_source_policy_guard_geometry_tolerance_science_sampler_RNG_journal_native_GPU_or_production_change':True,
    'saved_checkpoint_remains170400_and69_historical_attempts_need_genuine_redo':True,
    'next_scope_proposal_only':'If ROOT accepts this localization, freeze ONE independent exact complete head-member physical/cumulative+PTS/discard-padding observation for only shard0022/member aedf14e56cf50416170171503e0dee9a.webm at offset6190319616 size261223, all original physical samples once, full99labels, native/clean hashes and derived geometry, unchanged guards. No execution or remedy admitted here.',
    'completion_sealer_identity':identity(me),'completion_launcher_identity':identity(me.parent()),
    'completion_sealer_and_launcher_alive_at_write_require_external_exact_death_verification':True}
out=F/'ordered14_decoder_actual_completion_v1.json'
with out.open('x',encoding='utf-8',newline='\n')as file:
    json.dump(completion,file,indent=2,ensure_ascii=False,allow_nan=False);file.write('\n');file.flush();os.fsync(file.fileno())
print(json.dumps({'completion':pin(out),'all_four_job_exact_dead':True,'sealer':completion['completion_sealer_identity'],
    'launcher':completion['completion_launcher_identity'],'first_failed_row':6,'written':820872,'declared':820896,'no_further_audio_executed':True}))
