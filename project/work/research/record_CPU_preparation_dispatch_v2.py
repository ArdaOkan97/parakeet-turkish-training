"""Small reviewable dispatch receipt; preserve unrelated project fields."""
import json
from cpu_preparation_guard_v2 import AREA,PROJECT,now,sha256,write_json

def main():
    bindings={
        'compressed_additional_stage':{'session_id':86358,'status_file':'additional_stage_v2_status.json','source':'stage_yodas_additional_cpu_v2.py'},
        'independent_QC':{'session_id':65753,'status_file':'independent_QC_v2_status.json','source':'validate_alania_independent_cpu_v2.py'},
        'source_heldout_exclusions':{'session_id':27096,'status_file':'source_heldout_exclusions_v2_status.json','source':'exclude_alania_sources_cpu_v2.py'},
        'pilot_audio_exclusion':{'session_id':85589,'status_file':'pilot_audio_exclusion_v2_status.json','source':'check_alania_pilot_audio_exclusion_cpu_v2.py'},
        'full_accepted_pool_exclusion':{'session_id':80400,'status_file':'accepted_pool_exclusion_v2_status.json','source':'check_alania_accepted_pool_cpu_v2.py'}}
    for value in bindings.values():
        path=AREA/value['status_file'];status=json.loads(path.read_text(encoding='utf-8'))
        value.update(owner='/root/cpu_data_preparation',pid=status['pid'],started_at_utc=status['started_at_utc'],
            observed_phase=status['phase'],status_path=str(path),source_sha256=sha256(PROJECT/'work/research'/value['source']))
    receipt={'kind':'owned_parallel_CPU_preparation_dispatch_v2','updated_at_utc':now(),'jobs':bindings,
        'E_only_outputs':True,'GPU_used':False,'accepted_TRAIN_hours':0,'training_sources_and_environment_untouched':True,
        'do_not_duplicate_healthy_jobs':True,'independent_QC_report':str(AREA/'independent_QC_report_v2.json'),
        'independent_QC_ledger':str(AREA/'independent_QC_ledger_v2.jsonl'),
        'additional_stage_plan':str(AREA/'additional_stage_plan_v2.json'),
        'additional_stage_ledger':str(AREA/'additional_stage_ledger_v2.jsonl'),
        'source_heldout_exclusion_report':str(AREA/'source_heldout_exclusion_report_v2.json')}
    receipt['original72_independent_QC_milestone']=str(AREA/'original72_independent_QC_milestone_v2.json')
    receipt['pilot_audio_exclusion_actual_closure']={'session_id':85589,'exit_code':0,'tool_chunk':'74a80d',
        'report':str(AREA/'pilot_audio_exclusion_report_v2.json'),'report_sha256':sha256(AREA/'pilot_audio_exclusion_report_v2.json')}
    write_json(AREA/'owned_CPU_dispatch_v2.json',receipt)
    path=PROJECT/'outputs/parakeet_turkish/current_CPU_data_preparation_v1.json'
    old=path.read_bytes();state=json.loads(old.decode('utf-8-sig'));state['CPU_validation_v2']=receipt
    if path.read_bytes()!=old:raise RuntimeError('Concurrent project receipt update; retry merge')
    path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'dispatch':str(AREA/'owned_CPU_dispatch_v2.json'),'project_field':'CPU_validation_v2','unrelated_fields_preserved':True}))

if __name__=='__main__':main()
