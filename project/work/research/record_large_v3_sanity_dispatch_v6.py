"""Lightweight ownership/current-state receipt, preserving unrelated fields."""
import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

STORE=Path(r'E:\gpu_training_scratchpad')
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(STORE/'scratch/CPU_validation_v2_env/Lib/site-packages'))
import psutil
AREA=STORE/'analysis/CPU_validation_v6'


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.writing')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def main():
    session=int(sys.argv[1]) if len(sys.argv)>1 else 30710
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    state=json.loads((AREA/'full_large_v3_sanity_setup_v6_status.json').read_text())
    process=psutil.Process(state['pid'])
    assert any('prepare_large_v3_sanity_cpu_v6.py' in v for v in process.cmdline())
    source_names=['prepare_large_v3_sanity_cpu_v6.py','whisper_cpp_cpu_bindings_v6.py','run_large_v3_sanity_cpu_v6.py']
    sources=[]
    for name in source_names:
        path=PROJECT/'work/research'/name
        ast.parse(path.read_text(encoding='utf-8'))
        sources.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    pause=json.loads((AREA/'accepted_reference_worker_declared_pause_v6.json').read_text())
    reference=psutil.Process(pause['pid'])
    native_path=AREA/'full_large_v3_sanity_native_v6_status.json'
    native=json.loads(native_path.read_text()) if native_path.exists() else None
    report_path=AREA/'full_large_v3_sanity_report_v6.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else None
    value={'kind':'owned_bounded_full_large_v3_CPU_sanity_dispatch_v6','updated_at_utc':now,
           'setup_session':session,'setup_PID':process.pid,'setup_process_create_time':process.create_time(),
           'previous_setup_attempt':{'session':15276,'PID':7756,'actual_exit':1,
                 'reason':'Single long CDN stream ended after16726599bytes; durable partial preserved and download repaired to bounded4MiB Range requests'},
           'previous_resource_guard_attempt_closure':str(AREA/'actual_setup_attempt2_guard_closure_v6.json'),
           'setup_cmdline':process.cmdline(),'setup_state':state,
           'native_state':native,'actual_full_large_v3_recognitions':report['samples_completed'] if report else 0,
           'reference_pause':pause,'reference_current_process_status':reference.status(),
           'source_bindings':sources,'manifest_receipt':str(AREA/'worst_samples_selection_receipt_v6.json'),
           'fit_preflight':str(AREA/'full_large_v3_CPU_fit_preflight_v6.json'),
           'model_identity':'full original OpenAI Whisper-large-v3, official ggerganov whisper.cpp Q5_0 conversion',
           'no_turbo_distil_or_new_small_inference':True,'CPU_only':True,'accepted_TRAIN_hours':0}
    write(AREA/'owned_CPU_dispatch_v6.json',value)
    receipt_path=PROJECT/'outputs/parakeet_turkish/current_CPU_data_preparation_v1.json'
    receipt=json.loads(receipt_path.read_text(encoding='utf-8-sig'))
    receipt['CPU_validation_v6']={'updated_at_utc':now,'session':session,'setup_PID':process.pid,
         'status_path':str(AREA/'full_large_v3_sanity_setup_v6_status.json'),
         'ownership_receipt':str(AREA/'owned_CPU_dispatch_v6.json'),
         'manifest_receipt':value['manifest_receipt'],'fit_preflight':value['fit_preflight'],
         'actual_full_large_v3_recognitions':value['actual_full_large_v3_recognitions'],'accepted_TRAIN_hours':0,
         'reference_worker_PID':reference.pid,'reference_worker_state':reference.status()}
    receipt['CPU_validation_v5']={'completed_cached_text_audit':True,'original_unique_flags':106,
         'actual_flags_changed':0,'report':str(STORE/'analysis/CPU_validation_v5/cached_flagged_text_diagnostic_report_v5.json')}
    if 'CPU_validation_v4' in receipt:
        receipt['CPU_validation_v4']['operating_state']='paused_for_user_requested_large_v3_sanity' if reference.status()==psutil.STATUS_STOPPED else 'running'
        receipt['CPU_validation_v4']['pause_receipt']=str(AREA/'accepted_reference_worker_declared_pause_v6.json')
    write(receipt_path,receipt)
    print(json.dumps({'setup_session':session,'PID':process.pid,'reference_worker_status':reference.status(),
                      'downloaded_bytes':state.get('downloaded_bytes'),
                      'native_recognitions':value['actual_full_large_v3_recognitions'],
                      'sources_AST_verified':len(sources),'receipt':str(AREA/'owned_CPU_dispatch_v6.json')}))


if __name__=='__main__':main()
