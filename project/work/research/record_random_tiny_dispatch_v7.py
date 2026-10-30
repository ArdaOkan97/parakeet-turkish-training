"""Compact live/closed CPU receipts; unrelated project fields are preserved."""
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
AREA=STORE/'analysis/CPU_validation_v7'


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.writing')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,path)


def main():
    state=json.loads((AREA/'random_tiny_language_CPU_v7_status.json').read_text())
    report=json.loads((AREA/'tiny_LID_report_v7.json').read_text())
    process=psutil.Process(state['pid']) if psutil.pid_exists(state['pid']) else None
    if process:assert any('run_random_tiny_language_CPU_v7.py' in v for v in process.cmdline())
    source=PROJECT/'work/research/run_random_tiny_language_CPU_v7.py'
    closed_path=AREA/'actual_tiny_LID_CLOSED0_v7.json'
    closed=json.loads(closed_path.read_text()) if closed_path.exists() else None
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    receipt={'kind':'owned_random_tiny_language_CPU_v7','updated_at_utc':now,'owner':'/root/cpu_data_preparation',
             'session':67978,'PID':state['pid'],'process_alive':process is not None,
             'source':str(source),'source_SHA':hashlib.sha256(source.read_bytes()).hexdigest(),
             'status_path':str(AREA/'random_tiny_language_CPU_v7_status.json'),
             'report_path':str(AREA/'tiny_LID_report_v7.json'),'ledger_path':str(AREA/'tiny_LID_per_clip_ledger_v7.jsonl'),
             'random64_manifest':str(AREA/'random64_tiny_language_manifest_v7.jsonl'),
             'random24_manifest':str(AREA/'random24_full_large_v3_manifest_v7.jsonl'),
             'random64_completed':report['random64_completed'],
             'random64_screen_counts':report['random64_screen_counts'],
             'total_with_worst_supplement_completed':report['total_with_worst_supplement_completed'],
             'actual_CLOSED0':closed,'CPU_only':True,'tiny_transcription':False,'accepted_TRAIN_hours':0,
             'large_CPP_partial_bytes':(STORE/'models/whisper_cpp_full_large_v3_q5_0/ggml-large-v3-q5_0.bin.partial').stat().st_size,
             'V4_PID':1912,'V4_state':'suspended_until_ROOT_entire_QC_batch_release'}
    write(AREA/'owned_CPU_dispatch_v7.json',receipt)
    path=PROJECT/'outputs/parakeet_turkish/current_CPU_data_preparation_v1.json'
    value=json.loads(path.read_text(encoding='utf-8-sig'))
    value['CPU_validation_v7']=receipt
    if 'CPU_validation_v6' in value:
        value['CPU_validation_v6']['operating_state']='owned_waiting_setup_retired_by_explicit_successor_direction'
        value['CPU_validation_v6']['retirement_receipt']=str(STORE/'analysis/CPU_validation_v6/owned_waiting_setup_retirement_v7.json')
    write(path,value)
    print(json.dumps({k:receipt[k] for k in ('session','PID','process_alive','random64_completed','random64_screen_counts','total_with_worst_supplement_completed','large_CPP_partial_bytes')}))


if __name__=='__main__':main()
