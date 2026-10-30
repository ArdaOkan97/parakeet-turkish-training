"""Read-only SQLite verification of recovery registration and retained speed runs."""
from __future__ import annotations
import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
OUTPUT=ROOT/'outputs/parakeet_turkish'
RUN='adaptive_20261010_recovery_b8_01_b8_quality'
DB=OUTPUT/'trackio/parakeet-turkish.db'
def read(path,default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):
        return default
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
checks=[]
def check(label,value):
    if not value:
        raise AssertionError(label)
    checks.append(label)

speed_path=OUTPUT/'large_batch_speed_report.json'
speed=read(speed_path)
check('independently_audited_batch_speed_integrity',speed['integrity_passed'] is True)
report={'schema':1,'verified_at_utc':datetime.now(timezone.utc).isoformat(),'cpu_only':True,'gpu_used':False,
    'database_access':'SQLite URI mode=ro; PRAGMA query_only; one read-only transaction snapshot',
    'database_path':str(DB),'recovery_run_id':RUN,'dashboard_url':'http://127.0.0.1:7860/?project=parakeet-turkish',
    'mirror_script_sha256':sha(OUTPUT/'track_experiment.py'),'speed_report_sha256':sha(speed_path)}
with sqlite3.connect(DB.as_uri()+'?mode=ro',uri=True,timeout=5) as connection:
    connection.execute('PRAGMA query_only=ON')
    connection.execute('BEGIN')
    benchmarks=[]
    for family in ('typical','long'):
        name='batch_profile_effective16_'+family
        configs=connection.execute('SELECT run_id,config FROM configs WHERE run_name=?',(name,)).fetchall()
        check(name+'_registration_retained',bool(configs))
        stored=connection.execute('SELECT step,metrics FROM metrics WHERE run_name=? ORDER BY step,id',(name,)).fetchall()
        check(name+'_all5physical_batches_retained',[s for s,_ in stored]==[1,2,4,8,16])
        expected={r['batch_size']:r for r in speed['table'] if r['family']==family and r['timing_scope']=='end_to_end'}
        for step,value in stored:
            metric=json.loads(value)
            check(name+'_batch_'+str(step),metric['benchmark/physical_batch_size']==step)
            for field in ('audio_min_per_wall_min','peak_allocated_vram_gb','peak_reserved_vram_gb'):
                check(name+'_'+str(step)+'_'+field,metric['benchmark/'+field]==expected[step][field])
            check(name+'_'+str(step)+'_strict_failure_retained',metric['benchmark/adoption_eligible']==0 and metric['benchmark/same_shape_numerical_gate_passed']==0)
        benchmarks.append({'name':name,'registration_count':len(configs),'metric_records':len(stored),'physical_batch_steps':[s for s,_ in stored]})
    report['retained_benchmark_runs']=benchmarks
    configs=connection.execute('SELECT run_id,config,created_at FROM configs WHERE run_name=? ORDER BY id',(RUN,)).fetchall()
    stored=connection.execute('SELECT step,metrics,timestamp FROM metrics WHERE run_name=? ORDER BY id',(RUN,)).fetchall()
    history=read(OUTPUT/'runs'/RUN/'training_history.json',[])
    config=read(OUTPUT/'runs'/RUN/'run_config.json',{})
    report['recovery_registration']={'registered':bool(configs),'registration_count':len(configs),'metrics_records':len(stored),
        'source_config_published':bool(config),'source_history_rows':len(history)}
    if configs:
        stored_config=json.loads(configs[-1][1])
        extras=sorted(set(stored_config)-set(config))
        check('recovery_SQLite_only_known_Trackio_metadata',extras==['_Created','_Group','_Username'])
        check('recovery_SQLite_config_exact_source',{k:stored_config[k] for k in config}==config)
        check('recovery_config_run_identity',config['run_id']==RUN)
        merged={}
        for step,value,timestamp in stored:
            merged.setdefault(step,{}).update(json.loads(value))
        source_keys={'tdt_loss':'train/tdt_loss','replay_kl':'train/replay_kl','gradient_norm':'train/gradient_norm',
            'replay_duration_kl':'train/replay_duration_kl','replay_feature_distillation':'train/replay_feature_distillation',
            'peak_allocated_vram_gb':'resources/peak_torch_allocated_vram_gb',
            'current_allocated_vram_gb':'resources/current_allocated_vram_gb','current_reserved_vram_gb':'resources/current_reserved_vram_gb',
            'turkish_unique_clips_seen':'data/turkish_unique_clips_seen','turkish_epoch':'data/turkish_epoch',
            'turkish_clips_seen':'data/turkish_clips_seen','replay_clips_seen':'data/replay_clips_seen',
            'audio_seen_seconds':'data/audio_seen_seconds','turkish_audio_seen_seconds':'data/turkish_audio_seen_seconds',
            'replay_audio_seen_seconds':'data/replay_audio_seen_seconds','audio_minutes_per_wall_minute':'resources/audio_minutes_per_wall_minute',
            'train_wall_seconds':'resources/train_wall_seconds','evaluation_wall_seconds':'resources/evaluation_wall_seconds',
            'process_wall_seconds':'resources/process_wall_seconds','head_lr':'train/head_lr','lora_lr':'train/lora_lr',
            'new_token_update_multiplier':'train/new_token_update_multiplier'}
        verified_steps=[]
        for row in history:
            step=row['step']
            if step not in merged:
                continue
            values=merged[step]
            for key,target in source_keys.items():
                if key in row:
                    check('recovery_'+str(step)+'_'+key,values.get(target)==row[key])
            check('recovery_'+str(step)+'_elapsed',values.get('resources/elapsed_minutes')==row['elapsed_seconds']/60)
            for key in ('train/tdt_loss','train/replay_kl','train/gradient_norm'):
                check('recovery_'+str(step)+'_finite_'+key,math.isfinite(values[key]))
            if 'validation' in row:
                validation=row['validation']
                check('recovery_'+str(step)+'_dev_retention',values.get('retention/all_dev_gates_passed')==int(validation['retention_gate_passed']))
                for lang,delta in validation['retention_wer_deltas'].items():
                    check('recovery_'+str(step)+'_retention_'+lang,values.get('retention/wer_delta_'+lang+'_pp')==100*delta)
            verified_steps.append(step)
        for step,values in merged.items():
            summary=read(OUTPUT/f'results/{RUN}_step_{step}/summary.json',{})
            for key,value in summary.items():
                if key.startswith('validation_') and 'asr' in value:
                    target='dev/wer_'+key.removeprefix('validation_')+'_percent'
                    if target in values:
                        check('recovery_'+str(step)+'_'+target,values[target]==100*value['asr']['wer'])
        report['recovery_registration'].update(database_run_id=configs[-1][0],created_at=configs[-1][2],
            Trackio_added_config_metadata_keys=extras,
            verified_training_steps=verified_steps,last_verified_step=max(verified_steps,default=None),
            newest_metrics_timestamp=stored[-1][2] if stored else None)
        report['recovery_metrics_verified']=bool(verified_steps)
    else:
        report['recovery_metrics_verified']=False
    connection.rollback()

mirror_log=ROOT/'work/trackio_mirror_recovery.log'
text=mirror_log.read_text(encoding='utf-8',errors='replace') if mirror_log.exists() else ''
report['mirror_log']={'path':str(mirror_log),'contains_recovery_registration':('Mirroring '+RUN) in text,
    'contains_traceback':'Traceback' in text,'contains_initialization':'Trackio project initialized' in text}
report['loop_state']={k:v for k,v in read(OUTPUT/'runs/adaptive_20261010_recovery_b8_01/loop_status.json',{}).items()
    if k in ('status','active_run_id','deadline_at_utc','started_at_utc','updated_at_utc')}
report['passed']=True
report['verification_status']='complete' if report['recovery_metrics_verified'] else 'benchmarks_verified_waiting_for_recovery_history'
report['checks_passed']=len(checks);report['checks']=checks
report['limitations']=['Read-only persistent metrics verification; does not claim a browser-rendering or HTTP availability check.',
    'Unchanged mirror registers a new run only after a nonempty history is published; cache preparation/initial DEV can precede registration.',
    'Live snapshots can lag the newest training rows by the mirror polling interval; verified rows are compared exactly to source history.']
target=OUTPUT/'trackio_b8_readonly_verification.json'
target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'status':report['verification_status'],'checks_passed':len(checks),
    'recovery':report['recovery_registration'],'report_path':str(target)}),flush=True)
