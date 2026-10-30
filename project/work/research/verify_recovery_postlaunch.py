"""Read-only runtime/provenance verification; writes only a new sidecar report."""
from __future__ import annotations
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUTPUT))
from atomic_io_retry_v1 import read_bytes_shared, read_json, write_json


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    checks=[]
    def require(value,label):
        if not value:raise RuntimeError(label)
        checks.append(label)
    loop=OUTPUT/'runs/adaptive_20261010_recovery_01'
    state=read_json(loop/'loop_status.json',{})
    original=read_json(loop/'original_loop_status.json',{})
    report={'schema':1,'passed':False,'status':'failed','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'CPU file hashes and runtime metadata only; no GPU/model inference or heldout prediction parsing',
            'loop_id':state.get('loop_id'),'gpu_used':False,'script_sha256':sha(__file__)}
    try:
        require(state['loop_id']=='adaptive_20261010_recovery_01','Exact recovery loop identity')
        require(state['deadline_at_utc']==original['deadline_at_utc']=='2026-10-10T23:41:22.889692+00:00','Original absolute deadline retained')
        require(state['awake_request']=={'supported':True,'enabled':True,'flags':2147483649},'Windows keep-awake request recorded enabled')
        origin_dir=OUTPUT/'runs/adaptive_20261010_01'
        require(sha(origin_dir/'loop_status.json')==state['recovery']['original_status_snapshot_sha256']==sha(loop/'original_loop_status.json'),'Original failed loop bytes preserved')
        require(sha(origin_dir/'frozen_final_selection.json')==state['recovery']['original_frozen_selection_snapshot_sha256']==sha(loop/'original_frozen_final_selection.json'),'Original frozen selection bytes preserved')
        old_config=read_json(OUTPUT/'runs/adaptive_20261010_01_screen_B/run_config.json',{})
        protected_codes=dict(old_config['code_sha256'])
        protected_codes.update({'run_experiment_loop.py':original['controller_sha256'],
            'run_experiment_loop_continued.py':original['continuation']['controller_sha256'],
            'benchmark_strong_tr_test.py':original['strong_test_script_sha256'],
            'benchmark_strong_tr.py':'46bf78a57112d1adc7c11ec019be5c99356ea6d30f68b61bb49abc3cc6bca475',
            'verify_adaptive_results.py':'68558524b47e079b294399e0e54ba235dff81e047cba60ef954b2b637d089443'})
        for name,digest in protected_codes.items():require(sha(OUTPUT/name)==digest,'Frozen original source: '+name)
        original_quality=read_json(OUTPUT/'sleep_recovery_quality_verification.json',{})
        for path,digest in original_quality['preserved_evidence_sha256'].items():require(sha(path)==digest,'Preserved original evidence: '+Path(path).name)
        for path,digest in old_config['input_sha256'].items():require(sha(path)==digest,'Original TRAIN/heldout manifest bytes: '+Path(path).name)
        require(sha(ROOT/'work/adaptive_20261010_01_screen_B_best.pt')==original['best_development_candidate']['provenance']['weights_sha256'],'Original B selected optimizer/trainable checkpoint bytes')
        for name,digest in state['recovery']['previous_output_artifacts'].items():require(sha(loop/'previous_output_artifacts'/name)==digest,'Archived output pointer/report: '+name)
        new_sources={'run_experiment_loop_recovery.py':state['recovery']['controller_sha256'],
            'launch_with_io_retry_v1.py':state['recovery']['io_retry_launcher_sha256'],
            'atomic_io_retry_v1.py':state['recovery']['io_retry_helper_sha256'],
            'benchmark_strong_tr_test_v2.py':'91012f94ba80bd1c0bbee7f06855d3120d5c982e461184b30968c9fc0a4c9580',
            'verify_recovery_results_v1.py':'bec333b9fc0606da39168d7db0bf4c4f38a557b412b41fd8969b516adf41a361'}
        for name,digest in new_sources.items():require(sha(OUTPUT/name)==digest,'Frozen new recovery source: '+name)
        run_id='adaptive_20261010_recovery_01_screen_C'
        trial=[r for r in state['trials'] if r['run_id']==run_id][0]
        require(state.get('active_run_id')==run_id and trial['status']=='running','Fresh C owned trial active')
        require(trial['selected_configuration']==state['selected_configuration'],'C uses existing B1/accum4 configuration')
        require(trial['parameters']=={'head_lr':2e-5,'lora_lr':6e-4,'new_token_lr_multiplier':15},'Original planned C learning parameters')
        require(trial['optimizer_resume'] is False,'Fresh C optimizer policy explicit')
        wrapper=read_json(trial['io_retry_provenance_path'],{})
        require(wrapper['status']=='running' and wrapper['training_semantics_changed'] is False,'Actual runtime wrapper initialized without training semantic changes')
        for key,name in [('worker_sha256','train_lora_batched.py'),('launcher_sha256','launch_with_io_retry_v1.py'),('helper_sha256','atomic_io_retry_v1.py')]:
            require(wrapper[key]==sha(OUTPUT/name),'Runtime I/O provenance source '+name)
        require(wrapper['arguments'][1:]==trial['command'][trial['command'].index('--')+1:],'Actual wrapper worker arguments equal recorded launch')
        require(wrapper['retry_winerrors']==[5,32,33] and wrapper['retry_seconds']==30 and wrapper['unique_same_directory_temporary'] is True,'Bounded private-temp I/O retries recorded')
        cfg_path=OUTPUT/'runs'/run_id/'run_config.json'
        config=read_json(cfg_path,None)
        if config is None:
            report.update(passed=True,status='passed_current_checks_config_pending',pending=['Trainer run_config not yet written while cache audit runs'])
        else:
            require(config['run_id']==run_id and config['base_revision']=='541d1f99c6b0c3cd0b11a95167540bb8edefd82b','Actual C config run/base identity')
            require(config['resume_exact'] is None and Path(config['resume_weights']).resolve()==(ROOT/'work/lora_20261010_01_best.pt').resolve(),'Actual C warm start same original pilot; no exact-resume claim')
            warm=config['warm_start']
            require(warm['mode']=='new_optimizer_warm_start' and warm['source_run_id']=='lora_20261010_01' and warm['source_step']==1200,'Actual C warm-start provenance declares original pilot step1200/new optimizer')
            require(Path(warm['path']).resolve()==Path(trial['initialization_weights']).resolve()
                    and warm['sha256']==trial['initialization_sha256']==sha(warm['path']),'Actual C warm-start checkpoint bytes match runtime and controller')
            require(config['batch_size']==1 and config['accumulation']==4 and config['epochs']==.25 and config['steps']==3204,'Actual C batch/effective4/quarter-step cap')
            for key,value in trial['parameters'].items():require(config[key]==value,'Actual C learning argument '+key)
            require(config['code_sha256']==old_config['code_sha256'],'Actual C original training sources unchanged')
            require(config['input_sha256']==old_config['input_sha256'],'Actual C TRAIN and heldout manifest source hashes identical')
            require(config['turkish_clips_used']==25628 and config['replay_clips_used']==1553,'Actual C accepted TRAIN/replay counts unchanged')
            for key in ('tr_pool_sha256','replay_pool_sha256','extended_tokenizer_sha256'):
                require(config['config_identity'][key]==old_config['config_identity'][key],'Actual C accepted pool/tokenizer identity '+key)
            require(config['seed']==old_config['seed']==20261010,'Actual C original deterministic seed preserved')
            require(config['selection_split']=='FLEURS validation only','Actual C selection is DEV only')
            report.update(passed=True,status='complete',runtime_config_sha256=sha(cfg_path),pending=[])
        report.update(check_count=len(checks),checks=checks,original_source_sha256=protected_codes,
                      recovery_source_sha256=new_sources,active_run_id=run_id,awake_request=state['awake_request'],
                      deadline_at_utc=state['deadline_at_utc'],stage_wall_quota_seconds=trial['stage_wall_quota_seconds'],
                      requested_turkish_epochs=trial['epochs_planned'],
                      exposure_limit='C requests 0.25 epoch but its 998.78-second wall quota may truncate exposure; completion is not claimed.',
                      wrapper_provenance_sha256_at_read=sha(trial['io_retry_provenance_path']))
    except Exception as error:
        report.update(failure_type=type(error).__name__,failure_message=str(error),checks=checks,check_count=len(checks))
    write_json(OUTPUT/'recovery_postlaunch_verification.json',report)
    print(json.dumps({k:report.get(k) for k in ('passed','status','check_count','failure_message','pending','active_run_id','stage_wall_quota_seconds')}))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
