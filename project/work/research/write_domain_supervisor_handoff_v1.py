import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];OUTPUT=ROOT/'outputs/parakeet_turkish'
def record(path):
    with Path(path).open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':str(Path(path).resolve()),'sha256':digest}
script=OUTPUT/'supervise_domain30_same_science_v1.py'
proof=OUTPUT/'domain_same_science_supervisor_tiny_lifecycle_verification_v1.json'
test=json.loads(proof.read_text(encoding='utf-8'))
report={'schema':1,'kind':'domain_same_science_supervisor_handoff_v1','status':'inert_CPU_ready_for_ROOT_arming_after_actual_first_worker_identity',
    'supervisor':record(script),'focused_lifecycle_fixture':record(ROOT/'work/research/domain_supervisor_tiny_lifecycle_v1.py'),
    'focused_lifecycle_proof':record(proof),'focused_checks':test['check_count'],'passed':test['passed'],
    'complementary_exact_helper_proof':record(OUTPUT/'domain_same_science_exact_resume_tiny_verification_v1.json'),
    'exact_resume_handoff':record(OUTPUT/'domain_same_science_resume_handoff_v1.json'),
    'original_first_controller':record(OUTPUT/'run_hf_domain30_arm_v1.py'),
    'exact_controller':record(OUTPUT/'resume_domain30_same_science_v1.py'),
    'fresh_native_preloader':record(OUTPUT/'launch_domain30_same_science_resume_v1.py'),
    'reviewed_HF_supervisor_pattern':record(OUTPUT/'supervise_hf_same_science_v2.py'),
    'frozen_domain_runtime_chain':record(OUTPUT/'domain30_runtime_chain_pins_v1.json'),
    'peer_source_reviews':{'retention_design':'No remaining concrete closure/exclusive-runtime blocker',
        'confidence_design':'No remaining concrete scientific identity or clock/closure blocker'},
    'automatic_scope':{'same_run_id':True,'scientific_identity_changed':False,'configured_updates':13009,'epochs':1,
        'epoch_semantics':'minimum-domain exposure, not uniform draw-equivalent epochs',
        'per_successor_window_hours_maximum':3,'supervisor_wall_hours_maximum':12,
        'qualifying_budget_stop_reasons':['soft_max_wall_seconds; stop_category budget; no resource/manual cause',
            'controller_stop_file AND cooperative_stop_reason window_budget_checkpoint_boundary AND budget category'],
        'automatic_state_reset':False,'automatic_resource_review_clearance':False,
        'all_retention_and_internal_stops_preserved':True,'resource_or_failure_retry':False,
        'unchanged_state':['Adam','Turkish/replay samplers','all RNG','BEST','ordered history','cumulative exposure/audio/wall counters',
            'retention/plateau counters','next DEV epoch threshold','original native-ready/source-closure input identity'],
        'runtime':'Existing cap8GB/end4GB/native750MB and system/actual-worker-private/progress guards',
        'no_TEST_export_or_goal_completion':True},
    'ownership_scope':{'initial_first_controller':'ROOT-declared actual launcher/worker PID metadata and pinned child-close→completion→atomic-terminal path',
        'initial_supervisor_independent_OS_exit_query':False,'cross_exec_OpenProcess_escalation':False,
        'successors':'Supervisor-owned Popen.poll/wait, actual exit0 required before any budget successor',
        'resource_exit1':'May be verified and recorded, but ends for ROOT review without another launch',
        'bounded_horizon':'No new launch after deadline; owned helper gets cooperative checkpoint request and bounded600s close grace',
        'initial_horizon_expiry':'No cross-job termination; initial controller retains its own finite deadline/guards and ROOT is told no successor was launched'},
    'CLI':{'python':str(ROOT/'work/training-env/Scripts/python.exe'),'script':str(script),
        'arguments':['--source-loop','<actual live FIRST domain controller directory>',
            '--run-id','<same original domain run ID>','--supervisor-dir','<unused NEW supervisor evidence directory>',
            '--launcher-pid','<actual first controller owned launcher PID>','--worker-pid','<actual native worker self PID>',
            '--successor-prefix','<unused safe successor folder prefix>','--window-hours','3','--max-hours','12',
            '--gpu-window-authorized'],
        'arming_requires':'First worker run_config + native-ready + actual identity published and controller status running/fresh <=30s; final selection is not required yet',
        'optional_metadata_only':'Use --bootstrap-only in a separate NEW evidence directory; it launches no GPU successor. Its directory cannot be reused for the real supervisor'},
    'fixture_repeat_cli':[str(ROOT/'work/training-env/Scripts/python.exe'),str(ROOT/'work/research/domain_supervisor_tiny_lifecycle_v1.py'),
        '--output','<NEW workspace proof JSON>'],
    'limitations':['Actual domain training/closure/automatic GPU handoff has not been executed',
        'Fixture was synthetic and mocked Popen, source-state selection and completion calls; actual source ownership/order and generated-command code were exercised',
        'The complementary18-check helper fixture exercised original-ready/closure/config/history-prefix/hook-delta semantics separately',
        'No actual model/checkpoint state, GPU, audio, dataset scan or existing-source mutation was performed'],
    'created_at_utc':datetime.now(timezone.utc).isoformat()}
target=OUTPUT/'domain_same_science_supervisor_handoff_v1.json'
with target.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,ensure_ascii=False)
print(json.dumps({'path':str(target),'sha256':record(target)['sha256'],'source_sha256':record(script)['sha256']}))
