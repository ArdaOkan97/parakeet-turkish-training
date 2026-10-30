import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];OUTPUT=ROOT/'outputs/parakeet_turkish'
def record(path):
    with Path(path).open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':str(Path(path).resolve()),'sha256':digest}
helper=OUTPUT/'resume_domain30_same_science_v1.py';outer=OUTPUT/'launch_domain30_same_science_resume_v1.py'
proof=OUTPUT/'domain_same_science_exact_resume_tiny_verification_v1.json'
test=json.loads(proof.read_text(encoding='utf-8'))
value={'schema':1,'kind':'domain_same_science_exact_resume_handoff_v1','status':'inert_ready_for_closed_domain_window',
    'passed_focused_CPU_checks':test['passed'],'focused_check_count':test['check_count'],
    'helper':record(helper),'fresh_preloader':record(outer),'focused_fixture_report':record(proof),
    'repeatable_fixture':record(ROOT/'work/research/domain_same_science_tiny_check_v1.py'),
    'frozen_first_launch_controller':record(OUTPUT/'run_hf_domain30_arm_v1.py'),
    'frozen_runtime_receipt':record(OUTPUT/'domain30_runtime_chain_pins_v1.json'),
    'peer_source_review':{'retention_design':'No remaining concrete runtime-routing blocker; manual stop requires ROOT decision',
        'confidence_design':'No remaining concrete scientific identity/reset blocker'},
    'scope':{'same_run_id':True,'source_worker_or_first_controller_changed':False,
        'source_runtime_chain_changed':False,'scientific_config_identity_changed':False,
        'actual_domain_worker_started':False,'actual_checkpoint_or_model_read':False,
        'full_training_state_tensor_audit_performed':False,'ML_audio_imports_or_GPU_use':False},
    'exact_state_contract':{'resume_input':'Opaque byte-identical archived full latest checkpoint; never rewritten/deserialized by parent',
        'BEST':'Closed BEST snapshot archived and SAME work BEST retained for frozen worker validation',
        'worker_restores':['Adam','Turkish domain and independent replay samplers','Python/Torch/CUDA RNG',
            'full inherited history','qualifying BEST/selected step','plateau and retention-failure counters',
            'next domain-epoch DEV threshold','cumulative audio/training/evaluation counters'],
        'parent_checks':['Same source/config/recipe/tokenizer/layout/pool identity remains the worker exact-resume predicate',
            'Original domain native-ready and initial qualifying source-closure path/SHA remain unchanged',
            'New actual process/preload evidence is separate from science identity and never replaces old ready',
            'Complete prior runtime/selected durable frontier before closed-state archive',
            'Final history prefix equals inherited history; new cache-hook calls equal final minus inherited steps',
            'Cumulative3/10 FLEURS quota and minimum-domain clock; full coverage required at13009'],
        'approved_worker_argv_changes':['--resume-weights removed','--resume-exact immutable full latest archive',
            '--stop-file NEW path','--max-wall-seconds NEW external window budget'],
        'original_horizon':'epochs1 / steps13009 unchanged across windows',
        'recipe':'B8x2/rank8alpha16/original8199/native3/KD10/LRs/replay50/explicit same replay seed unchanged',
        'memory':'Cap8GB/endcleanup4GB/native inactive750MB, existing system/actual-self/progress guards unchanged',
        'resource_stop':'Requires explicit --resource-review-cleared after ROOT review; no resets or guard waiver',
        'quality_stop':'Refuse same-state launch after retained consecutive-retention stop, plateau, target or configured horizon'},
    'future_invocation':{'python':str(ROOT/'work/training-env/Scripts/python.exe'),'script':str(helper),
        'flags':['--source-loop','<closed first-domain or previous exact-window evidence directory>',
            '--run-id','<same original domain run ID>','--output-dir','<unused NEW continuation evidence directory>',
            '--source-closed-confirmed','--window-hours','3'],
        'CPU_preflight':'Add --validate-only --preflight-output <NEW workspace JSON>; validates and byte-archives closed state, no GPU',
        'read_only_preview':'Add --prepare-only; requires an already-published domain config/ready but does not archive checkpoint bytes',
        'actual_launch':'ROOT alone removes --validate-only and adds --gpu-window-authorized after predecessor owned exit and exclusive GPU ownership',
        'resource_review_flag':'Add --resource-review-cleared only when actual preceding resource review has been resolved'},
    'repeatable_fixture_cli':[str(ROOT/'work/training-env/Scripts/python.exe'),str(ROOT/'work/research/domain_same_science_tiny_check_v1.py'),
        '--output','<NEW report JSON>'],
    'limitations':['Actual first domain arm has not been launched; actual closed-source preflight and fresh native preload remain future',
        'The fixture used opaque synthetic checkpoint bytes and mocked native preload/dispatch, not model-state tensor validation',
        'ROOT confirms owned process exit/exclusive GPU; the helper does not claim an independent OS exit proof',
        'Budget window completion leaves the quality goal incomplete; no TEST, export, merge or data change'],
    'created_at_utc':datetime.now(timezone.utc).isoformat()}
target=OUTPUT/'domain_same_science_resume_handoff_v1.json'
with target.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,ensure_ascii=False)
print(json.dumps({'path':str(target),'sha256':record(target)['sha256'],'checks':test['check_count']}))
