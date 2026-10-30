from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, sys

root = Path(__file__).resolve().parents[1]
out = root / 'outputs/parakeet_turkish'
sys.path.insert(0, str(out))
from paired_windows_commit_telemetry_v1 import performance_info

def record(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}

identity_path = root / 'work/trackio_continuation_mirror_v9_01_identity.json'
identity = json.loads(identity_path.read_text(encoding='utf-8'))
assert identity['pid'] == 4180 and identity['status'] == 'stopped'
assert identity['stop_reason'] == 'cooperative_stop_file' and identity['trackio_finish_returned'] is True
assert record(out / 'track_continuation_v9.py')['sha256'] == '076ab5f58fe6cbcd2b061af6346d98fa58aaa500700b853a8d35b356fe44a674'
assert record(out / 'track_continuation_v10.py')['sha256'] == '935309ae110cb37b8c690a6389dc260bc3aded83eab053652712459d898b4f54'
state_path = root / 'work/trackio_continuation_mirror_state_v9.json'
state = json.loads(state_path.read_text(encoding='utf-8'))
target = out / 'Trackio_V9_actual_CLOSED_ROOT_handoff_V10_v1.json'
assert not target.exists()
receipt = {'schema':1,'kind':'ROOT_actual_closed_Trackio_V9_for_V10_handoff_v1',
    'status':'actual_CLOSED','actual_owned_session':12608,'actual_exit_code':0,'actual_exit_tool_chunk':'5b590c',
    'identity':record(identity_path),'source_state':record(state_path),'source_runs_count':len(state['runs']),
    'source_script':record(out/'track_continuation_v9.py'),'target_script':record(out/'track_continuation_v10.py'),
    'same_project':'parakeet-turkish','no_Trackio_DB_or_historical_archive_deletion':True,
    'original_V8_archive':record(root/'work/trackio_continuation_mirror_state_v8.json'),
    'root_latest26_tiny_proof':record(out/'Trackio_V10_latest_ROOT_tiny_verification_v1.json'),
    'after_source_closure_system_memory':performance_info(),'created_at_utc':datetime.now(timezone.utc).isoformat()}
target.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'handoff':record(target),'source_runs':len(state['runs']),'system_memory':receipt['after_source_closure_system_memory']}))
