from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/parakeet_turkish'
def rec(path):
    with path.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':digest}
identity=ROOT/'work/trackio_continuation_mirror_v10_01_identity.json'
value=json.loads(identity.read_text(encoding='utf-8'))
assert value['pid']==39732 and value['status']=='stopped' and value['trackio_finish_returned'] is True
index=ROOT/'work/trackio_continuation_mirror_state_v10.json'
state=json.loads(index.read_text(encoding='utf-8'))
proof=OUT/'Trackio_V11_ROOT_uniform_verification_v1.json'
verified=json.loads(proof.read_text(encoding='utf-8'))
assert verified['passed'] and verified['assertions']==53 and verified['rejections']==10
target=OUT/'track_continuation_v11.py'
assert rec(target)['sha256']==verified['source_sha256']=='6a58eb0e8ccf25dae1263afcfc356f1e361d832d4d252698593edcb83d850b1e'
receipt={'schema':1,'kind':'ROOT_actual_closed_Trackio_V10_for_V11_handoff_v1','status':'actual_CLOSED',
    'actual_owned_session':47566,'actual_exit_code':0,'actual_exit_tool_chunk':'45339e',
    'identity':rec(identity),'source_state':rec(index),'source_runs_count':len(state['runs']),
    'source_script':rec(OUT/'track_continuation_v10.py'),'target_script':rec(target),
    'root_tiny_proof':rec(proof),'same_project':'parakeet-turkish',
    'no_Trackio_DB_or_historical_archive_deletion':True,
    'preserve_inherited_V9_and_own_V10_shards':True,
    'created_at_utc':datetime.now(timezone.utc).isoformat()}
dest=OUT/'Trackio_V10_actual_CLOSED_ROOT_handoff_V11_v1.json'
assert not dest.exists()
dest.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'receipt':rec(dest),'source_runs_count':len(state['runs'])}))
