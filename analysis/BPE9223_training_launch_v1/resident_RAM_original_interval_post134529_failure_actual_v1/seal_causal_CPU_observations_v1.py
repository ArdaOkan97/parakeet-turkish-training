"""Actual external wait/death bindings for the finite causal observations."""
import json,hashlib
from pathlib import Path
import psutil
F=Path(__file__).parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def pin(p):
    raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def death(who):
    now=None
    try:now=psutil.Process(who['pid']).create_time()
    except psutil.NoSuchProcess:pass
    assert now is None or abs(now-who['create_time'])>=.001
    return {'original_identity':who,'observed_current_create_time':now,'original_identity_dead':True}
closure=read(F/'actual_failure_closure.json');probe=read(F/'probe_actual_01/report.json');member=read(F/'full_member_actual_01/report.json')
result={'kind':'ASR18_original_interval134530_actual_causal_CPU_closure_receipt_v1',
    'failure_observer':{'actual_TOOL_exit_code':0,'actual_TOOL_chunk_id':'6ea5eb','report':pin(F/'actual_failure_closure.json'),
        'actual_death':death({k:closure['observer'][k]for k in ('pid','create_time')})},
    'ordered_decoder_probe':{'actual_TOOL_exit_code':0,'actual_TOOL_chunk_id':'5a900e','report':pin(F/'probe_actual_01/report.json'),
        'completion':pin(F/'probe_actual_01/CPU_completion.json'),'actual_death':death(probe['producer'])},
    'complete_member_observation':{'actual_TOOL_exit_code':0,'actual_TOOL_chunk_id':'09aa70','report':pin(F/'full_member_actual_01/report.json'),
        'completion':pin(F/'full_member_actual_01/CPU_completion.json'),'actual_death':death(member['producer'])},
    'actual_failing_update':134530,'actual_update_index':9,'actual_row_index':3,
    'source_id':'alania:9105d11d7d12c9c166028f98935dc107_00000','physical_native_samples':600190,'declared_native_samples':600192,
    'physical_clean16k_samples':200064,'valid_mel_frames':1250,'physical_feature_frames':1251,'encoder_frames':157,'full_target_tokens':82,
    'actual_frontend_or_model_GPU_run_claimed':False,'production_remedy_admitted':False,'no_source_or_scientific_state_changes':True}
out=F/'actual_causal_CPU_closure_receipt.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(pin(out)),flush=True)
