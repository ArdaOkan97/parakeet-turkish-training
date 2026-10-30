"""Metadata-only correction of the unchanged historical genesis authority argument."""
import copy,hashlib,json,sys
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
N=Path(__file__).parent;D=N/'resident_RAM_single_packet_resource_source_v1'
sys.path[:0]=[str(S),str(R)]
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import read_json
from fastmss_owned_process_closure_v2 import dead
old=D/'final_execution_handoff_v1.json';result=read_json(old)
boot=read_json(Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01\invocations\owner_000010\boot.json'))
original=boot['latest_failure'];require(record(original['path'])==original,'Exact original source12 genesis authority')
v=read_json(original['path'])
require(v['kind']=='ASR18_actual_ACKed260_singlepass_control_namespace_failure_TOOL1_v1'
    and v['actual_tool_exit_code']==1 and v['new_training_GPU_children']==v['new_CPU_children']==0
    and v['durable_ACK_committed']is True and dead(v['producer']),
    'Unchanged original genesis authority gate actually passes')
result['ROOT_execution_review_required_fields']['latest_failure']=original
args=result['exact_future_wrapper_arguments'];args[args.index('--latest-failure')+1]=original['path']
result['historical_genesis_authority_argument']=original
result['new_actual133137_failure_is_separately_typed_abnormal_resume']=True
result['supersedes_metadata_argument_only_handoff']=record(old)
new=D/'final_execution_handoff_v2.json';require(not new.exists(),'Preserve original inactive handoff')
new.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(record(new)),flush=True)
