import json,sys
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from bpe_broad_audio_v1 import record
from fastmss_BPE9223_cpu_common_v4 import environment,resource_guard
environment();resource_guard()
F=Path(__file__).parent;O=F.parent/'finite_serial350DEV_configuration_01'
value={'kind':'ASR18_source_bound_actual_tool_exit_v1','owner':'/root/cpu_data_preparation','actual_exit_code':0,
    'actual_exec_tool_exit_code':0,'actual_tool_chunk_id':'02ddf1','native_PID':37352,
    'published_marker':record(O/'source_resource_review_handoff.json'),
    'source':record(P/'work/research/prepare_ASR18_finite_BPE9223_serial_DEV_configuration_v1.py'),
    'guard_closure':record(F/'metadata_proposal_guard_01/closure.json'),
    'additional_512MB_180s_guard_closure':record(F/'metadata512_closure.json'),
    'actual_peak_private_bytes':32161792,'actual_audio_model_GPU_or_payload_reads':0,
    'scope':'finite20 scalar/source/PRE/process metadata only; not hard-metadata success or DEV admission',
    'supersession_reason':'memory owner requested next proposal bind packet02 after corrected HARD metadata actual0'}
target=O/'actual_metadata_prepare_review_CLOSED0.json';assert not target.exists();target.write_text(json.dumps(value,indent=2))
print(json.dumps(record(target)))
raw=(F/'run_metadata512.py').read_text();raw=raw.replace("F=Path(__file__).parent;script=", "F=Path(__file__).parent/'metadata_prepare_review_02';assert not F.exists();F.mkdir();script=")
target=F/'run_metadata512_proposal02.py';assert not target.exists();target.write_text(raw)
