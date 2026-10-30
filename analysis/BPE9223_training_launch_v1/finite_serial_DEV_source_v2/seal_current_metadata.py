import json,sys
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from bpe_broad_audio_v1 import record
from fastmss_BPE9223_cpu_common_v4 import environment,resource_guard
environment();resource_guard();F=Path(__file__).parent;N=F.parent
cases=[(N/'CPU_preparation_v5','b542a2',21752,N/'CPU_preparation_v5/preflight_report.json',N/'CPU_preparation_v5/preflight_guard_01/closure.json',N/'CPU_preparation_v5/build_and_preflight.py'),
    (N/'finite_serial350DEV_configuration_02','5347ca',4720,N/'finite_serial350DEV_configuration_02/source_resource_review_handoff.json',F/'metadata_proposal_guard_02/closure.json',P/'work/research/prepare_ASR18_finite_BPE9223_serial_DEV_configuration_v1.py')]
for folder,chunk,pid,report,guard,source in cases:
    g=json.loads(guard.read_text());assert g['pid']==pid and g['actual_exit_code']==0 and g['monitor_clean'] and g['monitor_joined']
    value={'kind':'ASR18_source_bound_actual_tool_exit_v1','owner':'/root/cpu_data_preparation','actual_exit_code':0,
        'actual_exec_tool_exit_code':0,'actual_tool_chunk_id':chunk,'native_PID':pid,'published_marker':record(report),
        'source':record(source),'guard_closure':record(guard),'actual_audio_model_GPU_or_payload_reads':0}
    if pid==4720:value['additional_512MB_guard_closure']=record(F/'metadata_prepare_review_02/metadata512_closure.json')
    target=folder/'actual_owned_metadata_CLOSED0.json';assert not target.exists();target.write_text(json.dumps(value,indent=2));print(json.dumps(record(target)))
raw=(F/'run_metadata512.py').read_text().replace("F=Path(__file__).parent;script=", "F=Path(__file__).parent/'metadata_configuration_02';assert not F.exists();F.mkdir();script=")
target=F/'run_metadata512_configuration02.py';assert not target.exists();target.write_text(raw)
