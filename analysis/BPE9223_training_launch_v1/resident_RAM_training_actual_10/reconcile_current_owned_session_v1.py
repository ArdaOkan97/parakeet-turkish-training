import json,sys
from datetime import datetime,timezone
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01');N=Path(__file__).parent
boot=read_json(B/'invocations/owner_000009/boot.json');path=P/'work/gpu_training_owner.json';lease=read_json(path)
require(lease['owner']=='/root/memory_diagnosis' and lease['actual_owned_worker_pid']==boot['producer']['pid'] and lease['actual_GPU_running'] is True,'Only actual owned live native metadata')
prior=lease.get('current_owned_session');lease.setdefault('owned_session_pointer_history',[]).append({'at_utc':datetime.now(timezone.utc).isoformat(),'previous_session':prior,'new_current_session':62574,'same_live_native':boot['producer'],'only_metadata_pointer_reconciled':True,'admitted_source_or_environment_changed':False})
lease.update(current_owned_session=62574,previous_closed_owned_session=39651,actual_owned_launcher_pid=boot['producer']['ppid'],current_configuration=boot['configuration'],current_resident_source=boot['source'],latest_source_bound_original_interval_failure=record(N.parent/'resident_RAM_original_interval_post47727_failure_actual_v1/actual_failure_closure.json'))
write_json(path,lease);print(json.dumps({'current_owned_session':62574,'native':boot['producer'],'phase':lease['status']}))
