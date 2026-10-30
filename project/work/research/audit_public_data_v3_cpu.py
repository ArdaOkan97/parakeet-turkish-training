"""Bounded proof-chain audit: no repeated full WAV scan, Torch or CUDA."""
from pathlib import Path
import sys,json,os,time
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(D))
import public_turkish_data_policy_v3 as policy
from resource_watchdog_v2 import sample_memory
target=D/'tr_public_v3_input_verification.json'
if target.exists():raise FileExistsError(target)
start=time.perf_counter();before=sample_memory(os.getpid(),'CPU_v3_proof_chain_validation')
audit=policy.audit_public_training_inputs(ROOT/'work/data/tr_public_v3_train.jsonl',ROOT/'work/data/replay_train.jsonl',D/'tr_public_v3_preparation.json',verify_actual_audio=False)
after=sample_memory(os.getpid(),'CPU_v3_proof_chain_validation_complete')
value=dict(schema=1,kind='public_data_v3_CPU_input_proof_chain_verification',passed=True,CPU_only=True,pid=os.getpid(),
    completed_at_utc=datetime.now(timezone.utc).isoformat(),audit=audit,policy_sha256=policy.sha256(policy.__file__),
    trainer_sha256=policy.sha256(D/'train_lora_batched_data_v3.py'),
    no_Torch_imported='torch' not in sys.modules,no_GPU_execution=True,
    audio_scope='71,684 actual WAVs previously fully audited plus3,199 newly recovered actual WAVs audited by pinnedv3prep; no duplicated fullWAVscan here; trainer defaults a freshfullactualWAVaudit atlaunch',
    owned_process_memory_observations=[before,after],elapsed_seconds=time.perf_counter()-start)
target.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,clips=audit['identity']['accepted_clips'],hours=audit['identity']['accepted_actual_PCM_hours'],elapsed_seconds=value['elapsed_seconds'])))
