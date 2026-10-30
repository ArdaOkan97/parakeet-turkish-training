"""Full bounded-CPU v2 input verification with self-process memory observations."""
from pathlib import Path
import sys
import json
import os
import threading
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'outputs/parakeet_turkish'))
import public_turkish_data_policy_v2 as policy
from resource_watchdog_v1 import sample_memory
from atomic_io_retry_v1 import write_json

target=ROOT/'outputs/parakeet_turkish/tr_public_v2_input_verification.json'
if target.exists():raise FileExistsError(target)
samples=[];stop=threading.Event()
def sample():
    try:samples.append(sample_memory(os.getpid(),'CPU_full_public_data_input_verification'))
    except Exception as error:samples.append({'telemetry_failed':type(error).__name__})
def monitor():
    while not stop.wait(5):
        if len(samples)<512:sample()
sample()
thread=threading.Thread(target=monitor,daemon=True);thread.start()
report={'schema':1,'kind':'public_data_v2_full_CPU_input_verification','passed':False,'CPU_only':True,
    'pid':os.getpid(),'started_at_utc':datetime.now(timezone.utc).isoformat(),
    'policy_sha256':policy.sha256(policy.__file__),'trainer_sha256':policy.sha256(ROOT/'outputs/parakeet_turkish/train_lora_batched_data_v2.py')}
print(json.dumps({'CPU_only':True,'owned_pid':os.getpid(),'policy_sha256':report['policy_sha256']}),flush=True)
try:
    result=policy.audit_public_training_inputs(ROOT/'work/data/tr_public_v2_train.jsonl',ROOT/'work/data/replay_train.jsonl',
        ROOT/'outputs/parakeet_turkish/tr_public_v2_preparation.json')
    report.update(passed=True,status='CPU_inputs_verified_future_GPU_dispatch_pending',audit=result,
        math_source_comparison='Training operations and optimizer-end line unchanged; fixture proof remains separately bound',
        no_optimizer_exposure_claimed=True,no_heldout_predictions_read=True)
except BaseException as error:
    report.update(status='failed',failure_type=type(error).__name__,failure_message=str(error));raise
finally:
    stop.set();thread.join(timeout=1);sample()
    report.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),owned_process_memory_observations=samples,
        policy_sha256_after=policy.sha256(policy.__file__),
        observed_peak_private_bytes=max((s.get('worker_private_memory',{}).get('private_bytes',0) for s in samples),default=0),
        peak_private_measurement_limit='Five-second CPU process samples; not a continuous or retrospective preparation peak')
    write_json(target,report)
    print(json.dumps({'passed':report['passed'],'status':report['status'],'report':str(target)}),flush=True)
