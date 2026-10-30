"""Bind minimal 250-update cadence dispatch; preserve every older receipt."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/parakeet_turkish'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=D/'public_turkish_v3_launch_preflight_v2.json';value=json.loads(old.read_text(encoding='utf-8'))
value.update(kind='public_turkish_v3_future_training_CPU_readiness_v3',prior_readiness_path=str(old.resolve()),prior_readiness_sha256=sha(old),
    production_memory_launcher_path=str((D/'launch_with_memory_guard_v7.py').resolve()),production_memory_launcher_sha256=sha(D/'launch_with_memory_guard_v7.py'),
    production_trainer_path=str((D/'train_lora_batched_data_v4.py').resolve()),production_trainer_sha256=sha(D/'train_lora_batched_data_v4.py'),
    production_cache_launcher_path=str((D/'launch_with_cache_reclaim_v4.py').resolve()),production_cache_launcher_sha256=sha(D/'launch_with_cache_reclaim_v4.py'),
    explicit_DEV_update_interval_supported=True,preferred_DEV_update_interval=250,
    DEV_policy='Explicit--eval-every-updates250 OR existingepochthreshold/finalstep; cadenceexact-resumeconfigbound; loss/Adam/samplemathunchanged')
for name in ('train_lora_batched_data_v4.py','launch_with_cache_reclaim_v4.py','launch_with_memory_guard_v7.py','fast_cadence_v4_cpu_verification.json'):
    value['artifacts_sha256'][str((D/name).resolve())]=sha(D/name)
value['CPU_guard_checks']['minimal_cadence_and_dispatch_source_guards']=19
target=D/'public_turkish_v3_launch_preflight_v3.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,proof_path=str(target.resolve()),proof_sha256=sha(target))))
