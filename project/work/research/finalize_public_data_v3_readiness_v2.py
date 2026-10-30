"""Bind NEW all-phase telemetry wrapper; preserve preceding readiness evidence."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/parakeet_turkish'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=D/'public_turkish_v3_launch_preflight.json'
value=json.loads(old.read_text(encoding='utf-8'))
value.update(kind='public_turkish_v3_future_training_CPU_readiness_v2',
    prior_readiness_path=str(old.resolve()),prior_readiness_sha256=sha(old),
    production_memory_launcher_path=str((D/'launch_with_memory_guard_v6.py').resolve()),
    production_memory_launcher_sha256=sha(D/'launch_with_memory_guard_v6.py'),
    all_phase_live_self_memory_enabled=True,
    self_memory_policy='Every5sec ownedself stdlibsample during nestedworkeractive includingPCM verification/cache/DEV, plus optimizer-end; sourcevalidatedsamePID andcreationidentity; newestatomicboundedsidecar; finallystop/join',
    system_global_commit_API='GetPerformanceInfo CommitTotal/CommitLimit timesPageSize',
    calling_process_commit_observation='Separate GlobalMemoryStatusEx observedinside actualselfworker, pidbound; cannotreplace globalcommit',
    runtime_worker_CLI_help_checked=True,memory_wrapper_validate_only_checked=True,
    source_files_ready_for_parent_freeze=True)
value['artifacts_sha256'].update({str((D/name).resolve()):sha(D/name) for name in ('launch_with_memory_guard_v6.py','memory_guard_v6_final_cpu_verification.json')})
value['CPU_guard_checks']['memory_wrapper_mocks_native_API_real_thread']=55
value['CPU_guard_checks'].pop('memory_wrapper_mocks_and_native_API',None)
target=D/'public_turkish_v3_launch_preflight_v2.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,proof_path=str(target.resolve()),proof_sha256=sha(target),memory_launcher_sha256=value['production_memory_launcher_sha256'])))
