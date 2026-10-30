from pathlib import Path
import json, hashlib
ROOT=Path(__file__).resolve().parents[2]
D=ROOT/'outputs/parakeet_turkish'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
prior=D/'public_turkish_v3_launch_preflight_v3.json'
target=D/'public_turkish_v3_launch_preflight_v4.json'
if target.exists():raise FileExistsError(target)
report=json.loads(prior.read_text(encoding='utf-8'))
report['kind']='public_turkish_v3_future_training_CPU_readiness_v4'
report['prior_readiness_path']=str(prior)
report['prior_readiness_sha256']=sha(prior)
for role,name in [('io','launch_with_io_retry_v2.py'),('cache','launch_with_cache_reclaim_v5.py'),('memory','launch_with_memory_guard_v8.py')]:
    path=D/name
    report['production_'+role+'_launcher_path']=str(path)
    report['production_'+role+'_launcher_sha256']=sha(path)
    report['artifacts_sha256'][str(path)]=sha(path)
proof=D/'data_v4_actual_io_dispatch_cpu_verification.json'
data=json.loads(proof.read_text(encoding='utf-8'))
assert data['passed'] and data['CPU_only'] and data['check_count']==19
report['artifacts_sha256'][str(proof)]=sha(proof)
report['CPU_guard_checks']['actual_nested_IO_dispatch_and_original_serializer']=19
report['actual_nested_dispatch_proof_path']=str(proof)
report['actual_nested_dispatch_scope']='Actual memory8/cache5/IO2 chain, real trainer4 parser, unchanged original checkpoint serializer through atomic adapter; CPU tiny run substitute, no GPU training claim'
report['supersedes_prior_IO_dispatch_claim']='Prior cache4/memory7 delegated to IO1 which rejects future trainer4 before GPU; sources and failed-run evidence retained unchanged'
report['optional_follow_on']['TR_only_metadata_cache']='Investigate exact CPU encoder-frame derivation to avoid frozen TR encoding entirely; replay feature cache remains required; runtime launch readiness does not depend on this follow-on'
target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'path':str(target),'sha256':sha(target),'passed':True}))
