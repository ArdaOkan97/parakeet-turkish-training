"""NEW end-of-completed-update3GB cleanup sibling; no ML/state/GPU action."""
from pathlib import Path
import ast
import hashlib
import re

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'outputs/parakeet_turkish'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def names(text):
    for old,new in (
        ('tsc_bpe_row_schedule_transition_v1','tsc_bpe_row_schedule_memory_v2'),
        ('train_lora_batched_tsc_bpe_row_schedule_v1','train_lora_batched_tsc_bpe_row_schedule_memory_v2'),
        ('run_TSC_bpe_row_schedule_v1','run_TSC_bpe_row_schedule_memory_v2')):text=text.replace(old,new)
    return text
def new(name,text):
    p=HERE/name
    assert not p.exists(),name
    ast.parse(text);p.write_text(text,encoding='utf-8');return sha(p)

helper=names((HERE/'tsc_bpe_row_schedule_transition_v1.py').read_text(encoding='utf-8'))
helper_sha=new('tsc_bpe_row_schedule_memory_v2.py',helper)
worker=names((HERE/'train_lora_batched_tsc_bpe_row_schedule_v1.py').read_text(encoding='utf-8'))
worker=worker.replace('and evidence.get("source_row_lr") == 1e-4,',
 'and evidence.get("source_row_lr") == 1e-4 and evidence.get("declared_allocator_policy") == {"cap_gb": 8, "end_cache_gb": 3, "native_inactive_threshold_bytes": 750000000},')
worker=worker.replace('"native_backend_provenance_sha256": file_sha(args.native_backend_provenance),',
 '"native_backend_provenance_sha256": file_sha(args.native_backend_provenance), "completed_update_cache_cleanup_threshold_bytes": 3_000_000_000, "resource_policy_change": "end-of-completed-update cache reclamation4GB->3GB; math and schedule unchanged",')
worker_sha=new('train_lora_batched_tsc_bpe_row_schedule_memory_v2.py',worker)
old_tree=ast.parse((HERE/'train_lora_batched_tsc_bpe_row_schedule_v1.py').read_text(encoding='utf-8'))
tree=ast.parse(worker)
run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
sync=[n.lineno for n in ast.walk(run) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.cuda.synchronize']
assert sync==[471],sync
controller=names((HERE/'run_TSC_bpe_row_schedule_v1.py').read_text(encoding='utf-8'))
controller=re.sub(r"WORKER_SHA = '[0-9a-f]+'", "WORKER_SHA = '"+worker_sha+"'",controller)
controller=re.sub(r"TRANSITION_SHA = '[0-9a-f]+'", "TRANSITION_SHA = '"+helper_sha+"'",controller)
controller=controller.replace('4_000_000_000','3_000_000_000').replace("'end_cache_gb': 4", "'end_cache_gb': 3")
controller=controller.replace("'kind': CONTROLLER_KIND, 'phase': PHASE, 'run_id': args.run_id,", "'kind': CONTROLLER_KIND, 'phase': PHASE, 'run_id': args.run_id,\n        'resource_policy_change': 'end-of-completed-update cache reclamation4GB->3GB; math/schedule/cap8/native750MB unchanged',\n        'declared_allocator_policy': {'cap_gb':8,'end_cache_gb':3,'native_inactive_threshold_bytes':750000000},")
controller=controller.replace("'cleanup_threshold_bytes': 3_000_000_000, 'native_max_joint_bytes':", "'cleanup_threshold_bytes': 3_000_000_000, 'resource_policy_change': plan['resource_policy_change'], 'native_max_joint_bytes':")
controller=controller.replace("'Actual cap8/end4/all-phase telemetry completion required'", "'Actual cap8/end3/all-phase telemetry completion required'")
controller=controller.replace("'native_max_joint_bytes': NATIVE_MAX_BYTES, 'allocator_cap_bytes': 8_000_000_000,", "'native_max_joint_bytes': NATIVE_MAX_BYTES, 'allocator_cap_bytes': 8_000_000_000, 'completed_update_cache_cleanup_threshold_bytes': 3_000_000_000,")
controller_sha=new('run_TSC_bpe_row_schedule_memory_v2.py',controller)

fixture=(ROOT/'work/research/check_TSC_bpe_row_schedule_controller_v1.py').read_text(encoding='utf-8')
fixture=names(fixture)
fixture=fixture.replace("'declared_acquisition': {'explicit_fixture_admission_mock': True}","'resource_policy_change':'explicit_resource_policy_fixture', 'declared_acquisition': {'explicit_fixture_admission_mock': True}")
fixture=fixture.replace("passed('actual_worker_source_optimizer_callback_frame_admitted')", "passed('actual_worker_source_optimizer_callback_frame_admitted')\n    actual_caller=m.make_caller(SimpleNamespace(),worker_path,SimpleNamespace())\n    require(actual_caller.threshold_bytes==3_000_000_000,'Actual caller must use3GB policy')\n    passed('actual_scoped_caller_constructor_uses3GB_completed_update_threshold')")
fixture=fixture.replace("'inactive_TSC_BPE_scheduled_row_dispatch_CPU_verification_v1'", "'inactive_TSC_BPE_scheduled_row_memory_dispatch_CPU_verification_v2'")
fixture=fixture.replace("'actual_controller_dispatch_function_executed': True,", "'actual_controller_dispatch_function_executed': True, 'only_memory_policy_changed':True,")
p=ROOT/'work/research/check_TSC_bpe_row_schedule_memory_controller_v2.py'
assert not p.exists();ast.parse(fixture);p.write_text(fixture,encoding='utf-8')
print({'helper':helper_sha,'worker':worker_sha,'sync':sync[0],'controller':controller_sha})
