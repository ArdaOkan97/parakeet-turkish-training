"""NEW startup source-admission lifetime cleanup; no ML/state/GPU execution."""
from pathlib import Path
import ast
import hashlib
import re

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'outputs/parakeet_turkish'
old=['tsc_bpe_row_schedule_memory_v2.py','train_lora_batched_tsc_bpe_row_schedule_memory_v2.py','run_TSC_bpe_row_schedule_memory_v2.py']
new=['tsc_bpe_row_schedule_memory_v3.py','train_lora_batched_tsc_bpe_row_schedule_memory_v3.py','run_TSC_bpe_row_schedule_memory_v3.py']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rename(text):
    for a,b in sorted(zip(old,new),key=lambda x:len(x[0]),reverse=True):text=text.replace(a,b).replace(a[:-3],b[:-3])
    return text
def write(name,text):
    p=HERE/name;assert not p.exists();ast.parse(text);p.write_text(text,encoding='utf-8');return sha(p)

h=rename((HERE/old[0]).read_text(encoding='utf-8'))
h+='''\n\ndef release_admission_caches():
    """After actual source300/initial saves only: drop redundant JSON admission templates.

    This clears this worker's caches, not files or another process. Authoritative
    current config/history/sampler/optimizer/RNG are caller-held and untouched.
    Cached contexts contain JSON provenance, never the deserialized source207.
    """
    from tsc_bpe_acquisition_transition_v1 import _context as original_context
    before = {"scheduled_admission_entries": _context.cache_info().currsize,
              "original_acquisition_admission_entries": original_context.cache_info().currsize}
    _context.cache_clear(); original_context.cache_clear()
    return {**before, "scheduled_admission_entries_after": _context.cache_info().currsize,
        "original_acquisition_admission_entries_after": original_context.cache_info().currsize,
        "checkpoint_tensors_in_admission_cache": False, "authoritative_runtime_mutated": False}
'''
helper_sha=write(new[0],h)
w=rename((HERE/old[1]).read_text(encoding='utf-8'))
w=w.replace('load_validate_phase1, initial_source_equivalence)', 'load_validate_phase1, initial_source_equivalence, release_admission_caches)')
point='    save(latest_path, start_step)\n    model.train(); model.encoder.eval()'
assert point in w
cleanup='''    save(latest_path, start_step)
    # Direct original/source tensors were already deleted above in predecessorv2.
    # expanded still holds the cloned inherited-runtime alias; both its audit
    # fields and context have now completed their last use and durable writes.
    import gc, os
    from launch_with_memory_guard_TSC_resume_v3 import safe_sample
    before_release = safe_sample(os.getpid(), "before_startup_admission_release")
    inherited_runtime_removed = expanded.pop("inherited_runtime", None) is not None
    del expanded, context
    cache_release = release_admission_caches()
    garbage_collected = gc.collect()
    after_release = safe_sample(os.getpid(), "after_startup_admission_release")
    write_json(run_dir / "source_admission_release.json", {"kind": "scheduled_row_startup_admission_release_v3",
        "passed": True, "run_id": args.run_id, "phase": PHASE, "before_first_optimizer_update": True,
        "initial_BEST_and_latest_committed": True, "actual_source300_sha256": file_sha(proof_path(args)),
        "source_step": start_step, "local_scheduled_updates": acquisition.optimizer_updates_completed,
        "row_Adam_clock": acquisition.source_acquisition_updates + acquisition.optimizer_updates_completed,
        "expanded_inherited_runtime_alias_removed": inherited_runtime_removed,
        "direct_original_source_deletion_already_present_in_predecessor": True,
        "authoritative_history_rows_retained": len(history), "cache_release": cache_release,
        "gc_collected": garbage_collected, "host_memory_before": before_release, "host_memory_after": after_release,
        "retention_relief_magnitude_or_recurrence_prevention_claimed": False})
    del before_release, after_release
    print("completed startup admission alias release before first optimizer update", flush=True)
    model.train(); model.encoder.eval()'''
w=w.replace(point,cleanup)
w=w.replace('"resource_policy_change": "end-of-completed-update cache reclamation4GB->3GB; math and schedule unchanged",',
    '"resource_policy_change": "end-cache3GB retained; redundant startup admission aliases/caches released after source300 and initial saves; math and schedule unchanged",')
worker_sha=write(new[1],w)
tree=ast.parse(w);run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
sync=[n.lineno for n in ast.walk(run) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.cuda.synchronize'];assert len(sync)==1
c=rename((HERE/old[2]).read_text(encoding='utf-8'))
c=re.sub(r"WORKER_SHA = '[0-9a-f]+'","WORKER_SHA = '"+worker_sha+"'",c)
c=re.sub(r"TRANSITION_SHA = '[0-9a-f]+'","TRANSITION_SHA = '"+helper_sha+"'",c)
c=re.sub(r'EXPECTED_SYNC_LINE = [0-9]+','EXPECTED_SYNC_LINE = '+str(sync[0]),c)
c=c.replace("'resource_policy_change': 'end-of-completed-update cache reclamation4GB->3GB; math/schedule/cap8/native750MB unchanged',",
    "'resource_policy_change': 'end-cache3GB retained; startup admission aliases/caches released after source300+initial commits; math/schedule/cap8/native750MB unchanged',")
point="    source_pins(boundary_context(plan['worker_arguments']))\n    return {'new_acquisition_updates'"
assert point in c
c=c.replace(point,'''    release = read(run / 'source_admission_release.json', {})
    require(release.get('kind') == 'scheduled_row_startup_admission_release_v3'
        and release.get('passed') is True and release.get('run_id') == plan['run_id']
        and release.get('before_first_optimizer_update') is True
        and release.get('initial_BEST_and_latest_committed') is True
        and release.get('actual_source300_sha256') == sha(run / PROOF_NAME)
        and release.get('source_step') == inherited and release.get('local_scheduled_updates') == 0
        and release.get('row_Adam_clock') == plan['source_acquisition_updates']
        and release.get('expanded_inherited_runtime_alias_removed') is True
        and release.get('authoritative_history_rows_retained') == inherited
        and release.get('cache_release', {}).get('scheduled_admission_entries_after') == 0
        and release.get('cache_release', {}).get('original_acquisition_admission_entries_after') == 0,
        'Actual startup-only source admission release evidence required')
    source_pins(boundary_context(plan['worker_arguments']))
    return {'new_acquisition_updates' ''')
c=c.replace("'new_acquisition_updates' :", "'new_acquisition_updates':")
controller_sha=write(new[2],c)
fixture=rename((ROOT/'work/research/check_TSC_bpe_row_schedule_memory_controller_v2.py').read_text(encoding='utf-8'))
fixture=fixture.replace('inactive_TSC_BPE_scheduled_row_memory_dispatch_CPU_verification_v2','inactive_TSC_BPE_scheduled_row_memory_dispatch_CPU_verification_v3')
p=ROOT/'work/research/check_TSC_bpe_row_schedule_memory_controller_v3.py';assert not p.exists();ast.parse(fixture);p.write_text(fixture,encoding='utf-8')
print({'helper':helper_sha,'worker':worker_sha,'sync':sync[0],'controller':controller_sha})
