from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'outputs/parakeet_turkish'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

prior = HERE / 'run_TSC_row_schedule_memory_queue_v2.py'
target = HERE / 'run_TSC_row_schedule_memory_queue_v3.py'
assert not target.exists()
text = prior.read_text(encoding='utf-8')
text = text.replace('row_schedule_memory_v2', 'row_schedule_memory_v3')
text = text.replace('row_schedule_memory_queue_v2', 'row_schedule_memory_queue_v3')
text = text.replace("== 59763", "== 29031").replace("== '68a1dc'", "== '37f468'")
old = """    require(approval['kind'] == 'ROOT_actual_CLOSED_resource_stop_changed_policy_approval_v1'
            and approval['passed'] is True and approval['allow_next_changed_cache_policy'] is True
            and approval['native_complete_and_sources_unchanged'] is True
            and approval['next_controller_sha256'] == sha(CONTROLLER)
            and approval['new_end_cache_threshold_bytes'] == 3_000_000_000
            and approval['old_end_cache_threshold_bytes'] == 4_000_000_000
            and approval['cap_bytes'] == 8_000_000_000,
            'Changed cache policy and actual successful native save are required')"""
new = """    require(approval['kind'] == 'ROOT_actual_CLOSED_resource_stop_startup_host_release_approval_v3'
            and approval['passed'] is True and approval['allow_next_startup_host_release'] is True
            and approval['native_complete_and_sources_unchanged'] is True
            and approval['next_controller_sha256'] == sha(CONTROLLER)
            and approval['end_cache_threshold_bytes'] == 3_000_000_000
            and approval['cap_bytes'] == 8_000_000_000
            and approval['unchanged_guard_thresholds'] is True,
            'Changed host lifetime policy and actual successful native save are required')"""
assert old in text
text = text.replace(old,new)
needle = "    config = read(run / 'run_config.json')"
extra = """    release = read(run / 'source_admission_release.json')
    require(release['kind'] == 'scheduled_row_startup_admission_release_v3'
            and release['passed'] is True and release['before_first_optimizer_update'] is True
            and release['actual_source300_sha256'] == sha(run / 'TSC_BPE_row_schedule_initial_source_equivalence.json')
            and release['expanded_inherited_runtime_alias_removed'] is True
            and release['authoritative_history_rows_retained'] == 7000
            and release['cache_release']['scheduled_admission_entries_after'] == 0
            and release['cache_release']['original_acquisition_admission_entries_after'] == 0,
            'Actual changed host-release policy must execute before main admission')
"""
assert text.count(needle)==1
text = text.replace(needle, extra+needle)
target.write_text(text,encoding='utf-8')
plan_path = HERE / 'experiment_plan.json'
plan=read(plan_path)
item=next(x for x in plan['ordered_experiments'] if x['id']=='03')
item.update(status='resource_stopped_repair_in_progress',actual_updates=9,actual_GPU_trial_completed=False,
            full1000_claimed=False,actual_queue_ROOT_closed_session=29031,
            actual_queue_ROOT_closed_exit_chunk='37f468',
            next_changed_host_policy='Release startup-only inherited runtime alias and cached JSON admission templates after source300 and initial state commits; keep authoritative live state.')
plan_path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'queue':str(target),'sha256':sha(target),'plan_updated':True}))
