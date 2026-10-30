"""CPU-only guards and actual parent/fork proof for the new final auditor."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUTPUT))
import verify_continued_loop_results_v2 as audit


def main():
    output=OUTPUT/'continued_loop_v2_auditor_cpu_verification.json'
    if output.exists():raise FileExistsError(output)
    check=audit.CompactChecks();cases=[]
    policy={'reserved_limit_gb':12,'reserved_observations':3,'commit_limit_fraction':.9,
        'commit_critical_fraction':.95,'commit_headroom_bytes':2_000_000_000,'commit_observations':3,
        'commit_min_interval_seconds':5,'commit_telemetry_max_age_seconds':15,
        'memory_observations_phase_tagged':True,'resource_stop_blocks_all_further_GPU':True}
    audit.audit_watchdog_policy(policy,check);cases.append('Exact changed resource policy accepted')
    rejected=[]
    for key,value in [('commit_critical_fraction',.99),('commit_observations',1),
        ('commit_headroom_bytes',0),('commit_min_interval_seconds',1),('reserved_limit_gb',13),
        ('resource_stop_blocks_all_further_GPU',False)]:
        bad=copy.deepcopy(policy);bad[key]=value
        try:audit.audit_watchdog_policy(bad,audit.CompactChecks())
        except audit.AuditError:rejected.append(key)
        else:raise AssertionError(key)
    from resource_watchdog_v1 import POLICY
    stamp='2026-10-11T05:31:09.743080+00:00'
    row={'policy':POLICY,'worker_private_memory':{'pid':12345,'available':True,'observed_at_utc':stamp,
        'private_bytes':10_000_000_000,'working_set_bytes':8_000_000_000,'peak_working_set_bytes':12_000_000_000},
        'system_memory':{'available':True,'observed_at_utc':stamp,'commit_fraction':.91,
            'commit_used_bytes':45_500_000_000,'commit_limit_bytes':50_000_000_000,'commit_available_bytes':4_500_000_000},
        'observed_monotonic':100.,'moderate_commit_streak':1,
        'worker_phase':{'inferred_from_bounded_log_tail':True}}
    trial={'pid':12345,'watchdog_observations':[row]}
    audit.audit_watchdog_observations(trial,check);cases.append('Single91% transient allowed without weakening critical guard')
    bad=copy.deepcopy(trial);bad['watchdog_observations'][0]['worker_private_memory']={'pid':12345,
        'available':False,'observed_at_utc':stamp,'error':5}
    audit.audit_watchdog_observations(bad,check);cases.append('Private memory read error retained; valid system guard still audited')
    corruptions=[('three_high_polls',lambda r:r.update(moderate_commit_streak=3)),
        ('wrong_owned_PID',lambda r:r['worker_private_memory'].update(pid=54321)),
        ('unlabelled_phase',lambda r:r['worker_phase'].update(inferred_from_bounded_log_tail=False)),
        ('critical95percent',lambda r:r['system_memory'].update(commit_fraction=.95,commit_used_bytes=47_500_000_000,commit_available_bytes=2_500_000_000)),
        ('critical_headroom',lambda r:r['system_memory'].update(commit_fraction=.962,commit_used_bytes=48_100_000_000,commit_available_bytes=1_900_000_000)),
        ('inconsistent_fields',lambda r:r['system_memory'].update(commit_used_bytes=44_000_000_000))]
    for name,mutate in corruptions:
        bad=copy.deepcopy(trial);mutate(bad['watchdog_observations'][0])
        try:audit.audit_watchdog_observations(bad,audit.CompactChecks())
        except audit.AuditError:rejected.append(name)
        else:raise AssertionError(name)
    # The write guard executes before any file read or write.
    for loop in ('adaptive_20261010_continuation_01','adaptive_20261010_recovery_b8_01'):
        args=SimpleNamespace(loop_dir=OUTPUT/'runs'/loop,output_dir=OUTPUT/'runs'/loop,
            frozen_selection=OUTPUT/'runs'/loop/'frozen_final_selection.json',strong_test_comparison=OUTPUT/'unused.json')
        try:audit.main(args)
        except ValueError:rejected.append('old_loop_write_'+loop)
        else:raise AssertionError('old evidence overwritten')
    preflight=audit.json_load(OUTPUT/'continuation_v2_actual_preflight.json')
    parent_state={'prior_loop_id':'adaptive_20261010_continuation_01',
        'prior_deadline_at_utc':preflight['old_deadline_at_utc']}
    receipt=audit.audit_predecessor(preflight,parent_state,check)
    cases.append('Actual stopped-controller receipt/runtime/ordinary801 export audited')
    directory=OUTPUT/'runs/adaptive_20261010_continuation_02'
    state={'fork_manifest_path':str(directory/'fork_snapshot/fork_manifest.json'),
        'fork_manifest_sha256':audit.file_sha(directory/'fork_snapshot/fork_manifest.json'),
        'initial_fork_archive_path':str(directory/'fork_initial_states_archive.json'),
        'initial_fork_archive_sha256':audit.file_sha(directory/'fork_initial_states_archive.json')}
    audit.audit_full_fork(state,directory,receipt,check)
    cases.append('Both actual1773 initialarchives fully reloaded and raw-compared to immutable parent')
    import torch
    assert not torch.cuda.is_initialized()
    result={'schema':1,'passed':True,'audit_source_sha256':audit.file_sha(Path(audit.__file__)),
        'positive_checks':check.count,'full_tensor_leaves':check.tensor_count,'corruptions_rejected':len(rejected),
        'cases':cases,'rejections':rejected,'CUDA_initialized':False,
        'scope':'Actual preserved predecessor receipt/runtime/export and two full1773 archive-state comparisons; synthetic watchdog/write-boundary guards. No future candidate inference/export/outcome claim.'}
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('passed','positive_checks','full_tensor_leaves','corruptions_rejected','audit_source_sha256')}))


if __name__=='__main__':main()
