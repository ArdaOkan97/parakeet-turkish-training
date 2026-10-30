"""Bounded real-handle CPU exit drain with separate observer/family limits.

The outer native coordinator is not part of the reviewed CPU-child allowance.
Its unchanged ParentGuard remains mandatory; only captured Popen descendants
are summed against the CPU-family limit. This helper never attests a report,
native guard, scientific state, or Functions tool completion.
"""
import os
import subprocess
import time

from asr18_owned_family_observation_v4 import same_process
from bpe_broad_audio_v1 import require
from resource_watchdog_v2 import sample_memory


def wait_actual_CPU_family_exit(child, launcher, known, *, guard,
                                CPU_family_private_allowance_bytes, timeout=2):
    require(isinstance(child, subprocess.Popen) and child.pid == launcher['pid'],
            'Actual owned CPU Popen handle required')
    require(type(timeout) in (int, float) and 0 < timeout <= 2,
            'At most two seconds for captured CPU exit drain')
    require(type(CPU_family_private_allowance_bytes) is int
            and 0 < CPU_family_private_allowance_bytes <= 2_000_000_000,
            'Explicit unchanged CPU-child family allowance required')
    require((launcher['pid'], launcher['create_time']) in known
            and known[(launcher['pid'], launcher['create_time'])] == launcher,
            'Original captured launcher must remain in the known identity set')
    require(all(identity['pid'] != os.getpid() for identity in known.values()),
            'Observer must not be counted as an owned CPU descendant')
    code = child.poll()
    require(not same_process(launcher) or type(code) is int and code == 0,
            'A live wait-pending launcher cannot enter normal exit handling')
    began = time.monotonic(); samples = 0; family_peak = 0; parent_peak = 0
    draining = []
    while True:
        own = guard(); samples += 1
        require(own['paired_sampling_within_one_second']
                and own['system_memory']['available']
                and own['worker_private_memory']['available']
                and own['worker_private_memory']['pid'] == os.getpid(),
                'Fresh paired exact observer/system guard during CPU exit drain')
        parent_private = own['worker_private_memory']['private_bytes']
        require(type(parent_private) is int and 0 < parent_private < 2_000_000_000,
                'Unchanged independent native-parent private limit during drain')
        parent_peak = max(parent_peak, parent_private)
        private = 0; live = []
        for identity in tuple(known.values()):
            if not same_process(identity): continue
            observed = sample_memory(identity['pid'], 'outer_owned_CPU_exit_drain')
            worker = observed['worker_private_memory']
            if not worker['available'] or not same_process(identity):
                require(not same_process(identity), 'Live captured CPU private telemetry unavailable')
                continue
            require(observed['paired_sampling_within_one_second']
                    and observed['system_memory']['available']
                    and worker['pid'] == identity['pid']
                    and type(worker['private_bytes']) is int and worker['private_bytes'] > 0,
                    'Fresh paired exact captured CPU private telemetry required')
            private += worker['private_bytes']; live.append(identity)
        family_peak = max(family_peak, private)
        require(private <= CPU_family_private_allowance_bytes,
                'Unchanged reviewed whole CPU-child allowance exceeded during drain')
        code = child.poll()
        if code is not None:
            require(type(code) is int and code == 0, 'Nonzero CPU launcher cannot become normal completion')
            actual = child.wait(timeout=0)
            require(type(actual) is int and actual == 0, 'Real owned CPU Popen wait0 required')
            if not any(same_process(identity) for identity in known.values()):
                return {'kind': 'ASR18_outer_CPU_actual_wait0_captured_family_drain_v1',
                    'launcher': launcher, 'actual_Popen_wait_code': actual,
                    'all_captured_owned_identities_observed_dead': True,
                    'guard_samples': samples, 'captured_CPU_family_private_peak_bytes': family_peak,
                    'separate_parent_private_peak_bytes': parent_peak,
                    'CPU_family_private_allowance_bytes': CPU_family_private_allowance_bytes,
                    'separate_parent_private_limit_bytes': 2_000_000_000,
                    'captured_identities_observed_draining': draining,
                    'wall_seconds': time.monotonic()-began,
                    'normal_native_exit0_claimed': False,
                    'clean_guard_or_report_validated_here': False,
                    'tool_CLOSED0_claimed': False}
        for identity in live:
            if identity not in draining: draining.append(identity)
        require(time.monotonic()-began < timeout,
                'Captured CPU identities or actual handle exceeded bounded exit drain')
        time.sleep(.02)
