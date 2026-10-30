"""INACTIVE conservative pre-update commit-pressure scheduling gate.
No workload forecast, measured saving, allocation, model, sampler or journal operation.
Existing actual monitor/check callbacks retain their exceptions and every guard.
"""
import time

POLICY = {
    'kind': 'ASR18_native_preupdate_additional_commit_pressure_buffer_v1',
    'native_actual_commit_headroom_bytes': 2_000_000_000,
    'native_actual_physical_headroom_bytes': 1_500_000_000,
    'full_additional_prefetch_host_reservation_bytes': 268_435_456,
    'additional_native_commit_pressure_buffer_bytes': 2_500_000_000,
    'maximum_wait_seconds': 300,
    'poll_interval_seconds': 0.5,
    'freed_memory_credit_claimed': False,
    'workload_forecast_claimed': False,
}

def wait_for_native_update_commit_pressure(*, observe, check, policy, clock=time.monotonic, sleep=time.sleep):
    """Run before packet take/propose/native; return only after stricter admission.
    The observe callback is unchanged original Monitor._observe, so a crossing
    of actual2GB/1.5GB floors still raises immediately. The check callback is
    unchanged cooperative stop/monitor guard. No exception is caught/relabelled.
    """
    if policy != POLICY:
        raise ValueError('Exact separately reviewed native pressure scheduling policy')
    required_commit = (policy['native_actual_commit_headroom_bytes']
        + policy['full_additional_prefetch_host_reservation_bytes']
        + policy['additional_native_commit_pressure_buffer_bytes'])
    required_physical = (policy['native_actual_physical_headroom_bytes']
        + policy['full_additional_prefetch_host_reservation_bytes'])
    began = clock()
    observations = 0
    while True:
        check()
        sample = observe()
        observations += 1
        system = sample['system_memory']
        if (system['commit_available_bytes'] >= required_commit
            and system['physical_available_bytes'] >= required_physical):
            check()
            return {
                'kind': 'ASR18_actual_native_preupdate_commit_pressure_admission_v1',
                'observations': observations,
                'wait_seconds': clock() - began,
                'required_commit_available_bytes': required_commit,
                'required_physical_available_bytes': required_physical,
                'actual_commit_available_bytes': system['commit_available_bytes'],
                'actual_physical_available_bytes': system['physical_available_bytes'],
                'no_native_sampler_draw_or_journal_operation_in_gate': True,
                'workload_forecast_or_measured_saving_claimed': False,
            }
        elapsed = clock() - began
        if elapsed >= policy['maximum_wait_seconds']:
            error = TimeoutError('Bounded native pre-update commit-pressure wait exhausted; preserve state, no unchanged retry')
            error.native_pressure_denial = {
                'kind': 'ASR18_actual_native_preupdate_commit_pressure_denial_v1',
                'last_denied_sample': sample,
                'required_commit_available_bytes': required_commit,
                'required_physical_available_bytes': required_physical,
                'observed_wait_seconds': elapsed,
                'maximum_wait_seconds': policy['maximum_wait_seconds'],
                'observations': observations,
                'parameters': dict(policy),
                'no_packet_propose_native_or_authoritative_operation_admitted': True,
            }
            raise error
        sleep(min(policy['poll_interval_seconds'], policy['maximum_wait_seconds'] - elapsed))
