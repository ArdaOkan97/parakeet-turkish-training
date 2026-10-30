"""Close private CPU preparation at a coherent periodic DEV frontier."""
from atomic_io_retry_v1 import read_json
from bpe_broad_audio_v1 import record, require
from bpe_broad_production_state_v1 import digest


def close_private_preparation_for_DEV(stream, queued, *, committed_updates, guard):
    """Discard only the exact unconsumed future plan, then require genuine CPU wait0."""
    require(stream is not None, 'One owned resident CPU before periodic DEV quiescence')
    discarded = 0
    plan_pin = None
    if queued is not None:
        plan, plan_pin = queued
        require(record(plan_pin['path']) == plan_pin and read_json(plan_pin['path']) == plan,
                'Keep the exact previously submitted private plan')
        require(plan['initial_committed_updates'] == committed_updates and 0 < len(plan['updates']) <= 20,
                'Only the wholly unconsumed immediate private successor')
        for index, entry in enumerate(plan['updates']):
            guard()
            packet = stream.next_update(digest(plan), index)
            metadata = packet.get('metadata', packet['update'])
            require(packet['plan_sha256'] == digest(plan) and metadata['update_index'] == index
                    and all(metadata[key] == entry[key] for key in
                            ('native_transaction_sha256', 'before_sampler_sha256', 'after_sampler_sha256', 'batch_id')),
                    'Exact ordered private packet before discard; never native consumption')
            stream.release(packet['token'])
            del packet, metadata
            discarded += 1
    guard()
    closed = stream.close()
    require(closed['actual_Popen_wait_code'] == 0 and closed['transport_threads_joined'] is True
            and closed['pipe_handles_closed'] is True, 'Genuine CPU wait0 and released transport before DEV')
    return {'kind': 'ASR18_actual_periodic_DEV_private_CPU_quiescence_v1',
            'optimizer_updates': committed_updates, 'unconsumed_private_plan': plan_pin,
            'private_packets_discarded': discarded, 'actual_native_updates': 0,
            'authoritative_sampler_replay_or_journal_advance': False,
            'private_plan_regenerated_or_reordered': False, 'CPU_actual0': closed,
            'same_exact_plan_must_be_resubmitted_after_DEV': plan_pin is not None,
            'full_prefetch_and_DEV_reservations_unchanged': True}
