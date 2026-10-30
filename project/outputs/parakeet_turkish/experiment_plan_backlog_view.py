"""Render the current ordered plan for the canonical Backlog task."""
from __future__ import annotations


def description(plan):
    labels = {
        'tested_not_improved': 'Tested; no WER improvement',
        'tested_resource_stop_not_improved': 'Tested; memory guard stopped at993 updates; no WER improvement',
        'running': 'Training is running',
        'running_smoke': 'GPU smoke test is running',
        'resource_stopped_repair_in_progress': 'Stopped by the memory guard; repair and fresh measured trial in progress',
        'queued_in_local_process': 'Queued in the local process',
        'prepared_waiting_for_fixed_control_closure': 'Implemented; waiting for the preceding run to close',
        'prepared_waiting_for_preceding_trial': 'Implemented; real GPU and WER trial still pending',
        'audit_complete_quality_control_untested': 'Loss-scaling audit complete; WER control still untested',
        'next_for_implementation': 'Next to implement and run',
        'conditional_next': 'Run once a tested recipe improves WER and passes retention',
        'preparing_full_data_priority': 'Immediate next run; full-data training and Portuguese admission are being prepared',
        'partial_training_resource_stop_deferred': 'Partial run stopped by memory guard; further screens deferred until the full-data pass',
        'planned': 'Planned; still untested',
        'planned_if_512_underperforms': 'Conditional; if512 pieces still underperform',
    }
    lines = ['Execute this ordered experiment plan without waiting for routine user input.',
             'Status is generated from experiment_plan.json; actual controller/native completion and reviewed DEV results are authoritative.', '']
    for position, item in enumerate(plan['ordered_experiments'], 1):
        label = labels.get(item['status'], item['status'].replace('_', ' '))
        lines.append(f"{position}. {item['name']} (experiment {item['id']}) - {label}.")
    lines.extend(['',
        'After each trial, record actual WER, retention, token adoption, applied LR, throughput and unique dataset coverage. Preserve the qualifying champion and implement or launch the next defensible experiment.',
        'The immediate user priority is a full uniform pass over all253798 Turkish clips from the best tested original8199 BEST6500 recipe. Preserve all207 weights/Adam and RNG; explicitly migrate the sampler and add Portuguese. Further short screens wait until the full-pass learning curve is reviewed.',
        'Keep one GPU training child at a time. Inspect and repair resource failures before further dispatch; never retry an unchanged failing runtime.',
        'The active15-minute chat heartbeat follows this plan. Final completion requires competitive Turkish WER, multilingual retention, held-out comparison, independently verified export and confidence evaluation.',
        'References: experiment_plan.json and experiment_queue_state.json.'])
    return '\n'.join(lines)
