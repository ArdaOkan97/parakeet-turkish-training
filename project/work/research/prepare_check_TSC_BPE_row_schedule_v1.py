"""Create NEW scoped scheduled-contract fixtures from frozen stdlib fixtures."""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[2]

def replace(text):
    for old, new in (
        ('run_TSC_bpe_row_lr_v1', 'run_TSC_bpe_row_schedule_v1'),
        ('tsc_bpe_row_lr_transition_v1', 'tsc_bpe_row_schedule_transition_v1'),
        ('continued_acquisition_declaration', 'scheduled_acquisition_declaration'),
        ('allow_continued_row_only_acquisition', 'allow_scheduled_row_only_acquisition'),
        ('declared_row_lr', 'declared_row_schedule'),
        ('active_row_lr', 'peak_row_lr'), ('--active-row-lr', '--peak-row-lr'),
        ('--resume-row-lr', '--resume-row-schedule'), ('--row-lr-source-receipt', '--row-schedule-source-receipt'),
        ('continued_row_LR', 'scheduled_row'), ('TSC_BPE_row_LR', 'TSC_BPE_row_schedule'),
    ): text = text.replace(old, new)
    return text

text = replace((ROOT / 'work/research/check_TSC_bpe_row_lr_controller_v1.py').read_text(encoding='utf-8'))
text = text.replace("'require': m.require, 'file_sha': m.sha, 'PHASE': m.PHASE, 'GEOMETRY': m.GEOMETRY}",
    "'require': m.require, 'file_sha': m.sha, 'PHASE': m.PHASE, 'GEOMETRY': m.GEOMETRY, 'schedule_identity': m.row_schedule}")
text = text.replace("'actual_controller_dispatch_function_executed': True,", "'actual_controller_dispatch_function_executed': True, 'scheduled_utility_sha256': m.sha(HERE / 'bpe_newrows_scheduled_acquisition_v1.py'),")
path = ROOT / 'work/research/check_TSC_bpe_row_schedule_controller_v1.py'
assert not path.exists(); ast.parse(text); path.write_text(text, encoding='utf-8')

text = replace((ROOT / 'work/research/check_TSC_bpe_row_lr_boundary_v1.py').read_text(encoding='utf-8'))
text = text.replace("    old=ast.parse", "    from bpe_newrows_scheduled_acquisition_v1 import lr_for_update\n    for peak in (1e-3, 3e-4):\n        check(lr_for_update(1, peak)==peak/50 and lr_for_update(50, peak)==peak and lr_for_update(1000, peak)==peak*.1, 'schedule_endpoints_'+str(peak))\n        altered=copy.deepcopy(r);altered['declared_row_schedule']['row_schedule']['horizon_updates']=1001\n        rejected(lambda:h.validate_receipt(altered,'synthetic_rowLR'), 'schedule_clock_change_rejected_'+str(peak))\n    old=ast.parse")
path = ROOT / 'work/research/check_TSC_bpe_row_schedule_boundary_v1.py'
assert not path.exists(); ast.parse(text); path.write_text(text, encoding='utf-8')
print('Created NEW scheduled stdlib fixtures')
