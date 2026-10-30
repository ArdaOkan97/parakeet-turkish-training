"""Create only NEW inactive scheduled-row siblings; no ML/data/state reads."""
from pathlib import Path
import ast
import hashlib

HERE = Path(__file__).resolve().parents[2] / 'outputs/parakeet_turkish'
UTILITY_SHA = 'f72485b7d2a0c242cf3a65c77a9be8e514c281d74c1a6ed924d3ea32eb8e0875'

def rep(text, old, new):
    if old not in text: raise ValueError('Missing exact predecessor: ' + old[:100])
    return text.replace(old, new)

def write(name, text):
    path = HERE / name
    if path.exists(): raise ValueError('New sibling already exists: ' + name)
    ast.parse(text)
    path.write_text(text, encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()

def common(text):
    for old, new in (
        ('tsc_bpe_row_lr_transition_v1', 'tsc_bpe_row_schedule_transition_v1'),
        ('train_lora_batched_tsc_bpe_row_lr_v1', 'train_lora_batched_tsc_bpe_row_schedule_v1'),
        ('run_TSC_bpe_row_lr_v1', 'run_TSC_bpe_row_schedule_v1'),
        ('HF_TSC_BPE512_new_rows_continued_acquisition_v1', 'HF_TSC_BPE512_new_rows_scheduled_acquisition_v1'),
        ('HF_TSC_BPE512_new_rows_continued_acquisition_state_v1', 'HF_TSC_BPE512_new_rows_scheduled_acquisition_state_v1'),
        ('HF_TSC_BPE512_new_rows_continued_acquisition_identity_v1', 'HF_TSC_BPE512_new_rows_scheduled_acquisition_identity_v1'),
        ('TSC_BPE_row_LR_initial_source_equivalence.json', 'TSC_BPE_row_schedule_initial_source_equivalence.json'),
        ('TSC_BPE512_row_LR_before_update_source_equivalence_v1', 'TSC_BPE512_row_schedule_before_update_source_equivalence_v1'),
        ('before_row_lr_updates', 'before_row_schedule_updates'),
        ('resume_row_lr', 'resume_row_schedule'), ('row_lr_source_receipt', 'row_schedule_source_receipt'),
        ('--resume-row-lr', '--resume-row-schedule'), ('--row-lr-source-receipt', '--row-schedule-source-receipt'),
        ('active_row_lr', 'peak_row_lr'), ('--active-row-lr', '--peak-row-lr'),
        ('declared_row_lr', 'declared_row_schedule'),
        ('continued_acquisition_declaration', 'scheduled_acquisition_declaration'),
    ): text = text.replace(old, new)
    return text

h = common((HERE / 'tsc_bpe_row_lr_transition_v1.py').read_text(encoding='utf-8'))
h = rep(h, 'ROOT_owned_closed_TSC_BPE_phase1_for_row_LR_v1', 'ROOT_owned_closed_TSC_BPE_phase1_for_row_schedule_v1')
h = rep(h, 'allow_continued_row_only_acquisition', 'allow_scheduled_row_only_acquisition')
h = rep(h, 'from tsc_bpe_acquisition_transition_v1 import (',
    'from bpe_newrows_scheduled_acquisition_v1 import schedule_identity\n\nfrom tsc_bpe_acquisition_transition_v1 import (')
h = rep(h, 'SOURCE_PINS = {', 'SOURCE_PINS = {\n    "bpe_newrows_scheduled_acquisition_v1.py": "' + UTILITY_SHA + '",')
h = rep(h, 'if name != "bpe_newrows_continued_acquisition_v1.py"',
    'if name not in ("bpe_newrows_continued_acquisition_v1.py", "bpe_newrows_scheduled_acquisition_v1.py")')
h = rep(h, 'return {"phase": PHASE, "peak_row_lr": peak_row_lr, "source_row_lr": 1e-4,',
    'return {"phase": PHASE, "peak_row_lr": peak_row_lr, "row_schedule": schedule_identity(peak_row_lr), "source_row_lr": 1e-4,')
h = rep(h, '    from bpe_newrows_continued_acquisition_v1 import BPEContinuedAcquisition\n    acquisition = BPEContinuedAcquisition(',
    '    require(sha(HERE / "bpe_newrows_scheduled_acquisition_v1.py") == SOURCE_PINS["bpe_newrows_scheduled_acquisition_v1.py"], "Reviewed scheduled-row utility changed")\n    from bpe_newrows_scheduled_acquisition_v1 import BPEScheduledAcquisition\n    acquisition = BPEScheduledAcquisition(')
h = rep(h, '"continued_row_identity": acquisition.phase_identity()', '"scheduled_row_identity": acquisition.phase_identity()')
h = h.replace('row-only LR challenger.', 'row-only warmup/cosine challenger.').replace('declared active-row-LR transition', 'declared local warmup/cosine transition')
helper_sha = write('tsc_bpe_row_schedule_transition_v1.py', h)

w = common((HERE / 'train_lora_batched_tsc_bpe_row_lr_v1.py').read_text(encoding='utf-8'))
w = rep(w, 'from bpe_newrows_continued_acquisition_v1 import BPEContinuedAcquisition',
    'from bpe_newrows_scheduled_acquisition_v1 import BPEScheduledAcquisition, schedule_identity, lr_for_update')
w = rep(w, 'and evidence.get("peak_row_lr") == args.peak_row_lr',
    'and evidence.get("peak_row_lr") == args.peak_row_lr\n        and evidence.get("row_schedule") == schedule_identity(args.peak_row_lr)')
w = rep(w, '"bpe_newrows_acquisition_v1.py", "bpe_newrows_continued_acquisition_v1.py", "native_tdt_checkpointed_v4.py",',
    '"bpe_newrows_acquisition_v1.py", "bpe_newrows_continued_acquisition_v1.py", "bpe_newrows_scheduled_acquisition_v1.py", "native_tdt_checkpointed_v4.py",')
w = rep(w, '"acquisition_lr": args.peak_row_lr, "source_row_lr": 1e-4, "source_acquisition_updates":',
    '"acquisition_lr": args.peak_row_lr, "peak_row_lr": args.peak_row_lr, "row_schedule": schedule_identity(args.peak_row_lr), "source_row_lr": 1e-4, "source_acquisition_updates":')
w = rep(w, '"acquisition_updates_completed": acquisition.optimizer_updates_completed,\n            **totals',
    '"acquisition_updates_completed": acquisition.optimizer_updates_completed,\n            **totals') if False else w
w = rep(w, '"row_Adam_clock": acquisition.source_acquisition_updates + acquisition.optimizer_updates_completed,\n            **totals',
    '"row_Adam_clock": acquisition.source_acquisition_updates + acquisition.optimizer_updates_completed,\n            "scheduled_acquisition_updates": acquisition.optimizer_updates_completed,\n            "last_applied_row_lr": acquisition.last_applied_lr, "scheduled_next_update": acquisition.optimizer_updates_completed + 1 if acquisition.optimizer_updates_completed < 1000 else None,\n            **totals')
w = rep(w, '"acquisition_lr": args.peak_row_lr, "peak_row_lr": args.peak_row_lr, "source_row_lr": 1e-4, "head_lr":',
    '"acquisition_lr": acquisition.current_row_lr, "current_row_lr": acquisition.current_row_lr, "active_row_lr": acquisition.current_row_lr, "peak_row_lr": args.peak_row_lr, "source_row_lr": 1e-4, "head_lr":')
w = rep(w, '"peak_row_lr": args.peak_row_lr, "source_row_lr": 1e-4,\n                "encoder_LoRA_frozen":',
    '"peak_row_lr": args.peak_row_lr, "source_row_lr": 1e-4,\n                "current_row_lr": acquisition.current_row_lr, "last_applied_row_lr": acquisition.last_applied_lr,\n                "scheduled_acquisition_updates": acquisition.optimizer_updates_completed, "scheduled_next_update": 2,\n                "row_schedule": schedule_identity(args.peak_row_lr), "scheduled_state_save_load_passed": True,\n                "encoder_LoRA_frozen":')
w = rep(w, '    _verify_smoke_phase_reload', '    _verify_smoke_phase_reload') if False else w
w = rep(w, '            _verify_smoke_phase_reload(acquisition, saved_phase, runtime)',
    '            _verify_smoke_phase_reload(acquisition, saved_phase, runtime)\n            require(saved_phase["scheduled_acquisition_updates"] == 1 and saved_phase["continued_acquisition_updates"] == 1\n                and saved_phase["acquisition_optimizer_updates"] == acquisition.source_acquisition_updates + 1\n                and saved_phase["current_row_lr"] == saved_phase["last_applied_lr"] == acquisition.current_row_lr == lr_for_update(1, args.peak_row_lr)\n                and saved_phase["scheduled_next_update"] == 2 and saved_phase["row_schedule"] == schedule_identity(args.peak_row_lr),\n                "Actual scheduled smoke saved appliedLR/localN/AdamK+N/next point differs")')
w = rep(w, '"row_Adam_clock": acquisition.source_acquisition_updates + acquisition.optimizer_updates_completed,\n        "acquisition_end_step":',
    '"row_Adam_clock": acquisition.source_acquisition_updates + acquisition.optimizer_updates_completed,\n        "scheduled_acquisition_updates": acquisition.optimizer_updates_completed, "peak_row_lr": args.peak_row_lr,\n        "current_row_lr": acquisition.current_row_lr, "last_applied_row_lr": acquisition.last_applied_lr,\n        "row_schedule": schedule_identity(args.peak_row_lr),\n        "acquisition_end_step":')
w = rep(w, '"acquisition_updates_completed": acquisition.optimizer_updates_completed,\n        "selected_step":',
    '"acquisition_updates_completed": acquisition.optimizer_updates_completed,\n        "selected_step":') if False else w
w = rep(w, '"acquisition_updates_completed": acquisition.optimizer_updates_completed,\n        "config_identity":',
    '"acquisition_updates_completed": acquisition.optimizer_updates_completed,\n        "config_identity":') if False else w
w = rep(w, '"acquisition_updates_completed": acquisition.optimizer_updates_completed,\n        "selected_step":',
    '"acquisition_updates_completed": acquisition.optimizer_updates_completed,\n        "selected_step":') if False else w
w = rep(w, '"continued_optimizer": "acquired three-table AdamK retained; declared active rowLR; no6boost",',
    '"continued_optimizer": "acquired three-table AdamK retained; local warmup50/cosine1000/floor0.1peak; no6boost",\n        "row_schedule": schedule_identity(args.peak_row_lr), "row_LR_clock": "local scheduled N; Adam stays K+N",')
w = w.replace('row-only LR challenger.', 'row-only warmup/cosine challenger.').replace('only rowLR changes to1e-3 or3e-4.', 'only local row-LR schedule changes (warmup50/cosine1000/floor0.1peak).')
w = w.replace('actual_acquired_source300_equal_row_LR_phase0', 'actual_acquired_source300_equal_row_schedule_phase0')
worker_sha = write('train_lora_batched_tsc_bpe_row_schedule_v1.py', w)
tree = ast.parse(w)
run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
sync = [n.lineno for n in ast.walk(run) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'torch.cuda.synchronize']
assert len(sync) == 1

c = common((HERE / 'run_TSC_bpe_row_lr_v1.py').read_text(encoding='utf-8'))
c = rep(c, 'guarded_TSC_BPE512_continued_row_LR_controller_v1', 'guarded_TSC_BPE512_scheduled_row_controller_v1')
c = rep(c, "WORKER_SHA = '7f3f71c0ad6091c9345d866bebaca81d78e2de38b2e818ff6223cb8d5ee644f1'", "WORKER_SHA = '" + worker_sha + "'")
c = rep(c, "TRANSITION_SHA = 'f8647e1214759479c2ee81e57977cfb5f6e576fa54c8c324da4991173ba729ee'", "TRANSITION_SHA = '" + helper_sha + "'")
c = rep(c, 'EXPECTED_SYNC_LINE = 469', 'EXPECTED_SYNC_LINE = ' + str(sync[0]))
c = rep(c, 'UTILITY_PINS = {', "UTILITY_PINS = {\n    'bpe_newrows_scheduled_acquisition_v1.py': '" + UTILITY_SHA + "',")
insert = "\n\ndef row_schedule(peak):\n    from bpe_newrows_scheduled_acquisition_v1 import schedule_identity\n    return schedule_identity(peak)\n\n\ndef applied_row_lr(local, peak):\n    from bpe_newrows_scheduled_acquisition_v1 import lr_for_update\n    return 1e-4 if local == 0 else lr_for_update(local, peak)\n"
c = rep(c, '\n\ndef check_phase_arguments(argv):', insert + '\n\ndef check_phase_arguments(argv):')
c = rep(c, "'peak_row_lr': args.peak_row_lr, 'source_row_lr': 1e-4,", "'peak_row_lr': args.peak_row_lr, 'row_schedule': row_schedule(args.peak_row_lr), 'source_row_lr': 1e-4,")
c = rep(c, "'peak_row_lr': plan['peak_row_lr'], 'source_row_lr': 1e-4,", "'peak_row_lr': plan['peak_row_lr'], 'row_schedule': row_schedule(plan['peak_row_lr']), 'source_row_lr': 1e-4,")
c = rep(c, "'acquisition_lr': plan['peak_row_lr'], 'source_row_lr': 1e-4,", "'acquisition_lr': plan['peak_row_lr'], 'peak_row_lr': plan['peak_row_lr'], 'row_schedule': row_schedule(plan['peak_row_lr']), 'source_row_lr': 1e-4,")
c = rep(c, "optimizer_identity.get('peak_row_lr') == optimizer_identity.get('base_lr') == plan['peak_row_lr']", "optimizer_identity.get('peak_row_lr') == optimizer_identity.get('active_row_lr') == optimizer_identity.get('base_lr') == plan['peak_row_lr']\n        and optimizer_identity.get('row_schedule') == row_schedule(plan['peak_row_lr'])\n        and optimizer_identity.get('dynamic_applied_lr_is_not_scientific_identity') is True")
c = rep(c, "    if plan['smoke_updates']:\n        smoke = read", "    require(selection.get('scheduled_acquisition_updates') == local\n        and selection.get('row_schedule') == row_schedule(plan['peak_row_lr'])\n        and selection.get('peak_row_lr') == plan['peak_row_lr']\n        and selection.get('current_row_lr') == applied_row_lr(local, plan['peak_row_lr'])\n        and selection.get('last_applied_row_lr') == (None if local == 0 else applied_row_lr(local, plan['peak_row_lr'])),\n        'Actual local schedule/appliedLR frontier differs')\n    for row in history[inherited:]:\n        n = row['acquisition_updates_completed']\n        require(row.get('scheduled_acquisition_updates') == n and row.get('row_Adam_clock') == plan['source_acquisition_updates'] + n\n            and row.get('peak_row_lr') == plan['peak_row_lr']\n            and row.get('acquisition_lr') == row.get('current_row_lr') == row.get('last_applied_row_lr') == applied_row_lr(n, plan['peak_row_lr']),\n            'Published applied LR must follow actual localN scheduled Adam step')\n    if plan['smoke_updates']:\n        smoke = read")
c = rep(c, "and smoke.get('peak_row_lr') == plan['peak_row_lr'] and smoke.get('source_row_lr') == 1e-4", "and smoke.get('peak_row_lr') == plan['peak_row_lr'] and smoke.get('source_row_lr') == 1e-4\n            and smoke.get('scheduled_acquisition_updates') == 1 and smoke.get('scheduled_next_update') == 2\n            and smoke.get('current_row_lr') == smoke.get('last_applied_row_lr') == applied_row_lr(1, plan['peak_row_lr'])\n            and smoke.get('row_schedule') == row_schedule(plan['peak_row_lr'])\n            and smoke.get('scheduled_state_save_load_passed') is True")
c = c.replace('Only active rowLR changes.', 'Only a declared local50-update warmup/cosine schedule changes.')
c = c.replace('future ROOT-owned CLOSED currentheadlr qualified BEST; no actual source admitted', 'future ROOT-owned CLOSED qualifying acquired phase1 point7000/K500; no actual source admitted')
c = c.replace('activeLR1e-3 or3e-4, decay0, no6boost', 'warmup50/cosine1000/floor0.1peak; peak1e-3 or3e-4, decay0, no6boost')
c = c.replace('actual GPU proof required before optimizer creation', 'actual GPU proof required before first scheduled Adam update')
controller_sha = write('run_TSC_bpe_row_schedule_v1.py', c)
print({'helper': helper_sha, 'worker': worker_sha, 'sync': sync[0], 'controller': controller_sha})
