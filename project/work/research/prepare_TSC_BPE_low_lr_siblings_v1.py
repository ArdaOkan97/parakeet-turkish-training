"""Stdlib-only creation of three inactive one-variable challenger siblings."""
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / "outputs/parakeet_turkish"

def changed(source, changes):
    for before, after in changes:
        if before not in source:
            raise ValueError("Missing exact predecessor text: " + before[:100])
        source = source.replace(before, after)
    return source

def create(name, source):
    path = HERE / name
    if path.exists():
        raise ValueError("New sibling already exists: " + name)
    path.write_text(source, encoding="utf-8")

helper = (HERE / "tsc_bpe_lora_transition_v2.py").read_text(encoding="utf-8")
helper = changed(helper, [
    ('STATE_KIND = "HF_TSC_BPE512_new_rows_plus_encoder_LoRA_state_v1"', 'STATE_KIND = "HF_TSC_BPE512_new_rows_plus_encoder_LoRA_low_lr_state_v1"'),
    ('RECEIPT_KIND = "ROOT_owned_closed_TSC_BPE_phase1_for_LoRA_v1"', 'RECEIPT_KIND = "ROOT_owned_closed_TSC_BPE_phase1_for_low_LR_LoRA_v1"'),
    ('WORKER = "train_lora_batched_tsc_bpe_lora_native4_v2.py"', 'WORKER = "train_lora_batched_tsc_bpe_lora_low_lr_v1.py"'),
    ('OUTER = "run_TSC_bpe_lora_native4_v2.py"', 'OUTER = "run_TSC_bpe_lora_low_lr_v1.py"'),
    ('    "bpe_newrows_lora_adaptation_v1.py": "2f70811d67af5fc950fa547727dc5ae18fa895a1e5e2cb572c4ff40bbc42476b",', '    "bpe_newrows_lora_adaptation_v1.py": "2f70811d67af5fc950fa547727dc5ae18fa895a1e5e2cb572c4ff40bbc42476b",\n    "bpe_newrows_lora_adaptation_low_lr_v1.py": "PENDING",'),
    ('"encoder_LoRA_learning_rate": 3e-4, "weight_decay": 0.,', '"encoder_LoRA_learning_rate": 3e-5, "source_encoder_LoRA_learning_rate": 3e-4,\n        "active_lora_lr_change_only": True, "source_phase2_latest_used": False, "weight_decay": 0.,'),
    ('        and receipt.get("allow_new_rows_plus_encoder_LoRA") is True', '        and receipt.get("allow_new_rows_plus_encoder_LoRA") is True\n        and receipt.get("source_phase2_latest_used") is False'),
    ('if name != "bpe_newrows_lora_adaptation_v1.py"', 'if name not in ("bpe_newrows_lora_adaptation_v1.py", "bpe_newrows_lora_adaptation_low_lr_v1.py")'),
    ('0 < args.adaptation_max_updates <= 10000', '0 < args.adaptation_max_updates <= 1000'),
    ('require(args.native_max_joint_bytes == NATIVE_MAX_BYTES', 'require(args.native_max_joint_bytes == NATIVE_MAX_BYTES\n        and args.adapt_lora_lr == 3e-5'),
    ('require(sha(HERE / "bpe_newrows_lora_adaptation_v1.py") == SOURCE_PINS["bpe_newrows_lora_adaptation_v1.py"],', 'require(sha(HERE / "bpe_newrows_lora_adaptation_low_lr_v1.py") == SOURCE_PINS["bpe_newrows_lora_adaptation_low_lr_v1.py"],'),
    ('from bpe_newrows_lora_adaptation_v1 import BPENewRowsLoRAAdaptation', 'from bpe_newrows_lora_adaptation_low_lr_v1 import BPENewRowsLoRAAdaptation'),
])
helper = helper.replace('"boundary": {**boundary,', '"boundary": {**boundary, "declared_adaptation": adaptation_declaration(),')
helper = helper.replace('"source_receipt": context["receipt_record"], "source_selected_step":', '"source_receipt": context["receipt_record"], "declared_adaptation": adaptation_declaration(), "source_selected_step":')
create("tsc_bpe_lora_low_lr_transition_v1.py", helper)

worker = (HERE / "train_lora_batched_tsc_bpe_lora_native4_v2.py").read_text(encoding="utf-8")
worker = worker.replace("tsc_bpe_lora_transition_v2", "tsc_bpe_lora_low_lr_transition_v1")
worker = worker.replace("bpe_newrows_lora_adaptation_v1", "bpe_newrows_lora_adaptation_low_lr_v1")
worker = worker.replace("run_TSC_bpe_lora_native4_v2.py", "run_TSC_bpe_lora_low_lr_v1.py")
worker = changed(worker, [
    ('and evidence.get("geometry") == GEOMETRY,', 'and evidence.get("geometry") == GEOMETRY\n        and evidence.get("lora_lr") == args.adapt_lora_lr == 3e-5\n        and evidence.get("source_original_lora_lr") == args.lora_lr == 3e-4,'),
    ('"row_lr": 1e-4, "lora_lr": 3e-4, "max_updates":', '"row_lr": 1e-4, "lora_lr": args.adapt_lora_lr, "source_original_lora_lr": args.lora_lr, "max_updates":'),
    ('constant LRs1e-4/3e-4/decay0', 'constant active LRs1e-4/3e-5/decay0; original LoRA moments/clock retained with dormant sourceLR3e-4'),
    ('p.add_argument("--adaptation-max-updates", type=int, default=1000)', 'p.add_argument("--adapt-lora-lr", type=float, default=3e-5,\n                   help="Active LoRA LR; source --lora-lr stays dormant3e-4")\n    p.add_argument("--adaptation-max-updates", type=int, default=250)'),
    ('1 <= args.adaptation_max_updates <= 10000', '1 <= args.adaptation_max_updates <= 1000 and args.adapt_lora_lr == 3e-5'),
    ('constant1e-4/3e-4 phase', 'constant active1e-4/3e-5 with original LoRA moments/clock and dormant source3e-4 phase'),
])
create("train_lora_batched_tsc_bpe_lora_low_lr_v1.py", worker)

controller = (HERE / "run_TSC_bpe_lora_native4_v2.py").read_text(encoding="utf-8")
controller = controller.replace("train_lora_batched_tsc_bpe_lora_native4_v2.py", "train_lora_batched_tsc_bpe_lora_low_lr_v1.py")
controller = controller.replace("tsc_bpe_lora_transition_v2.py", "tsc_bpe_lora_low_lr_transition_v1.py")
controller = controller.replace("bpe_newrows_lora_adaptation_v1.py", "bpe_newrows_lora_adaptation_low_lr_v1.py")
controller = changed(controller, [
    ("WORKER_SHA = '15825a12c35b1262b28509c921a5a763ecf17696ee02c58446d5a52a33218be3'", "WORKER_SHA = 'PENDING'"),
    ("TRANSITION_SHA = '22407adf7301fae61c823cf391c87c486bc60e48d748935210b2ed91c8a77469'", "TRANSITION_SHA = 'PENDING'"),
    ("'bpe_newrows_lora_adaptation_low_lr_v1.py': '2f70811d67af5fc950fa547727dc5ae18fa895a1e5e2cb572c4ff40bbc42476b'", "'bpe_newrows_lora_adaptation_low_lr_v1.py': 'PENDING'"),
    ("EXPECTED_SYNC_LINE = 458", "EXPECTED_SYNC_LINE = 0"),
    ("STATE_KIND = 'HF_TSC_BPE512_new_rows_plus_encoder_LoRA_state_v1'", "STATE_KIND = 'HF_TSC_BPE512_new_rows_plus_encoder_LoRA_low_lr_state_v1'"),
    ("CONTROLLER_KIND = 'guarded_TSC_BPE512_new_rows_plus_encoder_LoRA_controller_v1'", "CONTROLLER_KIND = 'guarded_TSC_BPE512_new_rows_plus_encoder_LoRA_low_lr_controller_v1'"),
    ("and float(option(argv, '--lora-lr')) == 3e-4", "and float(option(argv, '--lora-lr')) == 3e-4\n        and float(option(argv, '--adapt-lora-lr')) == 3e-5"),
    ("'--acquisition-max-updates', '--adaptation-max-updates'", "'--acquisition-max-updates', '--adapt-lora-lr', '--adaptation-max-updates'"),
    ("acquisition_lr=float(option(argv, '--acquisition-lr')),", "acquisition_lr=float(option(argv, '--acquisition-lr')), adapt_lora_lr=float(option(argv, '--adapt-lora-lr')),"),
    ("('--adaptation-source-receipt', receipt_path), ('--acquisition-lr', 1e-4),", "('--adaptation-source-receipt', receipt_path), ('--acquisition-lr', 1e-4), ('--adapt-lora-lr', 3e-5),"),
    ("'row_lr': 1e-4, 'lora_lr': 3e-4,", "'row_lr': 1e-4, 'lora_lr': 3e-5, 'source_original_lora_lr': 3e-4,"),
    ("default=1000)", "default=250)"),
    ("'default_adaptation_max_updates': 1000", "'default_adaptation_max_updates': 250"),
])
controller = controller.replace("<= 10000", "<= 1000")
controller = controller.replace("atclockS with3e-4", "atclockS with active3e-5 (source3e-4 moments retained)")
controller = controller.replace("atclockS/LR3e-4", "atclockS/activeLR3e-5 (dormant sourceLR3e-4)")
create("run_TSC_bpe_lora_low_lr_v1.py", controller)
print("Created three inactive low-LR siblings; runtime hashes remain PENDING until peer utility freeze.")
