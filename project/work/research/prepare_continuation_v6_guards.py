from pathlib import Path

root = Path(__file__).resolve().parents[2]
p = root / "work/research/test_full_nemo_continuation_v3.py"
text = p.read_text(encoding="utf-8")
for before, after in (
    ("continuation_v3", "continuation_v6"), ("native_B4_250", "native_B2_250"),
    ("B4", "B2"), ("full_nemo_turkish_v3", "full_nemo_turkish_v6"),
    ("train_full_nemo_turkish_v4", "train_full_nemo_turkish_v7"),
    ("full_nemo_turkish_v4", "full_nemo_turkish_v7"),
    ('== 4 and cfg["trainer"]["accumulate_grad_batches"] == 4', '== 2 and cfg["trainer"]["accumulate_grad_batches"] == 8'),
    ('== 4000 and cfg["trainer"]["check_val_every_n_epoch"]', '== 8000 and cfg["trainer"]["check_val_every_n_epoch"]'),
    ('("allocator_cap_gb", 9)', '("allocator_cap_gb", 8)'),
    ('"completed": 1000, "ready": 1000', '"completed": 2000, "ready": 2000'),
    ('= 1001', '= 2001'), ("and1000 microbatches", "and2000 microbatches"),
):
    text = text.replace(before, after)
marker = 'prepared = {"run_id": fixture.name, "entry_sha256": cfg["source_pilot_entry_sha256"],'
assert text.count(marker) == 1
text = text.replace(marker, marker + '\n            "config_sha256": cfg["source_pilot_config_sha256"],')
extra = '''
check("stock periodic checkpoints retain latest with no serialized counter rewrite", "periodic = ModelCheckpoint(" in source and "every_n_train_steps=500, save_top_k=1" in source and "def on_save_checkpoint" not in source)
check("every completedDEV checkpoint saved even when unqualified", "recovery_dev_step_{step}.ckpt" in source and "completed_DEV_boundary" in source)
check("periodic resume explicitly disabled until cursor review", cfg["periodic_checkpoint_policy"]["periodic_checkpoint_auto_resume_enabled"] is False)
check("first resumed250 is not a stock periodic trigger", cfg["periodic_checkpoint_policy"]["every_n_train_steps"] == 500 and 250 % 500 != 0)
'''
text = text.replace('report = {"schema": 1,', extra + '\nreport = {"schema": 1,')
(root / "work/research/test_full_nemo_continuation_v6.py").write_text(text, encoding="utf-8")
