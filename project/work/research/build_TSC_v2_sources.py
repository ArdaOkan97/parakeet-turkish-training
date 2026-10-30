"""One-time stdlib source assembly; never edits frozen v1 files."""
from pathlib import Path
import hashlib

HERE = Path(__file__).resolve().parents[2]/'outputs/parakeet_turkish'
pins = {'tsc_training_inputs_v1.py':'7123bd6978294af4fe15c6c475699db825af98d245f1940d827ee68f961f3f7c',
        'train_lora_batched_tsc_v1.py':'1bf121e4d25b39112c8a05412b856ac4d7c8315c92819f001d7183146d2a0f8b'}
for name,digest in pins.items():
    assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest

text=(HERE/'tsc_training_inputs_v1.py').read_text(encoding='utf-8')
text=text.replace('import tsc_supplemental_data_policy_v1 as policy','import tsc_supplemental_data_policy_v2 as policy')
text=text.replace('a10b1cbaa94b4e6810f6a5fd749de4b7743bbaf68504c63bbcd5070e8cdd7cdb',
                  '3ead93ded7ab55ad0da197d375c74504e3c7d97c63e51ac824b98cc11e5a6fe4')
text=text.replace('tsc_supplemental_data_policy_v1.py','tsc_supplemental_data_policy_v2.py')
text=text.replace('TSC_supplemental_TR_frame_target_inputs_v1','TSC_supplemental_TR_frame_target_inputs_v2')
text=text.replace('root_owned_TSC_TRAIN_candidate_approval_v1','root_owned_TSC_TRAIN_candidate_approval_v2')
text=text.replace('root_owned_TSC_closed_source_v1','root_owned_TSC_closed_source_v2')
text=text.replace('def approve(approval_path, import_proof_path, *, training=False):',
                  'def approve(approval_path, import_proof_path, *, training=False, quarantine_proof_path=None):')
needle='    if training:\n        require(approval.get("allow_isolated_training_arm") is True,'
replacement='''    require(quarantine_proof_path is not None
            and approval.get("quarantine_proof") == file_record(quarantine_proof_path),
            "Explicit ROOT approval must bind this actual uncertainty derivative")
    if training:
        require(approval.get("allow_isolated_training_arm") is True,'''
assert text.count(needle)==1; text=text.replace(needle,replacement)
text=text.replace('        replay_pool_path=refs["replay_pool"]["path"],',
                  '        quarantine_proof_path=refs["quarantine_proof"]["path"], replay_pool_path=refs["replay_pool"]["path"],')
text=text.replace('    approve(args.root_approval, args.tsc_import_proof)',
                  '    approve(args.root_approval, args.tsc_import_proof, quarantine_proof_path=args.tsc_quarantine_proof)')
text=text.replace('        replay_pool_path=args.replay_pool, replay_pool_sha256=',
                  '        quarantine_proof_path=args.tsc_quarantine_proof, replay_pool_path=args.replay_pool, replay_pool_sha256=')
text=text.replace('    refs = {"import_proof": file_record(args.tsc_import_proof),',
                  '    refs = {"quarantine_proof": file_record(args.tsc_quarantine_proof), "import_proof": file_record(args.tsc_import_proof),')
text=text.replace('              "TSC_clips": len(supplemental), "replay_clips": 1595,',
                  '              "TSC_clips": len(supplemental), "replay_clips": 1595, "TSC_quarantine": context.quarantine,')
text=text.replace('    approve(args.tsc_training_approval, context.proof_path, training=True)',
                  '    approve(args.tsc_training_approval, context.proof_path, training=True, quarantine_proof_path=context.quarantine_path)')
text=text.replace('("tsc-import-proof", "root-approval", "replay-pool", "replay-config", "output-dir", "report")',
                  '("tsc-import-proof", "tsc-quarantine-proof", "root-approval", "replay-pool", "replay-config", "output-dir", "report")')
text=text.replace('    source_science = source["config_identity"]["training_arguments"]',
                  '    source_science = common_source_recipe(source, closure)')
text=text.replace('"Actual CLOSED qualifying uniform source is required"','"Actual CLOSED qualifying original8199 uniform/domain source is required"')
text=text.replace('            "source_scientific_arguments": source_science,',
                  '            "source_scientific_arguments": source_science, "source_sampling_transition": source_sampling_transition(source, closure),')
text=text.replace('                                             "Set one uniform combined-pool pass and ceil(actualCombinedN/8) update horizon"],',
                  '                                             "Set one uniform combined-pool pass and ceil(actualCombinedN/8) update horizon",\n                                             "Explicit source sampler transition where applicable; initial new Adam/sampler/RNG, unchanged replay data with no replay-order equality claim"],')
insert='''COMMON_TRAINING_KEYS = frozenset(("epochs", "steps", "batch_size", "accumulation", "replay_fraction", "bucket_size", "seed",
    "max_tokens", "head_lr", "lora_lr", "new_token_lr_multiplier", "replay_kl", "duration_kl", "feature_distillation",
    "rank", "alpha", "last_n_layers", "chunk_size", "tdt_implementation", "warmup_steps", "schedule", "min_lr_ratio",
    "eval_per_epoch", "eval_epoch_fraction", "eval_every_updates", "min_improvement", "no_encoder_checkpoint",
    "phonetic_sh_donors", "train_limit", "early_stop_patience", "max_retention_failures", "target_turkish_wer"))
DOMAIN_TRAINING_KEYS = frozenset(("sampling_mode", "fleurs_fraction", "replay_seed"))
DOMAIN_SOURCES = {"train_lora_batched_domain30_v1.py":"7242b0ce65619f8df38593e89c6ed5b7a01487d1d1174c2e28f73d00bff8ee4f",
    "turkish_domain_sampler_v2.py":"822d0ddb608d719112e4a4285ad42a76ef7f1e4ede0cf4d97211fe38f97c62fd",
    "domain30_training_identity_v1.py":"151bf1b3236e3e23c269f7be2c2e94bf12a71b1e547b64a70f26d6f77ae81f5e"}


def common_source_recipe(source, closure):
    """Return only the identical common recipe; source sampler policy is declared separately."""
    recipe = source["config_identity"]["training_arguments"]
    extra = set(recipe)-COMMON_TRAINING_KEYS
    require(COMMON_TRAINING_KEYS.issubset(recipe) and extra in (set(), set(DOMAIN_TRAINING_KEYS)),
            "Only the reviewed common recipe and exact three domain sampler arguments are supported")
    if extra:
        identity = source["config_identity"].get("domain_sampling_identity", {})
        require(recipe["sampling_mode"] in ("domain30", "uniform") and recipe["fleurs_fraction"] == .3
                and type(recipe["replay_seed"]) is int
                and identity.get("sampling_mode") == recipe["sampling_mode"]
                and identity.get("replay_seed") == recipe["replay_seed"]
                and identity.get("turkish_seed") == recipe["seed"]
                and identity.get("replay_rng_independent_of_turkish") is True
                and identity.get("fleurs_fraction_numerator") == 3 and identity.get("fleurs_fraction_denominator") == 10
                and identity.get("batch_size") == 8 and identity.get("accumulation") == 2
                and identity.get("replay_per_update") == 8,
                "Actual original independent-replay domain sampling identity differs")
        require(closure.get("allow_source_sampler_transition_to_combined_uniform") is True
                and closure.get("source_sampling_metadata") == identity,
                "ROOT must explicitly approve domain/independent-replay to combined-uniform/shared-RNG transition")
        for name, digest in DOMAIN_SOURCES.items():
            require(source["code_sha256"].get(name) == digest and policy.sha256(HERE/name) == digest,
                    "Closed domain source implementation differs: "+name)
    else:
        require(not source["config_identity"].get("domain_sampling_identity"), "Do not hide domain source identity")
    return {key: value for key, value in recipe.items() if key in COMMON_TRAINING_KEYS}


def source_sampling_transition(source, closure):
    common_source_recipe(source, closure)
    recipe = source["config_identity"]["training_arguments"]
    independent = "sampling_mode" in recipe
    return {"source_mode": recipe.get("sampling_mode", "legacy_uniform"),
        "source_sampling_metadata": source["config_identity"].get("domain_sampling_identity"),
        "source_fleurs_fraction": recipe.get("fleurs_fraction"), "source_replay_seed": recipe.get("replay_seed"),
        "target_mode": "combined_uniform", "source_replay_RNG_independent": independent,
        "target_replay_RNG_independent": False, "original_replay_pool_teacher_labels_unchanged": True,
        "replay_order_bitwise_identity_claimed": False,
        "scope": "Model-only qualifying source warm start; NEW Adam/sampler/RNG; declared sampler-policy change, not exact continuation"}


'''
needle='def closed_source_identity(args):';assert text.count(needle)==1;text=text.replace(needle,insert+needle)
path=HERE/'tsc_training_inputs_v2.py';assert not path.exists();path.write_text(text,encoding='utf-8',newline='\n')
worker=(HERE/'train_lora_batched_tsc_v1.py').read_text(encoding='utf-8')
worker=worker.replace('from tsc_training_inputs_v1 import','from tsc_training_inputs_v2 import')
worker=worker.replace('train_lora_batched_tsc_v1.py','train_lora_batched_tsc_v2.py')
worker=worker.replace('tsc_training_inputs_v1.py','tsc_training_inputs_v2.py')
worker=worker.replace('"tsc_supplemental_data_policy_v1.py",',
                      '"tsc_supplemental_data_policy_v2.py", "tsc_supplemental_data_policy_v1.py", "prepare_TSC_quarantine_v2.py",')
worker=worker.replace('launch_native_training_tsc_v1.py','launch_native_training_tsc_v2.py')
path=HERE/'train_lora_batched_tsc_v2.py';assert not path.exists();path.write_text(worker,encoding='utf-8',newline='\n')
print('NEW helper and worker created; frozen v1 untouched')
