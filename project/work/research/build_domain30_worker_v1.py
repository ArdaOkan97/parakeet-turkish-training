"""One-time stdlib construction of NEW inactive scientific worker; no imports."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = root / "outputs/parakeet_turkish"
parent = out / "train_lora_batched_data_v11.py"
target = out / "train_lora_batched_domain30_v1.py"
source = parent.read_text(encoding="utf-8")


def replace(old, new):
    global source
    if source.count(old) != 1:
        raise ValueError("Expected exactly one source block: " + old[:100])
    source = source.replace(old, new)


tree = ast.parse(source)
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BalancedBucketSampler")
lines = source.splitlines(keepends=True)
source = "".join(lines[:node.lineno-1] + lines[node.end_lineno:])
replace('"""Official native NeMo TDT public-data-v3 LoRA candidate; DEV selection only.\n\nWarm starts load exactly the incoming trainable layout with a NEW optimizer.\nExact resume restores optimizer/sampler/RNG of the SAME run and data/config.\nUses vetted CV validated/train-role and FLEURS TRAIN; fixed TEST stays untouched.\n"""',
    '"""Inactive domain30 Turkish sampling arm; original8199 model/loss recipe.\n\nFirst launch uses a CLOSED qualifying source with fresh Adam/sampler/RNG.\nExact resume is allowed only within this NEW scientific identity.\nDomain30 epochs mean minimum FLEURS/CV exposure; replay uses an independent RNG.\nThe optional uniform comparator shares exactly that replay-seed contract.\nFixed TEST remains untouched; all five original DEV retention gates apply.\n"""')
replace('from train_lora import checkpoint_encoder_blocks',
    'from train_lora import checkpoint_encoder_blocks\nfrom turkish_domain_sampler_v2 import DomainBalancedBucketSampler\nfrom domain30_training_identity_v1 import validate_recipe, closed_source_identity')
replace('    process_start = time.perf_counter(); public_data_audit = audit_public_training_inputs(',
    '    process_start = time.perf_counter(); source_identity = closed_source_identity(args); public_data_audit = audit_public_training_inputs(')
replace('    vocabulary = extend_processor_and_model(processor, model, [row["text"] for row in tr])',
    '    vocabulary = extend_processor_and_model(processor, model, [row["text"] for row in tr])\n    if (old_size != 8193 or model.config.vocab_size != 8199 or model.config.blank_token_id != 8192\n            or vocabulary["extended_tokenizer_sha256"] != "b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75"):\n        raise ValueError("Domain arm requires unchanged8199 token geometry and original blank8192")')
replace('        validate_trainable_checkpoint(model, resume, encoder_lora_metadata(model))\n        model.load_state_dict',
    '        validate_trainable_checkpoint(model, resume, encoder_lora_metadata(model))\n        if (not args.resume_exact and (resume.get("run_id") != source_identity["source_run_id"]\n                or resume.get("step") != source_identity["selected_step"])):\n            raise ValueError("Loaded weights are not the closed source selected checkpoint")\n        model.load_state_dict')
replace('    sampler = BalancedBucketSampler(tr, replay, seed=args.seed, batch_size=args.batch_size,\n                                    accumulation=args.accumulation, replay_fraction=args.replay_fraction,\n                                    bucket_size=args.bucket_size)\n    epoch_steps = math.ceil(args.epochs * len(tr) / sampler.tr_per_update)',
    '    sampler = DomainBalancedBucketSampler(tr, replay, seed=args.seed, replay_seed=args.replay_seed,\n            sampling_mode=args.sampling_mode, batch_size=args.batch_size, accumulation=args.accumulation,\n            replay_fraction=args.replay_fraction, fleurs_fraction=args.fleurs_fraction, bucket_size=args.bucket_size,\n            cycler_factory=BucketCycler, cycler_source_sha256=file_sha(OUTPUT / "train_lora_batched_data_v11.py"))\n    epoch_steps = sampler.updates_for_epochs(args.epochs)')
replace('"train_lora_batched_data_v11.py", "hf_exact_runtime_continuation_v1.py",',
    '"train_lora_batched_domain30_v1.py", "turkish_domain_sampler_v2.py", "turkish_domain_sampler_v1.py", "domain30_training_identity_v1.py", "train_lora_batched_data_v11.py",')
replace('"launch_native_training_v5.py",', '"launch_native_training_domain30_v1.py",')
replace('    input_paths = [args.tr_manifest,', '    input_paths = [args.source_closure, args.tr_manifest,')
replace('                       "source_sha256":source_sha, "input_sha256":input_sha, "native_loss":native_loss_identity(args),',
    '                       "source_sha256":source_sha, "input_sha256":input_sha, "native_loss":native_loss_identity(args),\n                       "domain_sampling_identity":sampler.identity, "closed_source":source_identity,')
replace('("epochs","steps","batch_size","accumulation","replay_fraction",',
    '("sampling_mode","fleurs_fraction","replay_seed","epochs","steps","batch_size","accumulation","replay_fraction",')
replace('sampler.tr.draws != resume["step"]*sampler.tr_per_update', 'sampler.tr_draws != resume["step"]*sampler.tr_per_update')
replace('              "sampling":f"without-replacement per-stream shuffled cycles, reservoir{args.bucket_size} two-dimensional T/U length buckets; replay language uniform per clip",',
    '              "sampling":{"mode":args.sampling_mode,"fleurs_fraction_of_TR_clips":args.fleurs_fraction if args.sampling_mode == "domain30" else None,\n                  "sampler_identity":sampler.identity,"initial_pool":sampler.coverage(),\n                  "replay_language_uniform_per_clip":True,"replay_independent_of_Turkish_rng":True,\n                  "legacy_shared_rng_replay_sequence_equivalence_claim":False,\n                  "epochs_semantics":"minimum domain exposure" if args.sampling_mode == "domain30" else "uniform draw-equivalent",\n                  "all_pool_pass_requires_each_source_seen":True,"same_replay_seed_matches_new_uniform_and_domain_workers":True},\n              "scientific_change":"Turkish domain clip mixture + isolated replay RNG; new Adam and local exposure counters",\n              "closed_source":source_identity,')
replace('        full_epoch = abs(epoch-round(epoch)) <= sampler.tr_per_update/len(tr) + 1e-9',
    '        previous_evaluation_epochs = [row["turkish_epoch"] for row in history[:-1] if "validation" in row]\n        previous_epoch = previous_evaluation_epochs[-1] if previous_evaluation_epochs else 0.\n        full_epoch = math.floor(epoch + 1e-10) > math.floor(previous_epoch + 1e-10)')
replace('"turkish_clips_seen":sampler.tr.draws,"turkish_unique_clips_seen":len(sampler.tr.seen),',
    '"turkish_clips_seen":sampler.tr_draws,"turkish_unique_clips_seen":sampler.tr_unique,\n                  "Turkish_domain_coverage":sampler.coverage(),\n                  "turkish_audio_minutes_per_train_wall_minute":language_audio["tr"]/train_wall,')
replace('                 "latest_weights":str(latest_path),"turkish_epochs_completed":sampler.turkish_epoch,',
    '                 "latest_weights":str(latest_path),"turkish_epochs_completed":sampler.turkish_epoch,\n                 "turkish_epoch_semantics":"minimum_domain_exposure" if args.sampling_mode == "domain30" else "uniform_draw_equivalent",\n                 "Turkish_domain_coverage":sampler.coverage(),"sampling_identity":sampler.identity,')
# Remove legacy migration escape paths: only the NEW scientific identity may exact-resume.
tree = ast.parse(source)
for name, body in (("runtime_migration_evidence", 'def runtime_migration_evidence(args):\n    return None\n'),
                   ("exact_resume_identity_matches", 'def exact_resume_identity_matches(saved, args, current):\n    return args.resume_runtime_migration is None and saved.get("config_identity") == current\n')):
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = source.splitlines(keepends=True)
    source = "".join(lines[:node.lineno-1]) + body + "".join(lines[node.end_lineno:])
    tree = ast.parse(source)
replace('    p.add_argument("--batch-size",type=int,default=2)', '    p.add_argument("--batch-size",type=int,default=8)')
replace('    p.add_argument("--eval-per-epoch",type=float,default=1.); p.add_argument("--eval-every-updates",type=int)',
    '    p.add_argument("--eval-per-epoch",type=float,default=1.); p.add_argument("--eval-every-updates",type=int,default=250)')
replace('    p.add_argument("--head-lr",type=float,default=2e-5)', '    p.add_argument("--head-lr",type=float,default=5e-5)')
replace('    p.add_argument("--new-token-lr-multiplier",type=float,default=15.)', '    p.add_argument("--new-token-lr-multiplier",type=float,default=6.)')
replace('    p.add_argument("--replay-kl",type=float,default=5.)', '    p.add_argument("--replay-kl",type=float,default=10.)')
replace('    p.add_argument("--seed",type=int,default=20261010)',
    '    p.add_argument("--seed",type=int,default=20261010)\n    p.add_argument("--sampling-mode",choices=("domain30","uniform"),default="domain30")\n    p.add_argument("--fleurs-fraction",type=float,default=.3)\n    p.add_argument("--replay-seed",type=int,required=True)\n    p.add_argument("--source-closure",type=Path,required=True)')
replace('    if args.tdt_implementation == "nemo":\n        native_loss_identity(args)\n    return args',
    '    validate_recipe(args)\n    if args.tdt_implementation == "nemo":\n        native_loss_identity(args)\n    return args')
ast.parse(source)
if target.exists():
    raise FileExistsError("Do not overwrite an existing worker")
target.write_text(source, encoding="utf-8", newline="\n")
print(target)
