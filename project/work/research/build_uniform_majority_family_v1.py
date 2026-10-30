"""Source-only construction of NEW inactive majority siblings; no ML imports."""
from pathlib import Path
import ast, hashlib

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'outputs/parakeet_turkish'
def change(text,old,new,count=1):
    if text.count(old)!=count:raise ValueError((old[:110],text.count(old),count))
    return text.replace(old,new)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert digest(HERE/'train_lora_batched_tsc_uniform_full_v1.py')=='b9e550ec145c56a7a75521a997a8a6a33bd98eee7d76c1cb0cabfb116a40631c'
assert digest(HERE/'run_TSC_uniform_full_v1.py')=='dfbac1e473274728f287169a9b3db3edeb03f39046d694c615c026ae7e2ada4f'
w=(HERE/'train_lora_batched_tsc_uniform_full_v1.py').read_text(encoding='utf-8')
w=change(w,w[:w.index('from __future__')],'''"""INACTIVE declared14TR/2replay continuation of a CLOSED uniform frontier.

All207 weights/Adam, actual TR queue suffix, six replay queues/RNG, inherited
history/failure/evaluation clocks carry. Initial350 reproduces sourceH exactly,
including failed gates. Original uniform origin6500 and PT grace are retained.
Only quota/retention-density coefficients and remaining-pass horizon change.
"""
''')
w=change(w,'from tsc_uniform_full_transition_v1 import (preflight as uniform_preflight,\n    load_validate_source, compare_source_development, validate_portuguese_inputs,\n    proof_path, uniform_declaration, audit_source_optimizer, record as file_record)\nfrom tsc_uniform_full_sampler_v1 import TSCUniformFullSampler',
'''from tsc_uniform_majority_transition_v1 import (preflight as majority_preflight,
    audit_parent_state, compare_source_development, validate_portuguese_inputs,
    proof_path, majority_declaration)
from tsc_uniform_full_transition_v1 import audit_source_optimizer, record as file_record
from tsc_uniform_majority_sampler_v1 import TSCUniformMajoritySampler''')
w=change(w,'"TSC_uniform_full_207_state_transition_v1"','"TSC_uniform_majority_207_state_transition_v1"')
w=change(w,'"declared": uniform_declaration(args.portuguese_seed)',
    '"declared": majority_declaration(context["source_step"],context["prior_uniform_draws"])')
w=change(w,'"fresh_optimizer_or_warmup": False, "Portuguese_replay_and_six_retention_languages": True',
    '"fresh_optimizer_or_warmup": False, "Portuguese_grace_reset":False, "source_failed_gates_preserved":True')
w=change(w,'not source_step <= step <= 38225','not source_step <= step <= context["configured_end_step"]')
w=change(w,'not source_step<selected<=step','not context["uniform_origin_step"]<selected<=step')
w=change(w,'Own exact state changed inherited original6500 history','Own exact state changed inherited CLOSED frontier history')
w=change(w,'def load_indexed_pool(args, processor, run_dir, heldout_paths):','def load_indexed_pool(args, processor, run_dir, heldout_paths, context):')
w=change(w,'''        tr, audit = prepare_approved_TR_rows(args, processor, args.indexed_tr_directory,
                                             cache_size=256, heldout_paths=heldout_paths)
        audit["identity"].update(uniform_full_pass_transition=True,
            target_replay_rng_independent_of_turkish=True, target_replay_seed=args.replay_seed,
            Portuguese_replay_pool_added_explicitly=True, replay_language_balance_count=6,
            old_five_future_replay_sequence_claimed=False)
        write_json(audit_path, audit)''',
'''        # Reopen the complete source-admitted immutable carrier. This validates
        # its bytes/provenance/input contract without rebuilding253798 targets.
        audit=copy.deepcopy(context["indexed_audit"])
        tr,audit=reopen_approved_TR_rows(args,args.indexed_tr_directory,audit,cache_size=256)
        write_json(audit_path,audit)''')
w=change(w,'context=uniform_preflight(args)','context=majority_preflight(args)')
w=change(w,'load_indexed_pool(args,processor,run_dir,heldout_paths)','load_indexed_pool(args,processor,run_dir,heldout_paths,context)')
w=change(w,'source=torch.load(args.resume_uniform_full,map_location="cpu",weights_only=True)','source=torch.load(args.resume_majority,map_location="cpu",weights_only=True)')
w=change(w,'"after_source6500_CPU_state_load"','"after_CLOSED_frontier_CPU_state_load"')
w=change(w,'load_validate_source(model,source,context,torch_module=torch)','audit_parent_state(model,source,context,torch_module=torch)')
w=change(w,'sampler=TSCUniformFullSampler(','sampler=TSCUniformMajoritySampler(')
w=change(w,'sampler.migrate_source_state(source["sampler"],context["parent_identity"]["domain_sampling_identity"])','sampler.migrate_closed_state(source["sampler"],context["parent_identity"]["domain_sampling_identity"])')
w=change(w,'''    source_step=context["source_step"];configured_steps=source_step+sampler.uniform_pass_updates
    if source_step!=6500 or configured_steps!=38225 or sampler.uniform_pass_updates!=31725:raise ValueError("One complete253798 new uniform pass required")''',
'''    source_step=context["source_step"];uniform_origin=context["uniform_origin_step"]
    configured_steps=sampler.configured_end_step
    if (uniform_origin!=6500 or configured_steps!=context["configured_end_step"]
            or sampler.remaining_update_budget!=context["remaining_update_budget"]
            or sampler.quota_transition["prior_uniform_draws"]!=context["prior_uniform_draws"]):
        raise ValueError("Declared same-pass14/2 remaining frontier/horizon changed")
    source_best_checkpoint=context.get("source_best_checkpoint")
    parked_original_champion=context["parked_original_champion"]''')
w=change(w,'"after_original300_and_Portuguese50_source_DEV"','"after_actual_source_H_fixed350_DEV"')
for old,new in [('run_TSC_uniform_full_v1.py','run_TSC_uniform_majority_v1.py'),('tsc_uniform_full_sampler_v1.py','tsc_uniform_majority_sampler_v1.py')]:
    w=w.replace(old,new)
# Only NEW sources are added; the frozen original helper remains imported for
# the hardened named Adam audit and lineage input contracts.
w=w.replace('"tsc_uniform_full_transition_v1.py"','"tsc_uniform_majority_transition_v1.py"')
w=change(w,'input_paths=[args.uniform_full_source_receipt,proof_path(args)',
    'input_paths=[args.majority_source_receipt,args.uniform_full_source_receipt,proof_path(args)')
w=change(w,'input_sha={str(path.resolve()):file_sha(path) for path in input_paths}',
    'input_sha={**context["parent_identity"]["input_sha256"],**{str(path.resolve()):file_sha(path) for path in input_paths}}')
w=change(w,'config_identity={"public_turkish_data_policy"',
    'config_identity={**context["parent_identity"],"public_turkish_data_policy"')
w=change(w,'"uniform_full_transition":transition_identity(args,context,sampler_transition)',
    '"majority_transition":transition_identity(args,context,sampler_transition)')
w=change(w,'next_eval=resume["next_eval_epoch"] if args.resume_exact else next_evaluation_epoch(0.,args.eval_epoch_fraction,args.eval_per_epoch)',
    'next_eval=resume["next_eval_epoch"]')
w=change(w,'''    if not args.resume_exact:
        # The old-five champion stays parked. Actual Portuguese96/1095 is
        # measured honestly; there is no eligible all-six BEST at startup.
        best, selected_step = math.inf, None
''','')
w=change(w,'''config={**arguments,"steps":configured_steps,"source_step":source_step,"new_uniform_pass_updates":31725,
        "phase":"HF_TSC_8199_uniform_fullpass_with_Portuguese_v1","evaluation_languages":list(LANGUAGES),
        "uniform_full_pass":{"source_step":source_step,"additional_update_budget":31725,"end_step":38225,
            "expected_new_unique_TR_clips":253798,"original_domain_epoch_is_lineage_only":True},
        "method":"original8199 full207 head+encoderLoRA best recipe; new fulluniformTR and approvedPortuguese replay",''',
'''config={**arguments,"steps":configured_steps,"source_step":source_step,"uniform_origin_step":uniform_origin,
        "new_uniform_pass_updates":sampler.remaining_update_budget,
        "phase":"HF_TSC_8199_uniform_Turkish_majority_fullpass_v1","evaluation_languages":list(LANGUAGES),
        "uniform_full_pass":{"source_step":uniform_origin,"transition_source_step":source_step,
            "additional_update_budget":sampler.remaining_update_budget,"end_step":configured_steps,
            "expected_new_unique_TR_clips":253798,"original_domain_epoch_is_lineage_only":True,
            "prior_uniform_draws":context["prior_uniform_draws"],"quota_transition":sampler.quota_transition},
        "method":"Full207 state-preserving existing uniform-pass14TR/2replay allocation; retention-density40/2/4",''')
w=change(w,'"source_champion_parked":context["source_checkpoint"]',
    '"source_champion_parked":parked_original_champion,"inherited_parent_all6_BEST":source_best_checkpoint')
w=change(w,'"original_base_v3_summary_unchanged":True,"new_uniform_epoch_clock":True',
    '"original_base_v3_summary_unchanged":True,"new_uniform_epoch_clock":False')
w=change(w,'"uniform_sampler_transition":sampler_transition',
    '"uniform_sampler_transition":sampler_transition,"retention_density_normalization":majority_declaration(source_step,context["prior_uniform_draws"]),"source_next_eval_epoch_preserved":next_eval')
w=change(w,'if selected_step is not None:save(best_path,start_step,"fulluniform_declared_source6500")',
    'if selected_step is not None and selected_step==start_step:save(best_path,start_step,"majority_SOURCE_H_is_actual_all6_BEST")')
w=change(w,'from tsc_uniform_full_transition_v1 import release_admission_caches',
    'from tsc_uniform_majority_transition_v1 import release_admission_caches')
w=change(w,'local_updates = step-source_step','local_updates = step-uniform_origin')
w=change(w,'"Turkish_domain_coverage":sampler.coverage(),"uniform_full_updates":step-source_step,',
    '"Turkish_domain_coverage":sampler.coverage(),"uniform_full_updates":step-uniform_origin,\n                  "majority_updates":step-source_step,"majority_source_step":source_step,"uniform_origin_step":uniform_origin,')
w=change(w,'"weights":str(best_path if selected_step is not None else latest_path),',
    '"weights":str((best_path if best_path.exists() else Path(source_best_checkpoint["path"])) if selected_step is not None else latest_path),\n                 "selected_checkpoint_role":"own_majority_all6_BEST" if selected_step is not None and selected_step>source_step else "inherited_parent_all6_BEST" if selected_step is not None else "unqualified_current_latest",')
w=change(w,'"source_step":source_step,"uniform_full_updates":completed-source_step,',
    '"source_step":source_step,"uniform_origin_step":uniform_origin,"uniform_full_updates":completed-uniform_origin,\n                 "majority_updates":completed-source_step,')
w=w.replace('TSC_uniform_full_fresh_preload_v1','TSC_uniform_majority_fresh_preload_v1')
w=change(w,'p.add_argument("--replay-fraction",type=float,default=.5)','p.add_argument("--replay-fraction",type=float,default=.125)')
w=change(w,'p.add_argument("--replay-kl",type=float,default=10.)','p.add_argument("--replay-kl",type=float,default=40.)')
w=change(w,'p.add_argument("--duration-kl",type=float,default=.5)','p.add_argument("--duration-kl",type=float,default=2.)')
w=change(w,'p.add_argument("--feature-distillation",type=float,default=1.)','p.add_argument("--feature-distillation",type=float,default=4.)')
w=change(w,'p.add_argument("--resume-uniform-full",type=Path,required=True)',
    'p.add_argument("--resume-uniform-full",type=Path,required=True)\n    p.add_argument("--resume-majority",type=Path,required=True)\n    p.add_argument("--majority-source-receipt",type=Path,required=True)')
w=change(w,'args.replay_kl!=10 or args.schedule!="constant"',
    'args.replay_fraction!=.125 or args.replay_kl!=40 or args.duration_kl!=2 or args.feature_distillation!=4 or args.schedule!="constant"')
w=change(w,'Original qualified full207 B8x2/KD10 constant recipe required',
    'Declared full207 B8x2/14TR2RP/KD40-duration2-feature4 constant recipe required')
w=change(w,'    uniform_preflight(args)','    majority_preflight(args)')
ast.parse(w)
dest=HERE/'train_lora_batched_tsc_uniform_majority_v1.py'
if dest.exists():raise ValueError('NEW worker already exists')
dest.write_text(w,encoding='utf-8',newline='\n')
print(digest(dest))

