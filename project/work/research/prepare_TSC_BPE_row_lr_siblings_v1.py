"""Create NEW inactive row-LR challenger siblings. No ML/data execution."""
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]/'outputs/parakeet_turkish'
def rep(text,a,b):
    if a not in text:raise ValueError('Missing predecessor: '+a[:80])
    return text.replace(a,b)
def write(name,text):
    p=HERE/name
    if p.exists():raise ValueError('New sibling already exists')
    p.write_text(text,encoding='utf-8')

h=(HERE/'tsc_bpe_lora_transition_v2.py').read_text(encoding='utf-8')
h=rep(h,'PHASE = "HF_TSC_BPE512_new_rows_plus_encoder_LoRA_v1"','PHASE = "HF_TSC_BPE512_new_rows_continued_acquisition_v1"')
h=rep(h,'STATE_KIND = "HF_TSC_BPE512_new_rows_plus_encoder_LoRA_state_v1"','STATE_KIND = "HF_TSC_BPE512_new_rows_continued_acquisition_state_v1"')
h=rep(h,'RECEIPT_KIND = "ROOT_owned_closed_TSC_BPE_phase1_for_LoRA_v1"','RECEIPT_KIND = "ROOT_owned_closed_TSC_BPE_phase1_for_row_LR_v1"')
h=rep(h,'WORKER = "train_lora_batched_tsc_bpe_lora_native4_v2.py"','WORKER = "train_lora_batched_tsc_bpe_row_lr_v1.py"')
h=rep(h,'OUTER = "run_TSC_bpe_lora_native4_v2.py"','OUTER = "run_TSC_bpe_row_lr_v1.py"')
h=rep(h,'PROOF_NAME = "TSC_BPE_LoRA_initial_source_equivalence.json"','PROOF_NAME = "TSC_BPE_row_LR_initial_source_equivalence.json"')
h=rep(h,'PROOF_KIND = "TSC_BPE512_LoRA_before_update_source_equivalence_v1"','PROOF_KIND = "TSC_BPE512_row_LR_before_update_source_equivalence_v1"')
h=rep(h,'    "bpe_newrows_lora_adaptation_v1.py": "2f70811d67af5fc950fa547727dc5ae18fa895a1e5e2cb572c4ff40bbc42476b",','    "bpe_newrows_continued_acquisition_v1.py": "PENDING",')
begin=h.index('def adaptation_declaration():')
end=h.index('\ndef original_pointer(',begin)
h=h[:begin]+'''def continued_acquisition_declaration(active_row_lr):
    require(active_row_lr in (1e-3, 3e-4), "Only declared rowLR1e-3 or3e-4")
    return {"phase": PHASE, "active_row_lr": active_row_lr, "source_row_lr": 1e-4,
        "encoder_LoRA_active": False, "shared204_frozen": True, "weight_decay": 0.,
        "active_table_names": ["decoder.embedding.weight", "joint.head.weight", "joint.head.bias"],
        "effective_new_rows": [8199, 8711], "source_original207_retained_dormant": True,
        "phase1_three_table_Adam_moments_and_clock_retained": True,
        "phase1_acquired_rows_retained": True, "donor_reinitialization_used": False,
        "single_global3_clip_norm": 1., "six_character_delta_multiplier_applied": False,
        "native_max_joint_bytes": NATIVE_MAX_BYTES,
        "phase1_sampler_RNG_history_exposure_counters_preserved": True,
        "new_sampler_target_fingerprint_migration": False, "source_phase2_latest_used": False,
        "source_LR_audited_before_moment_copy": True, "original207_exact_resume_claimed": False,
        "external_phase_exact_resume_supported": False, "unfreeze_allowed": False}


def validate_receipt(receipt, target_run_id, active_row_lr=None):
    if active_row_lr is None:
        active_row_lr = receipt.get("declared_row_lr", {}).get("active_row_lr")
    require(receipt.get("schema") == 1 and receipt.get("kind") == RECEIPT_KIND
        and receipt.get("status") == "actual_CLOSED_qualified_phase1"
        and receipt.get("target_run_id") == target_run_id
        and receipt.get("allow_continued_row_only_acquisition") is True
        and receipt.get("source_phase2_latest_used") is False
        and receipt.get("declared_row_lr") == continued_acquisition_declaration(active_row_lr),
        "Explicit ROOT CLOSED phase1 row-LR approval required")
    source_run = receipt.get("source_run_id")
    require(isinstance(source_run, str) and isinstance(target_run_id, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,110}", source_run) is not None
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,110}", target_run_id) is not None
        and target_run_id != source_run, "Distinct safe phase1/new run IDs required")
    values = [receipt.get(key) for key in ("original_step", "acquisition_updates", "chosen_step", "completed_steps")]
    require(all(type(value) is int and value > 0 for value in values)
        and values[2] == values[0] + values[1] and values[2] <= values[3],
        "Positive acquired K and originalS+K frontier required")
    require(receipt.get("chosen_checkpoint_role") in ("latest", "best")
        and receipt.get("retention_gate_passed_at_chosen_step") is True,
        "Chosen acquired state must have its own qualifying fixedDEV")
    exits = receipt.get("owned_actual_exits", [])
    require(any(item.get("role") == "trainer_controller" for item in exits)
        and all(item.get("exit_code") == 0 and (item.get("chunk_id") or item.get("tool_chunk"))
                and item.get("session_id") and item.get("observed_at_utc") for item in exits),
        "Actual ROOT-owned CLOSED0 evidence required")
    return receipt

''' + h[end:]
h=rep(h,'if name != "bpe_newrows_lora_adaptation_v1.py"','if name != "bpe_newrows_continued_acquisition_v1.py"')
h=h.replace('getattr(args, "resume_adaptation", None)','getattr(args, "resume_row_lr", None)')
h=h.replace('args.adaptation_source_receipt','args.row_lr_source_receipt').replace('args.resume_adaptation','args.resume_row_lr')
h=h.replace('args.adaptation_max_updates','args.acquisition_max_updates')
h=rep(h,'require(args.native_max_joint_bytes == NATIVE_MAX_BYTES','require(args.active_row_lr == context["receipt"]["declared_row_lr"]["active_row_lr"]\n        and args.native_max_joint_bytes == NATIVE_MAX_BYTES')
h=h.replace('<= 10000','<= 1000')
start=h.index('    require(sha(HERE / "bpe_newrows_lora_adaptation_v1.py")')
end=h.index('\n\ndef proof_path(',start)
h=h[:start]+'''    require(sha(HERE / "bpe_newrows_continued_acquisition_v1.py") == SOURCE_PINS["bpe_newrows_continued_acquisition_v1.py"],
            "Reviewed continued-row utility changed")
    from bpe_newrows_continued_acquisition_v1 import BPEContinuedAcquisition
    acquisition = BPEContinuedAcquisition(model, phase1_state, original207,
        original_source_record=context["receipt"]["artifacts"]["original_source_checkpoint"],
        phase1_record=context["receipt"]["artifacts"]["chosen_checkpoint"],
        active_row_lr=context["receipt"]["declared_row_lr"]["active_row_lr"], torch_module=torch)
    require(acquisition.cumulative_step == context["source_step"] and acquisition.optimizer_updates_completed == 0,
            "Construction must not advance acquired Adam/runtime")
    return {"acquisition": acquisition, "vocabulary": expanded["vocabulary"],
        "boundary": {**boundary, "original207_audit": expanded["boundary"],
            "actual_acquired_table_learning": learned, "continued_row_identity": acquisition.phase_identity()},
        "expansion_evidence": expanded["expansion_evidence"], "inherited_runtime": acquisition.inherited_runtime}
''' + h[end:]
h=h.replace('"before_phase2_updates": True','"before_row_lr_updates": True')
h=h.replace('"source_receipt": context["receipt_record"], "source_selected_step":','"source_receipt": context["receipt_record"], "declared_row_lr": context["receipt"]["declared_row_lr"], "source_selected_step":')
write('tsc_bpe_row_lr_transition_v1.py',h)

w=(HERE/'train_lora_batched_tsc_bpe_acquire_v1.py').read_text(encoding='utf-8')
w=w.replace('native_tdt_checkpointed_v3','native_tdt_checkpointed_v4').replace('run_TSC_bpe_acquire_v1.py','run_TSC_bpe_row_lr_v1.py')
w=rep(w,'from tsc_bpe_acquisition_transition_v1 import (preflight as acquisition_preflight,\n    load_validate_expand, initial_source_equivalence)','from tsc_bpe_row_lr_transition_v1 import (preflight as acquisition_preflight,\n    load_validate_phase1, initial_source_equivalence)')
w=rep(w,'from bpe_target_sampler_transition_v1 import prepare_sampler_state\nfrom bpe_newrows_acquisition_v1 import BPEOnlyAcquisition, artifact_record','from bpe_newrows_continued_acquisition_v1 import BPEContinuedAcquisition\nfrom bpe_newrows_acquisition_v1 import artifact_record')
w=w.replace('PHASE = "HF_TSC_BPE512_new_rows_only_acquisition_v1"','PHASE = "HF_TSC_BPE512_new_rows_continued_acquisition_v1"')
w=w.replace('TSC_BPE_acquisition_initial_source_equivalence.json','TSC_BPE_row_LR_initial_source_equivalence.json')
w=w.replace('args.resume_acquisition','args.resume_row_lr').replace('args.acquisition_source_receipt','args.row_lr_source_receipt')
w=w.replace('args.acquisition_lr','args.active_row_lr')
w=rep(w,'and evidence.get("geometry") == GEOMETRY,','and evidence.get("geometry") == GEOMETRY\n        and evidence.get("active_row_lr") == args.active_row_lr\n        and evidence.get("source_row_lr") == 1e-4,')
w=rep(w,'    source = torch.load(args.resume_row_lr, map_location="cpu", weights_only=True)\n    expanded = load_validate_expand(model, processor, source, context, torch_module=torch)','    original_record = context["receipt"]["artifacts"]["original_source_checkpoint"]\n    original = torch.load(original_record["path"], map_location="cpu", weights_only=True)\n    source = torch.load(args.resume_row_lr, map_location="cpu", weights_only=True)\n    expanded = load_validate_phase1(model, processor, original, source, context, torch_module=torch)\n    acquisition = expanded["acquisition"]\n    runtime = expanded["inherited_runtime"]')
w=rep(w,'    migrated, sampler_transition = prepare_sampler_state(source["sampler"], context["config"], sampler.identity, consumer_acceptance)\n    sampler.load_state_dict(migrated)\n    start_step = source["step"]','    sampler_transition = "none; exact acquired phase1 targets and queues"\n    sampler.load_state_dict(runtime["sampler"])\n    start_step = context["source_selected_step"]\n    require(start_step == acquisition.cumulative_step and source["step"] == start_step, "Acquired frontier differs")')
begin=w.index('    source_names = tuple(source["weights"])')
end=w.index('    source_sha = ',begin)
w=w[:begin]+'    parked = original_record\n'+w[end:]
w=w.replace('source["python_rng"]','runtime["python_rng"]').replace('source["torch_rng"]','runtime["torch_rng"]').replace('source["cuda_rng"]','runtime["cuda_rng"]')
w=w.replace('"tsc_bpe_acquisition_transition_v1.py",','"tsc_bpe_row_lr_transition_v1.py",')
w=w.replace('"bpe_newrows_acquisition_v1.py", "bpe_target_sampler_transition_v1.py",','"bpe_newrows_acquisition_v1.py", "bpe_newrows_continued_acquisition_v1.py", "native_tdt_checkpointed_v4.py",')
for key in ('history','patience','retention_failures','next_eval_epoch','initial_gate','train_wall_seconds','evaluation_wall_seconds','audio_seconds','language_audio_seconds'):
    w=w.replace('source["'+key+'"]','runtime["'+key+'"]')
w=rep(w,'    best, selected_step = source["best_score"], source["selected_step"]','    best, selected_step = context["chosen_gate"]["turkish_wer"], start_step')
w=rep(w,'    del source, migrated, baseline_processor','    del original, source, runtime, baseline_processor\n    acquisition.inherited_runtime = None')
w=rep(w,'"acquisition_lr": args.active_row_lr, "max_updates":','"acquisition_lr": args.active_row_lr, "source_row_lr": 1e-4, "source_acquisition_updates": acquisition.source_acquisition_updates, "max_updates":')
w=rep(w,'"acquisition_updates_completed_initial": 0,','"acquisition_updates_completed_initial": 0, "source_acquisition_updates": acquisition.source_acquisition_updates,\n        "original_source_selected_step": acquisition.source_step,')
w=rep(w,'"fresh_optimizer": "three full tables; active512rows only; AdamW1e-4/decay0/constant/no6boost",','"fresh_optimizer": False, "continued_optimizer": "acquired three-table AdamK retained; declared active rowLR; no6boost",')
w=rep(w,'        record = {"step": step, "acquisition_updates_completed": acquisition.optimizer_updates_completed,','        record = {"step": step, "acquisition_updates_completed": acquisition.optimizer_updates_completed,\n            "source_acquisition_updates": acquisition.source_acquisition_updates,\n            "row_Adam_clock": acquisition.source_acquisition_updates + acquisition.optimizer_updates_completed,')
w=rep(w,'"acquisition_lr": args.active_row_lr, "head_lr": 0., "lora_lr": 0.,','"acquisition_lr": args.active_row_lr, "active_row_lr": args.active_row_lr, "source_row_lr": 1e-4, "head_lr": 0., "lora_lr": 0.,')
w=rep(w,'"inherited_history_steps": start_step, "acquisition_updates_completed": acquisition.optimizer_updates_completed,','"inherited_history_steps": start_step, "acquisition_updates_completed": acquisition.optimizer_updates_completed,\n        "original_source_selected_step": acquisition.source_step, "source_acquisition_updates": acquisition.source_acquisition_updates,\n        "row_Adam_clock": acquisition.source_acquisition_updates + acquisition.optimizer_updates_completed,')
w=w.replace('"--resume-acquisition"','"--resume-row-lr"').replace('"--acquisition-source-receipt"','"--row-lr-source-receipt"')
w=w.replace('p.add_argument("--acquisition-lr", type=float, default=1e-4)','p.add_argument("--active-row-lr", type=float, choices=(1e-3, 3e-4), default=1e-3)')
w=w.replace('p.add_argument("--acquisition-max-updates", type=int, default=500)','p.add_argument("--acquisition-max-updates", type=int, default=1000)')
w=w.replace('args.active_row_lr == 1e-4','args.active_row_lr in (1e-3, 3e-4)').replace('<= 10000','<= 1000')
w=rep(w,'    runtime_state = {"run_id": run_id, "step": step, "source_selected_step": source_selected_step,','    require(step == acquisition.cumulative_step and sampler.updates == step and len(history) == step, "Durable row continuation frontier differs")\n    runtime_state = {"run_id": run_id, "step": step, "source_selected_step": source_selected_step,\n        "source_acquisition_updates": acquisition.source_acquisition_updates,\n        "original_source_selected_step": acquisition.source_step,\n        "acquisition_updates_completed": acquisition.optimizer_updates_completed,')
w=w.replace('target identities, pending orders and cursors carry exactly.','target identities, pending orders and cursors carry exactly.')
w=w.replace('Original replay pool, independent RNG, pending orders and cursors carry from selected BEST. TR target fingerprint changes; future target-length bucket sorts are declared, not old-recipe ASR continuation.','Original replay pool, independent RNG, acquired phase1 pending orders/targets/cursors carry exactly; no target migration.')
write('train_lora_batched_tsc_bpe_row_lr_v1.py',w)

c=(HERE/'run_TSC_bpe_acquire_v1.py').read_text(encoding='utf-8')
c=c.replace('train_lora_batched_tsc_bpe_acquire_v1.py','train_lora_batched_tsc_bpe_row_lr_v1.py').replace('tsc_bpe_acquisition_transition_v1.py','tsc_bpe_row_lr_transition_v1.py')
c=rep(c,"WORKER_SHA = '82faecd3c19762b182d9bff1bac9a61317bd99ceb6327db4356bba3b2587cd21'","WORKER_SHA = 'PENDING'")
c=rep(c,"TRANSITION_SHA = '0c888b2bc1cdbde8fc7c8062a7c6bd6a404d6aa09b62050b93942ec16705d038'","TRANSITION_SHA = 'PENDING'")
c=rep(c,'EXPECTED_SYNC_LINE = 459','EXPECTED_SYNC_LINE = 0')
c=c.replace("PHASE = 'HF_TSC_BPE512_new_rows_only_acquisition_v1'","PHASE = 'HF_TSC_BPE512_new_rows_continued_acquisition_v1'")
c=c.replace("STATE_KIND = 'HF_TSC_BPE512_new_rows_only_acquisition_state_v1'","STATE_KIND = 'HF_TSC_BPE512_new_rows_continued_acquisition_state_v1'")
c=c.replace('guarded_TSC_BPE512_new_rows_only_acquisition_controller_v1','guarded_TSC_BPE512_continued_row_LR_controller_v1')
c=c.replace('TSC_BPE_acquisition_initial_source_equivalence.json','TSC_BPE_row_LR_initial_source_equivalence.json').replace('TSC_BPE512_acquisition_before_update_source_equivalence_v1','TSC_BPE512_row_LR_before_update_source_equivalence_v1')
c=rep(c,"    'bpe_newrows_acquisition_v1.py': 'bc26638f4874aaa5e92029e86c30bef0624a477514957d8f8d054d7e92088724',","    'bpe_newrows_acquisition_v1.py': 'bc26638f4874aaa5e92029e86c30bef0624a477514957d8f8d054d7e92088724',\n    'bpe_newrows_continued_acquisition_v1.py': 'PENDING',")
c=rep(c,"    'native_tdt_checkpointed_v3.py': '839194dddf83c3b08bf608d01cb540b2f160feee426a91373d14cfc426ab7494',","    'native_tdt_checkpointed_v3.py': '839194dddf83c3b08bf608d01cb540b2f160feee426a91373d14cfc426ab7494',\n    'native_tdt_checkpointed_v4.py': '24b9cf5b8f3e11d5b043c22d0940da3735a46f8f70b6f5810b97f17d545cd90e',")
c=c.replace('native_tdt_checkpointed_v3 import','native_tdt_checkpointed_v4 import')
c=c.replace("HERE / 'native_tdt_checkpointed_v3.py'","HERE / 'native_tdt_checkpointed_v4.py'")
c=c.replace("float(option(argv, '--acquisition-lr')) == 1e-4","float(option(argv, '--active-row-lr')) in (1e-3, 3e-4)")
c=c.replace("'--resume-acquisition', '--acquisition-source-receipt',","'--resume-acquisition', '--acquisition-source-receipt', '--resume-row-lr', '--row-lr-source-receipt',")
c=c.replace("'--acquisition-lr',","'--acquisition-lr', '--active-row-lr',")
# Previous replacement changes calls as well; correct only the explicit new route below.
c=c.replace("option(argv, '--acquisition-source-receipt')","option(argv, '--row-lr-source-receipt')")
c=c.replace("resume_acquisition=scoped(option(argv, '--resume-acquisition'))","resume_row_lr=scoped(option(argv, '--resume-row-lr'))")
c=c.replace("acquisition_source_receipt=scoped(option(argv, '--row-lr-source-receipt'))","row_lr_source_receipt=scoped(option(argv, '--row-lr-source-receipt'))")
c=c.replace("acquisition_lr=float(option(argv, '--acquisition-lr'))","active_row_lr=float(option(argv, '--active-row-lr'))")
c=c.replace("context['receipt']['artifacts']['best_checkpoint']","context['receipt']['artifacts']['chosen_checkpoint']")
c=c.replace("best = scoped(receipt['artifacts']['best_checkpoint']['path'])","best = scoped(receipt['artifacts']['chosen_checkpoint']['path'])")
c=c.replace("('--resume-acquisition', best)","('--resume-row-lr', best)")
c=c.replace("('--acquisition-source-receipt', receipt_path), ('--acquisition-lr', '--active-row-lr', 1e-4)","('--row-lr-source-receipt', receipt_path), ('--active-row-lr', args.active_row_lr)")
c=c.replace("'declared_acquisition': importlib.import_module(Path(TRANSITION).stem).acquisition_declaration(),","'declared_acquisition': importlib.import_module(Path(TRANSITION).stem).continued_acquisition_declaration(args.active_row_lr),\n        'active_row_lr': args.active_row_lr, 'source_row_lr': 1e-4,\n        'source_acquisition_updates': context['acquisition_updates'], 'original_source_selected_step': context['original_step'],")
c=c.replace("'parent_Adam_loaded_into_active_optimizer': False","'parent_Adam_loaded_into_active_optimizer': True")
c=c.replace("'acquisition_lr': 1e-4, 'max_updates':","'acquisition_lr': plan['active_row_lr'], 'source_row_lr': 1e-4,\n        'source_acquisition_updates': plan['source_acquisition_updates'], 'max_updates':")
c=c.replace("proof.get('before_optimizer_creation') is True","proof.get('before_row_lr_updates') is True")
c=c.replace("result.add_argument('--acquisition-max-updates', type=int, default=500)","result.add_argument('--active-row-lr', type=float, choices=(1e-3, 3e-4), default=1e-3)\n    result.add_argument('--acquisition-max-updates', type=int, default=1000)")
c=c.replace('<= 10000','<= 1000')
c=c.replace("'default_acquisition_max_updates': 500","'default_acquisition_max_updates': 1000")
c=rep(c,"'acquisition_phase': PHASE, 'geometry': GEOMETRY,","'acquisition_phase': PHASE, 'geometry': GEOMETRY,\n        'active_row_lr': plan['active_row_lr'], 'source_row_lr': 1e-4,")
write('run_TSC_bpe_row_lr_v1.py',c)
print('Created three inactive row-LR draft siblings; source pins remain pending.')
