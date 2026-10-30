"""Create NEW inactive shared12+row3 siblings; source-only, no ML/data reads."""
from pathlib import Path
import ast
import hashlib

HERE=Path(__file__).resolve().parents[2]/'outputs/parakeet_turkish'
UTILITY_SHA='384d01abe9b7105761491b8acda5d3de4eb8c40ae679eee2701269079cf9967c'
def rep(text,a,b):
    if a not in text:raise ValueError('Missing predecessor: '+a[:100])
    return text.replace(a,b)
def write(name,text):
    p=HERE/name
    if p.exists():
        if p.read_text(encoding='utf-8') == text:return hashlib.sha256(p.read_bytes()).hexdigest()
        raise ValueError('New sibling exists with different bytes:'+name)
    ast.parse(text);p.write_text(text,encoding='utf-8')
    return hashlib.sha256(p.read_bytes()).hexdigest()
def common(t):
    for a,b in (
        ('tsc_bpe_lora_transition_v2','tsc_bpe_shared_head_transition_v1'),
        ('train_lora_batched_tsc_bpe_lora_native4_v2','train_lora_batched_tsc_bpe_shared_head_v1'),
        ('run_TSC_bpe_lora_native4_v2','run_TSC_bpe_shared_head_v1'),
        ('bpe_newrows_lora_adaptation_v1','bpe_newrows_shared_head_adaptation_v1'),
        ('BPENewRowsLoRAAdaptation','BPENewRowsSharedHeadAdaptation'),
        ('HF_TSC_BPE512_new_rows_plus_encoder_LoRA','HF_TSC_BPE512_new_rows_plus_shared_heads'),
        ('ROOT_owned_closed_TSC_BPE_phase1_for_LoRA_v1','ROOT_owned_closed_TSC_BPE_phase1_for_shared_heads_v1'),
        ('TSC_BPE_LoRA_initial_source_equivalence.json','TSC_BPE_shared_head_initial_source_equivalence.json'),
        ('TSC_BPE512_LoRA_before_update_source_equivalence_v1','TSC_BPE512_shared_heads_before_update_source_equivalence_v1'),
        ('before_phase2_updates','before_shared_head_updates'),
        ('resume_adaptation','resume_shared_heads'),('adaptation_source_receipt','shared_head_source_receipt'),
        ('--resume-adaptation','--resume-shared-heads'),('--adaptation-source-receipt','--shared-head-source-receipt'),
    ):t=t.replace(a,b)
    return t

h=common((HERE/'tsc_bpe_lora_transition_v2.py').read_text(encoding='utf-8'))
h=rep(h,'"2f70811d67af5fc950fa547727dc5ae18fa895a1e5e2cb572c4ff40bbc42476b"','"'+UTILITY_SHA+'"')
a=h.index('def adaptation_declaration():');b=h.index('\ndef validate_receipt(',a)
h=h[:a]+'''def shared_head_declaration(row_lr, shared_lr):
    require(row_lr in (1e-3, 3e-4) and shared_lr in (1e-5, 3e-6), "Only declared row/shared learning rates")
    return {"phase": PHASE, "row_lr": row_lr, "shared_lr": shared_lr,
        "source_row_lr": 1e-4, "source_shared_lr": 1e-5,
        "active_parameter_tensors": 15, "active_shared_head_tensors": 12,
        "active_table_names": ["decoder.embedding.weight", "joint.head.weight", "joint.head.bias"],
        "effective_new_rows": [8199, 8711], "frozen_encoder_LoRA_tensors": 192,
        "encoder_LoRA_active": False, "acoustic_encoder_no_grad": True,
        "encoder_projector_outside_no_grad": True, "shared_decoder_differentiable": True,
        "old_six_blank_duration_table_rows_frozen": True,
        "original12_shared_head_moments_copied_by_name": True,
        "original3_resized_table_moments_active": False, "original192_LoRA_moments_active": False,
        "original207_optimizer_artifact_retained_dormant": True,
        "phase1_three_table_Adam_moments_and_clock_retained": True,
        "phase1_acquired_rows_retained": True, "donor_reinitialization_used": False,
        "source_moments_or_clocks_reset": False,
        "two_distinct_Adam_clocks": True, "single_global15_clip_norm": 1.,
        "row_weight_decay": 0., "shared_source_weight_decay_retained": .01,
        "six_character_delta_multiplier_applied": False, "native_max_joint_bytes": NATIVE_MAX_BYTES,
        "phase1_sampler_RNG_history_exposure_counters_preserved": True,
        "new_sampler_target_fingerprint_migration": False, "source_phase2_latest_used": False,
        "same_science_or_original207_exact_resume_claimed": False,
        "external_phase2_exact_resume_supported": False, "further_unfreeze_allowed": False}

''' + h[b:]
h=rep(h,'def validate_receipt(receipt, target_run_id):','def validate_receipt(receipt, target_run_id, row_lr=None, shared_lr=None):\n    declared = receipt.get("declared_shared_heads", {})\n    if row_lr is None: row_lr = declared.get("row_lr")\n    if shared_lr is None: shared_lr = declared.get("shared_lr")')
h=rep(h,'and receipt.get("allow_new_rows_plus_encoder_LoRA") is True\n        and receipt.get("declared_adaptation") == adaptation_declaration(),',
    'and receipt.get("allow_shared_head_adaptation") is True\n        and receipt.get("source_phase2_latest_used") is False\n        and receipt.get("declared_shared_heads") == shared_head_declaration(row_lr, shared_lr),')
h=rep(h,'require(args.native_max_joint_bytes == NATIVE_MAX_BYTES',
    'require(args.row_lr == context["receipt"]["declared_shared_heads"]["row_lr"]\n        and args.shared_lr == context["receipt"]["declared_shared_heads"]["shared_lr"]\n        and args.native_max_joint_bytes == NATIVE_MAX_BYTES')
h=h.replace('<= 10000','<= 1000')
h=rep(h,'phase1_record=context["receipt"]["artifacts"]["chosen_checkpoint"], torch_module=torch)',
    'phase1_record=context["receipt"]["artifacts"]["chosen_checkpoint"],\n        row_lr=context["receipt"]["declared_shared_heads"]["row_lr"],\n        shared_lr=context["receipt"]["declared_shared_heads"]["shared_lr"], torch_module=torch)')
h=rep(h,'"source_receipt": context["receipt_record"], "source_selected_step":',
    '"source_receipt": context["receipt_record"], "declared_shared_heads": context["receipt"]["declared_shared_heads"], "source_selected_step":')
h=h.replace('separate-clock195','separate-clock15').replace('encoder LoRA adaptation','shared-head adaptation')
helper_sha=write('tsc_bpe_shared_head_transition_v1.py',h)

w=common((HERE/'train_lora_batched_tsc_bpe_lora_native4_v2.py').read_text(encoding='utf-8'))
w=rep(w,'from batched_lora_tr_metadata_v1 import batch_objective, collate_training_batch',
    'from batched_lora_tr_metadata_v1 import BatchObjective, sampled_preservation, collate_training_batch')
w=rep(w,'from train_lora import checkpoint_encoder_blocks\n','')
row=(HERE/'train_lora_batched_tsc_bpe_row_lr_v1.py').read_text(encoding='utf-8')
a=row.index('def batch_objective(');b=row.index('\ndef native_loss_identity(',a)
objective=row[a:b]
objective=rep(objective,'            ep = model.encoder_projector(features)','        ep = model.encoder_projector(features)')
objective=objective.replace('no_grad ONLY around encoder/projector','no_grad ONLY around acoustic encoder')
objective=objective.replace('Frozen decoder/joint parameters remain in the autograd graph so that native\n    TDT and full-vocabulary KD can differentiate candidate embeddings/head rows.',
    'The encoder projector and shared decoder stay differentiable; native TDT\n    and full-vocabulary KD reach all12 shared parameters plus the new table rows.')
w=rep(w,'\ndef native_loss_identity(args):','\n'+objective+'\ndef native_loss_identity(args):')
w=rep(w,'and evidence.get("geometry") == GEOMETRY,',
    'and evidence.get("geometry") == GEOMETRY\n        and evidence.get("row_lr") == args.row_lr and evidence.get("shared_lr") == args.shared_lr\n        and evidence.get("active_parameter_tensors") == 15\n        and evidence.get("frozen_encoder_LoRA_tensors") == 192,')
w=w.replace('adaptation.lora_optimizer','adaptation.shared_optimizer').replace('"lora_optimizer"','"shared_optimizer"')
w=w.replace('adaptation.lora_group','adaptation.shared_group').replace('adaptation.adapters','adaptation.shared').replace('adaptation.adapter_names','adaptation.shared_names')
w=w.replace('smoke_lora_before','smoke_shared_before').replace('nonzero_lora','nonzero_shared').replace('changed_lora','changed_shared')
w=w.replace('LoRA_gradients_finite=192','shared_gradients_finite=12').replace('LoRA_nonzero_gradient_names','shared_nonzero_gradient_names')
w=w.replace('require(nonzero_shared, "Actual differentiable encoder-LoRA gradient absent")',
    'require(len(nonzero_shared) == 12, "Actual differentiable gradient must reach all12 shared heads")')
w=w.replace('require(changed_shared, "Actual encoder-LoRA parameter delta absent")',
    'require(len(changed_shared) == 12, "Actual parameter delta must reach all12 shared heads")')
w=w.replace('"LoRA_Adam_clock"','"shared_Adam_clock"').replace('"lora_clock_S_plus_N"','"shared_clock_S_plus_N"')
w=w.replace('"LoRA192_gradients_finite_and_any_delta_nonzero"','"shared12_gradients_finite_and_all_deltas_nonzero"').replace('"LoRA_changed_names"','"shared_changed_names"')
w=w.replace('"protected_old_six_blank_duration_and_frozen12_rawbytes_exact"','"protected_old_six_blank_duration_and_frozen192_encoder_LoRA_rawbytes_exact"')
w=w.replace('"all195_Adam_moments_clocks_and_groups_exact"','"all15_Adam_moments_clocks_and_groups_exact"')
w=w.replace('TSC_BPE_LoRA_actual_one_update_smoke_v1','TSC_BPE_shared_heads_actual_one_update_smoke_v1')
w=rep(w,'    # All24 attention blocks retain the original nonreentrant checkpoint path.\n    # Base weights remain frozen, but encoder and projector forwards carry gradients.\n    checkpoint_encoder_blocks(model)',
    '    # The acoustic encoder is frozen and executes under no_grad; no encoder activation graph is retained.\n    require(args.no_encoder_checkpoint is False, "Preserve dormant source checkpoint-policy metadata")')
w=rep(w,'"active_parameter_tensors": 195, "row_lr": 1e-4, "lora_lr": 3e-4,',
    '"active_parameter_tensors": 15, "active_shared_head_tensors": 12, "frozen_encoder_LoRA_tensors": 192, "row_lr": args.row_lr, "shared_lr": args.shared_lr,')
w=rep(w,'"old15_head_moments_active": False, "original207_Adam_artifact_retained_dormant": True,',
    '"original12_shared_head_moments_active": True, "original3_resized_table_moments_active": False, "original192_encoder_LoRA_moments_active": False, "original207_Adam_artifact_retained_dormant": True,')
w=w.replace('"source_optimizer": "original207 artifact parked; only original192 named LoRA states copied",',
    '"source_optimizer": "original207 artifact parked; only original12 named shared-head states copied",')
w=w.replace('"active_optimizers": "phase1 three-table AdamK plus copied original192 LoRA AdamS; constant LRs1e-4/3e-4/decay0",',
    '"active_optimizers": "phase1 three-table AdamK plus copied original12 shared-head AdamS; declared constant row/shared LR; source shared decay0.01 retained",')
w=w.replace('"encoder_LoRA_active": True','"encoder_LoRA_active": False').replace('"encoder_activation_checkpointing": True','"encoder_activation_checkpointing": False')
w=w.replace('"encoder_projector_weights_frozen_but_differentiable": True, "shared_decoder_frozen_but_differentiable": True,',
    '"encoder_projector_active_outside_no_grad": True, "shared_decoder_active_and_differentiable": True,')
w=w.replace('"row_lr": adaptation.row_group["lr"], "head_lr": 0., "lora_lr": adaptation.shared_group["lr"],',
    '"row_lr": adaptation.row_group["lr"], "shared_lr": adaptation.shared_group["lr"], "head_lr": adaptation.shared_group["lr"], "lora_lr": 0.,')
w=rep(w,'p.add_argument("--acquisition-lr", type=float, default=1e-4,\n                   help="Inherited row-Adam constant; only1e-4 is admitted")',
    'p.add_argument("--row-lr", type=float, choices=(1e-3, 3e-4), default=1e-3)\n    p.add_argument("--shared-lr", type=float, choices=(1e-5, 3e-6), default=1e-5)')
w=rep(w,'require(args.acquisition_lr == 1e-4 and type(args.adaptation_max_updates) is int',
    'require(args.row_lr in (1e-3, 3e-4) and args.shared_lr in (1e-5, 3e-6) and type(args.adaptation_max_updates) is int')
w=w.replace('<= 10000','<= 1000')
w=w.replace('new-row plus encoder-LoRA adaptation worker','new-row plus shared-head adaptation worker')
w=w.replace('rows and192 encoder LoRA tensors update','rows and12 shared-head tensors update').replace('original15\nhead moments remain dormant','original192 encoder-LoRA moments remain dormant')
w=w.replace('original192 encoder-LoRA phase','original12 shared-head phase').replace('independent3/192 Adam','independent3/12 Adam')
w=w.replace('three full tables with only512 permitted rows plus192 original encoder-LoRA tensors','three full tables with only512 permitted rows plus12 original shared-head tensors')
w=w.replace('BPE+LoRA B8x2','BPE+shared12 B8x2').replace('frozen12 bytes','frozen192 encoder-LoRA bytes').replace('encoder-LoRA gradient','shared-head gradient')
w=w.replace('Declared paired rowK/LoRAS constant1e-4/3e-4 phase with all-block checkpointing required','Declared rowK/sharedS constant rates and dormant source recipe required')
worker_sha=write('train_lora_batched_tsc_bpe_shared_head_v1.py',w)
tree=ast.parse(w);run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
sync=[n.lineno for n in ast.walk(run) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.cuda.synchronize']
assert len(sync)==1

c=common((HERE/'run_TSC_bpe_lora_native4_v2.py').read_text(encoding='utf-8'))
c=rep(c,"WORKER_SHA = '15825a12c35b1262b28509c921a5a763ecf17696ee02c58446d5a52a33218be3'","WORKER_SHA = '"+worker_sha+"'")
c=rep(c,"TRANSITION_SHA = '22407adf7301fae61c823cf391c87c486bc60e48d748935210b2ed91c8a77469'","TRANSITION_SHA = '"+helper_sha+"'")
c=rep(c,'EXPECTED_SYNC_LINE = 458','EXPECTED_SYNC_LINE = '+str(sync[0]))
c=rep(c,"'2f70811d67af5fc950fa547727dc5ae18fa895a1e5e2cb572c4ff40bbc42476b'","'"+UTILITY_SHA+"'")
c=c.replace("float(option(argv, '--acquisition-lr')) == 1e-4", "float(option(argv, '--row-lr')) in (1e-3, 3e-4)\n        and float(option(argv, '--shared-lr')) in (1e-5, 3e-6)")
c=c.replace("'--candidate-tr-metadata-proof', '--acquisition-lr',", "'--candidate-tr-metadata-proof', '--acquisition-lr', '--row-lr', '--shared-lr',")
c=c.replace("acquisition_lr=float(option(argv, '--acquisition-lr')),", "row_lr=float(option(argv, '--row-lr')), shared_lr=float(option(argv, '--shared-lr')),")
c=c.replace("('--shared-head-source-receipt', receipt_path), ('--acquisition-lr', 1e-4),", "('--shared-head-source-receipt', receipt_path), ('--row-lr', args.row_lr), ('--shared-lr', args.shared_lr),")
c=c.replace("'--shared-head-source-receipt', '--smoke-updates')", "'--shared-head-source-receipt', '--acquisition-lr', '--row-lr', '--shared-lr', '--smoke-updates')")
c=c.replace(".adaptation_declaration()", ".shared_head_declaration(args.row_lr, args.shared_lr)")
c=rep(c,"'declared_adaptation': importlib.import_module(Path(TRANSITION).stem).shared_head_declaration(args.row_lr, args.shared_lr),", "'declared_shared_heads': importlib.import_module(Path(TRANSITION).stem).shared_head_declaration(args.row_lr, args.shared_lr),\n        'row_lr': args.row_lr, 'shared_lr': args.shared_lr,")
c=c.replace("'named192_original_LoRA_moments_copied': True", "'named12_original_shared_head_moments_copied': True")
c=c.replace("'lora_Adam_source_clock'", "'shared_Adam_source_clock'").replace("'LoRA_Adam_clock'", "'shared_Adam_clock'").replace("'lora_clock_S_plus_N'", "'shared_clock_S_plus_N'")
c=c.replace("'row_lr': 1e-4, 'lora_lr': 3e-4,", "'row_lr': plan['row_lr'], 'shared_lr': plan['shared_lr'],")
c=c.replace("'active_parameter_tensors': 195, 'fixed_shared_head_tensors': 12,", "'active_parameter_tensors': 15, 'active_shared_head_tensors': 12, 'frozen_encoder_LoRA_tensors': 192,")
c=rep(c,"'active_parameter_tensors': 195, 'row_lr': plan['row_lr'], 'shared_lr': plan['shared_lr'],", "'active_parameter_tensors': 15, 'active_shared_head_tensors': 12, 'frozen_encoder_LoRA_tensors': 192, 'row_lr': plan['row_lr'], 'shared_lr': plan['shared_lr'],")
c=c.replace("'old15_head_moments_active': False, 'original207_Adam_artifact_retained_dormant': True,", "'original12_shared_head_moments_active': True, 'original3_resized_table_moments_active': False, 'original192_encoder_LoRA_moments_active': False, 'original207_Adam_artifact_retained_dormant': True,")
c=c.replace("optimizers.get('active_parameter_tensors') == 195", "optimizers.get('active_parameter_tensors') == 15")
c=c.replace("and optimizers.get('fixed_shared_head_tensors') == 12\n        and optimizers.get('old15_head_moments_active') is False", "and optimizers.get('active_shared_head_tensors') == 12\n        and optimizers.get('frozen_encoder_LoRA_tensors') == 192\n        and optimizers.get('active_row_lr') == plan['row_lr'] and optimizers.get('active_shared_lr') == plan['shared_lr']\n        and optimizers.get('original12_shared_moments_active') is True\n        and optimizers.get('original3_resized_table_moments_active') is False\n        and optimizers.get('original192_encoder_LoRA_moments_active') is False")
c=c.replace("proof.get('before_shared_head_updates')", "proof.get('before_shared_head_updates')")
c=c.replace('TSC_BPE_LoRA_actual_one_update_smoke_v1','TSC_BPE_shared_heads_actual_one_update_smoke_v1')
c=c.replace('LoRA192_gradients_finite_and_any_delta_nonzero','shared12_gradients_finite_and_all_deltas_nonzero').replace('protected_old_six_blank_duration_and_frozen12_rawbytes_exact','protected_old_six_blank_duration_and_frozen192_encoder_LoRA_rawbytes_exact')
c=c.replace('<= 10000','<= 1000')
c=rep(c,"result.add_argument('--adaptation-max-updates', type=int, default=1000)", "result.add_argument('--row-lr', type=float, choices=(1e-3, 3e-4), default=1e-3)\n    result.add_argument('--shared-lr', type=float, choices=(1e-5, 3e-6), default=1e-5)\n    result.add_argument('--adaptation-max-updates', type=int, default=1000)")
c=rep(c,"and plan['adaptation_max_updates'] == bound and plan['smoke_updates'] == smoke,", "and plan['row_lr'] == float(option(argv, '--row-lr')) == context['receipt']['declared_shared_heads']['row_lr']\n        and plan['shared_lr'] == float(option(argv, '--shared-lr')) == context['receipt']['declared_shared_heads']['shared_lr']\n        and plan['adaptation_max_updates'] == bound and plan['smoke_updates'] == smoke,")
c=c.replace('new rows8199:8711 of three tables and192 encoder LoRA tensors update','new rows8199:8711 of three tables and12 shared-head tensors update')
c=c.replace('encoder-LoRA phase','shared-head phase').replace('a separate LoRA Adam copies','a separate shared-head Adam copies').replace('192 named original states','12 named original states')
c=c.replace("original207 archive preserved; old15 head moments dormant; named192 LoRA states copied atS", "original207 archive preserved; original192 LoRA and3 resized moments dormant; named12 shared-head states copied atS")
c=c.replace("three acquired tables atclockK/LR1e-4 plus192 encoder LoRA atclockS/LR3e-4, one195 global clip, no6boost", "three acquired tables atclockK/declaredrowLR plus12 shared-head tensors atclockS/declaredsharedLR, one15 global clip, no6boost")
c=c.replace('original192 encoder-LoRA','original12 shared-head').replace('195-tensor/two-Adam','15-tensor/two-Adam')
controller_sha=write('run_TSC_bpe_shared_head_v1.py',c)
print({'helper':helper_sha,'worker':worker_sha,'sync':sync[0],'controller':controller_sha})
