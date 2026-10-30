"""Prepare NEW minimal shared-head stdlib route and receipt fixtures."""
from pathlib import Path
import ast
import hashlib
import re

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'outputs/parakeet_turkish'
worker=HERE/'train_lora_batched_tsc_bpe_shared_head_v1.py'
helper=HERE/'tsc_bpe_shared_head_transition_v1.py'
controller=HERE/'run_TSC_bpe_shared_head_v1.py'
tree=ast.parse(worker.read_text(encoding='utf-8'))
run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
sync=[n.lineno for n in ast.walk(run) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.cuda.synchronize']
assert len(sync)==1
c=controller.read_text(encoding='utf-8')
c=re.sub(r"WORKER_SHA = '[0-9a-f]+'", "WORKER_SHA = '"+hashlib.sha256(worker.read_bytes()).hexdigest()+"'",c)
c=re.sub(r"TRANSITION_SHA = '[0-9a-f]+'", "TRANSITION_SHA = '"+hashlib.sha256(helper.read_bytes()).hexdigest()+"'",c)
c=re.sub(r'EXPECTED_SYNC_LINE = [0-9]+','EXPECTED_SYNC_LINE = '+str(sync[0]),c)
controller.write_text(c,encoding='utf-8')

text=(ROOT/'work/research/check_TSC_bpe_lora_native4_controller_v2.py').read_text(encoding='utf-8')
for a,b in (
    ('run_TSC_bpe_lora_native4_v2.py','run_TSC_bpe_shared_head_v1.py'),
    ('--resume-adaptation','--resume-shared-heads'),('--adaptation-source-receipt','--shared-head-source-receipt'),
    ('declared_adaptation','declared_shared_heads'),('lora_optimizer','shared_optimizer'),
    ('LoRAS_schema','sharedS_schema'),('inactive_TSC_BPE_LoRA_native4_controller_dispatch_CPU_verification_v2','inactive_TSC_BPE_shared_heads_controller_dispatch_CPU_verification_v1')
):text=text.replace(a,b)
text=text.replace("'--acquisition-lr', '.0001',", "'--row-lr', '.001', '--shared-lr', '.00001',")
a=text.index('    # Compare genuine new worker scientific functions');b=text.index('    publications = []',a)
text=text[:a]+'''    # Actual objective AST changes only the projector's no_grad scope.
    import copy
    old=ast.parse((HERE/'train_lora_batched_tsc_bpe_row_lr_v1.py').read_text(encoding='utf-8'))
    before=copy.deepcopy(next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='batch_objective'))
    after=copy.deepcopy(functions['batch_objective'])
    before.body.pop(0);after.body.pop(0)
    autocast=next(n for n in before.body if isinstance(n,ast.With))
    no_grad=autocast.body[0]
    projector=no_grad.body.pop()
    require(ast.unparse(projector)=='ep = model.encoder_projector(features)', 'Expected projector assignment')
    autocast.body.insert(1,projector)
    require(ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False),
        'Only acoustic encoder no_grad; all loss/U-normalization/KD math otherwise identical')
    passed('actual_objective_AST_only_projector_moved_outside_no_grad_preserves_native_and_KD_math')
    source=(HERE/'train_lora_batched_tsc_bpe_shared_head_v1.py').read_text(encoding='utf-8')
    require('checkpoint_encoder_blocks(model)' not in source and 'adaptation.inherited_runtime = None' in source,
        'Frozen encoder graph and old runtime lifetime')
    protected=ast.unparse(functions['_protected_snapshot'])
    require('if name in adaptation.shared_names:' in protected and 'adaptation.adapter_names' not in protected,
        'All192 frozen LoRA and protected table rows must be checked; shared12 excluded')
    passed('actual_frozen192_protected_row_snapshot_and_released_source_runtime_scope')

''' +text[b:]
text=text.replace("'declared_shared_heads': {'explicit_fixture_admission_mock': True}","'row_lr': .001, 'shared_lr': .00001, 'declared_shared_heads': {'explicit_fixture_admission_mock': True}")
text=text.replace('native_max_joint_bytes=1850000000)', 'native_max_joint_bytes=1850000000,row_lr=.001,shared_lr=.00001)')
path=ROOT/'work/research/check_TSC_bpe_shared_head_controller_v1.py'
assert not path.exists();ast.parse(text);path.write_text(text,encoding='utf-8')

text=(ROOT/'work/research/check_TSC_bpe_row_lr_boundary_v1.py').read_text(encoding='utf-8')
text=text.replace('tsc_bpe_row_lr_transition_v1','tsc_bpe_shared_head_transition_v1')
text=text.replace('for lr in (1e-3,3e-4):', 'for lr,shared in ((1e-3,1e-5),(1e-3,3e-6),(3e-4,1e-5),(3e-4,3e-6)):')
text=text.replace('allow_continued_row_only_acquisition','allow_shared_head_adaptation').replace('declared_row_lr','declared_shared_heads')
text=text.replace('h.continued_acquisition_declaration(lr)','h.shared_head_declaration(lr,shared)')
text=text.replace("h.validate_receipt(r,'synthetic_rowLR',lr)","h.validate_receipt(r,'synthetic_rowLR',lr,shared)")
text=text.replace("h.validate_receipt(altered,'synthetic_rowLR',lr)","h.validate_receipt(altered,'synthetic_rowLR',lr,shared)")
text=text.replace('h.continued_acquisition_declaration(1e-4)','h.shared_head_declaration(1e-4,1e-5)').replace('h.continued_acquisition_declaration(3e-5)','h.shared_head_declaration(1e-3,3e-4)')
text=text.replace('TSC_BPE_row_LR_boundary_CPU_verification_v1','TSC_BPE_shared_head_boundary_CPU_verification_v1').replace('inactive_TSC_BPE_continued_row_LR_boundary','inactive_TSC_BPE_shared_head_boundary')
path=ROOT/'work/research/check_TSC_bpe_shared_head_boundary_v1.py'
assert not path.exists();ast.parse(text);path.write_text(text,encoding='utf-8')
print({'worker_sha':hashlib.sha256(worker.read_bytes()).hexdigest(),'helper_sha':hashlib.sha256(helper.read_bytes()).hexdigest(),'controller_sha':hashlib.sha256(controller.read_bytes()).hexdigest(),'sync':sync[0]})
