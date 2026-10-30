"""Stdlib-only source/config assembly. Never imports Torch or executes the oracle."""
import ast, copy, hashlib, json
from pathlib import Path

P=Path(r'C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
R=P/'work/research'; S=P/'outputs/parakeet_turkish'
A=Path(r'E:/gpu_training_scratchpad/analysis/ASR18_memory_diagnosis_v1')
OLD=A/'blocked_FP32_joint_configuration_01'
NEW=A/'blocked_BF16_joint_configuration_01'
SOURCE=R/'test_ASR18_blocked_BF16_joint_v1.py'

def pin(path):
    path=Path(path).resolve(); raw=path.read_bytes()
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n')

assert not SOURCE.exists() and not NEW.exists()
text=(R/'test_ASR18_blocked_FP32_joint_v1.py').read_text(encoding='utf-8')
text=text.replace('blocked_FP32','blocked_BF16').replace('Blocked FP32','Blocked BF16')
text=text.replace('NEW full-U/T-block FP32 cross-GEMM oracle','NEW full-U/T-block BF16 cross-GEMM oracle')
def replace(old,new):
    global text
    assert old in text,old
    text=text.replace(old,new)

replace("'FP32_gradient_relative_L2_strictly_less_than':.001,'BF16_operand_gradient_relative_L2_strictly_less_than':.03,",
        "'BF16_joint_all_gradient_relative_L2_strictly_less_than':.03,")
replace("    require(record(c['prior_tolerance_source']['path'])==c['prior_tolerance_source'], 'Pinned existing tolerance authority')",
        "    require(record(c['prior_tolerance_source']['path'])==c['prior_tolerance_source'], 'Pinned existing BF16-autocast tolerance authority')\n"
        "    require(record(c['original_HF_source']['path'])==c['original_HF_source'], 'Pinned actual original activation/head recipe')")
replace("    from tdt_joint_canonical_2d_fp32_v2 import apply as canonical_joint\n",'')
replace("    from tdt_dense_fp32_joint_view_v1 import DenseFP32JointView,POLICY as REFERENCE_POLICY\n"
        "    from tdt_blocked_fp32_joint_view_v1 import BlockedFP32JointView,POLICY",
        "    from tdt_blocked_bf16_joint_view_v1 import DenseBF16JointView,BlockedBF16JointView,REFERENCE_POLICY,POLICY\n"
        "    require(POLICY==c['candidate_joint_policy'] and REFERENCE_POLICY==c['reference_joint_policy']\n"
        "        and POLICY['original_HF_source']==c['original_HF_source'], 'Exact predeclared BF16 joint policies and original HF source')")
replace('    from tdt_blocked_full_native_v1 import bounded_tdt_loss as candidate_loss',
        '    from tdt_blocked_bf16_full_native_v1 import bounded_tdt_loss as candidate_loss')
replace('        def __init__(self,width):super().__init__();self.head=torch.nn.Linear(width,8716)',
        '        def __init__(self,width):\n'
        '            super().__init__();self.activation=torch.nn.ReLU(inplace=False);self.head=torch.nn.Linear(width,8716)')
replace("        original=Joint(saved['ep'].shape[-1]).to('cuda');original.load_state_dict(saved['joint']);canonical_joint(original)",
        "        original=Joint(saved['ep'].shape[-1]).to('cuda');original.load_state_dict(saved['joint'])")
replace('        reference_joint=DenseFP32JointView(original);joint=BlockedFP32JointView(original);head=tuple(joint.parameters())',
        '        reference_joint=DenseBF16JointView(original);joint=BlockedBF16JointView(original);head=tuple(joint.parameters())\n'
        '        require(reference_joint.activation is joint.activation is original.activation and not original.activation.inplace,\n'
        "            'Same actual original-style ReLU object, no in-place mutation')")
replace("'official_full_vocabulary_dense_FP32_reference'","'official_full_vocabulary_dense_BF16_reference'")
replace('                tolerance=.03 if expected.dtype==torch.bfloat16 else .001',
        '                # BF16 GEMM policy applies to all adjoints, including FP32-stored head Parameters.\n'
        '                tolerance=.03')
replace("                'time_ranges_including_tail':time_ranges,'full_predictor_positions':int(labels.shape[1])+1,",
        "                'time_ranges_including_tail':time_ranges,'full_predictor_positions':int(labels.shape[1])+1,\n"
        "                'saved_projected_input_dtypes':[str(e.dtype),str(d.dtype)],\n"
        "                'joint_operands_addition_activation_and_GEMM_dtype':'bfloat16',\n"
        "                'lexical_normalization_duration_logsoftmax_and_DP_dtype':'float32',")
replace("        'cross_policy_gate':c['cross_policy_gate'],'prior_tolerance_source':c['prior_tolerance_source'],",
        "        'cross_policy_gate':c['cross_policy_gate'],'prior_tolerance_source':c['prior_tolerance_source'],\n"
        "        'original_HF_source':c['original_HF_source'],")
replace("'canonical_T1_or_BF16_trajectory_equivalence_claimed':False,",
        "'canonical_T1_FP32_trajectory_equivalence_claimed':False,'full_clip_BF16_GEMM_byte_equivalence_claimed':False,")
ast.parse(text)
with SOURCE.open('x',encoding='utf-8',newline='\n') as f:f.write(text)

module=ast.parse((S/'tdt_blocked_bf16_joint_view_v1.py').read_text(encoding='utf-8'))
policy=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='POLICY' for t in n.targets))
reference_policy={k:v for k,v in policy.items() if k not in ('maximum_time_block','maximum_complete_joint_union_bytes',
    'conservative_FP32_packed_work_unit_bytes','full_clip_BF16_GEMM_byte_equivalence_claimed')}
reference_policy['kind']='ASR18_full_clip_original_style_BF16_joint_v1'
hf=pin(policy['original_HF_source']['path']);assert hf==policy['original_HF_source']
primitives=[pin(S/'tdt_blocked_full_native_allocation_budget_v1.py'),pin(S/'tdt_blocked_bf16_joint_view_v1.py'),
            pin(S/'tdt_blocked_bf16_full_native_v1.py')]
assert [p['sha256'] for p in primitives]==[
    'a4d129c6a7b10034a3b71f4b5175414a68cde56020c5d9b5228f3a34d4479fd7',
    '75ef1f1c2ad68247bc8f6431e7d2ba964138b9963e7672d3d5c790eaa217cf95',
    'd3e3c855bcb409f64deac918a53666b33bbb86feeb0e13690df7ce3cb80f4f31']
NEW.mkdir()
old_manifest=A/'blocked_FP32_joint_source_manifest_v1.json'
manifest=copy.deepcopy(load(old_manifest)); by_path={p['path']:p for p in manifest['bindings']}
for p in [pin(SOURCE),*primitives,hf]:by_path[p['path']]=p
manifest.update(kind='ASR18_blocked_BF16_joint_cheap_fixture_source_manifest_v1',
    bindings=sorted(by_path.values(),key=lambda p:p['path']),source_only=True,
    new_driver=pin(SOURCE),primitive_sources=primitives,original_HF_source=hf,
    reused_blocked_FP32_fixture_manifest=pin(old_manifest))
manifest_path=A/'blocked_BF16_joint_source_manifest_v1.json';write(manifest_path,manifest)

gate={'raw_loss_max_abs_strictly_less_than':.01,'BF16_joint_all_gradient_relative_L2_strictly_less_than':.03,
      'all4_RNG_exact':True,'tensor_byte_equality_reported_not_required':True}
c=copy.deepcopy(load(OLD/'configuration_pending_execution_review.json'))
c.update(kind='ASR18_blocked_BF16_trained_joint_cross_policy_oracle_configuration_v1',source=pin(SOURCE),
    source_manifest=pin(manifest_path),cross_policy_gate=gate,original_HF_source=hf,
    candidate_joint_policy=policy,reference_joint_policy=reference_policy,
    independent_reference='official full-vocabulary native loss +DenseBF16JointView +original full dense local autograd',
    candidate='source-budgeted T<=32/FULL-U BlockedBF16JointView +compactDP +original local autograd.grad',
    canonical_T1_FP32_trajectory_claimed=False,full_clip_BF16_GEMM_byte_equivalence_claimed=False,
    FP32_normalization_duration_logsoftmax_and_DP_unchanged=True,
    BF16_autocast_existing_joint_gradient_gate_applies_to_all_ep_dp_weight_bias=True)
c.pop('canonical_T1_or_BF16_trajectory_claimed',None)
c.pop('no_tolerance_widening',None)
config_path=NEW/'configuration_pending_execution_review.json';write(config_path,c)
proposal=copy.deepcopy(load(OLD/'source_resource_review_proposal.json'))
proposal.update(kind='ASR18_blocked_BF16_joint_cheap_cross_GEMM_fixture_source_resource_proposal_v1',
    scope='trained_joint_fullU_blocked_BF16_cross_GEMM_oracle_v1',source=pin(SOURCE),
    configuration=pin(config_path),source_manifest=pin(manifest_path),primitive_sources=primitives,
    original_HF_source=hf,candidate_joint_policy=policy,independent_reference_joint_policy=reference_policy,
    predeclared_cross_policy_gate=gate,reused_blocked_FP32_fixture_configuration=pin(OLD/'configuration_pending_execution_review.json'))
proposal['launch'].update(script=pin(SOURCE),config=pin(config_path),
    actual_folder=str(A/'blocked_BF16_joint_actual_01'),expected_review_scope=proposal['scope'])
proposal['limits']=[
    'Source-only derivative; GPU launch exclusively memory_diagnosis after exact ROOT review and fresh admission.',
    'Same trained T15 encoder prefix and ALL original U574/484 labels, not whole ASR/update/full utterance proof.',
    'NEW explicit original-style BF16 addition/ReLU/GEMM; SAME BF16 full official reference versus blocked BF16 compact DP.',
    'No canonical FP32/T1 trajectory or full-clip BF16 GEMM byte-equivalence claim; prior BF16 .03 relativeL2 applies to ALL adjoints including FP32 head storage, lossabs<.01 predeclared before results.',
    'Full FP32 conservative allocation accounting retained, unchanged host forecast; no host saving or throughput/adoption claim.',
    'Actual fixed contexts prior_compact_cells=0; native grouped caller must independently charge all other live compact contexts.',
    'No actual ASR/Adam/audio/draw/update, teacher or preservation change; no installed/shared/admitted file edits.']
proposal['strict_invariants']=[
    'Same exact original-style activation object and two FP32 trained head Parameter objects/pointers/versions/bytes; no .grad assignment.',
    'Actual view explicitly uses BF16 addition/ReLU/GEMM in both arms; FP32 lexical normalization, duration logsoftmax and official DP.',
    'Complete finite loss/ep/dp/head weight/head bias metrics with maxabs, relativeL2, rawSHA and reported byte equality; no tensor exclusions.',
    'All4 RNG exact per arm and full fixture; unchanged exact actual Popen0/native0/joined guards/captured identity death required.']
proposal['source_assembly_only']=True
proposal_path=NEW/'source_resource_review_proposal.json';write(proposal_path,proposal)
handoff={'kind':'ASR18_blocked_BF16_joint_cheap_fixture_handoff_v1','source':pin(SOURCE),
    'configuration':pin(config_path),'source_manifest':pin(manifest_path),'proposal':pin(proposal_path),
    'GPU_execution_performed':False,'allowed_executor_owner':'/root/memory_diagnosis','production_enabled':False}
handoff_path=NEW/'handoff.json';write(handoff_path,handoff)
print(json.dumps({'handoff':pin(handoff_path),'source':pin(SOURCE),'configuration':pin(config_path),
                  'manifest':pin(manifest_path),'proposal':pin(proposal_path),'bindings':len(manifest['bindings'])},indent=2))
