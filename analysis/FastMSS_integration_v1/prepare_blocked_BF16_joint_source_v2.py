"""Stdlib-only two-line setup correction; no numerical imports or oracle execution."""
import ast,copy,hashlib,json
from pathlib import Path
P=Path(r'C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');R=P/'work/research'
A=Path(r'E:/gpu_training_scratchpad/analysis/ASR18_memory_diagnosis_v1')
OLD=A/'blocked_BF16_joint_configuration_01';NEW=A/'blocked_BF16_joint_configuration_02'
SOURCE=R/'test_ASR18_blocked_BF16_joint_v2.py'
def pin(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
assert not SOURCE.exists() and not NEW.exists()
prior=R/'test_ASR18_blocked_BF16_joint_v1.py';text=prior.read_text(encoding='utf-8')
old='    from tdt_blocked_bf16_joint_view_v1 import DenseBF16JointView,BlockedBF16JointView,REFERENCE_POLICY,POLICY'
assert text.count(old)==1
text=text.replace(old,'    from tdt_joint_canonical_2d_fp32_v2 import apply as canonical_joint\n'+old)
old="        original=Joint(saved['ep'].shape[-1]).to('cuda');original.load_state_dict(saved['joint'])"
assert text.count(old)==1
text=text.replace(old,old+';canonical_joint(original)')
ast.parse(text)
with SOURCE.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
NEW.mkdir()
old_manifest=A/'blocked_BF16_joint_source_manifest_v1.json';manifest=copy.deepcopy(load(old_manifest))
by_path={p['path']:p for p in manifest['bindings']};by_path[str(SOURCE.resolve())]=pin(SOURCE)
manifest.update(kind='ASR18_blocked_BF16_joint_cheap_fixture_source_manifest_v2',
    bindings=sorted(by_path.values(),key=lambda p:p['path']),new_driver=pin(SOURCE),
    preserved_unexecuted_BF16_fixture1_manifest=pin(old_manifest),
    sole_successor_delta='Restore canonical_joint import and apply after trained-head load, supplying inherited canonical marker/shared lock; original activation remains untouched.')
manifest_path=A/'blocked_BF16_joint_source_manifest_v2.json';write(manifest_path,manifest)
c=copy.deepcopy(load(OLD/'configuration_pending_execution_review.json'))
c.update(source=pin(SOURCE),source_manifest=pin(manifest_path))
config_path=NEW/'configuration_pending_execution_review.json';write(config_path,c)
proposal=copy.deepcopy(load(OLD/'source_resource_review_proposal.json'))
proposal.update(source=pin(SOURCE),configuration=pin(config_path),source_manifest=pin(manifest_path),
    preserved_unexecuted_BF16_fixture1_source=pin(prior),
    sole_successor_delta=manifest['sole_successor_delta'])
proposal['launch'].update(script=pin(SOURCE),config=pin(config_path),actual_folder=str(A/'blocked_BF16_joint_actual_02'))
proposal_path=NEW/'source_resource_review_proposal.json';write(proposal_path,proposal)
handoff={'kind':'ASR18_blocked_BF16_joint_cheap_fixture_handoff_v2','source':pin(SOURCE),
    'configuration':pin(config_path),'source_manifest':pin(manifest_path),'proposal':pin(proposal_path),
    'GPU_execution_performed':False,'allowed_executor_owner':'/root/memory_diagnosis','production_enabled':False}
handoff_path=NEW/'handoff.json';write(handoff_path,handoff)
print(json.dumps({'handoff':pin(handoff_path),'source':pin(SOURCE),'configuration':pin(config_path),
    'manifest':pin(manifest_path),'proposal':pin(proposal_path),'bindings':len(manifest['bindings'])},indent=2))
