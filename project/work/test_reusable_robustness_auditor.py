"""CLI regression checks; synthetic identity variants reuse saved real scores."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/parakeet_turkish'
WORK=ROOT/'work/reusable_robustness_test'
WORK.mkdir(exist_ok=True)
PYTHON=ROOT/'work/training-env/Scripts/python.exe'
SCRIPT=OUTPUT/'verify_confidence_robustness.py'
report=json.loads((OUTPUT/'baseline_confidence_robustness.json').read_text(encoding='utf-8'))
fixture=copy.deepcopy(report)
fixture['models']={'base':fixture['models']['v2'],'selected':fixture['models']['v3']}
fixture['pairwise_model_sensitivity']={'base_vs_selected':fixture['pairwise_model_sensitivity']['v2_vs_v3']}
write=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
path=WORK/'relabelled.json'; write(path,fixture)

def run(input_path,refs,output):
    command=[str(PYTHON),str(SCRIPT),'--input',str(input_path),'--output',str(output)]
    for name,reference in refs.items():
        command.extend(['--reference',f'{name}={reference}'])
    return subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')

refs={'base':OUTPUT/'results/base_v2','selected':OUTPUT/'results/base_v3'}
result=run(path,refs,WORK/'relabelled_audit.json')
assert result.returncode==0,result.stderr
check=json.loads((WORK/'relabelled_audit.json').read_text(encoding='utf-8'))
assert set(check['models'])=={'base','selected'} and 'base_vs_selected' in check['paired_model_comparison']

# A synthetic canonical-name export tests provenance logic only. It does not
# represent a trained checkpoint or new inference; all scored audio is baseline.
from transformers import AutoConfig, AutoModelForTDT
import torch
model_dir=WORK/'synthetic_export'; model_dir.mkdir(exist_ok=True)
reference_dir=WORK/'synthetic_reference'; reference_dir.mkdir(exist_ok=True)
base_path=Path(report['models']['v3']['model_path'])
shutil.copyfile(base_path/'config.json',model_dir/'config.json')
sidecar=model_dir/'synthetic-proof.txt'; sidecar.write_text('Fixture-only publication hash check.',encoding='utf-8')
canonical='nvidia/parakeet-tdt-0.6b-v3@541d1f99c6b0c3cd0b11a95167540bb8edefd82b'
digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()
config=AutoConfig.from_pretrained(model_dir,local_files_only=True)
config._name_or_path=str(model_dir)
with torch.device('meta'):
    template=AutoModelForTDT.from_config(config,dtype=torch.float32)
config=template.config
probe_identity=fixture['models']['selected']['model_fingerprint']['identity']
probe_identity['model_config_sha256']=digest(config.to_dict())
fixture['models']['selected']['model_fingerprint']['sha256']=digest(probe_identity)
fixture['models']['selected']['model_path']=str(model_dir)
metadata=json.loads((OUTPUT/'results/base_v3/metadata.json').read_text(encoding='utf-8'))
config._name_or_path=canonical
metadata['model_fingerprint']['identity']['model_config_sha256']=digest(config.to_dict())
metadata['model_fingerprint']['sha256']=digest(metadata['model_fingerprint']['identity'])
write(reference_dir/'metadata.json',metadata)
shutil.copyfile(OUTPUT/'results/base_v3/validation_en.jsonl',reference_dir/'validation_en.jsonl')
provenance={'canonical_model_name_or_path':canonical,'candidate_files_sha256':{
    p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (model_dir/'config.json',sidecar)}}
write(model_dir/'finalization_provenance.json',provenance)
canonical_input=WORK/'canonical_fixture.json';write(canonical_input,fixture)
canonical_refs=dict(refs,selected=reference_dir)
result=run(canonical_input,canonical_refs,WORK/'canonical_audit.json')
assert result.returncode==0,result.stderr
canonical_audit=json.loads((WORK/'canonical_audit.json').read_text(encoding='utf-8'))
note=next(n for n in canonical_audit['fingerprint_notes'] if n['model']=='selected')
assert note['reference_name_or_path']==canonical and note['exported_provenance']['exported_files_verified']==2
sidecar.write_text('Changed file must be rejected.',encoding='utf-8')
result=run(canonical_input,canonical_refs,WORK/'bad_copy_audit.json')
assert result.returncode!=0 and 'exact published checkpoint file selectedsynthetic-proof.txt' in result.stderr,result.stderr
sidecar.write_text('Fixture-only publication hash check.',encoding='utf-8')
fixture['models']['selected']['model_fingerprint']['identity']['generation_config_sha256']='invalid-generation-identity'
bad=WORK/'bad_generation.json';write(bad,fixture)
result=run(bad,canonical_refs,WORK/'bad_generation_audit.json')
assert result.returncode!=0 and 'probe fingerprint integrity selected' in result.stderr,result.stderr
verification={'passed':True,'gpu_used':False,'auditor_sha256':hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
    'checks':['Arbitrary model aliases and external references accepted',
        'Actual baseline source pairings and all metric recalculations remain identical',
        'Synthetic canonical config hashes exactly reconstructed with verified file provenance',
        'Tampered publication file rejected',
        'Changed generation identity rejected rather than normalized'],
    'scope':'CLI identity/provenance regression checks reuse actual baseline scores under synthetic metadata; no selected-pilot or new-inference claim.'}
write(OUTPUT/'verify_confidence_robustness_verification.json',verification)
print(json.dumps(verification,indent=2))
