"""Narrow NEW CPU4 source closure: reviewed9223 hard context only."""
import ast,hashlib,json
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
O=Path(__file__).parent;N=O.parent
old=json.loads((N/'CPU_preparation_v1/source_handoff_v3.json').read_text())
context=S/'asr18_BPE9223_hard_envelope_context_v1.py'
assert hashlib.sha256(context.read_bytes()).hexdigest()=='7968517267a14e8c88d297374593bf36266fc8613c4b7da18cd8e72bb83b90ed'
mapping={Path(v['path']).stem:Path(v['path']).stem[:-1]+'4' for v in old['sources'].values()}
sources={};checks=[]
for key,pin in old['sources'].items():
    original=Path(pin['path']);raw=original.read_text(encoding='utf-8');assert hashlib.sha256(original.read_bytes()).hexdigest()==pin['sha256']
    expected=raw
    for before,after in mapping.items():expected=expected.replace(before,after)
    if key=='bridge':
        expected=expected.replace('asr18_hard_envelope_schedule_context_v1','asr18_BPE9223_hard_envelope_context_v1').replace('24cb88c113800c5eede05863d51eaf33ace7a57c4fdcd3e1dd38a4e6c7f60b59','7968517267a14e8c88d297374593bf36266fc8613c4b7da18cd8e72bb83b90ed')
    ast.parse(expected);target=original.with_name(mapping[original.stem]+'.py');assert not target.exists()
    target.write_text(expected,encoding='utf-8',newline='\n')
    sources[key]={'path':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    checks.append(key+': exact import/path substitution'+(' + sole hard-context name/hash' if key=='bridge' else ''))
helper=S/'asr18_BPE9223_CPU_window_configuration_v2.py';text=helper.read_text()
text=text.replace('fastmss_BPE9223_feature_protocol_v3','fastmss_BPE9223_feature_protocol_v4')
text=text.replace(old['sources']['protocol']['sha256'],sources['protocol']['sha256'])
ast.parse(text);target=S/'asr18_BPE9223_CPU_window_configuration_v3.py';assert not target.exists();target.write_text(text,encoding='utf-8',newline='\n')
report={'kind':'ASR18_BPE9223_CPU4_narrow_source_closure_v1','sources':sources,
    'baseline_sources':old['sources'],'checks':checks,'hard_context':{'path':str(context),'bytes':context.stat().st_size,'sha256':hashlib.sha256(context.read_bytes()).hexdigest()},
    'configuration_deriver':{'path':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()},
    'full_branch_math_or_resource_changes':False,'actual_audio_GPU_or_model_jobs':0}
(O/'generation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
