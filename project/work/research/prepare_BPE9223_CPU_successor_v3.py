"""NEW source closure for exact registered full-owner CPU payload paths."""
import ast
import hashlib
from pathlib import Path
import re

P=Path(__file__).resolve().parents[2]
S=P/'outputs/parakeet_turkish';R=P/'work/research'
NAMES=['fastmss_BPE9223_plan_bridge_v2','fastmss_BPE9223_feature_engine_v2',
    'fastmss_BPE9223_cpu_common_v2','fastmss_BPE9223_feature_protocol_v2',
    'fastmss_BPE9223_recorded_room_train_v2','run_ASR18_BPE9223_CPU_window_v2',
    'produce_FastMSS_BPE9223_reference_CPU_v2','produce_FastMSS_BPE9223_frontend_CPU_v2']


def main():
    from ASR18_continuous_data_guard_v5 import DataIOGuard,ROOT
    from atomic_io_retry_v1 import write_json
    g=DataIOGuard('BPE9223_CPU_source_successor_v3',heavy=False,maximum_bytes=5_000_000);g.wait()
    texts={}
    for name in NAMES:
        p=(S if name.startswith('fastmss_') else R)/(name+'.py')
        text=p.read_text(encoding='utf-8')
        for old in NAMES:text=text.replace(old,old[:-1]+'3')
        text=text.replace('asr18_BPE9223_source_context_v1','asr18_BPE9223_source_context_v2')
        texts[name[:-1]+'3']=text
    name='fastmss_BPE9223_feature_protocol_v3';text=texts[name]
    before="""    require(type(template) is dict and set(template) == template_keys
        and template['kind'] == TEMPLATE"""
    after="""    require(type(template) is dict and set(template) in (template_keys, template_keys | {'checkpoint_storage_namespace'})
        and template['kind'] == TEMPLATE"""
    assert text.count(before)==1;text=text.replace(before,after)
    before="""    launch = ROOT / 'analysis/BPE9223_training_launch_v1'
    require(root != launch and root.is_relative_to(launch.resolve())
        and not root.is_relative_to(CACHE.resolve())
        and scoped(proposal['output_root']) == root / proposal['window_id'],
        'Private outputs derive only under this reviewed launch namespace')"""
    after="""    launch = ROOT / 'analysis/BPE9223_training_launch_v1'
    if 'checkpoint_storage_namespace' in template:
        checkpoint_root = scoped(template['checkpoint_storage_namespace'])
        require(type(template['checkpoint_storage_namespace']) is str
            and checkpoint_root.is_relative_to((ROOT / 'runs').resolve())
            and checkpoint_root.name == 'checkpoints' and root == checkpoint_root / 'transient'
            and re.fullmatch(r'p[0-9]{9}', proposal['window_id']) is not None
            and scoped(proposal['output_root']) == root / proposal['window_id'] / 'CPU' / 'payload',
            'Exact reviewed checkpoint/transient/p9/CPU/payload full-owner layout')
    else:
        require(root != launch and root.is_relative_to(launch.resolve())
            and not root.is_relative_to(CACHE.resolve())
            and scoped(proposal['output_root']) == root / proposal['window_id'],
            'Exact reviewed finite launch/direct layout')"""
    assert text.count(before)==1;texts[name]=text.replace(before,after)
    sha=hashlib.sha256(texts['fastmss_BPE9223_plan_bridge_v3'].encode()).hexdigest()
    name='fastmss_BPE9223_feature_engine_v3'
    texts[name],count=re.subn(r'BRIDGE_SHA = "[0-9a-f]{64}"','BRIDGE_SHA = "'+sha+'"',texts[name]);assert count==1
    helper=(S/'asr18_BPE9223_CPU_window_configuration_v1.py').read_text(encoding='utf-8')
    helper=helper.replace('fastmss_BPE9223_feature_protocol_v2','fastmss_BPE9223_feature_protocol_v3')
    protocol_sha=hashlib.sha256(texts['fastmss_BPE9223_feature_protocol_v3'].encode()).hexdigest()
    helper,count=re.subn(r"PROTOCOL_SHA256 = '[0-9a-f]{64}'","PROTOCOL_SHA256 = '"+protocol_sha+"'",helper);assert count==1
    helper=helper.replace("set(value) == TEMPLATE_KEYS", "set(value) in (TEMPLATE_KEYS, TEMPLATE_KEYS | {'checkpoint_storage_namespace'})")
    before="""    protocol.require(output == namespace / window_id, 'Exact template-derived private E output')"""
    after="""    expected = namespace / window_id / 'CPU' / 'payload' if 'checkpoint_storage_namespace' in value else namespace / window_id
    protocol.require(output == expected, 'Exact template-derived private E output')"""
    assert helper.count(before)==1;helper=helper.replace(before,after)
    texts['asr18_BPE9223_CPU_window_configuration_v2']=helper
    result={}
    for name,text in texts.items():
        ast.parse(text);path=(R if name.startswith(('run_','produce_')) else S)/(name+'.py');assert not path.exists()
        with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
        g.pace(len(text.encode()));result[name]=dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    import sys
    sys.path.insert(0,str(S));import fastmss_BPE9223_recorded_room_train_v3 as aug
    area=ROOT/'analysis/BPE9223_training_launch_v1/CPU_preparation_v1';target=area/'augmentation_identity_v3.json';assert not target.exists()
    _,identity=aug.policy_identity(aug.pin(area/'recorded_room_TRAIN_policy_v1.json',g.wait),g.wait);write_json(target,identity)
    print(dict(sources=result,augmentation_identity=aug.pin(target,g.wait),old_versions_preserved=True,
        sole_semantic_delta='Exact ROOT-reviewed E/runs/checkpoints/transient/p9/CPU/payload layout added; old finite layout preserved',
        context_source='asr18_BPE9223_source_context_v2.py',no_audio_or_model=True),flush=True)


if __name__=='__main__':main()
