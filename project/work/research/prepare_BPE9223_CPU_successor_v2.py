"""Preserve frozenv1; NEWv2 fixes unified population key and rebinds imports."""
import ast
import hashlib
from pathlib import Path
import re

P = Path(__file__).resolve().parents[2]
S = P/'outputs/parakeet_turkish'
R = P/'work/research'
NAMES = ['fastmss_BPE9223_plan_bridge_v1', 'fastmss_BPE9223_feature_engine_v1',
    'fastmss_BPE9223_cpu_common_v1', 'fastmss_BPE9223_feature_protocol_v1',
    'fastmss_BPE9223_recorded_room_train_v1', 'run_ASR18_BPE9223_CPU_window_v1',
    'produce_FastMSS_BPE9223_reference_CPU_v1', 'produce_FastMSS_BPE9223_frontend_CPU_v1']


def main():
    from ASR18_continuous_data_guard_v5 import DataIOGuard, ROOT
    from atomic_io_retry_v1 import write_json
    guard = DataIOGuard('BPE9223_CPU_source_successor_v2', heavy=False, maximum_bytes=5_000_000)
    guard.wait()
    texts = {}
    for name in NAMES:
        folder = S if name.startswith('fastmss_') else R
        text = (folder/(name+'.py')).read_text(encoding='utf-8')
        for old in NAMES:
            text = text.replace(old, old[:-1]+'2')
        texts[name[:-1]+'2'] = text
    key = 'fastmss_BPE9223_plan_bridge_v2'
    before = 'same(plan["finalized_population"], DATA_evidence["population"])'
    assert texts[key].count(before) == 1
    texts[key] = texts[key].replace(before, 'same(plan["finalized_population"], DATA_evidence["BPE9223_population"])')
    bridge_sha = hashlib.sha256(texts[key].encode()).hexdigest()
    engine = 'fastmss_BPE9223_feature_engine_v2'
    texts[engine], count = re.subn(r'BRIDGE_SHA = "[0-9a-f]{64}"', 'BRIDGE_SHA = "'+bridge_sha+'"', texts[engine])
    assert count == 1
    result = {}
    for name, text in texts.items():
        ast.parse(text)
        folder = S if name.startswith('fastmss_') else R
        path = folder/(name+'.py'); assert not path.exists()
        with path.open('x',encoding='utf-8',newline='\n') as stream:
            stream.write(text)
        guard.pace(len(text.encode()))
        result[name] = dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    import sys
    sys.path.insert(0,str(S))
    import fastmss_BPE9223_recorded_room_train_v2 as augmentation
    area = ROOT/'analysis/BPE9223_training_launch_v1/CPU_preparation_v1'
    policy = augmentation.pin(area/'recorded_room_TRAIN_policy_v1.json',guard.wait)
    _, identity = augmentation.policy_identity(policy,guard.wait)
    identity_path=area/'augmentation_identity_v2.json'; assert not identity_path.exists()
    write_json(identity_path,identity)
    print(dict(sources=result,augmentation_identity=augmentation.pin(identity_path,guard.wait),
        sole_semantic_fix='coordinator finalized_population binds unified evidence.BPE9223_population',
        old_v1_preserved=True,audio_or_models=False),flush=True)


if __name__=='__main__':
    main()
