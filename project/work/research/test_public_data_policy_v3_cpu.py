"""Small source-isolation/data-v2 fixtures, no Torch or GPU."""
from pathlib import Path
import sys
import json
import ast
import io
import hashlib
import tempfile
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'outputs/parakeet_turkish'))
import public_turkish_data_policy_v3 as policy
checks=[]
def check(name,value):
    if not value: raise AssertionError(name)
    checks.append(name)
def reject(name,function):
    try:function()
    except ValueError:checks.append(name)
    else:raise AssertionError(name)
def row(number=1,**changes):
    split=changes.get('dataset_split','validated')
    value=dict(dataset=policy.CV_REPO,dataset_revision=policy.CV_REV,dataset_config='default',
        dataset_split=split,dataset_file=f'data/{split}-00000-of-00001.parquet',training_role='train',
        source_id=f'{policy.CV_REPO}:default:{split}:None:data/{split}-00000-of-00001.parquet#row={number}',
        sentence_id=None,language='tr',text='Bir Türkçe eğitim cümlesi.',reference='Bir Türkçe eğitim cümlesi.',
        audio_sha256='embedded'+str(number),wav_sha256='wav'+str(number))
    value.update(changes);return value
heldout=row(999,text='Saklı kıyaslama metni.',reference='Saklı kıyaslama metni.',dataset_split='validation')
policy.source_isolation_v3([row(),row(2)],[],[heldout])
check('distinct-audio-training-text-repeats-allowed',True)
for name,changes in [('missing-role',{'training_role':None}),('false-role',{'training_role':'validation'}),
    ('wrong-revision',{'dataset_revision':'untrusted'}),('wrong-file',{'dataset_file':'data/test.parquet'}),
    ('wrong-source-id',{'source_id':'other:default:validated:1'}),('wrong-config',{'dataset_config':'other'}),
    ('wrong-language',{'language':'en'}),('wrong-split',{'dataset_split':'test'})]:
    reject(name,lambda changes=changes:policy.source_isolation_v3([row(**changes)],[],[heldout]))
for name,changes in [('heldout-source',{'source_id':heldout['source_id']}),
    ('heldout-reference',{'text':heldout['text']}),('heldout-embedded',{'audio_sha256':heldout['audio_sha256']}),
    ('heldout-wav',{'wav_sha256':heldout['wav_sha256']})]:
    reject(name,lambda changes=changes:policy.source_isolation_v3([row(**changes)],[],[heldout]))
fleurs=row(dataset=policy.FL_REPO,dataset_revision=policy.FL_REV,dataset_config='tr_tr',dataset_split='train',
    dataset_file='parquet-data/tr_tr/train-00000-of-00001.parquet',source_id='google/fleurs:tr_tr:train:123:file.wav',sentence_id=123)
global_heldout=dict(heldout,dataset=policy.FL_REPO,language='en',sentence_id=123)
reject('global-FLEURS-sentence-isolation',lambda:policy.source_isolation_v3([fleurs],[],[global_heldout]))
reject('duplicate-source-ids',lambda:policy.source_isolation_v3([row(),row()],[],[heldout]))
with tempfile.TemporaryDirectory(dir=ROOT/'work/research',prefix='data_policy_v3_cpu_') as directory:
    path=Path(directory)/'long.wav';signal=np.zeros(31*16000,dtype=np.int16)
    sf.write(path,signal,16000,subtype='PCM_16')
    record=dict(audio_filepath=str(path),wav_sha256=policy.sha256(path),duration=31.,source_id='fixture')
    reject('training-duration-over30-excluded',lambda:policy.audio_evidence(record))
    check('heldout-long-clip-canonical-hash-supported',policy.audio_evidence(record,training=False)==hashlib.sha256(signal.astype('<i2').tobytes()).hexdigest())
    reject('wrong-canonical-audio-rejected',lambda:policy.audio_evidence(record,'wrong',training=False))
    reject('implicit-data-subset-disallowed',lambda:policy.audit_public_training_inputs(path,path,path,train_limit=3))
    reject('implicit-supplement-disallowed',lambda:policy.audit_public_training_inputs(path,path,path,supplement=path))

old=ROOT/'outputs/parakeet_turkish/train_lora_batched.py'
new=ROOT/'outputs/parakeet_turkish/train_lora_batched_data_v3.py'
old_tree=ast.parse(old.read_text(encoding='utf-8'));new_tree=ast.parse(new.read_text(encoding='utf-8'))
def calls(tree,name):
    return [ast.dump(node,include_attributes=False) for node in ast.walk(tree) if isinstance(node,ast.Call) and ast.unparse(node.func)==name]
for name in ('batch_objective','loss.backward','optimizer.step','torch.nn.utils.clip_grad_norm_','BalancedBucketSampler','save_training_state','development_gate'):
    check('unchanged-training-operation-'+name,calls(old_tree,name)==calls(new_tree,name))
sync=[node for node in ast.walk(new_tree) if isinstance(node,ast.Call) and ast.unparse(node.func)=='torch.cuda.synchronize']
check('same-safe-update-hook-line539',len(sync)==1 and sync[0].lineno==539)
check('data-policy-bound-to-new-config-identity','public_turkish_data_policy' in new.read_text(encoding='utf-8'))
check('explicit-ungrouped-TDT-policy',policy.POLICY['tokenizer_decode_group_tokens'] is False)
check('new-trainer-explicit-ungrouped-TDT', 'skip_special_tokens=True, group_tokens=False' in new.read_text(encoding='utf-8'))
check('Torch-not-imported-by-policy-tests','torch' not in sys.modules)
target=ROOT/'outputs/parakeet_turkish/public_data_policy_v3_cpu_verification.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(dict(schema=1,passed=True,CPU_only=True,check_count=len(checks),checks=checks,
    policy_sha256=policy.sha256(policy.__file__),trainer_sha256=policy.sha256(new),
    full_input_verification_pending=True,future_GPU_dispatch_integration_pending=True),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,checks=len(checks),trainer_sha256=policy.sha256(new),policy_sha256=policy.sha256(policy.__file__))))
