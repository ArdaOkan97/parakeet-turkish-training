"""Synthetic-only frozen-winner/TEST pairing/comparison corruption fixtures."""
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUTPUT))
import benchmark_strong_tr_test as module

fixture=ROOT/'work/strong_test_cpu_fixture'
module.WORKSPACE=fixture
module.OUTPUT=fixture/'outputs/parakeet_turkish'
out=module.OUTPUT;data=fixture/'work/data';run=out/'runs/cpu_winner'
model=fixture/'work/models/whisper-large-v3';testout=out/'results/strong_whisper_large_v3_greedy_test'
selected=out/'results/cpu_winner_adapted'
for p in (data,run,model,testout,selected,out/'results/base_v3',out/'results/strong_whisper_large_v3_greedy_dev_verified'):
    p.mkdir(parents=True,exist_ok=True)
def write(path,value):
    module.write_json(path,value)
def jsonl(path,values):
    path.write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values),encoding='utf-8')
checks=[]
def check(name,value):
    checks.append({'name':name,'passed':bool(value)})
    if not value:raise AssertionError(name)
def reject(name,func):
    try:func()
    except (RuntimeError,ValueError,FileNotFoundError,KeyError):check(name,True)
    else:check(name,False)
def modify(path,transform,func):
    original=path.read_bytes()
    try:
        value=module.read(path);transform(value);write(path,value);func()
    finally:path.write_bytes(original)

examples=[]
for i in range(100):
    wav=data/f'{i}.wav';sf.write(wav,np.zeros(1600,np.float32),16000,subtype='PCM_16')
    text='Işık, şeker ve güzel bir gün.'
    examples.append({'source_id':f'fixture:tr:test:{i}','sentence_id':i//2,'language':'tr',
        'dataset':'google/fleurs','dataset_config':'tr_tr','dataset_split':'test',
        'dataset_revision':module.FLEURS_REVISION,'audio_sha256':f'original-{i}',
        'wav_sha256':module.sha256(wav),'audio_filepath':str(wav),'duration':.1,
        'reference':text,'text':text,'reference_normalized':module.normalize_text(text,'tr')})
manifest=data/'test_tr.jsonl';jsonl(manifest,examples)
basepred=[{key:r[key] for key in module.PAIR_KEYS}|{'split':'test','hypothesis':r['reference']} for r in examples]
jsonl(out/'results/base_v3/test_tr.jsonl',basepred)
training={'accepted':True,'run_id':'cpu_winner','selected_step':7,'turkish_validation_wer':.1,
          'selection_split':'FLEURS validation only','weights':str(fixture/'work/winner.pt')}
(fixture/'work/winner.pt').write_bytes(b'synthetic fixture trainables only')
write(run/'training_selection.json',training);write(run/'run_config.json',{'run_id':'cpu_winner'})
write(run/'training_history.json',[{'step':7,'validation':{'turkish_wer':.1,'retention_gate_passed':True}}])
final={**training,'merged':True,'heldout_tests_evaluated':True,'final_results_tag':selected.name}
write(run/'selection.json',final)
write(run/'adapted_results_path.json',{'tag':selected.name,'summary':str(selected/'summary.json')})
write(run/'finalization_provenance.json',{'run_id':'cpu_winner','selected_step':7,
    'selected_state_sha256':module.sha256(fixture/'work/winner.pt')})
frozen={'run_id':'cpu_winner','selected_step':7,'wer':.1,'weights':str(fixture/'work/winner.pt'),
        'selection_split':'FLEURS validation only','frozen_at_utc':'2026-01-01T00:00:00+00:00',
        'provenance':{'selected_step':7,'selection_sha256':module.sha256(run/'training_selection.json'),
        'run_config_sha256':module.sha256(run/'run_config.json'),'history_sha256':module.sha256(run/'training_history.json'),
        'weights_sha256':module.sha256(fixture/'work/winner.pt')}}
freeze=fixture/'work/frozen_final_selection.json';write(freeze,frozen)
f,fa=module.verified_frozen_selection(freeze,'cpu_winner');check('complete finalized frozen winner accepted',True)
reject('missing explicit frozen selection rejected',lambda:module.verified_frozen_selection(fixture/'work/missing.json'))
reject('different CLI winner rejected',lambda:module.verified_frozen_selection(freeze,'other_winner'))
for name,change in [('TEST-selected winner',lambda v:v.update(selection_split='FLEURS test')),
                    ('missing frozen timestamp',lambda v:v.pop('frozen_at_utc')),
                    ('invalid frozen step',lambda v:v.update(selected_step=-1)),
                    ('nonfinite frozen WER',lambda v:v.update(wer=float('nan'))),
                    ('frozen weights mismatch',lambda v:v['provenance'].update(weights_sha256='incorrect')),
                    ('frozen config mismatch',lambda v:v['provenance'].update(run_config_sha256='incorrect'))]:
    modify(freeze,change,lambda n=name:reject(n,lambda:module.verified_frozen_selection(freeze)))
for name,change in [('unmerged winner',lambda v:v.update(merged=False)),('missing final TEST',lambda v:v.update(heldout_tests_evaluated=False)),('different final step',lambda v:v.update(selected_step=8))]:
    modify(run/'selection.json',change,lambda n=name:reject(n,lambda:module.verified_frozen_selection(freeze)))
example_rows,inputaudit=module.verified_test_rows(manifest);check('fixed100 ordered TEST WAV/pair validation accepted',len(example_rows)==100)
reject('DEV manifest refused',lambda:module.verified_test_rows(data/'validation_tr.jsonl'))
original_manifest=manifest.read_bytes()
for name,changed in [('duplicate source',[*examples[:-1],examples[0]]),('wrong split',[{**examples[0],'dataset_split':'validation'},*examples[1:]]),
                    ('wrong reference',[{**examples[0],'reference':'different'},*examples[1:]]),('source audio mismatch',[{**examples[0],'audio_sha256':'changed'},*examples[1:]])]:
    try:jsonl(manifest,changed);reject(name,lambda:module.verified_test_rows(manifest))
    finally:manifest.write_bytes(original_manifest)
(out/'benchmark_strong_tr.py').write_bytes((OUTPUT/'benchmark_strong_tr.py').read_bytes())
(out/'metrics.py').write_bytes((OUTPUT/'metrics.py').read_bytes())
(model/'model.safetensors').write_bytes(b'synthetic fixture asset only')
(model/'config.json').write_text('{}',encoding='utf-8')
lock=fixture/'work/strong_baseline_model.lock.json'
write(lock,{'repository':module.REPOSITORY,'revision':module.REVISION,'local_path':str(model)})
assetsha={p.name:module.sha256(p) for p in model.iterdir()}
devsummary=out/'results/strong_whisper_large_v3_greedy_dev_verified/summary.json'
write(devsummary,{'status':'complete','repository':module.REPOSITORY,'revision':module.REVISION,'protocol':module.PROTOCOL,
    'script_sha256':module.sha256(out/'benchmark_strong_tr.py'),'metrics_code_sha256':module.sha256(out/'metrics.py'),
    'model_lock_sha256':module.sha256(lock),'model_files_sha256':assetsha})
assets=module.verified_assets(model);check('matched pinned fixture assets accepted',True)
modify(lock,lambda v:v.update(revision='different'),lambda:reject('different model revision rejected',lambda:module.verified_assets(model)))
old=(model/'config.json').read_bytes()
try:
    (model/'config.json').write_text('{"changed":true}',encoding='utf-8')
    reject('different tokenizer/frontend asset rejected',lambda:module.verified_assets(model))
finally:(model/'config.json').write_bytes(old)
strong=[];adapted=[]
for i,r in enumerate(examples):
    s={key:r[key] for key in module.PAIR_KEYS}|{'split':'test','hypothesis':r['reference'],
        'generated_token_ids':module.EXPECTED_PREFIX+[42,module.EXPECTED_EOS],'eos_emitted':True,'inference_seconds':.01}
    a={key:r[key] for key in module.PAIR_KEYS}|{'split':'test','hypothesis':r['reference'] if i%3 else 'ışık şeker güzel'}
    strong.append(s);adapted.append(a)
module.validate_predictions(strong,examples,whisper=True);check('full-prefix terminal EOS accepted',True)
bad=copy.deepcopy(strong);bad[0]['generated_token_ids']=[42,module.EXPECTED_EOS]
reject('missing language/task prefix rejected',lambda:module.validate_predictions(bad,examples,whisper=True))
bad=copy.deepcopy(strong);bad[0]['generated_token_ids']=module.EXPECTED_PREFIX+[42]
reject('missing EOS rejected',lambda:module.validate_predictions(bad,examples,whisper=True))
bad=copy.deepcopy(strong);bad[0]['generated_token_ids']+= [42]
reject('tokens after EOS rejected',lambda:module.validate_predictions(bad,examples,whisper=True))
jsonl(testout/'test_tr.jsonl',strong);jsonl(selected/'test_tr.jsonl',adapted)
def score(predictions):
    result=module.score_transcriptions(predictions,language='tr');result.pop('details');return result
write(selected/'summary.json',{'test_tr':{'asr':score(adapted)}})
write(testout/'summary.json',{'status':'complete','repository':module.REPOSITORY,'revision':module.REVISION,
    'split':'test','language':'tr','protocol':module.PROTOCOL,'metrics':score(strong),'input_audit':inputaudit,**assets,
    'script_sha256':module.sha256(Path(module.__file__)),'metrics_code_sha256':module.sha256(out/'metrics.py'),
    'predictions_sha256':module.sha256(testout/'test_tr.jsonl')})
args=SimpleNamespace(output_dir=testout,comparison_output=out/'adaptive_strong_baseline_test_comparison.json')
report=module.cpu_compare(args,f,fa,examples,inputaudit,assets)
check('CPU comparison all100 matches',report['passed'] and report['utterances']==100)
check('sentence-cluster bootstrap50 units',report['paired_sentence_bootstrap']['units']==50
      and report['paired_sentence_bootstrap']['resampling_unit']=='sentence cluster')
check('correct delta direction',report['wer_delta_percentage_points']>0 and report['baseline_asr']['wer']==0)
check('same point estimates',report['paired_sentence_bootstrap']['wer']['adapted']==score(adapted)['wer'])
modify(selected/'summary.json',lambda v:v['test_tr']['asr'].update(word_errors=-1),
       lambda:reject('selected summary metric corruption rejected',lambda:module.cpu_compare(args,f,fa,examples,inputaudit,assets)))
modify(testout/'summary.json',lambda v:v['protocol'].update(num_beams=5),
       lambda:reject('different greedy protocol rejected',lambda:module.cpu_compare(args,f,fa,examples,inputaudit,assets)))
result={'schema':1,'passed':all(c['passed'] for c in checks),'check_count':len(checks),'gpu_used':False,
        'scope':'Synthetic100-record PCM/source/reference fixtures and finalized fake winner provenance; no real heldout inference or model loading',
        'checks':checks,'script_sha256':module.sha256(Path(module.__file__)),
        'protected_dev_script_sha256':module.sha256(OUTPUT/'benchmark_strong_tr.py'),
        'synthetic_bootstrap_units':report['paired_sentence_bootstrap']['units'],'synthetic_resamples':2000}
(OUTPUT/'strong_test_guard_cpu_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ('passed','check_count','gpu_used')}))
