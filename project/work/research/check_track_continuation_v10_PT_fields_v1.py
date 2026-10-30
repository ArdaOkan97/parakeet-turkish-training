"""Extracted stdlib PT logger fields; byte-reader stubs, no mirror/DB/history."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
source=HERE/'track_continuation_v10.py';raw=source.read_bytes();tree=ast.parse(raw.decode('utf-8'))
names={'finite_number','portuguese_reference_record','required_eval_languages','portuguese_base_reference'}
nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
assert len(nodes)==len(names)
record={'path':str((ROOT/'work/research/synthetic_PT_reference.json').resolve()),'sha256':'a'*64,'bytes':123}
score={'utterances':50,'wer':.02,'cer':.01,'word_errors':2,'reference_words':100,'character_errors':2,'reference_characters':200}
proof={'kind':'Portuguese_retention_baselines_v1','passed':True,'actual_GPU_decode':True,'fixed_PT_DEV_count':50,
    'base_revision':'541d1f99c6b0c3cd0b11a95167540bb8edefd82b','original_six_language_baseline_unchanged':True,
    'no_TEST_prediction_or_score_used':True,'baseline_PT':{'validation_pt':{'asr':score}},'baseline_overlay':{'validation_pt':{'asr':score}}}
current=[proof]
ns={'Path':Path,'ROOT':ROOT,'math':math,'file_record':lambda path:copy.deepcopy(record),'read_json':lambda path,default:current[0]}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
checks=[];rejections=[]
def ok(condition,name):
    if not condition:raise AssertionError(name)
    checks.append(name)
def bad(call,name):
    try:call()
    except (ValueError,KeyError,TypeError):checks.append(name);rejections.append(name);return
    raise AssertionError('unexpected acceptance '+name)
old={'run_id':'synthetic_old'};new={'evaluation_languages':['en','es','it','fr','de','tr','pt'],'portuguese_retention_baseline_proof':record}
ok(ns['required_eval_languages'](old)==['en','es','it','fr','de','tr'],'historical six language final requirement unchanged')
ok(len(ns['required_eval_languages'](new))==7 and 'pt' in ns['required_eval_languages'](new),'Portuguese requires seventh actual final')
ok(ns['portuguese_base_reference'](old) is None,'historical runs do not invent PT reference')
ok(ns['portuguese_base_reference'](new)['metrics']=={'reference/base_v3_wer_pt_percent':2.},'actual PT reference kept under reference namespace')
for languages in (['en','es','it','fr','de','tr'],['en','es','it','fr','de','tr','pt','pt'],['en','es','it','fr','de','tr','ja']):
    bad(lambda value=languages:ns['required_eval_languages']({**new,'evaluation_languages':value}),'invalid PT final scope rejected '+str(languages))
bad(lambda:ns['portuguese_base_reference']({**new,'portuguese_retention_baseline_proof':{**record,'sha256':'b'*64}}),'changed PT proof bytes rejected')
bad(lambda:ns['portuguese_base_reference']({**new,'portuguese_retention_baseline_proof':'planned_path_only'}),'unbound PT path rejected')
for field,value in [('kind','CPU_readiness_only'),('actual_GPU_decode',False),('fixed_PT_DEV_count',49),
    ('original_six_language_baseline_unchanged',False),('no_TEST_prediction_or_score_used',False)]:
    current[0]={**proof,field:value};bad(lambda:ns['portuguese_base_reference'](new),'invalid PT actual proof '+field)
current[0]=copy.deepcopy(proof);current[0]['baseline_overlay']['validation_pt']={'asr':{**score,'word_errors':3}}
bad(lambda:ns['portuguese_base_reference'](new),'PT overlay alias mismatch rejected')
for field,value in [('word_errors',-1),('reference_words',0),('wer',.03),('cer',float('nan')),('utterances',49)]:
    current[0]=copy.deepcopy(proof);changed={**score,field:value}
    current[0]['baseline_PT']['validation_pt']={'asr':changed};current[0]['baseline_overlay']['validation_pt']={'asr':changed}
    bad(lambda:ns['portuguese_base_reference'](new),'invalid PT scored counts '+field)
current[0]=proof
ok(ns['portuguese_base_reference'](new)['record']==record,'verified record identity preserved')
follow=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='follow')
ok('required_eval_languages(config)' in ast.unparse(follow),'actual follow completion uses declared7 requirement')
ok(hashlib.sha256(source.read_bytes()).hexdigest()==hashlib.sha256(raw).hexdigest(),'reviewed source unchanged during fixture')
report={'schema':1,'kind':'inactive_Trackio_v10_PT_field_stdlib_verification_v1','passed':True,'source_sha256':hashlib.sha256(raw).hexdigest(),
    'checks':checks,'assertions':len(checks),'rejections':len(rejections),
    'scope':'Four actual source functions extracted by AST with explicit proof-byte-reader stubs. No Trackio import, real proof/history/shard/DB/mirror or GPU access.',
    'actual_Portuguese_baseline_admitted':False,'Trackio_or_DB_imported':False,'GPU_launched':False}
output=HERE/'Trackio_v10_PT_field_stdlib_verification_v1.json'
with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
print(json.dumps({'passed':True,'assertions':len(checks),'rejections':len(rejections),'source_sha256':report['source_sha256'],
    'report_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))
