"""Pure declared recipe/streamed history tests; no actual source/PT admission."""
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish';sys.path.insert(0,str(HERE))
import tsc_uniform_full_transition_v1 as boundary
import record_TSC_uniform_full_source_v1 as creator
checks=[];rejects=[]
def ok(condition,label):
    if not condition:raise AssertionError(label)
    checks.append(label)
def bad(call,label):
    try:call()
    except (ValueError,KeyError,TypeError):checks.append(label);rejects.append(label);return
    raise AssertionError('unexpected acceptance '+label)
declaration=boundary.uniform_declaration(701)
ok(declaration['new_uniform_pass_updates']==31725 and declaration['new_uniform_pass_end_step']==38225
    and declaration['new_uniform_TR_draws']==253800 and declaration['last_two_draws_next_cycle_repeats']==2,
    'whole fullpool horizon with honest final whole-batch repeats')
ok(declaration['all207_weights_and_Adam_states_retained'] and not declaration['fresh_optimizer_or_warmup'],
    'original all207 state preserved without newAdam')
ok(declaration['combined_replay_clips']==1914 and declaration['replay_languages']==['en','es','it','fr','de','pt']
    and declaration['six_language_future_replay_selection_changes_explicitly'] and not declaration['old_five_future_replay_sequence_claimed'],
    'Portuguese explicit append and future selection nuisance')
receipt={'schema':1,'kind':boundary.RECEIPT_KIND,'status':'actual_CLOSED_old_five_qualified_PT_recovery_source','target_run_id':'fixture_uniform_full',
    'source_run_id':boundary.PARENT_RUN,'selected_step':6500,'allow_uniform_fullpass':True,'BEST_not_latest_state':True,
    'declared_uniform_fullpass':declaration,'artifacts':{'best_checkpoint':{'sha256':boundary.BEST_SHA}}}
ok(boundary.validate_receipt(receipt,'fixture_uniform_full',701) is receipt,'new ROOT uniform purpose field schema accepted')
ok(declaration['Portuguese_recovery_grace_new_updates']==1000 and not declaration['initial_all_six_champion_claimed']
    and declaration['all_six_retention_required_for_every_BEST_selection'],'actual new PT failure recovery cannot masquerade as all-six champion')
for key,value in [('selected_step',7000),('BEST_not_latest_state',False),('allow_uniform_fullpass',False),
                  ('kind','ROOT_owned_closed_TSC_best_for_BPE512_acquisition_v1'),('target_run_id','another_run')]:
    changed={**receipt,key:value};bad(lambda value=changed:boundary.validate_receipt(value,'fixture_uniform_full',701),'reject wrong uniform purpose '+key)
for key,value in [('head_lr',5e-5),('lora_lr',1e-4),('new_uniform_pass_updates',1000),('original_lexical_size',8711),('combined_replay_clips',1595),
    ('Portuguese_recovery_grace_new_updates',2000),('initial_all_six_champion_claimed',True),('all_six_retention_required_for_every_BEST_selection',False)]:
    changed=copy.deepcopy(receipt);changed['declared_uniform_fullpass'][key]=value
    bad(lambda value=changed:boundary.validate_receipt(value,'fixture_uniform_full',701),'reject unapproved science '+key)
bad(lambda:boundary.validate_receipt(receipt,'fixture_uniform_full',702),'changed Portuguese seed rejected')
rows=[{'step':i,'text':'escaped \\" and braces{} / İstanbul','a':{'nested':[i,True,None]}} for i in range(1,15)]
with tempfile.TemporaryDirectory(prefix='uniform_boundary_',dir=ROOT/'work/research') as directory:
    path=Path(directory)/'history.json';path.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    ok(boundary._history_prefix(path,8,14)==boundary.canonical(rows[:8]),'streamed selected-history prefix matches canonical list encoding')
    bad(lambda:boundary._history_prefix(path,8,13),'wrong archived completed count rejected')
    wrong=copy.deepcopy(rows);wrong[7]['step']=10;path.write_text(json.dumps(wrong),encoding='utf-8')
    bad(lambda:boundary._history_prefix(path,8,14),'history discontinuity rejected')
    path.write_text(json.dumps(rows)+' bad trailing',encoding='utf-8')
    bad(lambda:boundary._history_prefix(path,8,14),'trailing malformed history rejected')
    path.write_text('[{"step":1,"loss":NaN}]',encoding='utf-8')
    bad(lambda:boundary._history_prefix(path,1,1),'nonfinite JSON history rejected')
ok(boundary.proof_path(SimpleNamespace(run_id='fixture_uniform_full'))==HERE/'runs/fixture_uniform_full'/boundary.PROOF_NAME,
    'actual source350 proof output is scoped and namespaced')
args=creator.parser().parse_args(['--target-run-id','fixture_uniform_full','--portuguese-seed','701',
    '--portuguese-preparation-proof','prep.json','--portuguese-baseline-proof','baseline.json','--validate-only'])
ok(args.validate_only and args.target_run_id=='fixture_uniform_full','ROOT-purpose creator CLI parser')
ok('torch' not in sys.modules and 'trackio' not in sys.modules,'no ML/logger imports')
report={'schema':1,'kind':'inactive_TSC_uniform_full_transition_stdlib_verification_v1','passed':True,
    'checks':checks,'assertions':len(checks),'rejections':len(rejects),
    'helper_sha256':hashlib.sha256(Path(boundary.__file__).read_bytes()).hexdigest(),
    'creator_sha256':hashlib.sha256(Path(creator.__file__).read_bytes()).hexdigest(),
    'scope':'Synthetic ROOT declaration, source horizon, CLI and tiny streamed history only; no actual receipt, complete archived history, source/PT tensors, Portuguese proof or GPU admission.',
    'actual_source_admitted':False,'actual_source350_measured':False,'checkpoint_bytes_read':0,'GPU_launched':False}
path=HERE/'TSC_uniform_full_transition_stdlib_verification_v2.json'
with path.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
print(json.dumps({'passed':True,'checks':len(checks),'rejections':len(rejects),'helper_sha256':report['helper_sha256'],
    'creator_sha256':report['creator_sha256'],'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
