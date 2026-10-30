"""Narrow source-only successor; all candidate imports/operations remain gated."""
import ast,copy,hashlib,json,os
from pathlib import Path
F=Path(__file__).parent;FIX=F.parent
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');R=P/'work/research';S=P/'outputs/parakeet_turkish'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n')as out:json.dump(v,out,indent=2,ensure_ascii=False);out.write('\n')
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
scope=F/'ROOT_source35_and_finite_v1_preexecution_findings_and_successor_preparation_scope_v1.json'
assert pin(scope)['sha256']=='a7b448c8570cd0b5734949df53822c7a414f1ad6ff71918e0e3cd9b084c89a7c'
pending=[]
def begin(old,new):assert not new.exists();return old.read_bytes().decode(),[]
def replace(t,l,a,b,n=1):
 if '\n'in a and t.count(a.replace('\n','\r\n'))==n:a=a.replace('\n','\r\n');b=b.replace('\n','\r\n')
 assert t.count(a)==n,(a[:140],t.count(a),n);l.append({'old_literal':a,'new_literal':b,'expected_count':n});return t.replace(a,b)
def finish(old,new,t,l):
 ast.parse(t);inv=t
 for e in reversed(l):
  assert inv.count(e['new_literal'])==e['expected_count'],e['new_literal'][:150]
  inv=inv.replace(e['new_literal'],e['old_literal'])
 assert inv.encode()==old.read_bytes();pending.append((old,new,t,l))

old=F/'asr18_one_postDEV_gc_refresh_v1.py';new=F/'asr18_one_postDEV_gc_refresh_v2.py';t,l=begin(old,new)
def h(a,b,n=1):
 global t;t=replace(t,l,a,b,n)
h('        self.attempted = False\n','        self.attempted = False\n        self.opportunity_consumed = False\n        self.ineligible_reason = None\n')
h("        require(not self.attempted and not self.completed, 'One event only; no fallback or second attempt')",
  "        require(not self.opportunity_consumed and not self.attempted and not self.completed, 'One first opportunity only; no fallback or second attempt')")
h('        self.attempted = True\n','        self.attempted = True\n        self.opportunity_consumed = True\n')
anchor='    def execute(self, barrier, *, capture, compare, guard, admit, operations, emit):\n'
method='''    def consume_ineligible(self, updates, *, reason, phase, scalar_was_constructed,
                           prefetch_absent, validation_stop, budget_stop, guard, emit):
        """Truthful first-opportunity receipt only; NO join/recreate/GC operation.

        Original terminal normal-stop flags take precedence. Cleaner absence
        requires that the original constructor never ran in this process.
        Missing/dead weak-label workers are never this no-worker alternative.
        """
        require(threading.current_thread() is threading.main_thread(), 'MainThread first-opportunity authority')
        require(not self.opportunity_consumed and not self.attempted and not self.completed,
                'First opportunity already consumed; no later DEV retry')
        require(type(updates) is int and updates == self.expected_update,
                'Only FIRST expected genuine scheduled periodic opportunity')
        require(all(type(v) is bool for v in (scalar_was_constructed, prefetch_absent, validation_stop, budget_stop)),
                'Exact original constructor/absence/normal-stop scalar flags')
        require(callable(guard) and callable(emit), 'Original guard and actual bounded receipt writer')
        if reason == 'original_coincident_normal_stop':
            require(validation_stop or budget_stop, 'Actual original terminal validation/budget flag required')
        elif reason == 'original_cleaner_phase_scalar_never_constructed':
            require(phase == 'cleaner_data_fine_tuning' and not scalar_was_constructed and prefetch_absent
                    and not validation_stop and not budget_stop,
                    'Genuine original cleaner absence only; never weak-label missing/dead fallback')
        else:
            raise ValueError('Unsupported ineligible reason; no GC fallback policy')
        guard()
        self.opportunity_consumed = True
        self.ineligible_reason = reason
        report = {'kind':'ASR18_actual_first_periodic_DEV_refresh_ineligible_v1', 'passed':True,
                  'optimizer_updates':updates, 'phase':phase, 'ineligible_reason':reason,
                  'first_opportunity_consumed':True, 'later_periodic_retry_allowed':False,
                  'GC_operation_attempted':False, 'actual_refresh_performed':False,
                  'scalar_join_drop_or_recreate_performed_here':False,
                  'original_scalar_was_constructed':scalar_was_constructed,
                  'original_prefetch_is_None':prefetch_absent,
                  'original_validation_stop':validation_stop, 'original_budget_stop':budget_stop,
                  'original_normal_exit_path_unchanged':bool(validation_stop or budget_stop),
                  'scalar_closure_fabricated':False, 'host_credit_bytes':0}
        emit(report)
        guard()
        return report

'''
h(anchor,method+anchor)
finish(old,new,t,l)

old=S/'asr18_one_postDEV_gc_refresh_checkpoint_transition_v1.py';new=S/'asr18_one_postDEV_gc_refresh_checkpoint_transition_v2.py';t,l=begin(old,new)
def v(a,b,n=1):
 global t;t=replace(t,l,a,b,n)
v('asr18_one_postDEV_gc_refresh_v1.py','asr18_one_postDEV_gc_refresh_v2.py')
v('from asr18_one_postDEV_gc_refresh_v1 import OnePostDEVRefresh','from asr18_one_postDEV_gc_refresh_v2 import OnePostDEVRefresh')
anchor="        'frequency':'at_most_one_first_genuine_scheduled_periodic_DEV_per_native_process',\n"
v(anchor,anchor+"        'ineligible_first_opportunity_consumed_without_GC':['original_cleaner_phase_scalar_never_constructed','original_coincident_normal_stop'],\n        'preexecution_successor_preparation_scope':record(EVENT/'ROOT_source35_and_finite_v1_preexecution_findings_and_successor_preparation_scope_v1.json'),\n")
finish(old,new,t,l)

old=R/'run_ASR18_BPE9223_resident_RAM_training_v35.py';new=R/'run_ASR18_BPE9223_resident_RAM_training_v36.py';t,l=begin(old,new)
def d(a,b,n=1):
 global t;t=replace(t,l,a,b,n)
d('from asr18_one_postDEV_gc_refresh_checkpoint_transition_v1 import',
  'from asr18_one_postDEV_gc_refresh_checkpoint_transition_v2 import')
start=t.index('                if not postDEV_refresh.attempted:')
end=t.index('            if validation_stop or budget_stop:',start)
prior=t[start:end]
body=prior[prior.index('                    require(not validation_stop'):]
body=body.replace('require(not validation_stop and not budget_stop and stream is None and queued is not None',
                  'require(stream is None and queued is not None')
body=body.replace('require(prefetch is not None and future_ticket is None and prefetch.outstanding is None',
                  "require(prefetch is not None and not prefetch.closed and prefetch.thread.is_alive()\r\n                        and prefetch.worker_status=={'closed':False,'error':None}\r\n                        and future_ticket is None and prefetch.outstanding is None")
# Preserve the old block's precise RAW endings while adding one indentation level.
body=''.join('    '+line if line.strip()else line for line in body.splitlines(keepends=True))
prefix='''                if not postDEV_refresh.opportunity_consumed:
                    require(optimizer.updates==postDEV_refresh.expected_update,
                        'Only first genuine scheduled periodic DEV in this native process')
                    if validation_stop or budget_stop:
                        postDEV_refresh.consume_ineligible(optimizer.updates,reason='original_coincident_normal_stop',
                            phase=engine.schedule.phase,scalar_was_constructed=bool(prefetch_generation),
                            prefetch_absent=prefetch is None,validation_stop=validation_stop,budget_stop=budget_stop,guard=guard,
                            emit=lambda value:write_json(invocation/'actual_one_postDEV_gc_refresh_ineligible.json',value))
                    elif prefetch is None and engine.schedule.phase=='cleaner_data_fine_tuning' and prefetch_generation==0:
                        postDEV_refresh.consume_ineligible(optimizer.updates,reason='original_cleaner_phase_scalar_never_constructed',
                            phase=engine.schedule.phase,scalar_was_constructed=False,prefetch_absent=True,
                            validation_stop=False,budget_stop=False,guard=guard,
                            emit=lambda value:write_json(invocation/'actual_one_postDEV_gc_refresh_ineligible.json',value))
                    else:
'''.replace('\n','\r\n')
d(prior,prefix+body)
finish(old,new,t,l)

old=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v35.py';new=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v36.py';t,l=begin(old,new)
def w(a,b,n=1):
 global t;t=replace(t,l,a,b,n)
w('run_ASR18_BPE9223_resident_RAM_training_v35','run_ASR18_BPE9223_resident_RAM_training_v36',10)
w('asr18_one_postDEV_gc_refresh_checkpoint_transition_v1','asr18_one_postDEV_gc_refresh_checkpoint_transition_v2',2)
finish(old,new,t,l)

# Freeze only after all AST and inverse checks, never importing a candidate.
for old,new,t,l in pending:new.write_bytes(t.encode())
records=[{'old':pin(old),'new':pin(new),'replacements':l,'full_RAW_forward_inverse_exact':True,
          'newline_policy':'Exact RAW literal replacements; original mixed endings preserved'}for old,new,t,l in pending]
base=json.loads((F/'proposed_source35_configuration_pending_CPU_and_actual_stop_v1.json').read_text())
base['one_postDEV_gc_refresh_policy']['helper_source']=pin(F/'asr18_one_postDEV_gc_refresh_v2.py')
base['one_postDEV_gc_refresh_policy']['ineligible_first_opportunity_consumed_without_GC']=[
 'original_cleaner_phase_scalar_never_constructed','original_coincident_normal_stop']
base['one_postDEV_gc_refresh_policy']['preexecution_successor_preparation_scope']=pin(scope)
cp=F/'proposed_source36_configuration_pending_CPU_and_actual_stop_v1.json';write(cp,base)
proof={'kind':'ASR18_INACTIVE_postDEV_helper2_source36_exact_successor_inverse_v1','ROOT_preparation_scope':pin(scope),
 'sources':records,'configuration':pin(cp),'capture_source_unchanged':pin(F/'asr18_postDEV_refresh_state_capture_v1.py'),
 'frozen35_and_helper1_unexecuted_history':True,'candidate_import_or_execution':False}
# The complete inverse chain is independently consumable, including direct admitted33.
earlier=json.loads((F/'source34_to35_resolver_static_binding_and_direct33_RAW_inverse_v1.json').read_text())
for row,which in ((records[2],0),(records[3],1)):
 prior=earlier['sources'][which];all_edits=prior['entire33_to35_replacements']+row['replacements']
 inverse=Path(row['new']['path']).read_bytes().decode()
 for e in reversed(all_edits):
  assert inverse.count(e['new_literal'])==e['expected_count'];inverse=inverse.replace(e['new_literal'],e['old_literal'])
 assert inverse.encode()==Path(prior['proper_admitted33']['path']).read_bytes()
 row['proper_admitted33']=prior['proper_admitted33'];row['entire33_to36_replacements']=all_edits;row['full_admitted33_RAW_inverse_exact']=True
oldtree=ast.parse(Path(records[2]['proper_admitted33']['path']).read_bytes());newtree=ast.parse(Path(records[2]['new']['path']).read_bytes())
main=lambda tr:next(n for n in tr.body if isinstance(n,ast.FunctionDef)and n.name=='main')
native=lambda tr:next(n for n in ast.walk(main(tr))if isinstance(n,ast.For)and isinstance(n.target,ast.Tuple)
    and [x.id for x in n.target.elts if isinstance(x,ast.Name)]==['index','entry'])
assert ast.dump(native(oldtree),include_attributes=False)==ast.dump(native(newtree),include_attributes=False)
proof['entire_native_update_for_loop_AST_exact_to33']=True
ledger=F/'helper2_source35_to36_and_full33_RAW_inverse_v1.json';write(ledger,proof)
hp=F/'inactive_source36_one_postDEV_refresh_source_handoff_v1.json'
write(hp,{'kind':'ASR18_INACTIVE_source36_one_postDEV_refresh_source_handoff_v1','ROOT_successor_preparation_scope':pin(scope),
 'helper_source':records[0]['new'],'capture_source':pin(F/'asr18_postDEV_refresh_state_capture_v1.py'),
 'transition_validator':records[1]['new'],'driver_source':records[2]['new'],'wrapper_source':records[3]['new'],
 'proposed_source_configuration':pin(cp),'exact_RAW_and_AST_inverse':pin(ledger),
 'proper_source33_base':pin(FIX.parent/'resident_RAM_original_interval170469_failure_actual_v1/production_configuration_source32_v1/configuration.json'),
 'original_seven_runtime_binding_kinds':['worker_driver_source','parent_driver_source','resident_RAM_stream_configuration',
  'resident_branch_namespace','checkpoint_interval_updates','source_checkpoint','source_configuration'],
 'future_binding_values':'actual driver36; BOTH stream and source configuration bind the same NEW materialized final configuration; original interval200; actual latest full saved checkpoint after separately admitted clean intentional stop',
 'future_invocation':25,'future_journal':'commits_000025.jsonl','future_output':'resident_RAM_training_actual_26',
 'completed_CPU_review':None,'actual_intentional_transition':None,'production_or_CPU_execution_admitted':False,
 'first_opportunity':'Once consumed: eligible genuine joined scalar event OR truthful cleaner-never-constructed no-GC receipt OR original terminal normal-stop no-GC receipt. Weak missing/dead is fatal; no later retry.',
 'validation':{'start':'restore+20','end':'max(start+1600,nextscheduledDEV+1020)',
  'whole_first400_final400_two_postDEV400_all_pauses_saves_originalrows':True,
  'save_excluded_sustained_decline_max_percent':5,'near_original_400_inclusive_target':'~75+ vs76.57 with DEV scopes labelled',
  'full401072_no_finite_stop':True}})
print(json.dumps({'helper':records[0]['new'],'validator':records[1]['new'],'driver':records[2]['new'],
 'wrapper':records[3]['new'],'config':pin(cp),'ledger':pin(ledger),'handoff':pin(hp)}))
