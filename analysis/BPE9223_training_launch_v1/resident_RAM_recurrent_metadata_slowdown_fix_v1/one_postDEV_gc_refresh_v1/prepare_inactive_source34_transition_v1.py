"""Immutable source/AST/config preparation only. NO candidate import or GC call."""
import ast,copy,difflib,hashlib,json,os
from pathlib import Path
F=Path(__file__).parent;FIX=F.parent
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
BASE=FIX.parent/'resident_RAM_original_interval170469_failure_actual_v1/production_configuration_source32_v1/configuration.json'
POLICY_KEY='one_postDEV_gc_refresh_policy';TRANSITION_KEY='prior_resident_RAM_one_postDEV_gc_refresh_checkpoint_transition'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
assert pin(BASE)['sha256']=='372de7cb3cd96427fed29edf36ad15f95c4cad063572280de031d488a5a135f2'
pending=[]
def start(old,new):
 assert not new.exists();return old.read_bytes().decode(),[]
def change(t,l,a,b,n=1):
 if '\n'in a and t.count(a.replace('\n','\r\n'))==n:a=a.replace('\n','\r\n');b=b.replace('\n','\r\n')
 assert t.count(a)==n,(a[:140],t.count(a),n)
 l.append({'old_literal':a,'new_literal':b,'expected_count':n});return t.replace(a,b)
def verify(old,new,t,l):
 ast.parse(t);inv=t
 for e in reversed(l):
  assert inv.count(e['new_literal'])==e['expected_count'],e['new_literal'][:150]
  inv=inv.replace(e['new_literal'],e['old_literal'])
 assert inv.encode()==old.read_bytes()
 pending.append((old,new,t,l))

old=S/'asr18_fresh_signature_collect_checkpoint_transition_v1.py'
new=S/'asr18_one_postDEV_gc_refresh_checkpoint_transition_v1.py';t,l=start(old,new)
def v(a,b,n=1):
 global t;t=change(t,l,a,b,n)
v('INACTIVE typed intentional source30 checkpoint transition for fresh signatures and original collect observation.',
  'INACTIVE typed intentional source33 checkpoint transition for one first-periodic-DEV GC refresh.')
v("SOURCE30_BASE=F/'production_configuration_v1/configuration.json'",
  "SOURCE33_BASE=F.parent/'resident_RAM_original_interval170469_failure_actual_v1/production_configuration_source32_v1/configuration.json'\nEVENT=F/'one_postDEV_gc_refresh_v1'")
v('def checked_fresh_signature_collect_checkpoint_resume(', 'def checked_one_postDEV_refresh_checkpoint_resume(')
v("base['prior_resident_RAM_fresh_signature_collect_checkpoint_transition']", "base['prior_resident_RAM_one_postDEV_gc_refresh_checkpoint_transition']")
v('ASR18_actual_source30_checkpoint_fresh_signature_collect_transition_v1','ASR18_actual_source33_checkpoint_one_postDEV_gc_refresh_transition_v1')
v("value['owned_session']==72166 and value['old_owner_invocation']==22 and value['new_owner_invocation']==23",
  "value['owned_session']==65067 and value['old_owner_invocation']==24 and value['new_owner_invocation']==25")
v("    checked_fresh_signature_collect_policy(base)\n    previous=read_json(SOURCE30_BASE);restored=copy.deepcopy(base)\n    del restored['prior_resident_RAM_fresh_signature_collect_checkpoint_transition']\n    del restored['fresh_metadata_signature_and_collect_observation_policy']\n    require(restored==previous,'Only fresh signature/collect observation policy and actual transition provenance differ from proper source30 base')",
  "    checked_one_postDEV_refresh_policy(base)\n    previous=read_json(SOURCE33_BASE);restored=legacy_source33_configuration(base)\n    require(restored==previous,'Only new GC lifecycle policy and actual intentional transition differ from proper source33 base')")
v('run_ASR18_BPE9223_resident_RAM_training_v30.py','run_ASR18_BPE9223_resident_RAM_training_v33.py')
v('run_ASR18_BPE9223_resident_RAM_autocontinue_v30.py','run_ASR18_BPE9223_resident_RAM_autocontinue_v33.py')
v('fastmss_resident_feature_stream_v11.py','fastmss_resident_feature_stream_v13.py')
v('invocations/owner_000022','invocations/owner_000024',3)
v('Exact actual source30/CPU11 invocation22','Exact actual source33/CPU13 invocation24')
v('ASR18_fresh_signature_collect_checkpoint_stop_v1','ASR18_one_postDEV_refresh_checkpoint_stop_v1')
v('ASR18_reviewed_fresh_signature_collect_checkpoint_cooperative_stop_v1','ASR18_reviewed_one_postDEV_refresh_checkpoint_cooperative_stop_v1')
v("intent['owned_session']==72166 and intent['old_owner_invocation']==22", "intent['owned_session']==65067 and intent['old_owner_invocation']==24")
v("who[0]['pid']==40012 and abs(who[0]['create_time']-1792743161.6231525)<.001",
  "who[0]['pid']==3652 and abs(who[0]['create_time']-1792764340.2396317)<.001")
v('Same admitted live owner and honest old source30/CPU11 provenance','Same admitted live owner and honest old source33/CPU13 provenance')
v('saved>=166000','saved>=174000')
v("tool['owned_session']==72166", "tool['owned_session']==65067")
v('commits_000022.jsonl','commits_000024.jsonl')
v('Only compact preserved new tail and original22 STAT, never historical journal/prefix/whole','Only compact preserved new tail and original24 STAT, never historical journal/prefix/whole')
v('Closed source30 whole22 remains untouched','Closed source33 whole24 remains untouched')
v('Old22 remains authoritative until after exact restore and new prefix-only branch','Old24 remains authoritative until after exact restore and new prefix-only branch')
begin=t.index('def checked_fresh_signature_collect_policy(base):')
end=t.index('def checked_archive_representation_policy(base):',begin)
old_policy=t[begin:end]
new_policy='''def legacy_source33_configuration(base):
    previous=read_json(SOURCE33_BASE);restored=copy.deepcopy(base)
    require('one_postDEV_gc_refresh_policy'in restored and
        'prior_resident_RAM_one_postDEV_gc_refresh_checkpoint_transition'in restored,
        'Both explicit new operational fields, no historical policy weakening')
    del restored['one_postDEV_gc_refresh_policy']
    del restored['prior_resident_RAM_one_postDEV_gc_refresh_checkpoint_transition']
    require(restored==previous,'Every proper source33 science/resource/CPU13/MAX2/225 value is exact')
    return restored


def checked_one_postDEV_refresh_policy(base):
    legacy_source33_configuration(base)
    policy=base['one_postDEV_gc_refresh_policy']
    require(policy=={
        'kind':'ASR18_one_first_periodic_DEV_quiescent_gc_refresh_policy_v1',
        'helper_source':record(EVENT/'asr18_one_postDEV_gc_refresh_v1.py'),
        'capture_source':record(EVENT/'asr18_postDEV_refresh_state_capture_v1.py'),
        'original_fingerprint_source':record(S/'asr18_live_state_fingerprint_v1.py'),
        'original_gc_scope_source':record(P/'work/research/run_ASR18_native_throughput_component_v15.py'),
        'original_native_source':record(S/'asr18_BPE9223_grouped_RAM_native_v1.py'),
        'original_scalar_prefetch_source':record(S/'asr18_BPE9223_private_scalar_prefetch_v1.py'),
        'preparation_scope':record(F/'ROOT_one_postDEV_gc_refresh_inactive_preparation_scope_v1.json'),
        'semantic_capture_amendment':record(F/'ROOT_one_postDEV_gc_refresh_inactive_preparation_semantic_verification_amendment_v1.json'),
        'frequency':'at_most_one_first_genuine_scheduled_periodic_DEV_per_native_process',
        'operations':['unfreeze','captured_actual_original_full_collect','freeze'],
        'scalar_join_drop_before_refresh_recreate_original_after':True,
        'exact_queued_plan_object_and_pin_retained_no_draw':True,
        'additional_bookkeeping_reserve_bytes':16777216,
        'full_event_commit_threshold_bytes':4535212672,
        'full_event_physical_threshold_bytes':1785212672,
        'maximum_original_fingerprint_transfer_slice_bytes':4194304,
        'two_added_idle_fingerprint_explicit_synchronizations':True,
        'all_original_native_perupdate_GC_sync_cache_calls_and_final_GC_teardown_unchanged':True,
        'opaque_framework_cycle_absence_proved':False,
        'completed_CPU_review':policy['completed_CPU_review']},'Exact one-event changed GC lifecycle hypothesis')
    proof=policy['completed_CPU_review']
    require(type(proof)is dict and record(proof['path'])==proof,'Genuine completed finite CPU ROOT review required')
    review=read_json(proof['path'])
    require(review['accepted']is True and review['reviewer']=='/root'and review['scope']==policy['kind']
        and review['helper_source']==policy['helper_source']and review['capture_source']==policy['capture_source']
        and record(review['actual_CPU_control_report']['path'])==review['actual_CPU_control_report']
        and review['all_exact_metadata_identities_dead']is True,
        'Real CPython/lifecycle proof and exact metadata closure; scientific GPU portion remains source/METHOD_STUB')
    import sys
    if str(EVENT)not in sys.path:sys.path.insert(0,str(EVENT))
    from asr18_one_postDEV_gc_refresh_v1 import OnePostDEVRefresh
    from asr18_postDEV_refresh_state_capture_v1 import capture_idle_state,compare_idle_state
    return OnePostDEVRefresh,capture_idle_state,compare_idle_state


'''
v(old_policy,new_policy)
# Exact compact original Monitor closure is a required transition gate.
anchor="    failure=read_json(e['actual_failure']['path']);tool=read_json(e['actual_TOOL']['path'])\n"
v(anchor,anchor+"    monitor=read_json(e['monitor_completion']['path'])\n    require(monitor['joined']is True and monitor['clean']is True and monitor['error']is None,\n        'Only clean original Monitor cooperative stop, never concurrent abnormal guard failure')\n")
verify(old,new,t,l)

old=R/'run_ASR18_BPE9223_resident_RAM_training_v33.py';new=R/'run_ASR18_BPE9223_resident_RAM_training_v34.py';t,l=start(old,new)
def d(a,b,n=1):
 global t;t=change(t,l,a,b,n)
anchor='    native_pressure_policy,wait_for_native_commit_pressure=checked_native_commit_pressure_policy(base)\n'
d(anchor,anchor+"    from asr18_one_postDEV_gc_refresh_checkpoint_transition_v1 import (checked_one_postDEV_refresh_policy,\n        checked_one_postDEV_refresh_checkpoint_resume,legacy_source33_configuration)\n    OnePostDEVRefresh,capture_idle_state,compare_idle_state=checked_one_postDEV_refresh_policy(base)\n    legacy_base=legacy_source33_configuration(base)\n    gc_transition=None if a.gc_refresh_resume is None else checked_one_postDEV_refresh_checkpoint_resume(\n        record(a.gc_refresh_resume),base=base,folder=folder,pointer_pin=pointer_pin)\n    require(gc_transition is not None or a.prior_process is not None,\n        'New GC policy enters only exact actual source33 intentional stop or genuine later normal0')\n    if gc_transition is not None:\n        require(a.invocation==25 and all(x is None for x in (a.prior_process,a.eighth_original_resume,a.fresh_signature_resume,\n            a.canonical_archive_resume,a.lookahead_resume,a.cushion225_resume,a.postrestore_selector_resume,a.packet_credit_resume,\n            a.native_headroom_resume,a.abnormal_resume,a.original_interval_resume,a.source_transition_resume,\n            a.DEV_frontend_resume,a.DEV_admission_resume,a.periodic_quiescence_resume)),\n            'Only actual source33 intentional checkpoint stop, no historical or normal0 alias')\n")
d('    checked_eighth_original_policy(base)\n','    checked_eighth_original_policy(legacy_base)\n')
d('    transition=signature_transition if signature_transition is not None else(',
  '    transition=gc_transition if gc_transition is not None else(signature_transition if signature_transition is not None else(')
d('        record(a.source_transition_resume),base=base,folder=folder,pointer_pin=pointer_pin))))',
  '        record(a.source_transition_resume),base=base,folder=folder,pointer_pin=pointer_pin)))))')
d('    preserved_tail=signature_transition if signature_transition is not None else(',
  '    preserved_tail=gc_transition if gc_transition is not None else(signature_transition if signature_transition is not None else(')
d("(transition if transition is not None and transition['unsaved_tail_bytes'] else None)))))",
  "(transition if transition is not None and transition['unsaved_tail_bytes'] else None))))))")
d("'actual_transition':record(a.fresh_signature_resume if a.fresh_signature_resume is not None else",
  "'actual_transition':record(a.gc_refresh_resume if a.gc_refresh_resume is not None else a.fresh_signature_resume if a.fresh_signature_resume is not None else")
d("(a.eighth_original_resume is not None or a.fresh_signature_resume is not None or a.canonical_archive_resume is not None or a.lookahead_resume is not None or a.cushion225_resume is not None or a.packet_credit_resume is not None)",
  "(a.gc_refresh_resume is not None or a.eighth_original_resume is not None or a.fresh_signature_resume is not None or a.canonical_archive_resume is not None or a.lookahead_resume is not None or a.cushion225_resume is not None or a.packet_credit_resume is not None)")
d("'predecessor':record(a.eighth_original_resume if a.eighth_original_resume is not None else",
  "'predecessor':record(a.gc_refresh_resume if a.gc_refresh_resume is not None else a.eighth_original_resume if a.eighth_original_resume is not None else")
d('    prefetch=None;prefetch_close_attempted=False\n',
  '    prefetch=None;prefetch_generation=0;prefetch_close_attempted_generation=None\n')
d('        nonlocal prefetch_close_attempted\n        if prefetch is None or prefetch_close_attempted:return\n        prefetch_close_attempted=True\n',
  '        nonlocal prefetch_close_attempted_generation\n        if prefetch is None or prefetch_close_attempted_generation==prefetch_generation:return\n        prefetch_close_attempted_generation=prefetch_generation\n')
anchor='        original_collect_observer=scopes.enter_context(ObserveOriginalCollectScope())\n'
d(anchor,anchor+'        postDEV_refresh=OnePostDEVRefresh(initial_updates)\n')
d('            prefetch=PrivateScalarPrefetch(engine.context,guard=guard,admit=admit)\n',
  '            prefetch=PrivateScalarPrefetch(engine.context,guard=guard,admit=admit)\n            prefetch_generation+=1\n')
anchor='                record_periodic(dev_pin,score,preservation,False)\n'
event='''                if not postDEV_refresh.attempted:
                    require(optimizer.updates==postDEV_refresh.expected_update,
                        'Only first genuine scheduled periodic DEV in this native process')
                    require(not validation_stop and not budget_stop and stream is None and queued is not None
                        and optimizer.updates==last_checkpoint==latest['committed_updates']
                        and latest['full_state_verified']is True and not optimizer.poisoned and not engine.schedule.poisoned,
                        'Successful complete persistence/DEV before new preparation, no normal-stop substitution')
                    cpu_closed=CPU_quiescence['CPU_actual0']
                    phase_preserved=read_json(preservation['path'])
                    require(cpu_closed['actual_Popen_wait_code']==0 and cpu_closed['transport_threads_joined']is True
                        and cpu_closed['pipe_handles_closed']is True
                        and cpu_closed['packet_credit']['all_owned_kernel_handles_closed']is True
                        and cpu_closed['packet_credit_owner']['all_owned_kernel_handles_closed']is True
                        and phase_preserved['actual_phase_private']['joined']is True
                        and phase_preserved['actual_phase_private']['error']is None
                        and phase_preserved['all207_threeAdam_all4RNG_unchanged']is True
                        and phase_preserved['both_cursors_journal_and_nextdraw_unchanged']is True,
                        'Genuine feature closure and successful real DEV preservation/monitor join')
                    require(prefetch is not None and future_ticket is None and prefetch.outstanding is None
                        and prefetch.held_result is None and engine.context.resolver is prefetch.original,
                        'Fully adopted idle original scalar worker; no unconsumed work to discard')
                    import weakref
                    old_scalar_ref=weakref.ref(prefetch)
                    prefetch_close_attempted_generation=prefetch_generation
                    scalar_closed=prefetch.close(discard_unconsumed=False)
                    require(scalar_closed['joined']is True and scalar_closed['worker_connections_closed_on_worker']is True
                        and scalar_closed['discarded_unconsumed_request']is False
                        and not prefetch.thread.is_alive()and prefetch.worker_status=={'closed':True,'error':None},
                        'Actual empty worker join/DB+locks closure before dropping old object/frame')
                    write_json(invocation/'first_periodic_refresh_scalar_worker_closure.json',
                        dict(scalar_closed,source=prefetch_policy['source'],generation=prefetch_generation))
                    prefetch=None
                    require(old_scalar_ref()is None,'Old joined scalar worker object/frame is genuinely dropped')
                    del old_scalar_ref
                    # The exact queued tuple/plan/pin and scientific owners stay
                    # live. These completed aliases have no remaining use on the
                    # non-fullpass periodic branch; never release a packet twice.
                    del plan,plan_pin,coordinator,writer,before,nxt,caller,ptr,proof,ck
                    del tx,entry,rows,metadata,native,source,binding,evidence,state,CPU_quiescence
                    del dev_pin,score,preservation,cpu_closed,phase_preserved,scalar_closed
                    refresh_barrier={'kind':'ASR18_actual_first_periodic_DEV_refresh_barrier_v1','updates':optimizer.updates,
                        'genuine_periodic_DEV_returned_and_recorded':optimizer.updates in evaluated,
                        'checkpoint_full_persistence_same_frontier':latest['full_state_verified']and latest['committed_updates']==optimizer.updates,
                        'feature_CPU_actual_wait0_joined_pipes_and_both_credit_sets_closed':True,
                        'DEV_preservation_and_phase_monitor_join_succeeded':True,
                        'optimizer_idle_unpoisoned':not optimizer.poisoned and not engine.schedule.poisoned,
                        'scalar_empty_joined_closed_errorNone_worker_DB_and_locks_closed':True,
                        'old_scalar_object_and_frame_dropped':prefetch is None,
                        'parent_original_context_and_locks_continuous':not engine.context.closed and len(engine.context.resolver.metadata_locks.handles)==3,
                        'no_live_packet_or_arrays':True,'checkpoint_writer_and_completed_aliases_dropped':True,
                        'queued_original_same_object_and_pin':record(queued[1]['path'])==queued[1],
                        'main_Monitor_active':monitor.thread.is_alive()and monitor.error is None,
                        'stream_is_None':stream is None,'future_ticket_is_None':future_ticket is None,
                        'resolver_is_original':engine.context.resolver is signature_scope.context.resolver,
                        'capture_contract':'original_bounded_live_fingerprint_and_identity_manifest_v1'}
                    def refresh_admission(increment):
                        require(increment==16777216,'Exact additional event reserve, no freed CPU/DEV credit')
                        E_reserve(increment);original=admit('one_postDEV_refresh_additional_bookkeeping',increment)
                        sample=monitor._observe();system=sample['system_memory']
                        require(system['commit_available_bytes']>=4535212672 and system['physical_available_bytes']>=1785212672,
                            'Full original225+FULLprefetch thresholds plus separate16MiB event reserve')
                        return {'passed':True,'original_admission':original,'sample':sample,
                            'additional_bookkeeping_reserve_bytes':increment,'required_commit_bytes':4535212672,
                            'required_physical_bytes':1785212672,'host_credit_bytes':0}
                    postDEV_refresh.execute_real(refresh_barrier,
                        capture=lambda:capture_idle_state(optimizer=optimizer,torch=torch,guard=guard,schedule=engine.schedule,
                            context=engine.context,journal=journal,lifecycle=lifecycle,periodic=periodic,queued=queued,
                            latest=latest,completed=completed),compare=compare_idle_state,guard=guard,admit=refresh_admission,
                        original_collect_observer=original_collect_observer,
                        emit=lambda value:write_json(invocation/'actual_one_postDEV_gc_refresh.json',value))
                    del refresh_barrier
                    prefetch=PrivateScalarPrefetch(engine.context,guard=guard,admit=admit)
                    prefetch_generation+=1
                    write_json(invocation/'private_metadata_worker_after_first_periodic_refresh_admission.json',
                        dict(prefetch.admission,generation=prefetch_generation,only_after_completed_refresh=postDEV_refresh.completed))
'''
d(anchor,anchor+event)
d("p.add_argument('--eighth-original-resume');", "p.add_argument('--gc-refresh-resume');p.add_argument('--eighth-original-resume');")
verify(old,new,t,l)

old=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v33.py';new=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v34.py';t,l=start(old,new)
def w(a,b,n=1):
 global t;t=change(t,l,a,b,n)
w('run_ASR18_BPE9223_resident_RAM_training_v33','run_ASR18_BPE9223_resident_RAM_training_v34',10)
anchor="    configuration=read_json(a.configuration);policy=configuration['private_scalar_prefetch_policy']\n"
w(anchor,anchor+"    from asr18_one_postDEV_gc_refresh_checkpoint_transition_v1 import (checked_one_postDEV_refresh_policy,\n        checked_one_postDEV_refresh_checkpoint_resume,legacy_source33_configuration)\n    checked_one_postDEV_refresh_policy(configuration)\n    legacy_configuration=legacy_source33_configuration(configuration)\n    require(review['one_postDEV_gc_refresh_policy']==configuration['one_postDEV_gc_refresh_policy'],\n        'Exact one-event GC lifecycle and original byte/science/guard policy')\n")
w('require(a.eighth_original_resume is None and a.fresh_signature_resume is None',
  'require(a.gc_refresh_resume is None and a.eighth_original_resume is None and a.fresh_signature_resume is None')
anchor='    elif a.eighth_original_resume is not None:\n'
w(anchor,"    elif a.gc_refresh_resume is not None:\n        require(all(x is None for x in (a.eighth_original_resume,a.fresh_signature_resume,a.canonical_archive_resume,a.lookahead_resume,\n            a.cushion225_resume,a.postrestore_selector_resume,a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume,\n            a.original_interval_resume,a.source_transition_resume,a.DEV_frontend_resume,a.DEV_admission_resume,a.periodic_quiescence_resume))\n            and review['gc_refresh_resume']==record(a.gc_refresh_resume)\n            and review['gc_refresh_transition_validator']==record(S/'asr18_one_postDEV_gc_refresh_checkpoint_transition_v1.py')\n            and a.start_invocation==review['initial_owner_invocation']==25,\n            'Only exact reviewed actual source33 intentional checkpoint stop, never normal0/historical failure')\n"+anchor)
w('    checked_eighth_original_policy(configuration)\n','    checked_eighth_original_policy(legacy_configuration)\n')
anchor='    if a.eighth_original_resume is not None:\n'
w(anchor,"    if a.gc_refresh_resume is not None:\n        checked_one_postDEV_refresh_checkpoint_resume(record(a.gc_refresh_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))\n"+anchor)
w("            if a.eighth_original_resume is not None:command+=['--eighth-original-resume',a.eighth_original_resume]",
  "            if a.gc_refresh_resume is not None:command+=['--gc-refresh-resume',a.gc_refresh_resume]\n            elif a.eighth_original_resume is not None:command+=['--eighth-original-resume',a.eighth_original_resume]")
w("p.add_argument('--eighth-original-resume');", "p.add_argument('--gc-refresh-resume');p.add_argument('--eighth-original-resume');")
verify(old,new,t,l)

# All source assertions/inverses precede immutable outputs.
for old,new,t,l in pending:new.write_bytes(t.encode())
records=[]
for old,new,t,l in pending:
 diff=F/(new.stem+'.diff')
 with diff.open('x',encoding='utf-8',newline='\n')as out:
  out.write(''.join(difflib.unified_diff(old.read_text().splitlines(True),t.splitlines(True),fromfile=str(old),tofile=str(new))))
 records.append({'old':pin(old),'new':pin(new),'replacements':l,'full_old_RAW_inverse_exact':True,
                 'newline_policy':'RAW literal edits retain all original mixed line endings; no normalization','diff':pin(diff)})
oldtree=ast.parse((R/'run_ASR18_BPE9223_resident_RAM_training_v33.py').read_bytes())
newtree=ast.parse((R/'run_ASR18_BPE9223_resident_RAM_training_v34.py').read_bytes())
funcs=lambda tree:{n.name:ast.dump(n,include_attributes=False)for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
of,nf=funcs(oldtree),funcs(newtree);unchanged=[n for n in of if n!='main'];assert all(of[n]==nf[n]for n in unchanged)
getmain=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
nested=lambda tree:{n.name:ast.dump(n,include_attributes=False)for n in getmain(tree).body if isinstance(n,ast.FunctionDef)}
on,nn=nested(oldtree),nested(newtree)
old_nested_unchanged=[n for n in on if n!='close_private_metadata'];assert all(on[n]==nn[n]for n in old_nested_unchanged)
native_loop=lambda tree:next(n for n in ast.walk(getmain(tree))if isinstance(n,ast.For)
    and isinstance(n.target,ast.Tuple)and [x.id for x in n.target.elts if isinstance(x,ast.Name)]==['index','entry'])
assert ast.dump(native_loop(oldtree),include_attributes=False)==ast.dump(native_loop(newtree),include_attributes=False)
candidate=json.loads(BASE.read_text())
candidate[POLICY_KEY]={
 'kind':'ASR18_one_first_periodic_DEV_quiescent_gc_refresh_policy_v1',
 'helper_source':pin(F/'asr18_one_postDEV_gc_refresh_v1.py'),'capture_source':pin(F/'asr18_postDEV_refresh_state_capture_v1.py'),
 'original_fingerprint_source':pin(S/'asr18_live_state_fingerprint_v1.py'),
 'original_gc_scope_source':pin(R/'run_ASR18_native_throughput_component_v15.py'),
 'original_native_source':pin(S/'asr18_BPE9223_grouped_RAM_native_v1.py'),
 'original_scalar_prefetch_source':pin(S/'asr18_BPE9223_private_scalar_prefetch_v1.py'),
 'preparation_scope':pin(FIX/'ROOT_one_postDEV_gc_refresh_inactive_preparation_scope_v1.json'),
 'semantic_capture_amendment':pin(FIX/'ROOT_one_postDEV_gc_refresh_inactive_preparation_semantic_verification_amendment_v1.json'),
 'frequency':'at_most_one_first_genuine_scheduled_periodic_DEV_per_native_process',
 'operations':['unfreeze','captured_actual_original_full_collect','freeze'],
 'scalar_join_drop_before_refresh_recreate_original_after':True,'exact_queued_plan_object_and_pin_retained_no_draw':True,
 'additional_bookkeeping_reserve_bytes':16777216,'full_event_commit_threshold_bytes':4535212672,
 'full_event_physical_threshold_bytes':1785212672,'maximum_original_fingerprint_transfer_slice_bytes':4194304,
 'two_added_idle_fingerprint_explicit_synchronizations':True,
 'all_original_native_perupdate_GC_sync_cache_calls_and_final_GC_teardown_unchanged':True,
 'opaque_framework_cycle_absence_proved':False,'completed_CPU_review':None}
candidate[TRANSITION_KEY]=None
inverse=copy.deepcopy(candidate);del inverse[POLICY_KEY];del inverse[TRANSITION_KEY];assert inverse==json.loads(BASE.read_text())
cp=F/'proposed_source34_configuration_pending_CPU_and_actual_stop_v1.json';write(cp,candidate)
ledger=F/'source33_to34_one_postDEV_refresh_entire_RAW_AST_inverse_v1.json'
write(ledger,{'kind':'ASR18_INACTIVE_source33_to34_postDEV_refresh_full_RAW_AST_inverse_v1','sources':records,
 'all_old_nonentry_AST_exact':unchanged,'all_old_nested_AST_exact_except_per_instance_close':old_nested_unchanged,
 'entire_native_update_for_loop_AST_exact':True,'original_checkpoint_and_DEV_body_order_retained':True,
 'proper_source33_base':pin(BASE),'configuration':pin(cp),'only_added_operational_fields':[POLICY_KEY,TRANSITION_KEY],
 'full_config_inverse_after_only_two_added_fields':True,'candidate_import_or_operation_executed':False})
handoff=F/'inactive_source34_postDEV_refresh_source_configuration_handoff_v1.json'
write(handoff,{'kind':'ASR18_INACTIVE_source34_one_postDEV_refresh_handoff_v1',
 'driver_source':records[1]['new'],'wrapper_source':records[2]['new'],'transition_validator':records[0]['new'],
 'entire_RAW_AST_inverse':pin(ledger),'proposed_source_configuration':pin(cp),'proper_source33_base':pin(BASE),
 'finite_CPU_source_handoff':pin(F/'inactive_one_postDEV_refresh_finite_CPU_source_handoff_v1.json'),
 'original_seven_runtime_binding_kinds':['worker_driver_source','parent_driver_source','resident_RAM_stream_configuration',
  'resident_branch_namespace','checkpoint_interval_updates','source_checkpoint','source_configuration'],
 'future_binding_values':'driver34/wrapper actual command source34; both stream and source configuration must bind the same NEW materialized final configuration; checkpoint from actual clean coherent stop only',
 'future_invocation':25,'future_journal':'commits_000025.jsonl','future_output':'resident_RAM_training_actual_26',
 'actual_stop_and_latest_verified_checkpoint':None,'completed_CPU_review':None,
 'selected_production_GC_policy_or_execution_binding':None,
 'future_typed_regression_requirement':'Execute entire actual new pre-restore selector+factory+post-restore branch dictionary and wrapper require+forwarding with actual closure, strict stale/None/wrong-selector/STAT-only negatives; source26 bug must not recur.',
 'validation':{'start':'restored+20','end':'max(start+1600,nextscheduledDEV+1020)',
  'all_contiguous_original20rows_all_natural_saves_realDEV_allpauses':True,
  'whole_first400_final400_two_postDEV400':True,'near_original_inclusive400_target':'~75+ versus76.569475, DEV-containing comparison explicitly labelled',
  'save_excluded_first_to_final_sustained_decline_max_percent':5,'no_finite_production_stop':True},
 'retention_risk':'Opaque reachable cycles frozen now may become unreachable later and persist until original final unfreeze. CPU proof cannot exclude framework cycles or prove production fit/speed.',
 'training_source33_65067_unchanged':True,'production_or_CPU_proof_or_stop_admitted':False})
print(json.dumps({'driver':records[1]['new'],'wrapper':records[2]['new'],'validator':records[0]['new'],
 'config':pin(cp),'raw_AST_inverse':pin(ledger),'handoff':pin(handoff)}))
