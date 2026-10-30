"""Immutable inactive candidate only. No import of candidate or production action."""
import ast,copy,difflib,hashlib,json,os
from pathlib import Path
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research';F=Path(__file__).parent
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
def start(p):
 raw=p.read_bytes();t=raw.decode().replace('\r\n','\n');assert t.replace('\n','\r\n').encode()==raw;return t,[]
def change(t,l,a,b,n=1):
 assert t.count(a)==n,(a[:150],t.count(a),n);l.append({'old_literal':a,'new_literal':b,'expected_count':n});return t.replace(a,b)
def freeze(old,new,t,l):
 ast.parse(t);raw=t.replace('\n','\r\n').encode();assert not new.exists();new.write_bytes(raw);inv=t
 for e in reversed(l):
  assert inv.count(e['new_literal'])==e['expected_count'],e['new_literal'][:120]
  inv=inv.replace(e['new_literal'],e['old_literal'])
 assert inv.replace('\n','\r\n').encode()==old.read_bytes()
 diff=F/(new.stem+'.diff')
 with diff.open('x',encoding='utf-8',newline='\n')as f:f.write(''.join(difflib.unified_diff(old.read_text().splitlines(True),t.splitlines(True),fromfile=str(old),tofile=str(new))))
 return {'old':pin(old),'new':pin(new),'replacements':l,'newline_policy':'LF replacement literals; restore original full CRLF bytes','full_old_raw_inverse_exact':True,'diff':pin(diff)}

old=S/'asr18_canonical_archive_checkpoint_transition_v1.py';new=S/'asr18_fresh_signature_collect_checkpoint_transition_v1.py';t,l=start(old)
def v(a,b,n=1):
 global t;t=change(t,l,a,b,n)
v('INACTIVE typed intentional source29 checkpoint transition for canonical archive bytes.','INACTIVE typed intentional source30 checkpoint transition for fresh signatures and original collect observation.')
v("SOURCE29_BASE=F.parent/'resident_RAM_two_logical_packet_lookahead_v1/production_configuration_v1/configuration.json'", "SOURCE30_BASE=F/'production_configuration_v1/configuration.json'")
v('def checked_canonical_archive_checkpoint_resume(', 'def checked_fresh_signature_collect_checkpoint_resume(')
v("base['prior_resident_RAM_canonical_archive_checkpoint_transition']", "base['prior_resident_RAM_fresh_signature_collect_checkpoint_transition']")
v('ASR18_actual_source29_checkpoint_canonical_archive_transition_v1','ASR18_actual_source30_checkpoint_fresh_signature_collect_transition_v1')
v("value['owned_session']==87302 and value['old_owner_invocation']==21 and value['new_owner_invocation']==22", "value['owned_session']==72166 and value['old_owner_invocation']==22 and value['new_owner_invocation']==23")
v("    checked_archive_representation_policy(base)\n    previous=read_json(SOURCE29_BASE);restored=copy.deepcopy(base)\n    del restored['prior_resident_RAM_canonical_archive_checkpoint_transition']\n    del restored['prepared_metadata_archive_representation_policy']\n    require(restored==previous,'Only archive representation and actual transition provenance differ from source29 base')", "    checked_fresh_signature_collect_policy(base)\n    previous=read_json(SOURCE30_BASE);restored=copy.deepcopy(base)\n    del restored['prior_resident_RAM_fresh_signature_collect_checkpoint_transition']\n    del restored['fresh_metadata_signature_and_collect_observation_policy']\n    require(restored==previous,'Only fresh signature/collect observation policy and actual transition provenance differ from proper source30 base')")
v('run_ASR18_BPE9223_resident_RAM_training_v29.py','run_ASR18_BPE9223_resident_RAM_training_v30.py')
v('run_ASR18_BPE9223_resident_RAM_autocontinue_v29.py','run_ASR18_BPE9223_resident_RAM_autocontinue_v30.py')
v('invocations/owner_000021','invocations/owner_000022',3)
v('Exact actual source29/CPU11 invocation21','Exact actual source30/CPU11 invocation22')
v('ASR18_canonical_archive_checkpoint_stop_v1','ASR18_fresh_signature_collect_checkpoint_stop_v1')
v('ASR18_reviewed_canonical_archive_checkpoint_cooperative_stop_v1','ASR18_reviewed_fresh_signature_collect_checkpoint_cooperative_stop_v1')
v("intent['owned_session']==87302 and intent['old_owner_invocation']==21", "intent['owned_session']==72166 and intent['old_owner_invocation']==22")
v("who[0]['pid']==37836 and abs(who[0]['create_time']-1792733651.0335317)<.001", "who[0]['pid']==40012 and abs(who[0]['create_time']-1792743161.6231525)<.001")
v('Same admitted live owner and honest old source29/CPU11 provenance','Same admitted live owner and honest old source30/CPU11 provenance')
v('saved>=162000','saved>=166000')
v("tool['owned_session']==87302", "tool['owned_session']==72166")
v('commits_000021.jsonl','commits_000022.jsonl')
v('Only compact preserved new tail and original21 STAT, never historical journal/prefix/whole','Only compact preserved new tail and original22 STAT, never historical journal/prefix/whole')
v('Closed source29 whole21 remains untouched','Closed source30 whole22 remains untouched')
v('Old21 remains authoritative until after exact restore and new prefix-only branch','Old22 remains authoritative until after exact restore and new prefix-only branch')
anchor='def checked_archive_representation_policy(base):'
extra='''def checked_fresh_signature_collect_policy(base):
    policy=base['fresh_metadata_signature_and_collect_observation_policy']
    require(policy=={
        'kind':'ASR18_fresh_metadata_signature_and_original_collect_observation_policy_v1',
        'signature_helper_source':record(F/'asr18_fresh_single_open_metadata_signature_v2.py'),
        'integration_helper_source':record(F/'asr18_fresh_metadata_signature_scope_v2.py'),
        'collect_observer_source':record(F/'asr18_observe_original_gc_collect_v2.py'),
        'original_context_source':record(S/'asr18_BPE9223_source_context_v2.py'),
        'original_geometry_source':record(S/'bpe_declared_geometry_index_v1.py'),
        'original_prefetch_source':record(S/'asr18_BPE9223_private_scalar_prefetch_v1.py'),
        'original_native_source':record(S/'asr18_BPE9223_grouped_RAM_native_v1.py'),
        'original_gc_scope_source':record(P/'work/research/run_ASR18_native_throughput_component_v15.py'),
        'fresh_original_check_count_path_size_integer_mtime_and_fallback':True,
        'all_three_original_readonly_locks_and_worker_join_before_alias_restore':True,
        'collect_original_builtin_once_same_args_result_exception':True,
        'collect_install_after_original_freeze_restore_before_original_teardown':True,
        'collect_bounded_per20_scalars_no_frames_or_payload_history':True,
        'all_original_byte_reservations_floors_225_policy_science_and_cadence_unchanged':True,
        'completed_CPU_review':policy['completed_CPU_review']},'Exact scoped signature and original collect observation policy')
    proof=policy['completed_CPU_review']
    require(type(proof)is dict and record(proof['path'])==proof,'Actual completed ROOT focused CPU proof required')
    review=read_json(proof['path'])
    require(review['accepted']is True and review['reviewer']=='/root'and review['scope']==policy['kind']
        and all(review[k]==policy[k]for k in ('signature_helper_source','integration_helper_source','collect_observer_source'))
        and record(review['actual_CPU_control_report']['path'])==review['actual_CPU_control_report']
        and review['semantic_passed']is True and review['candidate_timing_eligible_in_THIS_bounded_fixed_order']is True
        and review['all_exact_metadata_identities_dead']is True,'Actual semantic, bounded eligibility and exact-death gate')
    import sys
    if str(F)not in sys.path:sys.path.insert(0,str(F))
    from asr18_fresh_metadata_signature_scope_v2 import FreshMetadataSignatureScope
    from asr18_observe_original_gc_collect_v2 import ObserveOriginalCollectScope
    return FreshMetadataSignatureScope,ObserveOriginalCollectScope


'''
v(anchor,extra+anchor)
validator=freeze(old,new,t,l)

old=R/'run_ASR18_BPE9223_resident_RAM_training_v30.py';new=R/'run_ASR18_BPE9223_resident_RAM_training_v31.py';t,l=start(old)
def d(a,b,n=1):
 global t;t=change(t,l,a,b,n)
anchor='    from asr18_canonical_archive_checkpoint_transition_v1 import checked_archive_representation_policy,checked_canonical_archive_checkpoint_resume\n'
d(anchor,"    from asr18_fresh_signature_collect_checkpoint_transition_v1 import checked_fresh_signature_collect_policy,checked_fresh_signature_collect_checkpoint_resume\n    FreshMetadataSignatureScope,ObserveOriginalCollectScope=checked_fresh_signature_collect_policy(base)\n    signature_transition=None if a.fresh_signature_resume is None else checked_fresh_signature_collect_checkpoint_resume(\n        record(a.fresh_signature_resume),base=base,folder=folder,pointer_pin=pointer_pin)\n    if signature_transition is not None:\n        require(a.invocation==23 and all(x is None for x in (a.prior_process,a.canonical_archive_resume,a.lookahead_resume,a.cushion225_resume,\n            a.postrestore_selector_resume,a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume,\n            a.source_transition_resume,a.original_interval_resume,a.DEV_frontend_resume,a.DEV_admission_resume,a.periodic_quiescence_resume)),\n            'Only exact actual source30 fresh-signature checkpoint stop, never historical or normal0')\n"+anchor)
d('    transition=canonical_transition if canonical_transition is not None else(', '    transition=signature_transition if signature_transition is not None else(canonical_transition if canonical_transition is not None else(')
d('        record(a.source_transition_resume),base=base,folder=folder,pointer_pin=pointer_pin)))', '        record(a.source_transition_resume),base=base,folder=folder,pointer_pin=pointer_pin))))')
d('    preserved_tail=canonical_transition if canonical_transition is not None else(', '    preserved_tail=signature_transition if signature_transition is not None else(canonical_transition if canonical_transition is not None else(')
d("(transition if transition is not None and transition['unsaved_tail_bytes'] else None))))", "(transition if transition is not None and transition['unsaved_tail_bytes'] else None)))))")
d("'actual_transition':record(a.canonical_archive_resume if a.canonical_archive_resume is not None else", "'actual_transition':record(a.fresh_signature_resume if a.fresh_signature_resume is not None else a.canonical_archive_resume if a.canonical_archive_resume is not None else")
d('(a.canonical_archive_resume is not None or a.lookahead_resume is not None or a.cushion225_resume is not None or a.packet_credit_resume is not None)', '(a.fresh_signature_resume is not None or a.canonical_archive_resume is not None or a.lookahead_resume is not None or a.cushion225_resume is not None or a.packet_credit_resume is not None)',2)
d("'predecessor':record(a.canonical_archive_resume if a.canonical_archive_resume is not None else", "'predecessor':record(a.fresh_signature_resume if a.fresh_signature_resume is not None else a.canonical_archive_resume if a.canonical_archive_resume is not None else")
anchor='        from asr18_BPE9223_private_scalar_prefetch_v1 import PrivateScalarPrefetch\n'
d(anchor,anchor+"        signature_scope=scopes.enter_context(FreshMetadataSignatureScope(engine.context,private_worker_closed=lambda:\n            prefetch is None or(not prefetch.thread.is_alive()and prefetch.worker_status['closed'])))\n        original_collect_observer=scopes.enter_context(ObserveOriginalCollectScope())\n        write_json(invocation/'actual_fresh_signature_collect_observation_admission.json',\n            {'policy':base['fresh_metadata_signature_and_collect_observation_policy'],\n             'signature_scope':signature_scope.status(),'original_collect_installed_after_startup_freeze':True})\n")
d("            metadata_timing.update(kind='ASR18_actual_private_metadata_window_timing_v1',", "            metadata_timing['original_collect_observation']=original_collect_observer.take_window()\n            metadata_timing.update(kind='ASR18_actual_private_metadata_window_timing_v1',")
d("p.add_argument('--canonical-archive-resume');", "p.add_argument('--fresh-signature-resume');p.add_argument('--canonical-archive-resume');")
driver=freeze(old,new,t,l)

old=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v30.py';new=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v31.py';t,l=start(old)
def w(a,b,n=1):
 global t;t=change(t,l,a,b,n)
w('run_ASR18_BPE9223_resident_RAM_training_v30','run_ASR18_BPE9223_resident_RAM_training_v31',10)
anchor='    from asr18_canonical_archive_checkpoint_transition_v1 import checked_archive_representation_policy,checked_canonical_archive_checkpoint_resume\n'
w(anchor,"    from asr18_fresh_signature_collect_checkpoint_transition_v1 import checked_fresh_signature_collect_policy,checked_fresh_signature_collect_checkpoint_resume\n    checked_fresh_signature_collect_policy(configuration)\n    require(review['fresh_metadata_signature_and_collect_observation_policy']==configuration['fresh_metadata_signature_and_collect_observation_policy'],\n        'Exact reviewed fresh signatures and observational original collect policy')\n"+anchor)
w('require(a.canonical_archive_resume is None and a.lookahead_resume is None', 'require(a.fresh_signature_resume is None and a.canonical_archive_resume is None and a.lookahead_resume is None')
anchor='    elif a.canonical_archive_resume is not None:\n'
w(anchor,"    elif a.fresh_signature_resume is not None:\n        require(all(x is None for x in (a.canonical_archive_resume,a.lookahead_resume,a.cushion225_resume,a.postrestore_selector_resume,\n            a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume,a.original_interval_resume,a.source_transition_resume,\n            a.DEV_frontend_resume,a.DEV_admission_resume,a.periodic_quiescence_resume))\n            and review['fresh_signature_resume']==record(a.fresh_signature_resume)\n            and review['fresh_signature_transition_validator']==record(S/'asr18_fresh_signature_collect_checkpoint_transition_v1.py')\n            and review['human_approval']==record(Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/ROOT_direct_human_then_fix_authorization_v1.json'))\n            and a.start_invocation==review['initial_owner_invocation']==23,\n            'Only exact reviewed actual source30 signature/collect transition, never normal0')\n"+anchor)
w('    if a.canonical_archive_resume is not None:\n', "    if a.fresh_signature_resume is not None:\n        checked_fresh_signature_collect_checkpoint_resume(record(a.fresh_signature_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))\n    if a.canonical_archive_resume is not None:\n")
w("            if a.canonical_archive_resume is not None:command+=['--canonical-archive-resume',a.canonical_archive_resume]", "            if a.fresh_signature_resume is not None:command+=['--fresh-signature-resume',a.fresh_signature_resume]\n            elif a.canonical_archive_resume is not None:command+=['--canonical-archive-resume',a.canonical_archive_resume]")
w("p.add_argument('--canonical-archive-resume');", "p.add_argument('--fresh-signature-resume');p.add_argument('--canonical-archive-resume');")
wrapper=freeze(old,new,t,l)

base=F/'production_configuration_v1/configuration.json';candidate=json.loads(base.read_text())
candidate['fresh_metadata_signature_and_collect_observation_policy']={
 'kind':'ASR18_fresh_metadata_signature_and_original_collect_observation_policy_v1',
 'signature_helper_source':pin(F/'asr18_fresh_single_open_metadata_signature_v2.py'),
 'integration_helper_source':pin(F/'asr18_fresh_metadata_signature_scope_v2.py'),
 'collect_observer_source':pin(F/'asr18_observe_original_gc_collect_v2.py'),
 'original_context_source':pin(S/'asr18_BPE9223_source_context_v2.py'),
 'original_geometry_source':pin(S/'bpe_declared_geometry_index_v1.py'),
 'original_prefetch_source':pin(S/'asr18_BPE9223_private_scalar_prefetch_v1.py'),
 'original_native_source':pin(S/'asr18_BPE9223_grouped_RAM_native_v1.py'),
 'original_gc_scope_source':pin(R/'run_ASR18_native_throughput_component_v15.py'),
 'fresh_original_check_count_path_size_integer_mtime_and_fallback':True,
 'all_three_original_readonly_locks_and_worker_join_before_alias_restore':True,
 'collect_original_builtin_once_same_args_result_exception':True,
 'collect_install_after_original_freeze_restore_before_original_teardown':True,
 'collect_bounded_per20_scalars_no_frames_or_payload_history':True,
 'all_original_byte_reservations_floors_225_policy_science_and_cadence_unchanged':True,
 'completed_CPU_review':None}
candidate['prior_resident_RAM_fresh_signature_collect_checkpoint_transition']=None
cp=F/'proposed_source31_configuration_pending_ROOT_CPU_and_actual_stop_v1.json';write(cp,candidate)
inv=copy.deepcopy(candidate);del inv['fresh_metadata_signature_and_collect_observation_policy'];del inv['prior_resident_RAM_fresh_signature_collect_checkpoint_transition'];assert inv==json.loads(base.read_text())
oldtree=ast.parse((R/'run_ASR18_BPE9223_resident_RAM_training_v30.py').read_text());newtree=ast.parse((R/'run_ASR18_BPE9223_resident_RAM_training_v31.py').read_text())
funcs=lambda tree:{x.name:ast.dump(x,include_attributes=False)for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
o,n=funcs(oldtree),funcs(newtree);same=[k for k in o if k!='main'];assert all(o[k]==n[k]for k in same)
ledger=F/'source30_to31_fresh_signature_collect_exact_raw_inverse_v1.json'
write(ledger,{'kind':'ASR18_inactive_fresh_signature_collect_source31_raw_inverse_v1','validator':validator,'driver':driver,'wrapper':wrapper,
 'all_old_driver_nonentry_AST_exact':same,'proper_source30_base':pin(base),'proposed_configuration':pin(cp),
 'config_inverse_after_only_two_added_fields':True,'native1_GCscope_CPU11_helper2_helper3_guards_science_cadence_unchanged':True,'production_admitted':False})
hp=F/'inactive_source31_fresh_signature_collect_transition_handoff_v1.json'
write(hp,{'kind':'ASR18_INACTIVE_fresh_signature_collect_source31_transition_handoff_v1','driver_source':driver['new'],'wrapper_source':wrapper['new'],
 'transition_validator':validator['new'],'raw_inverse':pin(ledger),'proposed_configuration':pin(cp),'proper_source30_base':pin(base),
 'focused_CPU_completion':pin(F/'actual_fresh_signature_collect_CPU_completion_v1.json'),
 'human_authorization':pin(F/'ROOT_direct_human_then_fix_authorization_v1.json'),'completed_CPU_review':None,'actual_transition':None,
 'future_invocation':23,'future_output':str(F.parent/'resident_RAM_training_actual_24'),'future_journal':'commits_000023.jsonl',
 'seven_runtime_binding_kinds':['worker_driver_source','parent_driver_source','resident_RAM_stream_configuration','resident_branch_namespace','checkpoint_interval_updates','source_checkpoint','source_configuration'],
 'future_pointer':'Only actual latest full natural200 pointer after separately reviewed intentional stop',
 'validation_formula':{'start':'restored+20','end':'max(start+1600,nextscheduledDEV+1020)','all_original20rows_all_saves_realDEV_pauses':True,'no_finite_training_stop':True,
 'comparisons':'all inclusive interval; first400/final400; two full400 postDEV blocks; save-excluded decline<=5%, near original76.57 inclusive (~75+), no guard/science/state/reader regression'},
 'production_stop_or_launch_admitted':False,'source30_owned_session72166_unchanged':True,
 'next_requirements':['ROOT completed focused CPU gate','ROOT exact source/config/intent/observer/sealer review','actual intentional stop+clean joined Monitor+TOOL1+exact deaths+tail seal',
 'actual closure/config entire real pre/postrestore/wrapper regression metadata proof','final ROOT exact execution binding','ONE full-state recovery/genuine redo and durable long actual validation']})
print(json.dumps({'handoff':pin(hp),'raw_inverse':pin(ledger),'validator':validator['new'],'driver':driver['new'],'wrapper':wrapper['new'],'candidate':pin(cp)}))
