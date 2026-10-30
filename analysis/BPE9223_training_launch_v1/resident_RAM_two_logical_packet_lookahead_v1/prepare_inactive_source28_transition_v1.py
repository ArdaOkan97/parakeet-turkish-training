import ast,copy,difflib,hashlib,json
from pathlib import Path
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
F=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_two_logical_packet_lookahead_v1')
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,indent=2);f.write('\n')
reports=[]
for kind in ['training','autocontinue']:
 old=R/('run_ASR18_BPE9223_resident_RAM_'+kind+'_v27.py');new=R/('run_ASR18_BPE9223_resident_RAM_'+kind+'_v28.py')
 raw=old.read_bytes();text=raw.decode('utf-8-sig').replace('\r\n','\n');edits=[]
 def edit(a,b,n=1):
  global text
  assert text.count(a)==n,(kind,a[:100],text.count(a),n)
  text=text.replace(a,b);edits.append({'old_literal':a,'new_literal':b,'expected_count':n})
 if kind=='training':
  edit('from fastmss_resident_feature_stream_v10 import checked_resource_profile','from fastmss_resident_feature_stream_v11 import checked_resource_profile')
  edit('from fastmss_resident_feature_stream_v10 import ResidentFeatureProducer','from fastmss_resident_feature_stream_v11 import ResidentFeatureProducer')
  a='    from asr18_resident_source_transition_v2 import checked_source_transition_resume\n    transition=None if a.source_transition_resume is None else checked_source_transition_resume(\n        record(a.source_transition_resume),base=base,folder=folder,pointer_pin=pointer_pin)'
  b='''    from asr18_approved_lookahead_checkpoint_transition_v1 import checked_lookahead_checkpoint_resume
    lookahead_transition=None if a.lookahead_resume is None else checked_lookahead_checkpoint_resume(
        record(a.lookahead_resume),base=base,folder=folder,pointer_pin=pointer_pin)
    if lookahead_transition is not None:
        require(a.invocation==21 and all(x is None for x in (a.prior_process,a.cushion225_resume,
            a.postrestore_selector_resume,a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume,
            a.source_transition_resume,a.original_interval_resume,a.DEV_frontend_resume,a.DEV_admission_resume,
            a.periodic_quiescence_resume)), 'Only the exact approved actual source27 stop, never historical or normal0')
    from asr18_resident_source_transition_v2 import checked_source_transition_resume
    transition=lookahead_transition if lookahead_transition is not None else(None if a.source_transition_resume is None else checked_source_transition_resume(
        record(a.source_transition_resume),base=base,folder=folder,pointer_pin=pointer_pin))'''
  edit(a,b)
  edit('    preserved_tail=interval_failure if interval_failure is not None else(abnormal if abnormal is not None else(transition if transition is not None and transition[\'unsaved_tail_bytes\'] else None))',
       '    preserved_tail=lookahead_transition if lookahead_transition is not None else(interval_failure if interval_failure is not None else(abnormal if abnormal is not None else(transition if transition is not None and transition[\'unsaved_tail_bytes\'] else None)))')
  edit("'actual_transition':record(a.source_transition_resume)","'actual_transition':record(a.lookahead_resume if a.lookahead_resume is not None else a.source_transition_resume)")
  edit('(a.cushion225_resume is not None or a.packet_credit_resume is not None)', '(a.lookahead_resume is not None or a.cushion225_resume is not None or a.packet_credit_resume is not None)',2)
  edit("'predecessor':record(a.postrestore_selector_resume", "'predecessor':record(a.lookahead_resume if a.lookahead_resume is not None else a.postrestore_selector_resume")
  edit("p.add_argument('--postrestore-selector-resume');", "p.add_argument('--lookahead-resume');p.add_argument('--postrestore-selector-resume');")
 else:
  count=text.count('run_ASR18_BPE9223_resident_RAM_training_v27');edit('run_ASR18_BPE9223_resident_RAM_training_v27','run_ASR18_BPE9223_resident_RAM_training_v28',count)
  edit("review['CPU_source']==record(S/'fastmss_resident_feature_stream_v10.py')","review['CPU_source']==record(S/'fastmss_resident_feature_stream_v11.py')")
  edit('from fastmss_resident_feature_stream_v10 import checked_packet_credit_policy','from fastmss_resident_feature_stream_v11 import checked_packet_credit_policy')
  edit("        require(a.cushion225_resume is None and", "        require(a.lookahead_resume is None and a.cushion225_resume is None and")
  edit('    elif a.cushion225_resume is not None:', '''    elif a.lookahead_resume is not None:
        require(all(x is None for x in (a.cushion225_resume,a.postrestore_selector_resume,a.packet_credit_resume,
            a.native_headroom_resume,a.abnormal_resume,a.original_interval_resume,a.source_transition_resume,
            a.DEV_frontend_resume,a.DEV_admission_resume,a.periodic_quiescence_resume))
            and review['lookahead_resume']==record(a.lookahead_resume)
            and review['lookahead_transition_validator']==record(S/'asr18_approved_lookahead_checkpoint_transition_v1.py')
            and review['human_approval']==configuration['resident_RAM_packet_credit_policy']['human_approval']
            and review['packet_credit_policy']==configuration['resident_RAM_packet_credit_policy']
            and review['native_update_commit_pressure_policy']==configuration['native_update_commit_pressure_policy']
            and a.start_invocation==review['initial_owner_invocation']==21,
            'Only exact reviewed approved intentional source27 checkpoint transition, never normal0')
    elif a.cushion225_resume is not None:''')
  edit('    if a.cushion225_resume is not None:\n        from', '''    if a.lookahead_resume is not None:
        from asr18_approved_lookahead_checkpoint_transition_v1 import checked_lookahead_checkpoint_resume
        checked_lookahead_checkpoint_resume(record(a.lookahead_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))
    if a.cushion225_resume is not None:
        from''')
  edit("            if a.cushion225_resume is not None:command+=['--cushion225-resume',a.cushion225_resume]", "            if a.lookahead_resume is not None:command+=['--lookahead-resume',a.lookahead_resume]\n            elif a.cushion225_resume is not None:command+=['--cushion225-resume',a.cushion225_resume]")
  edit("p.add_argument('--postrestore-selector-resume');", "p.add_argument('--lookahead-resume');p.add_argument('--postrestore-selector-resume');")
 ast.parse(text);assert not new.exists();new.write_bytes(text.replace('\n','\r\n').encode())
 inverse=text
 for e in reversed(edits):
  assert inverse.count(e['new_literal'])==e['expected_count'],e['new_literal'][:100]
  inverse=inverse.replace(e['new_literal'],e['old_literal'])
 assert inverse.replace('\n','\r\n').encode()==raw
 old_ast=ast.parse(raw);new_ast=ast.parse(text)
 same_top=[]
 oldfunctions={n.name:n for n in old_ast.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 newfunctions={n.name:n for n in new_ast.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 for name,node in oldfunctions.items():
  if name!='main':assert ast.dump(node,include_attributes=False)==ast.dump(newfunctions[name],include_attributes=False);same_top.append(name)
 oldmain=oldfunctions['main'];newmain=newfunctions['main']
 oldwhile=[n for n in ast.walk(oldmain)if isinstance(n,ast.While)];newwhile=[n for n in ast.walk(newmain)if isinstance(n,ast.While)]
 if kind=='training':assert len(oldwhile)==len(newwhile)==3 and all(ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)for a,b in zip(oldwhile,newwhile))
 item={'old_source':pin(old),'new_source':pin(new),'edits_on_LF_normalized_text':edits,
  'old_and_new_newlines':'CRLF','full_raw_old_bytes_reconstructed_exact':True,'all_old_nonmain_top_AST_exact':same_top,
  'native_three_while_AST_exact':True if kind=='training'else'wrapper forwarding intentionally adds only new typed flag'}
 reports.append(item)
 (F/(kind+'27_to28.diff')).write_text(''.join(difflib.unified_diff(raw.decode().replace('\r\n','\n').splitlines(True),text.splitlines(True),fromfile=str(old),tofile=str(new))),encoding='utf-8')
helper=S/'asr18_approved_lookahead_checkpoint_transition_v1.py';ast.parse(helper.read_text(encoding='utf-8-sig'))
config=json.loads((F/'proposed_configuration_v1.json').read_text());config['prior_resident_RAM_approved_lookahead_checkpoint_transition']=None
cfg=F/'proposed_configuration_with_transition_v1.json';write(cfg,config)
write(F/'source27_to28_exact_raw_inverse_ledger_v1.json',{'kind':'ASR18_INACTIVE_source28_exact_inverse_v1','reports':reports,
 'transition_validator':pin(helper),'proposed_configuration':pin(cfg),'production_or_stop_executed':False,
 'historical_validators_not_rewritten_or_source_aliased':True,
 'numerical_commit_DEV_checkpoint_native_loops_exact':True,'future_seven_runtime_binding_kinds_retained':True})
handoff={'kind':'ASR18_INACTIVE_source28_approved_lookahead_checkpoint_transition_source_handoff_v1',
 'driver_source':pin(R/'run_ASR18_BPE9223_resident_RAM_training_v28.py'),
 'wrapper_source':pin(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v28.py'),
 'transition_validator':pin(helper),'exact_raw_inverse_ledger':pin(F/'source27_to28_exact_raw_inverse_ledger_v1.json'),
 'proposed_configuration':pin(cfg),'CPU_source':pin(S/'fastmss_resident_feature_stream_v11.py'),
 'helper_source':pin(F/'asr18_owned_packet_credit_v2.py'),
 'human_approval':config['resident_RAM_packet_credit_policy']['human_approval'],
 'completed_CPU_review_slot':None,'actual_future_transition_closure_slot':None,
 'source27_running_unchanged':True,'stop_or_production_execution_admitted':False,
 'draft_future_names':{'initial_invocation':21,'output':'resident_RAM_training_actual_22','new_journal':'commits_000021.jsonl'},
 'remaining_stages':['ROOT finite actual CPU acceptance','ROOT successor/observer/intent pre-stop acceptance',
 'intentional supported stop only early after natural verified200 publication, actual TOOL1/deaths/tail retained',
 'serialize final config with genuine completed CPU review and actual closure pin',
 'ONE separately reviewed metadata/entire pre+postrestore dictionary/wrapper forwarding control',
 'ROOT final exact execution binding, restore before native, prefix-only branch after restore, genuinely redo any tail',
 'actual continued checkpoint-inclusive streaming throughput against original target80; no guarantee or finite stop']}
write(F/'inactive_source28_checkpoint_transition_handoff_v1.json',handoff)
print(json.dumps({'handoff':pin(F/'inactive_source28_checkpoint_transition_handoff_v1.json'),'sources':[x['new_source']for x in reports],
 'validator':pin(helper),'config':pin(cfg),'AST_and_raw_inverse_only':True}))
