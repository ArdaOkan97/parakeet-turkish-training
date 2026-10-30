"""Preparation only: immutable source30, exact raw inverses, no production action."""
import ast,copy,difflib,hashlib,json
from pathlib import Path
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
F=Path(__file__).parent
LOOK=F.parent/'resident_RAM_two_logical_packet_lookahead_v1'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
 with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
def begin(p):
 raw=p.read_bytes();text=raw.decode().replace('\r\n','\n');assert raw==text.replace('\n','\r\n').encode();return text,[]
def change(text,ledger,old,new,count=1):
 assert text.count(old)==count,(old[:120],text.count(old),count)
 ledger.append(dict(old_literal=old,new_literal=new,expected_count=count));return text.replace(old,new)
def freeze(old,new,text,ledger):
 ast.parse(text);data=text.replace('\n','\r\n').encode();assert not new.exists();new.write_bytes(data)
 inverse=text
 for e in reversed(ledger):
  assert inverse.count(e['new_literal'])==e['expected_count'],e['new_literal'][:100]
  inverse=inverse.replace(e['new_literal'],e['old_literal'])
 assert inverse.replace('\n','\r\n').encode()==old.read_bytes()
 diff=F/(new.stem+'.diff')
 with diff.open('x',encoding='utf-8')as f:f.write(''.join(difflib.unified_diff(old.read_text().splitlines(True),text.splitlines(True),fromfile=str(old),tofile=str(new))))
 return dict(old=pin(old),new=pin(new),newline_policy='Both sources original CRLF; replacement literals LF normalized then CRLF byte restoration',replacements=ledger,full_old_raw_bytes_reconstructed=True,diff=pin(diff))

# New typed lane derives from the already-reviewed actual cooperative-stop lane.
old=S/'asr18_approved_lookahead_checkpoint_transition_v3.py';new=S/'asr18_canonical_archive_checkpoint_transition_v1.py';t,l=begin(old)
def v(a,b,n=1):
 global t;t=change(t,l,a,b,n)
v('INACTIVE typed intentional source27 checkpoint transition for approved lookahead.','INACTIVE typed intentional source29 checkpoint transition for canonical archive bytes.')
v("F=ROOT/'analysis/BPE9223_training_launch_v1/resident_RAM_two_logical_packet_lookahead_v1'\nAPPROVAL=F.parent/'resident_RAM_source27_speed_review_v1/ROOT_direct_human_lookahead_and_checkpoint_restart_approval_v1.json'\nSOURCE27_BASE=F.parent/'resident_RAM_source26_postrestore_selector_failure_actual_v1/inactive_selector_fix_v2/production_configuration_v1/configuration.json'", "F=ROOT/'analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1'\nAPPROVAL=F/'ROOT_direct_human_then_fix_authorization_v1.json'\nSOURCE29_BASE=F.parent/'resident_RAM_two_logical_packet_lookahead_v1/production_configuration_v1/configuration.json'\nCODEC=F/'asr18_canonical_prepared_archive_v1.py'\nARCHIVE_SCOPE='ASR18_canonical_bytes_prepared_archive_policy_v1'")
v('def checked_lookahead_checkpoint_resume(', 'def checked_canonical_archive_checkpoint_resume(')
v("base['prior_resident_RAM_approved_lookahead_checkpoint_transition']", "base['prior_resident_RAM_canonical_archive_checkpoint_transition']")
v("ASR18_actual_approved_source27_checkpoint_lookahead_transition_v1", "ASR18_actual_source29_checkpoint_canonical_archive_transition_v1")
v("value['owned_session']==56620 and value['old_owner_invocation']==20 and value['new_owner_invocation']==21", "value['owned_session']==87302 and value['old_owner_invocation']==21 and value['new_owner_invocation']==22")
v("approval['sha256']=='f92036ea0b494897d85cf15d69c8eba07fb1abfd97febcf0a18099a0d8caf9fb'", "approval['sha256']=='e02734964bd14b3e1bc6fc25d9146995233d738b80a96d6e2d0719c74d071ca6'")
v("read_json(APPROVAL)['answer']=='Approve lookahead and checkpoint restart'", "read_json(APPROVAL)['human_message']=='then fix?'")
v('Exact direct approval of logical lookahead exception and checkpoint transition', 'Exact direct authorization to fix recurrent slowdown through a reviewed checkpoint transition')
start="    policy=base['resident_RAM_packet_credit_policy']\n"
end="    e=value['evidence']\n"
a=t.index(start);b=t.index(end,a);oldblock=t[a:b]
newblock="""    checked_archive_representation_policy(base)
    previous=read_json(SOURCE29_BASE);restored=copy.deepcopy(base)
    del restored['prior_resident_RAM_canonical_archive_checkpoint_transition']
    del restored['prepared_metadata_archive_representation_policy']
    require(restored==previous,'Only archive representation and actual transition provenance differ from source29 base')
"""
v(oldblock,newblock)
v('run_ASR18_BPE9223_resident_RAM_training_v27.py','run_ASR18_BPE9223_resident_RAM_training_v29.py')
v('run_ASR18_BPE9223_resident_RAM_autocontinue_v27.py','run_ASR18_BPE9223_resident_RAM_autocontinue_v29.py')
v('fastmss_resident_feature_stream_v10.py','fastmss_resident_feature_stream_v11.py')
v('invocations/owner_000020','invocations/owner_000021',3)
v('Exact actual source27/CPU10 invocation20','Exact actual source29/CPU11 invocation21')
v('ASR18_approved_two_logical_packet_checkpoint_stop_v1','ASR18_canonical_archive_checkpoint_stop_v1')
v('ASR18_reviewed_approved_lookahead_checkpoint_cooperative_stop_v1','ASR18_reviewed_canonical_archive_checkpoint_cooperative_stop_v1')
v("intent['owned_session']==56620 and intent['old_owner_invocation']==20", "intent['owned_session']==87302 and intent['old_owner_invocation']==21")
v("who[0]['pid']==9704 and abs(who[0]['create_time']-1792727133.7017913)<.001", "who[0]['pid']==37836 and abs(who[0]['create_time']-1792733651.0335317)<.001")
v('Same admitted live owner and honest old source27/CPU10 provenance','Same admitted live owner and honest old source29/CPU11 provenance')
v('saved>=157000','saved>=162000')
v("tool['owned_session']==56620", "tool['owned_session']==87302")
v('commits_000020.jsonl','commits_000021.jsonl')
v('Only compact preserved new tail and original20 STAT, never historical18/19 or old whole journal','Only compact preserved new tail and original21 STAT, never historical journal/prefix/whole')
v('Closed source27 whole20 remains untouched','Closed source29 whole21 remains untouched')
v('Old20 remains authoritative until after exact restore and new prefix-only branch','Old21 remains authoritative until after exact restore and new prefix-only branch')
extra="""

def checked_archive_representation_policy(base):
    policy=base['prepared_metadata_archive_representation_policy']
    require(policy=={
        'kind':ARCHIVE_SCOPE,'helper_source':record(CODEC),
        'record_encoding':'original_canonical_UTF8_bytes_after_positive_commit',
        'checkpoint_encoding':'stream_original_canonical_JSON_array_into_original_exclusive_gzip',
        'historical_record_mutation_forbidden':True,'checkpoint_interval_updates':200,
        'native_GC_guard_and_science_unchanged':True,
        'completed_CPU_review':policy['completed_CPU_review']},'Exact immutable original-canonical archive representation policy')
    proof=policy['completed_CPU_review'];require(type(proof)is dict and record(proof['path'])==proof,
        'Actual completed ROOT codec proof required before admitted production')
    review=read_json(proof['path'])
    require(review['accepted']is True and review['reviewer']=='/root'and review['scope']==ARCHIVE_SCOPE
        and review['helper_source']==policy['helper_source']
        and record(review['actual_CPU_control_report']['path'])==review['actual_CPU_control_report']
        and review['all_exact_metadata_identities_dead']is True,'Typed completed actual codec CPU review')
    import importlib.util
    spec=importlib.util.spec_from_file_location('asr18_canonical_prepared_archive_v1',CODEC)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
"""
v(t[-1:],t[-1:]+extra) if False else None
# The appended function is explicitly invertible without editing historical functions.
l.append(dict(old_literal='',new_literal=extra,expected_count=1,operation='append_only'))
t+=extra
# freeze needs special inverse for append_only, performed separately below.
def freeze_append(old,new,text,ledger):
 appended=ledger[-1];assert appended.get('operation')=='append_only'and text.endswith(appended['new_literal'])
 ast.parse(text);assert not new.exists();new.write_bytes(text.replace('\n','\r\n').encode())
 inverse=text[:-len(appended['new_literal'])]
 for e in reversed(ledger[:-1]):
  assert inverse.count(e['new_literal'])==e['expected_count'],e['new_literal'][:100]
  inverse=inverse.replace(e['new_literal'],e['old_literal'])
 assert inverse.replace('\n','\r\n').encode()==old.read_bytes()
 diff=F/(new.stem+'.diff')
 with diff.open('x',encoding='utf-8')as f:f.write(''.join(difflib.unified_diff(old.read_text().splitlines(True),text.splitlines(True),fromfile=str(old),tofile=str(new))))
 return dict(old=pin(old),new=pin(new),newline_policy='CRLF restored exactly',replacements=ledger,full_old_raw_bytes_reconstructed=True,diff=pin(diff))
validator=freeze_append(old,new,t,l)

old=R/'run_ASR18_BPE9223_resident_RAM_training_v29.py';new=R/'run_ASR18_BPE9223_resident_RAM_training_v30.py';t,l=begin(old)
def d(a,b,n=1):
 global t;t=change(t,l,a,b,n)
d('    native_pressure_policy,wait_for_native_commit_pressure=checked_native_commit_pressure_policy(base)\n', '    native_pressure_policy,wait_for_native_commit_pressure=checked_native_commit_pressure_policy(base)\n    from asr18_canonical_archive_checkpoint_transition_v1 import checked_archive_representation_policy,checked_canonical_archive_checkpoint_resume\n    archive_codec=checked_archive_representation_policy(base)\n    canonical_transition=None if a.canonical_archive_resume is None else checked_canonical_archive_checkpoint_resume(\n        record(a.canonical_archive_resume),base=base,folder=folder,pointer_pin=pointer_pin)\n    if canonical_transition is not None:\n        require(a.invocation==22 and all(x is None for x in (a.prior_process,a.lookahead_resume,a.cushion225_resume,\n            a.postrestore_selector_resume,a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume,\n            a.source_transition_resume,a.original_interval_resume,a.DEV_frontend_resume,a.DEV_admission_resume,\n            a.periodic_quiescence_resume)),\n            \'Only exact reviewed actual source29 archive transition, never historical or normal0\')\n')
d('    transition=lookahead_transition if lookahead_transition is not None else(None if a.source_transition_resume is None else checked_source_transition_resume(\n        record(a.source_transition_resume),base=base,folder=folder,pointer_pin=pointer_pin))', '    transition=canonical_transition if canonical_transition is not None else(lookahead_transition if lookahead_transition is not None else(None if a.source_transition_resume is None else checked_source_transition_resume(\n        record(a.source_transition_resume),base=base,folder=folder,pointer_pin=pointer_pin)))')
d('    preserved_tail=lookahead_transition if lookahead_transition is not None else(interval_failure if interval_failure is not None else(abnormal if abnormal is not None else(transition if transition is not None and transition[\'unsaved_tail_bytes\'] else None)))', '    preserved_tail=canonical_transition if canonical_transition is not None else(lookahead_transition if lookahead_transition is not None else(interval_failure if interval_failure is not None else(abnormal if abnormal is not None else(transition if transition is not None and transition[\'unsaved_tail_bytes\'] else None))))')
d("'actual_transition':record(a.lookahead_resume if a.lookahead_resume is not None else a.source_transition_resume)","'actual_transition':record(a.canonical_archive_resume if a.canonical_archive_resume is not None else a.lookahead_resume if a.lookahead_resume is not None else a.source_transition_resume)")
d('(a.lookahead_resume is not None or a.cushion225_resume is not None or a.packet_credit_resume is not None)', '(a.canonical_archive_resume is not None or a.lookahead_resume is not None or a.cushion225_resume is not None or a.packet_credit_resume is not None)',2)
d("'predecessor':record(a.lookahead_resume if a.lookahead_resume is not None else", "'predecessor':record(a.canonical_archive_resume if a.canonical_archive_resume is not None else a.lookahead_resume if a.lookahead_resume is not None else")
d("prepared_archive.append({'binding':binding,'metadata':metadata,'native':native})", "prepared_archive.append(archive_codec.encode_record({'binding':binding,'metadata':metadata,'native':native}))")
d("with gzip.open(ck/'prepared_metadata.json.gz','xb')as f:f.write(canonical(prepared_archive))", "archive_codec.write_gzip_archive(ck/'prepared_metadata.json.gz',prepared_archive)")
d("p.add_argument('--lookahead-resume');", "p.add_argument('--canonical-archive-resume');p.add_argument('--lookahead-resume');")
driver=freeze(old,new,t,l)

old=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v29.py';new=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v30.py';t,l=begin(old)
def w(a,b,n=1):
 global t;t=change(t,l,a,b,n)
w('run_ASR18_BPE9223_resident_RAM_training_v29','run_ASR18_BPE9223_resident_RAM_training_v30',8)
w("    require(a.postrestore_selector_resume is None or", "    from asr18_canonical_archive_checkpoint_transition_v1 import checked_archive_representation_policy,checked_canonical_archive_checkpoint_resume\n    checked_archive_representation_policy(configuration)\n    require(review['prepared_metadata_archive_representation_policy']==configuration['prepared_metadata_archive_representation_policy'],\n        'Exact reviewed canonical archive representation; all old protections retained')\n    require(a.postrestore_selector_resume is None or")
w("require(a.lookahead_resume is None and a.cushion225_resume is None", "require(a.canonical_archive_resume is None and a.lookahead_resume is None and a.cushion225_resume is None")
w("    elif a.lookahead_resume is not None:\n", "    elif a.canonical_archive_resume is not None:\n        require(all(x is None for x in (a.lookahead_resume,a.cushion225_resume,a.postrestore_selector_resume,a.packet_credit_resume,\n            a.native_headroom_resume,a.abnormal_resume,a.original_interval_resume,a.source_transition_resume,\n            a.DEV_frontend_resume,a.DEV_admission_resume,a.periodic_quiescence_resume))\n            and review['canonical_archive_resume']==record(a.canonical_archive_resume)\n            and review['canonical_archive_transition_validator']==record(S/'asr18_canonical_archive_checkpoint_transition_v1.py')\n            and review['human_approval']==record(Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/ROOT_direct_human_then_fix_authorization_v1.json'))\n            and a.start_invocation==review['initial_owner_invocation']==22,\n            'Only exact reviewed actual source29 archive transition; never normal0')\n    elif a.lookahead_resume is not None:\n")
w("    if a.lookahead_resume is not None:\n", "    if a.canonical_archive_resume is not None:\n        checked_canonical_archive_checkpoint_resume(record(a.canonical_archive_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))\n    if a.lookahead_resume is not None:\n")
w("            if a.lookahead_resume is not None:command+=['--lookahead-resume',a.lookahead_resume]", "            if a.canonical_archive_resume is not None:command+=['--canonical-archive-resume',a.canonical_archive_resume]\n            elif a.lookahead_resume is not None:command+=['--lookahead-resume',a.lookahead_resume]")
w("p.add_argument('--lookahead-resume');", "p.add_argument('--canonical-archive-resume');p.add_argument('--lookahead-resume');")
wrapper=freeze(old,new,t,l)

base=LOOK/'production_configuration_v1/configuration.json';configuration=json.loads(base.read_text())
configuration['prepared_metadata_archive_representation_policy']={
 'kind':'ASR18_canonical_bytes_prepared_archive_policy_v1','helper_source':pin(F/'asr18_canonical_prepared_archive_v1.py'),
 'record_encoding':'original_canonical_UTF8_bytes_after_positive_commit',
 'checkpoint_encoding':'stream_original_canonical_JSON_array_into_original_exclusive_gzip',
 'historical_record_mutation_forbidden':True,'checkpoint_interval_updates':200,
 'native_GC_guard_and_science_unchanged':True,'completed_CPU_review':None}
configuration['prior_resident_RAM_canonical_archive_checkpoint_transition']=None
candidate=F/'proposed_source30_configuration_pending_proofs_and_actual_stop_v1.json';write(candidate,configuration)
inverse=copy.deepcopy(configuration);del inverse['prepared_metadata_archive_representation_policy'];del inverse['prior_resident_RAM_canonical_archive_checkpoint_transition'];assert inverse==json.loads(base.read_text())
oldtree=ast.parse((R/'run_ASR18_BPE9223_resident_RAM_training_v29.py').read_text());newtree=ast.parse((R/'run_ASR18_BPE9223_resident_RAM_training_v30.py').read_text())
body=lambda tree:{x.name:ast.dump(x,include_attributes=False) for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
ob,nb=body(oldtree),body(newtree);unchanged=[k for k in ob if k!='main'];assert all(ob[k]==nb[k]for k in unchanged)
ledger=F/'source29_to30_canonical_archive_exact_raw_inverse_v1.json'
write(ledger,{'kind':'ASR18_inactive_canonical_archive_source30_exact_raw_inverse_v1','validator':validator,'driver':driver,'wrapper':wrapper,'all_old_driver_nonentry_AST_exact':unchanged,'proper_source_base':pin(base),'candidate_configuration':pin(candidate),'configuration_inverse_after_only_two_added_fields':True,'native_GC_CPU_DEV_prefetch_and_original_guards_unchanged':True,'production_admitted':False})
handoff={'kind':'ASR18_inactive_canonical_archive_source30_handoff_v1','driver_source':driver['new'],'wrapper_source':wrapper['new'],'transition_validator':validator['new'],'archive_helper':pin(F/'asr18_canonical_prepared_archive_v1.py'),'raw_inverse':pin(ledger),'proposed_configuration':pin(candidate),'proper_source_base':pin(base),'human_authorization':pin(F/'ROOT_direct_human_then_fix_authorization_v1.json'),'completed_CPU_review':None,'actual_transition':None,'production_admitted':False,'future_invocation':22,'future_output':str(F.parent/'resident_RAM_training_actual_23'),'future_journal':'commits_000022.jsonl','seven_runtime_binding_kinds':['worker_driver_source','parent_driver_source','resident_RAM_stream_configuration','resident_branch_namespace','checkpoint_interval_updates','source_checkpoint','source_configuration'],'future_pointer':'Only actual latest full natural200 pointer after reviewed stop','validation_formula':{'start':'restored+20','end':'max(start+1600,next_scheduled4000boundary+220)','all_natural_checkpoints_and_DEV_included':True,'no_finite_training_stop':True},'next_requirements':['ROOT completed focused codec CPU acceptance','frozen exact current-owner stop intent/observer/sealer with gap0or1','ROOT pre-stop review','actual intentional stop/death/TOOL/tail seal','whole real pre/postrestore dictionary and wrapper typed metadata proof','ROOT final execution binding','ONE launch/fullrestore-before-native/savedprefix-afterrestore/genuine redo','long actual production validation across DEV']}
hp=F/'inactive_source30_canonical_archive_transition_handoff_v1.json';write(hp,handoff)
print(json.dumps({'handoff':pin(hp),'inverse':pin(ledger),'driver':driver['new'],'wrapper':wrapper['new'],'validator':validator['new'],'candidate':pin(candidate)}))
