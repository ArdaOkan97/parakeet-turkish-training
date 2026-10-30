import ast,copy,difflib,hashlib,json
from pathlib import Path
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
F=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_two_logical_packet_lookahead_v1')
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,indent=2);f.write('\n')
def freeze(old,new,edits):
 raw=Path(old).read_bytes();text=raw.decode('utf-8-sig').replace('\r\n','\n')
 for a,b,n in edits:
  assert text.count(a)==n,(str(new),a[:100],text.count(a),n)
  text=text.replace(a,b)
 ast.parse(text);assert not Path(new).exists();Path(new).write_bytes(text.replace('\n','\r\n').encode())
 inverse=text
 for a,b,n in reversed(edits):assert inverse.count(b)==n,(b[:100],inverse.count(b),n);inverse=inverse.replace(b,a)
 assert inverse.replace('\n','\r\n').encode()==raw.lstrip(b'\xef\xbb\xbf')
 # BOM in Set-Content sources is declared and restored independently.
 restored=inverse.replace('\n','\r\n').encode()
 if raw.startswith(b'\xef\xbb\xbf'):restored=b'\xef\xbb\xbf'+restored
 assert restored==raw
 label=Path(new).stem
 (F/(label+'.diff')).write_text(''.join(difflib.unified_diff(raw.decode('utf-8-sig').replace('\r\n','\n').splitlines(True),text.splitlines(True),fromfile=str(old),tofile=str(new))),encoding='utf-8')
 return {'old_source':pin(old),'new_source':pin(new),'old_UTF8_BOM':raw.startswith(b'\xef\xbb\xbf'),'new_UTF8_BOM':False,
  'old_and_new_newlines':'CRLF','edits_on_LF_normalized_text':[{'old_literal':a,'new_literal':b,'expected_count':n}for a,b,n in edits],
  'full_raw_inverse_exact_including_declared_BOM':True}
validation='''
def validate_intentional_optimizer_journal_gap(failure,published_journal_frontier):
    """Keep the real optimizer counter distinct from successfully published events."""
    actual=failure['actual_optimizer_updates']
    require(type(actual)is int and type(published_journal_frontier)is int
        and actual-published_journal_frontier in (0,1)
        and 'Actual owner cooperative stop'in failure['error'],
        'Only actual cooperative stop with zero or one unpublished optimizer counter advance')
    gap=actual-published_journal_frontier;trace=failure['traceback'];site='no_unpublished_optimizer_counter'
    if gap:
        journal_guard=('in commit_RAM_transaction'in trace and 'in append'in trace
            and 'self.namespace.check(); self.binding()'in trace and 'self.guard()'in trace)
        native_poststep=('in native_update'in trace and 'norm=float(optimizer.step())'in trace
            and 'run_ASR18_native_throughput_component_v15.py'in trace)
        require(journal_guard or native_poststep,
            'One unpublished counter requires exact original pre-append or post-step cleanup cooperative guard trace')
        site='journal_preappend_original_guard'if journal_guard else'native_poststep_original_cleanup_guard'
    return {'actual_optimizer_frontier':actual,'published_journal_frontier':published_journal_frontier,
        'unpublished_optimizer_counter_advance':gap,'source_backed_cooperative_stop_site':site,
        'missing_journal_event_or_completed_native_receipt_fabricated':False,
        'full_checkpoint_restore_and_genuine_redo_required':True}

'''
helper1=S/'asr18_approved_lookahead_checkpoint_transition_v1.py';helper2=S/'asr18_approved_lookahead_checkpoint_transition_v2.py'
records=[]
records.append(freeze(helper1,helper2,[
 ('\ndef checked_lookahead_checkpoint_resume(', '\n'+validation+'\ndef checked_lookahead_checkpoint_resume(',1),
 ("    saved=value['last_fully_saved_optimizer_updates'];actual=value['actual_optimizer_updates']", "    saved=value['last_fully_saved_optimizer_updates'];actual=value['actual_optimizer_updates']\n    published=value['published_journal_optimizer_updates']\n    gap=validate_intentional_optimizer_journal_gap(failure,published)\n    require(gap==value['optimizer_vs_journal_frontier'],'Exact actual counter/journal distinction; no missing event fabricated')",1),
 ("and type(actual)is int and saved<=actual<saved+200", "and type(actual)is int and saved<=published<=actual<saved+200",1),
 ("require(len(events)==actual-saved and all", "require(len(events)==published-saved and all",1),
 ("zip(range(saved+1,actual+1),events)", "zip(range(saved+1,published+1),events)",1),
 ("'preserved_tail_positive_updates':list(range(saved+1,actual+1))", "'preserved_tail_positive_updates':list(range(saved+1,published+1))",1)]))
sealer1=F/'seal_actual_approved_source27_checkpoint_transition_v1.py';sealer2=F/'seal_actual_approved_source27_checkpoint_transition_v2.py'
records.append(freeze(sealer1,sealer2,[
 ('from resource_watchdog_v2 import sample_memory','from resource_watchdog_v2 import sample_memory\nfrom asr18_approved_lookahead_checkpoint_transition_v2 import validate_intentional_optimizer_journal_gap',1),
 ("    require(len(events)==actual-saved and all(x['optimizer_updates']", "    published=saved+len(events);gap=validate_intentional_optimizer_journal_gap(failure,published)\n    require(saved<=published<=actual and all(x['optimizer_updates']",1),
 ("zip(range(saved+1,actual+1),events)","zip(range(saved+1,published+1),events)",1),
 ("'last_fully_saved_optimizer_updates':saved,'actual_optimizer_updates':actual,", "'last_fully_saved_optimizer_updates':saved,'actual_optimizer_updates':actual,\n        'published_journal_optimizer_updates':published,'optimizer_vs_journal_frontier':gap,",1)]))
observer1=F/'observe_approved_natural_checkpoint_stop_v1.py';observer2=F/'observe_approved_natural_checkpoint_stop_v2.py'
records.append(freeze(observer1,observer2,[
 ("and saved<=actual<=saved+20:","and saved<=actual<=saved+20 and progress['last_saved_updates']==saved:",1),
 ("require(saved<=fresh['actual_optimizer_updates']<=saved+20 and fresh['source']==intent['old_driver_source'],", "require(saved<=fresh['actual_optimizer_updates']<=saved+20 and fresh['last_saved_updates']==saved\n                    and fresh['source']==intent['old_driver_source'],",1)]))
driver28=R/'run_ASR18_BPE9223_resident_RAM_training_v28.py';driver29=R/'run_ASR18_BPE9223_resident_RAM_training_v29.py'
records.append(freeze(driver28,driver29,[('from asr18_approved_lookahead_checkpoint_transition_v1 import','from asr18_approved_lookahead_checkpoint_transition_v2 import',1)]))
wrapper28=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v28.py';wrapper29=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v29.py'
wtext=wrapper28.read_text();count=wtext.count('run_ASR18_BPE9223_resident_RAM_training_v28')
records.append(freeze(wrapper28,wrapper29,[
 ('run_ASR18_BPE9223_resident_RAM_training_v28','run_ASR18_BPE9223_resident_RAM_training_v29',count),
 ('asr18_approved_lookahead_checkpoint_transition_v1','asr18_approved_lookahead_checkpoint_transition_v2',2)]))
intent=json.loads((F/'inactive_approved_source27_checkpoint_transition_intent_v1.json').read_text())
intent['next_driver_source']=pin(driver29);intent['next_wrapper_source']=pin(wrapper29)
intent['stop_point']='First next natural fullyverified200 outside4000DEV, BOTH original+fresh progress.last_saved_updates==saved, emitted progress<=saved+20'
intent['optimizer_and_journal_frontiers_may_differ_by_one']=True
intent['no_unpublished_optimizer_counter_is_promoted_to_a_journal_event']=True
write(F/'inactive_approved_source27_checkpoint_transition_intent_v2.json',intent)
config=json.loads((F/'proposed_configuration_with_transition_v1.json').read_text())
config['resident_RAM_packet_credit_policy']['actual_CPU_control_review']=pin(F/'ROOT_owned_two_logical_packet_one_receiver_actual_CPU_review_v1.json')
write(F/'candidate_configuration_completed_CPU_pending_actual_transition_v1.json',config)
write(F/'checkpoint_stop_edge_revision_exact_raw_inverse_v1.json',{'kind':'ASR18_INACTIVE_intentional_stop_race_representation_revision_v1','records':records,
 'production_original_native_commit_guards_unchanged':True,'actual_stop_or_control_executed':False,
 'old_transport_CPU_proof_reused_without_execution':True})
handoff={'kind':'ASR18_INACTIVE_checkpoint_stop_edge_revision_and_source29_handoff_v1',
 'transition_intent':pin(F/'inactive_approved_source27_checkpoint_transition_intent_v2.json'),
 'observer_source':pin(observer2),'closure_sealer_source':pin(sealer2),'transition_validator':pin(helper2),
 'driver_source':pin(driver29),'wrapper_source':pin(wrapper29),
 'source_inverse_ledger':pin(F/'checkpoint_stop_edge_revision_exact_raw_inverse_v1.json'),
 'candidate_configuration':pin(F/'candidate_configuration_completed_CPU_pending_actual_transition_v1.json'),
 'actual_CPU_review':pin(F/'ROOT_owned_two_logical_packet_one_receiver_actual_CPU_review_v1.json'),
 'human_approval':intent['human_approval'],'requested_ROOT_scope':'ASR18_approved_two_logical_packet_checkpoint_stop_v1',
 'stop_written':False,'actual_closure_slot':None,
 'narrow_edge':'Counter frontier and published journal frontier retained separately; exactly0or1 gap only under pinned original cooperative-stop trace; no fabricated event/native receipt; all after saved genuinely redo',
 'postwriter_edge':'Require both progress last_saved_updates==saved before exclusive stop write; still no atomic zero-tail promise',
 'unchanged':'helper2/CPU11/proof323c/numericalbody/commit/DEV/save/guards/225/fullreservations/source27liveenvironment',
 'execution_admitted':False}
write(F/'inactive_source29_checkpoint_stop_edge_handoff_v1.json',handoff)
print(json.dumps({'handoff':pin(F/'inactive_source29_checkpoint_stop_edge_handoff_v1.json'),
 'sources':[x['new_source']for x in records],'candidate':handoff['candidate_configuration'],'intent':handoff['transition_intent'],'actual_stop':False}))
