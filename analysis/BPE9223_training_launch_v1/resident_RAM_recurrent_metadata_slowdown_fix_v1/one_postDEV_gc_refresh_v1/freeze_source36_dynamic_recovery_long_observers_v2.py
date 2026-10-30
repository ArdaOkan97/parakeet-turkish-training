"""Freeze inactive observer sources/metadata only; no observer or candidate execution."""
import ast,difflib,hashlib,json,os
from pathlib import Path
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');R=P/'work/research';S=P/'outputs/parakeet_turkish'
F=Path(__file__).resolve().parent;OLD=F.parent
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
old=OLD/'observe_source31_dynamic_durable_validation_v1.py';raw=old.read_bytes();text=raw.decode();edits=[]
def replace(a,b,n):
 global text
 assert text.count(a)==n,(a,text.count(a),n);text=text.replace(a,b)
 edits.append({'old_literal':a,'new_literal':b,'expected_count':n})
replace('source31','source36',text.count('source31'))
replace("HANDOFF=F/'final_source36_signature_collect_transition_execution_handoff_v1.json'",
 "HANDOFF=F/'final_source36_one_postDEV_refresh_transition_execution_handoff_v1.json'",1)
replace('invocations/owner_000023','invocations/owner_000025',1)
replace("A=F.parent/'resident_RAM_training_actual_24'","A=F.parent.parent/'resident_RAM_training_actual_26'",1)
replace('native_000023.log','native_000025.log',1)
replace("and m['commit_available_bytes']>=2268435456+extra", "and m['commit_available_bytes']>=(4518435456+extra if admission else 2268435456)",1)
replace("def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))",
 "def read(p):\n p=Path(p);require(p.stat().st_size<=8*1024**2,'Bounded compact JSON only');return json.loads(p.read_text(encoding='utf-8-sig'))",1)
replace(" p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}",
 " p=Path(p).resolve();require(p.stat().st_size<=8*1024**2,'Bounded source/metadata pin');b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}",1)
anchor=" require(read(pe['DEV']['path'])['passed']is True,'Actual evaluation report passed completion, no quality-pass claim')\n"
addition=""" event_path=I/'actual_one_postDEV_gc_refresh.json';ineligible_path=I/'actual_one_postDEV_gc_refresh_ineligible.json'
 require(event_path.exists()!=ineligible_path.exists(),'Exactly one consumed first-periodic opportunity, no later retry/fallback')
 if event_path.exists():
  event=read(event_path)
  require(event['passed']is True and event['optimizer_updates']==dev
   and event['operations_completed']==['unfreeze','captured_original_full_collect','freeze']
   and event['before_after_capture_exact']is True and event['before_capture_sha256']==event['after_capture_sha256']
   and event['bookkeeping_reserve_bytes']==16777216 and event['host_credit_bytes']==0,
   'Actual sole idle-event receipt/source-backed semantic compare; no new fingerprint or GC call by observer')
  opportunity={'actual_refresh_performed':True,'receipt':pin(event_path),'actual_receipt':event,
   'original_capture_source':review['one_postDEV_gc_refresh_policy']['capture_source'],
   'scalar_closed':pin(I/'first_periodic_refresh_scalar_worker_closure.json'),
   'fresh_scalar_admission':pin(I/'private_metadata_worker_after_first_periodic_refresh_admission.json')}
 else:
  event=read(ineligible_path)
  require(event['passed']is True and event['optimizer_updates']==dev and event['first_opportunity_consumed']is True
   and event['later_periodic_retry_allowed']is False and event['actual_refresh_performed']is False
   and event['GC_operation_attempted']is False and event['scalar_closure_fabricated']is False,
   'Truthful consumed ineligible opportunity, never fabricate a refresh or closure')
  opportunity={'actual_refresh_performed':False,'receipt':pin(ineligible_path),'actual_receipt':event,
   'GC_refresh_hypothesis_was_not_exercised':True}
"""
replace(anchor,anchor+addition,1)
replace("'actual_scheduled_DEV':pe,'actual_normal_CPU_quiescence':", "'actual_scheduled_DEV':pe,'first_periodic_refresh_opportunity':opportunity,'actual_normal_CPU_quiescence':",1)
replace("'first_and_final_and_both_postDEV400_inclusive_near_reference':all(x['audio_minutes_per_wall_minute_pause_inclusive']>=75 for x in [first,last,*post]),",
 "'all_comparable_nonDEV400_inclusive_near_reference':all(x['audio_minutes_per_wall_minute_pause_inclusive']>=75 for x in [first,last,*post] if not x['contains_scheduled_DEV_pause']),\n   'first400_contains_DEV_pause_and_is_not_training_save_only_reference':first['contains_scheduled_DEV_pause'],",1)
ast.parse(text);inverse=text
for x in reversed(edits):
 assert inverse.count(x['new_literal'])==x['expected_count'],x['new_literal'];inverse=inverse.replace(x['new_literal'],x['old_literal'])
assert inverse.encode()==raw
new=F/'observe_source36_dynamic_durable_validation_v1.py';assert not new.exists();new.write_bytes(text.encode())
diff=new.with_suffix('.diff')
with diff.open('x',encoding='utf-8',newline='\n') as f:f.write(''.join(difflib.unified_diff(old.read_text().splitlines(True),text.replace('\r\n','\n').splitlines(True),fromfile=str(old),tofile=str(new))))
recovery=F/'seal_source36_dynamic_natural_recovery_v1.py';ast.parse(recovery.read_text(encoding='utf-8-sig'))
ledger=F/'source36_dynamic_recovery_long_metadata_observers_source_ledger_v1.json'
write(ledger,{'long_observer':{'old':pin(old),'new':pin(new),'replacements':edits,'newline_normalization':'NONE: exact raw literal edits',
 'entire_RAW_forward_and_inverse_exact':True,'diff':pin(diff)},
 'recovery_observer_is_new_dynamic_source_not_claimed_algorithm_equivalent':pin(recovery),
 'source_AST_syntax_checked_only_no_import_or_observer_execution':True})
handoff=F/'inactive_source36_dynamic_recovery_and_long_validation_observers_handoff_v1.json'
write(handoff,{'kind':'ASR18_INACTIVE_source36_dynamic_recovery_and_long_validation_sources_v1',
 'preparation_scope':pin(F/'ROOT_source36_typed_transition_checkpoint_stop_inactive_preparation_scope_v1.json'),
 'recovery_observer_source':pin(recovery),'long_validation_observer_source':pin(new),'source_ledger':pin(ledger),
 'candidate_source':pin(R/'run_ASR18_BPE9223_resident_RAM_training_v36.py'),
 'recovery':'Only after final exact ROOT launch: actual full restore-before-copy/config7/closedROOTboot reuse, only saved-offset..old actual+1 NEW25 events, natural saved+200/post+220, old24 STAT only, healthy5 and bounded unchanged reader captures; no archive/tensor/audio reads.',
 'recovery_inputs_chosen_only_after_actual_stop':'Actual last fully verified saved checkpoint and actual optimizer/published frontiers; no artificial fixed recovery frontier.',
 'future_final_execution_handoff_path':str(F/'final_source36_one_postDEV_refresh_transition_execution_handoff_v1.json'),
 'predeclared_validation':{'start':'actual_restored+20','end':'max(start+1600,nextscheduledDEV+1020)',
  'all_original20_rows_and200_endpoints_saves_realDEV_and_pauses':True,'whole_first400_final400_two_postDEV400':True,
  'near_original_inclusive_nonDEV400_floor':75.0,'original_checkpoint_inclusive_reference':76.569475,
  'maximum_sustained_save_excluded_first_to_final_decline_percent':5.0,'DEV_containing_first400_labelled_not_unfairly_compared_to_training_save_only':True,
  'one_actual_opportunity_receipt_and_scalar_closure_recreate_metadata':True,'no_new_archive_GC_profiler_or_fingerprint_operation':True,
  'ineligible_consumed_receipt_cannot_be_called_an_actual_GC_trial':True,'production_finite_stop':False,'warm_only_success_or_causal_guarantee':False},
 'recovery_guard':{'internal_seconds':180,'sampled_self_private_bytes':536870912,'full_prospective_commit_bytes':5055306368,
  'full_prospective_physical_bytes':2305306368,'runtime_commit_with_FULLprefetch':2268435456,'runtime_physical_with_FULLprefetch':1768435456,
  'E_free_bytes':100000000000,'JSON_source_cap_bytes':8388608,'new_event_segment_cap_bytes':4194304},
 'long_observer_guard':'Same180s/512MiB/full225+prospective512/FULLprefetch/originalfloors/E100GB; bounded8MiB current native+timing snapshots; all complete rows only.',
 'actual_observers_or_production_execution_admitted':False})
print(json.dumps({'recovery':pin(recovery),'long_observer':pin(new),'handoff':pin(handoff),'ledger':pin(ledger)}),flush=True)
