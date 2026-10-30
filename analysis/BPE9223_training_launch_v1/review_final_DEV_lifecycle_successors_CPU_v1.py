"""Exact six-source lifecycle import correction; real new-mean plan metadata only."""
import ast,difflib,importlib,json,sys
from pathlib import Path
L=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1');sys.path.insert(0,str(L));m=importlib.import_module('audit_and_fit_append_only_v5')
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1');O=N/'independent_DEV_lifecycle_successors_review_v1';O.mkdir(exist_ok=True);m.AREA=O
S=m.PROJECT/'outputs/parakeet_turkish';R=m.PROJECT/'work/research';sys.path.insert(0,str(S))
def rec(path):return {k:v for k,v in m.record(path).items() if k in ('path','bytes','sha256')}
def main():
 g=m.Guard();g.phase='exact_six_source_DEV_lifecycle_mean_binding_correction'
 pairs=[(S/'asr18_full_run_serial_DEV_9223_v2.py',S/'asr18_full_run_serial_DEV_9223_v3.py','dd05653efe75c0514ebf850e58c8389adbaa3526cb8365604ce783b57d2e9789'),
  (S/'asr18_full_run_DEV_runtime_9223_v2.py',S/'asr18_full_run_DEV_runtime_9223_v3.py','c73b3d3b11f8f8b2ecd796144b86b4731ce5da402832805faa881fae13e7b473'),
  (S/'asr18_BPE9223_full_run_controller_v1.py',S/'asr18_BPE9223_full_run_controller_v2.py','93622a9f4b9ffeca0f0b91a8a41b0c29c035752e5423a130491d0dcfc9992864'),
  (R/'run_ASR18_BPE9223_full_run_owner_v1.py',R/'run_ASR18_BPE9223_full_run_owner_v2.py','78d7d97a4719a7d95bd54ea6060f5339f7643daa451eb7c222b6529bdc6f82d8'),
  (S/'asr18_BPE9223_full_run_native_v2.py',S/'asr18_BPE9223_full_run_native_v3.py','18a8f76fbe9f73d9fefa53fe831382c60b74a90bd3b92e4878c8cc35c33e813c'),
  (R/'asr18_BPE9223_owned_GPU_runtime_v2.py',R/'asr18_BPE9223_owned_GPU_runtime_v3.py','f4c71c31e6c0de063e32cc6996f8272a4c440b9d4324eedc314252c710cd375a')]
 replacements={'bpe_full_run_lifecycle_v2':'asr18_BPE9223_lifecycle_v3'}
 for old,new,expected in pairs:
  assert m.sha(new)==expected;replacements[old.stem]=new.stem;replacements[m.sha(old)]=expected
 deltas=[];activeimports=set()
 for old,new,expected in pairs:
  transformed=old.read_text()
  for before,after in replacements.items():transformed=transformed.replace(before,after)
  assert ast.dump(ast.parse(transformed))==ast.dump(ast.parse(new.read_text())),str(new)
  assert transformed==new.read_text(),str(new)+' nonliteral source delta'
  diff=''.join(difflib.unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),fromfile=old.name,tofile=new.name))
  target=O/(new.name+'.diff');target.write_text(diff)
  deltas.append({'baseline':rec(old),'successor':rec(new),'exact_literal_only_delta':rec(target)})
  for node in ast.walk(ast.parse(new.read_text())):
   if isinstance(node,ast.ImportFrom) and node.module:activeimports.add(node.module)
   elif isinstance(node,ast.Import):activeimports.update(x.name for x in node.names)
 assert 'bpe_full_run_lifecycle_v2' not in activeimports and 'asr18_full_run_serial_DEV_9223_v2' not in activeimports and 'asr18_full_run_DEV_runtime_9223_v2' not in activeimports
 import asr18_full_run_serial_DEV_9223_v2 as previous
 import asr18_full_run_serial_DEV_9223_v3 as current
 import asr18_BPE9223_lifecycle_v3 as lifecycle
 assert current.progress is lifecycle.progress and current.milestones is lifecycle.milestones
 planpath=N/'full_configuration_01/prepared_original_first20_plan.json';assert m.sha(planpath)=='143e0807673daa69796409acc45d6cd03e78ddf73ecc3f672d13b50df4e90591'
 plan=json.loads(planpath.read_text());state=plan['final_private_planning_sampler'];frontier=current.progress(state)
 assert frontier['optimizer_updates']==20 and frontier['broad_draws']==280 and frontier['cleaner_draws']==0
 assert state['population_mean_U']['weak_label_continued_training']==90746310/2631115 and state['population_mean_U']['cleaner_data_fine_tuning']==10621773/352768
 try:previous.progress(state)
 except ValueError as error:old_rejection=str(error)
 else:raise AssertionError('Old mean path should reject the real9223 sampler')
 mapper=S/'asr18_DEV_weight_only_mapper_9223_v1.py';mapper_tree=ast.parse(mapper.read_text())
 selection=next(n for n in mapper_tree.body if isinstance(n,ast.FunctionDef) and n.name=='selection_plan')
 imports=[n for n in ast.walk(selection) if isinstance(n,ast.ImportFrom) and n.module=='asr18_full_run_serial_DEV_9223_v2']
 assert len(imports)==1 and [a.name for a in imports[0].names]==['checkpoint_format_view']
 format_tree=ast.parse((S/'asr18_full_run_serial_DEV_9223_v2.py').read_text());fmt=next(n for n in format_tree.body if isinstance(n,ast.FunctionDef) and n.name=='checkpoint_format_view')
 assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('progress','milestones','next_boundary_request','official_DEV') for n in ast.walk(fmt))
 assert not any(k in sys.modules for k in ('torch','numpy','transformers','nemo'));g.check()
 report={'kind':'ASR18_independent_six_successor_DEV_lifecycle_source_metadata_review_v1','passed':True,'reviewer':'/root/label_bpe_audit','scope':'exact literal source delta and genuine plan scalar progress only; no native/DEV execution authority',
  'source_deltas':deltas,'six_sources_equal_prior_after_only_declared_import_self_pin_substitutions':True,
  'active_six_have_no_old_lifecycle_or_serial_runtime_import':True,'new_serial_progress_milestones_are_exact_current_lifecycle3_functions':True,
  'actual_newmean_sampler_plan':rec(planpath),'actual_newmean_frontier':frontier,'actual_new_mean_U':state['population_mean_U'],'previous_active_mean_route_rejected_same_real_plan':old_rejection,
  'legacy_mapper_import_scope':'Only oldserial2.checkpoint_format_view; no lifecycle/progress/milestones call in this function. Prior all207/854 format/selection scope reused.',
  'all_scoring_retention_parameter_restore_resource_and_process_bodies_unchanged':True,'new_concrete_source_blockers':[],
  'remaining':['Receive and independently verify final resealed source manifest/config/policy/CPU template maps point to these successor pins before final ROOT execution review'],
  'actual_audio_GPU_model_payload_jobs':0,'production_admitted_here':False}
 (O/'source_findings_v1.json').write_text(json.dumps(report,indent=2)+'\n');g.publish(completed=True,findings=rec(O/'source_findings_v1.json'));print(json.dumps({'passed':True,'findings':rec(O/'source_findings_v1.json')}))
if __name__=='__main__':main()
