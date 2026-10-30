"""Read-only state controls on an actually closed original-clean feature window."""
from dataclasses import replace
import argparse,json,os,shutil,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];R=Path(r'E:\gpu_training_scratchpad');A=R/'analysis/FastMSS_integration_v1'
sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from fastmss_clean_feature_window_v1 import (FeatureWindowReader,read_bound,record,write_new,cleanup_window,create_window,
 finalize_window,owner_folder,inventory,load_plan,scoped)
from fastmss_ordered_draw_binding_v1 import AugmentationDrawContext
from resource_watchdog_v2 import sample_memory
def guard():
 v=sample_memory(os.getpid(),'read_only_clean_feature_window_state');w,s=v['worker_private_memory'],v['system_memory']
 assert w['available'] and s['available'] and v['paired_sampling_within_one_second'] and w['private_bytes']<2_000_000_000
 assert s['commit_used_bytes']/s['commit_limit_bytes']<.90 and s['commit_available_bytes']>=2_000_000_000 and shutil.disk_usage(R).free>=100_000_000_000
def main(output):
 assert os.environ['CUDA_VISIBLE_DEVICES']=='-1';out=scoped(output);out.mkdir(exist_ok=False);checks={}
 def check(name,value):assert value,name;checks[name]=True
 def rejects(name,function):
  try:function()
  except (ValueError,KeyError,TypeError,IndexError,AssertionError):checks[name]=True
  else:raise AssertionError(name)
 run=A/'actual_cleaner_feature_window_01';report=read_bound(record(run/'report.json',guard),guard);owner=report['owner'];manifest=report['manifest']
 actual=json.loads(Path(str(run)+'_guard/closure.json').read_bytes());assert actual['actual_exit_code']==0 and actual['monitor_clean'] and actual['monitor_joined']
 declared,folder=owner_folder(owner,guard);before={p.name:record(p,guard) for p in folder.iterdir()};plan=load_plan(declared['plan'],guard);u=plan['updates'][0]
 contexts=[AugmentationDrawContext(**v) for v in u['contexts']]
 def reader(**kw):return FeatureWindowReader(owner,manifest,guard=guard,**kw)
 def take(r,index,context=None,transaction=None):return r.take_row(0,index,context=contexts[index] if context is None else context,
  native_transaction_sha256=u['expected_native_transaction_sha256'] if transaction is None else transaction,before_sampler_sha256=u['expected_before_sampler_sha256'])
 r=reader();rejects('incomplete_payload_rejected',lambda:r.prepared_payload(0))
 for index in range(len(contexts)):
  arrays=take(r,index);check('exact_array_keys_'+str(index),set(arrays)=={'input_features','attention_mask'});del arrays
 payload=r.prepared_payload(0);check('all_clean_protection_applied_false',payload['protection_kind']=='FastMSS_explicit_clean_phase_protection_v1' and all(v['augmentation']['applied'] is False for v in payload['rows']))
 payload['rows'][0]['augmentation']['applied']=True;check('payload_defensive_copy',r.prepared_payload(0)['rows'][0]['augmentation']['applied'] is False)
 rejects('row_cannot_be_reused',lambda:take(r,0))
 r=reader();rejects('out_of_order_rejected',lambda:take(r,1));rejects('failed_reader_poisoned',lambda:take(r,0))
 r=reader();rejects('weak_phase_context_rejected',lambda:take(r,0,replace(contexts[0],phase='weak_label_continued_training')))
 r=reader();rejects('wrong_transaction_rejected',lambda:take(r,0,transaction='0'*64))
 checkpoint=plan['source_parent_checkpoint']
 rejects('blind_skip_without_actual_verifier_rejected',lambda:reader(resume={'checkpoint':checkpoint,'committed_window_updates':1}))
 rejects('incomplete_resume_attestation_rejected',lambda:reader(resume={'checkpoint':checkpoint,'committed_window_updates':1},verify_resume=lambda value:{'passed':True}))
 rejects('missing_feature_child_closure_rejected',lambda:finalize_window(owner,[],guard=guard))
 boundary={'kind':'coherent_checkpoint_boundary','owner_sha256':owner['sha256'],'window_manifest_sha256':manifest['sha256'],
  'committed_window_updates':0,'checkpoint':checkpoint}
 rejects('unconsumed_window_retained_at_intermediate_boundary',lambda:cleanup_window(owner,closures=[v['actual_feature_closure'] for v in report['updates']],
  manifest_binding=manifest,boundary=boundary,verify_boundary=lambda value:True,
  evidence_path=out/'MUST_NOT_BE_WRITTEN.json',guard=guard))
 check('negative_cleanup_created_no_archive',not(out/'MUST_NOT_BE_WRITTEN.json').exists())
 rejects('one_shared_window_at_a_time',lambda:create_window(declared['plan'],window_id='MUST_NOT_CREATE_SECOND_WINDOW',augmentation_identity=declared['augmentation_identity'],frontend=declared['frontend'],guard=guard))
 check('all_feature_files_bytes_and_bindings_unchanged',before=={p.name:record(p,guard) for p in folder.iterdir()} and inventory(declared,folder)==sum(v['bytes'] for v in before.values()))
 check('no_Torch_frontend_or_audio_modules',not any(v in sys.modules for v in ('torch','transformers','soundfile','scipy','pyarrow')))
 write_new(out/'validation.json',{'passed':True,'checks':checks,'source':record(__file__,guard),'actual_closed_CPU_window_report':record(run/'report.json',guard),
  'owner':owner,'manifest':manifest,'cache_read_only_all_files_unchanged':True,'actual_optimizer_updates':0,'authoritative_frontiers_advanced':False,
  'resume_test_scope':'rejection only; no actual cleaner native checkpoint exists yet; callback true used only to exercise earlier-count cleanup rejection',
  'feature_window_retained_for_native_owner':True,'no_native_or_production_approval':True})
 print(json.dumps({'passed':True,'checks':len(checks),'no_cache_mutation':True}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
