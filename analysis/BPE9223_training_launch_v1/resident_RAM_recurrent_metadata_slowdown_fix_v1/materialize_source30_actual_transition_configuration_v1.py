"""INACTIVE metadata assembler. Run only after separately ROOT-reviewed actual stop.
No validator/control, journal read, lease, GPU, restore or prefix-copy operation.
"""
import argparse,copy,hashlib,json
from pathlib import Path
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research';F=Path(__file__).parent
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
def require(v,m):
 if not v:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
 with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
def main(a):
 review=read(a.ROOT_review)
 require(review['accepted']is True and review['reviewer']=='/root'and review['scope']=='ASR18_canonical_archive_checkpoint_stop_v1',
  'Only exact reviewed stop metadata may be materialized; no production admission')
 require(review['materialization_source']==pin(__file__),'ROOT binds exact metadata assembler')
 for v in review['input_pins']:require(pin(v['path'])==v,'Frozen pre-stop source inputs unchanged')
 closure=F/'actual_source29_canonical_archive_checkpoint_transition_01/actual_transition_closure.json';value=read(closure)
 require(value['kind']=='ASR18_actual_source29_checkpoint_canonical_archive_transition_v1'and value['owned_session']==87302
  and value['actual_outer_TOOL_exit_code']==1 and value['normal0_claimed']is False and value['all_five_exact_dead']is True,
  'Genuine supported source29 intentional stop; actual final acceptance still ROOT responsibility')
 require(value['evidence']['ROOT_pre_stop_review']==pin(a.ROOT_review),'Exact actual pre-stop provenance')
 monitor=value['monitor_completion'];require(type(monitor)is dict and monitor['joined']is True and monitor['clean']is True and monitor['error']is None,
  'Actual original Monitor closure clean; concurrent guard failure is not an intentional transition')
 configuration=read(F/'candidate_source30_configuration_completed_CPU_pending_actual_stop_v1.json')
 require(configuration['prior_resident_RAM_canonical_archive_checkpoint_transition']is None,'Only None actual closure slot replaced')
 configuration['prior_resident_RAM_canonical_archive_checkpoint_transition']=pin(closure)
 sourcebase=F.parent/'resident_RAM_two_logical_packet_lookahead_v1/production_configuration_v1/configuration.json';previous=read(sourcebase)
 inverse=copy.deepcopy(configuration);del inverse['prepared_metadata_archive_representation_policy'];del inverse['prior_resident_RAM_canonical_archive_checkpoint_transition']
 require(inverse==previous,'Every original source configuration value exact after only two operational additions')
 folder=F/'production_configuration_v1';require(not folder.exists(),'One immutable final metadata materialization');folder.mkdir()
 config=folder/'configuration.json';write(config,configuration)
 keys=['worker_driver_source','parent_driver_source','resident_RAM_stream_configuration','resident_branch_namespace','checkpoint_interval_updates','source_checkpoint','source_configuration']
 oldruntime=read(B/'invocations/owner_000021/configuration.json');oldrestored=copy.deepcopy(oldruntime)
 olddiff={}
 for key in keys:
  if oldruntime[key]!=previous.get(key):olddiff[key]={'source':previous.get(key),'actual_runtime':oldruntime[key]}
  if key in previous:oldrestored[key]=previous[key]
  else:del oldrestored[key]
 require(oldrestored==previous,'Old actual runtime differs from proper source base only seven declared runtime kinds')
 pointer=value['evidence']['pointer'];saved=value['last_fully_saved_optimizer_updates'];latest=read(B/'latest_checkpoint.json')
 require(latest['pointer']==pointer and latest['committed_updates']==saved and latest['full_state_verified']is True,'Actual latest full saved checkpoint selected AFTER stop')
 future=dict(worker_driver_source=pin(R/'run_ASR18_BPE9223_resident_RAM_training_v30.py'),parent_driver_source=pin(R/'run_ASR18_BPE9223_resident_RAM_training_v30.py'),
  resident_RAM_stream_configuration=pin(config),resident_branch_namespace=str(B),checkpoint_interval_updates=200,source_checkpoint=pointer,source_configuration=pin(config))
 validation={'start_updates':saved+20,'end_updates':max(saved+20+1600,((saved//4000)+1)*4000+220),
  'formula':'start=restored+20; end=max(start+1600,next_scheduled4000boundary+220)',
  'entire_contiguous_original_timing_rows_required':True,'all_natural_save_and_actual_DEV_time_included':True,
  'actual_cleanup_boundary_scalar_archives_at_first_and_post_DEV_end_checkpoint':True,
  'no_cherry_pick_or_short_warm_success':True,'no_finite_training_stop':True,'full_target_updates':401072,'administrative_stop':None}
 inversepath=F/'final_source30_configuration_and_future7_runtime_bindings_v1.json'
 write(inversepath,dict(proper_source_base=pin(sourcebase),source_configuration=pin(config),inverse_after_only_two_operational_fields_exact=True,
  prior_source_to_actual_runtime_differences=olddiff,original_runtime_binding_kinds=keys,future_actual_runtime_bindings=future,actual_restore_frontier=saved,predeclared_long_validation=validation))
 output=F.parent/'resident_RAM_training_actual_23';require(not output.exists()and not(B/'invocations/owner_000022').exists()and not(B/'journals/commits_000022.jsonl').exists(),'Fresh next owner22/output23/journal22, no reuse')
 rootfuture=F/('ROOT_source30_canonical_archive_saved'+str(saved)+'_final_execution_review_v1.json')
 args=['--review',str(rootfuture),'--configuration',str(config),'--pointer',pointer['path'],'--latest-failure',read(B/'invocations/owner_000021/boot.json')['latest_failure']['path'],
  '--branch',str(B),'--output',str(output),'--start-invocation','22','--queue-bytes','268435456','--canonical-archive-resume',str(closure)]
 require(len(args)==18,'Exact 18-string future wrapper argument array')
 handoff={'kind':'ASR18_source30_actual_canonical_archive_transition_final_metadata_handoff_v1','production_admitted':False,
  'driver_source':future['worker_driver_source'],'wrapper_source':pin(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v30.py'),
  'RAM_native_source':pin(S/'asr18_BPE9223_grouped_RAM_native_v1.py'),'CPU_source':pin(S/'fastmss_resident_feature_stream_v11.py'),
  'DEV_helper_source':pin(S/'asr18_resident_serial350_DEV_v2.py'),'canonical_archive_transition_validator':pin(S/'asr18_canonical_archive_checkpoint_transition_v1.py'),
  'archive_helper_source':pin(F/'asr18_canonical_prepared_archive_v1.py'),'source_configuration':pin(config),'initial_checkpoint':pointer,
  'latest_failure':read(B/'invocations/owner_000021/boot.json')['latest_failure'],'canonical_archive_resume':pin(closure),
  'initial_owner_invocation':22,'bounded_RAM_queue_bytes':268435456,'periodic_DEV_interval_updates':4000,
  'prepared_metadata_archive_representation_policy':configuration['prepared_metadata_archive_representation_policy'],
  'packet_credit_policy':configuration['resident_RAM_packet_credit_policy'],'native_update_commit_pressure_policy':configuration['native_update_commit_pressure_policy'],
  'periodic_DEV_CPU_quiescence_policy':configuration['periodic_DEV_CPU_quiescence_policy'],
  'private_scalar_prefetch_source':configuration['private_scalar_prefetch_policy']['source'],
  'private_scalar_prefetch_actual_CPU_review':configuration['private_scalar_prefetch_policy']['actual_CPU_control_review'],
  'additional_prefetch_host_reservation_bytes':268435456,'human_approval':pin(F/'ROOT_direct_human_then_fix_authorization_v1.json'),
  'configuration_inverse_and_runtime7':pin(inversepath),'predeclared_actual_production_validation':validation,
  'actual_stop_ROOT_review':pin(a.ROOT_review),'actual_stop_closure':pin(closure),'exact_future_wrapper_arguments':args,
  'required_next_steps':['Separate ROOT typed-metadata scope','ONE actual typed-metadata0/exactdeaths','Final ROOT exact execution binding','ONE actual full-state restart','genuine redo then natural standard200 and positive','complete long contiguous through-DEV production validation'],
  'no_memory_saving_speed_or_complete_cause_claim':True}
 hp=F/'final_source30_actual_canonical_archive_transition_execution_handoff_v1.json';write(hp,handoff)
 print(json.dumps({'configuration':pin(config),'inverse_runtime7':pin(inversepath),'handoff':pin(hp),'actual_restored_frontier':saved,'validation':validation}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ROOT-review',required=True);main(p.parse_args())
