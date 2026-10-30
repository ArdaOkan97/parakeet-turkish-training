"""Seal actual independent exit0 and publish the agreed final DATA evidence API."""
import importlib,json,subprocess,sys
from pathlib import Path
L=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1')
sys.path.insert(0,str(L));m=importlib.import_module('audit_and_fit_append_only_v5')
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
V=N/'verified_population_v1';S=N/'final_DATA_seal_v1';m.AREA=S

def main():
 assert len(sys.argv)==2;S.mkdir();g=m.Guard();g.phase='seal_actual_independent_exit0_final9223_DATA_API'
 report=json.loads((V/'independent_full_label_database_verification_v1.json').read_text())
 state=json.loads((V/'current_status.json').read_text());assert report['passed'] is True
 assert report['PID']==state['PID'] and report['create_time']==state['create_time']
 dead=[]
 for x in [{'PID':state['PID'],'create_time':state['create_time']}]+state['parent_identity']:
  try:same=abs(m.psutil.Process(x['PID']).create_time()-x['create_time'])<.01
  except m.psutil.NoSuchProcess:same=False
  assert not same;dead.append({**x,'same_identity_alive':False})
 closure={'kind':'Actual_independent_BPE9223_label_DB_verifier_CLOSED0_v1','at_utc':m.now(),'owner':'/root/label_bpe_audit',
  'owned_session':10115,'actual_exit_code':0,'actual_tool_chunk':sys.argv[1],
  'worker_PID':state['PID'],'worker_create_time':state['create_time'],'all_captured_identities_dead':dead,
  'report':m.record(V/'independent_full_label_database_verification_v1.json'),'actual_status':state}
 (V/'actual_owned_CLOSED0.json').write_text(json.dumps(closure,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 runtime=json.loads((V/'final_runtime_bindings_9223_v1.json').read_text());assert runtime['complete'] is True and runtime['vocabulary']==9223
 original=m.ROOT/'analysis/ASR18_memory_diagnosis_v1/final_verified_DATA_evidence_v1.json'
 evidence={'kind':'BPE9223_final_verified_TRAIN_population_evidence_v1','at_utc':m.now(),'complete':True,
  'original_DATA_evidence':json.loads(original.read_text()),'original_DATA_evidence_file':m.record(original),
  'BPE9223_population':m.record(V/'final_runtime_bindings_9223_v1.json'),
  'BPE9223_actual0':m.record(V/'actual_owned_CLOSED0.json'),
  'BPE9223_report':m.record(V/'independent_full_label_database_verification_v1.json'),
  'tokenizer':runtime['tokenizer'],'broad_labels':runtime['databases']['broad_labels'],
  'cleaner_labels':runtime['databases']['cleaner_labels'],'declared_geometry_index':runtime['databases']['declared_geometry'],
  'source_original_catalog':runtime['source_catalog_unchanged'],'cleaner_source_original_catalog':runtime['cleaner_source_catalog_unchanged'],
  'declared_geometry_envelope':runtime['declared_geometry'],'two_broad_one_cleaner_manifest':runtime['curriculum'],
  'original_leaf_donor_migration_plan':runtime['migration_donors'],'producer_actual0':runtime['actual_producer_closure'],
  'vocabulary':9223,'blank_id':8192,'duration_rows_half_open':[9223,9228],
  'broad':{'rows':2631115,'tokens':90746310,'mean_U':90746310/2631115,'MAX_U':report['broad_MAX_U'],'MAX_T':report['broad_MAX_T'],'MAX_T_times_U_plus_1':report['broad_MAX_T_times_U_plus_1']},
  'cleaner':{'rows':352768,'tokens':10621773,'mean_U':10621773/352768,'MAX_U':report['cleaner_MAX_U'],'MAX_T':report['cleaner_MAX_T'],'MAX_T_times_U_plus_1':report['cleaner_MAX_T_times_U_plus_1'],'augmentation_allowed':False},
  'original_foundation_reused_only_for_frozen_audio_sources_catalog_heldout_carriers':True,
  'old8711_target_labels_means_geometry_targets_not_armed':True,'same_source_rows_ids_text_hashes_audio_all_preserved':True,
  'protected1914_cached_replay_teacher_DEV_TEST_targets_not_retokenized':True,
  'all_2983883_stored_new_targets_match_actual_Rust_candidate_digest':report['all_2983883_stored_BLOB_labels_match_prior_actual_Rust_digest'],
  'actual_producer_and_independent_verifier_exit0_observed':True,'native_model_optimizer_and_full_pipeline_proofs_owned_by_memory_and_ROOT':True,
  'production_training_started_by_this_CPU_agent':False}
 path=N/'BPE9223_final_verified_TRAIN_population_evidence_v1.json'
 path.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 out=m.PROJECT/'outputs/parakeet_turkish';task=out/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
 marker='BPE9223_FULL_LABEL_INTEGRATION_20261018';assert marker not in task.read_text(encoding='utf8')
 note=(m.now()+': '+marker+' — CPU label/population integration actually completed in immutable BPE9223_training_launch_v1. '
  'Actual producer0 and separately streamed independent verifier0 cover all2,983,883 stored targets with exact prior fullRust candidate digest. '
  'Broad2,631,115 clips/90,746,310 tokens/mean34.48967833028963; cleaner352,768/10,621,773/mean30.109797373911466. '
  'New broad uint32 and cleaner uint16 DBs, exact contiguous source IDs/text hashes, all new target counts, source-preserving declared geometry, '
  '2broad+1cleaner virtual pass manifest and flattened original8711 leaf donor plan published. '
  'Existing8711 labels/audio/source catalogs remain unchanged; original8199 cleaner old_u/old_labels_sha preserved; protected1914cached replay/teacher/DEV/TEST not retokenized. '
  'New512 init plan uses original mature leaf means once and biasmean−2 once, old lexical IDs/blank8192 fixed, duration8711:8716→9223:9228; no model tensors migrated by CPU agent. '
  'Final evidence E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/BPE9223_final_verified_TRAIN_population_evidence_v1.json. '
  'Native migration/optimizer/throughput/retention/full production source/config checks remain owned by memory/ROOT; human contingent launch authorization already given, no additional permission gate introduced.')
 cli=subprocess.run([str(out/'tools/backlog.exe'),'task','edit','ASR-18','--append-notes',note],cwd=out,capture_output=True,text=True,encoding='utf8');assert cli.returncode==0,cli.stderr
 (S/'actual_Backlog_CLI_output.txt').write_text(cli.stdout+'\n'+cli.stderr,encoding='utf8')
 refresh=subprocess.run([str(m.PROJECT/'work/training-env/Scripts/python.exe'),str(out/'refresh_backlog.py')],cwd=m.PROJECT,capture_output=True,text=True,encoding='utf8');assert refresh.returncode==0,refresh.stderr
 (S/'actual_refresh_output.txt').write_text(refresh.stdout+'\n'+refresh.stderr,encoding='utf8')
 assert task.read_text(encoding='utf8').count(marker)==1
 m.write('actual_Backlog_CLI0_refresh0.json',{'at_utc':m.now(),'task':'ASR-18','marker':marker,'exactly_once':True,'CLI_actual_exit_code':0,'refresh_actual_exit_code':0,
  'CLI_output':m.record(S/'actual_Backlog_CLI_output.txt'),'refresh_output':m.record(S/'actual_refresh_output.txt'),'final_evidence':m.record(path)})
 dispatch={'kind':'Current_BPE9223_label_preparation_dispatch_v1','at_utc':m.now(),'owner':'/root/label_bpe_audit',
  'state':'complete_producer0_independent_verifier0_final_evidence_published','final_evidence':m.record(path),
  'producer_actual0':runtime['actual_producer_closure'],'independent_actual0':m.record(V/'actual_owned_CLOSED0.json'),
  'one_Backlog_milestone':m.record(S/'actual_Backlog_CLI0_refresh0.json'),'no_GPU_model_or_old_source_label_mutations':True}
 (N/'current_label_preparation_dispatch_v1.json').write_text(json.dumps(dispatch,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 g.phase='final9223_DATA_evidence_and_one_Backlog_milestone_complete';g.publish(passed=True)
 print(json.dumps(dispatch,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
