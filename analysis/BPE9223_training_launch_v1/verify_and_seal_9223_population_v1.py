"""After real producer exit0: stream new DBs, verify full targets and mappings.

No tokenizer refit/text re-audit/audio/model/GPU operation.
"""
import array,hashlib,importlib,itertools,json,os,sys,time
from pathlib import Path
L=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1')
sys.path.insert(0,str(L));m=importlib.import_module('audit_and_fit_append_only_v5')
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
P=N/'label_generation_v1';V=N/'verified_population_v1';m.AREA=V

def main():
 assert len(sys.argv)==2;actual_chunk=sys.argv[1]
 V.mkdir();g=m.Guard();g.phase='actual_label_producer_exit0_and_independent_DB_verification'
 pop=json.loads((P/'complete_label_population_v1.json').read_text());state=json.loads((P/'current_status.json').read_text())
 assert pop['complete'] is True and pop['PID']==state['PID'] and pop['create_time']==state['create_time']
 identities=[{'PID':state['PID'],'create_time':state['create_time']}]+state['parent_identity'];dead=[]
 for x in identities:
  try:same=abs(m.psutil.Process(x['PID']).create_time()-x['create_time'])<.01
  except m.psutil.NoSuchProcess:same=False
  assert not same;dead.append({**x,'same_identity_alive':False})
 closure={'kind':'Actual_BPE9223_label_population_producer_CLOSED0_v1','at_utc':m.now(),'owner':'/root/label_bpe_audit',
  'owned_session':89834,'actual_exit_code':0,'actual_tool_chunk':actual_chunk,'all_captured_identities_dead':dead,
  'producer_population':m.record(P/'complete_label_population_v1.json'),'actual_status':state}
 (P/'actual_owned_CLOSED0.json').write_text(json.dumps(closure,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 assert g.process.memory_info().private<=128000000
 mem=m.memory();forecast=256000000;limit=mem['commit_headroom_bytes']/(1-mem['commit_ratio'])
 assert mem['commit_ratio']+forecast/limit<.9 and mem['commit_headroom_bytes']-forecast>=2000000000
 for name,rec in pop['databases'].items():assert m.stat(rec['path'])=={k:rec[k] for k in ('path','bytes','mtime_ns')}
 opened=[]
 def db(path):
  x=m.ro(path);opened.append(x);return x
 cat=db(m.BROAD);prior=db(m.BROAD_LABELS);new=db(pop['databases']['broad_labels']['path'])
 og=db(m.ROOT/'indexes/ASR18_declared_training_geometry_v1.sqlite');ng=db(pop['databases']['declared_geometry']['path'])
 digest=hashlib.sha256();bc=bt=maxU=maxT=maxCells=0;high=[]
 iters=[cat.execute('SELECT ordinal,unique_id FROM clips ORDER BY ordinal'),
  prior.execute('SELECT ordinal,target_tokens,text_sha256,conservative_estimated_joint_bytes FROM labels ORDER BY ordinal'),
  new.execute('SELECT ordinal,target_ids_u32,target_tokens,text_sha256,conservative_estimated_joint_bytes FROM labels ORDER BY ordinal'),
  og.execute('SELECT * FROM geometry ORDER BY ordinal'),ng.execute('SELECT * FROM geometry ORDER BY ordinal')]
 for c,o,n,geo0,geo in itertools.zip_longest(*iters):
  assert None not in (c,o,n,geo0,geo);bc+=1
  assert c[0]==o[0]==n[0]==geo0[0]==geo[0]==bc
  ordinal,sid=c;_,oldU,oldsha,oldDense=o;_,blob,U,textsha,dense=n
  assert textsha==oldsha and len(blob)==4*U and U>0
  ids=array.array('I');ids.frombytes(blob);assert all(x<9223 and x!=8192 for x in ids)
  assert geo[:7]==geo0[:7] and geo[8]==geo0[8]=='declared' and geo[7]==U and geo0[7]==oldU
  estT=oldDense//((oldU+1)*8716*4);assert oldDense==estT*(oldU+1)*8716*4
  assert dense==estT*(U+1)*9228*4
  T=geo[6];maxU=max(maxU,U);maxT=max(maxT,T);maxCells=max(maxCells,T*(U+1));bt+=U
  digest.update((m.canonical(['broad',ordinal,sid,list(ids)])+'\n').encode());g.pace(len(blob)+len(sid.encode())+200)
  if len(high)<8 and U>458:high.append({'ordinal':ordinal,'source_id':sid,'T':T,'U':U})
  if bc%50000==0:g.publish(verified_broad_rows=bc,verified_broad_tokens=bt)
 assert bc==2631115 and bt==90746310 and maxU==517
 progress=new.execute('SELECT * FROM progress').fetchall();assert progress==[(1,bc,bc,bt,pop['broad']['conservative_dense_joint_over1p7GB_rows'])]
 for x in opened:x.close()
 opened.clear();prior=db(m.CLEAN_LABELS);new=db(pop['databases']['cleaner_labels']['path'])
 cc=ct=cmaxU=cmaxT=cmaxCells=0
 oldfields='ordinal,source_id,text_sha,frames,old_u,old_labels_sha'
 a=prior.execute('SELECT '+oldfields+' FROM labels ORDER BY ordinal');b=new.execute('SELECT * FROM labels ORDER BY ordinal')
 for o,n in itertools.zip_longest(a,b):
  assert o is not None and n is not None and tuple(n[:6])==tuple(o)
  ordinal,sid,textsha,T,oldU,oldsha,U,newsha,blob,dense=n
  assert ordinal==cc and len(blob)==2*U and U>0
  ids=array.array('H');ids.frombytes(blob);assert all(x<9223 and x!=8192 for x in ids)
  assert hashlib.sha256(json.dumps(list(ids),sort_keys=True).encode()).hexdigest()==newsha
  assert dense==T*(U+1)*9228*4
  cc+=1;ct+=U;cmaxU=max(cmaxU,U);cmaxT=max(cmaxT,T);cmaxCells=max(cmaxCells,T*(U+1))
  digest.update((m.canonical(['cleaner',ordinal+1,sid,list(ids)])+'\n').encode());g.pace(len(blob)+len(sid.encode())+200)
  if cc%50000==0:g.publish(verified_cleaner_rows=cc,verified_cleaner_tokens=ct)
 assert cc==352768 and ct==10621773 and cmaxU==161
 for x in opened:x.close()
 assert digest.hexdigest()==pop['ordered_all2983883_new_label_ids_sha256']
 prior_validation=json.loads((L/'corpus_driven_BPE9223_v2/full_TRAIN_validation.json').read_text())
 assert digest.hexdigest()==prior_validation['ordered_candidate_labels_sha256']
 for rec in pop['databases'].values():assert m.record(rec['path'])==rec
 original_geom=m.ROOT/'analysis/BPE_weak_then_clean_v1/ASR18_final_declared_training_geometry_v1.json'
 oldenv=json.loads(original_geom.read_text())
 source_catalog_pin=oldenv['bindings']['final_catalog']
 assert {k:source_catalog_pin[k] for k in ('path','bytes','mtime_ns')}==pop['source_catalog_unchanged']
 cleaner_provenance=m.PROJECT/'work/data/TR_expanded_WorldSpeech_index_v1/index_provenance.json'
 cleaner_index=json.loads(cleaner_provenance.read_text())
 cleaner_source_pin={**cleaner_index['files']['rows.jsonl'],'mtime_ns':pop['cleaner_source_catalog_unchanged']['mtime_ns']}
 assert cleaner_source_pin['path']==pop['cleaner_source_catalog_unchanged']['path'] and cleaner_source_pin['bytes']==pop['cleaner_source_catalog_unchanged']['bytes']
 envelope={'kind':'BPE9223_complete_source_bound_declared_geometry_v1','at_utc':m.now(),'complete':True,
  'tokenizer':pop['tokenizer'],'geometry_index':pop['databases']['declared_geometry'],
  'broad_labels':pop['databases']['broad_labels'],'original_audio_geometry_provenance_only':m.record(original_geom),
  'all_non_target_geometry_fields_exactly_unchanged':True,
  'population':{'retained_rows':bc,'total_target_tokens':bt,'population_mean_U':bt/bc,'MAX_T':maxT,'MAX_U':maxU,'MAX_T_times_U_plus_1':maxCells,
   'MAX_native_samples':oldenv['population']['MAX_native_samples'],'MAX_clean16k_samples':oldenv['population']['MAX_clean16k_samples']},
  'hard_target_examples':high,'unknown_declared_geometry_rows':0,'physical_frontend_frames':'clean16k_samples//160+1',
  'valid_mask_frames':'clean16k_samples//160','encoder_frames':'ceil(valid_mask_frames/8)',
  'runtime_contract':oldenv['runtime_contract'],'TRAIN_admitted':False}
 m.write('declared_geometry_envelope_9223_v1.json',envelope)
 curriculum={'kind':'BPE9223_two_broad_one_cleaner_full_pass_manifest_v1','at_utc':m.now(),'complete':True,
  'tokenizer':pop['tokenizer'],'broad':{'rows':bc,'tokens':bt,'population_mean_U':bt/bc,'passes':2,'ordinal_domain':[1,bc],'labels':pop['databases']['broad_labels']},
  'cleaner':{'rows':cc,'tokens':ct,'population_mean_U':ct/cc,'passes':1,'ordinal_domain':[0,cc-1],'labels':pop['databases']['cleaner_labels'],'augmentation_allowed':False},
  'ordering':'Native caller virtual complete pass cursor preserves source RNG; producer chooses no seeds or draw frontiers.',
  'broad_pass1_to2_and_broad_to_cleaner_boundaries_must_not_be_crossed_in_a_window':True,
  'retention_failure_does_not_shorten_declared_passes':True,'total_TR_draws':2*bc+cc,'replay_clock_separate':True,
  'old8711_label_databases_sources_audio_and_protected1914_replay_unchanged':True}
 m.write('two_broad_one_cleaner_manifest_9223_v1.json',curriculum)
 report={'kind':'Actual_independent_BPE9223_full_label_database_verification_v1','at_utc':m.now(),'passed':True,
  'PID':os.getpid(),'create_time':g.main_identity['create_time'],'actual_producer_closure':m.record(P/'actual_owned_CLOSED0.json'),
  'population':m.record(P/'complete_label_population_v1.json'),'databases':pop['databases'],
  'broad_rows':bc,'broad_tokens':bt,'broad_mean_U':bt/bc,'broad_MAX_U':maxU,'broad_MAX_T':maxT,'broad_MAX_T_times_U_plus_1':maxCells,
  'cleaner_rows':cc,'cleaner_tokens':ct,'cleaner_mean_U':ct/cc,'cleaner_MAX_U':cmaxU,'cleaner_MAX_T':cmaxT,'cleaner_MAX_T_times_U_plus_1':cmaxCells,
  'all_2983883_stored_BLOB_labels_match_prior_actual_Rust_digest':True,'ordered_label_IDs_sha256':digest.hexdigest(),
  'broad_sourceID_ordinal_textSHA_matches_old_exactly':True,'cleaner_original8199_source_metadata_preserved_all_rows':True,
  'all_non_target_broad_geometry_fields_preserved_all_rows':True,'row_counts_ordinals_blob_widths_token_ranges_hashes_and_means_complete':True,
  'protected1914_replay_teacher_DEV_TEST_audio_untouched':True,'no_partial_mean_armed':True,
  'elapsed_seconds':time.monotonic()-g.started,'peak_private_bytes':g.peak}
 m.write('independent_full_label_database_verification_v1.json',report)
 runtime={'kind':'BPE9223_complete_training_label_population_runtime_bindings_v1','at_utc':m.now(),'complete':True,
  'vocabulary':9223,'blank_id':8192,'joint_width':9228,'duration_rows_half_open':[9223,9228],
  'tokenizer':pop['tokenizer'],'databases':pop['databases'],'source_catalog_unchanged':source_catalog_pin,
  'cleaner_source_catalog_unchanged':cleaner_source_pin,'original_cleaner_index_provenance':m.record(cleaner_provenance),
  'source_hashes_reused_from_sealed_original_foundation_and_index_with_current_stat_and_all_target_ID_digest_match':True,
  'complete_label_population':m.record(P/'complete_label_population_v1.json'),
  'declared_geometry':m.record(V/'declared_geometry_envelope_9223_v1.json'),
  'curriculum':m.record(V/'two_broad_one_cleaner_manifest_9223_v1.json'),
  'migration_donors':m.record(P/'row_migration_and_leaf_initialization_plan.json'),
  'independent_full_DB_verification':m.record(V/'independent_full_label_database_verification_v1.json'),
  'actual_producer_closure':m.record(P/'actual_owned_CLOSED0.json'),
  'original_audio_source_foundation_only':m.record(m.ROOT/'analysis/BPE_weak_then_clean_v1/ASR18_final_DATA_preparation_handoff_v1.json'),
  'old_foundation8711_target_means_must_not_be_used':True,
  'broad_mean_U':bt/bc,'cleaner_mean_U':ct/cc,'broad_label_BLOB':'little-endian uint32,4*target_tokens',
  'cleaner_label_BLOB':'little-endian uint16,2*new_u','cleaner_old_u_old_labels_sha':'original8199 metadata retained',
  'no_training_lexical_permutation_blank_end_is_export_only':True,'sampler_RNG_frontier_LR_schedule_not_chosen_here':True,
  'native_model_optimizer_runtime_source_and_full_final_admission_owned_by_ROOT_and_memory':True,
  'independent_verifier_actual_closure_still_required_before_launch':True,'production_launched_by_this_CPU_agent':False}
 m.write('final_runtime_bindings_9223_v1.json',runtime)
 g.phase='complete_independent9223_label_verification_and_runtime_bindings';g.publish(passed=True)
 print(json.dumps({'runtime_bindings':m.record(V/'final_runtime_bindings_9223_v1.json'),'report':m.record(V/'independent_full_label_database_verification_v1.json'),'seconds':time.monotonic()-g.started}),flush=True)
if __name__=='__main__':main()
