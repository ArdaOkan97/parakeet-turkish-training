"""Actual Rust relabeling only: same immutable TRAIN audio/source rows, new DBs.

No BPE refit, audio read, protected replay retokenization, model or GPU work.
"""
import array,collections,hashlib,importlib,json,os,sqlite3,sys,time
from pathlib import Path
L=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1')
sys.path.insert(0,str(L));m=importlib.import_module('audit_and_fit_append_only_v5')
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
OUT=N/'label_generation_v1';INDEX=N/'indexes';m.AREA=OUT
TOKEN=L/'corpus_driven_BPE9223_v2/tokenizer.json'
OLDG=m.ROOT/'indexes/ASR18_declared_training_geometry_v1.sqlite'

def write(name,value):m.write(name,value)
def digest_ids(pool,ordinal,sid,ids):return (m.canonical([pool,ordinal,sid,ids])+'\n').encode()
def labels_sha(ids):return hashlib.sha256(json.dumps(list(ids),sort_keys=True).encode()).hexdigest()

def create(path,sql):
 assert not path.exists();db=sqlite3.connect(path)
 db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA synchronous=FULL')
 db.execute('PRAGMA cache_size=-32768');db.execute('PRAGMA mmap_size=0');db.execute('PRAGMA temp_store=MEMORY')
 db.execute(sql);db.commit();return db

def donor_plan(candidate):
 pieces={i:p for p,i in candidate['model']['vocab'].items()};leaves={};donors=[]
 additions=json.loads((L/'corpus_driven_BPE9223_v2/added_pieces.json').read_text())
 for x in additions:
  i=x['id'];assert 8711<=i<9223
  def donor(j):return [j] if j<8711 else leaves[j]
  ids=donor(x['left_id'])+donor(x['right_id']);assert 0<len(ids)<=1024 and all(0<=j<8711 and j!=8192 for j in ids)
  assert ''.join(pieces[j] for j in ids)==x['piece']
  leaves[i]=ids;donors.append({'HF_id':i,'piece':x['piece'],'left_id':x['left_id'],'right_id':x['right_id'],
   'merge_rank':x['merge_rank'],'original_leaf_donor_ids':ids,'weighted_pair_frequency':x['weighted_pair_frequency']})
 source=m.PROJECT/'outputs/parakeet_turkish/asr18_narrow_training_primitives_v1.py'
 result={'kind':'BPE9223_original_mature_leaf_row_initialization_plan_v1','at_utc':m.now(),
  'candidate_tokenizer':m.record(TOKEN),'previous_policy_source_readonly':m.record(source),
  'source_lexical_rows':8711,'target_lexical_rows':9223,'source_head_rows':8716,'target_head_rows':9228,
  'blank_id_unchanged':8192,'old_lexical_identity_half_open':[0,8711],
  'old_duration_to_new_rows':[{'old':8711+d,'new':9223+d} for d in range(5)],
  'new_lexical_half_open':[8711,9223],'initialization':'For each new row, arithmetic mean of ORIGINAL mature8711 leaf rows, occurrences retained; embedding/head mean once, bias leaf mean minus2 ONCE.',
  'recursive_averaging_or_repeated_bias_penalty':False,'RNG_policy':'CPU metadata job draws no RNG; native owner must preserve all RNG around Parameter constructors.',
  'optimizer_policy':'Existing source rows/moments/scalar clocks unchanged; memory owner creates separate new-row Adam state on the same fresh Adam objects at step0; oldviews retain mature2000.',
  'training_token_ID_permutation':False,'blank_at_end_is_future_export_only':True,
  'protected1914_replay_and_teacher_targets_unchanged':True,'donors':donors,'model_migration_executed':False}
 write('row_migration_and_leaf_initialization_plan.json',result)

def main():
 OUT.mkdir();INDEX.mkdir(exist_ok=True);g=m.Guard();g.phase='fresh_9223_relabeling_admission'
 assert sys.byteorder=='little' and array.array('I').itemsize==4 and array.array('H').itemsize==2
 assert m.sha(TOKEN)=='f2d80ffee7ed1b006a83b52cc77bfb28bed4ba850352b5222bb3a49d6cfbac4a'
 assert m.sha(L/'ROOT_final_BPE9223_delivery_review_v1.json')=='0a6b0e7cdc151551931cd9d61312128b1d01ba391d99761f7925bfa67daadb6b'
 val=json.loads((L/'corpus_driven_BPE9223_v2/full_TRAIN_validation.json').read_text())
 sources={k:m.stat(p) for k,p in [('broad_text',m.BROAD),('old_broad_labels',m.BROAD_LABELS),('clean_text',m.CLEAN),('old_clean_labels',m.CLEAN_LABELS),('old_geometry',OLDG)]}
 base=g.process.memory_info().private;parents=sum(p.memory_info().private for p in g.parents);mem=m.memory();forecast=512000000
 limit=mem['commit_headroom_bytes']/(1-mem['commit_ratio'])
 assert base<=128000000 and mem['commit_ratio']+(forecast-base)/limit<.9 and mem['commit_headroom_bytes']-(forecast-base)>=2000000000
 assert parents+forecast<2000000000
 write('recipe_and_source_bindings_pinned_before_labels.json',{'kind':'BPE9223_immutable_relabeling_recipe_v1','at_utc':m.now(),
  'source':m.record(__file__),'guard_source':m.record(L/'audit_and_fit_append_only_v5.py'),
  'candidate':m.record(TOKEN),'existing_full_Rust_validation':m.record(L/'corpus_driven_BPE9223_v2/full_TRAIN_validation.json'),
  'sources':sources,'forecast_leaf_bytes':forecast,'actual_baseline':base,'actual_parents':parents,'memory':mem,
  'fixed_broad_rows':2631115,'fixed_broad_new_tokens':90746310,'fixed_cleaner_rows':352768,'fixed_cleaner_new_tokens':10621773,
  'no_audio_decode_BPE_refit_source_reaudit_quality_filter_or_replay_change':True,'complete_population_only':True,
  'old_cleaner_old_u_and_old_labels_sha_original8199_preserved':True,'CPU_thread_caps':1,'maximum_IO_bytes':16000000000})
 from tokenizers import Tokenizer
 assert not any(x in sys.modules for x in ('torch','transformers','nemo'))
 tokenizer=Tokenizer.from_file(str(TOKEN));candidate=json.loads(TOKEN.read_text());donor_plan(candidate)
 broad_path=INDEX/'broad_labels_9223_v1.sqlite';clean_path=INDEX/'cleaner_labels_9223_v1.sqlite';geo_path=INDEX/'declared_geometry_9223_v1.sqlite'
 broad=create(broad_path,'CREATE TABLE labels (ordinal INTEGER PRIMARY KEY, target_ids_u32 BLOB, target_tokens INTEGER, text_sha256 TEXT, conservative_estimated_joint_bytes INTEGER)')
 broad.execute('CREATE TABLE progress (id INTEGER PRIMARY KEY CHECK(id=1), frontier INTEGER, rows INTEGER, tokens INTEGER, large_geometry_review_rows INTEGER)');broad.commit()
 geometry=create(geo_path,'CREATE TABLE geometry (ordinal INTEGER PRIMARY KEY, source_component INTEGER NOT NULL, native_sr INTEGER, native_samples INTEGER, clean16k_samples INTEGER, mel_frames INTEGER, encoder_frames INTEGER, target_tokens INTEGER NOT NULL, status TEXT NOT NULL)')
 src=m.ro(m.BROAD);src.execute('ATTACH DATABASE ? AS labels_old',(m.BROAD_LABELS.as_uri()+'?mode=ro&immutable=1',));src.execute('ATTACH DATABASE ? AS oldgeo',(OLDG.as_uri()+'?mode=ro&immutable=1',))
 query='SELECT c.ordinal,c.unique_id,c.dataset,c.text,l.target_tokens,l.text_sha256,l.conservative_estimated_joint_bytes,g.source_component,g.native_sr,g.native_samples,g.clean16k_samples,g.mel_frames,g.encoder_frames,g.target_tokens,g.status FROM clips c JOIN labels_old.labels l ON l.ordinal=c.ordinal JOIN oldgeo.geometry g ON g.ordinal=c.ordinal ORDER BY c.ordinal'
 dig=hashlib.sha256();stats={};bcount=btokens=large=0;groups={};maxgeo={};g.phase='actual_Rust_broad9223_labels_and_preserved_audio_geometry'
 def process_broad(batch):
  nonlocal bcount,btokens,large
  labels=[];geos=[];enc=tokenizer.encode_batch([x[3] for x in batch])
  for row,e in zip(batch,enc):
   ordinal,sid,dataset,text,oldU,textsha,oldjoint,component,sr,native,cleanN,mel,T,geoU,status=row
   assert ordinal==bcount+1 and geoU==oldU and status=='declared' and hashlib.sha256(text.encode()).hexdigest()==textsha
   ids=e.ids;U=len(ids);assert U>0 and all(0<=x<9223 and x!=8192 for x in ids)
   denominator=(oldU+1)*8716*4;assert oldjoint%denominator==0
   estimatedT=oldjoint//denominator;assert estimatedT>=T
   dense=estimatedT*(U+1)*9228*4
   labels.append((ordinal,array.array('I',ids).tobytes(),U,textsha,dense))
   geos.append((ordinal,component,sr,native,cleanN,mel,T,U,status))
   bcount+=1;btokens+=U;large+=dense>1700000000
   group=groups.setdefault(dataset,{'rows':0,'tokens':0,'max_U':0,'max_T':0,'max_T_times_U_plus1':0})
   group['rows']+=1;group['tokens']+=U;group['max_U']=max(group['max_U'],U);group['max_T']=max(group['max_T'],T);group['max_T_times_U_plus1']=max(group['max_T_times_U_plus1'],T*(U+1))
   for key,value in [('MAX_U',U),('MAX_T',T),('MAX_T_times_U_plus_1',T*(U+1))]:
    if value>maxgeo.get(key,{}).get('value',0):maxgeo[key]={'value':value,'ordinal':ordinal,'source_id':sid,'T':T,'U':U}
   dig.update(digest_ids('broad',ordinal,sid,ids));g.pace(len(text.encode())+len(ids)*4+200)
  broad.executemany('INSERT INTO labels VALUES (?,?,?,?,?)',labels);geometry.executemany('INSERT INTO geometry VALUES (?,?,?,?,?,?,?,?,?)',geos)
  if bcount%8192==0:
   broad.execute('INSERT OR REPLACE INTO progress VALUES (1,?,?,?,?)',(bcount,bcount,btokens,large));broad.commit();geometry.commit();g.publish(broad_rows=bcount,broad_tokens=btokens)
 batch=[]
 for row in src.execute(query):
  batch.append(row)
  if len(batch)==256:process_broad(batch);batch.clear()
 if batch:process_broad(batch)
 src.close();assert bcount==2631115 and btokens==90746310
 broad.execute('INSERT OR REPLACE INTO progress VALUES (1,?,?,?,?)',(bcount,bcount,btokens,large));broad.commit();geometry.commit();broad.close();geometry.close()
 clean=create(clean_path,'CREATE TABLE labels (ordinal INTEGER PRIMARY KEY, source_id TEXT NOT NULL UNIQUE, text_sha TEXT NOT NULL, frames INTEGER NOT NULL, old_u INTEGER NOT NULL, old_labels_sha TEXT NOT NULL, new_u INTEGER NOT NULL, new_labels_sha TEXT NOT NULL, label_ids BLOB NOT NULL, full_joint_bytes INTEGER NOT NULL)')
 old=m.ro(m.CLEAN_LABELS);cursor=iter(old.execute('SELECT ordinal,source_id,text_sha,frames,old_u,old_labels_sha,new_u,new_labels_sha FROM labels ORDER BY ordinal'))
 ccount=ctokens=0;cgroups={};cmaxU=cmaxT=0;cmaxcells=0;g.phase='actual_Rust_cleaner9223_labels_original8199_metadata_preserved'
 def process_clean(batch):
  nonlocal ccount,ctokens,cmaxU,cmaxT,cmaxcells
  out=[];enc=tokenizer.encode_batch([x[1]['text'] for x in batch])
  for (ordinal,row),e in zip(batch,enc):
   prior=next(cursor);po,sid,textsha,T,oldU,oldsha,priorU,priorsha=prior
   assert po==ordinal-1==ccount and sid==row['source_id'] and textsha==hashlib.sha256(row['text'].encode()).hexdigest()
   assert T==row['encoder_frames'] and oldU==row['target_tokens'] and oldsha==labels_sha(row['tr_metadata_labels'])
   ids=e.ids;U=len(ids);assert U>0 and all(0<=x<9223 and x!=8192 for x in ids)
   out.append((po,sid,textsha,T,oldU,oldsha,U,labels_sha(ids),array.array('H',ids).tobytes(),T*(U+1)*9228*4))
   ccount+=1;ctokens+=U;cmaxU=max(cmaxU,U);cmaxT=max(cmaxT,T);cmaxcells=max(cmaxcells,T*(U+1))
   group=cgroups.setdefault(row['dataset'],{'rows':0,'tokens':0,'max_U':0});group['rows']+=1;group['tokens']+=U;group['max_U']=max(group['max_U'],U)
   dig.update(digest_ids('cleaner',ordinal,sid,ids));g.pace(len(row['text'].encode())+len(ids)*2+200)
  clean.executemany('INSERT INTO labels VALUES (?,?,?,?,?,?,?,?,?,?)',out)
  if ccount%8192==0:clean.commit();g.publish(cleaner_rows=ccount,cleaner_tokens=ctokens)
 batch=[]
 with m.CLEAN.open('rb') as f:
  for ordinal,line in enumerate(f,1):
   g.pace(len(line));row=json.loads(line);assert row['language']=='tr' and row['dataset_split'] in ('train','validated')
   batch.append((ordinal,row))
   if len(batch)==256:process_clean(batch);batch.clear()
 if batch:process_clean(batch)
 assert next(cursor,None) is None;old.close();clean.commit();clean.close()
 assert ccount==352768 and ctokens==10621773
 assert dig.hexdigest()==val['ordered_candidate_labels_sha256'],('Prior actual Rust candidate digest differs',dig.hexdigest())
 for key,path in [('broad_text',m.BROAD),('old_broad_labels',m.BROAD_LABELS),('clean_text',m.CLEAN),('old_clean_labels',m.CLEAN_LABELS),('old_geometry',OLDG)]:assert m.stat(path)==sources[key]
 g.phase='closed_new_label_databases_hash_and_population_publication';g.check()
 records={k:m.record(path) for k,path in [('broad_labels',broad_path),('cleaner_labels',clean_path),('declared_geometry',geo_path)]}
 population={'kind':'BPE9223_complete_same_audio_TR_training_population_v1','at_utc':m.now(),'complete':True,'PID':os.getpid(),'create_time':g.main_identity['create_time'],
  'tokenizer':m.record(TOKEN),'source_catalog_unchanged':m.stat(m.BROAD),'cleaner_source_catalog_unchanged':m.stat(m.CLEAN),
  'source_label_audit_already_complete':m.record(L/'all_TRAIN_extension_v5/source_label_audit.json'),
  'source_recipe':m.record(OUT/'recipe_and_source_bindings_pinned_before_labels.json'),'databases':records,
  'broad':{'rows':bcount,'tokens':btokens,'mean_U':btokens/bcount,'ordinal_min':1,'ordinal_max':bcount,'groups':groups,'declared_geometry_maxima':maxgeo,'conservative_dense_joint_over1p7GB_rows':large},
  'cleaner':{'rows':ccount,'tokens':ctokens,'mean_U':ctokens/ccount,'ordinal_min':0,'ordinal_max':ccount-1,'groups':cgroups,'MAX_U':cmaxU,'MAX_T':cmaxT,'MAX_T_times_U_plus_1':cmaxcells,'augmentation_allowed':False,'original8199_old_metadata_fields_preserved':True},
  'ordered_all2983883_new_label_ids_sha256':dig.hexdigest(),'matches_prior_actual_full_Rust_validation_exactly':True,
  'curriculum':{'broad_complete_passes':2,'cleaner_complete_passes':1,'broad_TR_draws':2*bcount,'cleaner_TR_draws':ccount,'total_TR_draws':2*bcount+ccount,
   'ordering':'Caller-owned virtual CompletePassCursor; no seed/RNG/frontier chosen by this producer','cross_pass_window_forbidden':True},
  'protected1914_replay_targets_teacher_and_all_source_audio_unchanged':True,'no_quality_language_disagreement_filters':True,
  'runtime_checks_before_consumed_draw':['source identity/text hash/full target count and IDs','declared native/rate/resampled/physical-valid/encoder geometry','heldout PCM exclusion','resource admission'],
  'runtime_mismatch_action':'Stop before draw; no crop/drop/skip/pad or partial-mean arming; reconcile source/catalog/labels/denominator explicitly.',
  'TRAIN_admitted_by_this_CPU_producer':False,'producer_closure_and_independent_population_verification_required':True,
  'elapsed_seconds':time.monotonic()-g.started,'peak_private_bytes':g.peak,'actual_process_IO_bytes':g.process.io_counters().read_bytes+g.process.io_counters().write_bytes-g.initial_IO}
 write('complete_label_population_v1.json',population);g.phase='complete9223_labels_population_waiting_actual_exit_and_independent_verify';g.publish(passed=True)
 print(json.dumps({'population':m.record(OUT/'complete_label_population_v1.json'),'databases':records,'seconds':time.monotonic()-g.started}),flush=True)
if __name__=='__main__':main()
