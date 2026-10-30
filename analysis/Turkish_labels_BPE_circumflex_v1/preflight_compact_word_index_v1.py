"""Readonly compact-fit shape preflight. No corpus re-audit or active DB edits."""
import array, hashlib, importlib, json, sys, time
from pathlib import Path

D=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1')
sys.path.insert(0,str(D))
base=importlib.import_module('audit_and_fit_append_only_v5')
V5=D/'all_TRAIN_extension_v5'
AREA=D/'compact_fit_v6'
AREA.mkdir(exist_ok=True)
base.AREA=AREA

def main():
 g=base.Guard();g.phase='readonly_immutable_word_index_shape_preflight'
 st=json.loads((V5/'current_status.json').read_text())
 dead=[]
 for x in [{'PID':st['PID'],'create_time':st['create_time']}]+st['parent_identity']:
  try:
   p=base.psutil.Process(x['PID']);same=abs(p.create_time()-x['create_time'])<.01
  except base.psutil.NoSuchProcess:same=False
  assert not same,('Old identity still alive',x)
  dead.append({**x,'same_identity_alive':False})
 partial=json.loads((V5/'learned_merges_partial.json').read_text())
 closure={'kind':'Actual_owned_label_BPE_worker_CLOSED1_v1','owner':'/root/label_bpe_audit',
  'at_utc':base.now(),'owned_session':99344,'actual_exit_code':1,'actual_tool_chunk':'d32e9a',
  'failure':'Finite16GB actual process IO quota','last_observed_status':st,
  'all_captured_identities_dead':dead,'complete_source_audit_reusable':base.record(V5/'source_label_audit.json'),
  'sealed_exact_merge_prefix':base.record(V5/'learned_merges_partial.json'),'sealed_prefix_count':len(partial),
  'mutable_working_pairs_members_not_safe_to_resume':True,'source_word_counts_immutable':True,
  'GPU_Adam_or_active_source_changes':False}
 base.write('actual_v5_CLOSED1.json',closure)
 (V5/'actual_owned_CLOSED1.json').write_text(json.dumps(closure,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 ptr={'kind':'Current_label_BPE_owned_dispatch_pointer_v1','at_utc':base.now(),'owner':'/root/label_bpe_audit',
  'status':'previous_fit_IO_guard_closed1_supported_compact_successor_preflight',
  'current_status_path':str(AREA/'current_status.json'),'actual_previous_closure':str(V5/'actual_owned_CLOSED1.json'),
  'complete_source_audit':str(V5/'source_label_audit.json'),'safe_sealed_prefix_count':len(partial),
  'no_GPU_or_active_tokenizer_model_source_label_mutations':True}
 for name in ('current_dispatch.json','current_status.json'):(D/name).write_text(json.dumps(ptr,indent=2)+'\n')
 db=base.ro(V5/'training_counts.sqlite')
 n=0;adj=0;maxlen=0;maxfreq=0;sumfreq=0;rows=0;digest=hashlib.sha256()
 tok=json.loads((D/'coverage_only_8712/tokenizer.json').read_text())['model']['vocab']
 maxid=0
 for wid,raw,freq in db.execute('SELECT id,seq,freq FROM words ORDER BY id'):
  seq=json.loads(raw);length=len(seq);n+=length;adj+=max(0,length-1);maxlen=max(maxlen,length)
  maxfreq=max(maxfreq,freq);sumfreq+=freq;maxid=max(maxid,max((tok[s] for s in seq),default=0));rows+=1
  digest.update((base.canonical([wid,raw,freq])+'\n').encode());g.pace(len(raw.encode())+24)
  if rows%100000==0:g.publish(words=rows,sequence_cells=n)
 k=db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]
 positive=db.execute('SELECT COUNT(*) FROM pairs WHERE freq>0').fetchone()[0]
 members=db.execute('SELECT COUNT(*) FROM members').fetchone()[0]
 maxpair=db.execute('SELECT MAX(freq) FROM pairs').fetchone()[0]
 db.close();g.check()
 result={'kind':'Compact_BPE_index_readonly_shape_preflight_v1','at_utc':base.now(),
  'immutable_words_database':base.stat(V5/'training_counts.sqlite'),
  'immutable_word_rows_ordered_sha256':digest.hexdigest(),'word_types':rows,
  'initial_sequence_cells_N':n,'initial_adjacencies_upper_member_count':adj,
  'weighted_word_occurrences':sumfreq,'maximum_sequence_length':maxlen,'maximum_word_frequency':maxfreq,
  'maximum_initial_token_ID':maxid,'current_all_created_pair_rows_K_upper_initial':k,
  'current_positive_pair_rows':positive,'current_partial_mutable_members_NOT_frontier':members,
  'maximum_current_weighted_pair_count':maxpair,'sealed_merge_prefix_count':len(partial),
  'source_audit':base.record(V5/'source_label_audit.json'),
  'sealed_merge_prefix':base.record(V5/'learned_merges_partial.json'),
  'metadata_only_no_corpus_reaudit_no_learning_changes':True}
 base.write('readonly_shape_preflight.json',result)
 g.phase='readonly_shape_preflight_complete';g.publish(result=str(AREA/'readonly_shape_preflight.json'))
 print(json.dumps(result))

if __name__=='__main__':main()
