"""Observe actual fit exit0 and test the next validator's real boundary inputs."""
import hashlib,importlib,json,os,sys
from pathlib import Path
D=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1')
sys.path.insert(0,str(D));m=importlib.import_module('audit_and_fit_append_only_v5')
FIT=D/'compact_fit_v6';V=D/'validation_preflight_v2';V.mkdir(exist_ok=True);m.AREA=V

def main():
 g=m.Guard();g.phase='actual_fit_exit0_and_small_real_boundary_validation'
 state=json.loads((FIT/'current_status.json').read_text());dead=[]
 for x in [{'PID':state['PID'],'create_time':state['create_time']}]+state['parent_identity']:
  try:same=abs(m.psutil.Process(x['PID']).create_time()-x['create_time'])<.01
  except m.psutil.NoSuchProcess:same=False
  assert not same;dead.append({**x,'same_identity_alive':False})
 closure={'kind':'Actual_owned_compact_BPE_fit_CLOSED0_v1','at_utc':m.now(),'owner':'/root/label_bpe_audit',
  'owned_session':10694,'actual_exit_code':0,'actual_tool_chunk':'982907','all_captured_identities_dead':dead,
  'actual_completed_fit':m.record(FIT/'completed_fit_handoff.json'),'actual_status':state,
  'exact_sealed256_prefix_replay':m.record(FIT/'actual_sealed256_prefix_replay.json'),'full_corpus_validation_pending':True}
 (FIT/'actual_owned_CLOSED0.json').write_text(json.dumps(closure,ensure_ascii=False,indent=2)+'\n')
 from tokenizers import Tokenizer
 old=Tokenizer.from_file(str(m.INCUMBENT));new=Tokenizer.from_file(str(FIT/'tokenizer.json'))
 examples=[]
 db=m.ro(m.BROAD)
 for ordinal in (1,835251,835252,2631115):
  row=db.execute('SELECT ordinal,unique_id,dataset,text,original_split FROM clips WHERE ordinal=?',(ordinal,)).fetchone()
  assert row;examples.append(['broad',*row])
 db.close()
 import struct
 offsets=m.PROJECT/'work/data/TR_expanded_WorldSpeech_index_v1/offsets.u64'
 with offsets.open('rb') as off,m.CLEAN.open('rb') as text:
  for ordinal in (1,66,163,312,72846,72847,352768):
   off.seek((ordinal-1)*8);at=struct.unpack('<Q',off.read(8))[0];text.seek(at);row=json.loads(text.readline())
   assert row['language']=='tr' and row['dataset_split'] in ('train','validated')
   examples.append(['cleaner',ordinal,row['source_id'],row['dataset'],row['text'],row['dataset_split']])
 checks=[]
 for pool,ordinal,sid,dataset,text,split in examples:
  a=old.encode(text);b=new.encode(text)
  assert old.decode(a.ids,skip_special_tokens=False)==new.decode(b.ids,skip_special_tokens=False)
  assert a.ids.count(0)==b.ids.count(0)
  checks.append({'pool':pool,'ordinal':ordinal,'source_id':sid,'split':split,'incumbent_tokens':len(a.ids),'candidate_tokens':len(b.ids)})
 source=D/'validate_and_package_corpus_BPE9223_v2.py'
 compile(source.read_text(encoding='utf8'),str(source),'exec')
 m.write('small_real_boundary_fixture.json',{'at_utc':m.now(),'passed':True,'actual_examples':checks,
  'fixture_does_not_claim_full_validation':True,'source':m.record(source)})
 plan={'kind':'Pinned_full_TRAIN_Rust_validation_and_package_phase_v2','at_utc':m.now(),
  'source':m.record(source),'guard_source':m.record(D/'audit_and_fit_append_only_v5.py'),
  'fit_actual_closure':m.record(FIT/'actual_owned_CLOSED0.json'),'fit_tokenizer':m.record(FIT/'tokenizer.json'),
  'source_label_audit':m.record(D/'all_TRAIN_extension_v5/source_label_audit.json'),
  'small_actual_boundary_fixture':m.record(V/'small_real_boundary_fixture.json'),
  'actual_all_TRAIN_text_validation_rows':2983883,'six_language_TRAIN_replay_rows':1914,
  'no_source_label_reaudit_or_word_frequency_refit':True,'batch_texts':256,
  'forecast_leaf_private_bytes':512000000,'maximum_leaf_private_bytes':1800000000,
  'maximum_actual_whole_family_private_bytes':2000000000,'CPU_thread_caps':1,
  'fresh_system90yield_95critical_2GBheadroom_E100GBreserve':True,
  'actual_and_logical_IO_limit_bytes':16000000000,'seconds_limit':3600,
  'no_GPU_models_active_source_labels_or_tokenizer_mutations':True}
 m.write('full_validation_pinned_plan.json',plan)
 g.phase='fit_closure_and_boundary_fixture_complete';g.publish(passed=True)
 print(json.dumps({'closure':m.record(FIT/'actual_owned_CLOSED0.json'),'plan':m.record(V/'full_validation_pinned_plan.json'),'actual_boundaries':len(checks)}))
if __name__=='__main__':main()
