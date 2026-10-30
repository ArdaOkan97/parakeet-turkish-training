"""Finite exact metadata fit using durable word counts; no corpus rescan."""
import argparse,gc,hashlib,importlib,json,os,sys,time
from pathlib import Path
D=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1')
sys.path.insert(0,str(D));m=importlib.import_module('audit_and_fit_append_only_v5')
from compact_pair_trainer_v1 import CompactTrainer
AREA=D/'compact_fit_v6';V5=D/'all_TRAIN_extension_v5'
m.AREA=AREA

def main():
 p=argparse.ArgumentParser();p.add_argument('--review',required=True);args=p.parse_args()
 g=m.Guard();g.phase='compact_fit_fresh_resource_admission'
 proposal=json.loads((AREA/'RAM_resource_proposal_v1.json').read_text())
 review=json.loads(Path(args.review).read_text())
 assert review['accepted'] is True
 assert review['proposal_sha256']==m.sha(AREA/'RAM_resource_proposal_v1.json')
 assert review['trainer_sha256']==m.sha(D/'compact_pair_trainer_v1.py')
 assert review['driver_sha256']==m.sha(__file__)
 assert proposal['forecast_leaf_private_bytes']<1800000000
 assert m.sha(m.INCUMBENT)==m.INCUMBENT_SHA
 shape=json.loads((AREA/'readonly_shape_preflight.json').read_text())
 assert m.stat(V5/'training_counts.sqlite')==shape['immutable_words_database']
 audit=json.loads((V5/'source_label_audit.json').read_text())
 assert sum(x['normalized_char_counts'].get('Û',0) for x in audit['groups'].values())==0
 assert m.sha(V5/'source_label_audit.json')==shape['source_audit']['sha256']
 assert m.sha(V5/'learned_merges_partial.json')==shape['sealed_merge_prefix']['sha256']
 sealed=json.loads((V5/'learned_merges_partial.json').read_text());assert len(sealed)==256
 baseline=g.process.memory_info().private;parents=sum(x.memory_info().private for x in g.parents)
 assert baseline<=proposal['runtime_baseline_cap_bytes']
 forecast=proposal['forecast_leaf_private_bytes']
 mem=m.memory();assert mem['commit_ratio']<.90 and mem['commit_headroom_bytes']-forecast>=2000000000
 # memory() uses total system commit; derive a fresh projected ratio from its
 # ratio/headroom pair rather than substituting physical available memory.
 commit_limit=mem['commit_headroom_bytes']/(1-mem['commit_ratio'])
 assert mem['commit_ratio']+(forecast-baseline)/commit_limit<.90
 assert parents+forecast<2000000000
 m.write('actual_preallocation_admission.json',{'at_utc':m.now(),'PID':os.getpid(),
  'review':m.record(args.review),'proposal':m.record(AREA/'RAM_resource_proposal_v1.json'),
  'actual_baseline_private':baseline,'actual_parent_private':parents,'memory':mem,
  'forecast_leaf_private_bytes':forecast,'forecast_whole_private_bytes':forecast+parents,
  'projected_system_commit_ratio':mem['commit_ratio']+(forecast-baseline)/commit_limit,'passed':True})
 base=json.loads(m.INCUMBENT.read_text());candidate=json.loads(json.dumps(base))
 assert 'Û' not in candidate['model']['vocab'] and len(candidate['model']['vocab'])==8711
 t=CompactTrainer(candidate,g,proposal['live_pair_key_cap'],proposal['cumulative_posting_cell_cap'],
  shape['word_types'],shape['initial_sequence_cells_N'])
 digest=hashlib.sha256();db=m.ro(V5/'training_counts.sqlite');g.phase='compact_original_word_index_build'
 start=time.monotonic()
 for wid,raw,freq in db.execute('SELECT id,seq,freq FROM words ORDER BY id'):
  seq=json.loads(raw);ids=[candidate['model']['vocab'][x] for x in seq]
  t.add_word(wid-1,ids,freq);digest.update((m.canonical([wid,raw,freq])+'\n').encode());g.pace(len(raw.encode())+24)
  if wid%100000==0:g.publish(indexed_words=wid,metrics=t.metrics())
 db.close()
 assert digest.hexdigest()==shape['immutable_word_rows_ordered_sha256']
 assert len(t.sequences)==shape['word_types'] and t.initial_cells==shape['initial_sequence_cells_N']
 assert sum(t.weights)==shape['weighted_word_occurrences']
 m.write('actual_index_build.json',{'at_utc':m.now(),'seconds':time.monotonic()-start,'word_rows_ordered_sha256':digest.hexdigest(),
  'metrics':t.metrics(),'private_bytes':g.process.memory_info().private})
 additions=[];g.phase='exact_sealed256_prefix_replay_then_remaining256_merges'
 for step in range(512):
  actual=t.step();additions.append(actual)
  if step<len(sealed):
   expected=dict(sealed[step])
   for key in ('id','left_id','right_id'):
    assert expected[key]!=8711
    if expected[key]>8711:expected[key]-=1
   assert actual==expected,('Sealed prefix differs',step,actual,expected)
  if step+1==256:m.write('actual_sealed256_prefix_replay.json',{'at_utc':m.now(),'passed':True,
   'exact_greedy_frequency_tie_piece_rank_affected_types_and_ID_projection':True,
   'original_prefix':m.record(V5/'learned_merges_partial.json'),'metrics':t.metrics(),
   'elapsed_seconds':time.monotonic()-g.started})
  if (step+1)%16==0:
   m.write('learned_merges_partial.json',additions);g.publish(learned_merges=step+1,last_piece=actual['piece'],metrics=t.metrics())
 assert len(candidate['model']['vocab'])==9223 and len(candidate['model']['merges'])==14500
 for piece,identifier in base['model']['vocab'].items():assert candidate['model']['vocab'][piece]==identifier
 assert candidate['model']['merges'][:13988]==base['model']['merges']
 for key in base:
  if key!='model':assert candidate[key]==base[key]
 assert all('Û' not in x['piece'] and x['weighted_pair_frequency']>0 for x in additions)
 m.write('tokenizer.json',candidate);m.write('added_pieces.json',additions)
 metrics=t.metrics();del t;gc.collect();g.check()
 m.write('completed_fit_handoff.json',{'kind':'Exact_all_TRAIN_compact_BPE9223_fit_v6','at_utc':m.now(),'passed':True,
  'PID':os.getpid(),'create_time':g.main_identity['create_time'],'merges':512,'vocabulary':9223,
  'immutable_words_ordered_digest':digest.hexdigest(),'source_audit':m.record(V5/'source_label_audit.json'),
  'all983TRAINtext_scope':'2,631,115 broad +352,768 cleaner TRAIN clip occurrences; counts reused exactly',
  'source_audit_not_repeated':True,'sealed256_replay_exact':True,'unsupported_manual_Ucircumflex_not_added':True,
  'metrics':metrics,'tokenizer':m.record(AREA/'tokenizer.json'),'added_pieces':m.record(AREA/'added_pieces.json'),
  'elapsed_seconds':time.monotonic()-g.started,'peak_private_bytes':g.peak,
  'actual_process_IO_bytes':g.process.io_counters().read_bytes+g.process.io_counters().write_bytes-g.initial_IO,
  'full_corpus_candidate_validation_still_required':True,'models_GPU_active_assets_changed':False})
 g.phase='fit512_complete_full_corpus_candidate_validation_pending';g.publish(passed=True)
 print(json.dumps({'passed':True,'handoff':m.record(AREA/'completed_fit_handoff.json'),'seconds':time.monotonic()-g.started}),flush=True)

if __name__=='__main__':main()
