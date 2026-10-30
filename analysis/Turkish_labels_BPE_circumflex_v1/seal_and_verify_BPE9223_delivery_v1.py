"""After observed worker exit0, independently verify the small delivered package."""
import hashlib,importlib,json,sys,zipfile
from pathlib import Path
D=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1')
sys.path.insert(0,str(D));m=importlib.import_module('audit_and_fit_append_only_v5')
OUT=D/'corpus_driven_BPE9223_v2';SEAL=D/'final_delivery_verification_v1';SEAL.mkdir(exist_ok=True);m.AREA=SEAL

def main():
 g=m.Guard();g.phase='actual_exit0_and_independent_small_package_verification'
 state=json.loads((OUT/'current_status.json').read_text());dead=[]
 for x in [{'PID':state['PID'],'create_time':state['create_time']}]+state['parent_identity']:
  try:same=abs(m.psutil.Process(x['PID']).create_time()-x['create_time'])<.01
  except m.psutil.NoSuchProcess:same=False
  assert not same;dead.append({**x,'same_identity_alive':False})
 handoff=json.loads((OUT/'completed_handoff.json').read_text())
 assert handoff['passed'] and handoff['vocabulary']==9223
 assert handoff['PID']==state['PID'] and handoff['create_time']==state['create_time']
 manifest=json.loads((OUT/'manifest.json').read_text())
 for filename,record in manifest['artifacts'].items():assert m.record(OUT/filename)==record,filename
 assert m.record(handoff['package']['path'])==handoff['package']
 with zipfile.ZipFile(handoff['package']['path']) as z:
  assert z.testzip() is None
  expected=[OUT.name+'/'+x for x in [*manifest['artifacts'],'manifest.json']]
  assert sorted(z.namelist())==sorted(expected)
  for name in z.namelist():
   local=OUT/name.split('/',1)[1]
   assert hashlib.sha256(z.read(name)).hexdigest()==m.sha(local)
 old=json.loads(m.INCUMBENT.read_text());new=json.loads((OUT/'tokenizer.json').read_text())
 assert len(old['model']['vocab'])==8711 and len(new['model']['vocab'])==9223
 for piece,identifier in old['model']['vocab'].items():assert new['model']['vocab'][piece]==identifier
 assert new['model']['merges'][:13988]==old['model']['merges']
 assert len(new['model']['merges'])==14500 and 'Û' not in new['model']['vocab']
 for key in old:
  if key!='model':assert old[key]==new[key]
 for key in old['model']:
  if key not in ('vocab','merges'):assert old['model'][key]==new['model'][key]
 additions=json.loads((OUT/'added_pieces.json').read_text())
 for index,x in enumerate(additions):
  assert x['id']==8711+index and x['merge_rank']==13988+index and x['weighted_pair_frequency']>0
  assert x['left_id']<x['id'] and x['right_id']<x['id']
  assert new['model']['vocab'][x['left']]==x['left_id'] and new['model']['vocab'][x['right']]==x['right_id']
  assert x['piece']==x['left']+x['right'] and new['model']['vocab'][x['piece']]==x['id']
 val=json.loads((OUT/'full_TRAIN_validation.json').read_text());assert val['passed'] and val['all_TRAIN_rows']==2983883
 groups=val['groups'];oldtotal=sum(x['incumbent_tokens'] for x in groups.values());newtotal=sum(x['candidate_tokens'] for x in groups.values())
 assert oldtotal==113443643 and sum(x['rows'] for x in groups.values())==2983883
 replay=val['six_language_original1914_TRAIN_replay'];assert sum(x['rows'] for x in replay.values())==1914
 audit=json.loads((D/'all_TRAIN_extension_v5/source_label_audit.json').read_text())
 chars={c:sum(x['source_char_counts'].get(c,0) for x in audit['groups'].values()) for c in m.CHARS}
 normalized={c:sum(x['normalized_char_counts'].get(c,0) for x in audit['groups'].values()) for c in m.CHARS}
 assert chars==normalized and chars['Û']==0
 closure={'kind':'Actual_final_BPE_validation_package_CLOSED0_v1','at_utc':m.now(),'owner':'/root/label_bpe_audit',
  'owned_session':43339,'actual_exit_code':0,'actual_tool_chunk':ACTUAL_TOOL_CHUNK,
  'all_captured_identities_dead':dead,'worker_completed_handoff':m.record(OUT/'completed_handoff.json'),
  'full_validation':m.record(OUT/'full_TRAIN_validation.json'),'actual_status':state,'no_active_model_tokenizer_labels_changed':True}
 (OUT/'actual_owned_CLOSED0.json').write_text(json.dumps(closure,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 summary={'kind':'Independent_small_BPE9223_package_verification_v1','at_utc':m.now(),'passed':True,
  'all_zip_members_CRC_SHA_and_disk_manifest_exact':True,'all_old8711_IDs_13988_merges_settings_specials_exact':True,
  '512_positive_count_acyclic_append_merges_exact':True,'manual_unobserved_Ucircumflex_absent':True,
  'actual_all_TRAIN_Rust_validation_rows':2983883,'original_tokens':oldtotal,'candidate_tokens':newtotal,
  'token_reduction_fraction':1-newtotal/oldtotal,'source_char_counts':chars,'normalized_char_counts':normalized,
  'existing_unknown_TRAIN_rows':sum(x['existing_unknown_rows'] for x in audit['groups'].values()),
  'candidate_unknown_TRAIN_rows':sum(x['unknown_rows'] for x in groups.values()),
  'six_language1914_replay_encoding_changes':sum(x['encoding_changes'] for x in replay.values()),
  'six_language1914_replay_decode_changes':sum(x['decode_changes'] for x in replay.values()),
  'package':handoff['package'],'candidate':handoff['candidate'],'actual_closure':m.record(OUT/'actual_owned_CLOSED0.json')}
 m.write('verified_delivery_summary.json',summary)
 rows=['Dataset | Clips | â | î | û | Â | Î | Û | ğ | Ğ','---|---:|---:|---:|---:|---:|---:|---:|---:|---:']
 for key,x in audit['groups'].items():rows.append(key+' | '+str(x['rows'])+' | '+' | '.join(str(x['source_char_counts'].get(c,0)) for c in m.CHARS))
 report=('Turkish label audit and corpus-driven BPE candidate\n\n'
  'All 2,983,883 actual source training labels match their existing8711 tokenizer IDs/text hashes. '
  'Raw and normalized character counts below are identical; no audited diacritic/case loss was found. '
  'The historical field rows_with_circumflex includes all eight audit characters âîûÂÎÛğĞ; it is not a circumflex-only count.\n\n'+
  '\n'.join(rows)+'\n\n'
  'The model already has â8024/î8001/û8113/Â8180/Î8087/ğ8193/Ğ8194. '
  'Observed uppercase exposure is sparse:Â104/Î1/Ğ84 characters;Û0. Vocabulary slots do not prove current positive training exposure. '
  'No synthetic capitalization or transcript correction was added. TSC has no source circumflex spellings, rather than a pipeline stripping them.\n\n'
  'The finalized9223 candidate preserves all8711 original IDs/13988 merge ranks and adds512 frequency-selected Latin merges using every Turkish broad+clean TRAIN clip occurrence. '
  'Duplicate texts/source overlaps remain counted occurrences. New merges are not trained model weights or an ASR quality gain. '
  'No manually forcedÛ was retained. The optional8712 coverage patch remains separate and unadopted.\n\n'
  f'Actual Rust full-text validation: {oldtotal:,} → {newtotal:,} tokens ({100*(1-newtotal/oldtotal):.3f}% fewer). '
  'Decoded text and existing unknown counts stayed equal on every row; exact unused-atom-shadow ID projection passed. '
  f'Existing unknowns remain in {summary["existing_unknown_TRAIN_rows"]:,} TRAIN rows. Inherited byte_fallback=true has zero actual byte tokens; this is not complete Unicode coverage.\n\n'
  f'Protected six-language TRAIN replay comparison: {summary["six_language1914_replay_encoding_changes"]} of1914 encodings changed; '
  f'{summary["six_language1914_replay_decode_changes"]} decoded texts changed. Actual protected cached labels were not rewritten. '
  'Stable token IDs do not promise unchanged segmentation or retention. See full_TRAIN_validation.json for each language.\n\n'
  'Before adoption: explicit model embedding/output initialization for512 new IDs, optimizer extension preserving mature moments/clocks/RNG, '
  '5 TDT duration-row relocation8711..8715→9223..9227, new TRAIN labels/means/geometry and full six-language retention checks. '
  'Current8711 training/model/config/labels remain unchanged. RustJSON loading is tested; HF/TDT model loading is unverified.\n')
 (SEAL/'audit_and_BPE_report.txt').write_text(report,encoding='utf8')
 delivery={**handoff,'kind':'Final_BPE9223_verified_delivery_handoff_v3','at_utc':m.now(),
  'actual_validation_closure':m.record(OUT/'actual_owned_CLOSED0.json'),
  'independent_package_verification':m.record(SEAL/'verified_delivery_summary.json'),
  'readable_audit_report':m.record(SEAL/'audit_and_BPE_report.txt'),'actual_worker_closed0':True}
 (D/'finalized_delivery_handoff_v3.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 pointer={'kind':'Current_label_BPE_final_delivery_pointer_v3','at_utc':m.now(),'owner':'/root/label_bpe_audit',
  'status':'complete_finalized_tokenizer_candidate_not_adopted','delivery_handoff':m.record(D/'finalized_delivery_handoff_v3.json'),
  'candidate':handoff['candidate'],'package':handoff['package'],'actual_validation_closure':delivery['actual_validation_closure'],
  'no_GPU_or_active_tokenizer_model_source_label_mutations':True}
 for filename in ('current_dispatch.json','current_status.json','latest_finalized_corpus_BPE_handoff.json'):
  (D/filename).write_text(json.dumps(pointer,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 g.phase='final_delivery_verified';g.publish(passed=True)
 print(json.dumps({'handoff':m.record(D/'finalized_delivery_handoff_v3.json'),'summary':summary},ensure_ascii=False),flush=True)

if __name__=='__main__':
 assert len(sys.argv)==2
 ACTUAL_TOOL_CHUNK=sys.argv[1]
 main()
