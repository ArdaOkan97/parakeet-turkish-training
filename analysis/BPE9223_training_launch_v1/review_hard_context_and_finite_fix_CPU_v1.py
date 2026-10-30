"""Narrow hard9223 context and finite PRE-JSON fix review; no payload/audio/model."""
import ast,copy,difflib,importlib,json,sys,types
from pathlib import Path
L=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1');sys.path.insert(0,str(L));m=importlib.import_module('audit_and_fit_append_only_v5')
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1');O=N/'independent_hard9223_and_finite2_review_v1';O.mkdir(exist_ok=True);m.AREA=O
S=m.PROJECT/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
def rec(path):return {k:v for k,v in m.record(path).items() if k in ('path','bytes','sha256')}
def main():
 g=m.Guard();g.phase='hard_context_and_canonical_PRE_fix_metadata'
 hard=S/'asr18_BPE9223_hard_envelope_context_v1.py';old=S/'asr18_hard_envelope_schedule_context_v1.py'
 assert m.sha(hard)=='7968517267a14e8c88d297374593bf36266fc8613c4b7da18cd8e72bb83b90ed'
 assert m.sha(old)=='24cb88c113800c5eede05863d51eaf33ace7a57c4fdcd3e1dd38a4e6c7f60b59'
 f1=S/'asr18_finite_BPE9223_checkpoint_evidence_v1.py';f2=S/'asr18_finite_BPE9223_checkpoint_evidence_v2.py'
 assert m.sha(f1)=='f48386a080f3fa23a8e160735391b80c5c0477d48f69bb5211c3b75b36c9804f'
 assert m.sha(f2)=='a8898deb712a24cff66cdb00f4a849c6fc1956f21a5eb52a6bb1f40bdca3879b'
 original="    require(same(metadata_fingerprint(metadata),pre), 'All854 saved descriptors differ from independent actual PRE hashes')"
 changed="    actual_pre = metadata_fingerprint(metadata)\n    require(actual_pre['state_sha256'] == digest(actual_pre['state'])\n        == pre['state_sha256'] == digest(pre['state']),\n        'All854 saved descriptors differ from independent actual PRE canonical hashes')"
 assert f1.read_text().replace(original,changed)==f2.read_text()
 for a,b,pin in [('asr18_finite_BPE9223_serial_DEV_v1.py','asr18_finite_BPE9223_serial_DEV_v2.py','06e16d72d87ffa559d67715b2545c7151dbd27a1ae72b42617c0506e6b4a6d55'),('finite_BPE9223_DEV_runtime_v1.py','finite_BPE9223_DEV_runtime_v2.py','40586ba1a51afd924f33e909c5d255e2235ce982c79626a135b2b9ca2c955e6d')]:
  assert m.sha(S/b)==pin
  value=(S/a).read_text()
  for oldname,newname in [('asr18_finite_BPE9223_serial_DEV_v1','asr18_finite_BPE9223_serial_DEV_v2'),('finite_BPE9223_DEV_runtime_v1','finite_BPE9223_DEV_runtime_v2'),('asr18_finite_BPE9223_checkpoint_evidence_v1','asr18_finite_BPE9223_checkpoint_evidence_v2')]:value=value.replace(oldname,newname)
  assert ast.dump(ast.parse(value))==ast.dump(ast.parse((S/b).read_text()))
 import asr18_finite_BPE9223_checkpoint_evidence_v1 as oldhelper
 import asr18_finite_BPE9223_checkpoint_evidence_v2 as helper
 from asr18_full_run_serial_DEV_9223_v2 import checkpoint_format_view
 run=N/'sustained20_CPU_native_actual_01';summary=json.loads((run/'actual_sustained20_summary_v1.json').read_text())
 native=json.loads(Path(summary['native_report']['path']).read_text());assert rec(summary['native_report']['path'])==summary['native_report']
 metadata=json.loads(Path(native['checkpoint']['path']).read_text());assert rec(native['checkpoint']['path'])==native['checkpoint']
 pre=helper.normalize_fingerprint(json.loads(Path(native['PRE_save_fingerprint']['path']).read_text()));assert rec(native['PRE_save_fingerprint']['path'])==native['PRE_save_fingerprint']
 checkpoint_format_view(metadata);actual=helper.metadata_fingerprint(metadata)
 assert actual['state_sha256']==helper.digest(actual['state'])==pre['state_sha256']==helper.digest(pre['state'])
 assert not oldhelper.same(actual,pre)
 scalar=lambda *p:helper.decode_scalar_tree(helper.select(metadata['tree'],*p))
 assert type(scalar('optimizer','appended512_Adam_clock')) is int and scalar('optimizer','appended512_Adam_clock')==20
 assert scalar('optimizer','clocks')==[64270,2020,62270] and len(metadata['tensors'])==854
 assert len(actual['state']['weights'])==207 and sum(len(p['state']) for p in actual['state']['Adam_partitions'])==215
 assert set(actual['state']['RNG'])=={'python','numpy','torch','CUDA'}
 drift=copy.deepcopy(metadata);drift['tensors'][0]['sha256']='0'*64
 assert helper.metadata_fingerprint(drift)['state_sha256']!=pre['state_sha256']
 delta=''.join(difflib.unified_diff(old.read_text().splitlines(True),hard.read_text().splitlines(True),fromfile=old.name,tofile=hard.name));(O/'exact_hard_context_delta.diff').write_text(delta)
 # Independent single-row metadata lookup only. No audio, refit or database scan.
 evidence=json.loads((N/'BPE9223_final_verified_TRAIN_population_evidence_v1.json').read_text())
 db=m.ro(evidence['declared_geometry_index']['path']);geometry=db.execute('SELECT native_sr,native_samples,clean16k_samples,mel_frames,encoder_frames,target_tokens FROM geometry WHERE ordinal=1051133').fetchone();db.close()
 db=m.ro(evidence['broad_labels']['path']);label=db.execute('SELECT target_tokens,length(target_ids_u32) FROM labels WHERE ordinal=1051133').fetchone();db.close()
 assert geometry[2:]==(2547840,15924,1991,517) and label==(517,2068)
 tree=ast.parse(hard.read_text());definitions=[copy.deepcopy(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
 ordinals=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='ORDINALS' for x in n.targets))
 constants={k:next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id==k for x in n.targets)) for k in ('KIND','CURSOR','CONTEXT')}
 require=helper.require;digest=helper.digest;cases=[]
 def rejected(name,operation):
  try:operation()
  except (ValueError,KeyError) as error:cases.append({'name':name,'rejected':True,'reason':str(error)})
  else:raise AssertionError('Invalid hard context fixture accepted '+name)
 initial={'carried_replay':{'updates':0},'broad_cursor':{'draws':0},'cleaner_cursor':{'draws':0}}
 rows={o:{'source_id':'fixture:'+str(o),'encoder_frames':1,'target_tokens':1,'tr_metadata_labels':[1],'declared_geometry':{'clean16k_samples':1280}} for o in ordinals}
 rows[1051133]={'source_id':'kiraat:src01792-00990','encoder_frames':1991,'target_tokens':517,'tr_metadata_labels':[1]*517,'declared_geometry':{'clean16k_samples':2547840}}
 class OriginalContext:
  def __init__(self,arguments,guard):
   self.snapshot={'turkish_rng':'source_snapshot_not_a_reset'};self.identity={'fixture_population9223':True};self.resolver=types.SimpleNamespace(row=lambda ordinal:copy.deepcopy(rows[ordinal]))
  def new_schedule(self):return types.SimpleNamespace(state_dict=lambda:copy.deepcopy(initial))
 prior={'finite_recovered_checkpoint':{'path':'IN_MEMORY_ONLY.packet'},'first_plan':{'path':'IN_MEMORY_ONLY.plan','sha256':'143e0807673daa69796409acc45d6cd03e78ddf73ecc3f672d13b50df4e90591'}}
 config={'plan':prior['first_plan']};packet={'configuration':{'path':'IN_MEMORY_ONLY.config'},'optimizer_updates':20,'diagnostic_only':True,'production_enabled':False}
 plan={'initial_committed_updates':0,'updates':[{'TR_draws':[{'ordinal':o} for o in ordinals],'rows':[copy.deepcopy(rows[o]) for o in ordinals]} for _ in range(20)]}
 def read_bound(pin,guard):return copy.deepcopy(config if pin['path'].endswith('.config') else plan)
 scope={'deepcopy':copy.deepcopy,'require':require,'digest':digest,'ORDINALS':ordinals,**constants,'WEAK':'weak_label_continued_training','OriginalContext':OriginalContext,
  'checked_binding':lambda pin:(copy.deepcopy(packet),None,None),'read_bound':read_bound}
 exec(compile(ast.Module(body=definitions,type_ignores=[]),str(hard),'exec'),scope)
 assert scope['verified_prior_rows'](prior,g.check)==plan['updates'][0]['rows'][:13]
 context=scope['HardEnvelopeContext']({'DATA_evidence':{'BPE9223_population':{'fixture':True}}},prior,guard=g.check)
 assert context.original_state==initial and initial['broad_cursor']['draws']==0 and context.rows[1051133]['target_tokens']==517
 assert context.cursor_identity['production_source_eligible'] is False and context.cursor_identity['hard_original_ordinal']==1051133
 for field,bad in (('optimizer_updates',19),('diagnostic_only',False),('production_enabled',True)):
  oldvalue=packet[field];packet[field]=bad;rejected('prior_'+field,lambda:scope['verified_prior_rows'](prior,g.check));packet[field]=oldvalue
 bad=copy.deepcopy(prior);bad['first_plan']['sha256']='0'*64;rejected('wrong_first_plan_pin',lambda:scope['verified_prior_rows'](bad,g.check))
 for field,bad in (('encoder_frames',1990),('target_tokens',516),('source_id','wrong')):
  oldvalue=rows[1051133][field];rows[1051133][field]=bad;rejected('hard_'+field,lambda:scope['HardEnvelopeContext']({},prior,guard=g.check));rows[1051133][field]=oldvalue
 initial['broad_cursor']['draws']=14;rejected('nonzero_source_cursor',lambda:scope['HardEnvelopeContext']({},prior,guard=g.check));initial['broad_cursor']['draws']=0
 cursor=scope['ExplicitHardCursor']({'fixture':'source_zero'},'derived_identity_not_global_RNG');start=cursor.state_dict();draws=cursor.preview(14)
 rejected('out_of_range_diagnostic_draw',lambda:cursor.preview(15));rejected('reordered_diagnostic_commit',lambda:cursor.commit(list(reversed(draws))))
 bad=copy.deepcopy(start);bad['draws']=7;rejected('partial_diagnostic_resume',lambda:cursor.load_state_dict(bad))
 cursor.commit(draws);assert cursor.remaining==0 and cursor.draws==14 and initial['broad_cursor']['draws']==0
 assert not any(k in sys.modules for k in ('torch','numpy','transformers','nemo'));g.check()
 report={'kind':'ASR18_independent_hard9223_context_and_finite_PRE_fix_source_review_v1','passed':True,'reviewer':'/root/label_bpe_audit',
  'scope':'source/API/metadata; not actual hard native/CPU/audio/admission','hard_source':rec(hard),'old_hard_source':rec(old),'exact_delta':rec(O/'exact_hard_context_delta.diff'),
  'finite_helper2':rec(f2),'serial2':rec(S/'asr18_finite_BPE9223_serial_DEV_v2.py'),'runtime2':rec(S/'finite_BPE9223_DEV_runtime_v2.py'),
  'helper_fix_only_canonical_PRE_comparison':True,'serial_runtime_import_only_AST':True,
  'actual_after20_metadata':native['checkpoint'],'actual_PRE_fingerprint':native['PRE_save_fingerprint'],'actual_saved854_207weights_215Adam_all4RNG_canonical_equals_PRE':True,
  'original_Python_equality_gap_reproduced_and_fixed':True,'changed_descriptor_still_changes_hash':True,'actual_payload_read_here':False,
  'actual_hard_ordinal':1051133,'single_row_geometry_native_rate_samples_cleanN_mel_T_U':list(geometry),'single_row_target_U_and_BLOB_bytes':list(label),
  'hard_context_first13_provenance_only_not_checkpoint_adoption':True,'original9223_schedule_zero_and_full_new_population_binding':True,
  'explicit_hard_cursor_never_advances_original_source_cursor':True,'protected_carried_replay_code_unchanged':True,'diagnostic_not_fullproduction_permutation_or_seed':True,
  'negative_metadata_controls':cases,'negative_control_count':len(cases),'mocked_context_IO_not_real_schedule_or_audio_proof':True,
  'new_concrete_source_blockers':[],'production_admission_here':False,'actual_model_audio_payload_GPU_jobs':0}
 (O/'source_findings_v1.json').write_text(json.dumps(report,indent=2)+'\n');g.publish(completed=True,findings=rec(O/'source_findings_v1.json'))
 print(json.dumps({'passed':True,'findings':rec(O/'source_findings_v1.json'),'negative_controls':len(cases)}))
if __name__=='__main__':main()
