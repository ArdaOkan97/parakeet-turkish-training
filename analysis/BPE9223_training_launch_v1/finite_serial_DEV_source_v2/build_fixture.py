from pathlib import Path
O=Path(__file__).parent
text=(O.parent/'finite_serial_DEV_source_v1/fixture.py').read_text()
for old,new in [('asr18_finite_BPE9223_checkpoint_evidence_v1','asr18_finite_BPE9223_checkpoint_evidence_v2'),('asr18_finite_BPE9223_serial_DEV_v1','asr18_finite_BPE9223_serial_DEV_v2'),('finite_BPE9223_DEV_runtime_v1','finite_BPE9223_DEV_runtime_v2')]:text=text.replace(old,new)
text=text.replace('pre=e.metadata_fingerprint(metadata)','pre=json.loads(json.dumps(e.metadata_fingerprint(metadata)))')
text=text.replace("read=lambda v:o[v['path']]", "read=lambda v:json.loads(json.dumps(o[v['path']]))")
anchor="import asr18_full_run_serial_DEV_9223_v2 as protected"
extra='''# Real finalized native20 metadata and JSON PRE transport; NO tensor payload I/O.
actual_train=OUT.parent/'sustained20_CPU_native_actual_01/train'
actual_metadata=e.read_bound(record(actual_train/'after20.state.json'))
checkpoint_format_view(actual_metadata)
actual_PRE=e.normalize_fingerprint(e.read_bound(record(actual_train/'PRE_save_fingerprint.json')))
actual_fingerprint=e.metadata_fingerprint(actual_metadata)
assert actual_fingerprint['state_sha256']==digest(actual_fingerprint['state'])==actual_PRE['state_sha256']==digest(actual_PRE['state'])
assert actual_fingerprint!=actual_PRE, 'Real transport must exercise tuple/list Python inequality'
checks.append('actual native20 saved854/215 metadata canonical fingerprint matches real JSON PRE despite tuple/list transport')
bad=copy.deepcopy(actual_PRE);bad['state']['RNG']['python'][1][0]^=1;bad['state_sha256']=digest(bad['state'])
assert digest(actual_fingerprint['state'])!=bad['state_sha256']
checks.append('real JSON PRE RNG drift remains rejected by canonical state digest')
bad=copy.deepcopy(actual_PRE);key=next(iter(bad['state']['weights']));bad['state']['weights'][key]['sha256']='0'*64;bad['state_sha256']=digest(bad['state'])
assert digest(actual_fingerprint['state'])!=bad['state_sha256']
checks.append('real JSON PRE weight descriptor drift remains rejected')
'''
assert text.count(anchor)==1;text=text.replace(anchor,extra+'\n'+anchor)
text=text.replace('ASR18_finite_BPE9223_serialDEV_stdlib_source_controls_v1','ASR18_finite_BPE9223_serialDEV_JSON_transport_controls_v2')
text=text.replace("'synthetic_process_and_payload_I_O_only':True", "'synthetic_process_and_payload_I_O_only':True,'real_native20_metadata_and_JSON_PRE_only':True,'actual_PRE_state_sha256':actual_PRE['state_sha256']")
target=O/'fixture.py';assert not target.exists();target.write_text(text,encoding='utf-8')
print(str(target))
