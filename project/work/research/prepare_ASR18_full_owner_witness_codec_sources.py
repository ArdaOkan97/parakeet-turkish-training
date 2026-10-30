"""NEW fixed-dictionary lossless witness codec, same byte/horizon quotas."""
import ast,hashlib,json,zlib
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def emit(p,t):
    assert not p.exists();ast.parse(t);compile(t,str(p),'exec');p.write_text(t,encoding='utf-8',newline='\n');return pin(p)
def write(p,v):
    assert not p.exists();p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n');return pin(p)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
dictionary=(b'ASR18 owner native_checkpoint producer sha256 bytes path source configuration ownership state_revision catalog coherent_latest_pointer protected_owned_winners nonce checkpoint_mappings_released candidate_payload_handles_released checkpoint_write_inflight restore_in_progress actual_current_checkpoint_pointer native_release_ack resident_native_boundary current_exact_recovery publication_cadence_evidence incremental_authoritative_journal creation_identity writer_epoch source session_owner_nonce auxiliary_manifest_sha256 all_CPU_auxiliary_producers_actually_closed0 native_auxiliary_writer_contexts_returned all_auxiliary_handles_released '
    br'E:\\gpu_training_scratchpad\\analysis\\ASR18_memory_diagnosis_v1\\ '
    br'E:\\gpu_training_scratchpad\\runs\\ASR18_full_owner_validation_08_ '
    br'C:\\Users\\ardao\\Documents\\Codex\\2026-10-10\\i-x20\\outputs\\parakeet_turkish\\ ')
# Read-only sizing against the genuine retained complete proof. The prospective
# archive record is explicitly NOT an actual archive/publication claim.
proof_path=Path(r'E:\gpu_training_scratchpad\runs\ASR18_full_owner_validation_08_01\control\p000000001.proof.1.json')
original=proof_path.read_bytes();event={'kind':'ASR18_exact_archived_operation_witness_v1','original':pin(proof_path),
    'mutation':{'kind':'actual_auxiliary_archive_verified','archive':{
        'path':str(proof_path.parents[1]/'checkpoint_storage/auxiliary/p000000001.auxiliary.xz'),
        'bytes':2097152,'sha256':'f'*64}},'raw_UTF8':original.decode('utf-8')}
raw=canonical(event);enc=zlib.compressobj(level=9,zdict=dictionary);packed=enc.compress(raw)+enc.flush()
dec=zlib.decompressobj(zdict=dictionary);decoded=dec.decompress(packed,65537)
assert len(raw)<=65536 and len(packed)<=2048 and decoded==raw and dec.eof and not dec.unused_data and not dec.unconsumed_tail
assert json.loads(decoded)['raw_UTF8'].encode()==original
check=write(A/'full_owner_witness_static_dictionary_readonly_sizing_v1.json',{
    'kind':'ASR18_readonly_actual_retained_witness_full_byte_codec_sizing_v1','actual_proof':pin(proof_path),
    'prospective_archive_binding_is_not_actual_archive_proof':True,'dictionary_bytes':len(dictionary),
    'dictionary_sha256':hashlib.sha256(dictionary).hexdigest(),'raw_event_bytes':len(raw),
    'old_zlib9_packed_bytes':len(zlib.compress(raw,9)),'fixed_dictionary_packed_bytes':len(packed),
    'full_original_UTF8_and_event_roundtrip':True,'decoder_eof':True,'decoder_unconsumed_or_unused_data':False,
    'raw_cap_bytes':65536,'packed_cap_bytes':2048,'stream_cap_bytes':805306368,
    'actual_audio_model_GPU_jobs':0,'actual_original_unlink':False})
old=S/'asr18_full_run_parent_controller_v10.py';new=S/'asr18_full_run_parent_controller_v11.py'
text=old.read_text(encoding='utf-8-sig')
needle="ROOT=Path(r'E:\\gpu_training_scratchpad');OWNER='/root/memory_diagnosis'\n"
assert text.count(needle)==1;text=text.replace(needle,needle+'WITNESS_ZDICT='+repr(dictionary)+'\n')
before='        raw=canonical(event);packed=zlib.compress(raw,9)'
after='        raw=canonical(event);encoder=zlib.compressobj(level=9,zdict=WITNESS_ZDICT)\n        packed=encoder.compress(raw)+encoder.flush()'
assert text.count(before)==1;text=text.replace(before,after)
before='        decoded=zlib.decompress(actual[40:])'
after="""        decoder=zlib.decompressobj(zdict=WITNESS_ZDICT)
        decoded=decoder.decompress(actual[40:],65537)
        require(decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data,
                'Complete fixed source-dictionary witness EOF, no trailing/unconsumed compressed data')"""
assert text.count(before)==1;text=text.replace(before,after);controller=emit(new,text)
oldparent=R/'run_ASR18_full_run_owner_v12.py';parent=R/'run_ASR18_full_run_owner_v13.py'
parent_pin=emit(parent,oldparent.read_text(encoding='utf-8-sig').replace('from asr18_full_run_parent_controller_v10 import run','from asr18_full_run_parent_controller_v11 import run'))
oldbuilder=R/'prepare_ASR18_full_owner_configuration_v7.py';builder=R/'prepare_ASR18_full_owner_configuration_v8.py'
builder_pin=emit(builder,oldbuilder.read_text(encoding='utf-8-sig').replace('from asr18_full_run_parent_controller_v10 import FullOwner','from asr18_full_run_parent_controller_v11 import FullOwner'))
mapping={str(old):controller,str(oldparent):parent_pin,str(oldbuilder):builder_pin}
manifest=read(A/'full_owner_source_contract_v8.json');manifest['bindings']=[mapping.get(x['path'],x) for x in manifest['bindings']]
for oldpath,newpin in mapping.items():manifest['direct_sources'].pop(Path(oldpath).name,None);manifest['direct_sources'][Path(newpin['path']).name]=newpin
manifest_pin=write(A/'full_owner_source_contract_v9.json',manifest)
identity=read(A/'full_owner_validation_identity_v8.json');identity['full_run_source_contract']=manifest_pin
identity['persistent_prefix_sources'].update(metadata_resolver=controller,native_parent=parent_pin,recovery_parent=parent_pin)
identity_pin=write(A/'full_owner_validation_identity_v9.json',identity)
inputs=read(A/'full_owner_exact_ROOT_review_inputs_v9.json')
inputs.update(source_manifest=manifest_pin,identity=identity,identity_binding=identity_pin,run_identity_sha256=digest(identity),
    owner_verifier_source=controller,parent_source=parent_pin,metadata_builder_source=builder_pin,
    configuration_output=str(A/'full_owner_eight_update_configuration_04'),witness_readonly_codec_sizing=check)
inputs['lifecycle_callback_sources']={k:controller for k in inputs['lifecycle_callback_sources']}
inputs['proof_callback_sources'].update(verify_pending_window=controller,verify_writer_authority=controller,verify_durable_resume=controller)
inputs['DEV_execution_sources'].update(parent_source=parent_pin,guard_source=parent_pin)
inputs['DEV_native_boundary_sources'].update(boundary_verifier=parent_pin,native_parent=parent_pin,recovery_parent=parent_pin)
inputs['DEV_source_bindings'].pop('asr18_full_run_parent_controller_v10.py',None);inputs['DEV_source_bindings'][new.name]=controller
inputs_pin=write(A/'full_owner_exact_ROOT_review_inputs_v10.json',inputs)
print(json.dumps({'controller':controller,'parent':parent_pin,'builder':builder_pin,'source_manifest':manifest_pin,
    'identity':identity_pin,'run_identity_sha256':digest(identity),'ROOT_inputs':inputs_pin,'readonly_codec_check':check,
    'raw_bytes':len(raw),'old_packed':len(zlib.compress(raw,9)),'new_packed':len(packed)}))
