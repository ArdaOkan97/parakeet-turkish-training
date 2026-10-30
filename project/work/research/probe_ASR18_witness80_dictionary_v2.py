"""One fixed generic-field dictionary candidate; no search or RUN mutation."""
import ast,hashlib,json,os,struct,sys,time,zlib
from pathlib import Path
P=Path(__file__).resolve().parents[2];sys.path[:0]=[str(P/'outputs/parakeet_turkish'),str(P/'work/research')]
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import canonical
from atomic_io_retry_v1 import read_json,write_json
from fastmss_owned_process_closure_v2 import identity
import run_ASR18_BPE9223_full_run_owner_v2 as parent
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
RUN=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_weak2_clean1_20261018_02')
GENERIC_FIELDS=('merged_CPU_live_task_boundary','merged_parent_process_exit0_claimed',
 'metadata_task_completed_no_CPU_inflight','every_preserved_control_attempt_closed0_claimed',
 'legacy_all_CPU_auxiliary_producers_actually_closed0_scope','actual_numerical_reference_frontend_and_external_cleanup_only',
 'producer_completion_scopes','actual_owner','actual_current_checkpoint_pointer','active_auxiliary_workspaces',
 'committed_window_updates','retained_metadata_bytes','auxiliary_archives','auxiliary_bytes','archive_bytes',
 'live_payload_bytes','live_pairs','exact_pass_tail','file_index','history','last_frame','accounting',
 'checkpoint_mappings_released','candidate_payload_handles_released','checkpoint_write_inflight',
 'restore_in_progress','native_release_ack','native_work_quiescent','resident_native_boundary',
 'current_exact_recovery','publication_cadence_evidence','incremental_authoritative_journal',
 'writer_source','writer_epoch','sole_current_writer','prior_bytes_never_rewritten_or_truncated',
 'producer_PID','producer_create_time','session_owner_nonce','all_CPU_auxiliary_producers_actually_closed0',
 'native_auxiliary_writer_contexts_returned','all_auxiliary_handles_released','auxiliary_manifest_sha256')

def dictionaries():
 tree=ast.parse((P/'outputs/parakeet_turkish/asr18_BPE9223_full_run_controller_v18.py').read_text(encoding='utf8'))
 old=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='WITNESS_ZDICT'for t in n.targets))
 # Fixed public schema terms only. No event-derived hashes, nonces, paths or values.
 terms=' '.join(GENERIC_FIELDS)
 normal=' '.join(json.dumps(k)+':' for k in GENERIC_FIELDS)
 escaped=' '.join(json.dumps(json.dumps(k)+':')[1:-1] for k in GENERIC_FIELDS)
 return old,old+b' '+(terms+' '+normal+' '+escaped).encode('ascii')

def main():
 began=time.monotonic();parent.storage()
 require(sys.flags.utf8_mode==1 and os.environ.get('PYTHONUTF8')=='1' and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1','Upfront CPU-only UTF8')
 require(all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'Four numerical thread caps1')
 folder=N/'witness80_dictionary_v2_actual_01';require(not folder.exists(),'Fresh bounded E output');folder.mkdir()
 guard=parent.ParentGuard(folder);exitcode=1
 try:
  guard.check();require(parent.fresh_admit(128000000)['passed'],'Prospective same128M admission')
  old,new=dictionaries();require(len(new)<=32768 and new.startswith(old),'One fixed bounded extension')
  failure_path=N/'full_training_operational_resource_actual_12/actual_committed80_witness_frame_archive_failure_TOOL1_v1.json'
  failure=read_json(failure_path);failure_pin=record(failure_path)
  pin=record(RUN/'control/p000000004.proof.11.json');original=Path(pin['path']).read_bytes()
  archive=record(RUN/'checkpoints/auxiliary/p000000004.auxiliary.xz')
  event={'kind':'ASR18_exact_archived_operation_witness_v1','original':pin,
      'mutation':{'kind':'actual_auxiliary_archive_verified','archive':archive},'raw_UTF8':original.decode('utf8')}
  raw=canonical(event);enc=zlib.compressobj(level=9,zdict=new);packed=enc.compress(raw)+enc.flush()
  dec=zlib.decompressobj(zdict=new);decoded=dec.decompress(packed,65537)
  require(decoded==raw and dec.eof and not dec.unused_data and not dec.unconsumed_tail,'Exact same event, strict EOF')
  require(json.loads(decoded)==event and json.loads(decoded)['raw_UTF8'].encode('utf8')==original,'Every event field and original UTF8 byte preserved')
  frames=0
  with (RUN/'control/storage_operation_witnesses.bin').open('rb')as stream:
   while header:=stream.read(40):
    guard.check();require(len(header)==40,'Complete immutable legacy header');size,raw_size=struct.unpack('<II',header[:8])
    payload=stream.read(size);d=zlib.decompressobj(zdict=old);r=d.decompress(payload,65537)
    require(size<=2048 and raw_size<=65536 and len(r)==raw_size and hashlib.sha256(r).digest()==header[8:]
      and d.eof and not d.unused_data and not d.unconsumed_tail,'Existing legacy frames preserved')
    json.loads(r);frames+=1
  guard.check();require(frames==10 and record(pin['path'])==pin and record(archive['path'])==archive and record(failure_path)==failure_pin,'No original archive/witness/failure mutation')
  require(time.monotonic()-began<60,'One finite candidate only')
  (folder/'candidate.zlib').write_bytes(packed);(folder/'dictionary_v2.bin').write_bytes(new)
  write_json(folder/'report.json',{'kind':'ASR18_one_fixed_generic_field_witness_dictionary_v2_measurement_v1','passed':True,
   'producer':identity(),'source':record(__file__),'actual_failure':failure_pin,'original_witness':pin,'archive':archive,
   'candidates_tested':1,'raw_event_bytes':len(raw),'packed_bytes':len(packed),'fits_unchanged_caps':len(raw)<=65536 and len(packed)<=2048,
   'raw_cap_bytes':65536,'packed_cap_bytes':2048,'stream_cap_bytes':805306368,'all10_existing_legacy_frames_verified':True,
   'dictionary_v1_adler32':zlib.adler32(old),'dictionary_v2_adler32':zlib.adler32(new),
   'dictionary_v2':record(folder/'dictionary_v2.bin'),'payload':record(folder/'candidate.zlib'),
   'dictionary_contains_only_legacy_bytes_and_fixed_generic_schema_terms':True,'full_original_and_event_roundtrip':True,
   'RUN_mutation':False,'GPU_audio_model_jobs':0,'wall_seconds':time.monotonic()-began})
  exitcode=0;print(json.dumps({'report':record(folder/'report.json'),'packed_bytes':len(packed),'fit':len(packed)<=2048}),flush=True)
 finally:guard.close()
 if exitcode:raise SystemExit(exitcode)

if __name__=='__main__':main()
