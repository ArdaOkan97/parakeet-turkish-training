"""Separate exact observations; failed cross-genesis all4 gate stays false.

Independent original-source bootstraps do not set NumPy RNG. This report never
removes NumPy from the old all4 oracle or manufactures a shared initial state.
It verifies own saved4 -> actual startup restore -> saved8 preservation, and
reports independent-oracle tensor/scientific equality as separate assertions.
"""
from pathlib import Path
from contextlib import contextmanager
import argparse,copy,hashlib,json,sys
P=Path(__file__).resolve().parents[2];sys.path[:0]=[str(P/'outputs/parakeet_turkish'),str(P/'work/research')]
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from fastmss_phase_aware_plan_bridge_v7 import read_bound
from fastmss_row_serial_cpu_v8 import environment,resource_guard
from fastmss_owned_process_closure_v2 import verify,dead
from asr18_streamed_persistence_fingerprint_v1 import normalize_fingerprint
from bpe_indexed_owned_checkpoint_storage_v3 import IndexedOwnedFutureCheckpointStore
from asr18_full_run_metadata_resolver_v1 import MetadataResolver
from bpe_streaming_checkpoint_semantics_v2 import decode_scalar_tree,select
from verify_ASR18_normal_resume_state_equivalence_v1 import journal,IGNORED_JOURNAL_EXECUTION_FIELDS
ROOT=Path(r'E:\gpu_training_scratchpad')

@contextmanager
def metadata_phase(name):
    require(name=='read_index','Only frozen read/index member resolution');resource_guard()
    try:yield
    finally:resource_guard()

def run(oracle_pin,candidate_pin,observed_pin,pause_pin,diagnostic_pin,output):
    environment();guard=resource_guard
    oracle=read_bound(oracle_pin,guard);candidate=read_bound(candidate_pin,guard)
    actual=read_bound(observed_pin,guard);pause=read_bound(pause_pin,guard)
    require(oracle['passed'] and candidate['passed'] and actual['actual_tool_exit_code']==0
        and actual['report']==candidate_pin and dead(candidate['producer'])
        and oracle['actual_optimizer_updates']==candidate['actual_optimizer_updates']==8
        and candidate['actual_owner_invocation_count']==2,'Actual successful split8 and owner0, not comparison success inference')
    for key in ('current_native_process0','current_recovery_process0'):
        verify(read_bound(candidate[key],guard),kind='ASR18_owned_GPU_process_completion_v1',owner='/root/memory_diagnosis',guard=guard)
    require(pause['actual_optimizer_updates']==4 and pause['actual_owner_invocation_count']==1,'Actual exact first4 pause authority')
    store=IndexedOwnedFutureCheckpointStore(pause['storage_owner'],guard=guard)
    try:
        resolver=MetadataResolver(store,control_root=Path(pause['namespace'])/'control',guard=guard,metadata_phase=metadata_phase)
        original=pause['current_capture']['pre_save_fingerprint'];require(not Path(original['path']).exists(),'First4 PRE genuinely archived')
        raw=bytearray()
        with resolver(original) as stream:
            while block:=stream.read(1024**2):
                guard();raw.extend(block);require(len(raw)<=16*1024**2,'Bounded exact selected original PRE')
        require(len(raw)==original['bytes'] and hashlib.sha256(raw).hexdigest()==original['sha256'],'Full selected original SHA/bytes/EOF')
        first=normalize_fingerprint(json.loads(raw));require(first['state']['updates']==4,'Actual archived first4 fingerprint')
    finally:store.close()
    restore_pin=record(Path(candidate['current_native_process0']['path']).parent/'actual_parent_restore.json')
    restore=read_bound(restore_pin,guard)
    require(restore['passed'] and restore['pointer']==pause['current_capture']['pointer']
        and restore['all207_threeAdam_all4RNG_match_actual_pre_save'] is True
        and restore['next_draw_packet_match'] is True and restore['restore_occurred_only_at_startup_before_native'] is True,
        'Actual native startup exact own all4/207/threeAdam and nextdraw restore BEFORE native5')
    states=[];tables=[];samplers=[];packets=[];journals=[];pointers=[]
    for report in (oracle,candidate):
        capture=report['current_checkpoint'];pointer=read_bound(capture['pointer'],guard);pointers.append(pointer)
        pre=normalize_fingerprint(read_bound(capture['pre_save_fingerprint'],guard));states.append(pre['state'])
        metadata=read_bound(pointer['checkpoint'],guard);require(len(metadata['tensors'])==845,'Full actual tensor table')
        tables.append(metadata['tensors']);samplers.append(decode_scalar_tree(select(metadata['tree'],'sampler')))
        packets.append(read_bound(capture['next_draw_packet'],guard));journals.append(journal(pointer['journal_prefix'],guard))
    exact={}
    for field in states[0]:
        if field in ('identity','RNG'):continue
        require(states[0][field]==states[1][field],'Independent oracle scientific mismatch: '+field)
        exact[field]=digest(states[0][field])
    rng={key:{'oracle_sha256':digest(states[0]['RNG'][key]),'candidate_sha256':digest(states[1]['RNG'][key]),
        'exact':states[0]['RNG'][key]==states[1]['RNG'][key]} for key in states[0]['RNG']}
    require(all(rng[key]['exact'] for key in ('python','torch','CUDA')) and rng['numpy']['exact'] is False,
        'Preserve actual independent-genesis NumPy mismatch; all other RNG fields exact')
    require(digest(first['state']['RNG']['numpy'])==digest(states[1]['RNG']['numpy']),
        'Own actual saved NumPy4 and final8 exact, no seed/reset/NumPy exclusion')
    for name,values in (('all845_dtype_shape_offsets_bytes_SHA',tables),('full_sampler',samplers),('full_nextdraw',packets),('scientific_journal',journals)):
        require(values[0]==values[1],'Independent exact field mismatch: '+name);exact[name]=digest(values[0])
    require(pointers[0]['payload']['sha256']==pointers[1]['payload']['sha256'] and pointers[0]['payload']['bytes']==pointers[1]['payload']['bytes'],
        'Actual whole tensor payloadSHA/bytes exact')
    archives=candidate['actual_obsolete_worker_control_archives']
    require(len(archives)==2 and all(x['passed'] and x['all_original_member_bytes_verified_before_exact_unlink']
        and not Path(x['original_folder']).exists() for x in archives),'Actual source-owned obsolete archive+lookup, protected current proofs retained')
    require(output.is_relative_to(ROOT) and not output.exists(),'Fresh separate readonly observations')
    write_json(output,{'kind':'ASR18_actual_normal_resume_separate_exact_field_observations_v1','observations_verified':True,
        'source':record(__file__),'oracle_report':oracle_pin,'candidate_report':candidate_pin,'actual_owner_TOOL0':observed_pin,
        'normal_pause_snapshot':pause_pin,'archived_first4_PRE':original,'actual_startup_restore':restore_pin,
        'failed_cross_genesis_all4_comparator_log':diagnostic_pin,'cross_genesis_full_all4_oracle_passed':False,
        'cross_genesis_failure_not_waived_or_relabelled':True,'independent_rng_comparison':rng,
        'own_NumPy_saved4_to_actual_startup_restore_to_saved8_exact':True,
        'own_NumPy4_sha256':digest(first['state']['RNG']['numpy']),'own_NumPy8_sha256':digest(states[1]['RNG']['numpy']),
        'NumPy_net_state_unchanged_for_observed4_to8_interval':True,
        'bootstrap_restores_only_python_Torch_CUDA_source':record(P/'outputs/parakeet_turkish/bpe_broad_streamed_bootstrap_v2.py'),
        'all207_threeAdam_clocks_and_all845_tensor_bytes_match_independent_oracle':True,
        'exact_field_digests':exact,'actual_obsolete_worker_control_archives':archives,
        'explicit_state_comparison_provenance':'identity differs; NumPy mismatch reported separately, never removed from failed original all4 gate',
        'explicit_journal_execution_provenance_fields':IGNORED_JOURNAL_EXECUTION_FIELDS,
        'float_tolerance_or_RNG_seed_reset_used':False,'production_approval_claimed':False})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('oracle-report','candidate-report','observed-owner0','pause-snapshot','failed-comparator-log','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();run(record(a.oracle_report),record(a.candidate_report),record(a.observed_owner0),record(a.pause_snapshot),
        record(a.failed_comparator_log),Path(a.output).resolve())
