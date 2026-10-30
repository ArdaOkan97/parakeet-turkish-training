"""Read-only exact split8/uninterrupted8 scientific metadata comparison.

Tensor byte hashes are the actual persisted/fresh-restored descriptors, not
numerical samples. No tensor payload/model/optimizer is loaded here.
"""
from pathlib import Path
import argparse,copy,hashlib,json,sys
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from asr18_streamed_persistence_fingerprint_v1 import normalize_fingerprint
from fastmss_phase_aware_plan_bridge_v7 import read_bound
from fastmss_row_serial_cpu_v8 import environment,resource_guard
ROOT=Path(r'E:\gpu_training_scratchpad')
IGNORED_JOURNAL_EXECUTION_FIELDS=[
 'prepared_evidence(record3)','native.precomputed_input_update(record3)',
 'native.augmentation.payload_sha256(feature-metadata provenance)',
 'native.graph_boundary_memory_observations[].before/after(resource telemetry)',
 'native.graph_boundary_memory_observations[].collected_objects/seconds(telemetry)']

def journal(prefix,guard):
    path=Path(prefix['path']);require(path.is_relative_to(ROOT) and path.stat().st_size==prefix['offset']<=2*1024**2,
        'Whole actual8 journal exact small bounded prefix')
    raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==prefix['sha256'],'Actual authoritative journal SHA')
    entries=[]
    for line in raw.splitlines():
        guard();entry=json.loads(line);require(entry['prepared_evidence']==entry['native']['precomputed_input_update'],
            'Same independently consumed source-bound feature metadata provenance')
        entry.pop('prepared_evidence');native=entry['native'];native.pop('precomputed_input_update')
        native['augmentation'].pop('payload_sha256')
        for item in native['graph_boundary_memory_observations']:
            require(item['Python_NumPy_Torch_CUDA_RNG_unchanged'] is True
                and item['live207_parameter_and_gradient_pointers_versions_unchanged'] is True,
                'Both original invariant gates remain true')
            for key in ('before','after','collected_objects','seconds'):item.pop(key)
        entries.append(entry)
    require(len(entries)==8 and [x['optimizer_updates'] for x in entries]==list(range(1,9)),
        'Exactly original chronological updates1..8; no dropped scientific event')
    return entries

def compare(oracle_report_pin,candidate_report_pin,output):
    environment();guard=resource_guard
    oracle=read_bound(oracle_report_pin,guard);candidate=read_bound(candidate_report_pin,guard)
    require(oracle['passed'] is True and candidate['passed'] is True
        and oracle['actual_optimizer_updates']==candidate['actual_optimizer_updates']==8
        and candidate['actual_owner_invocation_count']==2 and candidate['actual_normal_resume_request'] is not None,
        'Actual uninterrupted8 vs separately closed/fresh-owner split4/4 report')
    comparisons=[];states=[];pointers=[];packets=[];tensors=[];samplers=[];journals=[]
    for report in (oracle,candidate):
        guard();capture=report['current_checkpoint'];pointer=read_bound(capture['pointer'],guard)
        pre=normalize_fingerprint(read_bound(capture['pre_save_fingerprint'],guard))
        state=copy.deepcopy(pre['state']);state.pop('identity')
        require(state['updates']==8 and state['clocks']==[64258,2008,62258]
            and len(state['weights'])==207 and len(state['Adam_partitions'])==3,
            'Full mature all207/threeAdam frontier, no reset/float tolerance')
        metadata=read_bound(pointer['checkpoint'],guard)
        require(len(metadata['tensors'])==845 and metadata['payload']==pointer['payload'],
            'Full actual845 tensor descriptors and exact persisted pair')
        states.append(state);tensors.append(metadata['tensors']);pointers.append(pointer)
        packets.append(read_bound(capture['next_draw_packet'],guard))
        from bpe_streaming_checkpoint_semantics_v2 import decode_scalar_tree,select
        # Exact saved full sampler is part of source-typed semantic metadata;
        # no execution identity normalization applies to the sampler tree.
        samplers.append(decode_scalar_tree(select(metadata['tree'],'sampler')))
        journals.append(journal(pointer['journal_prefix'],guard))
    for name,values in (('all207_threeAdam_all4RNG_clocks',states),('all845_tensor_dtype_shape_offsets_bytes_SHA',tensors),
                        ('full_sampler',samplers),('full_next_draw_packet',packets),('chronological_scientific_journal',journals)):
        require(values[0]==values[1],'Exact split/uninterrupted mismatch: '+name)
        comparisons.append({'field':name,'exact':True,'sha256':digest(values[0])})
    require(pointers[0]['payload']['bytes']==pointers[1]['payload']['bytes']
        and pointers[0]['payload']['sha256']==pointers[1]['payload']['sha256'],
        'Full actual tensor payload bytes/SHA equality, independently persistence-verified and fresh restored')
    archives=candidate['actual_obsolete_worker_control_archives']
    require(len(archives)==2 and all(x['passed'] and x['all_original_member_bytes_verified_before_exact_unlink']
        and not Path(x['original_folder']).exists() for x in archives),
        'Actual obsolete first native/recovery controls archived and released, current proofs protected')
    result={'kind':'ASR18_actual_normal_owner_resume_split8_scientific_byte_equivalence_v1','passed':True,
        'source':record(__file__),'oracle_report':oracle_report_pin,'candidate_report':candidate_report_pin,
        'comparisons':comparisons,'tensor_count':845,'all207_threeAdam_all4RNG_exact':True,
        'ignored_state_field_only':'state.identity(execution provenance)',
        'ignored_journal_execution_fields_only':IGNORED_JOURNAL_EXECUTION_FIELDS,
        'actual_obsolete_worker_control_archives':archives,'float_tolerance_used':False,
        'actual_GPU_or_audio_jobs_here':0,'production_approval_claimed':False}
    require(output.is_relative_to(ROOT) and not output.exists(),'One fresh E-only comparison receipt')
    write_json(output,result)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('oracle-report','candidate-report','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();compare(record(args.oracle_report),record(args.candidate_report),Path(args.output).resolve())
