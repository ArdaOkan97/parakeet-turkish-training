"""Inactive concrete serial350 DEV glue for the full-run lifecycle.

No launcher, model construction, optimizer construction or import-time ML work.
The native owner must supply a separately reviewed optimizer-free worker and
owned executor. The completed finite diagnostic DEV is derivation evidence,
never authority to promote its checkpoint or reuse its <=20-update admission.
"""
from copy import deepcopy
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import threading

from bpe_broad_production_state_v1 import canonical, digest, require
from bpe_full_run_lifecycle_v2 import (
    REFERENCES, LIMITS, SCORER_SHA, GATE_SHA, bound, read_bound, official_DEV,
    milestones, progress,
)
from bpe_broad_audio_v1 import record
from bpe_streaming_checkpoint_semantics_v2 import (
    decode_scalar_tree, select, verify_saved_metadata,
)

ROOT = Path(r'E:\gpu_training_scratchpad')
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = PROJECT/'work'/'data'
OWNER = '/root/memory_diagnosis'
REQUEST_KIND = 'ASR18_full_run_recovered_boundary_serial350DEV_request_v2'
REPORT_KIND = 'ASR18_serial_complete350_boundary_DEV_v1'
OPTIMIZER_FREE_PROTOCOL = 'ASR18_selected207_all4RNG_zeroAdam_serialDEV_protocol_v1'
EXECUTION_KEYS = {'worker_source', 'parent_source', 'guard_source', 'bootstrap_source'}
DEPENDENCIES = (
    'bpe_full_run_lifecycle_v2.py', 'bpe_streaming_checkpoint_v1.py',
    'bpe_streaming_checkpoint_semantics_v2.py', 'bpe_streaming_training_state_v1.py',
    'fastmss_owned_process_closure_v2.py',
    'asr18_DEV_weight_only_mapper_9223_v1.py', 'asr18_optimizer_free_DEV_structure_9223_v1.py',
    'asr18_full_run_DEV_runtime_9223_v2.py',
    'bpe_broad_production_v3.py', 'bpe_broad_full207_optimizer_v1.py',
    'train_lora_batched_tsc_bpe_full_decoder_v1.py', 'benchmark.py', 'metrics.py',
    'tsc_bpe_full_decoder_transition_v1.py', 'runtime.py', 'confidence.py',
)


def serial():
    require(threading.current_thread() is threading.main_thread(), 'Serial main-thread DEV only')


def same(left, right):
    return left == right and canonical(left) == canonical(right)


def integer(value, low=0):
    require(type(value) is int and value >= low, 'Exact nonboolean DEV frontier required')
    return value


def sources():
    """Small source bindings; caller pins the complete transitive runtime too."""
    return {name: record(HERE/name) for name in DEPENDENCIES}


def new_E(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT.resolve()), 'All new DEV artifacts must stay on designated E')
    return path


def write_new(path, value):
    path = new_E(path)
    with path.open('xb') as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
    return record(path)


def canonical_DEV_paths():
    return {language: DATA/('portuguese_retention_v1' if language == 'pt' else '')/
            ('validation_'+language+'.jsonl') for language in REFERENCES}


def _saved_dictionary(node):
    require(type(node) is list and len(node) == 2 and node[0] == 'dict'
            and type(node[1]) is list and len(node[1]) <= 100000, 'Bounded tagged saved dictionary required')
    result = {}
    for pair in node[1]:
        require(type(pair) is list and len(pair) == 2 and type(pair[0]) is list
                and len(pair[0]) == 2 and pair[0][0] == 'scalar'
                and type(pair[0][1]) in (str, int) and pair[0][1] not in result,
                'Unique exact saved scalar dictionary keys required')
        result[pair[0][1]] = pair[1]
    return result


def checkpoint_format_view(metadata):
    """Check the actual frozen writer/state format without Torch or mutation.

    Format compatibility is deliberately independent of full-run admission. A
    historical diagnostic checkpoint may pass this function while it MUST fail
    checked_checkpoint. No actual tensor load/equality or process0 is implied.
    """
    require(type(metadata) is dict and set(metadata) == {'kind','schema_version','endianness','source',
            'maximum_CPU_tensor_slice_bytes','payload','tensors','tree','dynamic_module_serialization'}
            and metadata['kind'] == 'ASR18_streaming_named_tensor_checkpoint_v1'
            and type(metadata['schema_version']) is int and metadata['schema_version'] == 1
            and metadata['endianness'] == 'little' and metadata['maximum_CPU_tensor_slice_bytes'] == 4*1024**2
            and metadata['dynamic_module_serialization'] is False
            and same(metadata['source'], record(HERE/'bpe_streaming_checkpoint_v1.py')),
            'Exact frozen writer format uses payload, never raw_payload')
    payload = metadata['payload']
    require(type(payload) is dict and set(payload) == {'path','bytes','sha256'}
            and type(payload['path']) is str and type(payload['bytes']) is int and payload['bytes'] > 0
            and type(payload['sha256']) is str and len(payload['sha256']) == 64,
            'Exact saved tensor payload record required')
    entries = metadata['tensors']
    require(type(entries) is list and 0 < len(entries) <= 4096, 'Bounded actual named tensor entries')
    previous = 0
    for index, entry in enumerate(entries):
        require(type(entry) is dict and set(entry) == {'index','dtype','shape','offset','bytes','sha256'}
                and type(entry['index']) is int and entry['index'] == index
                and type(entry['shape']) is list and all(type(size) is int and size >= 0 for size in entry['shape'])
                and type(entry['offset']) is int and entry['offset'] % 64 == 0 and entry['offset'] >= previous
                and type(entry['bytes']) is int and entry['bytes'] >= 0
                and entry['offset']+entry['bytes'] <= payload['bytes']
                and type(entry['sha256']) is str and len(entry['sha256']) == 64,
                'Exact bounded aligned tensor extent before weight/RNG format access')
        previous = entry['offset']+entry['bytes']
    require(previous == payload['bytes'], 'Complete named tensor payload extent required')
    tree = metadata['tree']; top = _saved_dictionary(tree)
    require(set(top) == {'kind','diagnostic_only','identity','source_provenance','weights','optimizer','sampler',
                         'augmentation','python_rng','torch_rng','cuda_rng','numpy_rng'},
            'Exact frozen bpe_streaming_training_state_v1 field names required')
    scalar = lambda *path: decode_scalar_tree(select(tree, *path))
    require(scalar('kind') == 'ASR18_broad_caller_state_v1' and type(scalar('diagnostic_only')) is bool
            and type(scalar('identity')) is dict and type(scalar('source_provenance')) is dict,
            'Saved state/source scientific identity fields differ')
    updates = integer(scalar('optimizer','updates'))
    require(scalar('optimizer','kind') == 'BPE9223_broad_full207_three_AdamW_mixed_fresh_clock_v1'
            and scalar('optimizer','appended512_Adam_clock') == updates
            and type(scalar('optimizer','appended512_Adam_clock')) is int
            and same(scalar('optimizer','clocks'), [clock+updates for clock in (64250,2000,62250)]),
            'Exact named optimizer scalar clocks required, never loaded in DEV')
    sampler = scalar('sampler'); prefix = scalar('augmentation','committed_journal')
    require(type(sampler) is dict and type(prefix) is dict and set(prefix) == {'path','offset','sha256'}
            and type(prefix['path']) is str and type(prefix['offset']) is int and prefix['offset'] >= 0
            and type(prefix['sha256']) is str and len(prefix['sha256']) == 64,
            'Actual sampler and exact journal prefix scalar paths required')
    def tensor_entry(node, dtype):
        require(type(node) is list and len(node) == 2 and node[0] == 'tensor'
                and type(node[1]) is int and 0 <= node[1] < len(entries), 'Exact saved tensor reference')
        entry = entries[node[1]]
        require(entry['dtype'] == dtype and entry['bytes'] > 0, 'Expected actual weight/RNG tensor dtype')
        return entry
    weights = _saved_dictionary(top['weights'])
    require(len(weights) == 207 and all(type(name) is str for name in weights)
            and len({node[1] for node in weights.values() if type(node) is list and len(node) == 2}) == 207,
            'All207 unique named weight tensor references required')
    for node in weights.values():
        entry = tensor_entry(node, 'float32')
        require(math.prod(entry['shape'])*4 == entry['bytes'], 'FP32 named weight shape/bytes differ')
    torch_entry = tensor_entry(top['torch_rng'], 'uint8')
    require(len(torch_entry['shape']) == 1 and torch_entry['shape'][0] == torch_entry['bytes'] <= 1024**2,
            'Bounded CPU Torch RNG tensor shape required')
    cuda = top['cuda_rng']
    require(type(cuda) is list and len(cuda) == 2 and cuda[0] == 'list'
            and type(cuda[1]) is list and 1 <= len(cuda[1]) <= 8, 'Exact saved CUDA RNG list required')
    for node in cuda[1]:
        entry = tensor_entry(node, 'uint8')
        require(len(entry['shape']) == 1 and entry['shape'][0] == entry['bytes'] <= 1024**2,
                'Bounded CUDA RNG tensor shape required')
    python = scalar('python_rng'); local = random.Random(0); local.setstate(python)
    require(same(local.getstate(), python), 'Actual typed Python RNG scalar tree required')
    numpy = scalar('numpy_rng')
    require(type(numpy) is tuple and len(numpy) == 5 and numpy[0] == 'MT19937'
            and type(numpy[1]) is list and len(numpy[1]) == 624
            and all(type(value) is int and 0 <= value < 2**32 for value in numpy[1])
            and type(numpy[2]) is int and 0 <= numpy[2] <= 624
            and type(numpy[3]) is int and numpy[3] in (0,1)
            and type(numpy[4]) is float and math.isfinite(numpy[4]),
            'Exact NumPy RNG tuple/list scalar encoding from frozen state writer required')
    return {'kind':'ASR18_saved_checkpoint_format_view_for_DEV_v1','passed':True,
        'diagnostic_only':scalar('diagnostic_only'),'optimizer_updates':updates,'weight_count':207,
        'payload':deepcopy(payload),'saved_source_clocks':scalar('optimizer','clocks'),
        'sampler_state_sha256':digest(sampler),'journal_prefix':deepcopy(prefix),
        'all4_RNG_field_paths_format_checked':True,'tensor_values_loaded_or_compared':False,
        'full_run_checkpoint_admitted':False,'state_writer':record(HERE/'bpe_streaming_training_state_v1.py')}


def checked_policy(policy_pin, ROOT_review_pin, *, identity, execution):
    """Membership is independently ROOT pinned, not inferred from word totals."""
    policy = read_bound(policy_pin); review = read_bound(ROOT_review_pin)
    require(review['kind'] == 'ROOT_selective_optimizer_free_serial350DEV_policy_and_sources_review_v3'
            and review['accepted'] is True and review['reviewer'] == '/root'
            and review['canonical_fixed_DEV_manifests_independently_verified'] is True
            and review['scope'] == 'exact_selective_optimizer_free_serial_DEV_runtime_sources_and_policy'
            and same(review['scorer_policy'], policy_pin)
            and same(review['full_run_source_contract'], identity['full_run_source_contract'])
            and review['run_identity_sha256'] == digest(identity)
            and same(review['implementation'], record(__file__))
            and same(review['dependencies'], sources())
            and same(review['execution_sources'], execution),
            'Separate exact ROOT policy/runtime review is required; this source is inactive')
    require(type(execution) is dict and set(execution) == EXECUTION_KEYS,
            'Explicit future native-owner worker/parent/guard/bootstrap sources required')
    for pin in execution.values(): bound(pin)
    require(policy['official_scorer'] == record(HERE/'metrics.py')
            and policy['official_scorer']['sha256'] == SCORER_SHA
            and policy['integer_gate_contract'] == record(HERE/'tsc_bpe_full_decoder_transition_v1.py')
            and policy['integer_gate_contract']['sha256'] == GATE_SHA
            and same(policy['reference_words'], REFERENCES)
            and same(policy['retention_limits'], LIMITS)
            and same(review['canonical_DEV_manifests'], policy['DEV_manifests'])
            and set(policy['DEV_manifests']) == set(REFERENCES),
            'Immutable official scorer, six integer gates and independently pinned fixed DEV required')
    for language, path in canonical_DEV_paths().items():
        pin = policy['DEV_manifests'][language]; bound(pin)
        require(pin['path'] == str(path.resolve()), 'DEV route differs from the canonical original evaluator')
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        require(len(rows) == 50 and len({row['source_id'] for row in rows}) == 50
                and all(row['language'] == language and row['dataset_split'] == 'validation' for row in rows),
                'Exactly the independently pinned original50 per language; never TEST')
    return policy


def checked_checkpoint(pointer_pin, *, identity, expected_updates, boundary):
    """Check the actual saved scalar tree/payload/prefix, not pointer counters alone."""
    pointer = read_bound(pointer_pin)
    require(pointer['kind'] == 'ASR18_production_window_completed_checkpoint_pointer_v2'
            and pointer['diagnostic_only'] is False
            and pointer['writer_returned_and_payload_verified'] is True
            and pointer['all_window_updates_committed'] is True
            and integer(pointer['committed_updates'], 1) == expected_updates,
            'Full-run coherent actual checkpoint at the exact declared pass tail required')
    metadata = read_bound(pointer['checkpoint'])
    checkpoint_format_view(metadata)
    require(same(metadata['payload'], pointer['payload'])
            and Path(pointer['payload']['path']).resolve() ==
            Path(pointer['checkpoint']['path']).resolve().with_suffix('.tensors.bin'),
            'Exact fsynced checkpoint metadata/payload pair required')
    # Payload can exceed the small-JSON bound; record() streams it without loading.
    require(same(record(pointer['payload']['path']), pointer['payload']), 'Complete checkpoint payload hash differs')
    tree = metadata['tree']
    scalar = lambda *path: decode_scalar_tree(select(tree, *path))
    saved_identity, sampler = scalar('identity'), scalar('sampler')
    require(same(saved_identity, pointer['identity'])
            and same(saved_identity['caller_identity']['full_run_lifecycle_identity'], identity),
            'Saved scientific source identity must carry the exact full-run lifecycle identity')
    semantic = verify_saved_metadata(metadata, identity=saved_identity, optimizer_updates=expected_updates,
        sampler_state=sampler, journal_prefix=pointer['journal_prefix'], diagnostic_only=False)
    require(same(semantic, pointer['saved_semantic_state_verified'])
            and digest(sampler) == pointer['sampler_state_sha256']
            and progress(sampler)['optimizer_updates'] == expected_updates
            and boundary in milestones(sampler)
            and all(progress(sampler)[key] == boundary[key] for key in ('broad_draws', 'cleaner_draws', 'optimizer_updates'))
            and scalar('optimizer', 'clocks') == [clock+expected_updates for clock in (64250, 2000, 62250)],
            'Saved actual all-source clocks/sampler/pass-tail/prefix differ')
    return pointer, metadata, sampler


def read_logical_durable_head(pin):
    """Read ONE bounded immutable frame, never hash a whole appending stream."""
    import struct
    require(type(pin) is dict and set(pin)=={'kind','path','offset','framed_bytes','sha256','sequence'}
            and pin['kind']=='ASR18_logical_immutable_durable_head_descriptor_v2', 'Exact logical durable descriptor')
    path=Path(pin['path']).resolve();require(path.is_relative_to(ROOT.resolve()), 'E-only durable stream')
    offset=integer(pin['offset']);framed=integer(pin['framed_bytes'],41);sequence=integer(pin['sequence'],1)
    require(framed<=65536 and type(pin['sha256']) is str and len(pin['sha256'])==64, '64KiB complete framed record cap')
    with path.open('rb') as stream:
        stream.seek(offset);header=stream.read(40)
        require(len(header)==40 and struct.unpack('<Q',header[:8])[0]+40==framed, 'Full bounded frame header')
        raw=stream.read(framed-40)
    require(len(raw)==framed-40 and hashlib.sha256(raw).digest()==header[8:]
            and hashlib.sha256(raw).hexdigest()==pin['sha256'], 'Exact immutable frame SHA')
    value=json.loads(raw)
    require(value['kind']=='ASR18_packed_durable_event_and_head_record_v2'
            and type(value['sequence']) is int and value['sequence']==sequence, 'Exact durable record kind/sequence')
    head=value['head']
    require(head['kind']=='ASR18_packed_durable_full_run_lifecycle_head_v2'
            and type(head['sequence']) is int and head['sequence']==sequence
            and head['identity_sha256']==value['identity_sha256'] and same(head['owner'],value['owner'])
            and head['production_enabled'] is False and head['TEST_complete'] is False and head['export_complete'] is False
            and type(head['total_event_bytes']) is int and head['total_event_bytes']==offset+framed,
            'Exact owner/head/record logical frontier')
    return value


def _checked_current_native_boundary(evidence, *, request, pointer, read_artifact, verify_process, read_head):
    """Pure validation core; injected I/O is used only by explicit metadata fixtures."""
    pins=request['native_boundary_sources'];identity_sha=digest(request['identity'])
    expected=request['optimizer_updates']
    require(type(expected) is int and expected>=1, 'Exact nonboolean current frontier')
    require(evidence['kind']=='ASR18_full_run_native_exit_and_CURRENT_checkpoint_recovery_boundary_for_serial_DEV_v2'
            and evidence['owner']==OWNER and evidence['passed'] is True and evidence['diagnostic_only'] is False
            and same(evidence['source'],pins['boundary_verifier'])
            and evidence['production_recovery_scope']=='CURRENT_LATEST_ONLY_WITH_VERIFIED_DURABLE_PERSISTENCE_PREFIX'
            and same(evidence['checkpoint_pointer'],request['checkpoint_pointer'])
            and evidence['run_identity_sha256']==identity_sha and type(evidence['optimizer_updates']) is int
            and evidence['optimizer_updates']==expected and evidence['sampler_state_sha256']==pointer['sampler_state_sha256']
            and same(evidence['journal_prefix'],pointer['journal_prefix'])
            and evidence['all207_threeAdam_all4RNG_exact_checkpoint_restore_verified'] is True,
            'Non-diagnostic CURRENT actual native0/fresh recovery boundary required')
    capture=evidence['current_checkpoint_capture']
    require(same(capture['pointer'],request['checkpoint_pointer']) and same(capture['checkpoint'],pointer['checkpoint'])
            and capture['checkpoint_persistence_verified'] is True and capture['exact_GPU_restore_performed'] is False
            and capture['publication_role']=='completed_window_end'
            and capture['current_latest_recovery_required_before_phase_DEV_winner_advance'] is True,
            'Actual current persistence capture, never all-history recovery fiction')
    native_process=read_artifact(evidence['current_native_process0'])
    verify_process(native_process,kind='ASR18_owned_GPU_process_completion_v1',owner=OWNER)
    require(same(native_process['worker_source'],pins['native_worker'])
            and same(native_process['parent_source'],pins['native_parent']), 'Current native process source')
    native=read_artifact(native_process['report'])
    require(native['passed'] is True and native['diagnostic_only'] is False
            and native['run_identity_sha256']==identity_sha and same(native['current_checkpoint'],capture)
            and type(native['final_optimizer_updates']) is int and native['final_optimizer_updates']==expected
            and native['all_exact_GPU_checkpoint_restores_deferred'] is True,
            'Actual closed current native report/capture; no unbounded historical windows list')
    recovery_process=read_artifact(evidence['actual_recovery_process0'])
    verify_process(recovery_process,kind='ASR18_owned_GPU_process_completion_v1',owner=OWNER,report=evidence['recovery_report'])
    require(same(recovery_process['worker_source'],pins['recovery_worker'])
            and same(recovery_process['parent_source'],pins['recovery_parent']), 'Current recovery process source')
    recovery=read_artifact(evidence['recovery_report'])
    require(recovery['kind']=='ASR18_full_run_actual_CURRENT_checkpoint_recovered_v2'
            and recovery['passed'] is True and recovery['diagnostic_only'] is False
            and recovery['current_latest_checkpoint_recovered'] is True
            and recovery['historical_fresh_GPU_restores_claimed'] is False
            and same(recovery['current_native_process0'],evidence['current_native_process0'])
            and recovery['run_identity_sha256']==identity_sha
            and type(recovery['actual_native_updates_in_recovery_process']) is int
            and recovery['actual_native_updates_in_recovery_process']==0
            and type(recovery['actual_draws_consumed_in_recovery']) is int and recovery['actual_draws_consumed_in_recovery']==0
            and type(recovery['recovered_checkpoints']) is list and len(recovery['recovered_checkpoints'])==1,
            'ONE actual current fresh recovery, zero native updates/draws')
    recovered_pin=recovery['recovered_checkpoints'][0];recovered=read_artifact(recovered_pin)
    require(recovered['passed'] is True and same(recovered['pointer'],capture['pointer'])
            and same(recovered['checkpoint'],pointer['checkpoint'])
            and same(recovered['pre_save_fingerprint'],capture['pre_save_fingerprint'])
            and same(recovered['next_draw_packet'],capture['next_draw_packet'])
            and recovered['all207_threeAdam_all4RNG_match_actual_pre_save'] is True
            and recovered['both_samplers_journal_prefix_and_next_draw_exact'] is True
            and type(recovered['native_updates_performed']) is int and recovered['native_updates_performed']==0
            and type(recovered['draws_consumed']) is int and recovered['draws_consumed']==0
            and recovered['separate_fresh_GPU_recovery'] is True,
            'Current all207/threeAdam/all4RNG/samplers/prefix/nextdraw exact fresh recovery')
    ack=read_artifact(evidence['production_recovery_boundary'])
    require(ack['kind']=='ASR18_full_run_current_latest_recovery_and_verified_persistence_prefix_v1'
            and ack['diagnostic_only'] is False and ack['run_identity_sha256']==identity_sha
            and same(ack['final_checkpoint_pointer'],request['checkpoint_pointer'])
            and ack['fresh_recovery_scope']=='current_latest_only'
            and ack['all_persistence_verified_through_pending_prefix'] is True
            and ack['all_historical_checkpoints_freshly_restored'] is False
            and ack['current_latest_all207_threeAdam_all4RNG_sampler_prefix_nextdraw_match_actual_pre_save'] is True
            and type(ack['actual_recovery_native_updates']) is int and ack['actual_recovery_native_updates']==0
            and type(ack['actual_recovery_draws_consumed']) is int and ack['actual_recovery_draws_consumed']==0
            and same(ack['current_latest_exact_restore'],recovered_pin)
            and same(ack['persistence_prefix_coverage_index'],evidence['durable_persistence_prefix_verification']),
            'Exact durable adapter current-recovery acknowledgement')
    prefix=read_artifact(evidence['durable_persistence_prefix_verification'])
    require(prefix['kind']=='ASR18_verified_durable_persistence_prefix_coverage_v1'
            and prefix['passed'] is True and same(prefix['source'],pins['durable_prefix_verifier'])
            and prefix['run_identity_sha256']==identity_sha and same(prefix['pending_head'],ack['pending_head'])
            and same(prefix['pending'],ack['pending'])
            and same(prefix['final_checkpoint_pointer'],request['checkpoint_pointer'])
            and prefix['current_checkpoint_capture_sha256']==digest(capture)
            and prefix['all_completed_window_PRE_saved_live_persistence_verified'] is True
            and prefix['historical_fresh_GPU_restores_claimed'] is False
            and type(prefix['native_updates_consumed_in_verification']) is int and prefix['native_updates_consumed_in_verification']==0,
            'Independent exact source-bound durable-prefix coverage; not historical GPU restoration')
    pending_record=read_head(ack['pending_head']);current_record=read_head(evidence['durable_head'])
    pending=pending_record['head'];head=current_record['head']
    require(pending_record['identity_sha256']==head['identity_sha256']==identity_sha
            and same(pending['owner'],head['owner'])
            and evidence['durable_head']['path']==ack['pending_head']['path']
            and evidence['durable_head']['sequence']==ack['pending_head']['sequence']+1
            and evidence['durable_head']['offset']==ack['pending_head']['offset']+ack['pending_head']['framed_bytes']
            and same(pending['pending'],ack['pending'])
            and same(pending['lifecycle']['last_pointer'],request['checkpoint_pointer'])
            and current_record['operation']=='current_latest_recovery_and_persistence_prefix_acknowledged'
            and same(current_record['previous_head'],ack['pending_head'])
            and same(current_record['evidence']['recovery'],evidence['production_recovery_boundary'])
            and same(head['last_recovery'],evidence['production_recovery_boundary'])
            and type(head['pending']['publications']) is int and head['pending']['publications']==0
            and type(head['pending']['windows']) is int and head['pending']['windows']==0
            and same(head['lifecycle']['last_pointer'],request['checkpoint_pointer'])
            and head['lifecycle']['last_sampler_sha256']==pointer['sampler_state_sha256'],
            'Actual immutable post-recovery durable head must bind current checkpoint and cleared pending prefix')
    return evidence


def checked_native_boundary(pin, *, request, pointer):
    from fastmss_owned_process_closure_v2 import verify
    evidence=read_bound(pin);pins=request['native_boundary_sources']
    require(type(pins) is dict and set(pins)=={'boundary_verifier','native_worker','native_parent',
            'recovery_worker','recovery_parent','durable_prefix_verifier'}, 'Exact current boundary and independent prefix sources')
    for source in pins.values():bound(source)
    review=read_bound(request['ROOT_policy_review'])
    require(same(review['native_boundary_sources'],pins)
            and same(review['durable_prefix_verifier_source'],pins['durable_prefix_verifier']), 'Exact ROOT-pinned prefix verifier')
    return _checked_current_native_boundary(evidence,request=request,pointer=pointer,
        read_artifact=read_bound,verify_process=verify,read_head=read_logical_durable_head)


def checked_request(request):
    serial()
    require(request['kind'] == REQUEST_KIND and request['owner'] == OWNER
            and request['production_enabled'] is False
            and request['action']['kind'] == 'checkpoint_then_serial350DEV'
            and request['split'] == 'validation' and request['clips'] == 350
            and request['TEST_used'] is False and request['native_updates_in_DEV'] == 0
            and type(request['native_updates_in_DEV']) is int,
            'Explicit inactive optimizer-free fixed350 boundary request required')
    identity = request['identity']
    require(identity['kind'] == 'ASR18_reviewed_full_two_broad_one_cleaner_run_identity_v1'
            and identity['native_owner'] == OWNER
            and identity['required_broad_passes'] == 2 and identity['required_cleaner_passes'] == 1
            and same(identity['source_original_clocks'], [64250, 2000, 62250])
            and identity['source_optimizer_RNG_sampler_reset'] is False
            and same(identity['scorer_policy'], request['scorer_policy']), 'Exact full-run identity and policy')
    bound(identity['full_run_source_contract'])
    policy = checked_policy(request['scorer_policy'], request['ROOT_policy_review'],
                            identity=identity, execution=request['execution_sources'])
    pointer, metadata, sampler = checked_checkpoint(request['checkpoint_pointer'], identity=identity,
        expected_updates=integer(request['optimizer_updates'], 1), boundary=request['action']['boundary'])
    checked_native_boundary(request['native_checkpoint_actual0'], request=request, pointer=pointer)
    require(same(request['dependencies'], sources()) and same(request['implementation'], record(__file__)),
            'DEV source or dependency changed after exact review')
    return policy, pointer, metadata, sampler


def _tensor_hash(torch, tensor, guard):
    """Post-evaluation equality without another fullCPU207-weight snapshot."""
    require(tensor.is_contiguous(), 'Original contiguous all207 parameters required')
    raw = tensor.detach().view(torch.uint8).reshape(-1)
    result = hashlib.sha256()
    for start in range(0, raw.numel(), 4*1024**2):
        guard(); chunk = raw[start:start+4*1024**2].cpu()
        result.update(chunk.numpy().tobytes()); del chunk
    return {'dtype': str(tensor.dtype), 'shape': list(tensor.shape), 'sha256': result.hexdigest()}


def _capture_RNG(torch, np):
    n = np.random.get_state()
    return (random.getstate(), (n[0], n[1].copy(), *n[2:]), torch.get_rng_state().clone(),
            [value.clone() for value in torch.cuda.get_rng_state_all()])


def _restore_RNG(torch, np, rng):
    random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
    torch.cuda.set_rng_state_all(rng[3])
    actual = _capture_RNG(torch, np)
    require(actual[0] == rng[0] and actual[1][0] == rng[1][0]
            and np.array_equal(actual[1][1], rng[1][1]) and actual[1][2:] == rng[1][2:]
            and torch.equal(actual[2], rng[2]) and len(actual[3]) == len(rng[3])
            and all(torch.equal(a, b) for a, b in zip(actual[3], rng[3])), 'All4 child RNG families restore exactly')


def check_optimizer_free_report_protocol(report):
    """Additive protocol identity; the lifecycle's canonical reportkind stays exact."""
    selected = report['selective_weight_RNG_mapping']
    require(report['kind'] == REPORT_KIND and report['optimizer_free_protocol'] == OPTIMIZER_FREE_PROTOCOL
            and report['all207_weight_only_restore_byte_exact'] is True
            and report['all207_after_DEV_hashes_unchanged'] is True
            and report['all4_child_RNG_restored_exactly'] is True
            and report['optimizer_objects_constructed_or_loaded_in_DEV_worker'] is False
            and report['sampler_replay_and_augmentation_objects_created_or_advanced'] is False
            and type(report['actual_native_updates_in_DEV']) is int and report['actual_native_updates_in_DEV'] == 0
            and type(report['Adam_tensor_views_created']) is int and report['Adam_tensor_views_created'] == 0
            and type(selected['weights']) is int and selected['weights'] == 207
            and type(selected['Adam_tensor_views']) is int and selected['Adam_tensor_views'] == 0
            and type(selected['optimizer_objects_constructed']) is int and selected['optimizer_objects_constructed'] == 0
            and type(report['selected_saved_CUDA_RNG_states']) is int
            and report['selected_saved_CUDA_RNG_states'] >= 1
            and type(selected['tensor_views']) is int
            and selected['tensor_views'] == 208+report['selected_saved_CUDA_RNG_states'],
            'Exact optimizer-free protocol plus canonical lifecycle reportkind/207/all4RNG/zeroAdam fields')
    return True


def evaluate_boundary_checkpoint(model, processor, request_pin, folder, *, owned_worker_admission,
                                 torch, stage, admit, stop_check):
    """Concrete future sole-owned worker hook, with an already built BPE model.

    This function cannot launch or construct a model/optimizer. A NEW reviewed
    native-owner bootstrap/driver must admit it, prove the actual sole lease,
    maintain all resource guards, and actually close before seal_completed_DEV.
    """
    request = read_bound(request_pin)
    policy, pointer, metadata, sampler = checked_request(request)
    folder = new_E(folder); require(folder.is_dir(), 'Fresh owner-created E worker namespace required')
    from fastmss_owned_process_closure_v2 import identity as process_identity
    producer = process_identity()
    # The future parent binds this attestation to its exact live owned PID/ctime,
    # lease and resource admission, then supplies a genuine process0 afterwards.
    boot = read_bound(owned_worker_admission)
    require(boot['kind'] == 'ASR18_sole_owned_optimizer_free_serialDEV_worker_admission_v1'
            and boot['owner'] == OWNER and boot['sole_owned_GPU_child'] is True
            and boot['actual_training_optimizer_running'] is False
            and same(boot['producer'], producer) and same(boot['request'], request_pin)
            and same(boot['worker_source'], request['execution_sources']['worker_source'])
            and same(boot['parent_source'], request['execution_sources']['parent_source'])
            and boot['optimizer_objects_constructed_or_loaded_in_DEV_worker'] is False,
            'Actual independently admitted sole optimizer-free worker identity required')
    import numpy as np
    from asr18_DEV_weight_only_mapper_9223_v1 import WeightOnlyCheckpoint
    from bpe_broad_production_v3 import prohibit_model_transforms, frozen_parameter
    require(model.joint._ASR18_fp32_joint_policy['kind'] == 'ASR18_joint_canonical_T1_U128_FP32_v2'
            and model.config.vocab_size == 9223 and model.config.blank_token_id == 8192,
            'Original all207 BPE model and canonical joint policy required')
    prohibit_model_transforms(model, torch)
    admit('full_run_DEV_weight_only_restore_and_hash', 350_000_000)
    mapped = WeightOnlyCheckpoint(pointer['checkpoint'], torch=torch, guard=stop_check)
    selection = deepcopy(mapped.selection)
    def restore_selected(saved):
        weights = saved['weights']
        from asr18_optimizer_free_DEV_structure_9223_v1 import expected_names
        require(saved['diagnostic_only'] is False and len(weights) == 207
                and set(weights) == set(expected_names(model))
                and set(weights) == {name for name, p in model.named_parameters() if p.requires_grad}
                and saved['optimizer']['updates'] == request['optimizer_updates']
                and saved['optimizer']['clocks'] == [clock+request['optimizer_updates'] for clock in (64250, 2000, 62250)]
                and digest(saved['sampler']) == digest(sampler), 'All207 source clocks/names/sampler required')
        actual = dict(model.named_parameters())
        require(all(actual[name].shape == weight.shape and actual[name].dtype == weight.dtype == torch.float32
                    for name, weight in weights.items()), 'All207 shape/dtype must agree BEFORE transfer')
        result = model.load_state_dict(weights, strict=False)
        require(not result.unexpected_keys, 'No unexpected saved weight names')
        hashes = {name: _tensor_hash(torch, actual[name], stop_check) for name in weights}
        for name, hashed in hashes.items():
            index = select(metadata['tree'], 'weights', name)[1]; entry = metadata['tensors'][index]
            require(hashed == {'dtype':'torch.float32','shape':entry['shape'],'sha256':entry['sha256']},
                    'Actual all207 GPU restoration must match exact saved raw bytes')
        n = saved['numpy_rng']
        random.setstate(saved['python_rng']); np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        captured = _capture_RNG(torch, np)
        require(captured[0] == saved['python_rng'] and captured[1][0] == n[0]
                and captured[1][1].tolist() == n[1] and captured[1][2:] == n[2:]
                and torch.equal(captured[2], saved['torch_rng'])
                and len(captured[3]) == len(saved['cuda_rng'])
                and all(torch.equal(a,b) for a,b in zip(captured[3],saved['cuda_rng'])),
                'Every saved Python/NumPy/Torch/CUDA RNG must actually restore')
        return hashes, captured
    failure = None
    try:
        weight_hashes, rng = restore_selected(mapped.value)
    except BaseException as error:
        # Drop exception frames that can retain borrowed readonly tensor views.
        failure = type(error).__name__+': '+str(error); error.__traceback__ = None
    mapped.close()
    require(failure is None, 'Selective DEV restore failed: '+str(failure))
    require(selection['weights'] == 207 and selection['Adam_tensor_views'] == 0
            and selection['optimizer_objects_constructed'] == 0, 'No Adam view/object in DEV')
    model._ASR18_DEV_weights_restored = True
    gc.collect(); model.requires_grad_(False); model.eval(); torch.cuda.empty_cache()
    frozen = model._diagnostic_frozen_backing
    require(len(frozen) == 294 and all(frozen_parameter(model, name).device.type == 'cpu'
            and not frozen_parameter(model, name).requires_grad
            and frozen_parameter(model, name).data_ptr() == backing.data_ptr() for name, backing in frozen.items()),
            'All read-only streamed frozen backing pointers must remain unchanged')
    from asr18_optimizer_free_DEV_structure_9223_v1 import assert_ready
    assert_ready(model, torch=torch, names=weight_hashes)
    stage('full_run_optimizer_free_serial350DEV_ready', checkpoint_pointer=request['checkpoint_pointer'])
    # Reused finite measurement, NOT a whole-worker or new full-run fit approval.
    # The future parent must review full bootstrap/current-private + this allowance.
    admit('full_run_optimizer_free_serial350DEV_original_route', 2_100_000_000)
    import benchmark
    import train_lora_batched_tsc_bpe_full_decoder_v1 as trainer
    require(Path(benchmark.DATA).resolve() == DATA.resolve() == Path(trainer.DATA).resolve(),
            'Original fixed evaluator data route changed')
    tag = 'ASR18_full_run_DEV_'+folder.name
    result_folder = ROOT/'results'/tag
    require(not result_folder.exists() and not (ROOT/'results'/(tag+'_pt')).exists(),
            'Fresh prediction namespace only; no cached partial or historical predictions')
    original_inputs = benchmark.inputs_for
    def inputs(processor, row):
        stop_check(); return original_inputs(processor, row)
    benchmark.inputs_for = inputs
    try:
        scores = trainer.evaluate(model, processor, tag)
    finally:
        benchmark.inputs_for = original_inputs
        _restore_RNG(torch, np, rng)
    require(all(not path.name.startswith('test_') for path in result_folder.iterdir()), 'TEST must remain untouched')
    actual = dict(model.named_parameters())
    after_hashes = {name: _tensor_hash(torch, actual[name], stop_check) for name in weight_hashes}
    require(same(weight_hashes, after_hashes)
            and model._diagnostic_frozen_metrics['logical_current_matrix_bytes'] == 0,
            'All207 weights and streamed matrix lifetime must remain unchanged across DEV')
    gate = trainer.gate(scores)
    require(same(gate['retention_allowed_word_errors'], LIMITS)
            and gate['retention_early_stop_active'] is False, 'Immutable all6 monitor-only integer gates')
    # Revalidate the entire immutable source checkpoint and small saved semantics;
    # DEV has neither instantiated a sampler nor replay/augmentation transactions.
    checked_checkpoint(request['checkpoint_pointer'], identity=request['identity'],
        expected_updates=request['optimizer_updates'], boundary=request['action']['boundary'])
    report = {'kind': REPORT_KIND, 'optimizer_free_protocol': OPTIMIZER_FREE_PROTOCOL,
        'passed': True, 'serial_optimizer_free': True,
        'full350_DEV_complete': True, 'split': 'validation', 'TEST_used': False,
        'checkpoint_pointer': request['checkpoint_pointer'], 'scorer_policy': policy,
        'predictions': {language: record(result_folder/('validation_'+language+'.jsonl')) for language in REFERENCES},
        'scores': scores, 'source_request': request_pin, 'owned_worker_admission': owned_worker_admission,
        'runtime_completion_path': str(folder/'worker_completion.json'),
        'implementation': record(__file__),
        'producer': producer, 'worker_source': request['execution_sources']['worker_source'],
        'parent_source': request['execution_sources']['parent_source'], 'dependencies': sources(),
        'all207_weight_only_restore_byte_exact': True, 'all207_after_DEV_hashes_unchanged': True,
        'selective_weight_RNG_mapping': selection, 'Adam_tensor_views_created': 0,
        'selected_saved_CUDA_RNG_states': len(rng[3]),
        'weight_hashes': weight_hashes, 'optimizer_objects_constructed_or_loaded_in_DEV_worker': False,
        'actual_native_updates_in_DEV': 0, 'all4_child_RNG_restored_exactly': True,
        'sampler_replay_and_augmentation_objects_created_or_advanced': False,
        'saved_sampler_state_sha256': pointer['sampler_state_sha256'], 'saved_journal_prefix': pointer['journal_prefix'],
        'saved_source_clocks': [clock+request['optimizer_updates'] for clock in (64250, 2000, 62250)],
        'original_evaluator_result_folder': str(result_folder), 'native_checkpoint_actual0': request['native_checkpoint_actual0'],
        'production_recovery_scope':'current_latest_only_with_verified_persistence_prefix',
        'historical_fresh_GPU_restore_claimed':False,
        'production_recovery_boundary':read_bound(request['native_checkpoint_actual0'])['production_recovery_boundary'],
        'training_shortened_or_winner_promoted': False, 'production_approval_claimed': False}
    check_optimizer_free_report_protocol(report)
    report['official_saved_text_rescore'] = official_DEV(report, request['checkpoint_pointer'], policy)
    return write_new(folder/'serial350_worker_report.json', report)


def verified_completed_DEV(request_pin, worker_report_pin, actual_process0_pin):
    """Verify real wait0 without publishing or executing anything."""
    from fastmss_owned_process_closure_v2 import verify
    request = read_bound(request_pin); policy, pointer, _, _ = checked_request(request)
    worker = read_bound(worker_report_pin); process = read_bound(actual_process0_pin)
    check_optimizer_free_report_protocol(worker)
    verify(process, kind='ASR18_owned_GPU_process_completion_v1', owner=OWNER, report=worker_report_pin)
    require(same(process['worker_source'], request['execution_sources']['worker_source'])
            and same(process['parent_source'], request['execution_sources']['parent_source'])
            and same(process['guard_source'], request['execution_sources']['guard_source'])
            and same(worker['source_request'], request_pin)
            and same(worker['implementation'], record(__file__)) and same(worker['dependencies'], sources())
            and worker['all207_weight_only_restore_byte_exact'] is True
            and worker['all207_after_DEV_hashes_unchanged'] is True
            and worker['all4_child_RNG_restored_exactly'] is True
            and worker['optimizer_objects_constructed_or_loaded_in_DEV_worker'] is False
            and type(worker['actual_native_updates_in_DEV']) is int and worker['actual_native_updates_in_DEV'] == 0
            and worker['sampler_replay_and_augmentation_objects_created_or_advanced'] is False
            and worker['saved_sampler_state_sha256'] == pointer['sampler_state_sha256']
            and same(worker['saved_journal_prefix'], pointer['journal_prefix']),
            'Actual source-bound optimizer-free all207/full-RNG/unchanged prefix completion required')
    boot = read_bound(worker['owned_worker_admission'])
    require(boot['kind'] == 'ASR18_sole_owned_optimizer_free_serialDEV_worker_admission_v1'
            and boot['owner'] == OWNER and boot['sole_owned_GPU_child'] is True
            and boot['actual_training_optimizer_running'] is False
            and boot['optimizer_objects_constructed_or_loaded_in_DEV_worker'] is False
            and same(boot['producer'], process['producer']) and same(boot['request'], request_pin)
            and same(boot['worker_source'], process['worker_source'])
            and same(boot['parent_source'], process['parent_source']),
            'Actual DEV admission/report/process identities differ')
    # The worker's inner report exists before final model release/phase closure;
    # actual0 plus this separate runtime completion is required before sealing.
    runtime_path = Path(worker['runtime_completion_path']).resolve()
    require(runtime_path == Path(worker_report_pin['path']).resolve().parent/'worker_completion.json',
            'Exact owned runtime completion path')
    runtime_pin = record(runtime_path); runtime = read_bound(runtime_pin)
    from asr18_full_run_DEV_runtime_9223_v2 import resource_proposal
    require(runtime['kind'] == 'ASR18_optimizer_free_serial350DEV_runtime_completion_v1'
            and runtime['optimizer_free_protocol'] == OPTIMIZER_FREE_PROTOCOL
            and runtime['passed'] is True and same(runtime['worker_report'],worker_report_pin)
            and same(runtime['source'], request['execution_sources']['worker_source'])
            and same(runtime['parent_source'], request['execution_sources']['parent_source'])
            and same(runtime['request'],request_pin) and same(runtime['producer'],process['producer'])
            and same(runtime['resource_policy'],resource_proposal())
            and runtime['optimizer_objects_constructed'] == 0 and type(runtime['optimizer_objects_constructed']) is int
            and runtime['Adam_tensor_views_created'] == 0 and type(runtime['Adam_tensor_views_created']) is int
            and runtime['native_updates'] == 0 and type(runtime['native_updates']) is int
            and runtime['sampler_or_replay_objects_constructed_or_advanced'] is False,
            'Actual source-bound optimizer-free model-release/runtime completion required')
    phases = [read_bound(pin) for pin in runtime['phase_memory_records']]
    expected_budgets = {'optimizer_free_DEV_readonly_base':900_000_000,
        'optimizer_free_DEV_BPE_LoRA_structure':250_000_000,
        'full_run_DEV_weight_only_restore_and_hash':350_000_000,
        'full_run_optimizer_free_serial350DEV_original_route':2_100_000_000}
    require(len(phases) == 4 and [phase['name'] for phase in phases] == list(expected_budgets),
            'Every bootstrap/structure/restore/serial DEV phase is required')
    for phase in phases:
        observed = phase['actual']; before = integer(phase['before_private_bytes'],1)
        require(phase['covered'] is True and type(phase['increment_bytes']) is int
                and phase['increment_bytes'] == expected_budgets[phase['name']]
                and before+phase['increment_bytes'] <= resource_proposal()['whole_worker_private_forecast_bytes']
                and observed['joined'] is True and observed['error'] is None
                and integer(observed['available_sample_count'],3) >= 3
                and integer(observed['private_peak_bytes'],1) <= before+phase['increment_bytes'],
                'Measured clean phase resource proof required before DEV completion')
    score = official_DEV(worker, request['checkpoint_pointer'], policy)
    require(same(score, worker['official_saved_text_rescore']), 'Saved complete350 scores differ on independent rescore')
    sealed = deepcopy(worker)
    sealed.update(actual_owned_DEV_process0=actual_process0_pin, actual_worker_report=worker_report_pin,
                  actual_runtime_completion=runtime_pin,
                  actual_DEV_owner_verified=True, official_saved_text_rescore=score)
    return sealed


def seal_completed_DEV(request_pin, worker_report_pin, actual_process0_pin, output):
    """Owner parent callback AFTER real wait0; report existence is insufficient."""
    return write_new(output, verified_completed_DEV(request_pin, worker_report_pin, actual_process0_pin))


class SerialDEVCallbacks:
    """Lifecycle callbacks for a NEW separately reviewed owner executor; no launch defaults.

    execute_owned(request_pin) must be a NEW reviewed native-owner implementation
    returning {'worker_report', 'actual_process0'}. No default launch path exists.
    native_boundary(pointer_pin) returns the exact completed native0 evidence.
    Both are source bound. The future parent still durably checkpoints lifecycle
    acknowledgement/winner state before another pass; this adapter cannot do so.
    """
    def __init__(self, *, ROOT_policy_review, execution_sources, execute_owned, execute_owned_source,
                 native_boundary, native_boundary_source, native_boundary_sources, output_directory):
        self.review = deepcopy(ROOT_policy_review); self.execution = deepcopy(execution_sources)
        self.execute, self.native_boundary = execute_owned, native_boundary
        self.boundary_sources = deepcopy(native_boundary_sources)
        self.output = new_E(output_directory)
        require(self.output.is_dir(), 'Explicit fresh future DEV namespace required')
        for function, pin in ((execute_owned, execute_owned_source), (native_boundary, native_boundary_source)):
            bound(pin); require(callable(function) and str(Path(inspect.getsourcefile(function)).resolve()) == pin['path'],
                                'Exact source-bound future native-owner callbacks required')
        require(same(execute_owned_source, execution_sources['parent_source'])
                and same(native_boundary_source, execution_sources['parent_source']),
                'The actual native owner parent owns launch and prior native checkpoint verification')

    def serial_DEV(self, lifecycle_request):
        serial(); request = deepcopy(lifecycle_request)
        request.update(kind=REQUEST_KIND, owner=OWNER, production_enabled=False, TEST_used=False,
            native_updates_in_DEV=0, scorer_policy=request['identity']['scorer_policy'],
            ROOT_policy_review=self.review, execution_sources=self.execution, dependencies=sources(),
            implementation=record(__file__), native_boundary_sources=self.boundary_sources, native_checkpoint_actual0=self.native_boundary(request['checkpoint_pointer']))
        # The executor gets the pre-worker request and creates a SEPARATE bound
        # live worker-admission record; it must not mutate this immutable request.
        # That record is passed explicitly to evaluate_boundary_checkpoint below.
        directory = self.output/request['action']['boundary']['name']
        directory.mkdir(exist_ok=False)
        checked_request(request)
        request_pin = write_new(directory/'request.json', request)
        result = self.execute(request_pin)
        return seal_completed_DEV(request_pin, result['worker_report'], result['actual_process0'], directory/'complete_serial350DEV.json')

    def verify_DEV(self, lifecycle_request):
        serial(); report = read_bound(lifecycle_request['report'])
        check_optimizer_free_report_protocol(report)
        request_pin = report['source_request']; request = read_bound(request_pin)
        require(same(request['checkpoint_pointer'], lifecycle_request['checkpoint_pointer'])
                and same(request['identity'], lifecycle_request['identity'])
                and same(request['action']['boundary'], lifecycle_request['boundary'])
                and lifecycle_request['all207_weights_and_all_source_clocks_unchanged_required'] is True
                and lifecycle_request['all4RNG_and_replay_queues_unchanged_required'] is True,
                'Exact lifecycle boundary/source/prefix verification required')
        # Do not publish another report or launch another evaluator on verification.
        expected = verified_completed_DEV(request_pin, report['actual_worker_report'], report['actual_owned_DEV_process0'])
        require(same(expected, report), 'Sealed actual complete350 DEV report differs')
        return True
