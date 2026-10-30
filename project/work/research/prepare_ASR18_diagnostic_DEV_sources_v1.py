"""Source-only derivation of an explicitly diagnostic DEV adapter/runtime.

Never edits the production DEV6/runtime3 primitives or executes ML work.
"""
import ast
from pathlib import Path

P = Path(__file__).resolve().parents[2]
S = P/'outputs/parakeet_turkish'


def replace_function(text, name, replacement):
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = text.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+replacement.strip()+'\n'+''.join(lines[node.end_lineno:])


def publish(name, text):
    target = S/name
    assert not target.exists(), target
    ast.parse(text)
    target.write_text(text, encoding='utf-8', newline='\n')


dev = (S/'asr18_full_run_serial_DEV_v6.py').read_text(encoding='utf-8')
dev = dev[:dev.index('\nclass SerialDEVCallbacks:')]
dev = dev.replace('Inactive concrete serial350 DEV glue for the full-run lifecycle.',
                  'Explicitly diagnostic optimizer-free serial350 DEV adapter, never full-run authority.')
dev = dev.replace("REQUEST_KIND = 'ASR18_full_run_recovered_boundary_serial350DEV_request_v2'",
                  "REQUEST_KIND = 'ASR18_diagnostic_recovered_pipeline_serial350DEV_request_v1'")
dev = dev.replace("REPORT_KIND = 'ASR18_serial_complete350_boundary_DEV_v1'",
                  "REPORT_KIND = 'ASR18_diagnostic_optimizer_free_serial350DEV_report_v1'")
dev = dev.replace("'asr18_full_run_DEV_runtime_v3.py',", "'asr18_diagnostic_DEV_runtime_v1.py', 'asr18_full_run_serial_DEV_v6.py',\n    'asr18_deferred_GPU_recovery_v1.py', 'asr18_temporal_tiled_subsampling_v2.py',")
dev = replace_function(dev, 'checked_policy', r'''
def checked_policy(policy_pin, ROOT_review_pin, *, identity, execution):
    policy = read_bound(policy_pin); review = read_bound(ROOT_review_pin)
    require(review['kind'] == 'ROOT_diagnostic_optimizer_free_serial350DEV_policy_and_sources_review_v1'
            and review['accepted'] is True and review['reviewer'] == '/root'
            and review['canonical_fixed_DEV_manifests_independently_verified'] is True
            and same(review['scorer_policy'], policy_pin)
            and same(review['diagnostic_source_identity'], identity)
            and same(review['implementation'], record(__file__))
            and same(review['dependencies'], sources())
            and same(review['execution_sources'], execution), 'Exact diagnostic policy/source review required')
    require(type(execution) is dict and set(execution) == EXECUTION_KEYS,
            'Exact diagnostic worker/parent/guard/bootstrap sources')
    for pin in execution.values(): bound(pin)
    require(same(policy['official_scorer'],record(HERE/'metrics.py'))
            and policy['official_scorer']['sha256'] == SCORER_SHA
            and same(policy['integer_gate_contract'],record(HERE/'tsc_bpe_full_decoder_transition_v1.py'))
            and policy['integer_gate_contract']['sha256'] == GATE_SHA
            and same(policy['reference_words'],REFERENCES) and same(policy['retention_limits'],LIMITS)
            and same(review['canonical_DEV_manifests'],policy['DEV_manifests'])
            and set(policy['DEV_manifests']) == set(REFERENCES), 'Immutable official scorer and fixed seven-language DEV')
    for language,path in canonical_DEV_paths().items():
        pin=policy['DEV_manifests'][language]; bound(pin)
        require(pin['path'] == str(path.resolve()), 'Original canonical evaluator data route')
        rows=[json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        require(len(rows)==50 and len({row['source_id'] for row in rows})==50
                and all(row['language']==language and row['dataset_split']=='validation' for row in rows),
                'Exactly50 original DEV rows/language, never TEST')
    return policy
''')
dev = replace_function(dev, 'checked_checkpoint', r'''
def checked_checkpoint(pointer_pin, *, identity, expected_updates, boundary):
    pointer=read_bound(pointer_pin)
    require(pointer['kind']=='ASR18_production_window_completed_checkpoint_pointer_v2'
            and pointer['diagnostic_only'] is True
            and pointer['writer_returned_and_payload_verified'] is True
            and pointer['all_window_updates_committed'] is True
            and integer(pointer['committed_updates'],1)==expected_updates,
            'Exact actual diagnostic pipeline checkpoint, never promoted full-run state')
    metadata=read_bound(pointer['checkpoint']); checkpoint_format_view(metadata)
    require(same(metadata['payload'],pointer['payload'])
            and Path(pointer['payload']['path']).resolve()==Path(pointer['checkpoint']['path']).resolve().with_suffix('.tensors.bin')
            and same(record(pointer['payload']['path']),pointer['payload']), 'Complete immutable payload integrity')
    scalar=lambda *path: decode_scalar_tree(select(metadata['tree'],*path))
    saved_identity,sampler=scalar('identity'),scalar('sampler')
    require(same(saved_identity,pointer['identity'])
            and same(saved_identity['caller_identity']['session'],identity['native_session'])
            and same(saved_identity['caller_identity']['engine'],identity['native_engine_source']),
            'Exact original pipeline source/session/engine identity')
    semantic=verify_saved_metadata(metadata,identity=saved_identity,optimizer_updates=expected_updates,
        sampler_state=sampler,journal_prefix=pointer['journal_prefix'],diagnostic_only=True)
    require(same(semantic,pointer['saved_semantic_state_verified'])
            and digest(sampler)==pointer['sampler_state_sha256']
            and scalar('optimizer','updates')==expected_updates
            and scalar('optimizer','clocks')==[clock+expected_updates for clock in (64250,2000,62250)]
            and boundary['kind']=='diagnostic_pipeline_native_exit_and_every_capture_recovered'
            and integer(boundary['optimizer_updates'],1)==expected_updates,
            'Saved diagnostic clocks, full sampler and journal prefix must agree')
    return pointer,metadata,sampler
''')
dev = replace_function(dev, 'checked_native_boundary', r'''
def checked_native_boundary(pin, *, request, pointer):
    from fastmss_owned_process_closure_v2 import verify
    from asr18_deferred_GPU_recovery_v1 import verified_requests
    evidence=read_bound(pin); identity=request['identity']
    require(evidence['kind']=='ASR18_fullpopulation_firstwindow20_conv_operator_observed_TOOL_CLOSED0_v1'
            and evidence['owner']==OWNER and evidence['actual_tool_exit_code']==0
            and evidence['actual_tool_completion_observed'] is True
            and type(evidence['actual_tool_chunk_id']) is str and bool(evidence['actual_tool_chunk_id'])
            and evidence['production_enabled'] is False and evidence['qualified'] is False
            and same(evidence['report'],identity['source_report']), 'Actual owner-observed pipeline TOOL0 required')
    report=read_bound(evidence['report'])
    require(report['passed'] is True and report['production_enabled'] is False
            and report['qualified_checkpoint'] is False
            and integer(report['actual_native_updates'],1)==request['optimizer_updates']
            and report['all_deferred_recovery_obligations_discharged'] is True,
            'Successful diagnostic native+every-checkpoint recovery pipeline required')
    requests=verified_requests(identity['deferred_recovery_requests'],session=identity['native_session'],
        parent_source=identity['native_parent_source'],guard=lambda:None)
    captures=[capture for window in report['windows'] for capture in window['checkpoints']]
    require(same(captures,requests['checkpoints']) and same(captures[-1]['pointer'],request['checkpoint_pointer'])
            and requests['expected_global_new_updates']==request['optimizer_updates'], 'Exact complete capture chain/final pointer')
    for process_pin in requests['native_process_completions']:
        process=read_bound(process_pin); verify(process,kind='ASR18_owned_GPU_process_completion_v1',owner=OWNER)
        require(same(process['worker_source'],identity['native_parent_source'])
                and same(process['parent_source'],identity['native_parent_source']), 'Frozen actual native source')
        native=read_bound(process['report'])
        require(native['passed'] is True and native['production_enabled'] is False
                and native['qualified_checkpoint'] is False and len(native['windows'])>=1
                and integer(native['actual_native_updates_this_process'],1)>=1
                and native['all_exact_GPU_checkpoint_restores_deferred'] is True, 'Actual training process0 report')
    recovery_pin=report['all_captured_checkpoints_actual_fresh_GPU_recovery']
    require(same(evidence['actual_recovery_process0'],report['actual_recovery_process0'])
            and same(evidence['all_captured_checkpoint_recovery'],recovery_pin), 'Tool0 binds exact recovery evidence')
    process=read_bound(report['actual_recovery_process0'])
    verify(process,kind='ASR18_owned_GPU_process_completion_v1',owner=OWNER)
    require(same(process['worker_source'],identity['native_parent_source'])
            and same(process['parent_source'],identity['native_parent_source']), 'Frozen actual recovery parent/entry source')
    worker=read_bound(process['report']); recovery=read_bound(recovery_pin)
    require(worker['passed'] is True and worker['worker_mode']=='restore_only'
            and same(worker['recovery_report'],recovery_pin)
            and worker['all_captured_checkpoints_actually_recovered'] is True
            and recovery['kind']=='ASR18_actual_all_deferred_checkpoints_fresh_GPU_recovery_v1'
            and recovery['passed'] is True and recovery['all_captured_checkpoints_recovered'] is True
            and same(recovery['request'],identity['deferred_recovery_requests'])
            and same(recovery['native_process_completions'],requests['native_process_completions'])
            and integer(recovery['actual_global_native_updates'],1)==request['optimizer_updates']
            and integer(recovery['actual_native_updates_in_recovery_process'])==0
            and integer(recovery['actual_draws_consumed_in_recovery'])==0
            and len(recovery['recovered_checkpoints'])==len(captures), 'Actual separate ALL-capture recovery0')
    for capture,recovered_pin in zip(captures,recovery['recovered_checkpoints']):
        recovered=read_bound(recovered_pin)
        require(recovered['passed'] is True and same(recovered['pointer'],capture['pointer'])
                and same(recovered['checkpoint'],capture['checkpoint'])
                and same(recovered['pre_save_fingerprint'],capture['pre_save_fingerprint'])
                and same(recovered['next_draw_packet'],capture['next_draw_packet'])
                and recovered['all207_threeAdam_all4RNG_match_actual_pre_save'] is True
                and recovered['both_samplers_journal_prefix_and_next_draw_exact'] is True
                and integer(recovered['native_updates_performed'])==0
                and integer(recovered['draws_consumed'])==0 and recovered['separate_fresh_GPU_recovery'] is True,
                'Exact current diagnostic checkpoint all-state recovery cannot be inferred from persistence')
    require(same(pointer,read_bound(captures[-1]['pointer'])),'Final recovered pointer identity')
    return evidence
''')
dev = replace_function(dev, 'checked_request', r'''
def checked_request(request):
    serial()
    require(request['kind']==REQUEST_KIND and request['owner']==OWNER
            and request['production_enabled'] is False and request['diagnostic_only'] is True
            and request['qualified'] is False and request['action']['kind']=='diagnostic_checkpoint_then_serial350DEV'
            and request['split']=='validation' and request['clips']==350 and request['TEST_used'] is False
            and integer(request['native_updates_in_DEV'])==0, 'Explicit unqualified diagnostic serial350 request')
    identity=request['identity']
    require(identity['kind']=='ASR18_actual_fullpopulation_pipeline_diagnostic_DEV_identity_v1'
            and identity['native_owner']==OWNER and identity['source_optimizer_RNG_sampler_reset'] is False
            and same(identity['source_original_clocks'],[64250,2000,62250])
            and same(identity['scorer_policy'],request['scorer_policy']), 'Exact diagnostic source identity, not full-run fiction')
    for name in ('source_report','native_session','native_engine_source','native_parent_source','deferred_recovery_requests'):
        bound(identity[name])
    policy=checked_policy(request['scorer_policy'],request['ROOT_policy_review'],identity=identity,execution=request['execution_sources'])
    pointer,metadata,sampler=checked_checkpoint(request['checkpoint_pointer'],identity=identity,
        expected_updates=integer(request['optimizer_updates'],1),boundary=request['action']['boundary'])
    checked_native_boundary(request['native_checkpoint_actual0'],request=request,pointer=pointer)
    require(same(request['dependencies'],sources()) and same(request['implementation'],record(__file__)), 'Exact new adapter/dependencies')
    return policy,pointer,metadata,sampler
''')
dev = dev.replace("require(saved['diagnostic_only'] is False and len(weights) == 207", "require(saved['diagnostic_only'] is True and len(weights) == 207")
dev = dev.replace("tag = 'ASR18_full_run_DEV_'", "tag = 'ASR18_diagnostic_DEV_'")
dev = dev.replace("'passed': True, 'serial_optimizer_free': True,", "'passed': True, 'serial_optimizer_free': True, 'diagnostic_only': True, 'qualified': False,")
dev = dev.replace("require(report['kind'] == REPORT_KIND and report['optimizer_free_protocol'] == OPTIMIZER_FREE_PROTOCOL", "require(report['kind'] == REPORT_KIND and report['diagnostic_only'] is True and report['qualified'] is False\n            and report['optimizer_free_protocol'] == OPTIMIZER_FREE_PROTOCOL")
dev = dev.replace("official_DEV(report, request['checkpoint_pointer'], policy)", "diagnostic_saved_text_rescore(report, request['checkpoint_pointer'], policy)")
dev = dev.replace("official_DEV(worker, request['checkpoint_pointer'], policy)", "diagnostic_saved_text_rescore(worker, request['checkpoint_pointer'], policy)")
dev = dev.replace("from asr18_full_run_DEV_runtime_v3 import resource_proposal", "from asr18_diagnostic_DEV_runtime_v1 import resource_proposal")
dev = dev.replace("'ASR18_optimizer_free_serial350DEV_runtime_completion_v1'", "'ASR18_diagnostic_optimizer_free_serial350DEV_runtime_completion_v1'")
dev += '''\n\ndef diagnostic_saved_text_rescore(report, pointer, policy):
    """In-memory scorer-schema projection only, never a published full-run report."""
    require(report['kind']==REPORT_KIND and report['diagnostic_only'] is True and report['qualified'] is False,
            'Diagnostic source report must remain explicitly unqualified')
    scorer_input=dict(report,kind='ASR18_serial_complete350_boundary_DEV_v1')
    result=official_DEV(scorer_input,pointer,policy)
    require(report['kind']==REPORT_KIND and report['diagnostic_only'] is True,'Scoring cannot relabel the actual report')
    return result
'''
publish('asr18_diagnostic_serial_DEV_v1.py',dev)

runtime=(S/'asr18_full_run_DEV_runtime_v3.py').read_text(encoding='utf-8')
runtime=runtime.replace('asr18_full_run_serial_DEV_v6','asr18_diagnostic_serial_DEV_v1')
runtime=runtime.replace('asr18_full_run_DEV_runtime_v3.py','asr18_diagnostic_DEV_runtime_v1.py')
runtime=runtime.replace('ASR18_optimizer_free_serial350DEV_worker_configuration_v1','ASR18_diagnostic_optimizer_free_serial350DEV_worker_configuration_v1')
runtime=runtime.replace('ASR18_optimizer_free_serial350DEV_resource_proposal_v1','ASR18_diagnostic_optimizer_free_serial350DEV_resource_proposal_v1')
runtime=runtime.replace('ROOT_optimizer_free_serial350DEV_execution_configuration_review_v1','ROOT_diagnostic_optimizer_free_serial350DEV_execution_configuration_review_v1')
runtime=runtime.replace('ROOT_optimizer_free_serial350DEV_resource_review_v1','ROOT_diagnostic_optimizer_free_serial350DEV_resource_review_v1')
runtime=runtime.replace('ASR18_optimizer_free_serial350DEV_runtime_completion_v1','ASR18_diagnostic_optimizer_free_serial350DEV_runtime_completion_v1')
runtime=runtime.replace("and config['production_enabled'] is False", "and config['production_enabled'] is False and config['diagnostic_only'] is True")
runtime=runtime.replace("value = sample_memory(os.getpid(), label)", "value = stage(folder,label,torch)")
runtime=runtime.replace("require(measured <= proposal['whole_worker_private_forecast_bytes'], 'Whole DEV-worker forecast exceeded')", "require(measured <= proposal['whole_worker_private_forecast_bytes']\n                and value['worker_private_peak_bytes_lifetime'] <= proposal['whole_worker_private_forecast_bytes'],\n                'Actual current AND OS lifetime whole-worker cap')")
runtime=runtime.replace("joint = canonical_joint(model.joint)", "joint = canonical_joint(model.joint)\n        from asr18_temporal_tiled_subsampling_v2 import apply as conv_only\n        conv = conv_only(model,torch=torch,guard=guard)")
runtime=runtime.replace("finish_phase(); check_bindings(config)", "finish_phase(); check_bindings(config)\n        gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.synchronize()\n        require(torch.cuda.memory_reserved() <= 3_000_000_000,'Coherent completed evaluation endcache<=3GB')\n        emit('diagnostic_DEV_evaluation_coherent_endcache')")
runtime=runtime.replace("model = processor = None; gc.collect(); torch.cuda.empty_cache()", "model = processor = None; gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.synchronize()\n        require(torch.cuda.memory_reserved() <= 3_000_000_000,'Final model-release endcache<=3GB')")
runtime=runtime.replace("'structure':structure,'loader':loader,'logical_device':logical,'canonical_joint':joint,", "'structure':structure,'loader':loader,'logical_device':logical,'canonical_joint':joint,'conv_only_CNN':conv,\n        'diagnostic_only':True,'qualified':False,")
runtime=runtime.replace("'metrics.py','tdt_joint_canonical_2d_fp32_v2.py'):", "'metrics.py','tdt_joint_canonical_2d_fp32_v2.py','asr18_temporal_tiled_subsampling_v2.py'):")
publish('asr18_diagnostic_DEV_runtime_v1.py',runtime)
