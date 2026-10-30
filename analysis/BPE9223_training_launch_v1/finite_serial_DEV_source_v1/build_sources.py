"""One-time source-only clone; frozen ancestors are read, never edited."""
import ast
from pathlib import Path

P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
S=P/'outputs/parakeet_turkish'

def transform_functions(source,replacements,removed=()):
    tree=ast.parse(source);spans=[];lines=source.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in set(replacements)|set(removed):
            spans.append((node.lineno-1,node.end_lineno,replacements.get(node.name,'')))
    assert len(spans)==len(set(replacements)|set(removed))
    for start,end,value in sorted(spans,reverse=True):lines[start:end]=[value+'\n']
    return ''.join(lines)

serial=(S/'asr18_diagnostic_serial_DEV_9223_v1.py').read_text(encoding='utf-8')
serial=serial.replace('asr18_diagnostic_DEV_runtime_9223_v1','finite_BPE9223_DEV_runtime_v1')
serial=serial.replace("REQUEST_KIND = 'ASR18_diagnostic_recovered_pipeline_serial350DEV_request_v1'", "REQUEST_KIND = 'ASR18_finite_BPE9223_recovered_serial350DEV_request_v1'")
serial=serial.replace("REPORT_KIND = 'ASR18_diagnostic_optimizer_free_serial350DEV_report_v1'", "REPORT_KIND = 'ASR18_finite_BPE9223_optimizer_free_serial350DEV_report_v1'")
serial=serial.replace("'ROOT_diagnostic_optimizer_free_serial350DEV_policy_and_sources_review_v1'", "'ROOT_finite_BPE9223_optimizer_free_serial350DEV_policy_and_sources_review_v1'")
serial=serial.replace("'asr18_full_run_serial_DEV_9223_v2.py',", "'asr18_full_run_serial_DEV_9223_v2.py', 'asr18_finite_BPE9223_checkpoint_evidence_v1.py',")
serial=serial.replace("tag = 'ASR18_diagnostic_9223_DEV_'", "tag = 'ASR18_finite_9223_DEV_'")
serial=transform_functions(serial,{
'checked_checkpoint':'''def checked_checkpoint(pointer_pin, *, identity, expected_updates, boundary):
    from asr18_finite_BPE9223_checkpoint_evidence_v1 import checked_binding
    packet,metadata,sampler=checked_binding(pointer_pin)
    require(type(expected_updates) is int and expected_updates==20
        and packet['optimizer_updates']==expected_updates
        and identity['kind']=='ASR18_actual_finite_BPE9223_diagnostic_DEV_identity_v1'
        and same(identity['native_configuration'],packet['configuration'])
        and same(identity['native_parent_source'],packet['native_source'])
        and same(identity['source_report'],packet['overall_result'])
        and same(identity['owner_tool_actual0'],packet['owner_tool_actual0'])
        and boundary['kind']=='finite20_native_exit_and_fresh_checkpoint_recovered'
        and type(boundary['optimizer_updates']) is int and boundary['optimizer_updates']==expected_updates,
        'Honest finite20 packet/config/source/TOOL0; never a production checkpoint pointer')
    return packet,metadata,sampler''',
'checked_native_boundary':'''def checked_native_boundary(pin, *, request, pointer):
    from asr18_finite_BPE9223_checkpoint_evidence_v1 import checked_owner
    pins=request['native_boundary_sources']
    require(type(pins) is dict and set(pins)=={'boundary_verifier','native_worker','native_parent','recovery_worker','recovery_parent'},
        'Exact finite checkpoint/native/recovery evidence sources, no durable authority')
    for source in pins.values():bound(source)
    require(same(pins['boundary_verifier'],record(HERE/'asr18_finite_BPE9223_checkpoint_evidence_v1.py'))
        and all(same(pins[key],pointer['native_source']) for key in ('native_worker','native_parent','recovery_worker','recovery_parent'))
        and same(pin,pointer['owner_tool_actual0'])
        and same(read_bound(request['ROOT_policy_review'])['native_boundary_sources'],pins),
        'Exact separately ROOT-pinned finite native evidence and owner TOOL0')
    return checked_owner(read_bound(pin),source=pointer['native_source'],configuration=pointer['configuration'],report=pointer['overall_result'])''',
'checked_request':'''def checked_request(request):
    serial()
    require(request['kind']==REQUEST_KIND and request['owner']==OWNER
        and request['production_enabled'] is False and request['diagnostic_only'] is True
        and request['qualified'] is False and request['action']['kind']=='diagnostic_checkpoint_then_serial350DEV'
        and request['split']=='validation' and type(request['clips']) is int and request['clips']==350
        and request['TEST_used'] is False and integer(request['native_updates_in_DEV'])==0,
        'Explicit unqualified finite20 optimizer-free serial350 request')
    identity=request['identity']
    require(identity['kind']=='ASR18_actual_finite_BPE9223_diagnostic_DEV_identity_v1'
        and identity['native_owner']==OWNER and identity['source_optimizer_RNG_sampler_reset'] is False
        and same(identity['source_original_clocks'],[64250,2000,62250])
        and same(identity['scorer_policy'],request['scorer_policy']),
        'Exact finite original-source qualification identity, no durable/full-pass claim')
    for name in ('source_report','native_configuration','native_parent_source','owner_tool_actual0'):bound(identity[name])
    policy=checked_policy(request['scorer_policy'],request['ROOT_policy_review'],identity=identity,execution=request['execution_sources'])
    packet,metadata,sampler=checked_checkpoint(request['checkpoint_pointer'],identity=identity,
        expected_updates=integer(request['optimizer_updates'],1),boundary=request['action']['boundary'])
    checked_native_boundary(request['native_checkpoint_actual0'],request=request,pointer=packet)
    require(same(request['dependencies'],sources()) and same(request['implementation'],record(__file__)),
        'Exact new finite adapter/dependencies')
    return policy,packet,metadata,sampler'''
},removed=('read_logical_durable_head','_checked_current_native_boundary'))
serial=serial.replace('Explicitly diagnostic optimizer-free serial350 DEV adapter, never full-run authority.',
    'Finite20 optimizer-free serial350 DEV adapter, never durable/full-run authority.')
runtime=(S/'asr18_diagnostic_DEV_runtime_9223_v1.py').read_text(encoding='utf-8')
runtime=runtime.replace('asr18_diagnostic_serial_DEV_9223_v1','asr18_finite_BPE9223_serial_DEV_v1')
runtime=runtime.replace('asr18_diagnostic_DEV_runtime_9223_v1','finite_BPE9223_DEV_runtime_v1')
for name,value in [('asr18_finite_BPE9223_serial_DEV_v1.py',serial),('finite_BPE9223_DEV_runtime_v1.py',runtime)]:
    target=S/name
    assert not target.exists(),str(target)
    ast.parse(value)
    target.write_text(value,encoding='utf-8',newline='\n')
    print(str(target))
