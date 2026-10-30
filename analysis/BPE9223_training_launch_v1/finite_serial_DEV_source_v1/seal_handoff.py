"""Small inactive source/request handoff; no execution admission."""
import difflib,json,sys
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish'
sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import record
from fastmss_BPE9223_cpu_common_v3 import environment,resource_guard
environment();resource_guard()
import asr18_finite_BPE9223_serial_DEV_v1 as finite
import finite_BPE9223_DEV_runtime_v1 as runtime
from asr18_finite_BPE9223_checkpoint_evidence_v1 import KIND,TOOL_KIND,RUNNER
O=Path(__file__).parent
report=record(O/'fixture_report.json');guard=record(O/'fixture_guard_01/closure.json')
g=json.loads(Path(guard['path']).read_text());assert g['actual_exit_code']==0 and g['monitor_clean'] and g['monitor_joined']
actual={'kind':'ASR18_source_bound_actual_tool_exit_v1','owner':'/root/cpu_data_preparation',
    'actual_exit_code':0,'actual_exec_tool_exit_code':0,'actual_tool_chunk_id':'f9e476',
    'native_PID':28568,'published_marker':report,'guard_closure':guard,'source':record(O/'fixture.py'),
    'scope':'35 synthetic stdlib metadata/source controls only; no real process/payload evidence invented',
    'actual_audio_model_GPU_or_checkpoint_payload_jobs':0}
target=O/'actual_fixture_CLOSED0.json';assert not target.exists();target.write_text(json.dumps(actual,indent=2),encoding='utf-8')
source_names=('asr18_finite_BPE9223_serial_DEV_v1.py','finite_BPE9223_DEV_runtime_v1.py','asr18_finite_BPE9223_checkpoint_evidence_v1.py')
sources={n:record(S/n) for n in source_names}
for old,new in [('asr18_diagnostic_serial_DEV_9223_v1.py',source_names[0]),('asr18_diagnostic_DEV_runtime_9223_v1.py',source_names[1])]:
    diff=''.join(difflib.unified_diff((S/old).read_text().splitlines(True),(S/new).read_text().splitlines(True),fromfile=old,tofile=new))
    (O/(new+'.patch')).write_text(diff,encoding='utf-8')
request_example={'kind':finite.REQUEST_KIND,'owner':finite.OWNER,'production_enabled':False,'diagnostic_only':True,'qualified':False,
    'action':{'kind':'diagnostic_checkpoint_then_serial350DEV','boundary':{'kind':'finite20_native_exit_and_fresh_checkpoint_recovered','optimizer_updates':20}},
    'split':'validation','clips':350,'TEST_used':False,'native_updates_in_DEV':0,'optimizer_updates':20,
    'identity':{'kind':'ASR18_actual_finite_BPE9223_diagnostic_DEV_identity_v1','native_owner':finite.OWNER,
        'source_optimizer_RNG_sampler_reset':False,'source_original_clocks':[64250,2000,62250],
        'scorer_policy':'EXACT_SCORER_POLICY_RECORD','source_report':'ACTUAL_overall_result_RECORD',
        'native_configuration':'EXACT_runner2_CONFIGURATION_RECORD','native_parent_source':record(RUNNER),
        'owner_tool_actual0':'ACTUAL_OWNER_TOOL0_RECORD'},
    'scorer_policy':'EXACT_SCORER_POLICY_RECORD','ROOT_policy_review':'EXACT_NEW_ROOT_POLICY_REVIEW_RECORD',
    'execution_sources':{'worker_source':sources[source_names[1]],'parent_source':'EXACT_NEW_DEV_OWNER_PARENT_SOURCE_RECORD',
        'guard_source':'EXACT_OWNED_DEV_GUARD_SOURCE_RECORD','bootstrap_source':record(S/'asr18_optimizer_free_DEV_structure_9223_v1.py')},
    'checkpoint_pointer':'ACTUAL_FINITE_PACKET_RECORD_WITH_OWN_DISTINCT_KIND',
    'native_checkpoint_actual0':'ACTUAL_OWNER_TOOL0_RECORD',
    'native_boundary_sources':{'boundary_verifier':sources[source_names[2]],'native_worker':record(RUNNER),
        'native_parent':record(RUNNER),'recovery_worker':record(RUNNER),'recovery_parent':record(RUNNER)},
    'dependencies':finite.sources(),'implementation':sources[source_names[0]]}
handoff={'kind':'ASR18_finite_BPE9223_serial350DEV_inactive_source_handoff_v1','owner':'/root/cpu_data_preparation',
    'sources':sources,'baselines':{n:record(S/n) for n in ('asr18_diagnostic_serial_DEV_9223_v1.py','asr18_diagnostic_DEV_runtime_9223_v1.py')},
    'fixture_report':report,'fixture_actual0':record(target),'resource_policy':runtime.resource_proposal(),
    'request_example_IS_NOT_EXECUTABLE':request_example,
    'policy_review_kind':'ROOT_finite_BPE9223_optimizer_free_serial350DEV_policy_and_sources_review_v1',
    'policy_review_fields':['accepted=True','reviewer=/root','canonical_fixed_DEV_manifests_independently_verified=True',
        'scorer_policy','diagnostic_source_identity=exact request.identity','implementation','dependencies','execution_sources',
        'canonical_DEV_manifests','native_boundary_sources'],
    'finite_packet_kind':KIND,'packet_builder':'asr18_finite_BPE9223_checkpoint_evidence_v1.build_binding(*,configuration,overall_result,owner_tool_actual0) -> candidate dict; publish new E file then checked_binding(record(file))',
    'packet_authority':'finite20 recovered checkpoint only; checkpoint_pointer request field carries this OWN kind, never production pointer/head/ACK or pass qualification',
    'owner_tool_completion':{'kind':TOOL_KIND,'owner':finite.OWNER,'actual_native_exit_code':0,'actual_exec_tool_exit_code':0,
        'native_PID':'ACTUAL_PARENT_PID','producer':'ACTUAL_PARENT_pid_ppid_create_time','actual_tool_chunk_id':'ACTUALLY_OBSERVED_TOOL_CHUNK',
        'report':'ACTUAL_overall_result_RECORD','source':record(RUNNER),'configuration':'EXACT_CONFIGURATION_RECORD','guard_closure':'ACTUAL_PARENT_GUARD_RECORD'},
    'deltas':['Only finite checkpoint/native evidence/request validators replaced; all durable-current validators removed',
        'Exact new finite request/report/policy kind, source imports and fresh output tag',
        '17 complete evaluator/restore/RNG/report/scorer/helper AST bodies unchanged after declared module/tag substitutions',
        'Entire runtime AST identical after two module-name substitutions; budgets/207 restore/all4 RNG/zeroAdam/7x50 scorer unchanged'],
    'remaining':['ROOT exact source/policy/resource/config review','real runner2 native/fresh recovery/overall+owner TOOL0',
        'new owner DEV parent/config/request assembly and fresh sole-worker admission before any execution'],
    'execution_admitted':False,'actual_native_or_DEV_jobs':0,'production_admission_or_winner_promotion':False,
    'original_production_checker_unchanged':True}
target=O/'source_handoff_v1.json';assert not target.exists();target.write_text(json.dumps(handoff,indent=2),encoding='utf-8')
print(json.dumps(record(target)),flush=True)
