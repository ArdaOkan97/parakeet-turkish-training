"""Metadata/source preparation ONLY. Never imports or executes decoder/fixture."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish'
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1')
F=N/'resident_RAM_original_interval170469_failure_actual_v1'
I=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01/invocations/owner_000023')
PRIOR=N/'resident_RAM_recurrent_metadata_slowdown_fix_v1/thread_CPU_attribution_v1/launch_fixed_counter_controller_METHOD_STUB_v1.py'
FIXTURE=F/'observe_recorded170470_ordered14_decoder_v1.py'
LAUNCHER=F/'launch_recorded170470_ordered14_decoder_external_v1.py'
SCOPE='ASR18_source31_saved170400_ordered14_original_decoder_reproduction_CPU_v1'


def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    p=Path(path).resolve();data=p.read_bytes();require(len(data)<=8*1024**2,'Compact/source preparation only')
    return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def write(path,value):
    p=Path(path);require(not p.exists(),'Immutable fresh prepared file')
    p.write_bytes((json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'))


candidate_pin=pin(F/'actual_CPU_completed2469_plan154_index9_ordered14_candidates_v1.json')
candidate=read(candidate_pin['path']);config_pin=pin(I/'configuration.json');config=read(config_pin['path'])
plan_pin=candidate['recorded_target_plan'];require(pin(plan_pin['path'])==plan_pin,'Recorded plan exact')
plan=read(plan_pin['path']);update=plan['updates'][9]
require(update['global_update']==170470 and update['local_update_index']==9 and len(candidate['ordered14_original_TR_candidates'])==14,'Exact recorded candidate')
validated_pin=config['planning_arguments']['DATA_evidence']['original_DATA_evidence']['validated']
require(pin(validated_pin['path'])=={k:validated_pin[k]for k in('path','bytes','sha256')},'Closed foundation compact metadata exact')
validated=read(validated_pin['path']);inventory=validated['bindings']['source_identity_inventory'];heldout=validated['bindings']['heldout_PCM']
policies=[];policy=config['original_clip_geometry_overlay'];policy_pins=[]
while policy is not None:
    certificate=read(policy['certificate']['path']);raw=certificate['original_row']
    require(all(item['ordinal']!=raw['ordinal']and item['source_id']!=raw['unique_id']for item in candidate['ordered14_original_TR_candidates']),
        'All14 rows nonmatch exact seven overlay dispatch keys')
    policies.append({'source':policy['source'],'certificate':policy['certificate'],'ordinal':raw['ordinal'],'unique_id':raw['unique_id']})
    policy_pins.extend((policy['source'],policy['certificate'],policy['ROOT_review']));policy=policy.get('prior_overlay')
require(len(policies)==7,'Existing seven policies unchanged')
raw=config['raw_lifetime_review']
budgets=[]
for item in candidate['ordered14_original_TR_candidates']:
    d=item['full_original_plan_row']['declared_geometry'];c=d['clean16k_samples'];n=d['native_samples'];channels=1 if item['dataset']=='kiraat'else raw['maximum_runtime_channels'];frame=raw['maximum_decoder_frame_samples']
    hold=4*c;frame_raw=channels*frame*8*6+frame*4*2;decode=4*n+frame_raw+4*(c+128)+256*1024
    weak=update['phase']=='weak_label_continued_training'and c<=120*16000
    retained=hold+(hold if weak else 0);transform=32*c+1024**2 if weak else 0
    peak=max(hold+decode,hold+65536*8,retained+transform,retained+24*c+64*1024)
    fft=1<<(max(1,c+24000-1)-1).bit_length()
    workspace=max(8*1024**2,6*16*fft if weak else 0,(c//160+1)*257*8*10+8*1024**2)
    require(peak<=raw['raw_capacity_bytes']and workspace<=max(raw['decoder_workspace_allowance_bytes'],raw['transform_workspace_allowance_bytes'],raw['frontend_workspace_allowance_bytes']),
        'Original raw/workspace support, not audio correctness')
    budgets.append({'original_update_row_1based':item['original_update_row_1based'],'prospective_bytes':peak+33554432+workspace+raw['numerical_runtime_reserve_bytes']})
maximum=max(v['prospective_bytes']for v in budgets)
old=PRIOR.read_bytes();text=old.decode('utf-8');changes=[]
def replace(before,after,count=1):
    global text
    require(text.count(before)==count,'Exact replacement count: '+before[:100])
    text=text.replace(before,after);changes.append({'old_literal':before,'new_literal':after,'expected_count':count})
replace('One NEW finite thread-counter/nativeTID proof; no production attach; external full guards.',
    'ONE ordered14 original decoder reproduction, no frontend/model/GPU/production; external original full guards.')
replace("FIXTURE=F/'verify_fixed_counter_controller_METHOD_STUB_v1.py'","FIXTURE=F/'observe_recorded170470_ordered14_decoder_v1.py'")
replace("RESULT=F/'fixed_counter_controller_external_result_v1.json'","RESULT=F/'ordered14_decoder_external_result_v1.json'")
replace("OUTPUT=F/'fixed_counter_controller_actual_01'","OUTPUT=F/'ordered14_decoder_actual_01'")
replace('LIMIT=536870912;PREFETCH=268435456',f'LIMIT=8000000000;PREFETCH=268435456;PROSPECTIVE={maximum};CPU_RESERVE=2500000000')
replace("'ASR18_fixed60_counter_window_persistence_METHOD_STUB_CPU_v1'",repr(SCOPE))
replace("and review['fixture_source']==pin(FIXTURE)and review['launcher_source']==pin(__file__)",
    "and review['fixture_source']==pin(FIXTURE)and review['launcher_source']==pin(__file__)\n        and review['frozen_inputs']==pin(F/'ordered14_decoder_frozen_inputs_v1.json')\n        and review['input_pins']==json.loads((F/'ordered14_decoder_frozen_inputs_v1.json').read_text(encoding='utf-8'))['input_pins']")
replace("sample_memory(proc.pid,'finite_thread_counter_mapping_external_CPU')","sample_memory(proc.pid,'ordered14_original_decoder_external_CPU')")
replace('prospective=LIMIT if admission else 0','prospective=PROSPECTIVE if admission else 0')
replace("require(total<=LIMIT and time.monotonic()-began<120,'Sampled family512MiB/external120s')",
    "require(total+prospective<=LIMIT and time.monotonic()-began<240,'Sampled full family/private8GB/external240s; original raw prospective charged before launch')")
replace("system['commit_available_bytes']>=(4518435456+prospective if admission else 2000000000+PREFETCH)",
    "system['commit_available_bytes']>=(max(2000000000,CPU_RESERVE+prospective)+PREFETCH if admission else 2000000000+PREFETCH)")
replace("'Admission FULL225+prefetch+prospective512MiB; runtime originalfloors+FULLprefetch; E100GB'",
    "'Admission originalCPU2.5GBprospective+FULLprefetch+maximum original raw/media/workspace/numericalbudget; runtime original2GB/1.5GBfloors+FULLprefetch/E100GB'")
replace("stdout=F/'fixed_counter_controller_stdout.log';stderr=F/'fixed_counter_controller_stderr.log'",
    "stdout=F/'ordered14_decoder_stdout.log';stderr=F/'ordered14_decoder_stderr.log'")
replace("'ASR18_fixed_counter_window_METHOD_STUB_external_result_v1'","'ASR18_ordered14_original_decoder_external_result_v1'")
replace("'Preserve actual failed diagnostic; no automatic rerun'","'Preserve actual failed decoder observation; no automatic rerun'")
new=text.encode('utf-8');require(not LAUNCHER.exists(),'Fresh immutable launcher')
ast.parse(FIXTURE.read_bytes());ast.parse(new)
LAUNCHER.write_bytes(new)
inverse=text
for item in reversed(changes):
    require(inverse.count(item['new_literal'])==item['expected_count'],'Invert exact replacement')
    inverse=inverse.replace(item['new_literal'],item['old_literal'])
require(inverse.encode('utf-8')==old,'ENTIRE original external source byte inverse')
ledger=F/'ordered14_decoder_external_exact_raw_inverse_v1.json'
write(ledger,{'old':pin(PRIOR),'new':pin(LAUNCHER),'newline_normalization':'NONE; exact source LF bytes retained',
    'replacements':changes,'full_old_raw_byte_reconstructed':True})
source_paths=[S/name for name in('fastmss_resident_feature_stream_v11.py','fastmss_bounded_original_audio_v2.py',
    'fastmss_raw_lifetime_v1.py','fastmss_BPE9223_cpu_common_v12.py','fastmss_BPE9223_plan_bridge_v12.py',
    'fastmss_BPE9223_feature_engine_v12.py','bpe_broad_audio_v1.py','bpe_broad_audio_FLAC_records_v1.py',
    'bpe_broad_source_identity_v1.py','resource_watchdog_v2.py')]
source_paths += [P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v31.py']
values=[candidate_pin,config_pin,plan_pin,candidate['origin_plan'],candidate['CPU_ready'],candidate['actual_CPU_failure'],
    pin(F/'actual_failure_closure.json'),pin(F/'actual_failure_metadata_completion_v1.json'),
    pin(F/'ROOT_source31_original_interval170469_saved170400_compact_failure_and_candidate_review_v1.json'),
    pin(F/'ROOT_source31_failure170469_direct69_actual_positive_verification_v1.json'),
    {k:validated_pin[k]for k in('path','bytes','sha256')},inventory,heldout,validated['bindings']['preparation_plan'],
    pin('E:/gpu_training_scratchpad/storage_config.json'),pin('E:/gpu_training_scratchpad/use_storage.ps1'),
    pin('E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe')]
values += policy_pins+[pin(path)for path in source_paths]
unique={}
for value in values:
    expected={k:value[k]for k in('path','bytes','sha256')};actual=pin(value['path'])
    require(actual==expected,'Frozen canonical input pin exact: '+value['path'])
    unique[actual['path']]=actual
inputs=list(unique.values())
frozen=F/'ordered14_decoder_frozen_inputs_v1.json'
write(frozen,{'kind':'ASR18_inactive_recorded170470_ordered14_decoder_inputs_v1','candidate':candidate_pin,
    'configuration':config_pin,'source_identity_inventory':inventory,'heldout_PCM':heldout,
    'original_resource_policy':config['resident_RAM_resource_scheduling_policy'],
    'seven_overlay_nonmatches':policies,'original_raw_contract':raw,'original_per_row_prospective_bytes':budgets,
    'maximum_original_prospective_bytes':maximum,'input_pins':inputs})
diff=F/'ordered9_to_ordered14_decoder_observation_source.diff.txt'
priorprobe=N/'resident_RAM_original_interval_post142894_failure_actual_v1/observe_candidate142895_original_intervals_v1.py'
diff.write_bytes(''.join(difflib.unified_diff(priorprobe.read_text().splitlines(True),FIXTURE.read_text().splitlines(True),
    fromfile=str(priorprobe),tofile=str(FIXTURE))).encode('utf-8'))
handoff=F/'ordered14_original_decoder_inactive_source_handoff_v1.json'
write(handoff,{'kind':'ASR18_INACTIVE_exact_ordered14_original_decoder_reproduction_handoff_v1',
    'execution_admitted':False,'required_ROOT_scope':SCOPE,'fixture_source':pin(FIXTURE),'launcher_source':pin(LAUNCHER),
    'frozen_inputs':pin(frozen),'external_raw_inverse':pin(ledger),'fixture_delta_to_prior_ordered9':pin(diff),
    'prior_decoder_observation_source':pin(priorprobe),'input_pins':inputs,'original_resource_policy':config['resident_RAM_resource_scheduling_policy'],
    'candidate_global_update':170470,'recorded_plan':plan_pin,'recorded_plan_canonical_sha256':candidate['recorded_target_plan_canonical_sha256'],
    'local_update_index':9,'ordered_rows_1based':list(range(1,15)),
    'ordered_row_ids':[{'row_1based':v['original_update_row_1based'],'ordinal':v['ordinal'],'dataset':v['dataset'],'source_id':v['source_id']}for v in candidate['ordered14_original_TR_candidates']],
    'seven_overlay_dispatch':'All14 have BOTH ordinal and unique/source id nonmatches at each of seven policies; same ordinary chain to core and decoder2, one actual decoder2 read_original per row',
    'actual_code_scope':'Original finalized source inventory/receipt/index guards, raw ledger/geometry declarations, source decoder/resample/hash/heldout check; frozen plan/raw labels copied as metadata, no repeated original scalar DB label audit',
    'exception_evidence':'Full exact row/raw annotation/full labels flushed+fsynced before decode; first decoder exception copied immediately using explicit scalar locals+bounded first5/last8 frames; outer propagated exception separately preserved; no retained frames/tracebacks',
    'stop_first_actual_failure':True,'baseline_rerun_admitted':False,'fullmember_read_admitted':False,
    'no_frontend_transform_model_GPU_sampler_RNG_journal_or_new_draw':True,'source_geometry_tolerance_or_production_change':False,
    'limits':{'fixture_wall_seconds':180,'external_wall_seconds':240,'sampled_whole_family_private_bytes':8000000000,
        'CPU_original_private_bytes':8000000000,'CPU_prospective_commit_reserve_bytes':2500000000,
        'original_CPU_allocation_wait_seconds':300,'FULL_retained_prefetch_bytes':268435456,
        'runtime_commit_floor_plus_prefetch_bytes':2268435456,'runtime_physical_floor_plus_prefetch_bytes':1768435456,
        'max_original_raw_media_maxworkspace_numerical_prospective_bytes':maximum,
        'external_admission_commit_bytes':2500000000+268435456+maximum,
        'external_admission_physical_bytes':1500000000+268435456+maximum,
        'E_free_bytes':100000000000,'CUDA_VISIBLE_DEVICES':'-1','all4threadcaps':1,'priority':'BelowNormal',
        'family_is_sampled_not_allocation_hard_cap':True,'CPU_original_private_and_reserve_NOT_replaced_with512MiB':True},
    'original_raw_plan_and_full_frontend_transform_prospective_reservations_retained_without_executing_frontend':True,
    'output_path':str((F/'ordered14_decoder_actual_01').resolve()),
    'future_exact_external_arguments':['E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe','-Xutf8','-B',str(LAUNCHER),'--ROOT-review','<FUTURE_EXACT_ROOT_SCOPE_PATH>'],
    'expected_observation_result':'Exit0 means bounded diagnostic complete, NOT that audio passed: report records first actual exception or all14 successes; guard/cleanup/nonzero cannot be promoted',
    'no_training_admission_or_unchanged_retry':True})
print(json.dumps({'fixture':pin(FIXTURE),'launcher':pin(LAUNCHER),'inputs':pin(frozen),'handoff':pin(handoff),
    'raw_inverse':pin(ledger),'input_count':len(inputs),'max_prospective':maximum,'no_fixture_execution':True}))
