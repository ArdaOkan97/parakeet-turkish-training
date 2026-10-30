import ast,hashlib,json,os
from pathlib import Path
F=Path(__file__).parent;P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish'
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
    with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
old=F/'launch_original280_scalar_metadata_stage_timing_v1.py';new=F/'launch_fresh_signature_collect_observation_CPU_v1.py'
edits=[('observe_original280_scalar_metadata_stages_v1.py','verify_fresh_signature_and_collect_observation_CPU_v1.py',1),
 ('scalar_metadata_stage_timing_external_result_v1.json','fresh_signature_collect_observation_external_result_v1.json',1),
 ('scalar_metadata_stage_timing_actual_01','fresh_signature_collect_observation_actual_CPU_01',1),
 ('ASR18_original280_scalar_metadata_stage_timing_CPU_v1','ASR18_fresh_single_open_signature_and_collect_observation_CPU_v1',1),
 ('original280_scalar_metadata_external_CPU','fresh_signature_collect_external_CPU',1),
 ('time.monotonic()-began<180','time.monotonic()-began<240',1),
 ('Sampled family512MiB/external180s','Sampled family512MiB/external240s',1),
 ('scalar_metadata_stage_timing_stdout.log','fresh_signature_collect_stdout.log',1),
 ('scalar_metadata_stage_timing_stderr.log','fresh_signature_collect_stderr.log',1),
 ('ASR18_original280_scalar_metadata_stage_timing_external_result_v1','ASR18_fresh_signature_collect_observation_external_result_v1',1)]
raw=old.read_bytes();text=raw.decode('utf-8')
for a,b,n in edits:assert text.count(a)==n;text=text.replace(a,b)
ast.parse(text);assert not new.exists();new.write_bytes(text.encode('utf-8'));inverse=text
for a,b,n in reversed(edits):assert inverse.count(b)==n;inverse=inverse.replace(b,a)
assert inverse.encode('utf-8')==raw
ledger=F/'fresh_signature_collect_external_launcher_exact_inverse_v1.json'
write(ledger,{'old':pin(old),'new':pin(new),'newline_normalization':'NONE; original full byte stream preserved',
 'edits':[{'old_literal':a,'new_literal':b,'expected_count':n}for a,b,n in edits],'full_raw_inverse_exact':True})
fixture=F/'verify_fresh_signature_and_collect_observation_CPU_v1.py';ast.parse(fixture.read_text(encoding='utf-8'))
inputs=[fixture,new,F/'asr18_fresh_single_open_metadata_signature_v2.py',F/'asr18_fresh_metadata_signature_scope_v2.py',
 F/'asr18_observe_original_gc_collect_v2.py',F/'inactive_signature_scope_collect_observer_v2_exact_ledger.json',ledger,
 F/'ROOT_fresh_signature_collect_observation_inactive_helpers_source_review_v1.json',F/'ROOT_direct_human_then_fix_authorization_v1.json',
 F/'exact_already_planned280_scalar_metadata_input_v1.json',
 Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01\invocations\owner_000021\plans\plan_000149.json'),
 F.parent/'verified_population_v1/final_runtime_bindings_9223_v1.json',
 Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01\invocations\owner_000022\configuration.json'),
 S/'asr18_BPE9223_source_context_v2.py',S/'bpe_declared_geometry_index_v1.py',S/'asr18_BPE9223_private_scalar_prefetch_v1.py',
 S/'bpe_windows_final_metadata_lock_v1.py',S/'bpe_broad_production_state_v1.py',S/'bpe_broad_audio_v1.py',
 S/'bpe_runtime_declared_geometry_v1.py',S/'asr18_physical_feature_geometry_v2.py',
 S/'resource_watchdog_v2.py',S/'paired_windows_commit_telemetry_v1.py',
 Path(r'E:\gpu_training_scratchpad\storage_config.json'),Path(r'E:\gpu_training_scratchpad\use_storage.ps1')]
pins=[pin(p)for p in inputs]
out=F/'fresh_signature_collect_observation_actual_CPU_01'
limits={'fixture_wall_seconds':180,'external_wall_seconds':240,'family_private_cap_bytes':536870912,
 'full_prospective_family_reserve_bytes':536870912,'full_prefetch_reservation_bytes':268435456,
 'original_commit_floor_bytes':2000000000,'original_physical_floor_bytes':1500000000,'E_free_floor_bytes':100000000000,
 'process_local_CUDA_VISIBLE_DEVICES':'-1','all_four_math_thread_caps':1,'priority':'BelowNormal','UTF8':True,'B_no_bytecode':True}
h={'kind':'ASR18_INACTIVE_fresh_signature_original_collect_focused_CPU_source_handoff_v1',
 'fixture_source':pin(fixture),'launcher_source':pin(new),'input_pins':pins,'external_full_raw_inverse':pin(ledger),
 'output':str(out),'external_result':str(F/'fresh_signature_collect_observation_external_result_v1.json'),
 'future_exact_external_arguments':['--ROOT-review','EXACT_ROOT_SCOPE_NOT_YET_ISSUED'],
 'required_ROOT_scope_fields':{'accepted':True,'reviewer':'/root','scope':'ASR18_fresh_single_open_signature_and_collect_observation_CPU_v1',
  'fixture_source':pin(fixture),'launcher_source':pin(new),'input_pins':pins,'output':str(out)},
 'limits':limits,'actual_execution_admitted':False,'completed_CPU_review':None,'production_admitted':False,
 'proof':{'real':['Fresh same-handle canonical path/size/integerFILETIME/currentmutation/parentrename/unicode/spaces and owned handle closures.',
  'Actual Windows reparse creation only if supported in owned E scratch; unsupported is explicitly recorded, never promoted as a real reparse proof.',
  'Original row/declaration methods with original readonly3DB/4MiBcache/32LRU/denywrite handles, one exact280 candidate query pass and original plan oracle/cursor equality, all840 hot signature checks preserved.',
  'Real worker joins before all3 scoped aliases restore; genuine originalbuiltin collect of owned temporary cycle, original thresholds/freeze/enabled unchanged and observer released.',
  'Fixed ORIGINAL first then CANDIDATE second,2threads×280signaturecalls each cycling same3locked actualmetadata paths; actual oracle equality, bounded wall/threadCPU clocks and handle lifetimes.'],
 'method_stubs':['FullRunSourceContext.__new__/resolver scaffold is constructor/authority double, not repeated full population hash admission; original methods/DB locks are genuine.',
  'Final-path/API/CloseHandle failure return injection, wrong lockpath/noncallable/changedalias/join-defer/callback negatives explicitly labelled.',
  'gc return/args/exception/clockfailure/reentrancy passthrough uses labelled methodstubs; nested call-wall durations may overlap, never losttime or CPU.'],
 'timing_eligibility':'Overall job will not pass merely on semantics if CANDIDATE is slower than ORIGINAL in the one predeclared fixed two-thread observation. It is still not a live throughput/fit/causal proof; preserve every failure, no automatic retry.'},
 'frozen_future_actual_validation':{'start':'restored+20','end':'max(start+1600,nextscheduledDEV+1020)',
 'coverage':'All contiguous original20rows/all200saves/realDEV/allpauses, first400/final400 and two full400postDEV blocks; no finite stop/extraDEV.',
 'engineering_acceptance':'Near original76.57 checkpoint-inclusive400 (~75+) and <=5% save-excluded first-to-final sustained decline, no guard/science/state/checkpoint/reader regression; partial gain or recurrent decline keeps human fix OPEN. Different ordinals, no isolated-cause/forecast guarantee.'},
 'original_source30_session72166_must_continue_unchanged':True,
 'no_new_audio_model_GPU_native_or_authoritative_sampler_activity':True}
hp=F/'fresh_signature_collect_focused_CPU_source_handoff_v1.json';write(hp,h)
print(json.dumps({'fixture':pin(fixture),'launcher':pin(new),'inverse':pin(ledger),'handoff':pin(hp),'input_count':len(pins),'AST_only_no_control_run':True}))
