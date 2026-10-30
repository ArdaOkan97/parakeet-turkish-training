"""Preparation only: AST and immutable metadata; no sampler or toy execution."""
import ast, hashlib, json, os
from pathlib import Path

F=Path(__file__).parent.resolve();P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
S=P/'outputs/parakeet_turkish'
B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01')

def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')

assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
helper=F/'py_spy_nonblocking_bounded_helper_v1.py';toy=F/'py_spy_toy_target_v1.py'
toy_launcher=F/'launch_py_spy_same_interpreter_toy_v1.py';live_launcher=F/'launch_py_spy_existing_native40012_v1.py'
trees={str(p):ast.parse(p.read_text(encoding='utf-8'),filename=str(p)) for p in (helper,toy,toy_launcher,live_launcher)}
tree=trees[str(helper)]
resources={
 'family_private_cap_bytes':536870912,'full_prospective_family_reserve_bytes':536870912,
 'full_prefetch_reservation_bytes':268435456,'commit_floor_bytes':2000000000,'physical_floor_bytes':1500000000,
 'E_free_floor_bytes':100000000000,'external_wall_seconds':120,'poll_seconds':.1,
 'profile_byte_cap':67108864,'log_byte_cap_each':4194304,'priority':'BelowNormal'}
target={'pid':40012,'create_time':1792743161.6231525,'ppid':20284}
paths=[helper,toy,toy_launcher,live_launcher,F/'py_spy_0_4_2_win_amd64_v1/py-spy.exe',
 F/'py_spy_0_4_2_win_amd64_v1/preparation_receipt.json',F/'ROOT_pinned_py_spy_0_4_2_isolated_E_preparation_review_v1.json',
 F/'ROOT_direct_human_then_fix_authorization_v1.json',S/'resource_watchdog_v2.py',S/'paired_windows_commit_telemetry_v1.py',
 P/'work/nemo-native-env-v2/Scripts/python.exe',S/'asr18_BPE9223_grouped_RAM_native_v1.py',
 P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v30.py',
 B/'invocations/owner_000022/boot.json',B/'invocations/owner_000022/configuration.json',
 Path(r'E:\gpu_training_scratchpad\storage_config.json'),Path(r'E:\gpu_training_scratchpad\use_storage.ps1')]
pins=[pin(p) for p in paths]
assert pins[4]['sha256']=='6e83ab095f40453742d01a2719a178f308534015da519ec0659d2584139dc995'
assert pins[12]['sha256']=='eb148001e6851993072f10610e77af16566987b8c0f4ed19d33aee3b1724b3e2'
assert pins[14]['sha256']=='5f3bc165a47810c134d649a10c73726f42d18b9f11d42a79a44e5d7cf3f499a4'
common={'helper_source':pin(helper),'input_pins':pins,'resources':resources,
 'execution_not_yet_admitted':True,'production_source_environment_guards_or_state_changed':False,
 'process_local_environment':{'CUDA_VISIBLE_DEVICES':'-1','PYTHONUTF8':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','storage':'Read storage_config and dot-source use_storage.ps1 including TORCH_EXTENSIONS_DIR before one invocation'},
 'outer_interpreter':r'E:\gpu_training_scratchpad\scratch\BPE_broad_CPU_preflight_env_v2\Scripts\python.exe',
 'parent_training_environment_unchanged':True,
 'limits_and_cleanup':'120s wrapper bound, original paired-resource sampling of OWNED sampler/toy family at100ms. At failure, exact-owned profiler only may be terminated; existing native never signaled/suspended/restarted. Owned toy can receive its own stop file and be cleaned if needed. Required cleanup/nonzero/error cannot be promoted to pass.',
 'metadata_closure':'Wrapper report records actual Popen returns and all observed owned IDs; outer TOOL receipt plus a later metadata completion must bind exact wrapper/launcher deaths. No success from Popen creation alone.',
 'profile_interpretation':'All-thread nonblocking Python stacks with sampling errors/stderr preserved. Statistical wall observations, not pureCPU or CUDA-kernel timers; non-atomic reads may miss/partially read frames. No C-native unwinding or GIL-only filter.'}
toy_scope='ASR18_py_spy_same_interpreter_toy_compatibility_CPU_v1'
toyout=F/'py_spy_same_interpreter_toy_actual_01'
toyh={**common,'kind':'ASR18_INACTIVE_py_spy_toy_compatibility_source_handoff_v1','scope':toy_scope,'mode':'toy',
 'launcher_source':pin(toy_launcher),'toy_source':pin(toy),'output':str(toyout),
 'future_exact_external_arguments':['--ROOT-review','EXACT_ROOT_TOY_SCOPE_NOT_YET_ISSUED'],
 'same_native_interpreter':str(P/'work/nemo-native-env-v2/Scripts/python.exe'),
 'toy_duration_seconds':30,'sampler_duration_seconds':8,'sampler_rate_Hz':100,
 'toy_behavior':'Standard library only, deterministic Python work, tiny self-cycle collection, sleep plus one sleep thread; noTorch/NumPy/audio/model/GPU/data or training import. Exact same interpreter and actual toy Python frame required in real speedscope output.',
 'required_ROOT_scope_fields':{'accepted':True,'reviewer':'/root','scope':toy_scope,'mode':'toy','helper_source':pin(helper),'launcher_source':pin(toy_launcher),'toy_source':pin(toy),'resources':resources,'input_pins':pins,'output':str(toyout)},
 'required_completed_ROOT_toy_gate_fields':{'accepted':True,'reviewer':'/root','scope':'ASR18_py_spy_same_interpreter_toy_compatibility_actual_CPU_v1','executable':pin(paths[4]),'helper_source':pin(helper)},
 'toy_pass_does_not_admit_existing_native_sampling':True}
liveout=F/'py_spy_existing_native40012_actual_01'
cmd=[str(paths[4]),'record','--pid','40012','--duration','60','--rate','100','--nonblocking','--idle','--threads','--full-filenames','--format','speedscope','--output',str(liveout/'profile.speedscope.json')]
live_scope='ASR18_existing_source30_nonblocking_stack_sample_v1'
liveh={**common,'kind':'ASR18_INACTIVE_existing_native40012_nonblocking_sampling_source_handoff_v1','scope':live_scope,'mode':'live',
 'launcher_source':pin(live_launcher),'output':str(liveout),'target':target,'exact_sampler_arguments':cmd,
 'future_exact_external_arguments':['--ROOT-review','EXACT_ROOT_LIVE_SCOPE_NOT_YET_ISSUED'],
 'completed_toy_ROOT_review':None,'separate_live_ROOT_scope_required':True,
 'required_ROOT_scope_fields':{'accepted':True,'reviewer':'/root','scope':live_scope,'mode':'live','helper_source':pin(helper),'launcher_source':pin(live_launcher),'resources':resources,'input_pins':pins,'output':str(liveout),'target':target,'exact_sampler_arguments':cmd,'completed_toy_ROOT_review':'EXACT_GENUINE_COMPLETED_TOY_ROOT_PIN_REQUIRED'},
 'target_checks':'ExactPID/create_time, source30 boot and actual runtime source/config pins, original full401072/adminNone/nonpoisoned current22/noactualfailure, before and after; exact liveness during every polling loop. No generic PID selection or fallback. Native/CPU/reader are not owned cleanup targets.',
 'precision_limits':'Python wrappers can distinguish torch.cuda.synchronize/torch.cuda.memory.empty_cache when visible, versus native chained callsites or snapshot/assert/RNG functions. Builtin gc.collect and other C internals sharing the same Python line remain ambiguous. No pureGC/kernel/thermal/disk attribution from this alone.'}
attribution={'kind':'ASR18_READONLY_cleanup_path_sampling_attribution_proposal_v1','selected_production_remedy':None,
 'native_source':pin(S/'asr18_BPE9223_grouped_RAM_native_v1.py'),'native_call_order_changes':0,
 'existing_group_boundary':{'before_timer':'Initial backward CUDA synchronize and before snapshot complete before began=time.monotonic at line211.',
 'included':['NumPy/Python/Torch/CUDA RNG snapshots','207 parameter/gradient pointers and versions','gc.collect','second CUDA synchronize','torch.cuda.empty_cache','resource/CUDA snapshot','pointer/frontier/RNG equality checks','assert_model','logical envelope construction before seconds field'],
 'same_line_ambiguity':'gc.collect;torch.cuda.synchronize;torch.cuda.empty_cache;after=snapshot are chained on native1 line214. Python line-only stacks cannot promise full individual C-call precision.'},
 'native_final_cleanup':'optimizer.step/zero_grad/gc.collect/synchronize/empty_cache are chained at native1 line233, outside archived group boundary but inside native+commit clock.',
 'source_facts_not_causal_proof':['Original resident_gc_scope freezes once at entry and unfreezes at final exit; unchanged.','Monitor retains scalar samples/peaks, not a growing sample list.','Journal append retains only incremental hash/offset after write/fsync, not event history.','Canonical archive byte representation remains deployed; its1600 validation did not hold the improvement.'],
 'next_decision':'Use a separately admitted toy compatibility pass, then separately reviewed60s nonblocking Python stacks to choose a supported narrow component diagnosis. Do not select GC refresh/process/cache/guard/native change from source hypothesis alone.',
 'no_actual_toy_or_live_sampling_yet':True}
write(F/'py_spy_same_interpreter_toy_source_handoff_v1.json',toyh)
write(F/'py_spy_existing_native40012_source_handoff_v1.json',liveh)
write(F/'readonly_cleanup_sampling_attribution_v1.json',attribution)
print(json.dumps({'AST_sources':[pin(p) for p in (helper,toy,toy_launcher,live_launcher)],
 'toy_handoff':pin(F/'py_spy_same_interpreter_toy_source_handoff_v1.json'),
 'live_handoff':pin(F/'py_spy_existing_native40012_source_handoff_v1.json'),
 'attribution':pin(F/'readonly_cleanup_sampling_attribution_v1.json'),'executed_sampler_or_toy':False}))
