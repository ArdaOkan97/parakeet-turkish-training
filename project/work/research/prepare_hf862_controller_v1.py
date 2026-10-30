"""One-shot stdlib controller generation; no worker imports or GPU operations."""
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; O=ROOT/'outputs/parakeet_turkish'
names=('train_lora_batched_data_v11.py','launch_with_io_retry_v9.py','launch_with_cache_reclaim_v12.py',
    'launch_with_memory_guard_v15.py','launch_native_training_v5.py','hf_exact_runtime_continuation_v1.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pins={name:sha(O/name) for name in names}
oldtree=ast.parse((O/'train_lora_batched_data_v10.py').read_text(encoding='utf-8'))
newtree=ast.parse((O/'train_lora_batched_data_v11.py').read_text(encoding='utf-8'))
def fn(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
def cleanrun(tree):
    return ast.Module(body=[n for n in fn(tree,'run').body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='source_names' for t in n.targets))],type_ignores=[])
assert ast.dump(cleanrun(oldtree),include_attributes=False)==ast.dump(cleanrun(newtree),include_attributes=False)
for name in ('parser','save_training_state','validate_arguments','native_loss_identity','native_loss_function'):
    assert ast.dump(fn(oldtree,name),include_attributes=False)==ast.dump(fn(newtree,name),include_attributes=False)
assert (O/'train_lora_batched_data_v11.py').read_text(encoding='utf-8').splitlines()[543].strip()=='torch.cuda.synchronize()'
with (O/'HF_862_runtime_chain_pins_v1.json').open('x',encoding='utf-8') as f:
    json.dump({'schema':1,'passed':True,'source_sha256':pins,'training_update_AST_unchanged':True,'optimizer_end_line':544,'CUDA_initialized':False},f,indent=2)
text=(O/'resume_native_training_358_v1.py').read_text(encoding='utf-8')
for old,new in (('adaptive_20261011_native_split_memory_01','adaptive_20261011_hf_return_window_01'),
    ('adaptive_20261011_native_split_resume_01','adaptive_20261011_hf_return_01'),
    ('adaptive_20261011_native_memory_fix_01_public_winner_extension','adaptive_20261011_native_split_resume_01'),
    ('initial_state_358','initial_state_862'),('train_lora_batched_data_v10.py','train_lora_batched_data_v11.py'),
    ('launch_native_training_v4.py','launch_native_training_v5.py'),('launch_with_io_retry_v8.py','launch_with_io_retry_v9.py'),
    ('launch_with_cache_reclaim_v11.py','launch_with_cache_reclaim_v12.py'),('launch_with_memory_guard_v14.py','launch_with_memory_guard_v15.py')):
    text=text.replace(old,new)
tree=ast.parse(text)
start=text.index('SOURCE_PINS = {');end=text.index('\n\n\ndef require',start)
source_pins={n:sha(O/n) for n in ('native_tdt_checkpointed_v3.py','cuda_cache_reclaim_v2.py','resource_watchdog_v2.py',
    'paired_windows_commit_telemetry_v1.py','public_turkish_data_policy_v3.py','preload_nemo_linker_v1.py','run_experiment_loop_continuation_v8.py',*names)}
text=text[:start]+'SOURCE_PINS = '+repr(source_pins)+text[end:]
start=text.index('def prepare(args):');end=text.index('\n\ndef validate_identity',start)
prepare='''def prepare(args):
    require(0 < args.window_hours <= 3 and 5 <= args.poll_seconds <= 15, "Bounded3h window / fresh5–15s polls required")
    require(sha(FORK) == FORK_SHA, "Actual root CPU full-state862 fork changed")
    fork = read(FORK, {})
    require(fork["passed"] and fork["source_run_id"] == PARENT and fork["target_run_id"] == TARGET
        and fork["source_step"] == 862 and fork["inherited_selected_step"] == 0 and fork["CUDA_initialized"] is False,
        "Exact862/best0 fork required")
    for item in fork["source_evidence"].values():
        require(sha(item["path"]) == item["sha256"], "Closed source evidence changed")
    for name,digest in SOURCE_PINS.items():
        require(sha(HERE/name) == digest, "Frozen new deployment source changed: "+name)
    require(not (LOOP/"loop_status.json").exists(), "Do not overwrite existing controller state")
    require(not (HERE/"runs"/TARGET/"run_config.json").exists(), "New run configuration required")
    require(not (ROOT/"work"/(TARGET+"_latest.pt")).exists(), "Immutable input fork only; latest initially absent")
    for role,item in fork["fork_checkpoint"].items():
        require(sha(item["path"]) == item["sha256"], "Immutable checkpoint fork changed: "+role)
    latest,best = fork["fork_checkpoint"]["latest"],fork["fork_checkpoint"]["best"]
    require(sha(best["mutable_work_path"]) == best["sha256"], "Qualifying BEST0 changed")
    for step,item in fork["development_reuse"].items():
        for name,digest in item["files_sha256"].items():
            require(sha(Path(item["target_path"])/name)==digest,"Reused DEV changed")
    config = read(fork["source_run_config"]["path"],{})
    require(config["run_id"]==PARENT and config["config_identity"]==fork["source_config_identity"]
        and config["batch_size"]==8 and config["accumulation"]==2 and config["replay_fraction"]==.5,
        "Preserve scientific B8x2/replay50 configuration")
    paths = {key:LOOP/(TARGET+suffix) for key,suffix in {"native":"_native_backend.json", "io":"_io.json",
        "cache":"_cache.json", "memory":"_memory.json", "identity":"_identity.json", "stop":"_stop.txt", "log":".log"}.items()}
    require(all(not p.exists() for p in paths.values()), "Fresh runtime/evidence files required")
    transition = {"kind":"HF_already_native3_exact_runtime_continuation_v1","status":"CPU_prepared","passed":True,
        "source_run_id":PARENT,"target_run_id":TARGET,"source_step":862,"inherited_selected_step":0,
        "fork_latest":latest,"fork_best":best,"source_run_config":fork["source_run_config"],
        "source_config_identity_sha256":fork["source_config_identity_sha256"],"approved_source_changes":{},
        "added_identity_validator":{"filename":"hf_exact_runtime_continuation_v1.py","sha256":SOURCE_PINS["hf_exact_runtime_continuation_v1.py"]},
        "replaced_previous_migration":record(config["resume_runtime_migration"]),
        "native_backend_transition":{"old_path":config["native_backend_provenance"],"old_sha256":sha(config["native_backend_provenance"]),"new_path":str(paths["native"])},
        "declared_allocator_policy":{"cap_gb":8,"end_cache_gb":4,"native_inactive_threshold_bytes":750000000},
        "training_recipe_changed":False,"optimizer_sampler_RNG_history_patience_retention_counters_preserved":True,
        "latest_ES_retention_failure_preserved":True,"qualifying_best_step":0}
    for old,new in (("train_lora_batched_data_v10.py","train_lora_batched_data_v11.py"),
        ("launch_native_training_v4.py","launch_native_training_v5.py")):
        require(sha(HERE/old)==config["config_identity"]["source_sha256"][old],"Original frozen source changed")
        transition["approved_source_changes"][old]={"old_sha256":sha(HERE/old),"new_filename":new,"new_sha256":SOURCE_PINS[new]}
    migration = LOOP/"runtime_identity_continuation_v1.json"
    if migration.exists():
        require(read(migration,{})==transition,"Prepared identity receipt changed")
    else:
        with migration.open("x",encoding="utf-8") as f:json.dump(transition,f,indent=2,ensure_ascii=False)
    old_state=read(fork["source_evidence"]["controller_status"]["path"],{})
    argv=list(old_state["worker_arguments"]);original=list(argv)
    for flag,value in (("--run-id",TARGET),("--resume-weights",None),("--resume-exact",latest["path"]),
        ("--native-backend-provenance",paths["native"]),("--resume-runtime-migration",migration),
        ("--stop-file",paths["stop"]),("--max-wall-seconds",max(1,args.window_hours*3600-120))):
        replace_option(argv,flag,value)
    allowed={"--run-id","--resume-exact","--native-backend-provenance","--resume-runtime-migration","--stop-file","--max-wall-seconds"}
    def unchanged_arguments(values):
        values=list(values)
        for flag in allowed:replace_option(values,flag,None)
        return values
    require(unchanged_arguments(argv)==unchanged_arguments(original),"Scientific/cadence/data arguments changed")
    memory=["--provenance",str(paths["io"]),"--cache-provenance",str(paths["cache"]),"--memory-provenance",str(paths["memory"]),
        "--worker-identity",str(paths["identity"]),"--cap-gb","8","--threshold-gb","4","--",*argv]
    command=[str(previous.scoped(args.python)),str(HERE/"launch_native_training_v5.py"),
        "--native-provenance",str(paths["native"]),"--memory-launcher",str(HERE/"launch_with_memory_guard_v15.py"),
        "--memory-launcher-sha256",SOURCE_PINS["launch_with_memory_guard_v15.py"],"--trainer-sha256",SOURCE_PINS["train_lora_batched_data_v11.py"],"--",*memory]
    return {"passed":True,"run_id":TARGET,"loop_id":LOOP.name,"initial_step":862,"inherited_selected_step":0,
        "fork":record(FORK),"predecessor_exit_evidence":fork["source_evidence"]["controller_status"],
        "runtime_migration":record(migration),"paths":{k:str(p) for k,p in paths.items()},"command":command,
        "worker_arguments":argv,"code_sha256":SOURCE_PINS,"controller_sha256":sha(__file__),
        "scientific_arguments_unchanged":True,"exact_optimizer_sampler_RNG_continuation":True,
        "latest_ES_retention_failure_preserved":True,"heldout_tests_evaluated":False,"CUDA_initialized_by_preflight":False}
'''
text=text[:start]+prepare+text[end:]
text=text.replace('FORK_SHA = "1963c1e96f49d499b3004959635074789b28b2984a3cab33f94df9130c9831f6"',
    'FORK_SHA = "2aeeed4836a3e536893d527b1d3fa01afcfdc34444ca0a835ffd59d0f45265f1"')
text=text.replace('native_exact358_split_runtime_window_v1','HF_exact862_quality_runtime_window_v1').replace('else 358','else 862').replace('>= 358','>= 862').replace('"initial_step": 358','"initial_step": 862')
text=text.replace('Preserves scientific arguments, cadence250, Adam/sampler/RNG/history/best0.',
    'Preserves B8x2/50%replay/row6/cadence250 and all862 Adam/sampler/RNG/history/best0 counters.')
text=text.replace('Root-only bounded exact358 continuation with the verified split native loss.',
    'Root-only bounded exact862 HF quality continuation; official native3 loss kernel unchanged.')
text=text.replace('parser.add_argument("--kernel-verification", type=Path, required=True)',
    'parser.add_argument("--kernel-verification", type=Path)')
p=O/'resume_hf_training_862_v1.py'
with p.open('x',encoding='utf-8') as f:f.write(text)
ast.parse(text)
print(json.dumps({'pins':source_pins,'controller_sha256':sha(p),'training_update_AST_unchanged':True,'optimizer_end_line':544},indent=2))
