import ast,ctypes,hashlib,json,os,pathlib,shutil,time
import psutil
P=pathlib.Path(r'C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
D=pathlib.Path(r'E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1/next212000_failure_repair_v1')
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
class MS(ctypes.Structure):
    _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(n,ctypes.c_ulonglong)for n in ("totalphysical","physical","commitlimit","commit","totalvirtual","virtual","extended")]
def sample():
    m=MS();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
    return dict(commit=m.commit,physical=m.physical,private=psutil.Process().memory_info().private,E_free=shutil.disk_usage('E:/gpu_training_scratchpad').free)
def who(p):return dict(pid=p.pid,ctime=p.create_time(),ppid=p.ppid())
def sha(b):return hashlib.sha256(b).hexdigest()
def pin_bytes(p,b):return dict(path=str(p),bytes=len(b),sha256=sha(b))
def pin(p):return pin_bytes(p,p.read_bytes())
def astsha(b):return sha(ast.dump(ast.parse(b.decode('utf-8-sig')),include_attributes=False).encode())
def write_source(p,b):
    ast.parse(b.decode('utf-8-sig'))
    with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
    return pin_bytes(p,b)
def write_json(p,obj):
    with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    return pin(p)
t0=time.monotonic();s=sample()
initial=write_json(D/'CPUfaithful_APInegative_SOURCE_initial_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=5399239296,physical=2649239296,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
assert s['commit']>=5399239296 and s['physical']>=2649239296 and s['private']<=536870912 and s['E_free']>=100000000000

import difflib
def raw_diff(old,new,source_path,target_path,labels):
    work=old;records=[];shift=0
    for tag,a,b,c,d in difflib.SequenceMatcher(None,old,new,autojunk=False).get_opcodes():
        if tag=='equal':continue
        at=a+shift;before=work
        assert work[at:at+(b-a)]==old[a:b]
        work=work[:at]+new[c:d]+work[at+(b-a):]
        records.append(dict(offset_before=at,old_utf8=old[a:b].decode(),new_utf8=new[c:d].decode(),before_sha256=sha(before),after_sha256=sha(work)))
        shift+=(d-c)-(b-a)
    assert work==new
    for e in reversed(records):
        at=e['offset_before'];x=e['new_utf8'].encode();y=e['old_utf8'].encode()
        assert work[at:at+len(x)]==x
        work=work[:at]+y+work[at+len(x):]
    assert work==old
    return dict(base=pin_bytes(source_path,old),candidate=pin_bytes(target_path,new),ordered_edits=records,whole_byte_inverse_exact=True,semantic_changes=labels,base_AST_sha256=astsha(old),candidate_AST_sha256=astsha(new))

CPU_BODY_BINDING="    # Different venv PATHS are authoritative even though the launcher shims hash\n    # identically. No package install, sys.path injection or environment edit.\n    expected_parent=Path(r\"C:/Users/ardao/Documents/Codex/2026-10-10/i-x20/work/nemo-native-env-v2/Scripts/python.exe\").resolve()\n    expected_worker=Path(r\"E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe\").resolve()\n    assert Path(sys.executable).resolve()==expected_parent\n    assert Path(scope[\"fixture_body_python\"][\"path\"]).resolve()==expected_parent\n    assert Path(scope[\"original_worker_python\"][\"path\"]).resolve()==expected_worker\n    assert expected_parent!=expected_worker\n    for executable in (scope[\"fixture_body_python\"],scope[\"original_worker_python\"]):\n        assert executable[\"bytes\"]==274192 and executable[\"sha256\"]==\"cac98e94324b5462d41a5c6ad1847b57508a1068ba56014cf67a3d33a493dcdf\"\n        assert pin(executable[\"path\"])==executable\n    save(root/\"actual_parent_worker_interpreter_bindings.json\",dict(\n        parent=scope[\"fixture_body_python\"],actual_parent_executable=str(Path(sys.executable).resolve()),\n        worker_contract=scope[\"original_worker_python\"],same_shim_hash_does_not_replace_path_authority=True,\n        no_package_path_or_environment_mutation=True))\n"
WATCHDOG_BINDING="    # CPU-only successor: keep the original E watchdog interpreter/environment,\n    # but launch the faithful firstpacket parent with its separately typed P venv.\n    body_python=scope[\"fixture_body_python\"]\n    worker_python=scope[\"original_worker_python\"]\n    expected_parent=Path(r\"C:/Users/ardao/Documents/Codex/2026-10-10/i-x20/work/nemo-native-env-v2/Scripts/python.exe\").resolve()\n    expected_worker=Path(r\"E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe\").resolve()\n    assert Path(sys.executable).resolve()==expected_worker\n    assert Path(body_python[\"path\"]).resolve()==expected_parent\n    assert Path(worker_python[\"path\"]).resolve()==expected_worker\n    assert expected_parent!=expected_worker\n    for executable in (body_python,worker_python):\n        assert executable[\"bytes\"]==274192 and executable[\"sha256\"]==\"cac98e94324b5462d41a5c6ad1847b57508a1068ba56014cf67a3d33a493dcdf\"\n        raw=Path(executable[\"path\"]).read_bytes()\n        assert len(raw)==executable[\"bytes\"]and hashlib.sha256(raw).hexdigest()==executable[\"sha256\"]\n        assert executable in scope[\"input_pins\"]\n    save(root/\"actual_typed_parent_worker_interpreters.json\",dict(\n        watchdog_executable=str(Path(sys.executable).resolve()),fixture_body_python=body_python,\n        original_worker_python=worker_python,no_environment_or_library_change=True))\n"
API_REPLACEMENTS=json.loads("[[\"        def publication_failure(path,value):\\n            marker=root/\\\"fail_next_STATE_publication.json\\\"\\n            if Path(path).name==\\\"mirror_state.json\\\"and marker.exists():\\n                fault=read(marker)\\n                if value.get(\\\"offset\\\")==fault[\\\"when_offset\\\"]:\\n                    marker.unlink() # Owned diagnostic fault token only.\\n                    save(root/\\\"STATE_publication_failure_observed.json\\\",dict(attempted_offset=value[\\\"offset\\\"],original_STATE_write_called=False,fault_owner=\\\"METHOD_STUB one local write failure\\\"))\\n                    raise OSError(\\\"fixture local STATE publication failed\\\")\\n            return original_write(path,value)\\n\",\"        fixture_area=under(read(bindings)[\\\"area\\\"],root)\\n        def negative_STATE_publication_fence():\\n            # METHOD_STUB fixture observation only. The actual failing STATE or\\n            # response is untouched; prevent a later retry/progress write from\\n            # racing the parent's exact negative-state assertion.\\n            for observed,released in (\\n                (\\\"STATE_publication_failure_observed.json\\\",\\\"STATE_failure_fence_release.json\\\"),\\n                (\\\"response_loss_observed.json\\\",\\\"response_loss_fence_release.json\\\")):\\n                if (root/observed).exists()and not(root/released).exists():\\n                    deadline=time.monotonic()+15\\n                    while not(root/released).exists():\\n                        if time.monotonic()>=deadline:\\n                            raise TimeoutError(\\\"METHOD_STUB negative STATE observation fence15s\\\")\\n                        time.sleep(.01)\\n        def publication_failure(path,value):\\n            if Path(path).name==\\\"mirror_state.json\\\":\\n                negative_STATE_publication_fence()\\n            marker=root/\\\"fail_next_STATE_publication.json\\\"\\n            if Path(path).name==\\\"mirror_state.json\\\"and marker.exists():\\n                fault=read(marker)\\n                if value.get(\\\"offset\\\")==fault[\\\"when_offset\\\"]:\\n                    # This snapshot is inside the exact intercepted write,\\n                    # immediately before rejecting it, not a polling sample.\\n                    pre_state=read(path)\\n                    marker.unlink() # Owned diagnostic fault token only.\\n                    save(root/\\\"STATE_publication_failure_observed.json\\\",dict(\\n                        pre_STATE=pre_state,attempted_STATE=value,attempted_offset=value[\\\"offset\\\"],\\n                        original_STATE_write_called=False,\\n                        observation_barrier=\\\"METHOD_STUB later STATE publications await explicit release; bounded15s\\\",\\n                        fault_owner=\\\"METHOD_STUB one local write failure\\\"))\\n                    raise OSError(\\\"fixture local STATE publication failed\\\")\\n            return original_write(path,value)\\n\"],[\"                    save(root/\\\"response_loss_observed.json\\\",dict(original_HTTP200_consumed=True,SQLite_durable_commit_from_HTTP200_alone=\\\"NOT_CLAIMED\\\",fault_owner=\\\"METHOD_STUB transport response loss after real API return\\\"))\",\"                    pre_state=read(fixture_area/\\\"mirror_state.json\\\")\\n                    save(root/\\\"response_loss_observed.json\\\",dict(\\n                        pre_STATE=pre_state,original_HTTP200_consumed=True,\\n                        intercepted_log_id=fault[\\\"when_log_id\\\"],\\n                        SQLite_durable_commit_from_HTTP200_alone=\\\"NOT_CLAIMED\\\",\\n                        observation_barrier=\\\"METHOD_STUB later STATE publications await explicit release; bounded15s\\\",\\n                        fault_owner=\\\"METHOD_STUB transport response loss after real API return\\\"))\"],[\"            wait(lambda:(case/\\\"STATE_publication_failure_observed.json\\\").exists(),20)\\n            wait(lambda:\\\"local STATE publication failed\\\"in str(read(area/\\\"current_status.json\\\").get(\\\"error\\\")),10)\\n            assert read(area/\\\"mirror_state.json\\\")==checkpoint_state\\n            assert read(area/\\\"current_status.json\\\")[\\\"pending_batch\\\"][\\\"count\\\"]==64\\n\",\"            try:\\n                wait(lambda:(case/\\\"STATE_publication_failure_observed.json\\\").exists(),20)\\n                wait(lambda:\\\"local STATE publication failed\\\"in str(read(area/\\\"current_status.json\\\").get(\\\"error\\\")),10)\\n                fault=read(case/\\\"STATE_publication_failure_observed.json\\\")\\n                at_fence=read(area/\\\"mirror_state.json\\\")\\n                failed_status=read(area/\\\"current_status.json\\\")\\n                assert fault[\\\"original_STATE_write_called\\\"]is False\\n                assert fault[\\\"attempted_offset\\\"]==len(raw)+len(raw2)\\n                assert at_fence==fault[\\\"pre_STATE\\\"]\\n                # Journal frontiers are independent of unrelated progress_signature.\\n                assert {k:fault[\\\"pre_STATE\\\"].get(k)for k in (\\\"journal_path\\\",\\\"offset\\\",\\\"last_update\\\")}=={k:checkpoint_state.get(k)for k in (\\\"journal_path\\\",\\\"offset\\\",\\\"last_update\\\")}\\n                assert fault[\\\"pre_STATE\\\"][\\\"journal_path\\\"]==str(journal)and fault[\\\"pre_STATE\\\"][\\\"offset\\\"]==len(raw)and fault[\\\"pre_STATE\\\"][\\\"last_update\\\"]==64\\n                pending=failed_status[\\\"pending_batch\\\"]\\n                assert pending==dict(journal_path=str(journal),start_offset=len(raw),end_offset=len(raw)+len(raw2),last_update=128,count=64)\\n                assert failed_status[\\\"mirror_thread_alive\\\"]is True\\n                save(case/\\\"STATE_failure_fence_assertion.json\\\",dict(\\n                    exact_pre_write_STATE=fault[\\\"pre_STATE\\\"],actual_STATE_at_error_fence=at_fence,\\n                    attempted_STATE=fault[\\\"attempted_STATE\\\"],pending_batch=pending,\\n                    journal_frontier_unchanged=True,original_STATE_write_called=False,\\n                    scientific_owner=\\\"METHOD_STUB observation fence only\\\"))\\n            finally:\\n                save(case/\\\"STATE_failure_fence_release.json\\\",dict(release_owned_METHOD_STUB_fence=True))\\n\"],[\"            wait(lambda:(case/\\\"response_loss_observed.json\\\").exists(),20)\\n            wait(lambda:\\\"response lost\\\"in str(read(area/\\\"current_status.json\\\").get(\\\"error\\\")),10)\\n            assert read(area/\\\"mirror_state.json\\\")==checkpoint_state\\n\",\"            try:\\n                wait(lambda:(case/\\\"response_loss_observed.json\\\").exists(),20)\\n                wait(lambda:\\\"response lost\\\"in str(read(area/\\\"current_status.json\\\").get(\\\"error\\\")),10)\\n                loss=read(case/\\\"response_loss_observed.json\\\")\\n                at_fence=read(area/\\\"mirror_state.json\\\")\\n                failed_status=read(area/\\\"current_status.json\\\")\\n                assert loss[\\\"original_HTTP200_consumed\\\"]is True and loss[\\\"intercepted_log_id\\\"]==first_id\\n                assert at_fence==loss[\\\"pre_STATE\\\"]\\n                assert {k:loss[\\\"pre_STATE\\\"].get(k)for k in (\\\"journal_path\\\",\\\"offset\\\",\\\"last_update\\\")}=={k:checkpoint_state.get(k)for k in (\\\"journal_path\\\",\\\"offset\\\",\\\"last_update\\\")}\\n                assert loss[\\\"pre_STATE\\\"][\\\"journal_path\\\"]==str(journal)and loss[\\\"pre_STATE\\\"][\\\"offset\\\"]==len(raw)+len(raw2)and loss[\\\"pre_STATE\\\"][\\\"last_update\\\"]==128\\n                pending=failed_status[\\\"pending_batch\\\"]\\n                assert pending==dict(journal_path=str(journal),start_offset=len(raw)+len(raw2),end_offset=len(raw)+len(raw2)+len(raw3),last_update=192,count=64)\\n                assert failed_status[\\\"mirror_thread_alive\\\"]is True\\n                save(case/\\\"response_loss_fence_assertion.json\\\",dict(\\n                    exact_pre_lost_response_STATE=loss[\\\"pre_STATE\\\"],actual_STATE_at_error_fence=at_fence,\\n                    pending_batch=pending,journal_frontier_unchanged=True,\\n                    original_HTTP200_consumed=True,SQLite_durable_commit_from_HTTP200_alone=\\\"NOT_CLAIMED\\\",\\n                    scientific_owner=\\\"METHOD_STUB observation fence only\\\"))\\n            finally:\\n                save(case/\\\"response_loss_fence_release.json\\\",dict(release_owned_METHOD_STUB_fence=True))\\n\"]]")

scopepath=D/'ROOT_CPUfaithful_parent_APInegative_fence_INACTIVE_SOURCE_scope_v1.json'
sr=scopepath.read_bytes();assert len(sr)==8006 and sha(sr)=='d385a6f7fdcc9c2af76d2b3eb176af789eafd06dd0619f0c5ad9d6b0bae1fe55'
scope=json.loads(sr)
binding_path=pathlib.Path(scope['ROOT_SOURCE_interpreter_bindings']['path'])
br=binding_path.read_bytes();assert len(br)==9347 and sha(br)=='28864e33f3177b16adaa7a7d028cd14e78736ac16e1b81b1b91e8154ce54e73a'
binding=json.loads(br)
base_cpu=D/'CPU14_original_startup_firstpacket_CPU_proof_v3.py'
base_watch=D/'owned_source_CPU_proof_watchdog_v1.py'
base_api=D/'Trackio7_real_API_mirror_scalar_CPU_integration_proof_v4.py'
cb=base_cpu.read_bytes();wb=base_watch.read_bytes();ab=base_api.read_bytes()
assert len(cb)==18417 and sha(cb)=='62a53d4682557fcb5f527a0f45c2a355d3b0dd74a247eace1677c260a949055f'
assert len(wb)==8988 and sha(wb)=='04d188dd774c4a9bed0ce896505a1b8563adbc1b5135c207da26d6266daad16a'
assert len(ab)==38186 and sha(ab)=='87d8a204861f4e9973aead577058a5487bf2a7bdfb97b3d9e240193973f090aa'
def replace_once(s,a,b):
    assert s.count(a)==1,(a[:100],s.count(a))
    return s.replace(a,b)
cpu=cb.decode()
# Actual v3 SOURCE wording is retained byte-exact; the interpreter check is at
# the original pre-import boundary, after numeric/resource/input pin checks.
# Anchor on the unchanged import statement rather than guessing comment text.
actual_anchor='    spec=importlib.util.spec_from_file_location("owned_original_CPU14_parent",scope["CPU14_source"]["path"])\n'
cpu=replace_once(cpu,actual_anchor,CPU_BODY_BINDING+actual_anchor)
cpu=replace_once(cpu,'    path=HERE/"owned_source_CPU_proof_watchdog_v1.py"','    path=HERE/"owned_CPU_parent_typed_proof_watchdog_v2.py"')
watch=wb.decode()
watch=replace_once(watch,'def execute(scope_path,fixture_path,mode):\n','def execute(scope_path,fixture_path,mode):\n    assert mode=="CPU","Typed-parent watcher is CPU proof only"\n')
anchor='    start=time.monotonic();known={};races=[];cleanup=[];guard_error=None;peak=0;last=initial;code=None\n'
watch=replace_once(watch,anchor,WATCHDOG_BINDING+anchor)
watch=replace_once(watch,'    argv=[sys.executable,"-X","utf8","-B",str(fixture_path),"--ROOT-review",str(scope_path),"--proof-body"]','    argv=[body_python["path"],"-X","utf8","-B",str(fixture_path),"--ROOT-review",str(scope_path),"--proof-body"]')
api=ab.decode()
for a,b in API_REPLACEMENTS:api=replace_once(api,a,b)
paths=dict(cpu=D/'CPU14_original_startup_firstpacket_CPU_proof_v4.py',watchdog=D/'owned_CPU_parent_typed_proof_watchdog_v2.py',api=D/'Trackio7_real_API_mirror_scalar_CPU_integration_proof_v5.py')
pins={k:write_source(paths[k],text.encode())for k,text in [('cpu',cpu),('watchdog',watch),('api',api)]}
def changed_top(old,new):
    def parts(s):
        return {getattr(n,'name',''):ast.dump(n,include_attributes=False)for n in ast.parse(s.decode()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a=parts(old);b=parts(new);assert set(a)==set(b)
    return [k for k in a if a[k]!=b[k]]
assert changed_top(cb,cpu.encode())==['fixture_body','main']
assert changed_top(wb,watch.encode())==['execute']
assert changed_top(ab,api.encode())==['server_child','fixture_body']
# Original CPU transport calls and first-fail observer are structurally exact.
def calls(s,names):
    return [ast.dump(n,include_attributes=False)for n in ast.walk(ast.parse(s))if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in names]
assert calls(cb.decode(),{'submit','wait_update_ready','next_update','release'})==calls(cpu,{'submit','wait_update_ready','next_update','release'})
ct=ast.parse(cpu);oldct=ast.parse(cb.decode())
def named_nested(tree,name):return next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name==name)
assert ast.dump(named_nested(ct,'observed_parent_fail'),include_attributes=False)==ast.dump(named_nested(oldct,'observed_parent_fail'),include_attributes=False)
cpu_labels=['CPU proof fixture_body verifies exact production P/nemo-native-env-v2 parent executable path and bytes/SHA, separately verifies original E worker path/pin; identical shims do not substitute for exact role paths',
'CPU proof main chooses fresh CPU-only typed-parent watchdog2; top-level E interpreter/environment unchanged',
'Original runner/CPU14/source, first-fail observer, submit/readiness/next/release and resources/science unchanged; no imports executed here']
watch_labels=['Reject non-CPU mode; after original numeric-before-full-predicate and input/env checks verify separately typed P parent and E worker executable paths/pins and persist scalar interpreter witness',
'Only fixture-body Popen argv executable becomes scope.fixture_body_python; E watchdog env, original E worker, capture/guards/caps/cleanup/polling/limits unchanged']
api_labels=['METHOD_STUB captures pre_STATE and attempted_STATE inside exact rejected STATE write; records original write not called',
'METHOD_STUB captures pre_STATE inside intercepted real successful response before intentional loss; no HTTP200 durability claim',
'Bounded15s publication-only METHOD_STUB fence prevents subsequent STATE retry/progress publication until controller observes original error/status and acknowledges; all original total/HTTP limits unchanged',
'Controller compares exact intercepted snapshots plus independent pre-append journal_path/offset/last_update, exact pending64 start/end/last/count/threadalive; releases owned fence in finally',
'Both500000-row shapes/cold8scalar+bulk64+version/original settled equivalence/SQLite timeout+latecommit+idempotence/warm/lifetime/cleanup unchanged; reader7/helper3/projector1/watchdog1 frozen']
cpu_ledger=raw_diff(cb,cpu.encode(),base_cpu,paths['cpu'],cpu_labels)
watch_ledger=raw_diff(wb,watch.encode(),base_watch,paths['watchdog'],watch_labels)
api_ledger=raw_diff(ab,api.encode(),base_api,paths['api'],api_labels)
cr=write_json(D/'CPU14_faithful_parent_interpreter_proof4_ordered_RAW_AST_v1.json',dict(kind='CPU4_watchdog2_whole_ordered_RAW_AST',CPU=cpu_ledger,watchdog=watch_ledger,top_function_deltas=dict(CPU=['fixture_body','main'],watchdog=['execute']),original_CPU_transport_and_first_fail_AST_unchanged=True,original_SOURCE_binding=pin_bytes(binding_path,br)))
ar=write_json(D/'Trackio7_negative_exact_STATE_proof5_ordered_RAW_AST_v1.json',dict(kind='API5_whole_ordered_RAW_AST',API=api_ledger,top_function_deltas=['server_child','fixture_body'],all_other_top_functions_AST_equal=True,scope=pin_bytes(scopepath,sr)))
common=dict(source_scope=pin_bytes(scopepath,sr),initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),candidate_import_execution=False,new_actual_execution_scope='UNKNOWN / REQUIRED',prior_CPU02=scope['ROOT_CPU02'],prior_API02=scope['ROOT_API02'])
ch=write_json(D/'CPU14_faithful_parent_interpreter_proof4_SOURCE_handoff_v1.json',dict(**common,kind='CPU4_typed_original_parent_INACTIVE_SOURCE_handoff_v1',candidate=pins['cpu'],watchdog=pins['watchdog'],RAW_AST=cr,original_SOURCE_interpreter_binding=pin_bytes(binding_path,br),fixture_body_python=scope['original_parent_python'],original_worker_python=scope['original_worker_python'],
proposed_top_level_prefix=[scope['original_worker_python']['path'],'-X','utf8','-B',str(paths['cpu']),'--ROOT-review','UNKNOWN'],
fresh_future_outputs=dict(body=str(D/'c_f03'),external=str(D/'c_e03')),
CPU14_unchanged=dict(bytes=36718,sha256='e3d01517ac3af9e481a27b0127474c7a8fbc2d9351f7fcc6aefcf36691675d0e'),runner_unchanged=dict(bytes=6296,sha256='7de36bc24022863bb7a819419dfaf6c49b646195ffa60a32cebafa94b52b7f0c'),
resource_limits_unchanged=dict(prospective_commit=8974029696,prospective_physical=6224029696,runtime_commit=6455594240,runtime_physical=5955594240,family_private=8000000000,worker_private=8000000000,watchdog_private=536870912,controller_private=536870912,body_wall=360,external_wall=420,packet_timeout=300,native_stall=600),
limitations=['Original production212 cause UNKNOWN','No package install, path injection, environment change, framework import or interpreter execution in SOURCE preparation','Scope must bind exact body and worker executable pins in future input_pins; same binary hash alone insufficient']))
ah=write_json(D/'Trackio7_negative_exact_STATE_proof5_SOURCE_handoff_v1.json',dict(**common,kind='API5_exact_negative_STATE_fences_INACTIVE_SOURCE_handoff_v1',candidate=pins['api'],RAW_AST=ar,
proposed_top_level_prefix=[str(P/'work/monitor-env/Scripts/python.exe'),'-X','utf8','-B',str(paths['api']),'--ROOT-review','UNKNOWN'],
fresh_future_outputs=dict(body=str(D/'a7_f03'),external=str(D/'a7_e03')),watchdog1_unchanged=scope['base_watchdog'],
reader3_helper3_projector1_unchanged=True,
observation_fences=dict(scientific_owner='METHOD_STUB only',max_wait_each_seconds=15,applies='only later STATE writes after injected fault until exact controller assertion and release',release_in_finally=True,no_HTTP_timeout_or_total_wall_increase=True),
resource_limits_unchanged=dict(prospective_commit=7989758336,prospective_physical=5239758336,runtime_commit=5471322880,runtime_physical=4971322880,family_private=2573741824,reader_private=1500000000,controller_private=536870912,watchdog_private=536870912,body_wall=160,external_wall=180,HTTP=20),
limitations=['API02 exact changed field/cause remains UNKNOWN','Fixture fence proves only tested invariant under injected fault, no production212 cause or durable throughput claim']))
prior=[]
for identity in [dict(pid=26148,ctime=1792862766.459404,ppid=25944),dict(pid=25944,ctime=1792862766.4470067,ppid=20996)]:
    try:q=psutil.Process(identity['pid']);same=q.create_time()==identity['ctime']and q.ppid()==identity['ppid']
    except psutil.NoSuchProcess:same=False
    prior.append(dict(identity=identity,exact_dead=not same))
assert all(x['exact_dead']for x in prior)
final=sample()
assert final['commit']>=2612368384 and final['physical']>=2112368384 and final['private']<=536870912 and final['E_free']>=100000000000 and time.monotonic()-t0<=180
result=write_json(D/'CPUfaithful_APInegative_SOURCE_result_v1.json',dict(candidates=pins,CPU_RAW_AST=cr,API_RAW_AST=ar,CPU_handoff=ch,API_handoff=ah,initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),prior_API02_completion_exact_deaths=prior,final=final,candidate_import_execution=False,wall=time.monotonic()-t0))
print(json.dumps(dict(candidates=pins,CPU_RAW_AST=cr,API_RAW_AST=ar,CPU_handoff=ch,API_handoff=ah,result=result,initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()))))

