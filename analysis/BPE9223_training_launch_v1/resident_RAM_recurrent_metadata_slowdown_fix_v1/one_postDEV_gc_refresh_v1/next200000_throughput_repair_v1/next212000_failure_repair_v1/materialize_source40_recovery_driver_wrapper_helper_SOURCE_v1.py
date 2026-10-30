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
initial=write_json(D/'source40_recovery_driver_wrapper_helper_SOURCE_initial_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=5399239296,physical=2649239296,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
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

helper_head="\"\"\"INACTIVE coherent source39 saved212000 startup/reader recovery.\nOriginal science and CPU14 are unchanged; no restore/control occurs here.\n\"\"\"\nimport copy\nfrom pathlib import Path\nimport psutil\nfrom atomic_io_retry_v1 import read_json\nfrom bpe_broad_audio_v1 import record,require\n\nP=Path(r\"C:/Users/ardao/Documents/Codex/2026-10-10/i-x20\")\nS=P/\"outputs/parakeet_turkish\"\nROOT=Path(r\"E:/gpu_training_scratchpad\")\nBRANCH=ROOT/\"runs/BPE9223_resident_RAM_20261018_01\"\nW=Path(r\"E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1\")\nX=Path(__file__).resolve().parent\nPOLICY=\"authoritative_current_plan_identity_and_main_path_attribution_policy\"\nPRIOR=\"prior_resident_RAM_source37_postDEV204000_throughput_transition\"\nREPAIR=\"original_CPU14_startup_reader7_recovery_policy\"\nFAILURE=\"prior_resident_RAM_source39_startup212000_failure\"\nEXTRA_HOST_BYTES=(72+8)*1024**2\n\ndef exact_dead(who):\n    try:return abs(psutil.Process(who[\"pid\"]).create_time()-who[\"create_time\"])>=.001\n    except psutil.NoSuchProcess:return True\n\ndef legacy_source37_configuration(base):\n    require(record(BASE_SOURCE[\"path\"])==BASE_SOURCE,\"Exact accepted source39 SOURCE dictionary\")\n    baseline=read_json(BASE_SOURCE[\"path\"])\n    candidate=copy.deepcopy(base)\n    require(REPAIR in candidate and FAILURE in candidate,\"Explicit new startup/reader policy and actual212000 failure\")\n    del candidate[REPAIR];del candidate[FAILURE]\n    require(candidate==baseline,\"ONLY two new operational fields; all original source39 science/LR/GC/225/FULL exact\")\n    proper=copy.deepcopy(baseline)\n    del proper[POLICY];del proper[PRIOR]\n    return proper\n\ndef checked_startup_reader_repair_policy(base):\n    p=base[REPAIR];cpu=p[\"CPU_startup_policy\"]\n    require(p[\"kind\"]==\"ASR18_source40_original_CPU14_startup_reader7_recovery_v1\"\n        and cpu[\"kind\"]==\"ASR18_original_CPU14_startup_receipts_launch_adapter_v1\"\n        and cpu[\"adapter_source\"]==record(X/\"CPU14_startup_launch_adapter_v15.py\")\n        and cpu[\"runner_source\"]==record(X/\"CPU14_owned_startup_only_receipt_runner_v2.py\")\n        and cpu[\"original_CPU14_source\"]==record(S/\"fastmss_resident_feature_stream_v14.py\")\n        and cpu[\"additional_host_reservation_bytes\"]==8*1024**2\n        and cpu[\"maximum_receipt_bytes\"]==1024**2 and cpu[\"four_coarse_hooks_once\"]is True\n        and cpu[\"no_per_update_observer\"]is True\n        and p[\"additional_host_reservation_bytes\"]==8*1024**2\n        and p[\"total_additional_native_host_bytes\"]==EXTRA_HOST_BYTES,\n        \"Exact startup-only adapter/runner and separately charged8MiB ABOVE original72; no privatecap or science change\")\n    require(p[\"reader_template\"]==READER_TEMPLATE and record(READER_TEMPLATE[\"path\"])==READER_TEMPLATE\n        and p[\"projection_source\"]==PROJECTION and record(PROJECTION[\"path\"])==PROJECTION\n        and p[\"API_helper_source\"]==API_HELPER and record(API_HELPER[\"path\"])==API_HELPER\n        and p[\"API_both_shapes_ROOT\"]==API_ROOT and record(API_ROOT[\"path\"])==API_ROOT,\n        \"Exact tested reader5/helper4/projector2 plus immutable ROOT both-shapes authority\")\n    proof_pin=cpu[\"actual_new_adapter_CPU_control_review\"]\n    require(type(proof_pin)is dict and record(proof_pin[\"path\"])==proof_pin,\n        \"Genuine completed NEW adapter CPU control review required; SOURCE-only gate cannot launch\")\n    proof=read_json(proof_pin[\"path\"])\n    require(proof[\"kind\"]==\"ASR18_original_CPU14_startup_adapter_actual_CPU_v1\"\n        and proof[\"accepted\"]is True and proof[\"reviewer\"]==\"/root\"\n        and proof[\"adapter_source\"]==cpu[\"adapter_source\"]and proof[\"runner_source\"]==cpu[\"runner_source\"]\n        and proof[\"CPU14_source\"]==cpu[\"original_CPU14_source\"]\n        and proof[\"full_first_packet_passed\"]is True\n        and proof[\"original_CPU14_worker_and_transport_unchanged\"]is True\n        and proof[\"all_exact_known_identities_dead\"]is True,\n        \"New launch-path FULL original firstpacket/lifetime/closure actual proof, not old CPU03 coverage waiver\")\n    reader_pin=p[\"actual_reader_replacement_ROOT\"]\n    require(type(reader_pin)is dict and record(reader_pin[\"path\"])==reader_pin,\n        \"Genuine separately admitted actual reader replacement ROOT required\")\n    reader=read_json(reader_pin[\"path\"])\n    require(reader[\"kind\"]==\"ROOT_reader7_actual_replacement_acceptance_v1\"\n        and reader[\"accepted\"]is True and reader[\"reviewer\"]==\"/root\"\n        and reader[\"template_source\"]==READER_TEMPLATE\n        and reader[\"projection_source\"]==PROJECTION and reader[\"API_helper_source\"]==API_HELPER\n        and reader[\"old_exact_pair_dead\"]is True and reader[\"mirror_thread_alive\"]is True\n        and reader[\"HTTP_ready\"]is True and reader[\"reader_ready\"]is True and reader[\"error\"]is None,\n        \"Actual new reader ready/thread-safe/current-journal acceptance before training\")\n    for identity in (reader[\"new_reader_identity\"],reader[\"new_reader_launcher_identity\"]):\n        observed=psutil.Process(identity[\"pid\"])\n        require(abs(observed.create_time()-identity[\"ctime\"])<.001 and observed.ppid()==identity[\"ppid\"],\n            \"Exact accepted new reader pair remains alive\")\n    status=read_json(reader[\"status_path\"])\n    require(status[\"pid\"]==reader[\"new_reader_identity\"][\"pid\"]\n        and status[\"create_time\"]==reader[\"new_reader_identity\"][\"ctime\"]\n        and status[\"mirror_thread_alive\"]is True and status[\"HTTP_ready\"]is True\n        and status[\"reader_ready\"]is True and status[\"error\"]is None,\n        \"Current local mirror/thread/HTTP health, processalive alone is insufficient\")\n    return p\n\n"
helper_resume="\ndef checked_current_plan_throughput_checkpoint_resume(pin,*,base,folder,pointer_pin):\n    require(Path(folder).resolve()==BRANCH.resolve()and pin==base[FAILURE]==FAILURE_ROOT\n        and record(pin[\"path\"])==pin,\"Exact accepted actual source39 startup212000 failure authority\")\n    checked_current_plan_throughput_policy(base)\n    value=read_json(pin[\"path\"])\n    require(value[\"kind\"]==\"ROOT_NEW212000_production_and_reader_failure_independent_seal_v1\"\n        and value[\"accepted\"]is True and value[\"reviewer\"]==\"/root\"\n        and value[\"actual212000_saved212000_gap0\"]is True\n        and value[\"production76082_consumed_ONCE_only_by_owner\"]is True\n        and value[\"native_primary_recorded_failure\"]==\"TimeoutError('Resident CPU update packet timed out')\"\n        and value[\"underlying_new_CPU_startup_cause\"]==\"UNKNOWN\"\n        and value[\"separate_native_CPU_wrapper_numeric_OS_exits\"]==\"UNRECORDED_UNKNOWN\",\n        \"Honest original timeout/gap0/unknown primary startup cause; no normal0 or cause fabrication\")\n    deaths=value[\"all_recorded_old_runtime_control_sealer_identities_exact_dead\"]\n    require(len(deaths)==12 and all(exact_dead(i)for i in deaths)\n        and all(any(i==expected for i in deaths)for expected in REQUIRED_OLD_IDENTITIES),\n        \"ALL recorded old runtime/control/sealer exactdead, including old and newCPU; readers are a separate reviewed lane\")\n    require(record(SAVED_ROOT[\"path\"])==SAVED_ROOT,\"Exact ROOT saved212000 full854/PRE-SAVED authority\")\n    saved=read_json(SAVED_ROOT[\"path\"])\n    require(saved[\"accepted\"]is True and saved[\"reviewer\"]==\"/root\"and saved[\"updates\"]==212000\n        and saved[\"weights\"]==207 and saved[\"Adam_states\"]==215\n        and saved[\"Adam_partitions\"]==[17,6,192]and saved[\"all4RNG\"]is True\n        and saved[\"named_tensor_references\"]==854\n        and saved[\"integer_Adam_JSON_keys_only_normalized\"]is True\n        and saved[\"semantic_identity_sampler_journal_nextdraw_equal\"]is True,\n        \"Original full207/215/3Adam/4RNG/nextdraw and canonical equality; no new payloadhash/GPUrestore claim\")\n    require(pointer_pin==SOURCE_DERIVED_POINTER and record(pointer_pin[\"path\"])==pointer_pin,\n        \"SOURCE-derived exact pointer bytes/hash checked BEFORE future restore\")\n    pointer=read_json(pointer_pin[\"path\"])\n    require(pointer[\"committed_updates\"]==212000 and pointer[\"diagnostic_only\"]is False\n        and pointer[\"all_window_updates_committed\"]is True,\"Full coherent saved212000 pointer\")\n    prefix=pointer[\"journal_prefix\"]\n    require(prefix[\"offset\"]==426026870\n        and prefix[\"sha256\"]==\"6621855eb4191790c3a741e3e1c19014f00db87b9e471d3a4c70802cbddaaf0a\"\n        and Path(prefix[\"path\"]).resolve().is_relative_to(BRANCH.resolve()),\n        \"Exact published saved J27prefix authority, never freshly claimed full-journal hash\")\n    expected=base[REPAIR][\"closed_original_journal_STAT\"]\n    require(type(expected)is dict and expected[\"kind\"]==\"STAT_ONLY_original_closed_journal_v1\"\n        and expected[\"path\"]==prefix[\"path\"]and expected[\"bytes\"]==prefix[\"offset\"],\n        \"Final ROOT-bound exact original J27 STAT required; no guessed mtime or prefix reread\")\n    actual=Path(expected[\"path\"]).stat()\n    require(actual.st_size==expected[\"bytes\"]and actual.st_mtime_ns==expected[\"mtime_ns\"],\n        \"Closed original journal STAT ONLY; no body/hash/replay/tail promotion\")\n    reader=read_json(base[REPAIR][\"actual_reader_replacement_ROOT\"][\"path\"])\n    require(reader[\"journal_path\"]==prefix[\"path\"]and reader[\"journal_offset\"]==prefix[\"offset\"]\n        and reader[\"verified_updates\"]>=212000,\"Accepted reader on saved J27 before new J28 handoff\")\n    return dict(saved_journal_prefix=prefix,whole_journal_including_preserved_tail=expected,\n        preserved_tail_positive_updates=0,unsaved_tail_bytes=0,actual_transition=pin,\n        full_checkpoint_restore_and_genuine_redo_required=True,\n        honest_original_optimizer_vs_journal_gap=0,no_unsaved_positive_updates_after212000=True)\n"

scope_path=D/'ROOT_source40_reader7_saved212000_INACTIVE_SOURCE_preparation_scope_v2.json'
scope=json.loads(scope_path.read_text(encoding='utf-8'))
assert scope['accepted']is True and scope['actual_execution_admitted']is False
W=D.parent;S=P/'outputs/parakeet_turkish';BRANCH=pathlib.Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
paths={k:D/v for k,v in dict(helper='asr18_current_plan_throughput_checkpoint_transition_v5.py',driver='run_ASR18_BPE9223_resident_RAM_training_v40.py',wrapper='run_ASR18_BPE9223_resident_RAM_autocontinue_v41.py',configuration='source40_saved212000_INACTIVE_configuration_v1.json',raw='source40_recovery_driver_wrapper_helper_full_RAW_AST_v1.json',handoff='source40_recovery_driver_wrapper_helper_SOURCE_handoff_v1.json',result='source40_recovery_driver_wrapper_helper_SOURCE_result_v1.json').items()}
assert not any(p.exists()for p in paths.values())
def get_bound(label):
    q=scope[label];p=pathlib.Path(q['path']);b=p.read_bytes()
    assert len(b)==q['bytes']and sha(b)==q['sha256'];return p,b
driver_base,db=get_bound('base_driver');wrapper_base,wb=get_bound('base_wrapper');config_base,cb=get_bound('original_source_configuration')
helper_base=W/'asr18_current_plan_throughput_checkpoint_transition_v4.py';hb=helper_base.read_bytes()
assert len(hb)==13808 and sha(hb)=='c59f522b046bbfd1b7c44b989f24521812bd90818e79824f86f6f20d92f71877'
helper_ast=ast.parse(hb.decode('utf-8-sig'));old_policy=next(n for n in helper_ast.body if isinstance(n,ast.FunctionDef)and n.name=='checked_current_plan_throughput_policy')
old_policy_text='\n'.join(hb.decode('utf-8-sig').splitlines()[old_policy.lineno-1:old_policy.end_lineno])+'\n'
assert old_policy_text.count("policy['additional_host_reservation_bytes'] == EXTRA_HOST_BYTES")==1
old_policy_text=old_policy_text.replace("policy['additional_host_reservation_bytes'] == EXTRA_HOST_BYTES","policy['additional_host_reservation_bytes'] == 72*1024**2")
old_policy_text=old_policy_text.replace('    from current_plan_identity_view_v3 import','    checked_startup_reader_repair_policy(base)\n    from current_plan_identity_view_v3 import')
pointer=dict(path=str(BRANCH/'checkpoints/update_000212000/pointer.json'),bytes=16120,sha256='9bef17e4fd0ed37f34f056595ca09ec91d2a69d80786165d6ca866b270bcbe07')
root_failure=scope['ROOT_failure212000'];root_saved=scope['ROOT_saved212000']
required_old=[dict(pid=34788,create_time=1792842518.865024,ppid=32332),dict(pid=32332,create_time=1792842518.85846,ppid=4964),dict(pid=4964,create_time=1792842518.480302,ppid=6664),dict(pid=6664,create_time=1792842518.4723651,ppid=37480),dict(pid=23496,create_time=1792848390.67016,ppid=39100),dict(pid=39100,create_time=1792848390.6293392,ppid=34788),dict(pid=33668,create_time=1792855655.0849347,ppid=34788)]
constants=dict(BASE_SOURCE=pin_bytes(config_base,cb),READER_TEMPLATE=scope['new_reader'],PROJECTION=scope['projection'],API_HELPER=scope['API_helper'],API_ROOT=scope['ROOT_API_both_shapes'],FAILURE_ROOT=root_failure,SAVED_ROOT=root_saved,SOURCE_DERIVED_POINTER=pointer,REQUIRED_OLD_IDENTITIES=required_old)
constant_text='\n'.join(k+'='+repr(v)for k,v in constants.items())+'\n\n'
helper=(helper_head+constant_text+old_policy_text+'\n'+helper_resume).encode()
# Literal replacements run against the exact ORIGINAL bytes with explicit counts.
def replacements(b,edits):
    text=b.decode('utf-8-sig')
    for old,new,count in edits:
        assert text.count(old)==count,(old,text.count(old),count)
        text=text.replace(old,new)
    return text.encode('utf-8')
explicitP='P=Path(r"C:/Users/ardao/Documents/Codex/2026-10-10/i-x20");S=P/\'outputs/parakeet_turkish\''
oldfactory='        from fastmss_resident_feature_stream_v14 import ResidentFeatureProducer\n        resident_source=record(inspect.getsourcefile(ResidentFeatureProducer))'
newfactory='''        import CPU14_startup_launch_adapter_v15 as startup_adapter
        startup_adapter.configure_startup_receipts(candidate_base['original_CPU14_startup_reader7_recovery_policy']['CPU_startup_policy'],record(config_path))
        ResidentFeatureProducer=startup_adapter.ResidentFeatureProducer
        resident_source=record(S/'fastmss_resident_feature_stream_v14.py')
        write_json(invocation/'actual_CPU_launch_adapter_binding.json',{'original_worker_source':resident_source,
            'launch_adapter_source':record(inspect.getsourcefile(startup_adapter)),
            'startup_runner_source':candidate_base['original_CPU14_startup_reader7_recovery_policy']['CPU_startup_policy']['runner_source'],
            'startup_only_no_per_update_observer':True,'additional_host_bytes_above_original72':8*1024**2})'''
de=[
("P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish'",explicitP,1),
('sys.path.insert(0,str(THROUGHPUT_FOLDER))',"sys.path.insert(0,str(THROUGHPUT_FOLDER))\nRECOVERY_FOLDER=Path(__file__).resolve().parent\nsys.path.insert(0,str(RECOVERY_FOLDER))",1),
('asr18_current_plan_throughput_checkpoint_transition_v4','asr18_current_plan_throughput_checkpoint_transition_v5',1),
('throughput_active and a.invocation==27','throughput_active and a.invocation==28',1),
('Only exact source37 post204000DEV cooperative stop, never normal0 or an old failure lane','Only exact accepted abnormal source39 saved212000 startup failure, never normal0 or an old failure lane',1),
('Exact actual204800 full saved prefix; all later tail historical and genuinely redone','Exact saved212000 full prefix; no unsaved positives or tail promotion',1),
('ASR18_readonly_saved204800_prefix_after_source37_throughput_transition_v1','ASR18_readonly_saved212000_prefix_after_source39_startup_failure_v1',1),
(oldfactory,newfactory,1)]
driver=replacements(db,de)
we=[
("P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish'",explicitP,1),
('sys.path.insert(0,str(THROUGHPUT_FOLDER))',"sys.path.insert(0,str(THROUGHPUT_FOLDER))\nRECOVERY_FOLDER=Path(__file__).resolve().parent\nsys.path.insert(0,str(RECOVERY_FOLDER))",1),
("driver=P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v39.py'","driver=RECOVERY_FOLDER/'run_ASR18_BPE9223_resident_RAM_training_v40.py'",1),
('asr18_current_plan_throughput_checkpoint_transition_v4','asr18_current_plan_throughput_checkpoint_transition_v5',2),
("record(THROUGHPUT_FOLDER/'asr18_current_plan_throughput_checkpoint_transition_v5.py')","record(RECOVERY_FOLDER/'asr18_current_plan_throughput_checkpoint_transition_v5.py')",1),
("and a.start_invocation==review['initial_owner_invocation']==27","and a.start_invocation==review['initial_owner_invocation']==28",1),
('Only reviewed current source37 post204000DEV cooperative checkpoint transition, no normal0 alias','Only reviewed coherent abnormal source39 saved212000 transition, no normal0 alias',1),
('run_ASR18_BPE9223_resident_RAM_training_v39','run_ASR18_BPE9223_resident_RAM_training_v40',14)]
# Explicit review binding for every NEW operational source and gate, before any GPU query.
extra="""    if throughput_active:
        repair=candidate_configuration['original_CPU14_startup_reader7_recovery_policy']
        require(review['CPU_startup_adapter_source']==repair['CPU_startup_policy']['adapter_source']
            and review['CPU_startup_runner_source']==repair['CPU_startup_policy']['runner_source']
            and review['actual_new_adapter_CPU_control_review']==repair['CPU_startup_policy']['actual_new_adapter_CPU_control_review']
            and review['actual_reader_replacement_ROOT']==repair['actual_reader_replacement_ROOT']
            and review['original_CPU14_startup_reader7_recovery_policy']==repair,
            'Exact reviewed new launch-path actual CPU and actual healthy reader gates')
"""
we.append(('    configuration=legacy_source37_configuration(candidate_configuration)if throughput_active else candidate_configuration',extra+'    configuration=legacy_source37_configuration(candidate_configuration)if throughput_active else candidate_configuration',1))
# Number of imports is source-derived, but value is not silently widened.
import_count=wb.decode('utf-8-sig').count('run_ASR18_BPE9223_resident_RAM_training_v39')-1
we[7]=(we[7][0],we[7][1],import_count)
wrapper=replacements(wb,we)
runner=pin(D/'CPU14_owned_startup_only_receipt_runner_v2.py');adapter=pin(D/'CPU14_startup_launch_adapter_v15.py')
cpu_source=pin(S/'fastmss_resident_feature_stream_v14.py')
config=json.loads(cb)
REPAIR='original_CPU14_startup_reader7_recovery_policy';FAILURE='prior_resident_RAM_source39_startup212000_failure'
assert REPAIR not in config and FAILURE not in config
config[REPAIR]=dict(kind='ASR18_source40_original_CPU14_startup_reader7_recovery_v1',
    additional_host_reservation_bytes=8*1024**2,total_additional_native_host_bytes=80*1024**2,
    CPU_startup_policy=dict(kind='ASR18_original_CPU14_startup_receipts_launch_adapter_v1',adapter_source=adapter,runner_source=runner,original_CPU14_source=cpu_source,
        worker_python_path=str(pathlib.Path('E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe').resolve()),
        additional_host_reservation_bytes=8*1024**2,maximum_receipt_bytes=1024**2,four_coarse_hooks_once=True,no_per_update_observer=True,
        actual_new_adapter_CPU_control_review=None,execution_review_path=None),
    reader_template=scope['new_reader'],projection_source=scope['projection'],API_helper_source=scope['API_helper'],
    API_both_shapes_ROOT=scope['ROOT_API_both_shapes'],actual_reader_replacement_ROOT=None,closed_original_journal_STAT=None)
config[FAILURE]=root_failure
inverse=dict(config);del inverse[REPAIR];del inverse[FAILURE];assert inverse==json.loads(cb)
cbytes=(json.dumps(config,ensure_ascii=False,indent=2)+'\n').encode()
pins={}
for k,b in [('helper',helper),('driver',driver),('wrapper',wrapper)]:pins[k]=write_source(paths[k],b)
with paths['configuration'].open('xb')as f:f.write(cbytes);f.flush();os.fsync(f.fileno())
pins['configuration']=pin_bytes(paths['configuration'],cbytes)
raws=[raw_diff(hb,helper,helper_base,paths['helper'],['SOURCE-only coherent abnormal212000 authority; future actual CPU/reader/STAT gates are None and reject','Original identity/phase policy72 preserved; extra8 separately increases dynamicnative reservation to80','SOURCE-derived pointer exactcheck BEFORE any future restore; no closed journal body hash']),
raw_diff(db,driver,driver_base,paths['driver'],[x[:2]for x in de]),raw_diff(wb,wrapper,wrapper_base,paths['wrapper'],[x[:2]for x in we])]
# Every unchanged top-level/nested definition is reported by complete original and candidate AST hashes.
def ast_nodes(b):
    tree=ast.parse(b.decode('utf-8-sig'))
    return {getattr(n,'name',str(i)):sha(ast.dump(n,include_attributes=False).encode())for i,n in enumerate(tree.body)}
rawpin=write_json(paths['raw'],dict(kind='ASR18_source40_recovery_complete_ordered_RAW_AST_SOURCE_v1',records=raws,whole_byte_inverse_all_exact=True,
    driver_original_nodes=ast_nodes(db),driver_candidate_nodes=ast_nodes(driver),wrapper_original_nodes=ast_nodes(wb),wrapper_candidate_nodes=ast_nodes(wrapper),
    configuration_only_added_fields=[REPAIR,FAILURE],configuration_inverse_dict_exact=True,
    configuration_original_bytes=pin_bytes(config_base,cb),configuration_candidate= pins['configuration']))
pressure_lines=[dict(line=i+1,source=x)for i,x in enumerate(driver.decode().splitlines())if 'throughput_reserve'in x or 'additional_bookkeeping_reservation_bytes'in x]
unknowns=dict(actual_new_adapter_CPU_control_review=None,actual_reader_replacement_ROOT=None,original_journal_STAT_mtime_ns=None,final_execution_review=None,
    current_actual_reader_health='FAILED reader6 mirror; no newreader activation yet',new_production_identities=None,production_latest_failure_ACK260_exact_path=None,
    new_launch_path_actual_firstpacket_proof='UNKNOWN: SOURCE-only; original CPU03 is provenance, not substituted actual control')
handoff=write_json(paths['handoff'],dict(kind='ASR18_source40_driver_wrapper_helper_INACTIVE_SOURCE_handoff_v1',SOURCE_only=True,actual_execution_admitted=False,
    preparation_scope=pin(scope_path),sources=pins,RAW=rawpin,startup_sources=dict(adapter=adapter,runner=runner,original_CPU14=cpu_source),
    ROOT_authorities={k:scope[k]for k in ('ROOT_API_both_shapes','ROOT_faithful_CPU03','ROOT_saved212000','ROOT_failure212000','ROOT_next_plan361')},
    pointer_SOURCE_derivation=dict(proposed_pointer=pointer,original_body_read=False,exact_BRANCH_operand=str(BRANCH),
        expressions=[dict(driver39_line=1295,text="ck=folder/'checkpoints'/('update_'+str(optimizer.updates).zfill(9));ck.mkdir()"),
                     dict(driver39_line=1312,text="coordinator.publish_checkpoint(...,pointer_path=ck/'pointer.json',...)"),
                     dict(driver39_line=1317,text="latest.pointer=record(ck/'pointer.json')")],
        future_actual_scope_must_check='bytes/SHA/committed212000/diagnosticFalse/allcommitted/fullstate before restore'),
    proposed_future_targets=dict(invocation=str(BRANCH/'invocations/owner_000028'),journal='freshJ28 path from original source; no overwrite',control='freshA30 path to be ROOT-bound'),
    proposed_wrapper_prefix=[str(P/'work/nemo-native-env-v2/Scripts/python.exe'),'-X','utf8','-B',str(paths['wrapper'])],
    proposed_wrapper_arguments_template=['--review',None,'--configuration',str(paths['configuration']),'--pointer',pointer['path'],'--latest-failure',None,'--branch',str(BRANCH),'--output',None,'--start-invocation','28','--throughput-resume',root_failure['path']],
    startup_coverage=dict(original_worker14_AST_unchanged=True,original_P_parent_E_worker_paths_unchanged=True,
        private_method_globals_launch_adapter_new_path=True,new_actual_proof_required=True,four_coarse_hooks_detach_before_first_original_call=True,no_per_update_wrapper_clock=True,
        full_firstpacket16_and_original_speculation_up_to3_packets48_audio_rows='future separately scoped proof; no audio read here'),
    reservations=dict(original_native225_and_FULL_unchanged=True,old_identity_phase_bytes=72*1024**2,new_startup_extra=8*1024**2,native_total_extra=80*1024**2,
        reader_cache256=256*1024**2,reader_API16=16*1024**2,reader_projection64=64*1024**2,reader_private_cap=1500000000,CPU_worker_private_cap=8000000000,
        all_original_self_family_time_limits_unchanged=True,actual_new_path_fit='UNKNOWN until separately admitted full-family proof',driver_dynamic_reservation_call_sites=pressure_lines),
    preservation=dict(original_three_fresh_canonical_digest_calls=True,original11_phase_scope_unchanged=True,source39_identity_policy_unchanged=True,full401072_adminNone=True,
        consumed180k_zero_second_calls=True,science_LR_207_Adam215_3_RNG4_unchanged=True,fullrestore212000_before_savedJ27prefix_copy_and_zero_native_draw_required=True,
        saved_published_prefix=dict(offset=426026870,sha256='6621855eb4191790c3a741e3e1c19014f00db87b9e471d3a4c70802cbddaaf0a'),published_authority_not_new_body_hash=True),
    unknown_FAIL_CLOSED_gates=unknowns,remaining_SOURCE_tasks=['new actual adapter firstpacket proof source','reader replacement actor source','typed/integration metadata fixture source'],
    entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),initial=initial))
final_sample=sample();assert time.monotonic()-t0<180 and final_sample['private']<=536870912 and final_sample['commit']>=2612368384 and final_sample['physical']>=2112368384 and final_sample['E_free']>=100000000000
result=write_json(paths['result'],dict(kind='SOURCE_only_materialization_result',passed=True,sources=pins,RAW=rawpin,handoff=handoff,initial=initial,
    entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),final_sample=final_sample,elapsed=time.monotonic()-t0,no_import_or_execution=True,
    direct_numeric_OS_exit='not yet recorded; shell must persist actual'))
print(json.dumps(dict(result=result,handoff=handoff,RAW=rawpin,sources=pins,initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent())),indent=2))

