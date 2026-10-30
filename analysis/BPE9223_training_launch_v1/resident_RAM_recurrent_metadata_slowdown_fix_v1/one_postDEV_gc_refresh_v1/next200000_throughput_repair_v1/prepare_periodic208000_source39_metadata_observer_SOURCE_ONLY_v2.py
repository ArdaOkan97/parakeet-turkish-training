import ast, hashlib, json, os, pathlib, shutil, sys, time, traceback
import psutil
W=pathlib.Path(__file__).resolve().parent
D=W.parent;P=pathlib.Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from resource_watchdog_v2 import sample_memory
BASE=D/'seal_periodic196000_source37_CPU14_helper3_max2_saved196200_v1.py'
NEW=W/'seal_periodic208000_source39_CPU14_helper3_max2_saved208200_v1.py'
RAW=W/'periodic208000_source39_metadata_observer_full_RAW_AST_v1.json'
HANDOFF=W/'periodic208000_source39_CPU14_metadata_observer_INACTIVE_handoff_v1.json'
RESULT=W/'periodic208000_source39_metadata_observer_source_preparation_result_v2.json'
INITIAL=W/'periodic208000_source39_metadata_observer_source_preparation_initial_v2.json'
def require(v,m):
    if not v:raise ValueError(m)
def pin(p):
    p=pathlib.Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
    with p.open('x',encoding='utf-8',newline='\n')as f:
        json.dump(v,f,ensure_ascii=False,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    return pin(p)
def identity(p):return dict(pid=p.pid,create_time=p.create_time(),ppid=p.ppid())
me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
entry=dict(identity=identity(me),launcher=identity(me.parent()),before_predicate=True)
began=time.monotonic();initial=sample_memory(me.pid,'periodic208000_SOURCE_ONLY_preparation')
write(INITIAL,dict(entry=entry,sample=initial,E_free=shutil.disk_usage('E:/gpu_training_scratchpad').free,TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR')))
result=dict(kind='ASR18_periodic208000_source39_metadata_observer_SOURCE_ONLY_preparation_v1',entry=entry,initial=pin(INITIAL),outputs={},error=None)
try:
    require(initial['system_memory']['commit_available_bytes']>=5130803840 and initial['system_memory']['physical_available_bytes']>=2380803840,'Original full prospective metadata512 plus separately charged source39 reserve72')
    require(me.memory_info().private<536870912 and shutil.disk_usage('E:/gpu_training_scratchpad').free>=100000000000,'Original self512/E100GB')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'Original isolated CPU caps')
    require(os.environ.get('TORCH_EXTENSIONS_DIR')=='E:\\gpu_training_scratchpad\\cache\\torch_extensions','Exact dot-sourced storage policy')
    require(all(not p.exists()for p in (NEW,RAW,HANDOFF,RESULT)),'Fresh inactive outputs')
    base=BASE.read_bytes();base_pin=pin(BASE)
    require(base_pin['sha256'].startswith('3375d6a1')and len(base)==18929,'Accepted entire196k SOURCE base only')
    text=base.decode('utf-8');edits=[]
    newline='\r\n'if base.count(b'\r\n')==base.count(b'\n')else'\n'
    def replace(old,new,meaning,count=None):
        global text
        old=old.replace('\r\n','\n').replace('\n',newline)
        new=new.replace('\r\n','\n').replace('\n',newline)
        current_bytes=text.encode('utf-8');old_bytes=old.encode('utf-8');new_bytes=new.encode('utf-8')
        n=current_bytes.count(old_bytes)
        require(n>0 and (count is None or n==count),'Exact declared source literal count: '+meaning)
        application_byte_offsets=[];search_from=0
        for unused in range(n):
            offset=current_bytes.index(old_bytes,search_from)
            current_bytes=current_bytes[:offset]+new_bytes+current_bytes[offset+len(old_bytes):]
            application_byte_offsets.append(offset);search_from=offset+len(new_bytes)
        text=current_bytes.decode('utf-8')
        edits.append(dict(old=old,new=new,count=n,meaning=meaning,application_byte_offsets=application_byte_offsets))
    replace('actual196k full350, source37','actual208k full350, source39','New genuine periodic target and current driver',1)
    replace("I=B/'invocations/owner_000026';A=Path(__file__).parent.parent.parent/'resident_RAM_training_actual_27'",
        "I=B/'invocations/owner_000027';A=Path(r'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_training_actual_29')",
        'Actual I27/A29; explicit absolute A because successor source lives one directory deeper',1)
    replace('196000','208000','All target milestone literals, paths, kinds and arithmetic')
    replace('196200','208200','All natural next-checkpoint literals and paths')
    replace('source37','source39','All current-source guard/ROOT key/messages and source filename literals')
    replace('fullpass_binding','startup_binding','Truthful startup-cohort local naming throughout')
    start=text.index('# A separate future ROOT gate binds')
    stop=text.index("configuration_pin=pin(I/'configuration.json')",start)
    old=text[start:stop]
    new="""# Exact future ROOT scope reuses closed source39 boot/startup and natural205000
# receipts. The actual204800 restore is NOT a prior periodic/fullpass phase.
execution_review=read(args.ROOT_review)
require(execution_review['kind']=='ROOT_periodic208000_source39_CPU14_metadata_only_exact_execution_scope_v1'
 and execution_review['accepted']is True and execution_review['reviewer']=='/root'
 and execution_review['metadata_source']==pin(Path(__file__)),
 'Separate exact ROOT execution binding for this immutable source')
startup_binding=execution_review['actual_startup_cohort']
require(startup_binding['restored_optimizer_updates']==204800
 and startup_binding['cohort_start_optimizer_updates']==204800,
 'Actual source39 restored204800 startup cohort, not an invented prior periodic/fullpass')
boot_ROOT_pin=startup_binding['ROOT_source39_boot_review']
natural205000_ROOT_pin=startup_binding['ROOT_natural205000_review']
require(pin(boot_ROOT_pin['path'])==boot_ROOT_pin and boot_ROOT_pin['bytes']==9500
 and boot_ROOT_pin['sha256']=='48b11594d36cddae4f3144b52bee03204a84c61e88d75f6398e6e676edd08b56',
 'Exact accepted boot ROOT receipt only; closed actual startup/ready not reopened')
require(pin(natural205000_ROOT_pin['path'])==natural205000_ROOT_pin and natural205000_ROOT_pin['bytes']==2887
 and natural205000_ROOT_pin['sha256']=='ad42b4335e8ae3364284c0c403246a0485fb9439bcdae1f843c626c667be2b9c',
 'Exact natural205000 ROOT receipt only; closed checkpoint/quality not reopened')
for root_pin in (boot_ROOT_pin,natural205000_ROOT_pin):
 root_review=read(root_pin['path']);require(root_review['accepted']is True and root_review['reviewer']=='/root','Genuine completed ROOT gate')
require(startup_binding['pre_periodic_CPU']=={'pid':16008,'create_time':1792842778.8873637,'ppid':41760}
 and startup_binding['pre_periodic_CPU_launcher']=={'pid':41760,'create_time':1792842778.867194,'ppid':34788},
 'Frozen actual source39 startup identities; no automatic rebind')
pre_periodic_ready_path=I/'CPU/actual_CPU_ready.json'
pre_periodic_ready_pin={'path':str(pre_periodic_ready_path),'bytes':10817,'sha256':'5b4ed5865d52fdbee57a6a9cf1e3c2a45a218a51024fcfd4b4998b5da6d0f7cf'}
require(startup_binding['pre_periodic_CPU_ready']==pre_periodic_ready_pin,
 'Scalar immutable ready witness reusing closed boot ROOT; no closed ready read')
require(startup_binding['pre_periodic_CPU']['ppid']==startup_binding['pre_periodic_CPU_launcher']['pid'],
 'Exact native CPU ancestry in the ROOT-bound startup cohort')
"""
    replace(old,new,'Replace prior192k periodic/fullpass gate with genuine204800 startup + accepted boot/natural205000 and closed scalar ready witness',1)
    replace("'bytes':226990,\n 'sha256':'39dc6a3aa4b7139d6209751de769d5aef63a507e7122f56a6720afd85bdb966d'",
        "'bytes':229576,\n 'sha256':'e8d7d6262d50dada9907343947ed7e8cb8e63a650d656f4a735be88bf265690c'",
        'Actual immutable I27 runtime configuration',1)
    start=text.index("boot_ROOT_pin=execution_review['source39_actual_boot_ROOT']")
    stop=text.index("state=read(B/'periodic_DEV.json')",start)
    old=text[start:stop]
    new="""require(execution_review['source39_actual_boot_ROOT']==boot_ROOT_pin,'Same accepted source39 boot authority')
inherited_pin={'path':str(I/'actual_inherited_consumed180000_refresh_opportunity.json'),'bytes':2059,
 'sha256':'fb540868deb3111ce6e0581e24ee292e159503067dcd6035835129d183297853'}
require(inherited_pin==execution_review['inherited_consumed180000_receipt'],
 'Closed accepted inherited180k consumed witness only; no second event or actual receipt reread')
"""
    replace(old,new,'Reuse accepted boot/inherited scalar witnesses without reopening closed actual boot/event metadata',1)
    old="previous=[v for v in state['completed_evaluations']if v['optimizer_updates']==192000];require(len(previous)==1,'One previous192k quality record');previous_errors=previous[0]['score']['word_errors']['tr']"
    new="""previous_quality=execution_review['previous_periodic_quality']
require(previous_quality['optimizer_updates']==204000 and previous_quality['TR_word_errors']==205 and previous_quality['reference_words']==825,
 'Original accepted previous204k quality scalar from separate exact ROOT scope')
previous_quality_ROOT_pin=previous_quality['ROOT_review']
require(pin(previous_quality_ROOT_pin['path'])==previous_quality_ROOT_pin,'Exact accepted prior quality ROOT receipt only')
previous_quality_ROOT=read(previous_quality_ROOT_pin['path'])
require(previous_quality_ROOT['accepted']is True and previous_quality_ROOT['reviewer']=='/root','No closed previous quality report/prediction reread')
previous_errors=previous_quality['TR_word_errors']"""
    replace(old,new,'No old192/204 quality state/report reopen; exact accepted204k scalar205 supplied by future ROOT scope',1)
    replace("who_old==startup_binding['pre_periodic_CPU'],\n 'Periodic closure belongs to the exact admitted192k successor cohort'",
        "who_old==startup_binding['pre_periodic_CPU'],\n 'Periodic closure belongs to the exact accepted source39 restored204800 startup cohort'",
        'Actual source39 old CPU cohort label',1)
    replace('Exact admitted192k successor CPU launcher dead','Exact source39 startup CPU launcher dead','Truthful source39 launcher provenance',1)
    old="newready_path=I/'CPU_after_periodic_DEV_000208000/actual_CPU_ready.json';newready=read(newready_path);CPU=newready['producer_identity']"
    new=old+"""
require(CPU==execution_review['post_periodic_CPU']and newready['source']['protocol']==pin(S/'fastmss_resident_feature_stream_v14.py'),
 'Genuine post208 CPU identity and source frozen by separate ROOT execution scope, never invented now')
new_CPU_launcher=execution_review['post_periodic_CPU_launcher']
require(CPU['ppid']==new_CPU_launcher['pid']and new_CPU_launcher['ppid']==34788 and exact_alive(new_CPU_launcher)
 and psutil.Process(CPU['pid']).ppid()==new_CPU_launcher['pid'],'Exact post208 CPU launcher/body ancestry')"""
    replace(old,new,'Freeze actual future post208 identity/ancestry in future scope instead of auto-binding ready producer',1)
    replace("progress['actual_optimizer_updates']>208200","progress['actual_optimizer_updates']>=208220",'Required actual post208220 positive frontier',1)
    replace("native['pid']==19320 and abs(native['create_time']-1792803517.7236164)<.001",
        "native['pid']==34788 and abs(native['create_time']-1792842518.865024)<.001",'Same actual current source39 native identity',1)
    replace("'owned_session':26493","'owned_session':76082",'Current sole-owned production session',1)
    replace("'TR_change_vs192000'","'TR_change_vs204000'",'Quality comparison uses accepted last204000 periodic only',1)
    old="'actual_prior_periodic_cohort':{'ROOT_review':prior_periodic_ROOT_pin,'origin_periodic_optimizer_updates':192000,'cohort_start_optimizer_updates':192000,'pre_periodic_CPU_ready':pin(pre_periodic_ready_path),'cohort_binding':startup_binding}"
    new="'actual_startup_cohort':{'ROOT_boot_review':boot_ROOT_pin,'ROOT_natural205000_review':natural205000_ROOT_pin,'restored_optimizer_updates':204800,'cohort_start_optimizer_updates':204800,'pre_periodic_CPU_ready_closed_scalar':pre_periodic_ready_pin,'cohort_binding':startup_binding}"
    replace(old,new,'Truthful output startup cohort; do not invent prior periodic/fullpass or reopen closed ready',1)
    replace("m['commit_available_bytes']>=2000000000+268435456 and m['physical_available_bytes']>=1500000000+268435456",
        "m['commit_available_bytes']>=2000000000+268435456+75497472 and m['physical_available_bytes']>=1500000000+268435456+75497472",
        'Separate72MiB above unchanged runtime floor plus fullprefetch',1)
    replace('4518435456+512*1024**2','4518435456+75497472+512*1024**2','Separate72MiB above original225/fullprefetch plus prospective metadata512')
    replace('1500000000+268435456+512*1024**2','1500000000+268435456+75497472+512*1024**2','Separate72MiB physical reserve above original/fullprefetch plus metadata512')
    replace('Original bounded metadata/headroom/E/full256MiB guards','Original bounded metadata/headroom/E/full256MiB plus separate72MiB guards','Honest stricter source39 runtime guard label',1)
    replace('Full approved225 plus prospective512MiB metadata reserve','Full approved225 plus separate72MiB and prospective512MiB metadata reserve','Honest stricter full prospective label',1)
    old="observation=A/'actual_periodic208000_postpositive_progress.json'"
    new="""reader_root=Path('E:/gpu_training_scratchpad/analysis/BPE9223_Trackio_live_v1')
reader=read(reader_root/'current_status.json')
require(reader['pid']==40048 and abs(reader['create_time']-1792844800.173076)<.001
 and exact_alive({'pid':40048,'create_time':1792844800.173076})
 and exact_alive({'pid':40132,'create_time':1792844800.1489947})
 and reader['error']is None and reader['training_worker_alive']is True
 and reader['native_producer']['pid']==34788 and Path(reader['journal_path']).name=='commits_000027.jsonl',
 'Current same-source restarted reader healthy on actual J27; no action or implicit rebind')
observation=A/'actual_periodic208000_postpositive_progress.json'"""
    replace(old,new,'Actual current healthy reader40048/J27 gate without reader action or history/DB/journal read',1)
    successor=text.encode('utf-8');ast_base=ast.parse(base.decode('utf-8'));ast_new=ast.parse(text)
    inverse=successor
    for edit in reversed(edits):
        a=edit['new'].encode();b=edit['old'].encode()
        require(len(edit['application_byte_offsets'])==edit['count'],'Exact declared application count')
        for offset in reversed(edit['application_byte_offsets']):
            require(inverse[offset:offset+len(a)]==a,'Exact ordered applied byte slice inverse')
            inverse=inverse[:offset]+b+inverse[offset+len(a):]
    require(inverse==base,'Entire accepted196 source whole-byte inverse')
    with NEW.open('xb')as f:f.write(successor);f.flush();os.fsync(f.fileno())
    result['outputs']['source']=pin(NEW)
    base_functions={v.name:ast.dump(v,include_attributes=False)for v in ast_base.body if isinstance(v,(ast.FunctionDef,ast.AsyncFunctionDef))}
    next_functions={v.name:ast.dump(v,include_attributes=False)for v in ast_new.body if isinstance(v,(ast.FunctionDef,ast.AsyncFunctionDef))}
    semantic_notes=[edit['meaning']for edit in edits]
    raw=dict(kind='ASR18_periodic208000_source39_full_ordered_RAW_AST_SOURCE_ONLY_v1',base=base_pin,successor=pin(NEW),
        ordered_edits=edits,whole_original_byte_inverse=True,reconstructed_original_sha256=hashlib.sha256(inverse).hexdigest(),
        base_whole_AST_sha256=hashlib.sha256(ast.dump(ast_base,include_attributes=False).encode()).hexdigest(),
        successor_whole_AST_sha256=hashlib.sha256(ast.dump(ast_new,include_attributes=False).encode()).hexdigest(),
        helper_AST_equality={k:base_functions[k]==next_functions[k]for k in base_functions},
        intentionally_changed_helper_guard_only='Adds separately charged72MiB; original resource values and call cadence remain visible',
        all_source_changes=semantic_notes,no_candidate_import_execution_or_pycompile=True,
        inverse_protocol='Reverse each literal edit and each exact application BYTE offset in reverse order; never globally replace unrelated equal new literals')
    result['outputs']['RAW']=write(RAW,raw)
    bootpin=pin(W/'ROOT_source39_I27_actual204800_boot_restore_savedprefix_CPU14_independent_acceptance_v1.json')
    naturalpin=pin(W/'ROOT_source39_natural205000_full854_PRE_SAVED_independent_acceptance_v1.json')
    inputs=[base_pin,pin(NEW),pin(RAW),bootpin,naturalpin,
        pin(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v39.py'),
        pin(P/'outputs/parakeet_turkish/fastmss_resident_feature_stream_v14.py'),
        pin(P/'outputs/parakeet_turkish/resource_watchdog_v2.py'),
        pin(pathlib.Path('E:/gpu_training_scratchpad/storage_config.json')),pin(pathlib.Path('E:/gpu_training_scratchpad/use_storage.ps1'))]
    handoff=dict(kind='ASR18_periodic208000_source39_CPU14_metadata_observer_INACTIVE_handoff_v1',metadata_source=pin(NEW),full_RAW=pin(RAW),input_pins=inputs,
        preparer=entry['identity'],launcher=entry['launcher'],initial=pin(INITIAL),SOURCE_ONLY=True,actual_observer_execution=False,
        actual_startup_cohort_proposal=dict(restored_optimizer_updates=204800,cohort_start_optimizer_updates=204800,
            ROOT_source39_boot_review=bootpin,ROOT_natural205000_review=naturalpin,
            pre_periodic_CPU=dict(pid=16008,create_time=1792842778.8873637,ppid=41760),
            pre_periodic_CPU_launcher=dict(pid=41760,create_time=1792842778.867194,ppid=34788),
            pre_periodic_CPU_ready=dict(path=str(pathlib.Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01/invocations/owner_000027/CPU/actual_CPU_ready.json')),bytes=10817,sha256='5b4ed5865d52fdbee57a6a9cf1e3c2a45a218a51024fcfd4b4998b5da6d0f7cf'),
            expected_native_updates_to_periodic=3200,expected_private_packets_drained='UNKNOWN_UNTIL_ACTUAL208q',expected_total_preparations='3200+ACTUALdrain; 3220 ONLY if actualdrain20'),
        future_post_periodic_CPU='UNKNOWN_UNTIL_ACTUAL208q/ready_and_separate_ROOT_binding_no_auto_rebind',
        future_previous_quality_ROOT_gate='Accepted204000 ROOTe3381653 exactpin supplied by future execution scope; scalar205/825 only no previous report/prediction re-audit',
        original_helpers_unchanged_except_separate72_guard=True,initial_before_full_prospective_predicate=True,
        guards=dict(prospective_commit=5130803840,prospective_physical=2380803840,runtime_commit=2343932928,runtime_physical=1843932928,
            source39_separate_reserve72=75497472,private512=536870912,seconds=180,E_free=100000000000,original225_FULLprefetch_preserved=True),
        no_new_or_repeated_GC_capture_sync=True,inherited180k_consumed=True,source39_I27_A29=True,
        actual_execution_prerequisites=['Real208000 original350/normalhelper3/actual3200+drain/q+ready/SAMEplan','Full208200/checkpointsemantic/actual>=208220','Exact immutable current source39/native34788 and future newCPU bound by separate ROOT','Healthy current40048/40132 reader on J27','Fresh outputs/full original numeric initial plus72 prospective/resource guards'],
        proposed_prefix=[str(P/'work/nemo-native-env-v2/Scripts/python.exe'),'-X','utf8','-B'],
        proposed_arguments=[str(NEW),'--ROOT-review',str(W/'ROOT_periodic208000_source39_CPU14_metadata_only_exact_execution_scope_v1.json')],
        pending='Independent whole-source/RAW/AST review plus separate exact ONE execution admission; None/UNKNOWN future bindings never executable')
    result['outputs']['handoff']=write(HANDOFF,handoff)
    require(time.monotonic()-began<180 and me.memory_info().private<536870912,'Original preparation wall/private guard')
    result['generator_correction']=dict(failed_source=pin(W/'prepare_periodic208000_source39_metadata_observer_SOURCE_ONLY_v1.py'),
        preserved_failed_result=pin(W/'periodic208000_source39_metadata_observer_source_preparation_result_v1.json'),
        preserved_failed_initial=pin(W/'periodic208000_source39_metadata_observer_source_preparation_initial_v1.json'),
        only_writer_bookkeeping_protocol_changed=True,no_candidate_execution=True)
    result['passed']=True
except BaseException as e:
    result['passed']=False;result['error']=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
finally:
    result['wall_seconds']=time.monotonic()-began
    result_pin=write(RESULT,result)
    print(json.dumps(dict(result=result_pin,passed=result.get('passed'),outputs=result['outputs'],entry=entry,error=result['error']),sort_keys=True))
require(result.get('passed'),'Preserve inactive source preparation failure; no automatic retry')
