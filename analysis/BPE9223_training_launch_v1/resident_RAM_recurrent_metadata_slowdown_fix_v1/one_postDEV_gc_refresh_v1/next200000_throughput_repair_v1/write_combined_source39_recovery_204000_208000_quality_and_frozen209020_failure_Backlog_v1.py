"""INACTIVE: one combined source39 recovery/quality/failed-durability Backlog milestone."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish'
sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
SPEC={'marker': 'BPE9223_SOURCE39_RECOVERY205000_DEV204000_208000_TR216_DURABILITY209020_FAIL_READER6_PENDING_20261024', 'note': 'BPE9223_SOURCE39_RECOVERY205000_DEV204000_208000_TR216_DURABILITY209020_FAIL_READER6_PENDING_20261024\nSource39 recovered from the saved204800 checkpoint with all207 weights, three Adam partitions, all four RNG states and both sampler frontiers restored before the saved J26 prefix was copied to J27; all87 historical unsaved updates were genuinely redone and natural205000 plus subsequent positive updates were independently accepted. The genuine full350 periodic204000 score was TR205/825=24.84848485%, and periodic208000 was TR216/825=26.18181818% (11 more errors, +1.33333333 percentage points). At208000 EN62 ES63 IT24 FR72 DE74 PT64; original caps69/52/30/74/70/66 miss ES by11 and DE by4, strictall6false; no winner, fullpass advancement, early stop or TEST. Natural208000/208200 full854 named tensors,207 weights,215 Adam states in17/6/192 partitions, all4 RNG states, clocks and independent PRE-SAVED equality passed; the same queued plan resumed with20 independently verified finite positive updates208001..208020 and healthy J27 reader. The oldCPU cohort closed normally:3200 native+20 actual wholly unconsumed drain=3220 acquisitions/posts, actualwait0, joined threads, closed pipes and both owned kernel handle sets, MAX2/LIVE1 preserved. The original frozen204820..209020 throughput study completed at72.33909615 audio min/wallmin inclusive, first40082.48011455, final40073.73268883:10.60549656% decline exceeds5%, so the permanent throughput repair remains unfinished. Two fixed postDEV400 rates73.67955321 and73.73268883 exceed70. Dashboard UI interventions are recorded, so this combined span is not an isolated source39 or reader6 cause proof. A tested incremental dashboard query replacement is prearmed for the natural212000 eval pause, NOT LIVE at this note; its frozen212020..217020 durability validation remains pending. Original learning rates, science, all resource guards, full prefetch and full401072 target remain unchanged.\n', 'closed_ROOT_inputs': {'boot': {'path': 'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\one_postDEV_gc_refresh_v1\\next200000_throughput_repair_v1\\ROOT_source39_I27_actual204800_boot_restore_savedprefix_CPU14_independent_acceptance_v1.json', 'bytes': 9500, 'sha256': '48b11594d36cddae4f3144b52bee03204a84c61e88d75f6398e6e676edd08b56'}, 'direct_redo': {'path': 'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\one_postDEV_gc_refresh_v1\\next200000_throughput_repair_v1\\ROOT_source39_direct_NEW204801_204888_genuine87redo_independent_acceptance_v1.json', 'bytes': 2135, 'sha256': '5e999ec84fae5220f930ae987e8f67d743acb3d20f76d0a6ea76b4b4c48824bb'}, 'standard205000': {'path': 'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\one_postDEV_gc_refresh_v1\\next200000_throughput_repair_v1\\ROOT_source39_natural205000_full854_PRE_SAVED_independent_acceptance_v1.json', 'bytes': 2887, 'sha256': 'ad42b4335e8ae3364284c0c403246a0485fb9439bcdae1f843c626c667be2b9c'}, 'post205020': {'path': 'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\one_postDEV_gc_refresh_v1\\next200000_throughput_repair_v1\\ROOT_source39_ONLY_NEW205001_205020_resumed_after_natural205000_independent_acceptance_v1.json', 'bytes': 1632, 'sha256': 'd8a78caab0aa435e0d1e7b0a59f8c599229f758db32e3db4e9962d7ec7750776'}, 'quality204000': {'path': 'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\one_postDEV_gc_refresh_v1\\next200000_throughput_repair_v1\\ROOT_NEW204000_original350_independent_word_character_DP_quality_acceptance_v1.json', 'bytes': 7170, 'sha256': 'e338165352dec7daabcddb94adf63332d7ecce8777041145b00b24c44679d782'}, 'full208000': {'path': 'E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1/ROOT_periodic208000_full350_source39_CPU14_max2_normal_saved208200_and_resumed_training_verification_v1.json', 'bytes': 9679, 'sha256': 'cac57ba30d142f10fa026ec46bfea5fc61e7291350d8817983f9e14ae54fe72b'}, 'direct20820': {'path': 'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\one_postDEV_gc_refresh_v1\\next200000_throughput_repair_v1\\ROOT_periodic208000_direct20_sameplan_broad2_resumed_positive_verification_v1.json', 'bytes': 8215, 'sha256': '7de95eda960ba773f7868742ecad18e104c208386be159e2d9a415b400a4592c'}, 'source39_durability': {'path': 'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\one_postDEV_gc_refresh_v1\\next200000_throughput_repair_v1\\ROOT_source39_original_frozen204820_209020_throughput_durability_acceptance_v1.json', 'bytes': 29172, 'sha256': '2f08a6ecbe668dbe6607d70c65b21d59dd10619e303320ee2bdca2c402c71524'}}, 'task': 'C:\\Users\\ardao\\Documents\\Codex\\2026-10-10\\i-x20\\outputs\\parakeet_turkish\\backlog\\tasks\\asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md', 'CLI_edit_command': ['C:\\Users\\ardao\\Documents\\Codex\\2026-10-10\\i-x20\\outputs\\parakeet_turkish\\tools\\backlog.exe', 'task', 'edit', 'ASR-18', '--append-notes', 'BPE9223_SOURCE39_RECOVERY205000_DEV204000_208000_TR216_DURABILITY209020_FAIL_READER6_PENDING_20261024\nSource39 recovered from the saved204800 checkpoint with all207 weights, three Adam partitions, all four RNG states and both sampler frontiers restored before the saved J26 prefix was copied to J27; all87 historical unsaved updates were genuinely redone and natural205000 plus subsequent positive updates were independently accepted. The genuine full350 periodic204000 score was TR205/825=24.84848485%, and periodic208000 was TR216/825=26.18181818% (11 more errors, +1.33333333 percentage points). At208000 EN62 ES63 IT24 FR72 DE74 PT64; original caps69/52/30/74/70/66 miss ES by11 and DE by4, strictall6false; no winner, fullpass advancement, early stop or TEST. Natural208000/208200 full854 named tensors,207 weights,215 Adam states in17/6/192 partitions, all4 RNG states, clocks and independent PRE-SAVED equality passed; the same queued plan resumed with20 independently verified finite positive updates208001..208020 and healthy J27 reader. The oldCPU cohort closed normally:3200 native+20 actual wholly unconsumed drain=3220 acquisitions/posts, actualwait0, joined threads, closed pipes and both owned kernel handle sets, MAX2/LIVE1 preserved. The original frozen204820..209020 throughput study completed at72.33909615 audio min/wallmin inclusive, first40082.48011455, final40073.73268883:10.60549656% decline exceeds5%, so the permanent throughput repair remains unfinished. Two fixed postDEV400 rates73.67955321 and73.73268883 exceed70. Dashboard UI interventions are recorded, so this combined span is not an isolated source39 or reader6 cause proof. A tested incremental dashboard query replacement is prearmed for the natural212000 eval pause, NOT LIVE at this note; its frozen212020..217020 durability validation remains pending. Original learning rates, science, all resource guards, full prefetch and full401072 target remain unchanged.\n'], 'refresh_command': ['C:\\Users\\ardao\\Documents\\Codex\\2026-10-10\\i-x20\\work\\training-env\\Scripts\\python.exe', '-Xutf8', '-B', 'C:\\Users\\ardao\\Documents\\Codex\\2026-10-10\\i-x20\\outputs\\parakeet_turkish\\refresh_backlog.py'], 'recipe_source': {'path': 'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\one_postDEV_gc_refresh_v1\\record_periodic196000_source37_backlog_v1.py', 'bytes': 8050, 'sha256': '4eef1b7124a10b9a412b9038680ed5d8329dd84f167de3a78a42e5fb19296757'}, 'preparation_scope': {'path': 'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\one_postDEV_gc_refresh_v1\\next200000_throughput_repair_v1\\ROOT_combined_source39_recovery_204000_208000_quality_and_frozen209020_failure_Backlog_INACTIVE_preparation_scope_v1.json', 'bytes': 7830, 'sha256': '50866a7ea19a1afb9da8365b9e62a1fbdcf1c79c36f93b3e847bec5605bc2051'}, 'future_scope_kind': 'ROOT_combined_source39_recovery_204000_208000_quality_and_frozen209020_failure_Backlog_exact_ONE_execution_scope_v1', 'actual_output_directory': 'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\one_postDEV_gc_refresh_v1\\next200000_throughput_repair_v1\\combined_source39_recovery_204000_208000_Backlog_actual_01', 'standby_actor': {'pid': 26648, 'create_time': 1792849617.2166405, 'ppid': 6320}, 'standby_launcher': {'pid': 6320, 'create_time': 1792849617.2091935, 'ppid': 21892}}
W=Path(__file__).resolve().parent
TRAINING_PY=P/'work/training-env/Scripts/python.exe'
TASK=Path(SPEC['task']);MARKER=SPEC['marker'];NOTE=SPEC['note'];OUT=Path(SPEC['actual_output_directory'])
BEGAN=time.monotonic()
def require(value,message):
 if not value:raise ValueError(message)
def pin(path):
 path=Path(path);raw=path.read_bytes();return {'path':str(path.resolve()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(path,value):
 with Path(path).open('x',encoding='utf-8',newline='\n')as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def ident(pid):
 p=psutil.Process(pid);return {'pid':pid,'create_time':p.create_time(),'ppid':p.ppid()}
def alive(who):
 try:return abs(psutil.Process(who['pid']).create_time()-who['create_time'])<.001 and psutil.Process(who['pid']).ppid()==who['ppid']
 except psutil.NoSuchProcess:return False
def same_pin(expected):
 actual=pin(expected['path']);require(actual['bytes']==expected['bytes']and actual['sha256']==expected['sha256']and Path(actual['path']).resolve()==Path(expected['path']).resolve(),'Exact immutable pin '+expected['path']);return actual
def guard(phase):
 sample=sample_memory(os.getpid(),phase);m=sample['system_memory']
 require(sample['paired_sampling_within_one_second']and sample['worker_private_memory']['private_bytes']<536870912,'Sampled self512MiB/paired guard')
 require(m['commit_available_bytes']>=2880803840 and m['physical_available_bytes']>=2380803840,'Original runtime+72 and separateactiveStandby512 reserve')
 require(shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-BEGAN<180,'E100GB/wall180')
 return sample

def main():
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 parser=argparse.ArgumentParser();parser.add_argument('--ROOT-review',required=True);args=parser.parse_args()
 require(Path(sys.executable).resolve()==TRAINING_PY.resolve(),'Established training-env job interpreter only')
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and sys.flags.utf8_mode==1,'CPUhidden/UTF8')
 require(all(os.environ.get(k)=='1'for k in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'caps1')
 require(Path(os.environ['TORCH_EXTENSIONS_DIR']).resolve()==Path('E:/gpu_training_scratchpad/cache/torch_extensions').resolve(),'Exact dot-sourced TORCH_EXTENSIONS_DIR')
 require(not OUT.exists(),'Fresh one-operation namespace; ambiguous retry forbidden');OUT.mkdir()
 ids=[ident(os.getpid()),ident(os.getppid()),ident(psutil.Process(os.getppid()).ppid())]
 sample=sample_memory(os.getpid(),'combined_source39_Backlog_initial');efree=shutil.disk_usage('E:/').free
 initial=OUT/'initial_numeric_BEFORE_full_prospective.json'
 write(initial,{'sample':sample,'required_commit':5667674752,'required_physical':2917674752,'extra_standby_reserve_bytes':536870912,'E_free_bytes':efree,'persisted_before_predicate_and_marker_read_and_mutation':True,'metadata_identities':ids})
 result={'kind':'ASR18_combined_source39_recovery_204000_208000_quality_and_failed_durability_Backlog_result_v1','marker':MARKER,'CLI_edit_invocations':0,'refresh_invocations':0,'actual_CLI_edit_exit':None,'actual_refresh_exit':None,'passed':False,'error':None,'subprocesses':[],'job_interpreter':str(TRAINING_PY),'refresh_interpreter':str(TRAINING_PY),'initial_numeric_sample':pin(initial),'metadata_identities_at_write':ids,'marker_count_before':None,'marker_count_after':None}
 inputs={};task_read_started=False
 def run(label,command):
  guard('combined_source39_Backlog_before_'+label)
  with(OUT/(label+'.stdout.log')).open('xb')as stdout,(OUT/(label+'.stderr.log')).open('xb')as stderr:
   process=subprocess.Popen(command,cwd=S,stdout=stdout,stderr=stderr)
   try:who=ident(process.pid)
   except psutil.NoSuchProcess:who={'pid':process.pid,'create_time':'UNRECORDED_UNKNOWN'}
   row={'label':label,'argv':command,'Popen_identity':who,'actual_Popen_wait_code':None};result['subprocesses'].append(row)
   write(OUT/(label+'_Popen_launch.json'),row)
   code=process.wait(timeout=60);row['actual_Popen_wait_code']=code
   row['recorded_Popen_identity_exact_dead']=None if who['create_time']=='UNRECORDED_UNKNOWN'else not alive(who)
  guard('combined_source39_Backlog_after_'+label)
  return code
 try:
  m=sample['system_memory'];require(sample['paired_sampling_within_one_second']and sample['worker_private_memory']['private_bytes']<536870912,'Actual paired/self512 before initial predicate')
  require(m['commit_available_bytes']>=5667674752 and m['physical_available_bytes']>=2917674752 and efree>=100000000000,'Full original225/FULL+72+prospective512 PLUS separateactiveStandby512; no freed credit')
  review=json.loads(Path(args.ROOT_review).read_bytes());require(review['kind']==SPEC['future_scope_kind']and review['accepted']is True and review['reviewer']=='/root','Separate exact ONE ROOT execution gate')
  require(review['source']==pin(Path(__file__)),'Exact reviewed immutable source')
  require(review['marker']==MARKER and review['note_text']==NOTE,'Exact approved complete milestone text')
  require(alive(SPEC['standby_actor'])and alive(SPEC['standby_launcher']),'Exact prearmed212 standby still live; charge its full512MiB')
  require(not(W/'actual_Trackio6_reader_repair_at212000_v1/result.json').exists()and not(P/'work/research/serve_BPE9223_Trackio_live_v6.py').exists(),'Reader6 pending at this note; no stale live/pending claim')
  for name,expected in SPEC['closed_ROOT_inputs'].items():inputs[name]=same_pin(expected)
  task_read_started=True;before=TASK.read_text(encoding='utf-8').count(MARKER);result['marker_count_before']=before
  require(before==0,'Canonical marker already exists; stop without duplicate edit/refresh')
  note_path=OUT/'milestone_note.txt'
  with note_path.open('x',encoding='utf-8',newline='\n')as f:f.write(NOTE);f.flush();os.fsync(f.fileno())
  write(OUT/'entry.json',{'kind':result['kind'],'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'marker':MARKER,'marker_count_before':before,'inputs':inputs,'task_before':pin(TASK),'one_edit_only':True,'metadata_identities':ids,'recipe_source':SPEC['recipe_source'],'ROOT_execution_review':pin(args.ROOT_review)})
  result['CLI_edit_invocations']=1;result['actual_CLI_edit_exit']=run('edit',SPEC['CLI_edit_command']);require(result['actual_CLI_edit_exit']==0,'Actual CLI edit failed; no retry/refresh')
  require(TASK.read_text(encoding='utf-8').count(MARKER)==1,'Marker exactly1 after ONE edit; no duplicate/refresh on ambiguity')
  result['refresh_invocations']=1;result['actual_refresh_exit']=run('refresh',SPEC['refresh_command']);require(result['actual_refresh_exit']==0,'Actual normal refresh failed; no retry')
  result['marker_count_after']=TASK.read_text(encoding='utf-8').count(MARKER);require(result['marker_count_after']==1,'Final canonical marker exactly1')
  result['passed']=True
 except BaseException as error:result['error']={'type':type(error).__name__,'message':str(error),'ambiguous_retry_forbidden':True}
 finally:
  if task_read_started:result['marker_count_after']=TASK.read_text(encoding='utf-8').count(MARKER);result['task_after']=pin(TASK)
  result['inputs']=inputs;result['source']=pin(Path(__file__));result['recipe_source']=SPEC['recipe_source']
  result['logs']={p.name:pin(p)for p in sorted(OUT.glob('*.log'))}
  result['entry']=pin(OUT/'entry.json')if(OUT/'entry.json').exists()else None
  result['note']=pin(OUT/'milestone_note.txt')if(OUT/'milestone_note.txt').exists()else None
  result['ROOT_input_pins_after']={name:same_pin(expected)for name,expected in SPEC['closed_ROOT_inputs'].items()}if inputs else {}
  result['wall_seconds']=time.monotonic()-BEGAN;result['no_training_reader_LR_GC_environment_guard_science_or_policy_change']=True
  write(OUT/'result.json',result)
 print(json.dumps({'result':pin(OUT/'result.json'),'passed':result['passed'],'edit':result['actual_CLI_edit_exit'],'refresh':result['actual_refresh_exit'],'marker_before':result['marker_count_before'],'marker_after':result['marker_count_after'],'metadata_identities_at_write':ids,'error':result['error']}),flush=True)
 return 0 if result['passed']else 1
if __name__=='__main__':raise SystemExit(main())
