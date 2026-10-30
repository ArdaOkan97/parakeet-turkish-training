"""INACTIVE SOURCE: original Source39 clean212000 DEV; Source41 I29 recovery and natural212200.
No payload/journal/plan/audio read, native operation, reader control, or bookkeeping.
Actual execution requires a separate exact ROOT scope for this immutable source.
"""
import argparse,ast,hashlib,json,os,shutil,sys,time,traceback,unicodedata
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
S=P/'outputs/parakeet_turkish'
sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01')
I=B/'invocations/owner_000029'
T=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_Trackio_live_v1')
HERE=Path(__file__).resolve().parent
BEGAN=time.monotonic()
NATIVE={'pid':27664,'create_time':1792877280.3418417,'ppid':23820}
CPU={'pid':16604,'create_time':1792877310.9130895,'ppid':36000}
CPU_LAUNCHER={'pid':36000,'create_time':1792877310.8928175,'ppid':27664}
READER={'pid':25012,'create_time':1792872869.5210469,'ppid':34368}
READER_LAUNCHER={'pid':34368,'create_time':1792872869.5039415,'ppid':21280}
DRIVER={'path':str(HERE/'run_ASR18_BPE9223_resident_RAM_training_v41_dual_config_v3.py'),'bytes':142995,'sha256':'bd1017bb044c21961a850861f7dc362c5fbcd83463c523428cd1ef0a185f4774'}
READER_SOURCE={'path':str(P/'work/research/serve_BPE9223_Trackio_live_v7.py'),'bytes':26204,'sha256':'d7bb6494d197cfe4b4f7bc8c5af1f4b720be199e2fb8d3077cb8af45e450da22'}
BOOT_ROOT={'path':str(HERE/'ROOT_source41_actual_boot_restore_ready_independent_acceptance_v2/acceptance.json'),'bytes':13564,'sha256':'97d7f58d40dd882b22f20daedbb82889b47b019baed2823bbf17d5d9362921f8'}
SCORER_INPUT={'path':str(HERE/'ROOT_pending212quality_immutable_scorer_reference_SOURCE_scope_v1/immutable_policy_SOURCE_input.json'),'bytes':3966,'sha256':'760c53a551a03d586e045d3eac50bc9c5195b4475b4030ee44e707462f466522'}
NORMALIZER_INPUT={'path':str(HERE/'ROOT_NEW212quality_original_normalizer_SOURCE_INPUT_v1/SOURCE_input.json'),'bytes':8924,'sha256':'d7d600c3762d504ade47f124e224092d48b589476d8f533827b219d63d58c1e7'}
DEV_SOURCE_INPUT={'path':str(HERE/'ROOT_pending212quality_DEV_helper_SOURCE_binding_v1/SOURCE_capture.json'),'bytes':8955,'sha256':'ae143beb23fff0381811ed0a8e684141676fb445d7b55f00fd333e2116520bf7'}
SAVED212000_POINTER={'path':str(B/'checkpoints/update_000212000/pointer.json'),'bytes':16120,'sha256':'9bef17e4fd0ed37f34f056595ca09ec91d2a69d80786165d6ca866b270bcbe07'}
LANGS=('tr','en','es','it','fr','de','pt')
REFERENCE_WORDS={'tr':825,'en':1045,'es':1289,'it':1158,'fr':1072,'de':1004,'pt':1095}
LIMITS={'en':69,'es':52,'it':30,'fr':74,'de':70,'pt':66}
READS=[]
CAPTURED=[]
def require(value,message):
 if not value:raise ValueError(message)
def identity(p):
 return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def exact_alive(who):
 try:
  p=psutil.Process(who['pid'])
  return abs(p.create_time()-who['create_time'])<.001 and p.ppid()==who['ppid']
 except psutil.NoSuchProcess:return False
def write_fresh(path,value):
 require(not path.exists(),'Fresh E-only output '+str(path))
 with path.open('x',encoding='utf-8',newline='\n')as f:
  json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def pin_bytes(path,data):
 return {'path':str(Path(path).resolve()),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def output_pin(path):
 return pin_bytes(path,path.read_bytes())
def sample():
 return sample_memory(os.getpid(),'original212000_Source41_readonly_metadata')
def check_numeric(v,commit,physical):
 require(v['paired_sampling_within_one_second'],'Original paired numeric sampling')
 require(v['system_memory']['commit_available_bytes']>=commit,'Full original prospective/runtime commit')
 require(v['system_memory']['physical_available_bytes']>=physical,'Full original prospective/runtime physical')
 require(v['worker_private_memory']['private_bytes']<=536870912,'Original self512MiB')
 require(shutil.disk_usage('E:/').free>=100000000000,'Original E100GB')
 require(time.monotonic()-BEGAN<=180,'Original metadata180s')
def guard():
 v=sample();check_numeric(v,3509949440,3009949440);return v
def read_once(path,expected=None,max_bytes=16777216):
 guard();p=Path(path).resolve();key=str(p).casefold()
 require(key not in READS,'Single-read metadata file '+str(p));READS.append(key)
 size=p.stat().st_size;require(0<=size<=max_bytes,'Bounded metadata input')
 data=p.read_bytes();require(len(data)<=max_bytes,'Bounded actual metadata input')
 actual=pin_bytes(p,data)
 CAPTURED.append({'pin':actual,'UTF8_text':data.decode('utf-8-sig')})
 if expected is not None:
  normalized={**expected,'path':str(Path(expected['path']).resolve())}
  require(actual==normalized,'Exact input bytes/SHA/resolved path '+str(p))
 return data,actual
def json_once(path,expected=None,max_bytes=16777216):
 data,actual=read_once(path,expected,max_bytes)
 return json.loads(data),actual
def pinned_json(expected):
 return json_once(expected['path'],expected)
def current_health():
 progress,_=json_once(B/'progress/latest.json')
 reader,_=json_once(T/'current_status.json')
 state,_=json_once(T/'mirror_state.json')
 require(progress['producer']==NATIVE and progress['source']==DRIVER,'Same Source41 native/source')
 require(progress['actual_optimizer_updates']>=212220 and progress['native_updates_this_process']>=220
  and progress['last_saved_updates']>=212200,'Natural212200 and resumed native updates')
 require(progress['planned_native_updates']==401072 and progress['administrative_update_stop']is None
  and progress['optimizer_poisoned']is False and progress['schedule_poisoned']is False
  and not(I/'actual_failure.json').exists(),'Healthy full401072/no finite stop')
 for who in (NATIVE,CPU,CPU_LAUNCHER,READER,READER_LAUNCHER):require(exact_alive(who),'Exact fixed current identity '+str(who['pid']))
 require(reader['pid']==READER['pid']and abs(reader['create_time']-READER['create_time'])<.001
  and Path(reader['source']).resolve()==Path(READER_SOURCE['path']).resolve()
  and reader['source_sha256']==READER_SOURCE['sha256'],'Same genuine reader7')
 require(reader['native_producer']==NATIVE and reader['training_worker_alive']is True
  and reader['mirror_thread_alive']is True and reader['HTTP_ready']is True and reader['reader_ready']is True
  and reader['error']is None and reader['API_dispatch']['failed']==0,'Actual reader current HTTP/thread/native health')
 jpath=B/'journals/commits_000029.jsonl'
 require(Path(reader['journal_path']).resolve()==jpath.resolve()and Path(state['journal_path']).resolve()==jpath.resolve(),'Reader actually on J29')
 require(reader['mirrored_update']>=212220 and reader['last_verified_checkpoint_update']>=212200
  and state['last_update']>=212220 and state['offset']>426460425,'Current advancing catch-up beyond accepted ROOT97 baseline')
 return progress,reader,state

def normalize_text(text: str, language: str = "tr") -> str:
    text = unicodedata.normalize("NFC", str(text or ""))
    if language.lower().split("_")[0].split("-")[0] == "tr":
        text = text.replace("I", "ı").replace("İ", "i")
    text = text.lower().replace("i\u0307", "i")
    # Punctuation is deleted (apostrophes stay inside the surrounding word).
    # Unicode separator/control characters become whitespace.
    text = "".join(
        " " if unicodedata.category(char)[0] in {"Z", "C"}
        else "" if unicodedata.category(char)[0] == "P"
        else char
        for char in text
    )
    return " ".join(text.split())


def distance(reference,hypothesis):
 previous=list(range(len(hypothesis)+1))
 for i,ref in enumerate(reference,1):
  row=[i]
  for j,hyp in enumerate(hypothesis,1):
   row.append(min(row[-1]+1,previous[j]+1,previous[j-1]+(ref!=hyp)))
  previous=row
 return previous[-1]
def score_rows(rows,language):
 details=[];we=ce=rw=rc=0
 for row in rows:
  guard()
  reference=normalize_text(row.get('reference',''),language);hypothesis=normalize_text(row.get('hypothesis',''),language)
  ew=distance(reference.split(),hypothesis.split());ec=distance(reference,hypothesis)
  nw=len(reference.split());nc=len(reference);we+=ew;ce+=ec;rw+=nw;rc+=nc
  details.append({'source_id':row['source_id'],'reference_normalized':reference,'hypothesis_normalized':hypothesis,
   'word_errors':ew,'reference_words':nw,'character_errors':ec,'reference_characters':nc})
 return {'utterances':len(rows),'word_errors':we,'reference_words':rw,'character_errors':ce,
  'reference_characters':rc,'wer':we/rw if rw else 0.0,'cer':ce/rc if rc else 0.0,'details':details}
def verify_quality(review,policy,DEV_binding):
 reportpath=B/'periodic_DEV/update_000212000/report.json'
 preservepath=B/'periodic_DEV/update_000212000/training_frontier_preservation.json'
 require(Path(review['quality_report']['path']).resolve()==reportpath.resolve()
  and Path(review['quality_preservation']['path']).resolve()==preservepath.resolve(),'Exact original212 source-derived paths')
 report,report_pin=json_once(reportpath,review['quality_report'])
 preservation,preservation_pin=json_once(preservepath,review['quality_preservation'])
 require(report['source']==DEV_binding['actual']and report['scorer_policy']==policy['pin'],'Original evaluator and immutable scoring policy')
 require(report['passed']is True and report['optimizer_updates']==212000 and report['checkpoint_pointer']==SAVED212000_POINTER
  and report['actual_native_updates_in_DEV']==0 and report['all207_threeAdam_all4RNG_unchanged']is True
  and report['TEST_used']is False and report['retention_early_stop']is False,'Genuine original Source39 full350 unchanged DEV')
 require(preservation['DEV']==report_pin and preservation['optimizer_updates']==212000
  and preservation['actual_native_updates_in_DEV']==0 and preservation['actual_sampler_or_replay_advance']is False,'Original212 frontier/DEV linkage')
 for key in ('all207_threeAdam_all4RNG_unchanged','both_cursors_journal_and_nextdraw_unchanged','fullpass_and_winner_state_unchanged_here'):
  require(preservation[key]is True,'Original frontier preservation '+key)
 # Original evaluator also emits calibration scores. Only seven validation sets are clean350.
 require(set(report['predictions'])==set(LANGS),'Exactly seven prediction languages')
 require({k for k in report['scores']if k.startswith('validation_')}=={'validation_'+k for k in LANGS},'Exactly seven original validation score keys')
 require(all(k in {'validation_'+l for l in LANGS}|{'calibration_'+l for l in LANGS}for k in report['scores']),'No unrelated/TEST score sets')
 scores={};captured={}
 for language in LANGS:
  predpin=report['predictions'][language]
  require(predpin==review['prediction_files'][language],'Exact newly scoped original212 prediction')
  preddata,predactual=read_once(predpin['path'],predpin)
  referencepin=policy['value']['DEV_manifests'][language]
  refdata,refactual=read_once(referencepin['path'],referencepin)
  rows=[json.loads(line)for line in preddata.decode('utf-8-sig').splitlines()if line.strip()]
  refs=[json.loads(line)for line in refdata.decode('utf-8-sig').splitlines()if line.strip()]
  require(len(rows)==len(refs)==50 and len({r['source_id']for r in refs})==50
   and [r['source_id']for r in rows]==[r['source_id']for r in refs],'Exact ordered original50 membership '+language)
  for ref,row in zip(refs,rows):
   require(row['language']==ref['language']==language and row['split']=='validation'and row['reference']==ref['text'],'Original clean reference '+language)
   for key in ('sentence_id','dataset_revision','audio_sha256','wav_sha256'):
    require(ref.get(key)==row.get(key),'Original provenance '+key)
  result=score_rows(rows,language);official=report['scores']['validation_'+language]['asr']
  for key in ('utterances','word_errors','reference_words','character_errors','reference_characters','wer','cer'):
   require(result[key]==official[key],'Original saved-text word/charDP equality '+language+'/'+key)
  require(result['reference_words']==REFERENCE_WORDS[language],'Frozen reference word total '+language)
  scores[language]=result
  captured[language]={'prediction_pin':predactual,'reference_pin':refactual,'predictions':rows,'references':refs}
  del preddata,refdata,rows,refs
 summary=report['official_saved_text_score']
 require(summary['full350_DEV_complete']is True and summary['TEST_used']is False
  and summary['reference_words']==REFERENCE_WORDS and summary['retention_limits']==LIMITS,'Original full350 official summary')
 require(summary['word_errors']=={k:scores[k]['word_errors']for k in LANGS}
  and summary['strict_all_six_passed']==all(scores[k]['word_errors']<=v for k,v in LIMITS.items())
  and summary['turkish_wer']==scores['tr']['word_errors']/825,'Exact integer gates and TR corpus fraction')
 return {'report_pin':report_pin,'preservation_pin':preservation_pin,'report':report,'preservation':preservation,
  'scores_word_char_DP':scores,'original350_and_reference_inputs':captured,
  'original_DEV_owner':'Source39/I27/native34788; historical normal4020 and separate72MiB, not current Source41 startup/+80',
  'retention_cap_misses':{k:scores[k]['word_errors']-v for k,v in LIMITS.items()if scores[k]['word_errors']>v},
  'no_checkpoint212000_or_q_ready_plan_reread':True}


def verify_natural212200(review):
 """Capture newly scoped metadata and check original persistence RECEIPTS only.
 Independent full854/reference geometry/PRE-SAVED canonical equality remains ROOT pending.
 No tensor-reference layout is guessed and no payload is opened.
 """
 CK=(B/'checkpoints/update_000212200').resolve()
 layout=review['natural212200_SOURCE_layout']
 require(layout['ROOT_SOURCE_derived_exact_bindings']is True
  and layout['independent_full854_PRE_SAVED_verification_required_after_capture']is True,
  'Future explicit ROOT SOURCE-derived layout gate; counts cannot stand for full854 validation')
 records=layout['metadata']
 require(set(records)=={'pointer','checkpoint','pre_save','next_draw','persistence'},'Every NEW natural212200 metadata role explicitly bound')
 fixed={'pointer':'pointer.json','checkpoint':'checkpoint.json','pre_save':'pre_save.json','next_draw':'next_draw.json','persistence':'persistence_verified.json'}
 for role,name in fixed.items():
  require(Path(records[role]['pin']['path']).resolve()==CK/name,'Exact SOURCE-derived NEW212200 '+role+' path')
 seen=set();captured={}
 for role in ('pointer','checkpoint','pre_save','next_draw','persistence'):
  item=records[role];p=Path(item['pin']['path']).resolve()
  require(isinstance(item['SOURCE_derivation'],str)and bool(item['SOURCE_derivation'].strip()),'Exact SOURCE expression/provenance '+role)
  require(p.suffix.lower()=='.json'and p.is_relative_to(B.resolve())
   and str(p).casefold()not in seen,'Explicit distinct bounded JSON metadata roles, never payload')
  seen.add(str(p).casefold())
  value,actual=json_once(p,item['pin'])
  captured[role]={'pin':actual,'value':value,'SOURCE_derivation':item['SOURCE_derivation']}
 persistence=captured['persistence']['value']
 require(persistence['passed']is True and persistence['optimizer_updates']==212200
  and persistence['named_tensors_verified']==854 and persistence['weights_verified']==207
  and persistence['Adam_states_verified']==215
  and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256']
  and persistence['saved_semantic_state_verified']['passed']is True,
  'Original persistence-reported counts/semantics/equality ONLY; independent ROOT validation still required')
 payload_stats=[]
 for item in layout['payload_STAT_ONLY']if layout['payload_STAT_ONLY']is not None else []:
  require(set(item)=={'path','SOURCE_derivation'}and isinstance(item['SOURCE_derivation'],str)and item['SOURCE_derivation'],
   'Explicit ROOT SOURCE-derived payload STAT-only binding')
  p=Path(item['path']).resolve()
  require(p.is_relative_to(B.resolve())and str(p).casefold()not in READS,'No payload read/hash and no metadata substitution')
  try:
   st=p.stat();status={'exists':True,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
  except FileNotFoundError:status={'exists':False,'normal_retention_removal_possible':True,'retention_not_changed':True}
  payload_stats.append({'path':str(p),'SOURCE_derivation':item['SOURCE_derivation'],'STAT':status,'body_read':False,'body_hash':False})
 # Unresolved payload namespace remains explicitly pending ROOT; no invented path or STAT claim.
 return {'optimizer_updates':212200,'metadata':captured,'original_persistence_receipt_counts_passed':True,
  'reported_named_refs':854,'reported_weights':207,'reported_Adam_states':215,
  'independent_full854_reference_geometry_PRE_SAVED_canonical_check':'PENDING_ROOT',
  'owner_independently_recomputed_full854':False,
  'ROOT_must_check':{'named_refs':854,'weights':207,'Adam_states':215,'Adam_partitions':[17,6,192],
   'RNG_sources':4,'clocks':[276450,214200,274450],'alignment_bytes':64,
   'original_scalar_and_integer_Adam_key_semantics':True,'independent_PRE_SAVED_canonical_equality':True},
  'payload_STAT_only':payload_stats if layout['payload_STAT_ONLY']is not None else 'PENDING_ROOT_SOURCE_operand_and_STAT',
  'payload_read_or_hash':False,'GPU_restore':False}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--ROOT-review',required=True);args=parser.parse_args()
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 actual=HERE/'original212000_source41_I29_metadata_actual_v1'
 require(not actual.exists(),'Fresh exact metadata output namespace')
 actual.mkdir()
 result_path=actual/'result.json';initial_path=actual/'initial_BEFORE_full_predicate.json'
 capture_path=actual/'NEW_metadata_CAPTURE.json';numeric_path=actual/'direct_numeric_exit.json'
 code=1
 try:
  initial=sample()
  write_fresh(initial_path,{'sample':initial,'required_commit':6028384896,'required_physical':3278384896,
   'sample_persisted_BEFORE_full_prospective_predicate':True,'self':identity(psutil.Process()),
   'launcher':identity(psutil.Process().parent()),'TORCH_EXTENSIONS_DIR':os.getenv('TORCH_EXTENSIONS_DIR'),
   'CUDA_VISIBLE_DEVICES':os.getenv('CUDA_VISIBLE_DEVICES')})
  check_numeric(initial,6028384896,3278384896)
  require(os.getenv('CUDA_VISIBLE_DEVICES')=='-1'
   and os.getenv('TORCH_EXTENSIONS_DIR')==r'E:\gpu_training_scratchpad\cache\torch_extensions','E policy/CPUhidden environment')
  review,reviewpin=json_once(args.ROOT_review)
  require(review['kind']=='ROOT_original212000_source41_I29_CPU14_metadata_only_exact_execution_scope_v1'
   and review['accepted']is True and review['reviewer']=='/root','Separate exact ONE actual ROOT execution admission')
  _,selfpin=read_once(__file__,review['metadata_source'])
  require(review['native']==NATIVE and review['current_CPU']==CPU and review['current_CPU_launcher']==CPU_LAUNCHER
   and review['reader_identity']==READER and review['reader_launcher_identity']==READER_LAUNCHER,'Exact frozen actual cohort/reader; no rebind')
  require(review['resources']['prospective_commit']==6028384896 and review['resources']['prospective_physical']==3278384896
   and review['resources']['runtime_commit']==3509949440 and review['resources']['runtime_physical']==3009949440
   and review['resources']['self_private']==536870912 and review['resources']['wall_seconds']==180
   and review['resources']['E_free_min']==100000000000,'Every original full source41/reader/ROOT+job reserve')
  require(Path(review['output_directory']).resolve()==actual.resolve(),'Exact fresh output directory')
  for expected in review['input_pins']:
   # SOURCE/ROOT/reference inputs only; original quality/natural state reads happen below exactly once.
   require(isinstance(expected,dict)and set(expected)=={'path','bytes','sha256'},'Exact three-field input pins')
  boot,bootpin=pinned_json(BOOT_ROOT)
  require(boot['accepted']is True and boot['reviewer']=='/root'
   and boot['all18_original_hashes_independently_verified']is True
   and boot['actual_full207_threeAdam_fourRNG_nextdraw_restore_before_native']is True
   and boot['original_GC_cache_science_guards_and_consumed180k_preserved']is True,'Closed ROOT97 only; no bootstrap original/capture re-audit')
  branch=boot['saved_J27_prefix_to_J29']
  require(branch['saved_prefix']['offset']==branch['new_active_journal']['offset']==426026870
   and branch['saved_prefix']['sha256']==branch['new_active_journal']['sha256']=='6621855eb4191790c3a741e3e1c19014f00db87b9e471d3a4c70802cbddaaf0a'
   and Path(branch['new_active_journal']['path']).resolve()==(B/'journals/commits_000029.jsonl').resolve()
   and branch['historical_unsaved_tail_updates']==0 and branch['native_updates_or_draws_in_copy']==0,'Accepted savedprefix-only/zero-draw transition')
  policy,policypin=pinned_json(SCORER_INPUT);normalizer,normalpin=pinned_json(NORMALIZER_INPUT);dev,devpin=pinned_json(DEV_SOURCE_INPUT)
  require(policy['value']['reference_words']==REFERENCE_WORDS and policy['value']['retention_limits']==LIMITS
   and normalizer['actual']==policy['value']['official_scorer'],'Exact immutable new-quality scorer/reference SOURCE authority')
  original_ast=ast.parse(normalizer['SOURCE'])
  original_normalizer=next(node for node in original_ast.body if isinstance(node,ast.FunctionDef)and node.name=='normalize_text')
  # Comparing the bound source normalizer with this source is metadata AST only, never importing/executing it.
  own_source=next(item['UTF8_text']for item in CAPTURED if item['pin']==selfpin)
  own_normalizer=next(node for node in ast.parse(own_source).body if isinstance(node,ast.FunctionDef)and node.name=='normalize_text')
  require(ast.dump(original_normalizer,include_attributes=False)==ast.dump(own_normalizer,include_attributes=False),'Exact original normalizer function AST')
  historical=review['historical_Source39_original212000_normal_phase']
  require(historical['accepted_ROOT_fact_only']is True and historical['native_optimizer_updates']==4000
   and historical['actual_private_drain']==20 and historical['completed_preparations']==4020
   and historical['Popen_wait_code']==0 and historical['bookkeeping_bytes_above_original']==75497472
   and historical['old_q_ready_or_pending_plan_read_admitted']is False,'Historical Source39 normal4020 facts only; distinct current Source41 startup212000/+80')
  progress,reader,mirror=current_health()
  quality=verify_quality(review,policy,dev)
  checkpoint=verify_natural212200(review)
  for expected in review['input_pins']:
   normalized={**expected,'path':str(Path(expected['path']).resolve())}
   require(any(item['pin']==normalized for item in CAPTURED),'All exact scoped input pins must have a captured single-read witness')

  write_fresh(actual/'postpositive_progress_snapshot.json',progress)
  write_fresh(actual/'current_reader_snapshot.json',{'status':reader,'mirror_state':mirror})
  write_fresh(capture_path,{'kind':'NEW_original212_quality_and_natural212200_METADATA_CAPTURE_v1',
   'inputs':CAPTURED,'no_payload_journal_plan_audio_or_closed_original_reads':True})
  result={'kind':'ASR18_original212000_quality_Source41_I29_natural212200_resumed_metadata_v1','passed':True,
   'ROOT_review':reviewpin,'metadata_source':selfpin,'ROOT97_boot_authority':bootpin,
   'original212000_quality':quality,'natural212200':checkpoint,
   'independent_full854_PRE_SAVED_acceptance':'PENDING_ROOT; original persistence counts alone are insufficient',
   'historical_Source39_normal_phase_ROOT_facts':historical,
   'current_Source41_startup_cohort':{'cohort_start':212000,'native':NATIVE,'CPU':CPU,'CPU_launcher':CPU_LAUNCHER,
    'bookkeeping_above_original_bytes':83886080,'ready_originals_CLOSED_and_not_reopened':True},
   'actual_optimizer_updates':progress['actual_optimizer_updates'],'last_saved_updates':progress['last_saved_updates'],
   'native_updates_this_process':progress['native_updates_this_process'],'full401072':True,'administrative_stop':None,
   'current_reader':reader,'capture':output_pin(capture_path),'initial':output_pin(initial_path),
   'last_sample':guard(),'self':identity(psutil.Process()),'launcher':identity(psutil.Process().parent()),
   'direct_new20_plan_or_journal_reads':False,'extra_DEV_GC_or_control':False,'Backlog_or_refresh':False,
   'durable_throughput_cause_or_fix_claim':False}
  write_fresh(result_path,result);code=0
 except BaseException as error:
  if not capture_path.exists():
   write_fresh(capture_path,{'kind':'PARTIAL_NEW_metadata_CAPTURE_v1','inputs':CAPTURED})
  if not result_path.exists():
   write_fresh(result_path,{'passed':False,'error_type':type(error).__name__,'error':str(error),
    'bounded_traceback':traceback.format_exc(limit=12),'capture':output_pin(capture_path),
    'self':identity(psutil.Process()),'launcher':identity(psutil.Process().parent())})
 finally:
  write_fresh(numeric_path,{'direct_numeric_os_exit':code,'self':identity(psutil.Process()),'launcher':identity(psutil.Process().parent())})
  print(json.dumps({'numeric_os_exit':code,'result':output_pin(result_path),'capture':output_pin(capture_path),
   'initial':output_pin(initial_path)if initial_path.exists()else None,'direct_numeric':output_pin(numeric_path)}),flush=True)
 return code
if __name__=='__main__':sys.exit(main())
