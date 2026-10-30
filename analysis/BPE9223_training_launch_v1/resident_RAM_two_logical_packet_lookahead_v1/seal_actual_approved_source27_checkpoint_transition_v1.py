"""INACTIVE metadata-only actual source27 stop closure sealer; no restore/launch.
Reads ONLY preserved post-checkpoint suffix (at most4MiB), compact records, source
and original20 STAT. Never reads old20 prefix/whole or old18/19/tensor/audio.
"""
import argparse,hashlib,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent;BEGIN=time.monotonic()
def require(v,m):
    if not v:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
    with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,indent=2);f.write('\n')
def dead(w):
    try:return abs(psutil.Process(w['pid']).create_time()-w['create_time'])>=.001
    except psutil.NoSuchProcess:return True
def guard():
    v=sample_memory(os.getpid(),'approved_actual_checkpoint_stop_metadata_seal');m=v['system_memory']
    require(v['paired_sampling_within_one_second']and v['worker_private_memory']['private_bytes']<536870912
        and time.monotonic()-BEGIN<180 and m['commit_available_bytes']>=2268435456
        and m['physical_available_bytes']>=1768435456 and shutil.disk_usage('E:/').free>=100000000000,
        'Metadata512MiB/180s/fullprefetch/originalfloors/E100GB')
    return v

def main(a):
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and os.environ.get('PYTHONUTF8')=='1'
        and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
        and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),
        'New metadata process-local E/CPU-only environment')
    review_pin=pin(a.ROOT_review);review=read(a.ROOT_review);intent_pin=review['transition_intent'];intent=read(intent_pin['path'])
    require(review['accepted']is True and review['reviewer']=='/root'
        and review['scope']=='ASR18_approved_two_logical_packet_checkpoint_stop_v1'
        and review['closure_sealer_source']==pin(__file__),'Exact reviewed metadata-only stop sealer')
    for item in review['input_pins']:require(pin(item['path'])==item,'Frozen pre-stop scope input unchanged')
    initial=guard();m=initial['system_memory'];require(m['commit_available_bytes']>=2805306368 and m['physical_available_bytes']>=2305306368,
        'Full prospective512MiB and256MiB original admission')
    runtime=intent['expected_runtime_identities'];require(all(dead(w)for w in runtime),'All five exact old production identities dead')
    tool=read(a.actual_TOOL);require(tool['owned_session']==56620 and tool['actual_TOOL_exit_code']==1 and tool['normal0_claimed']is False,
        'Real consumed OWNED56620 TOOL1, no numeric native/CPU/wrapper invention')
    branch=Path(intent['branch']);inv=branch/'invocations/owner_000020';control=Path(intent['old_control'])
    failure=read(inv/'actual_failure.json');latest=read(branch/'latest_checkpoint.json');saved=latest['committed_updates'];actual=failure['actual_optimizer_updates']
    require('Actual owner cooperative stop'in failure['error']and failure['last_checkpoint']==saved and saved<=actual<saved+200
        and latest['full_state_verified']is True,'Actual intentional exception at honest live/saved frontier')
    pointer=read(latest['pointer']['path']);persistence=read(latest['persistence']['path']);prefix=pointer['journal_prefix']
    require(pin(latest['pointer']['path'])==latest['pointer']and pin(latest['persistence']['path'])==latest['persistence']
        and saved%200==0 and saved%4000!=0 and pointer['committed_updates']==saved
        and persistence['passed']is True and persistence['optimizer_updates']==saved
        and persistence['named_tensors_verified']==854 and persistence['weights_verified']==207 and persistence['Adam_states_verified']==215
        and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256']
        and persistence['saved_semantic_state_verified']['passed']is True and persistence['journal_prefix']==prefix,
        'Actual full854/207/215 PRE-SAVED semantic saved state, no tensor load')
    journal=branch/'journals/commits_000020.jsonl';require(Path(prefix['path'])==journal,'Current source27 journal20 only')
    stat=journal.stat();tail_bytes=stat.st_size-prefix['offset'];require(0<=tail_bytes<=4*1024**2,'Only bounded actual later suffix')
    out=F/'actual_approved_source27_checkpoint_transition_01';require(not out.exists(),'One fresh actual closure');out.mkdir()
    with journal.open('rb')as f:f.seek(prefix['offset']);raw=f.read(tail_bytes);require(f.read(1)==b'','Actual exact bounded suffix ends at closed writer EOF')
    require(journal.stat().st_size==stat.st_size and journal.stat().st_mtime_ns==stat.st_mtime_ns,'Closed writer unchanged across bounded suffix preservation')
    events=[json.loads(x)for x in raw.splitlines()]
    require(len(events)==actual-saved and all(x['optimizer_updates']==x['native']['update']==x['prepared_evidence']['actual_positive_native_update']==u
        for u,x in zip(range(saved+1,actual+1),events)),'Every later positive attempt preserved in exact order; never replayed or promoted')
    tail_path=out/'preserved_unsaved_positive_tail.jsonl'
    with tail_path.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
    evidence={'pointer':latest['pointer'],'persistence':latest['persistence'],'old_driver':intent['old_driver_source'],
        'old_wrapper':intent['old_wrapper_source'],'old_CPU':intent['old_CPU_source'],
        'old_configuration':pin(inv/'configuration.json'),'old_boot':pin(inv/'boot.json'),
        'old_CPU_ready':intent['old_CPU_ready'],'current_child':pin(control/'current_child.json'),
        'actual_failure':pin(inv/'actual_failure.json'),'stop':pin(inv/'stop.json'),
        'transition_intent':intent_pin,'ROOT_pre_stop_review':review_pin,'actual_TOOL':pin(a.actual_TOOL),
        'native_log':pin(control/'native_000020.log')}
    monitor=inv/'monitor_completion.json'
    if monitor.exists():evidence['original_monitor_completion']=pin(monitor)
    lease=read(P/'work/gpu_training_owner.json')
    require(lease['owner']=='/root/memory_diagnosis'and lease['actual_GPU_running']is False
        and lease['actual_owned_worker_pid']is None,'Actual old lease released after process closure, no automatic takeover mutation')
    value={'kind':'ASR18_actual_approved_source27_checkpoint_lookahead_transition_v1','owned_session':56620,
        'old_owner_invocation':20,'new_owner_invocation':21,'actual_outer_TOOL_exit_code':1,'normal0_claimed':False,
        'intentional_cooperative_stop_observed':True,'human_approval':intent['human_approval'],
        'evidence':evidence,'all_five_runtime_identities':runtime,'all_five_exact_dead':True,
        'last_fully_saved_optimizer_updates':saved,'actual_optimizer_updates':actual,
        'unsaved_positive_tail_count':len(events),'saved_journal_prefix':prefix,
        'closed_old_journal_STAT':{'path':str(journal),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'whole_content_hashed':False},
        'preserved_unsaved_positive_tail':pin(tail_path),'old_prefix_or_whole_journal_read':False,
        'old18_or19_read':False,'tail_will_be_genuinely_redone_not_replayed':True,
        'native_numeric_OS_exit_code':None,'CPU_numeric_OS_exit_code':None,'wrapper_numeric_OS_exit_code':None,
        'unknown_separate_numeric_exits_not_invented':True,
        'monitor_completion':read(monitor)if monitor.exists()else None,
        'no_clean_monitor_report_inferred_if_absent':True,'actual_failure':failure,'final_guard':guard()}
    path=out/'actual_transition_closure.json';write(path,value)
    print(json.dumps({'actual_transition_closure':pin(path),'saved':saved,'actual':actual,'tail':pin(tail_path)}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ROOT-review',required=True);p.add_argument('--actual-TOOL',required=True);main(p.parse_args())
