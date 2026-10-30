"""INACTIVE reviewed checkpoint observer: the ONLY production mutation is stop.json.
No exit code, zero-tail, clean teardown, restore or recovery is inferred here.
Requires a separate exact ROOT pre-stop receipt after genuine CPU acceptance.
"""
import argparse,hashlib,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent;BEGIN=time.monotonic();PEAK=0

def require(v,m):
    if not v:raise ValueError(m)
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,indent=2);f.write('\n')
def identity(pid=None):
    p=psutil.Process(os.getpid()if pid is None else pid);return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def live(w):
    try:return abs(psutil.Process(w['pid']).create_time()-w['create_time'])<.001
    except psutil.NoSuchProcess:return False
def guard():
    global PEAK
    x=sample_memory(os.getpid(),'approved_checkpoint_stop_CPU_observer_only');m=x['system_memory'];private=x['worker_private_memory']['private_bytes'];PEAK=max(PEAK,private)
    require(x['paired_sampling_within_one_second']and private<536870912 and time.monotonic()-BEGIN<900
        and m['commit_available_bytes']>=2268435456 and m['physical_available_bytes']>=1768435456
        and shutil.disk_usage('E:/').free>=100000000000,'Observer512MiB/900s/fullprefetch/originalfloors/E100GB')
    return x

def main(a):
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and os.environ.get('PYTHONUTF8')=='1'
        and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
        and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),
        'New observer process-local E/CPU-only environment')
    review_pin=pin(a.ROOT_review);review=read(a.ROOT_review)
    require(review['accepted']is True and review['reviewer']=='/root'
        and review['scope']=='ASR18_fresh_signature_collect_checkpoint_stop_v1'
        and review['observer_source']==pin(__file__),'Separate exact ROOT pre-stop source/actualCPU acceptance')
    for item in review['input_pins']:require(pin(item['path'])==item,'Exact pre-stop input pin')
    intent_pin=review['transition_intent'];intent=read(intent_pin['path'])
    proof=read(review['actual_CPU_review']['path'])
    require(proof['accepted']is True and proof['reviewer']=='/root'
        and proof['scope']=='ASR18_fresh_metadata_signature_and_original_collect_observation_policy_v1'
        and proof['signature_helper_source']==pin(F/'asr18_fresh_single_open_metadata_signature_v2.py')
        and proof['integration_helper_source']==pin(F/'asr18_fresh_metadata_signature_scope_v2.py')
        and proof['collect_observer_source']==pin(F/'asr18_observe_original_gc_collect_v2.py')
        and proof['semantic_passed']is True and proof['candidate_timing_eligible_in_THIS_bounded_fixed_order']is True
        and proof['all_exact_metadata_identities_dead']is True,
        'Completed exact signature/collect observation proof BEFORE intentional stop')
    require(intent['owned_session']==72166 and intent['old_owner_invocation']==22
        and intent['new_owner_invocation']==23 and intent['human_approval']==review['human_approval']
        and intent['old_driver_source']==review['old_driver_source'],'Same exact admitted source30 owner')
    branch=Path(intent['branch']);inv=branch/'invocations/owner_000022';stop_path=inv/'stop.json'
    require(not stop_path.exists()and all(live(w)for w in intent['expected_runtime_identities']),
        'All exact old runtime identities alive and no earlier stop request')
    out=F/'fresh_signature_collect_checkpoint_stop_observer_actual_01';require(not out.exists(),'One fresh observer');out.mkdir()
    initial=guard();m=initial['system_memory']
    require(m['commit_available_bytes']>=2805306368 and m['physical_available_bytes']>=2305306368,
        'Full prospective512MiB plus256MiB and original floors')
    entry={'observer':identity(),'launcher':identity(os.getppid()),'ROOT_pre_stop_review':review_pin,'intent':intent_pin,'initial_guard':initial};write(out/'entry.json',entry)
    stopped=False
    try:
        while True:
            guard();require(all(live(w)for w in intent['expected_runtime_identities']),
                'Old runtime changed/exited before admitted intentional stop; preserve evidence, no guessed action')
            progress=read(branch/'progress/latest.json');latest=read(branch/'latest_checkpoint.json')
            require(progress['producer']['pid']==40012 and abs(progress['producer']['create_time']-1792743161.6231525)<.001
                and progress['source']==intent['old_driver_source']and not progress['optimizer_poisoned']
                and not progress['schedule_poisoned']and progress['planned_native_updates']==401072
                and progress['administrative_update_stop']is None,'Same healthy full objective before stop')
            saved=latest['committed_updates'];actual=progress['actual_optimizer_updates']
            if saved>=intent['minimum_natural_saved_updates']and saved%200==0 and saved%4000!=0 and latest['full_state_verified']is True and saved<=actual<=saved+20 and progress['last_saved_updates']==saved:
                pointer=read(latest['pointer']['path']);persistence=read(latest['persistence']['path'])
                require(pin(latest['pointer']['path'])==latest['pointer']and pin(latest['persistence']['path'])==latest['persistence']
                    and latest['source']==intent['old_driver_source']and pointer['committed_updates']==saved
                    and persistence['passed']is True and persistence['optimizer_updates']==saved
                    and persistence['named_tensors_verified']==854 and persistence['weights_verified']==207
                    and persistence['Adam_states_verified']==215
                    and persistence['saved_semantic_state_verified']['passed']is True
                    and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256']
                    and persistence['journal_prefix']==pointer['journal_prefix'],'Natural full200 PRE-SAVED/semantic publication')
                # Reread compact publication/progress immediately before the one exclusive write.
                require(read(branch/'latest_checkpoint.json')==latest and all(live(w)for w in intent['expected_runtime_identities']),
                    'Unchanged natural publication/exact owner at stop decision')
                fresh=read(branch/'progress/latest.json')
                require(saved<=fresh['actual_optimizer_updates']<=saved+20 and fresh['last_saved_updates']==saved
                    and fresh['source']==intent['old_driver_source'],
                    'Early after publication, far from next200 writer; zero-tail remains unclaimed')
                request={'kind':'ASR18_reviewed_fresh_signature_collect_checkpoint_cooperative_stop_v1',
                    'transition_intent':intent_pin,'ROOT_pre_stop_review':review_pin,'human_approval':intent['human_approval'],
                    'observer':entry['observer'],'observed_saved_optimizer_updates':saved,
                    'observed_native_updates':fresh['actual_optimizer_updates'],'natural_pointer':latest['pointer'],
                    'natural_persistence':latest['persistence'],'timestamp_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                    'race_and_possible_unsaved_tail_explicit':True,'normal0_claimed':False}
                write(stop_path,request);stopped=True
                write(out/'stop_requested.json',{'stop':pin(stop_path),'request':request})
                break
            time.sleep(1)
        # Observe exact deaths only. NEVER terminate a production process or consume its owned exec.
        while any(live(w)for w in intent['expected_runtime_identities']):guard();time.sleep(.25)
        write(out/'report.json',{'kind':'ASR18_actual_reviewed_checkpoint_stop_observer_report_v1','stop_requested':True,
            'stop':pin(stop_path),'all_five_exact_runtime_dead':True,'runtime_identities':intent['expected_runtime_identities'],
            'actual_Popen_or_TOOL_exit_code':'NOT_CONSUMED_BY_OBSERVER','numeric_native_CPU_wrapper_codes':'NOT_INFERRED',
            'ROOT_pre_stop_review':review_pin,'intent':intent_pin,'peak_private_bytes':PEAK,'final_guard':guard(),
            'actual_saved_frontier_and_tail_must_be_sealed_after_owned_TOOL_exit':True})
        print(json.dumps({'report':pin(out/'report.json'),'stop':pin(stop_path)}),flush=True)
    except BaseException as error:
        write(out/'actual_failure.json',{'error':repr(error),'stop_was_written':stopped,'observer':entry,
            'known_runtime_liveness':[{'identity':w,'exact_alive':live(w)}for w in intent['expected_runtime_identities']]})
        raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ROOT-review',required=True);main(p.parse_args())
