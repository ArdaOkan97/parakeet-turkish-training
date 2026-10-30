"""Actual admitted pause metadata/NEW suffix preservation. No payload read."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import psutil

P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
from asr18_one_postDEV_gc_refresh_checkpoint_transition_v2 import validate_intentional_optimizer_journal_gap
W=Path(__file__).resolve().parent
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
I=B/'invocations/owner_000026'
BEGIN=time.monotonic()


def require(value,message):
    if not value:raise ValueError(message)


def who(proc):return dict(pid=proc.pid,create_time=proc.create_time(),ppid=proc.ppid())


def alive(identity):
    try:
        proc=psutil.Process(identity['pid'])
        return proc.create_time()==identity['create_time']and proc.ppid()==identity['ppid']and proc.is_running()
    except psutil.NoSuchProcess:return False


def pin(path):
    path=Path(path);require(path.stat().st_size<=8388608,'Bounded compact/source metadata')
    raw=path.read_bytes();return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def read(path):
    path=Path(path);require(path.stat().st_size<=8388608,'Bounded compact JSON only')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf8',newline='\n')as stream:
        json.dump(value,stream,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return pin(path)


def guard():
    value=sample_memory(os.getpid(),'actual_pause204800_metadata_runtime')
    require(value['paired_sampling_within_one_second']is True
        and value['system_memory']['commit_available_bytes']>=2268435456
        and value['system_memory']['physical_available_bytes']>=1768435456
        and value['worker_private_memory']['private_bytes']<536870912
        and shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-BEGIN<180,
        'Original runtime/fullprefetch/private512MiB/E100GB/180s')
    return value


def main():
    me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    initial=sample_memory(me.pid,'actual_pause204800_metadata_initial_before_predicate')
    initial_pin=write(W/'pause204800_actual_sealer_initial_before_predicate_v1.json',
        dict(sample=initial,sealer=who(me),launcher=who(me.parent()),TORCH_EXTENSIONS_DIR=os.environ['TORCH_EXTENSIONS_DIR']))
    require(initial['paired_sampling_within_one_second']is True
        and initial['system_memory']['commit_available_bytes']>=5055306368
        and initial['system_memory']['physical_available_bytes']>=2305306368
        and initial['worker_private_memory']['private_bytes']<536870912
        and shutil.disk_usage('E:/').free>=100000000000,'Original FULL225+FULLprefetch+prospective512 admission')
    require(os.environ['CUDA_VISIBLE_DEVICES']=='-1'and all(os.environ[k]=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
        and Path(os.environ['TORCH_EXTENSIONS_DIR'])==Path('E:/gpu_training_scratchpad/cache/torch_extensions'),'Exact E CPU-only process-local environment')
    actor_dir=W/'postDEV204000_saved204800_checkpoint_pause_actual_01'
    actor=read(actor_dir/'result.json');stage=read(actor['prestop_evidence']['path'])
    require(actor['passed']is True and actor['stop_published']is True and stage['saved']==204800,'Actual new controller published once')
    preflight=read(W/'pause204800_actual_terminal_metadata_preflight_v1.json')
    failure=preflight['metadata']['actual_failure.json']['value']
    monitor=preflight['metadata']['monitor_completion.json']['value']
    scalar=preflight['metadata']['private_metadata_worker_completion.json']['value']
    latest=read(B/'latest_checkpoint.json');saved=latest['committed_updates']
    require(saved==204800,'Actual greatest published checkpoint; do not silently pin obsolete target')
    checkpoint=B/'checkpoints'/('update_'+str(saved).zfill(9))
    pointer_pin=pin(checkpoint/'pointer.json');pointer=read(pointer_pin['path'])
    persistence_pin=pin(checkpoint/'persistence_verified.json');persistence=read(persistence_pin['path'])
    require(latest['pointer']==pointer_pin and latest['persistence']==persistence_pin
        and latest['full_state_verified']is True and pointer['committed_updates']==saved
        and pointer['all_window_updates_committed']is True and pointer['diagnostic_only']is False,
        'Actual greatest full checkpoint pointer and publication')
    require(persistence['passed']is True and persistence['optimizer_updates']==saved
        and persistence['named_tensors_verified']==854 and persistence['weights_verified']==207 and persistence['Adam_states_verified']==215
        and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256']
        and persistence['saved_semantic_state_verified']['passed']is True,'Original published full854/207/215 semantic PRE-SAVED receipt')
    payload=Path(persistence['payload']['path']);payload_stat=payload.stat()
    require(payload_stat.st_size==persistence['payload']['bytes'],'Payload STAT ONLY; no payload hash or restore')
    saved_prefix=pointer['journal_prefix'];journal=Path(saved_prefix['path']);stat=journal.stat()
    offset=saved_prefix['offset'];count=stat.st_size-offset
    require(0<=count<=4*1024**2,'Only NEW bounded actual unsaved suffix, never old prefix/whole journal')
    original_stat=dict(kind='STAT_ONLY_original_closed_journal_v1',path=str(journal),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    # The only admitted ORIGINAL journal read. Preserve exact bytes BEFORE parse.
    with journal.open('rb')as stream:
        stream.seek(offset);raw=stream.read(count)
    require(len(raw)==count,'Exact immutable new suffix extent')
    copy_path=W/'source37_saved204800_unsaved_journal_suffix_COPY_v1.jsonl'
    with copy_path.open('xb')as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    tail_pin=dict(path=str(copy_path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    lines=raw.splitlines();rows=[json.loads(line)for line in lines if line.strip()]
    require(len(rows)<=199,'Exact natural200 tail bound')
    published=rows[-1]['optimizer_updates']if rows else saved
    require([r['optimizer_updates']for r in rows]==list(range(saved+1,published+1)), 'Every ordered actual published positive, no omission')
    require(all(r['kind']=='ASR18_committed_training_update_v1'and r['native']['update']==r['optimizer_updates']
        and math.isfinite(r['native']['mean_objective'])and r['native']['mean_objective']>0
        and math.isfinite(r['native']['global_grad_norm'])and r['native']['global_grad_norm']>0
        and len(r['native']['source_ids'])==16 and len(r['native']['languages'])==16
        and r['native']['languages'][:14]==['tr']*14
        and r['broad_draws']==14*r['optimizer_updates']-3 and r['replay_updates']==r['optimizer_updates']and r['cleaner_draws']==0
        for r in rows),'Actual finite positive native16row receipts and original broad2 partial-tail counters14u-3/u/0')
    require(len({r['transaction_sha256']for r in rows})==len(rows)
        and len({json.dumps(r['prepared_evidence']['token'],sort_keys=True,separators=(',',':'))for r in rows})==len(rows),
        'Every genuine new transaction/token unique')
    require(all(r['prepared_evidence']['producer_source_sha256']=='e3d01517ac3af9e481a27b0127474c7a8fbc2d9351f7fcc6aefcf36691675d0e'for r in rows),
        'Current unchanged CPU14 source on every actual consumed packet')
    gap=validate_intentional_optimizer_journal_gap(failure,published)
    require(failure['last_checkpoint']==saved and saved<=published<=failure['actual_optimizer_updates']<saved+200,
        'Actual optimizer frontier vs saved/published, original0or1 trace')
    require(monitor['joined']is True and monitor['clean']is True and monitor['error']is None,'Actual original Monitor joined clean')
    require(journal.stat().st_size==stat.st_size and journal.stat().st_mtime_ns==stat.st_mtime_ns,'Original suffix/prefix never modified')
    original_ids=[dict(pid=19320,create_time=1792803517.7236164,ppid=39856),dict(pid=39856,create_time=1792803517.7165897,ppid=21928),
        dict(pid=21928,create_time=1792803516.8916032,ppid=41376),dict(pid=42068,create_time=1792830275.0700862,ppid=40536),
        dict(pid=40536,create_time=1792830275.0545292,ppid=19320),dict(pid=38444,create_time=1792830275.0592997,ppid=40536)]
    captured=[d['identity']for d in stage['current_native_descendants']['descendants']]
    identities=list({(i['pid'],i['create_time']):i for i in original_ids+captured}.values())
    require(all(not alive(i)for i in identities),'ALL known original/current/captured identities exact dead including unclassified conhost')
    readers=[dict(pid=26892,create_time=1792652904.3359847,ppid=30128),dict(pid=30128,create_time=1792652904.283631,ppid=10644)]
    require(all(alive(i)for i in readers),'Same healthy reader identities remain untouched')
    reader_path=Path('E:/gpu_training_scratchpad/analysis/BPE9223_Trackio_live_v1/current_status.json')
    reader=read(reader_path)
    require(reader['error']is None and reader['training_worker_alive']is False,'Reader workerfalse expected after intentional stop')
    lease_path=P/'work/gpu_training_owner.json';lease=read(lease_path)
    require(lease['actual_GPU_running']is False and lease['actual_owned_worker_pid']is None,'Actual original lease released')
    tool_pin=write(W/'source37_owned26493_actual_TOOL1_post204800_pause_v1.json',dict(exit_code=1,chunk_id='7cff7f',exclusive_session=26493,
        consumed_once_after_exact_family_terminal=True,normal0_claimed=False,inner_native_CPU_wrapper_OS_codes='UNKNOWN_UNRECORDED'))
    actor_tool=write(W/'checkpoint204800_pause_actual_owned_TOOL0_v1.json',dict(exit_code=0,chunk_id='58e160',result=pin(actor_dir/'result.json')))
    tail_review=write(W/'source37_saved204800_new_tail_positive_metadata_v1.json',dict(kind='ASR18_actual_saved204800_new_tail_positive_metadata_v1',
        original_journal_STAT=original_stat,original_seek_start=offset,original_seek_end=stat.st_size,copy=tail_pin,
        positive_count=len(rows),first=saved+1 if rows else None,last=published,finite_positive_objective_and_gradient=True,
        all16_ordered_native_IDs_preserved=True,TR_rows=14,replay_rows=2,broad2_counter_formula='14*u-3',replay_counter='u',cleaner_counter=0,
        unique_tokens=True,unique_transactions=True,current_CPU14_source=True,optimizer_vs_published=gap,
        historical_tail_only_no_replay_or_promotion=True,source_bound_actual_row_keys=sorted(rows[0])if rows else []))
    evidence=dict(old_driver=pin(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v37.py'),
        old_wrapper=pin(P/'work/research/run_ASR18_BPE9223_resident_RAM_autocontinue_v37.py'),old_CPU=pin(S/'fastmss_resident_feature_stream_v14.py'),
        old_configuration=pin(I/'configuration.json'),pointer=pointer_pin,persistence=persistence_pin,
        actual_failure=preflight['metadata']['actual_failure.json']['pin'],monitor_completion=preflight['metadata']['monitor_completion.json']['pin'],
        scalar_completion=preflight['metadata']['private_metadata_worker_completion.json']['pin'],stop=pin(I/'stop.json'),
        watcher_result=pin(actor_dir/'result.json'),prestop_evidence=actor['prestop_evidence'],original_journal_stat=original_stat,
        tail_copy=tail_pin,tail_review=tail_review,production_TOOL=tool_pin,watcher_TOOL=actor_tool,
        ROOT_pause_scope=pin(W/'ROOT_source37_postDEV204000_saved204800_checkpoint_pause_exact_ONCE_scope_v1.json'))
    receipt=dict(kind='ASR18_actual_source37_postDEV204000_saved204800_throughput_transition_v1',owned_session=26493,
        old_owner_invocation=26,new_owner_invocation=27,actual_outer_TOOL_exit_code=1,actual_outer_TOOL_chunk_id='7cff7f',normal0_claimed=False,
        intentional_cooperative_stop_observed=True,actual_optimizer_updates=failure['actual_optimizer_updates'],published_journal_optimizer_updates=published,
        last_fully_saved_optimizer_updates=saved,optimizer_vs_journal_frontier=gap,
        all_recorded_runtime_identities=identities,all_recorded_runtime_identities_exact_dead=True,
        reader_identities=readers,reader_status=reader,reader_status_pin=pin(reader_path),lease=lease,
        known_metadata_identities=[actor['observer'],actor['launcher'],preflight['observer'],preflight['launcher']],
        evidence=evidence,saved_journal_prefix=saved_prefix,preserved_tail_positive_updates=len(rows),unsaved_tail_bytes=count,
        original_tail_modified=False,tail_is_historical_only_requires_genuine_redo=True,
        full_checkpoint_restore_and_genuine_redo_required=True,all_saved_payload_reads='STAT_ONLY_NO_HASH_OR_RESTORE',
        published_full854_semantic_PRE_SAVED_receipt=True,independent_ROOT_full_metadata_acceptance_pending=True,
        payload_STAT=dict(path=str(payload),bytes=payload_stat.st_size,mtime_ns=payload_stat.st_mtime_ns),
        inner_native_OS_exit=None,inner_CPU_OS_exit=None,wrapper_OS_exit=None,inner_OS_exits_status='UNKNOWN_UNRECORDED',
        original_I26_marker_permanently_preserved=True,no_LR_GC_science_guard_reader_change=True,
        inherited180000_event_consumed_no_second_event=True,actual_refresh_or_capture_or_sync_here=False,
        future_validation=dict(restored=saved,start=saved+20,end=max(saved+20+1600,208000+1020),all20rows_all_saves_all_pauses=True,
            primary_sustained_inclusive_rate_above=70,primary_inclusive_first_final_decline_max_percent=5),
        historical_metadata_identity_limitations=dict(denied_completion_PID16424_ctime_launcher='UNRECORDED',
            wrapper_interpreter_launcher_PID41376_ctime='UNRECORDED_NOT_INFERRED_FROM_PPID'),
        initial=initial_pin,final_guard=guard(),sealer=who(me),sealer_launcher=who(me.parent()),source=pin(__file__))
    final=write(W/'actual_source37_saved204800_checkpoint_pause_closed_v1.json',receipt)
    print(json.dumps(dict(receipt=final,tail=tail_review,tail_copy=tail_pin,saved=saved,actual=failure['actual_optimizer_updates'],published=published,
        gap=gap,sealer=who(me),launcher=who(me.parent()))),flush=True)


if __name__=='__main__':main()
