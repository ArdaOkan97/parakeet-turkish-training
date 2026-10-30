"""Readonly fixed-ordinal comparison; no model/audio/sampler or active-source edits.

The fixture reuses genuine admitted metadata bindings and original row methods.
It explicitly constructs only the metadata resolver view; it does not claim a
full training-context constructor, fresh corpus verification or production fit.
"""
import copy,hashlib,json,os,random,sys,threading,time,traceback
from collections import OrderedDict
from pathlib import Path
P=Path(r'C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from asr18_BPE9223_source_context_v2 import FullRunSourceContext9223,FinalBroadRows9223,BROAD_N,KIND
from bpe_declared_geometry_index_v1 import DeclaredGeometryIndex,open_final,signature
from bpe_windows_final_metadata_lock_v1 import FinalMetadataReadLocks
from asr18_BPE9223_readonly_metadata_pagecache_v1 import install,ALLOCATION_ADMISSION_BYTES
import psutil
out=Path(sys.argv[1]);require(not out.exists(),'Fresh E comparison records');out.mkdir()
require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and sys.flags.utf8_mode==1,'CPU-only upfront UTF8')
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
began=time.monotonic();stop=threading.Event();errors=[];peak=[0];samples=[0]
def guard():
    require(not errors,'Actual readonly CPU fixture monitor: '+repr(errors))
    require(time.monotonic()-began<180,'Bounded180s metadata-only fixture')
def sample():
    from resource_watchdog_v2 import sample_memory
    s=sample_memory(os.getpid(),'readonly_metadata_pagecache_CPU_comparison')
    peak[0]=max(peak[0],s['worker_private_memory']['private_bytes']);samples[0]+=1
    require(peak[0]<=512*1024**2 and s['system_memory']['commit_available_bytes']>=2000000000
        and s['system_memory']['physical_available_bytes']>=1500000000,'512MiB CPU fixture and unchanged real headroom')
    return s
def watch():
    try:
        while not stop.wait(.25):sample()
    except BaseException as e:errors.append(repr(e))
watcher=threading.Thread(target=watch,daemon=True);watcher.start()
def admit(name,increment):
    guard();s=sample();require(increment==ALLOCATION_ADMISSION_BYTES
        and s['system_memory']['commit_available_bytes']-increment>=2000000000
        and s['system_memory']['physical_available_bytes']-increment>=1500000000,'Full candidate cache+margin prospective headroom')
    return {'passed':True,'name':name,'prospective_increment_bytes':increment,'sample':s}
contexts=[]
try:
    N=Path(r'E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1');B=Path(r'E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
    cpin=record(N/'resident_RAM_DEV_frontend_source_v1/configuration.json');c=read_json(cpin['path'])
    runtime_pin=c['planning_arguments']['DATA_evidence']['BPE9223_population'];require(record(runtime_pin['path'])=={k:runtime_pin[k]for k in ('path','bytes','sha256')},'Original immutable runtime binding')
    runtime=read_json(runtime_pin['path'])
    pins={'final_catalog':runtime['source_catalog_unchanged'],'final_labels':runtime['databases']['broad_labels'],
        'declared_geometry_index':runtime['databases']['declared_geometry']}
    paths=[Path(pins[k]['path']).resolve()for k in ('final_catalog','final_labels','declared_geometry_index')]
    for pin in pins.values():require(signature(pin['path'])=={k:pin[k]for k in ('path','bytes','mtime_ns')},'Unchanged source signature from genuine original full-hash admission')
    frozen=[];expected=[]
    for index in (50,75,100,125,150,170):
        p=B/'invocations/owner_000005/plans'/('plan_'+str(index).zfill(6)+'.json');pin=record(p);plan=read_json(p)
        require(len(plan['updates'])==20 and plan['initial_committed_updates']>=4000
            and plan['initial_committed_updates']+20<=7400,'Past frozen real20-update plan only')
        rows=[r for update in plan['updates']for r in update['rows']if r['language']=='tr']
        require(len(rows)==280,'All14 TR rows per original update; no subset/filter')
        expected.extend((r['broad_original']['ordinal'],r)for r in rows)
        frozen.append({'plan':pin,'initial_updates':plan['initial_committed_updates'],'rows':len(rows)})
    rng_before=random.getstate()
    def open_context():
        resolver=FinalBroadRows9223.__new__(FinalBroadRows9223)
        resolver.guard=guard;resolver.cache=OrderedDict();resolver.poisoned=False;resolver.population=BROAD_N;resolver.connections=[]
        resolver.metadata_locks=FinalMetadataReadLocks(paths);resolver.signatures=[signature(p)for p in paths[:2]]
        for p in paths[:2]:resolver.connections.append(open_final(p))
        geometry=DeclaredGeometryIndex.__new__(DeclaredGeometryIndex)
        geometry.guard=guard;geometry.path=paths[2];geometry.signature=signature(paths[2]);geometry.connection=open_final(paths[2]);geometry.poisoned=False
        resolver.geometry=geometry
        context=FullRunSourceContext9223.__new__(FullRunSourceContext9223)
        context.closed=False;context.resolver=resolver;context.validated={'kind':KIND,'bindings':pins}
        contexts.append(context);return context
    baseline=open_context();candidate=open_context();configuration=install(candidate,guard=guard,admit=admit)
    def execute(context,profile,phase):
        guard();began_phase=time.monotonic();h=hashlib.sha256();count=0
        for ordinal,planned in expected:
            actual=context.resolver.row(ordinal)
            require(actual==planned,'Exact original full row/text/labels/geometry/source identity equals frozen plan')
            h.update(bytes.fromhex(digest(actual)));count+=1
        guard();elapsed=time.monotonic()-began_phase
        result={'profile':profile,'phase':phase,'wall_seconds':elapsed,'row_count':count,'all_full_rows_equal_frozen_plan':True,
            'full_row_sequence_sha256':h.hexdigest(),'SQLite_cache_KiB':[v.execute('PRAGMA cache_size').fetchone()[0]for v in [*context.resolver.connections,context.resolver.geometry.connection]]}
        print(json.dumps(result),flush=True);return result
    results=[execute(baseline,'4MiB','common_warmup'),execute(candidate,'64MiB','common_warmup')]
    for phase,context,profile in ((1,baseline,'4MiB'),(2,candidate,'64MiB'),(3,candidate,'64MiB'),(4,baseline,'4MiB')):
        results.append(execute(context,profile,'ABBA_'+str(phase)))
    require(len({v['full_row_sequence_sha256']for v in results})==1 and random.getstate()==rng_before
        and 'torch'not in sys.modules and 'numpy'not in sys.modules,'Same exact rows, no RNG/sampler/native/audio/numerical imports')
    summary={'kind':'ASR18_actual_frozen_plan_metadata_pagecache_CPU_comparison_v1','passed':True,'source':record(__file__),
        'helper_source':record(S/'asr18_BPE9223_readonly_metadata_pagecache_v1.py'),'configuration':cpin,'runtime_binding':runtime_pin,
        'metadata_bindings':pins,'frozen_plans':frozen,'actual_context_constructor_scope':'explicit metadata-only view; original row/declaration methods and actual locks; not full training context admission',
        'original_full_corpus_source_hash_review_reused':True,'full_metadata_source_SHA_repeated':False,
        'candidate_configuration':configuration,'results':results,
        'sampler_draws_or_optimizer_updates':0,'RNG_unchanged':True,'original_audio_or_model_loaded':False,
        'active_metadata_DB_source_or_training_process_changed':False,
        'OS_cache_scope':'Shared OS cache not flushed or controlled. Common warmup plus ABBA reports warm fixed-working-set comparison only; not cold full-corpus or production speed proof.',
        'total_wall_seconds':time.monotonic()-began}
    write_json(out/'report.json',summary);print(json.dumps({'passed':True,'report':record(out/'report.json')}),flush=True)
except BaseException as e:
    write_json(out/'failure.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
finally:
    for context in contexts:context.resolver.close()
    stop.set();watcher.join(3)
    write_json(out/'monitor_completion.json',{'samples':samples[0],'private_peak_bytes':peak[0],'errors':errors,'joined':not watcher.is_alive()})
