"""ONE first-pass, source-bound metadata observation; no training authority.

The ORIGINAL row/declaration methods execute unchanged. Proxies only time their
original calls in this separate process. No audio/model/numerical runtime,
authoritative sampler, DB hashing, cache flush, second query pass or production modification.
"""
import argparse
import ast
from collections import defaultdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import textwrap

import psutil

P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
import asr18_BPE9223_private_scalar_prefetch_v1 as prefetch
import asr18_BPE9223_source_context_v2 as context
import bpe_declared_geometry_index_v1 as geometry
from bpe_broad_production_state_v1 import canonical, digest, require, CompletePassCursor

F=Path(__file__).parent
INPUT=F/'exact_already_planned280_scalar_metadata_input_v1.json'
RUNTIME=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/verified_population_v1/final_runtime_bindings_9223_v1.json')
CONFIG=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01/invocations/owner_000021/configuration.json')
EXPECTED_NATIVE={'pid':37836,'create_time':1792733651.0335317}
OUTPUT=F/'scalar_metadata_stage_timing_actual_01'
BEGIN=time.monotonic()


def pin(path):
    path=Path(path).resolve();data=path.read_bytes()
    require(len(data)<=8*1024**2,'Only bounded compact/source input hashing, never DB payload')
    return {'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def load(path):
    path=Path(path);require(path.stat().st_size<=8*1024**2,'Compact JSON only')
    return json.loads(path.read_text(encoding='utf-8'))


def identity(proc):
    return {'pid':proc.pid,'create_time':proc.create_time(),'ppid':proc.ppid()}


def exact_alive(who):
    try:return abs(psutil.Process(who['pid']).create_time()-who['create_time'])<.001
    except psutil.NoSuchProcess:return False


class Measurements:
    """Flat LEAF aggregates; total row time overlaps all its component clocks."""
    def __init__(self):
        self.stats=defaultdict(lambda:{'calls':0,'wall_ns':0,'thread_cpu_ns':0,'max_wall_ns':0})

    def call(self,tag,fn,*args,**kwargs):
        wall=time.perf_counter_ns();cpu=time.thread_time_ns()
        try:return fn(*args,**kwargs)
        finally:
            elapsed=time.perf_counter_ns()-wall;used=time.thread_time_ns()-cpu
            value=self.stats[tag];value['calls']+=1;value['wall_ns']+=elapsed
            value['thread_cpu_ns']+=used;value['max_wall_ns']=max(value['max_wall_ns'],elapsed)


class CursorProxy:
    def __init__(self,cursor,tag,measure):self.cursor=cursor;self.tag=tag;self.measure=measure
    def fetchone(self):return self.measure.call(self.tag+'.fetchone',self.cursor.fetchone)


class ConnectionProxy:
    def __init__(self,connection,tag,measure):
        self.connection=connection;self.tag=tag;self.measure=measure;self.closed=False
    def execute(self,*args,**kwargs):
        cursor=self.measure.call(self.tag+'.execute',self.connection.execute,*args,**kwargs)
        return CursorProxy(cursor,self.tag,self.measure)
    def close(self):
        result=self.connection.close();self.closed=True;return result


def method_ast(fn):
    return hashlib.sha256(ast.dump(ast.parse(textwrap.dedent(inspect.getsource(fn))),include_attributes=False).encode()).hexdigest()


def run(review,inputs,plan,bindings,signatures,stop,status):
    measure=Measurements();reader=None;proxies=[];rows={};row_observations=[]
    originals={'context_signature':context.signature,'geometry_signature':geometry.signature,
               'deepcopy':context.deepcopy,'validate':geometry.validate_declared_row}
    original_row=context.FinalBroadRows9223.row
    original_declaration=geometry.DeclaredGeometryIndex.declaration
    def guard():
        if stop.is_set():raise InterruptedError('Metadata observer external/cooperative cancellation')
        require(time.monotonic()-BEGIN<120,'Bounded120s scalar metadata observation')
    try:
        sentinel=RuntimeError('METHOD_STUB_original_callback_exception_identity')
        def fail_stub():raise sentinel
        try:measure.call('METHOD_STUB_exception_passthrough',fail_stub)
        except RuntimeError as caught:require(caught is sentinel,'Timing wrapper preserves original exception identity')
        else:raise AssertionError('METHOD_STUB exception was swallowed')
        status['METHOD_STUB_timing_callback_exception_identity_preserved']=True
        guard();reader=prefetch._ThreadOwnedBroadRows(bindings,signatures,guard=guard)
        require(all(c.execute('PRAGMA query_only').fetchone()[0]==1 and
                    c.execute('PRAGMA cache_size').fetchone()[0]==-4096
                    for c in [*reader.connections,reader.geometry.connection]),'Original3 readonly4MiB caches')
        require(len(reader.cache)==0,'First pass, original empty32row LRU')
        for c,tag in zip(reader.connections,('catalog_sql','labels_sql')):
            proxies.append(ConnectionProxy(c,tag,measure))
        reader.connections=proxies[:]
        gp=ConnectionProxy(reader.geometry.connection,'geometry_sql',measure)
        proxies.append(gp);reader.geometry.connection=gp
        context.signature=lambda *a,**k:measure.call('catalog_labels_signature',originals['context_signature'],*a,**k)
        geometry.signature=lambda *a,**k:measure.call('geometry_signature',originals['geometry_signature'],*a,**k)
        context.deepcopy=lambda *a,**k:measure.call('row_deepcopy',originals['deepcopy'],*a,**k)
        geometry.validate_declared_row=lambda *a,**k:measure.call('geometry_validation',originals['validate'],*a,**k)
        require(context.FinalBroadRows9223.row is original_row and
                geometry.DeclaredGeometryIndex.declaration is original_declaration,
                'Original row and declaration function objects unchanged')
        oracle=[r for update in plan['updates'] for r in update['rows'][:14]]
        require(len(oracle)==280,'Exact20 original groups of14 broad rows; no replay queries')
        for expected,original in zip(inputs['original_ordered_rows'],oracle):
            require(original['ordinal']==expected['ordinal'] and original['source_id']==expected['source_id']
                    and digest(original)==expected['canonical_original_plan_row_sha256'],'Pinned original plan row oracle')
        process=psutil.Process();io_before=process.io_counters()._asdict()
        request_wall=time.perf_counter_ns();request_cpu=time.thread_time_ns();encoded=0
        final_cursor=plan['final_private_planning_sampler']['broad_cursor']
        initial_cursor=originals['deepcopy'](final_cursor);initial_cursor['draws']-=280
        require(initial_cursor['draws']>=0,'Exact original private cursor rewind, no new seed')
        cursor=CompletePassCursor(population=final_cursor['population'],identity=final_cursor['identity'],
            seed=final_cursor['seed'],passes=final_cursor['passes'])
        cursor.load_state_dict(initial_cursor);cloned_ordinals=[]
        for group in range(20):
            batch=measure.call('private_cursor_preview',cursor.preview,14)
            require([d['ordinal']for d in batch]==inputs['original_ordered280_ordinals'][group*14:group*14+14],
                    'Cloned original preview EXACTLY frozen already-planned ordinals')
            cloned_ordinals.extend(d['ordinal']for d in batch)
            for expected in inputs['original_ordered_rows'][group*14:group*14+14]:
                guard();ordinal=expected['ordinal']
                row=measure.call('row_total_INCLUDES_leaf_components',reader.row,ordinal)
                wire=measure.call('canonical_row_encoding',canonical,row)
                encoded+=len(wire);require(encoded<=prefetch.MAX_ROW_BYTES,'Original8MiB metadata prefix bound')
                require(row['source_id']==expected['source_id'] and hashlib.sha256(wire).hexdigest()==expected['canonical_original_plan_row_sha256'],
                        'Every full original row/labels/geometry matches already planned oracle')
                rows[ordinal]=row
                row_observations.append({'ordinal':ordinal,'source_id':row['source_id'],
                    'canonical_row_sha256':hashlib.sha256(wire).hexdigest(),'metadata_bytes':len(wire),
                    'wall_ns_cumulative':time.perf_counter_ns()-request_wall,
                    'thread_cpu_ns_cumulative':time.thread_time_ns()-request_cpu})
                del row,wire
            measure.call('private_cursor_commit_INCLUDES_own_preview',cursor.commit,batch)
        require(cursor.state_dict()==final_cursor and cloned_ordinals==inputs['original_ordered280_ordinals'],
                'Original cloned final cursor exact; no authoritative state or fresh row selection')
        output_digest=measure.call('whole_rows_canonical_digest',digest,rows)
        request_elapsed=time.perf_counter_ns()-request_wall;request_used=time.thread_time_ns()-request_cpu
        io_after=process.io_counters()._asdict()
        require(measure.stats['catalog_labels_signature']['calls']==560 and measure.stats['geometry_signature']['calls']==280
                and measure.stats['geometry_validation']['calls']==280 and measure.stats['row_deepcopy']['calls']==560,
                'Exact original three signatures/two copies/one declaration per unique row')
        require(all(measure.stats[k+'.'+op]['calls']==280 for k in ('catalog_sql','labels_sql','geometry_sql')
                    for op in ('execute','fetchone')),'Exactly840 original SQL queries; no second pass')
        require(len(reader.cache)==32 and not reader.poisoned and len(rows)==280,'Original32 row LRU/280 unique results')
        leaf_tags=('catalog_labels_signature','geometry_signature','row_deepcopy','geometry_validation',
                   'catalog_sql.execute','catalog_sql.fetchone','labels_sql.execute','labels_sql.fetchone',
                   'geometry_sql.execute','geometry_sql.fetchone')
        totals=measure.stats['row_total_INCLUDES_leaf_components']
        status.update({'passed':True,'actual_rows':280,'original_order_and_all_full_row_digests_equal':True,
            'rows_digest':output_digest,'encoded_bytes':encoded,'request_wall_ns_INCLUDES_oracle_checks':request_elapsed,
            'request_thread_cpu_ns_INCLUDES_oracle_checks':request_used,'flat_leaf_stage_clocks':dict(measure.stats),
            'row_total_minus_leaf_wall_ns':totals['wall_ns']-sum(measure.stats[k]['wall_ns']for k in leaf_tags),
            'row_total_minus_leaf_thread_cpu_ns':totals['thread_cpu_ns']-sum(measure.stats[k]['thread_cpu_ns']for k in leaf_tags),
            'owned_process_logical_IO_before':io_before,'owned_process_logical_IO_after':io_after,
            'owned_process_logical_IO_delta':{k:io_after[k]-io_before[k]for k in io_before},
            'per_row_scalar_observations':row_observations,
            'original_method_object_identity_unchanged':True,'original_row_method_AST':method_ast(original_row),
            'original_declaration_method_AST':method_ast(original_declaration),
            'private_cursor_final_state_exact':True,'private_cursor_preview_groups':20,
            'private_cursor_original_commit_groups':20,'authoritative_sampler_or_RNG_advance':False})
    except BaseException as error:
        status.update({'passed':False,'error':repr(error),'traceback':traceback.format_exc()})
    finally:
        try:
            if reader is not None:reader.close()
        except BaseException as error:
            status.update({'passed':False,'close_error':repr(error)})
        context.signature=originals['context_signature'];geometry.signature=originals['geometry_signature']
        context.deepcopy=originals['deepcopy'];geometry.validate_declared_row=originals['validate']
        status['process_local_original_globals_restored']=all((context.signature is originals['context_signature'],
            geometry.signature is originals['geometry_signature'],context.deepcopy is originals['deepcopy'],
            geometry.validate_declared_row is originals['validate']))
        status['all3_original_SQLite_close_calls_returned']=len(proxies)==3 and all(x.closed for x in proxies)
        status['all3_denywrite_handles_closed']=reader is not None and not reader.metadata_locks.handles
        status['worker_finished']=True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ROOT-review',required=True);args=parser.parse_args()
    me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('PYTHONUTF8')=='1'
        and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
        and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),
        'Separate E/CPU-only process-local environment')
    review=load(args.ROOT_review)
    require(review['accepted']is True and review['reviewer']=='/root'
            and review['scope']=='ASR18_original280_scalar_metadata_stage_timing_CPU_v1','Exact ROOT finite diagnostic scope')
    for bound in review['input_pins']:require(pin(bound['path'])==bound,'Every compact/source input pin before execution')
    require(review['fixture_source']==pin(__file__)and Path(review['output']).resolve()==OUTPUT.resolve(),'Exact fixture/fresh output')
    require(not OUTPUT.exists()and exact_alive(EXPECTED_NATIVE),'Same healthy source29/no repeated diagnostic')
    inputs=load(INPUT);runtime=load(RUNTIME);configuration=load(CONFIG);plan=load(inputs['source_plan']['path'])
    require(pin(inputs['source_plan']['path'])==inputs['source_plan']and digest(plan)==inputs['source_plan_canonical_sha256']
            and len(inputs['original_ordered280_ordinals'])==len(set(inputs['original_ordered280_ordinals']))==280,
            'Exact already selected280 distinct rows; no generated draw')
    require(configuration['planning_arguments']['DATA_evidence']['BPE9223_population']==pin(RUNTIME),'Same admitted population metadata')
    bindings={'final_catalog':runtime['source_catalog_unchanged'],'final_labels':runtime['databases']['broad_labels'],
              'declared_geometry_index':runtime['databases']['declared_geometry']}
    signatures=[{k:b[k]for k in ('path','bytes','mtime_ns')}for b in bindings.values()]
    require([geometry.signature(v['path'])for v in bindings.values()]==signatures,'Original immutable DB signatures, no repeated fullDBhash')
    OUTPUT.mkdir();status={'kind':'ASR18_original280_scalar_metadata_firstpass_stage_timing_CPU_v1',
        'passed':False,'identity':identity(me),'launcher_identity':identity(me.parent()),'CPU_only':True,
        'actual_GPU_native_model_audio_or_authoritative_state_operations':0,'ROOT_scope':pin(args.ROOT_review),
        'fixture_source':pin(__file__),'input_pins_before':review['input_pins'],'database_bindings_existing_admission_only':bindings,
        'new_whole_DB_SHA_admission_or_storage_cache_flush_performed':False,
        'limits':['Timing wrappers add bounded measurement overhead; row totals include leaf components and must NOT be added to them.',
            'Wall minus threadCPU includes IO and descheduling/GIL waits; this diagnostic cannot separate those causes.',
            'Owned-process IO counters are logical counters, not measured physical disk cache misses.',
            'Request wall/CPU includes explicit oracle verification; row_total excludes that verification.',
            'Original private cursor is cloned/rewound by the already-planned280, its preview/commit measured and final state/ordinals checked; authoritative draws/RNG untouched. No production/native timing attribution.']}
    stop=threading.Event();worker=threading.Thread(target=run,args=(review,inputs,plan,bindings,signatures,stop,status),daemon=True)
    worker.start();worker.join(timeout=max(.01,120-(time.monotonic()-BEGIN)))
    if worker.is_alive():stop.set();worker.join(timeout=10)
    require(not worker.is_alive(),'Actual diagnostic worker joined, no promoted timeout')
    status['worker_thread_joined']=True
    status['database_signatures_after_exact']=[geometry.signature(v['path'])for v in bindings.values()]==signatures
    status['input_pins_after']=[pin(v['path'])for v in review['input_pins']]
    status['all_compact_source_inputs_unchanged']=status['input_pins_after']==review['input_pins']
    status['same_native_identity_alive']=exact_alive(EXPECTED_NATIVE)
    status['passed']=bool(status['passed']and status['worker_finished']and status['all3_original_SQLite_close_calls_returned']
        and status['all3_denywrite_handles_closed']and status['process_local_original_globals_restored']
        and status['database_signatures_after_exact']and status['all_compact_source_inputs_unchanged']and status['same_native_identity_alive'])
    status['wall_seconds']=time.monotonic()-BEGIN
    out=OUTPUT/'report.json';out.write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':pin(out),'passed':status['passed'],'rows':status.get('actual_rows'),'error':status.get('error')}),flush=True)
    require(status['passed'],'Actual bounded metadata observation failed; preserve and review, no retry')


if __name__=='__main__':main()
