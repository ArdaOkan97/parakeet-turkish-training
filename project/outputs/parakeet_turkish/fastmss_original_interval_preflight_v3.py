"""INACTIVE readonly interval coverage records; never repair/filter a source row.

The observation adapter runs only in a separately admitted serial CPU process.
It traces scalar locals at the UNCHANGED decoder's final strict coverage guard.
An uncovered interval is blocked, never shortened, padded, skipped or certified.
No training/queue/plan integration and no corpus-scan entry point are provided.
"""
import hashlib,inspect,json,sys,threading
from pathlib import Path

KIND='ASR18_readonly_original_interval_preflight_v1'
MAX_RECORD_BYTES=65536
MAX_BATCH_ROWS=16
E_ROOT=Path(r'E:\gpu_training_scratchpad')

def require(value,message):
    if not value:raise ValueError(message)

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def record(path):
    path=Path(path);data=path.read_bytes()
    return {'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}

def specification(row,declaration,full_target_ids,*,source,member,decoder_source,runtime_authority):
    require(row['dataset']=='alania_yodas3' and len(full_target_ids)==declaration['target_tokens'],
        'Explicit original YODAS interval and every original target ID')
    require(row['ordinal']==declaration['ordinal'] and 0<=row['start']<row['end'] and
        declaration['native_sr']==48000 and type(declaration['native_samples'])is int and declaration['native_samples']>0,
        'Positive original declaration; no adjusted interval')
    require(Path(member['member_name']).stem==row['source_id'] and member['member_name'].endswith('.webm')
        and type(member['offset'])is int and member['offset']>=0 and type(member['bytes'])is int
        and 0<member['bytes'] and member['offset']+member['bytes']<=source['bytes'],
        'Exact bounded compressed source member identity')
    require(set(source)=={'path','bytes','mtime_ns','recorded_sha256'} and len(source['recorded_sha256'])==64
        and set(decoder_source)=={'path','bytes','sha256'},'Exact original source and decoder version binding')
    require(set(runtime_authority)=={'path','bytes','sha256'} and record(runtime_authority['path'])==runtime_authority,
        'Exact immutable CPU numerical/runtime authority; changed runtime invalidates cached geometry')
    value={'kind':KIND,'row':row,'declaration':declaration,'full_target_ids':full_target_ids,
        'source':source,'member':member,'decoder_source':decoder_source,'runtime_authority':runtime_authority}
    # Round-trip detaches all caller-owned dictionaries without retaining mutable rows.
    return json.loads(canonical(value))

def classify(spec,actual,*,evidence):
    require(spec['kind']==KIND,'Exact preflight specification')
    require(actual['source']==spec['source'] and actual['member']==spec['member']
        and actual['decoder_source']==spec['decoder_source'],'Observation belongs to exact source/member/decoder')
    written,count=actual['written'],actual['count']
    require(type(written)is int and type(count)is int and 0<=written<=count
        and count==spec['declaration']['native_samples'],'Truthful observed coverage under original reservation')
    require(type(actual['original_decoder_returned'])is bool and type(actual['strict_guards_passed'])is bool,
        'Explicit original decoder result; no certificate assumption')
    covered=(written==count and actual['original_decoder_returned'] and actual['strict_guards_passed'])
    value={'kind':KIND,'key':digest(spec),'specification':spec,'actual':actual,
        'status':'covered'if covered else'blocked_unresolved_original_interval',
        'native_sample_shortfall':count-written,'evidence':evidence,
        'no_source_or_annotation_change':True,'no_geometry_correction_authorized':True,
        'no_synthesized_cropped_filtered_or_skipped_samples':True}
    require(len(canonical(value))<=MAX_RECORD_BYTES,'Bounded one-row scalar preflight record')
    return value

def require_covered(records):
    require(isinstance(records,list) and 0<len(records)<=MAX_BATCH_ROWS,'Explicit finite batch, at most16 rows; never implicit corpus scan')
    for item in records:
        require(item['key']==digest(item['specification']) and item['status']=='covered'
            and item['actual']['written']==item['actual']['count'] and item['actual']['original_decoder_returned']
            and item['actual']['strict_guards_passed'],'Blocked original interval: stop before GPU consumption; never remove row')
    return True

def store_record(output_directory,item):
    root=Path(output_directory).resolve()
    require(root.is_relative_to(E_ROOT.resolve()) and root.is_dir(),'Explicit existing owned E output directory')
    require(item['key']==digest(item['specification']),'Immutable exact row/source/member/decoder key')
    data=canonical(item);require(len(data)<=MAX_RECORD_BYTES,'Bounded metadata only')
    path=root/(item['key']+'.json')
    with path.open('xb')as stream:stream.write(data)
    return record(path)

def read_record(pin,spec):
    path=Path(pin['path']).resolve()
    require(path.is_relative_to(E_ROOT.resolve()) and record(path)==pin and pin['bytes']<=MAX_RECORD_BYTES,
        'Exact bounded immutable E preflight record')
    value=json.loads(path.read_bytes())
    require(value['kind']==KIND and value['key']==digest(spec) and value['specification']==spec,
        'Source/member/annotation/labels/declaration/decoder changes invalidate cached observation')
    return value

def observe_unchanged_yodas(reader,row,declaration,full_target_ids,*,source,member,decoder_source,runtime_authority,guard):
    """Future CPU-only adapter; no decoder changes and no model/audio-file writes.

    Caller must own the original RawLifetimeLedger decode_resample scope, active
    row reservation, source authority/read locks and complete resource admission.
    Metadata tracing restores the previous trace on every exit. No training
    process may use this adapter, and partial buffers never leave the callback.
    """
    require(threading.current_thread()is threading.main_thread() and sys.gettrace()is None,
        'Separate serial CPU preflight with no existing tracing hook')
    function=reader.yodas;method=function.__func__
    require(record(inspect.getsourcefile(method))==decoder_source and
        method.__module__=='fastmss_bounded_original_audio_v2' and reader.active['declaration']==declaration,
        'Exact unchanged ordinary decoder and caller-owned full raw allocation scope')
    spec=specification(row,declaration,full_target_ids,source=source,member=member,decoder_source=decoder_source,runtime_authority=runtime_authority)
    lines,first=inspect.getsourcelines(method)
    matches=[first+i for i,line in enumerate(lines)if "require(written==count,'Original interval beyond source unresolved')"in line]
    require(len(matches)==1,'Exact source-bound original strict coverage guard')
    observed={};returned=False;error=None
    def trace(frame,event,arg):
        if frame.f_code is not method.__code__:return None
        if event=='line' and frame.f_lineno==matches[0]:
            guard();v=frame.f_locals;path=v['path'];stat=path.stat()
            actual_source={'path':str(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'recorded_sha256':v['job']['sha256']}
            actual_member={'member_name':v['name'],'offset':v['offset'],'bytes':v['size']}
            written=v['written'];pcm=v['pcm'];h=hashlib.sha256()
            # Hash only real written samples in bounded views; the unused reserved
            # tail is never exposed, resampled, serialized or used as audio.
            for start in range(0,written,65536):guard();h.update(memoryview(pcm[start:min(start+65536,written)]).cast('B'))
            observed.update(source=actual_source,member=actual_member,decoder_source=decoder_source,
                written=written,count=v['count'],native_f32_sha256=h.hexdigest(),decoded_frames=v['decoded'],
                original_PTS_begin=v['begin'],original_PTS_declared_finish=v['finish'],origin=str(v['origin']))
        return trace
    guard();sys.settrace(trace)
    try:
        wave,info=function(row);returned=True;del wave,info
    except ValueError as caught:error=str(caught)
    finally:sys.settrace(None)
    require(observed,'Source guard failed before final coverage observation; keep failure, never admit row')
    observed.update(original_decoder_returned=returned,strict_guards_passed=returned,
        original_error=error,unused_reserved_samples_never_read=True)
    return classify(spec,observed,evidence={'kind':'actual_separate_CPU_unchanged_decoder_observation',
        'preflight_source':record(__file__),'training_or_GPU_consumption':False})
