"""Prepare one exact terminal-interval certificate and inactive source, no execution."""
import ast,difflib,hashlib,json
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';N=Path(__file__).parent
pin=lambda p:{'path':str(Path(p).resolve()),'bytes':Path(p).stat().st_size,'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
r=json.loads((N/'full_member_actual_01/report.json').read_text());u=json.loads((N/'probe_actual_01/actual_failed_update_metadata.json').read_text());row=u['rows'][13]
c={'kind':'ASR18_source_bound_one_original_terminal_interval_geometry_certificate_v1','original_row':r['row'],'original_row_sha256':digest(r['row']),'original_declaration':r['declared_geometry'],'full_target_ids':row['tr_metadata_labels'],'full_target_ids_sha256':digest(row['tr_metadata_labels']),'raw_label_sha256':hashlib.sha256(r['row']['text'].encode()).hexdigest(),'original_TAR':r['original_TAR'],'recording_index_entry':r['recording_index_entry'],'member_sha256':r['member_sha256'],'actual_geometry':r['actual_interval_geometry'],'decoded_frames':657,'independent_full_member_decoded_frames':r['decoded_frames'],'max_abs_PTS_residual_samples':r['max_abs_PTS_residual_samples'],'native_f32_sha256':r['native_interval_f32_sha256'],'clean_f32_sha256':r['clean_f32_sha256'],'clean_i16_sha256':r['clean_i16_sha256'],'original_PTS_interval_begin':r['original_PTS_interval_begin'],'original_PTS_interval_declared_finish':r['original_PTS_interval_declared_finish'],'actual_PTS_interval_finish':r['actual_PTS_interval_finish'],'selected_physical_source_range':r['selected_physical_source_range'],'full_source_native_samples':r['full_source_native_samples'],'full_source_native_f32_sha256':r['full_source_native_f32_sha256'],'independent_full_member_observation':pin(N/'full_member_actual_01/report.json'),'actual_failure':pin(N/'actual_failure_closure.json'),'ordered_failure_localization':pin(N/'probe_actual_01/report.json'),'original_annotations_indexes_and_plans_unchanged':True,'all_original_selected_interval_physical_samples_once':True,'sampler_order_full_pass_denominator_labels_unchanged':True,'no_generic_EOF_tolerance_or_clamp':True,'historical_mel_frames':750,'actual_physical_mel_frames':750,'encoder_frames_unchanged':94}
certificate=N/'fourth_original_terminal_interval_geometry_certificate_v1.json';assert not certificate.exists();certificate.write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
oldpath=S/'fastmss_original_clip_geometry_overlay_v3.py';old=oldpath.read_text();tree=ast.parse(old);lines=old.splitlines(True);cls=next(x for x in tree.body if isinstance(x,ast.ClassDef));class_body=''.join(lines[cls.lineno-1:cls.end_lineno])
class_body=class_body.replace('self.third_overlay_policy','self.fourth_overlay_policy').replace('self.third_certificate','self.fourth_certificate').replace('decode_full_member(self.broad','decode_terminal_interval(self.broad').replace('its two prior certified rows','its three prior certified rows').replace('all_original_physical_samples_used_once=True','all_original_selected_interval_physical_samples_used_once=True').replace('ASR18_exact_original_full_clip_geometry_reconciliation_v1','ASR18_exact_original_terminal_interval_geometry_reconciliation_v1')
header='''"""One exact original nonzero terminal interval EOF, prior and ordinary paths unchanged.

Keep original annotations, larger native reservation, PTS selection/preroll and all
real interval samples. Only this certified one-sample declaration mismatch is used.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib,math,sqlite3,sys
import fastmss_original_clip_geometry_overlay_v3 as prior
from fastmss_original_clip_geometry_overlay_v3 import core,BoundedOriginalReader,raw_plan,chunked_hashes,canonical_source_path,source_identity_binding
from bpe_broad_audio_v1 import AV_TARGET,Extent,resample
ROW_ID='alania:e4a162635e1cf1f6599a40b38b8b7a57_00004'
ORDINAL=459954
require,digest,same,record=core.require,core.digest,core.same,core.record
match_original=prior.match_original

def checked_certificate(policy,guard):
    require(policy['source']==record(__file__,guard),'Exact physical interval overlay source')
    c=core.read_bound(policy['certificate'],guard);review=core.read_bound(policy['ROOT_review'],guard)
    require(review['accepted'] is True and review['reviewer']=='/root'
        and review['scope']=='ASR18_source_bound_fourth_original_terminal_interval_geometry_overlay_v1'
        and review['overlay_source']==policy['source'] and review['certificate']==policy['certificate'],
        'Exact ROOT reviewed nonzero terminal interval')
    require(c['kind']=='ASR18_source_bound_one_original_terminal_interval_geometry_certificate_v1'
        and c['original_row']['ordinal']==ORDINAL and c['original_row']['unique_id']==ROW_ID
        and c['original_row']['start']==45.648 and c['original_row']['end']==53.151
        and digest(c['original_row'])==c['original_row_sha256']
        and digest(c['full_target_ids'])==c['full_target_ids_sha256'] and len(c['full_target_ids'])==39,
        'Exact original nonzero annotation and full39 labels')
    require(c['original_declaration']==dict(ordinal=ORDINAL,source_component=0,native_sr=48000,
        native_samples=360144,clean16k_samples=120048,mel_frames=750,encoder_frames=94,target_tokens=39,status='declared')
        and c['actual_geometry']==dict(native_sr=48000,native_samples=360143,clean16k_samples=120048,
        mel_frames=750,encoder_frames=94,target_tokens=39) and c['decoded_frames']==657
        and c['original_PTS_interval_begin']==2191104 and c['original_PTS_interval_declared_finish']==2551248
        and c['actual_PTS_interval_finish']==2551247 and c['selected_physical_source_range']==[2191080,2551223]
        and c['full_source_native_samples']==2551223 and c['independent_full_member_decoded_frames']==2658
        and c['max_abs_PTS_residual_samples']=='24','Exact source-bound EOF and unchanged clean/T/U')
    observed=core.read_bound(c['independent_full_member_observation'],guard)
    require(observed['full_original_frames_observed_once'] is True
        and observed['selected_interval_original_samples_used_once'] is True
        and observed['full_source_range_equals_original_PTS_selected_PCM'] is True
        and observed['synthesized_samples']==0 and observed['selected_interval_skipped_samples']==0
        and observed['actual_interval_geometry']==c['actual_geometry']
        and observed['native_interval_f32_sha256']==c['native_f32_sha256']
        and all(observed[k]==c[k]for k in ('member_sha256','clean_f32_sha256','clean_i16_sha256',
            'recording_index_entry','original_TAR','original_PTS_interval_begin',
            'original_PTS_interval_declared_finish','actual_PTS_interval_finish',
            'selected_physical_source_range','full_source_native_samples','full_source_native_f32_sha256')),
        'Independent full-member and original PTS-selected interval PCM evidence')
    return c

def decode_terminal_interval(reader,row,certificate,*,guard,channels_bound,frame_samples_bound):
    """Original two-second preroll/PTS coordinates; exact certified EOF and PCM only."""
    import numpy as np
    if str(AV_TARGET)not in sys.path:sys.path.insert(0,str(AV_TARGET))
    import av
    require(av.__version__=='16.0.1','Pinned original PyAV')
    path,job=reader.verify_source(canonical_source_path(row,core.ROOT));original=certificate['original_TAR'];stat=path.stat()
    require(str(path)==original['path'] and stat.st_size==original['bytes']
        and stat.st_mtime_ns==original['mtime_ns'] and job['sha256']==original['recorded_sha256'],
        'Exact original source signature')
    index=core.ROOT/('indexes/yodas3_tar_'+row['shard']+'.sqlite');db=sqlite3.connect(index.as_uri()+'?mode=ro&immutable=1',uri=True)
    try:entry=db.execute('SELECT member_name,offset,bytes FROM recordings WHERE recording_id=?',(row['source_id'],)).fetchone()
    finally:db.close()
    require(entry is not None and dict(zip(('member_name','offset','bytes'),entry))==certificate['recording_index_entry'],
        'Exact original member, no replacement')
    name,offset,size=entry;sha=hashlib.sha256()
    with path.open('rb')as source:
        source.seek(offset);left=size
        while left:
            guard();block=source.read(min(1024**2,left));require(block,'Full original member');sha.update(block);left-=len(block)
    require(sha.hexdigest()==certificate['member_sha256'],'Exact original compressed bytes')
    start,end=Fraction(str(row['start'])),Fraction(str(row['end']));begin=round(start*48000);finish=round(end*48000)
    count=finish-begin;require(0<start<end and count==certificate['original_declaration']['native_samples']
        and begin==certificate['original_PTS_interval_begin'] and finish==certificate['original_PTS_interval_declared_finish'],
        'Unchanged original nonzero interval and larger raw allocation')
    with Extent(path,offset,size,guard)as extent,av.open(extent,mode='r',format='webm')as container:
        require(len(container.streams.audio)==1,'One original stream');stream=container.streams.audio[0];stream.codec_context.thread_count=1
        require(stream.codec_context.name=='opus' and stream.time_base==Fraction(1,1000),'Exact certified Opus/ms clock')
        first=next(container.decode(stream));channels=len(first.layout.channels);origin=first.pts*first.time_base
        require(first.pts is not None and origin==0 and first.sample_rate==48000 and 1<=channels<=channels_bound
            and first.format.is_planar and first.format.bytes<=8 and 0<first.samples<=frame_samples_bound,
            'Original first frame allocation envelope')
        guard();pcm=np.empty(count,dtype=np.float32);written=decoded=0;first_selected_pts=None
        target=origin+max(Fraction(0),start-2);container.seek(math.floor(target/stream.time_base),stream=stream,backward=True,any_frame=False)
        for frame in container.decode(stream):
            guard();decoded+=1
            require(frame.pts is not None and frame.sample_rate==48000 and len(frame.layout.channels)==channels
                and frame.format.is_planar and frame.format.bytes<=8 and 0<frame.samples<=frame_samples_bound,
                'Every original frame before ndarray allocation')
            frame_begin=round((frame.pts*frame.time_base-origin)*48000);frame_end=frame_begin+frame.samples
            if frame_end<=begin:continue
            require(frame_begin<finish,'No original frames beyond exact certified terminal interval')
            raw=frame.to_ndarray();require(raw.shape==(channels,frame.samples),'Original planar shape')
            if np.issubdtype(raw.dtype,np.integer):raw=raw.astype(np.float32)/float(2**(np.iinfo(raw.dtype).bits-1))
            else:raw=raw.astype(np.float32,copy=False)
            raw=raw.mean(axis=0,dtype=np.float32)if channels>1 else raw[0]
            low,high=max(begin,frame_begin),min(finish,frame_end)
            require(low-begin==written,'Original exact PTS gap/overlap guard, no filler')
            pcm[low-begin:high-begin]=raw[low-frame_begin:high-frame_begin];written=high-begin
            if first_selected_pts is None:first_selected_pts=str(frame.pts*frame.time_base)
            del raw
        require(decoded==certificate['decoded_frames'] and written==certificate['actual_geometry']['native_samples']
            and frame_end==certificate['actual_PTS_interval_finish'] and frame.samples==815,
            'Exact certified actual EOF, no generic clamp or tolerance')
        logical=extent.logical_bytes_read
    actual=pcm[:written]
    require(hashlib.sha256(memoryview(actual).cast('B')).hexdigest()==certificate['native_f32_sha256'],
        'Every actual original PTS-selected sample exact')
    output=resample(actual,48000);f32,i16=chunked_hashes(output,guard=guard)
    require(len(output)==certificate['actual_geometry']['clean16k_samples'] and f32==certificate['clean_f32_sha256']
        and i16==certificate['clean_i16_sha256'],'Exact full original interval resampling and PCM hashes')
    return output,dict(format='YODAS_Opus_WebM_TAR_extent',source_file_sha256=job['sha256'],member_name=name,
        member_offset=offset,member_bytes=size,member_logical_read_bytes=logical,native_sample_rate=48000,
        native_samples=written,actual_native_channels=channels,decoded_frames=decoded,PTS_origin=str(origin),
        source_interval_seconds=[str(start),str(end)],first_selected_frame_PTS=first_selected_pts,
        full_physical_recording_used=False,full_physical_selected_interval_used=True,
        original_annotation_end_preserved=True,original_derived_native_samples=count,linear_reference=False,
        codec_preroll_seconds=2,initial_decoded_frame_retained=False,temporary_member_or_PCM_written=False,
        PyAV_version=av.__version__)

'''
new=header+class_body+'\n';path=S/'fastmss_original_clip_geometry_overlay_v4.py';assert not path.exists();ast.parse(new);path.write_text(new,encoding='utf-8');delta=N/'overlay3_to4.diff.txt';delta.write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(oldpath),tofile=str(path))),encoding='utf-8');out={'overlay_source':pin(path),'overlay_delta':pin(delta),'certificate':pin(certificate),'actual_failure':pin(N/'actual_failure_closure.json'),'ordered_probe_closure':pin(N/'actual_ordered_original_CPU_probe_closure_v1.json'),'independent_member_report':pin(N/'full_member_actual_01/report.json'),'independent_member_CPU_closure':pin(N/'actual_independent_terminal_member_CPU_closure_v1.json'),'scope':'Inactive one nonzero terminal-interval original PTS EOF certificate only; no frontend or GPU admission'};handoff=N/'fourth_terminal_overlay_inactive_handoff_v1.json';handoff.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps({'handoff':pin(handoff),**out}),flush=True)