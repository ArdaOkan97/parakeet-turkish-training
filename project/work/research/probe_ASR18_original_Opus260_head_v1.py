"""One real failed row: unchanged reader failure, frame clock, linear PCM."""
import argparse,gc,hashlib,json,math,os,sqlite3,struct,sys,time
from fractions import Fraction
from pathlib import Path
import psutil
P=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(P/'outputs/parakeet_turkish'),str(P/'work/research')]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require,ROOT,AV_TARGET,resample
from fastmss_owned_process_closure_v2 import identity,dead
from run_ASR18_BPE9223_full_run_owner_v19 import storage,sample_memory
import fastmss_BPE9223_cpu_common_v11 as core
import fastmss_bounded_original_audio_v1 as original
import produce_FastMSS_BPE9223_reference_CPU_v11 as reference

def main():
    a=argparse.ArgumentParser();a.add_argument('--failure',required=True);a.add_argument('--folder',required=True)
    args=a.parse_args();storage();core.environment();began=time.monotonic();io0=psutil.Process().io_counters()
    failure_pin=record(args.failure);failure=read_json(args.failure)
    require(failure['kind']=='ASR18_actual_RUN07_committed260_original_Opus_PTS_reference_failure_TOOL1_v1'
        and failure['actual_tool_exit_code']==1 and failure['preserved_production_updates']==260,
        'Exact genuine saved260 failure only')
    folder=Path(args.folder).resolve();require(folder.is_relative_to(ROOT) and folder.is_dir(),'Fresh owned E output from real observer')
    def guard():
        require(time.monotonic()-began<180,'Finite180s original-row probe')
        require(all(dead(v) for v in failure['all_captured_identities']),'All exact failed production identities stay dead')
        value=sample_memory(os.getpid(),'original_Opus260_head_probe')
        require(value['paired_sampling_within_one_second'] and value['worker_private_memory']['available']
            and value['worker_private_memory']['private_bytes']<=512000000,'Owned512MB CPU-only case')
        system=value['system_memory'];require(system['commit_fraction']<.95 and system['commit_available_bytes']>2000000000,
            'Unchanged system95/headroom guard')
        io=psutil.Process().io_counters();require(io.read_bytes+io.write_bytes-io0.read_bytes-io0.write_bytes<=16*1024**3,'Bounded16GiB IO')
        return value
    guard();config=read_json(failure['configuration']['path']);run=Path(config['run_namespace'])
    workspace=run/'checkpoints/transient/p000000014'
    protected=[failure['current_checkpoint_capture'][k] for k in ('pointer','checkpoint','pre_save_fingerprint','next_draw_packet')]
    pointer=read_json(protected[0]['path']);protected += [record(pointer['journal_prefix']['path']),record(run/'checkpoints/state.json'),record(run/'checkpoints/catalog.sqlite'),record(run/'durable/latest.json')]
    before=[record(p['path']) for p in protected];require(before==protected,'Exact saved260 metadata before any source read')
    _,request,plan,contract,validated,_=reference.load_window_request(workspace/'CPU/payload/reference_request.json','reference')
    update=plan['updates'][7];row=update['coordinator_update']['rows'][0];expected=update['rows'][0]
    require(row==failure['first_candidate_row_not_yet_reproduced'] and expected==failure['first_candidate_expected_target']
        and row['start']==0.0 and row['end']==15.76,'Exact real first candidate update7/row0, full original interval')
    np,_=core.numpy_runtime()
    reader=core.OriginalRows(validated,raw_contract=contract['raw_lifetime_review'],
        private_limit_bytes=contract['resource_review']['reference_maximum_private_bytes'],frontend_enabled=False,guard=guard)
    try:
        raw,declaration=reader.resolve(row,expected,update['phase'])
        reproduced=None
        try:
            wave=reader(row,expected,update['phase']);del wave
        except ValueError as error:
            reproduced=str(error)
        require(reproduced=='Original PTS gap/overlap unresolved; no synthesized samples','Actual unchanged reader must reproduce exact failure')
        source,job=reader.broad.verify_source(core.canonical_source_path(raw,ROOT))
        source_binding=reader.broad.final_source_identity.entries[str(source)]
    finally:reader.close()
    del reader;gc.collect();guard()
    index=ROOT/f"indexes/yodas3_tar_{raw['shard']}.sqlite"
    db=sqlite3.connect(index.as_uri()+'?mode=ro&immutable=1',uri=True)
    try:entry=db.execute('SELECT member_name,offset,bytes FROM recordings WHERE recording_id=?',(raw['source_id'],)).fetchone()
    finally:db.close()
    require(entry is not None,'Same real original compressed source extent');name,offset,size=entry
    if str(AV_TARGET) not in sys.path:sys.path.insert(0,str(AV_TARGET))
    import av,numpy as np
    require(av.__version__=='16.0.1','Same pinned PyAV decoder')
    heads={}
    for seek in (False,True):
        with original.BoundedExtent(source,offset,size,guard) as extent,av.open(extent,mode='r',format='webm') as container:
            stream=container.streams.audio[0];stream.codec_context.thread_count=1
            first=next(container.decode(stream));origin=first.pts*first.time_base
            initial={'PTS':first.pts,'time_base':str(first.time_base),'samples':first.samples,'sample_rate':first.sample_rate,
                'codec':stream.codec_context.name,'stream_time_base':str(stream.time_base),
                'extradata_hex':bytes(stream.codec_context.extradata or b'').hex()}
            if seek:
                container.seek(math.floor(origin/stream.time_base),stream=stream,backward=True,any_frame=False)
                frames=container.decode(stream)
            else:
                import itertools
                frames=itertools.chain((first,),container.decode(stream))
            cursor=0;trace=[]
            for frame in frames:
                guard();clock=(frame.pts*frame.time_base-origin)*frame.sample_rate
                trace.append({'PTS':frame.pts,'time_base':str(frame.time_base),'samples':frame.samples,
                    'native_PTS_position_exact':str(clock),'native_PTS_position_rounded':round(clock),
                    'decoded_contiguous_sample_cursor':cursor,'PTS_minus_cursor_exact':str(clock-cursor)})
                cursor+=frame.samples
                if len(trace)==8:break
            heads['seek0' if seek else 'fresh_linear']={'initial_first':initial,'first8_frames':trace}
    count=declaration['native_samples'];require(count==756480 and declaration['native_sr']==48000,'Actual full declared native geometry')
    # Independent linear decoder ordering uses every returned sample exactly
    # once from the first decoded sample, without a seek or PTS placement.
    native=np.empty(count,dtype=np.float32);written=0;decoded=0;residuals=[]
    with original.BoundedExtent(source,offset,size,guard) as extent,av.open(extent,mode='r',format='webm') as container:
        stream=container.streams.audio[0];stream.codec_context.thread_count=1;origin=None
        for frame in container.decode(stream):
            guard();decoded+=1
            require(frame.pts is not None and frame.sample_rate==48000 and frame.format.is_planar
                and len(frame.layout.channels)<=contract['raw_lifetime_review']['maximum_runtime_channels']
                and frame.samples<=contract['raw_lifetime_review']['maximum_decoder_frame_samples'],'Existing frame workspace bounds')
            if origin is None:origin=frame.pts*frame.time_base
            exact=(frame.pts*frame.time_base-origin)*48000
            residuals.append(exact-written)
            values=frame.to_ndarray()
            if np.issubdtype(values.dtype,np.integer):values=values.astype(np.float32)/float(2**(np.iinfo(values.dtype).bits-1))
            else:values=values.astype(np.float32,copy=False)
            values=values.mean(axis=0,dtype=np.float32) if values.shape[0]>1 else values[0]
            copied=min(len(values),count-written);native[written:written+copied]=values[:copied];written+=copied
            del values
            if written==count:break
    require(written==count,'Full linear native interval, no synthesized sample or missing source')
    clean=resample(native,48000);require(len(clean)==declaration['clean16k_samples']==252160,'Full unchanged resampling count')
    f32,i16=core.chunked_hashes(clean,guard=guard)
    require(i16 not in set(core.read_bound(validated['bindings']['heldout_PCM'],guard)['PCM_i16_sha256']),'Real heldout PCM check')
    require(core.targets_sha(expected['full_target_ids'])==expected['target_ids_u32_sha256'] and len(expected['full_target_ids'])==72,
        'Exact unchanged full72 targets')
    geometry=core.geometry(len(clean));require(geometry['encoder_frames']==197,'Exact original full encoder197 geometry')
    require([record(p['path']) for p in protected]==before,'Saved260/journal/state/durable authoritative pins untouched')
    result=dict(kind='ASR18_actual_original_Opus260_head_and_linear_PCM_probe_v1',passed=True,producer=identity(),
        worker_source=record(__file__),parent_source=record(P/'work/research/run_ASR18_BPE9223_full_run_owner_v6.py'),
        failure=failure_pin,original_reader_source=record(original.__file__),actual_unchanged_failure_reproduced=reproduced,
        exact_failed_source_row=raw,exact_source_identity_entry=source_binding,compressed_member={'name':name,'offset':offset,'bytes':size},
        declared_geometry=declaration,head_frames=heads,linear_native_samples=written,linear_decoded_frames=decoded,
        linear_PTS_minus_sample_cursor_min=str(min(residuals)),linear_PTS_minus_sample_cursor_max=str(max(residuals)),
        linear_clean_samples=len(clean),linear_PCM_f32_sha256=f32,linear_PCM_i16_sha256=i16,full_target_tokens=72,
        target_ids_u32_sha256=expected['target_ids_u32_sha256'],full_frontend_geometry=geometry,
        actual_original_reader_fix_claimed=False,audio_samples_synthesized_or_skipped=False,
        independent_linear_read_uses_decoder_sample_order=True,PCM_or_member_written_to_disk=False,
        authoritative_saved260_metadata_unchanged=before,wall_seconds=time.monotonic()-began)
    write_json(folder/'report.json',result);print(json.dumps({'passed':True,'report':record(folder/'report.json')}),flush=True)

if __name__=='__main__':main()
