"""Actual complete failed row through decoder2; independent PCM and strict clock."""
import argparse,ast,gc,json,os,sys,time
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import psutil
P=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(P/'outputs/parakeet_turkish'),str(P/'work/research')]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require,ROOT
from fastmss_owned_process_closure_v2 import identity,dead
from run_ASR18_BPE9223_full_run_owner_v19 import storage,sample_memory
from fastmss_raw_lifetime_v1 import RawLifetimeLedger,raw_plan,chunked_hashes
from bpe_runtime_declared_geometry_v1 import verify_actual_row
import fastmss_BPE9223_cpu_common_v11 as core
import fastmss_bounded_original_audio_v2 as decoder
import produce_FastMSS_BPE9223_reference_CPU_v11 as reference

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--probe',required=True);parser.add_argument('--folder',required=True)
    args=parser.parse_args();storage();core.environment();began=time.monotonic();io0=psutil.Process().io_counters()
    probe_pin=record(args.probe);probe=read_json(args.probe);failure=read_json(probe['failure']['path'])
    require(probe['passed'] is True and probe['actual_unchanged_failure_reproduced']=='Original PTS gap/overlap unresolved; no synthesized samples','Real unchanged failure and independent full linear PCM required')
    folder=Path(args.folder).resolve();require(folder.is_relative_to(ROOT) and folder.is_dir(),'Actual owned E output')
    def guard():
        require(time.monotonic()-began<180,'Finite180s actual decoder proof')
        require(all(dead(v) for v in failure['all_captured_identities']),'Failed production stays quiescent')
        value=sample_memory(os.getpid(),'actual_original_Opus260_decoder2')
        require(value['paired_sampling_within_one_second'] and value['worker_private_memory']['private_bytes']<=512000000,'Fresh exact512MB owned CPU-only bound')
        require(value['system_memory']['commit_fraction']<.95 and value['system_memory']['commit_available_bytes']>2000000000,'Same system95 and2GB headroom')
        io=psutil.Process().io_counters();require(io.read_bytes+io.write_bytes-io0.read_bytes-io0.write_bytes<=16*1024**3,'Finite16GiB IO')
        return value
    guard();protected=probe['authoritative_saved260_metadata_unchanged'];require([record(v['path']) for v in protected]==protected,'Exact saved260 state before proof')
    config=read_json(failure['configuration']['path']);workspace=Path(config['run_namespace'])/'checkpoints/transient/p000000014'
    _,request,plan,contract,validated,_=reference.load_window_request(workspace/'CPU/payload/reference_request.json','reference')
    update=plan['updates'][7];row=update['coordinator_update']['rows'][0];expected=update['rows'][0]
    np,_=core.numpy_runtime();metadata=core.OriginalRows(validated,raw_contract=contract['raw_lifetime_review'],private_limit_bytes=contract['resource_review']['reference_maximum_private_bytes'],frontend_enabled=False,guard=guard)
    try:
        raw,declaration=metadata.resolve(row,expected,update['phase']);heldout=metadata.heldout
        require(raw==probe['exact_failed_source_row'] and declaration==probe['declared_geometry'],'Exact same original full source row and declaration')
    finally:metadata.close()
    del metadata;gc.collect();review=contract['raw_lifetime_review']
    reservation=raw_plan(native_samples=declaration['native_samples'],native_rate=declaration['native_sr'],clean_samples=declaration['clean16k_samples'],phase=update['phase'],role=expected['role'],channels_bound=review['maximum_runtime_channels'],frame_samples_bound=review['maximum_decoder_frame_samples'])
    require(reservation['workspace_incremental_envelopes']['decoder']<=review['decoder_workspace_allowance_bytes'],'Original decoder workspace bound')
    prospective=reservation['raw_peak_bytes']+reservation['compressed_media_cap_bytes']+reservation['workspace_incremental_envelopes']['decoder']+review['numerical_runtime_reserve_bytes']
    require(guard()['worker_private_memory']['private_bytes']+prospective<=512000000,'Whole child plus full preallocation fits unchanged512MB')
    core.admit(prospective,guard);ledger=RawLifetimeLedger(guard=guard);key='actual_saved260_full_failed_row';ledger.begin(key,reservation)
    reader=decoder.BoundedOriginalReader(ledger=ledger,source_inventory_binding=validated['bindings']['source_identity_inventory'],guard=guard,heldout_PCM_hashes=heldout)
    wave=reader.read_original(raw,job_key=key,declaration=declaration)
    actual=reader.last;geometry=verify_actual_row(declaration,raw,actual,full_target_tokens=len(expected['full_target_ids']))
    f32,i16=chunked_hashes(wave,guard=guard)
    require(f32==probe['linear_PCM_f32_sha256'] and i16==probe['linear_PCM_i16_sha256'],'Actual decoder2 equals independent full linear PCM sample-for-sample hashes')
    require(actual['initial_decoded_frame_retained'] is True and actual['linear_reference'] is True and actual['PTS_quantization_half_tick_samples']=='24','Actual repaired recording-origin branch exercised')
    require(len(wave)==252160 and core.geometry(len(wave))==probe['full_frontend_geometry'] and core.targets_sha(expected['full_target_ids'])==probe['target_ids_u32_sha256'],'Exact original full geometry and72 targets')
    # Execute the exact immutable decoder guard expression with scalar clocks.
    # These are negative guard controls, not fabricated source audio or frames.
    tree=ast.parse(Path(decoder.__file__).read_text(encoding='utf-8'))
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and len(node.args)==2 and isinstance(node.args[1],ast.Constant) and node.args[1].value=='Original Opus PTS exceeds exact half-tick quantization; no synthesized samples']
    require(len(calls)==1,'Exact production clock guard source binding')
    expression=compile(ast.Expression(calls[0].args[0]),str(decoder.__file__),'eval');clock_controls=[]
    for pts,accepted in ((14,True),(15,False),(12,False),(34,False)):
        passed=bool(eval(expression,{'abs':abs},{'frame':SimpleNamespace(pts=pts,time_base=Fraction(1,1000)),'stream':SimpleNamespace(time_base=Fraction(1,1000)),'origin':Fraction(0),'sr':48000,'sample_cursor':648}))
        require(passed==accepted,'Exact source clock guard must reject true gap/overlap, preserve half-tick quantization')
        clock_controls.append({'PTS_milliseconds':pts,'expected_accepted':accepted,'actual_accepted':passed,'cursor':648,'half_tick_native_samples':24})
    require([record(v['path']) for v in protected]==protected,'Saved260 pointer/PRE/journal/catalog/durable unchanged')
    result=dict(kind='ASR18_actual_original_Opus260_decoder2_full_PCM_equality_and_clock_guard_v1',passed=True,producer=identity(),worker_source=record(__file__),parent_source=record(P/'work/research/run_ASR18_BPE9223_full_run_owner_v6.py'),decoder_source=record(decoder.__file__),independent_actual_probe=probe_pin,exact_source_identity=reader.final_source_identity.entries[str(core.canonical_source_path(raw,ROOT))],full_decoded_row=raw,actual_decode=actual,actual_geometry_proof=geometry,full_frontend_geometry=core.geometry(len(wave)),full_targets=72,target_ids_u32_sha256=core.targets_sha(expected['full_target_ids']),independent_full_PCM_f32_sha256=f32,independent_full_PCM_i16_sha256=i16,exact_PCM_hashes_equal=True,source_clock_guard_scalar_controls=clock_controls,negative_controls_are_source_expression_only=True,no_audio_filled_skipped_or_synthesized=True,no_PCM_or_member_file_written=True,no_training_or_optimizer_draws=True,authoritative_saved260_pins_unchanged=protected,wall_seconds=time.monotonic()-began)
    write_json(folder/'report.json',result);print(json.dumps({'passed':True,'report':record(folder/'report.json')}),flush=True)

if __name__=='__main__':main()
