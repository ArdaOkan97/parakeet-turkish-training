"""Small metadata-only packed storage sizing; no payload or current-file writes."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

PROJECT=Path(__file__).resolve().parents[2];SOURCE=PROJECT/'outputs/parakeet_turkish'
sys.path.insert(0,str(SOURCE))
import bpe_checkpoint_storage_capacity_v1 as design
from fastmss_row_serial_cpu_v5 import environment,resource_guard
from fastmss_phase_aware_feature_window_v3 import write_new

AREA=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')


def main(policy_path,output):
    environment();resource_guard();output=Path(output).resolve()
    design.require(output.is_relative_to(design.ROOT.resolve()) and not output.exists(),'Fresh E sizing report only');output.mkdir()
    policy=json.loads(policy_path.read_text());allocation=design.E_allocation_unit()
    capacity=design.capacity_bound(policy,allocation_unit_bytes=allocation['allocation_unit_bytes'])
    checks=[]
    def check(name,value):design.require(value,name);checks.append(name)
    def reject(name,function):
        try:function()
        except ValueError:checks.append(name);return
        raise AssertionError('Expected capacity rejection: '+name)
    check('Full401072 positive-update publication horizon counted, no minimum windows',capacity['maximum_publications']==401072)
    check('Hot checkpoint entry count stays5 independent of horizon',capacity['maximum_hot_checkpoint_entries']==5)
    check('No complete retired history parsed for each publication',capacity['per_publication_parsed_history_entries']==0)
    check('Current pack index is bounded1024 records',capacity['maximum_open_pack_index_entries']==1024)
    samples=[]
    for folder,step in [('feature_window_native20_01',20),('cleaner_precomputed_after20_01',21)]:
        meta=AREA/folder/('after_update'+str(step)+'.state.json')
        pointer=AREA/folder/'latest_coherent_checkpoint.json'
        before=[design.record(meta),design.record(pointer)]
        value=json.loads(meta.read_text());point=json.loads(pointer.read_text())
        check('Actual '+str(step)+' pointer larger than unsupported64KiB',65536<before[1]['bytes']<=policy['maximum_pointer_bytes'])
        check('Actual '+str(step)+' metadata fits retained2MiB candidate cap',before[0]['bytes']<=policy['maximum_metadata_bytes'])
        check('Actual '+str(step)+' metadata/pointer payload IDs agree',point['payload']==value['payload'])
        sizing=design.measure_lossless_bundle([meta,pointer],guard=resource_guard)
        check('Lossless actual '+str(step)+' JSON pair fits512KiB capped record',design.require_record_fit(sizing))
        check('Actual '+str(step)+' independent roundtrip matches original raw bytes',sizing['lossless_roundtrip_sha256_equal'])
        check('Actual '+str(step)+' inputs unchanged',[design.record(meta),design.record(pointer)]==before)
        samples.append({'step':step,'metadata':before[0],'pointer':before[1],'payload_bytes_from_metadata_only':value['payload']['bytes'],'sizing':sizing})
    for name,change in [('Boolean count',{'maximum_publications':True}),('Incomplete horizon',{'maximum_publications':400000}),
                        ('Unreviewed doubled compressed cap',{'maximum_compressed_record_bytes':1048576})]:
        bad={**policy,**change};reject(name+' rejected',lambda bad=bad:design.capacity_bound(bad,allocation_unit_bytes=allocation['allocation_unit_bytes']))
    reject('Unknown nonpower2 allocation quantum rejected',lambda:design.capacity_bound(policy,allocation_unit_bytes=513))
    reject('Over-cap candidate cannot silently drop metadata fields',lambda:design.require_record_fit({'lossless_roundtrip_sha256_equal':True,'raw_bytes':8388609,'compressed_bytes':1}))
    reject('Incompressible/oversize result remains unresolved',lambda:design.require_record_fit({'lossless_roundtrip_sha256_equal':True,'raw_bytes':1048576,'compressed_bytes':524289}))
    reject('Boolean compressed size rejected',lambda:design.require_record_fit({'lossless_roundtrip_sha256_equal':True,'raw_bytes':1,'compressed_bytes':True}))
    check('No numerical/model/audio imports',not any(k in sys.modules for k in ('numpy','torch','scipy','transformers','pyarrow','soundfile','av')))
    check('No archives, payloads or fake checkpoints created',not list(output.iterdir()))
    free=shutil.disk_usage(design.ROOT).free
    report={'kind':'ASR18_actual_small_JSON_checkpoint_storage_sizing_v1','passed':True,'check_count':len(checks),'checks':checks,
        'source':design.record(__file__),'design_source':design.record(SOURCE/'bpe_checkpoint_storage_capacity_v1.py'),
        'proposal':design.record(policy_path),'actual_E_allocation_unit':allocation,'current_E_free_bytes':free,
        'disabled_capacity_arithmetic':capacity,'actual_small_JSON_pair_measurements':samples,
        'namespace_plus100GB_bound_below_current_free_arithmetic_only':capacity['minimum_initial_free_for_namespace_plus100GB_reserve']<free,
        'other_run_growth_not_budgeted_full_fit_not_claimed':True,'full_horizon_files_or_jobs_generated':False,
        'tensor_payloads_opened':False,'existing_checkpoint_or_source_files_changed':False,'actual_archive_retirement_or_catalog_implementation_executed':False,
        'GPU_model_audio_frontend_or_CACHE_job_started':False,'production_approval_claimed':False}
    write_new(output/'report.json',report)
    print(json.dumps({'passed':True,'checks':len(checks),'actual_small_JSON_only':True}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(Path(a.policy),Path(a.output))
