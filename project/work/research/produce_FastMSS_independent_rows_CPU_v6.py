"""Independent whole-clip references: serial original decode, metadata only."""
import argparse,gc,json,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
import fastmss_row_serial_cpu_v6 as core

def main(request_path,output):
    core.environment();guard=core.resource_guard
    request,plan,update,contract,validated=core.load_request(core.record(request_path,guard),guard)
    guard=core.stage_guard(contract,'reference')
    admission=core.admit(contract['resource_review']['reference_incremental_bytes'],guard)
    np,_=core.numpy_runtime();reader=core.OriginalRows(validated,raw_contract=contract['raw_lifetime_review'],
        private_limit_bytes=contract['resource_review']['reference_maximum_private_bytes'],frontend_enabled=False,guard=guard);refs=[]
    try:
        for raw,expected in zip(update['coordinator_update']['rows'],update['rows']):
            guard();wave=reader(raw,expected,update['phase'])
            refs.append(core.reference_entry(raw,expected,reader));del wave;gc.collect();reader.finish_raw_row()
    finally:reader.close()
    core.require(len(refs)==len(update['rows']),'All original rows, no exclusion or shortened batch')
    core.require(not any(k=='torch' or k.startswith(('torch.','transformers','pyarrow')) for k in sys.modules),'No Torch/frontend/Arrow in independent reference child')
    result={'kind':'FastMSS_phase_aware_independent_clean_PCM_references_v1','passed':True,
        'bridge':request['bridge'],'update_index':request['update_index'],'phase':update['phase'],
        'references':refs,'producer':core.identity(),'worker_source':core.record(__file__,guard),
        'parent_source':contract['sources']['controller'],'request':core.record(request_path,guard),
        'actual_optimizer_updates':0,'sampler_journal_frontier_advanced':False,
        'original_decoder_source':contract['sources']['broad_reader'],'NumPy':np.__version__,
        'raw_PCM_files_written':0,'simultaneous_raw_jobs':1,'serial_waveforms_not_retained':True,
        'actual_child_exit_observed_in_child':False,'CPU_stage_admission':admission}
    result['raw_lifetime_receipt']=reader.raw_receipt()
    core.write_new(core.scoped(output),result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.request,a.output)
