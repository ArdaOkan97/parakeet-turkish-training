"""Stdlib-only seal for a corrected hit-only consumer of genuine closed seed2."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import sys
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
E=Path(r'E:\gpu_training_scratchpad');A=E/'analysis/ASR18_memory_diagnosis_v1'
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_production_state_v1 import require

def record(path):
    path=Path(path).resolve();require(path.is_relative_to(P) or path.is_relative_to(E),'Project/E only')
    before=path.stat();h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(1024**2):h.update(b)
    require((before.st_size,before.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns),'Stable binding')
    return {'path':str(path),'bytes':before.st_size,'sha256':h.hexdigest()}

def main():
    require(os.environ['CUDA_VISIBLE_DEVICES']=='-1' and read_json(E/'storage_config.json')['storage_root']==str(E),'Masked E metadata only')
    oldfolder=A/'official_TDT_cache_configuration_02';oldpath=oldfolder/'configuration_pending_ROOT_execution_review.json'
    c=deepcopy(read_json(oldpath));seed=read_json(c['seed_actual_process_proof']);g=read_json(seed['guard']['path'])
    require(seed['actual_Popen_wait_code']==0 and seed['all_captured_identities_dead'] is True and g['actual_exit_code']==0
        and g['monitor_clean'] and g['monitor_joined'],'Actual closed seed2 only')
    require(read_json(seed['report']['path'])['saved_original_loss_gradient_byte_exact'] is True,'Exact saved original seed result')
    c['seed_configuration']=record(oldpath);c['seed_parent_source']=c['source_bindings']['harness']
    c['seed_actual_process_proof_binding']=record(c['seed_actual_process_proof'])
    pins=deepcopy(c['source_bindings']);pins['harness']=record(P/'work/research/run_ASR18_official_TDT_cache_pair_v3.py')
    pins['configuration_builder']=record(__file__)
    del pins['asr18_exact_TDT_kernel_cache_v2']
    pins['asr18_exact_TDT_kernel_cache_v3']=record(S/'asr18_exact_TDT_kernel_cache_v3.py')
    for pin in pins.values():
        require(record(pin['path'])==pin,'Exact current source')
        if Path(pin['path']).suffix=='.py':compile(Path(pin['path']).read_text(encoding='utf-8-sig'),pin['path'],'exec')
    c['source_bindings']=pins;c['hit_folder']=str(A/'official_TDT_cache_hit_actual_03')
    c['scope']+=' Hit-only canonical installed-codegen CC-key correction; preserved seed2 identity and inventory, no reseed or math change.'
    folder=A/'official_TDT_cache_configuration_03';folder.mkdir(exist_ok=False)
    path=folder/'configuration_pending_ROOT_execution_review.json';write_json(path,c)
    proposal=deepcopy(read_json(oldfolder/'resource_proposal.json'))
    proposal.update(configuration=record(path),source_bindings=pins)
    write_json(folder/'resource_proposal.json',proposal)
    handoff={'configuration':record(path),'resource_proposal':record(folder/'resource_proposal.json'),
        'harness':pins['harness'],'source_bindings':pins,'execution_admitted':False,'production_enabled':False,
        'preserved_seed_configuration':record(oldpath),'preserved_seed_actual0':record(c['seed_actual_process_proof']),
        'preserved_failed_hit_log':record(A/'official_TDT_cache_hit_actual_02/official_TDT/worker.log'),
        'exact_delta':'Hit-only source-bound seed2 consumer; derive CUBIN key via actual library._ensure_cc(None), require exact hardware pair and installed suffix. Same signatures/options/CUBIN/math, compile fallback disabled.'}
    write_json(folder/'handoff.json',handoff)
    print({'configuration':record(path),'handoff':record(folder/'handoff.json'),'source_count':len(pins)})
if __name__=='__main__':main()
