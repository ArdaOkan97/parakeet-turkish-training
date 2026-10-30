"""Bounded stdlib configuration seal for one inactive tiny official-TDT probe."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
E=Path(r'E:\gpu_training_scratchpad');A=E/'analysis/ASR18_memory_diagnosis_v1'
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_production_state_v1 import require
def record(path):
    path=Path(path).resolve();require(path.is_relative_to(P) or path.is_relative_to(E),'Existing project/E only')
    h=hashlib.sha256();before=path.stat()
    with path.open('rb') as f:
        while b:=f.read(1024**2):h.update(b)
    require((before.st_size,before.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns),'Stable pin')
    return {'path':str(path),'bytes':before.st_size,'sha256':h.hexdigest()}
def main():
    require(read_json(E/'storage_config.json')['storage_root']==str(E) and os.environ['CUDA_VISIBLE_DEVICES']=='-1','E-only CUDA-masked metadata seal')
    folder=A/'official_TDT_cache_configuration_01';folder.mkdir(exist_ok=False)
    harness=P/'work/research/run_ASR18_official_TDT_cache_pair_v1.py';helper=P/'work/research/run_ASR18_frozen_CNN_lifetime_pair_v8.py'
    files={'harness':harness,'telemetry_helper':helper,'configuration_builder':Path(__file__),
        'official_minimal_manifest':S/'native_tdt_minimal_import_manifest_v1.json'}
    queue=[harness,helper]
    while queue:
        f=queue.pop();tree=ast.parse(f.read_text(encoding='utf-8-sig'))
        for n in ast.walk(tree):
            names=([n.module] if isinstance(n,ast.ImportFrom) and n.module else [v.name for v in n.names] if isinstance(n,ast.Import) else [])
            for name in names:
                path=S/(name.split('.')[0]+'.py')
                if path.exists() and path not in files.values():files[name.split('.')[0]]=path;queue.append(path)
    official=P/'work/nemo-native-env-v1/Lib/site-packages/nemo'
    for relative,sha in read_json(S/'native_tdt_minimal_import_manifest_v1.json').items():
        path=official/relative;require(record(path)['sha256']==sha,'Exact original official loss source')
        files['official_'+relative.replace('/','_')]=path
    cuda_root=P/'work/nemo-native-env-v2/Lib/site-packages/numba_cuda/numba/cuda'
    for key,path in {'CUDA_dispatcher':cuda_root/'dispatcher.py','CUDA_cache_core':cuda_root/'core/caching.py',
        'CUDA_cache_config':cuda_root/'core/config.py','CUDA_codegen':cuda_root/'codegen.py'}.items():files[key]=path
    for f in files.values():
        if f.suffix=='.py':compile(f.read_text(encoding='utf-8-sig'),str(f),'exec')
    pins={key:record(path) for key,path in files.items()}
    stagefile=A/'feature_window_native20_01/stages.jsonl'
    with stagefile.open(encoding='utf-8') as f:evidence=[json.loads(next(f)) for _ in range(6)]
    startup=max(v['worker_private_memory']['private_bytes'] for v in evidence[:2])
    cold=evidence[5]['worker_private_memory']['private_bytes']-evidence[4]['worker_private_memory']['private_bytes']
    require(startup==1_100_308_480 and cold==1_551_867_904,'Exact previously closed entire first native-row host increase')
    policy={'worker_private_forecast_bytes':3_750_000_000,'whole_owned_family_forecast_bytes':3_942_000_000,
        'parent_private_cap_bytes':128_000_000,'launcher_private_reserve_bytes':64_000_000,
        'maximum_official_import_ready_private_bytes':1_250_000_000,'entire_cold_host_growth_allowance_bytes':1_600_000_000,
        'unchanged_variation_allowance_bytes':750_000_000,'tiny_full_V_tensor_hash_staging_allowance_bytes':64_000_000,
        'component_sum_bytes':3_664_000_000,'CUDA_component_peak_ceiling_bytes':64_000_000,'CUDA_allocator_cap_bytes':8_000_000_000,
        'global_moderate_fraction':.90,'global_critical_fraction':.95,'minimum_headroom_bytes':2_000_000_000,
        'minimum_E_free_bytes':100_000_000_000,'three_fresh_before_launch_and_each_call':True,
        'actual_current_and_OS_lifetime_private_cap_enforced':True,'cold_growth_is_measured_candidate_not_formal_NVVM_allocator_bound':True}
    saved=A/'official_TDT_cold_host_actual_01'
    c={'kind':'ASR18_official_TDT_exact_six_cache_experiment_v1','production_enabled':False,'actual_training_updates':0,
        'calls':2,'shape':[1,4,4,8716],'labels':[5,300,8710],'GPU_python':str(P/'work/nemo-native-env-v2/Scripts/python.exe'),
        'GPU_python_binary':record(P/'work/nemo-native-env-v2/Scripts/python.exe'),'source_bindings':pins,'resource_policy':policy,
        'maximum_worker_seconds':180,'official_NeMo_root':str(official),
        'cache_directory':str(A/'official_TDT_owned_exact_kernel_cache_01/kernel_objects'),
        'seed_folder':str(A/'official_TDT_cache_seed_actual_01'),'hit_folder':str(A/'official_TDT_cache_hit_actual_01'),
        'seed_actual_process_proof':str(A/'official_TDT_cache_seed_actual_01/actual_process0.json'),
        'saved_baseline':{'worker_report':record(saved/'official_TDT/worker_report.json'),'actual_process0':record(saved/'actual_process0.json')},
        'scientific_policy':{'full_vocabulary':8711,'blank':8192,'durations':[0,1,2,3,4],'sigma':0.,'omega':0.,
            'fastemit_lambda':0.,'clamp':0.,'dtype':'FP32','reduction':'none','Python_RNG':'Preserved exactly as frozen native_tdt_checkpointed_v4 wrapper',
            'data':'Deterministic complete FP32 logits; no vocabulary compression or actual audio/model inputs'},
        'scope':'One official loss cold forward/backward and identical warm repeat. No joint/model/encoder/207Adam/teacher/frontend/audio.'}
    path=folder/'configuration_pending_ROOT_execution_review.json';write_json(path,c)
    proposal={'kind':'ASR18_official_TDT_cold_host_component_resource_proposal_v1','accepted':False,'execution_admitted':False,
        'configuration':record(path),'resource_policy':policy,'source_bindings':pins,'actual_closed_stage_source':record(stagefile),
        'historical_startup_max_private_bytes':startup,'entire_first_native_row_private_growth_bytes':cold,
        'historical_before_and_after_first_row':evidence[4:6],
        'source_bound_basis':'Entire first native-row private growth rounded UP1.6GB; no CNN/BLAS subtraction or attribution. Startup1.25GB is a required actual gate before loss. Full cold growth/compiler allowance +unchanged750MB+64MB tiny tensor staging rounded UP3.75GB. Prospective/actual caps can fail; no unchanged retry.',
        'full_FP32_logits_bytes':1*4*4*8716*4,'logical_tensor_buffer_bound_bytes':64_000_000,
        'new_model_payload_or_audio_read':False,'ML_imports_in_configuration_job':any(x in sys.modules for x in ('torch','numpy','numba')),
        'native_training_forecast_changed':False}
    write_json(folder/'resource_proposal.json',proposal)
    handoff={'configuration':record(path),'resource_proposal':record(folder/'resource_proposal.json'),'harness':record(harness),
        'source_bindings':pins,'execution_admitted':False,'production_enabled':False}
    write_json(folder/'handoff.json',handoff)
    print(json.dumps({'configuration':record(path),'proposal':record(folder/'resource_proposal.json'),'handoff':record(folder/'handoff.json'),
        'harness':record(harness),'source_count':len(pins)}))
if __name__=='__main__':main()
