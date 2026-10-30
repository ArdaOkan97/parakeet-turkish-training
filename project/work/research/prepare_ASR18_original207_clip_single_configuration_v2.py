"""Metadata-only optimizer pair configuration; never imports ML."""
import ast,hashlib,json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
E=Path(r'E:\gpu_training_scratchpad');A=E/'analysis/ASR18_memory_diagnosis_v1'
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_production_state_v1 import require
def pin(path):
    p=Path(path).resolve();require(p.is_relative_to(P) or p.is_relative_to(E),'Project/E only');s=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(4*1024**2):h.update(b)
    require((s.st_size,s.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns),'Stable exact source')
    return {'path':str(p),'bytes':s.st_size,'sha256':h.hexdigest()}
def main():
    require(os.environ['CUDA_VISIBLE_DEVICES']=='-1' and read_json(E/'storage_config.json')['storage_root']==str(E),'CPU metadata E-only')
    folder=A/'original207_clip_single_configuration_02';folder.mkdir(exist_ok=False)
    files={'harness':P/'work/research/run_ASR18_original207_clip_single_v2.py','builder':Path(__file__),
        'telemetry_helper':P/'work/research/run_ASR18_frozen_CNN_lifetime_pair_v8.py'}
    queue=list(files.values())
    while queue:
        p=queue.pop();tree=ast.parse(p.read_text(encoding='utf-8-sig'))
        for n in ast.walk(tree):
            names=([n.module] if isinstance(n,ast.ImportFrom) and n.module else [v.name for v in n.names] if isinstance(n,ast.Import) else [])
            for name in names:
                p2=S/(name.split('.')[0]+'.py')
                if p2.exists() and p2 not in files.values():files[name.split('.')[0]]=p2;queue.append(p2)
    for p in files.values():compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
    policy={'worker_private_forecast_bytes':3_350_000_000,'whole_owned_family_forecast_bytes':3_542_000_000,
        'parent_private_cap_bytes':128_000_000,'launcher_private_reserve_bytes':64_000_000,
        'maximum_import_CUDA_ready_private_bytes':1_250_000_000,'source_and_state_transfer_allowance_bytes':700_000_000,
        'optimizer_clip_Adam_cold_and_workspace_allowance_bytes':512_000_000,'unchanged_variation_allowance_bytes':750_000_000,
        'staging_bytes':64_000_000,'CUDA_tensor_and_workspace_peak_ceiling_bytes':1_000_000_000,'CUDA_allocator_cap_bytes':8_000_000_000,
        'global90_threefresh_95critical_headroom2GB_E100GB_unchanged':True,'current_AND_OS_lifetime_cap':True}
    require(1_250_000_000+700_000_000+512_000_000+750_000_000+64_000_000==3_276_000_000<3_350_000_000,'Full conservative component sum')
    c={'kind':'ASR18_original207_clip_single_saved_default_component_v1','arms':['clip_single'],'maximum_worker_seconds':240,
        'actual_training_updates':0,'production_enabled':False,'GPU_python':pin(P/'work/nemo-native-env-v2/Scripts/python.exe'),
        'source_bindings':{k:pin(v) for k,v in files.items()},'resource_policy':policy,
        'evidence':{'source_projection':pin(A/'source_projection_v4/report.json'),'optimizer_preflight':pin(A/'optimizer_projected_v4/report.json'),
            'actual_short_integrated_summary':pin(A/'single_TR207_actual_01/actual_short_TR207_summary_v1.json')},
        'scope':'Two fresh optimizer-only processes; actual original207 BPE64250/LoRA62250/mature3Adam. No model/audio/loss/native trajectory or production proof.',
        'only_changed_execution_policy':'Clip forwarding wrapper adds ONLY foreach=False; original frozen full step/default AdamW foreach=None unchanged. One arm against saved default actual0, no baseline rerun.',
        'strict_gate':'Compare actual saved default full207 weights, all3 complete Adam states/clocks and scalar policies, all4RNG, full pre/postclip gradients and norm bytes. No metadata or numerical exclusions.',
        'gradient':'Each full FP32 parameter gradient generated serially CPU chunks<=4MiB, exact (int32 arange%257-128)/4096, copied to GPU; no RNG.'}
    proposal={'kind':'ASR18_original207_Adam_component_resource_candidate_v1','accepted':False,'resource_policy':policy,
        'arithmetic':{'import_CUDA_prerequisite':1_250_000_000,'source_transfer_constructor_allowance':700_000_000,
            'optimizer_cold_workspace':512_000_000,'unchanged_variation':750_000_000,'staging':64_000_000,'sum':3_276_000_000,
            'worker_cap':3_350_000_000,'family_cap':3_542_000_000},
        'basis':'Actual tiny Torch/CUDA prerequisite1.10GB below1.25GB, existing reviewed sequential original source/Adam transfer700MB, actual integrated Adam phase251.12MB within512MB combined cold+temporary allowance. This is a prospective component budget, not a proven total-private bound.',
        'CUDA_union':{'named207_weights':81_459_760,'all3_moments':162_919_520,'all207_gradients':81_459_760,
            'conservative_three_parameter_size_intermediates':244_379_280,'additional_stream_scalar_staging_allowance':64_000_000,'sum':634_218_320,'ceiling':1_000_000_000},
        'fresh_phase':'After actual source/gradient setup, actual current+512MBcold/workspace+750MBvariation+64MBstaging <=3.35GB, plus THREE fresh global samples with same full remaining +192MB family. Current AND actual OS-lifetime caps at every synchronized stage.',
        'hypothesis':'Actual original clip phase gained232MB currentprivate with only178KB CUDApeak. One clip-only foreach=False arm tests measured cause; no specific library attribution, saving or numerical equality assumed.',
        'not_claimed':'CUDA-only reduction is not host savings. Fresh-process phase and lifetime records may be censored by earlier peaks; preserve both arm reports on numerical mismatch.'}
    oracle_folder=A/'original207_Adam_pair_actual_01/default'
    oracle={'actual_process0':pin(oracle_folder/'actual_process0.json'),'report':pin(oracle_folder/'worker_report.json'),
        'configuration':pin(A/'original207_Adam_pair_configuration_01/configuration_pending_ROOT_execution_review.json'),
        'ROOT_review':pin(A/'ROOT_original207_Adam_pair_configuration01_source_resource_execution_review_v1.json')}
    c['saved_default_oracle']=oracle;c['evidence'].update({'saved_default_'+k:v for k,v in oracle.items()})
    c['evidence']['preserved_AdamFalse_pair_failure']=pin(A/'original207_Adam_pair_actual_01/actual_optimizer_pair_summary_v1.json')
    c['scope']='ONE new optimizer-only clipFalse process against saved actual default0; original207/3Adam, no model/audio/loss or native trajectory proof.'
    cp=folder/'configuration_pending_ROOT_execution_review.json';rp=folder/'resource_proposal.json';write_json(cp,c);write_json(rp,proposal)
    handoff={'configuration':pin(cp),'resource_proposal':pin(rp),'harness':c['source_bindings']['harness'],
        'source_bindings':c['source_bindings'],'source_pin_count':len(files),'execution_admitted':False,'production_enabled':False}
    write_json(folder/'handoff.json',handoff);print(json.dumps({k:handoff[k] for k in ('configuration','resource_proposal','harness','source_pin_count')}))
if __name__=='__main__':main()
