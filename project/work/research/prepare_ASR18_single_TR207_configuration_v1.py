"""Small source/resource seal; no original audio or numerical runtime import."""
import ast,hashlib,json,math,os,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
E=Path(r'E:\gpu_training_scratchpad');A=E/'analysis/ASR18_memory_diagnosis_v1';F=E/'analysis/FastMSS_integration_v1'
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_production_state_v1 import require
def pin(path):
    path=Path(path).resolve();require(path.is_relative_to(P) or path.is_relative_to(E),'Project/E only');st=path.stat();h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(1024**2):h.update(b)
    require((st.st_size,st.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns),'Stable source');return {'path':str(path),'bytes':st.st_size,'sha256':h.hexdigest()}
def main():
    require(os.environ['CUDA_VISIBLE_DEVICES']=='-1' and read_json(E/'storage_config.json')['storage_root']==str(E),'E metadata only')
    folder=A/'single_TR207_configuration_01';folder.mkdir(exist_ok=False);harness=P/'work/research/run_ASR18_single_TR207_memory_v1.py'
    helper=P/'work/research/run_ASR18_frozen_CNN_lifetime_pair_v8.py';original=P/'work/training-env/Lib/site-packages/transformers/models/parakeet/modeling_parakeet.py'
    files={'harness':harness,'configuration_builder':Path(__file__),'component_guard':helper,'original_model':original,
        'original_model_config':original.with_name('configuration_parakeet.py'),'original_frontend':original.with_name('feature_extraction_parakeet.py')}
    queue=[harness,helper]
    while queue:
        path=queue.pop();tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        for n in ast.walk(tree):
            names=([n.module] if isinstance(n,ast.ImportFrom) and n.module else [v.name for v in n.names] if isinstance(n,ast.Import) else [])
            for name in names:
                candidate=S/(name.split('.')[0]+'.py')
                if candidate.exists() and candidate not in files.values():files[name.split('.')[0]]=candidate;queue.append(candidate)
    official=P/'work/nemo-native-env-v1/Lib/site-packages/nemo'
    for relative,sha in read_json(S/'native_tdt_minimal_import_manifest_v1.json').items():
        path=official/relative;require(pin(path)['sha256']==sha,'Original official loss unchanged');files['official_'+relative.replace('/','_')]=path
    for f in files.values():compile(f.read_text(encoding='utf-8-sig'),str(f),'exec')
    pins={k:pin(v) for k,v in files.items()}
    reference=F/'actual_cleaner_TR_references_01/report.json';archive=F/'actual_cleaner_feature_window_durable_metadata_after_native21_v1.json'
    row=read_json(reference)['references'][0];old=read_json(archive);frontend=old['metadata']['owner.json']['frontend']
    descriptor=A/'feature_window_native20_01/after_update20.state.json';d=read_json(descriptor);dd=lambda node:{k[1]:v for k,v in node[1]}
    weights={k:d['tensors'][v[1]] for k,v in dd(dd(d['tree'])['weights']).items()}
    gradients=sum(v['bytes'] for v in weights.values());largest=max(v['bytes'] for v in weights.values())
    require(len(weights)==207 and gradients==81_459_760 and largest==22_312_960,'All207 exact FP32 descriptors')
    from tdt_full_native_allocation_budget_v1 import allocation_budget
    joint=allocation_budget(13,6,projected_elements=(13+6+1)*640,head_elements=8716*641,hidden_width=640)
    joint_tensors=sum(v for k,v in joint['parts'].items() if k!='joint_runtime_workspace_reserve')
    graph=64_000_000;stream=8_388_608;misc=8_000_000
    union=gradients+joint_tensors+graph+stream+misc
    require(union<384_000_000 and 2*gradients+misc<384_000_000,'Complete joint/grad/Adam temporal union within384MB')
    extra=1_073_741_824+400_000_000+750_000_000+384_000_000+64_000_000
    require(extra==2_671_741_824,'Explicit unchanged cold/backend/variation plus full logical/write allowance')
    policy={'bootstrap_worker_private_forecast_bytes':3_000_000_000,'bootstrap_whole_family_forecast_bytes':3_192_000_000,
        'actual_ready_private_prerequisite_bytes':2_100_000_000,'post_ready_full_native_increment_bytes':extra,
        'training_private_hard_cap_bytes':4_800_000_000,'training_whole_family_hard_cap_bytes':4_992_000_000,
        'backend_workspace_allowance_bytes':1_073_741_824,'separate_measured_TDT_cold_host_allowance_bytes':400_000_000,
        'unchanged_variation_allowance_bytes':750_000_000,'all_logical_graph_gradient_optimizer_staging_bytes':384_000_000,
        'checkpoint_write_increment_bytes':64_000_000,'recovery_entire_private_forecast_bytes':3_350_000_000,
        'recovery_increment_bytes':350_000_000,'parent_private_cap_bytes':128_000_000,'launcher_private_cap_bytes':64_000_000,
        'CUDA_peak_allocated_ceiling_bytes':2_000_000_000,'CUDA_allocator_cap_bytes':8_000_000_000,
        'global90_three_fresh_95critical_headroom2GB_E100GB_unchanged':True,'actual_current_and_OS_lifetime_caps':True,
        'first_native_requires_fresh_measured_ready_plus_entire_remaining_increment':True,'whole_future_fit_claimed_at_bootstrap':False}
    original_config=read_json(A/'full_DATA_two_window_configuration_CPU_actual_05/finite_two_window_configuration_pending_execution_review.json')
    c={'kind':'ASR18_single_real_TR207_memory_diagnostic_v1','production_enabled':False,'native_updates':1,'sampler_draws':0,
        'teacher_or_replay_forward':False,'selected_row':row['source_row'],'population_mean_U':read_json(reference)['cleaner_catalog_identity']['population_mean_U'],
        'CPU_python':pin(E/'scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe'),'GPU_python':pin(P/'work/nemo-native-env-v2/Scripts/python.exe'),
        'maximum_stage_seconds':{'prepare':120,'train':420,'recover':180},'source_bindings':pins,'resource_policy':policy,
        'evidence':{'clean_reference':pin(reference),'clean_feature_archive':pin(archive),'actual207_descriptor_metadata':pin(descriptor),
            'source_projection':pin(A/'source_projection_v4/report.json'),'source_optimizer_preflight':pin(A/'optimizer_projected_v4/report.json')},
        'frontend':frontend,'augmentation_identity':original_config['augmentation_identity'],
        'diagnostic_only':True,'qualified':False,'input_scope':'Original full1s CLEAN TRAIN ordinal1, full six targets, no augmentation/crop; no replay/sampler draws/recipe-quality assertion.'}
    path=folder/'configuration_pending_ROOT_execution_review.json';write_json(path,c)
    proposal={'kind':'ASR18_single_TR207_private_resource_proposal_v1','accepted':False,'configuration':pin(path),'source_bindings':pins,'resource_policy':policy,
        'descriptor_metadata':pin(descriptor),'actual_FP32_weight_and_gradient_bytes':gradients,'largest_parameter_bytes':largest,
        'complete_joint_allocation_budget':joint,'joint_tensor_bytes_excluding_separate_backend_runtime_reserve':joint_tensors,
        'raw_207_gradient_bytes':gradients,'conservative_all_Adam_foreach_sqrt_temporaries_bytes':gradients,
        'gradient_clip_norm_scalars_and_six_row_gain_and_hash_misc_allowance_bytes':misc,
        'largest_streamed_matrix_bytes':stream,'encoder_decoder_saved_graph_and_feature_staging_allowance_bytes':graph,
        'native_tensor_union_before_rounding_bytes':union,'logical_allowance_bytes':384_000_000,
        'sequential_Adam_union_bytes':2*gradients+misc,'conditional_actual_ready_plus_full_increment_at_ready_upper_bound_bytes':2_100_000_000+extra,
        'graph_basis':'T13/U6:32 checkpointed encoder blocks, width1024.64MB permits~1200 full FP32 T13x1024 buffers (37 per block), plus actual101x128 input/decoder length7x640. Attention pair terms13x13xheads are smaller. NEW actual component measurement must still fit; not every-shape/private proof.',
        'teacher_omission_basis':'Frozen sampled_preservation returns literal zeros before teacher.values() when replay_indices empty; original bootstrap creates no teacher. No replay recipe claim.',
        'phase_admission':'Bootstrap3GB+192MB only. After measured actualready<=2.1GB, require ready+2671741824<=4.8GB and THREE fresh full increment global90/headroom2GB admissions before first forward. Denial stops with no native/Adam update. All current and process-lifetime caps apply.',
        'bootstrap_source_recipe_changed':False,'original207_Adam_RNG_or_clocks_reset':False,'historical_native_peak_subtracted':False,
        'training_quality_or_full_recipe_claimed':False,'new_ML_or_audio_imported':any(v in sys.modules for v in ('torch','numpy','soundfile'))}
    write_json(folder/'resource_proposal.json',proposal);write_json(folder/'handoff.json',{'configuration':pin(path),'resource_proposal':pin(folder/'resource_proposal.json'),
        'harness':pin(harness),'source_bindings':pins,'execution_admitted':False,'production_enabled':False})
    print(json.dumps({'configuration':pin(path),'proposal':pin(folder/'resource_proposal.json'),'handoff':pin(folder/'handoff.json'),'sources':len(pins),'logical_union':union}))
if __name__=='__main__':main()
