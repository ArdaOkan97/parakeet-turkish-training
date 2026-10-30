"""Fixed eight-source/20-update resource proposal; metadata only, E outputs."""
import ast,copy,hashlib,json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
E=Path(r'E:\gpu_training_scratchpad');A=E/'analysis/ASR18_memory_diagnosis_v1';F=E/'analysis/FastMSS_integration_v1'
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_production_state_v1 import require
def pin(path):
    path=Path(path).resolve();require(path.is_relative_to(P) or path.is_relative_to(E),'Project/E only');h=hashlib.sha256();st=path.stat()
    with path.open('rb') as f:
        while b:=f.read(1024**2):h.update(b)
    require((st.st_size,st.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns),'Stable pin');return {'path':str(path),'bytes':st.st_size,'sha256':h.hexdigest()}
def main():
    require(os.environ['CUDA_VISIBLE_DEVICES']=='-1' and read_json(E/'storage_config.json')['storage_root']==str(E),'E metadata only')
    folder=A/'fixed_TR_replay207_configuration_01';folder.mkdir(exist_ok=False)
    prior=read_json(A/'single_TR207_configuration_03/configuration_pending_ROOT_execution_review.json');c=copy.deepcopy(prior)
    files={k:Path(v['path']) for k,v in prior['source_bindings'].items()}
    files.update(harness=P/'work/research/run_ASR18_fixed_TR_replay207_memory_v1.py',configuration_builder=Path(__file__),
        fixed_operations=P/'work/research/asr18_fixed_TR_replay207_operations_v1.py')
    queue=[files['fixed_operations']]
    while queue:
        f=queue.pop();tree=ast.parse(f.read_text(encoding='utf-8-sig'))
        for n in ast.walk(tree):
            names=([n.module] if isinstance(n,ast.ImportFrom) and n.module else [v.name for v in n.names] if isinstance(n,ast.Import) else [])
            for name in names:
                path=S/(name.split('.')[0]+'.py')
                if path.exists() and path not in files.values():files[name.split('.')[0]]=path;queue.append(path)
    for f in files.values():compile(f.read_text(encoding='utf-8-sig'),str(f),'exec')
    c['source_bindings']={k:pin(v) for k,v in files.items()}
    refs=read_json(c['evidence']['clean_reference']['path'])['references'];fixed=[]
    for v in refs:fixed.append({**v,'audio_file':v['source_audio_file']})
    replaypath=A/'window_replay_PCM_actual_01/report_plan_v2.json';rp=read_json(replaypath)
    for language in ('de','en','es','fr','it','pt'):fixed.append(next(v for v in rp['references'] if v['language']==language))
    require(len(fixed)==8 and max(v['source_row']['encoder_frames'] for v in fixed)==208 and max(len(v['full_target_ids']) for v in fixed)==95,'Exact fixed input envelope')
    c.update(kind='ASR18_fixed_TR_replay20_memory_diagnostic_v1',native_updates=20,teacher_or_replay_forward=True,
        fixed_references=fixed,maximum_stage_seconds={'prepare':240,'train':1800,'recover':300},
        input_scope='Fixed diagnostic14TR alternating2original cleaner rows +1oldfive/1PT originalreplay pair,20updates; source samplers unconsumed, no production/quality/full-corpus claim.')
    c['evidence'].update(weak_feature_archive=pin(F/'weak_fixed14_feature_window20_durable_metadata_after_native20_v1.json'),
        replay_reference=pin(replaypath),actual_short_overall=pin(A/'single_TR207_actual_01/overall_result.json'),
        actual_short_native=pin(A/'single_TR207_actual_01/train/actual_process0.json'),actual_short_stages=pin(A/'single_TR207_actual_01/train/stages.jsonl'))
    from tdt_full_native_allocation_budget_v1 import allocation_budget
    def joint(t,u):return allocation_budget(t,u,projected_elements=(t+u+1)*640,head_elements=8716*641,hidden_width=640)
    short=joint(13,6);long=joint(208,95);delta=long['total_budget_bytes']-short['total_budget_bytes'];require(delta==33_734_048,'Exact fixed geometry joint extension')
    stages=[json.loads(v) for v in Path(c['evidence']['actual_short_stages']['path']).read_text().splitlines()]
    ready=next(v['worker_private_memory']['private_bytes'] for v in stages if v['phase']=='actual_original207_threeAdam_all_setup_ready')
    peak=next(v['private_high_water']['lifetime_private_peak_bytes'] for v in stages if v['phase']=='after_one_actual_threeAdam_update')
    observed=peak-ready;require(observed==1_808_441_344,'Actual changed native full-forward/backward/Adam lifetime peak delta, not writer')
    extra=observed+750_000_000+delta+64_000_000+128_000_000;require(extra==2_784_175_392,'Explicit no-margin-shrink extension')
    c['resource_policy'].update(post_ready_full_native_increment_bytes=extra,training_private_hard_cap_bytes=4_900_000_000,
        training_whole_family_hard_cap_bytes=5_092_000_000,unchanged_variation_allowance_bytes=750_000_000,
        running_high_water_margin_bytes=750_000_000,graph_candidate_bytes=128_000_000,replay_runtime_extension_bytes=128_000_000,
        native_joint_tensor_extension_bytes=delta,CUDA_peak_allocated_ceiling_bytes=2_500_000_000)
    c['resource_policy']['writer_is_separate_fresh_64MB_phase_not_native_overlap']=True
    c['resource_policy']['actual_ready_includes_original_teacher_resident_tensors']=True
    c['resource_policy']['calibration_is_measured_one_short_row_not_universal_forecast_proof']=True
    path=folder/'configuration_pending_ROOT_execution_review.json';write_json(path,c)
    graph={'conservative_boundary_inputs32_actual_layers24_FP32':32*208*1024*4,
        'one_recompute_block64_FP32_width_equivalents_estimate':64*208*1024*4,
        'two_relative_BD_arrays':2*8*208*(2*208-1)*4,'two_attention_matrices':2*8*208*208*4,
        'decoder_saved_features_staging':8_000_000,'physical_CNN_local_stitch_excluding_separate_streamed_weight':24_000_000,'shared_positions_masks':4_000_000}
    require(sum(graph.values())<128_000_000,'Explicit candidate graph plus quadratic and physicalCNN terms')
    proposal={'kind':'ASR18_fixed_original_TR_replay20_resource_proposal_v1','accepted':False,'configuration':pin(path),
        'source_bindings':c['source_bindings'],'resource_policy':c['resource_policy'],
        'fixed_rows':[{'source_id':v['source_id'],'language':v['language'],'samples':v['expected_audio']['samples'],'T':v['source_row']['encoder_frames'],'U':len(v['full_target_ids'])} for v in fixed],
        'calibration_actual_ready':ready,'calibration_actual_native_OS_lifetime_peak_through_Adam':peak,'calibration_native_increment':observed,
        'source_bound_actual_calibration':c['evidence']['actual_short_native'],'graph_candidate_parts':graph,'graph_candidate_sum':sum(graph.values()),
        'graph_candidate_is_not_formal_allocator_bound':True,'original_nonreentrant_block_checkpoint_semantics_unchanged':True,
        'only_one_physical1_microbatch_graph_at_a_time':True,'joint_short':short,'joint_selected_max':long,
        'teacher_resident_weights_in_measured_ready_not_in_128MB_runtime_pad':True,
        'fresh_first_native_projection':'Actual teacher-inclusive ready<=2.1GB +2,784,175,392 worker increment <=4.9GB; global THREE fresh remaining increment+192MB parent/shim. Subsequent target=max(first target, actual highwater+750MB), cannot exceed4.9GB.',
        'write_projection':'No retained-graph/private drop assumed; fresh64MBwrite+192MBfamily must fit current wholeworker cap. Explicit postwrite current AND OS lifetime stage. Independent PREsave fingerprint verified afterwrite.',
        'recovery_projection':'Only after actual native0/dead; fresh3.35GB worker, exact350MB before EACH1/10/20 restore, all207/3Adam/all4RNG and untouched original source snapshot.',
        'universal_shape_or_quality_or_retention_or_production_claimed':False,'original_effective_rows_per_update':16,'source_sampler_commit':False}
    write_json(folder/'resource_proposal.json',proposal)
    write_json(folder/'handoff.json',{'configuration':pin(path),'resource_proposal':pin(folder/'resource_proposal.json'),'harness':pin(files['harness']),
        'operations':pin(files['fixed_operations']),'source_bindings':c['source_bindings'],'execution_admitted':False,'production_enabled':False})
    print(json.dumps({'configuration':pin(path),'handoff':pin(folder/'handoff.json'),'proposal':pin(folder/'resource_proposal.json'),'sources':len(files),'graph_candidate_sum':sum(graph.values())}))
if __name__=='__main__':main()
