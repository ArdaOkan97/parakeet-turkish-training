"""One full hard real-row proposal; source/metadata only, no ML/audio job."""
import ast,copy,hashlib,json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
E=Path(r'E:\gpu_training_scratchpad');A=E/'analysis/ASR18_memory_diagnosis_v1';R=P/'work/research';B=E/'analysis/BPE_weak_then_clean_v1'
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_production_state_v1 import require

def pin(path):
    path=Path(path).resolve();require(path.is_relative_to(P) or path.is_relative_to(E),'P/E only');h=hashlib.sha256();st=path.stat()
    with path.open('rb') as f:
        while b:=f.read(1024**2):h.update(b)
    require((st.st_size,st.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns),'Stable source')
    return {'path':str(path),'bytes':st.st_size,'sha256':h.hexdigest()}

def main():
    require(os.environ['CUDA_VISIBLE_DEVICES']=='-1' and read_json(E/'storage_config.json')['storage_root']==str(E),'Source-only E preparation')
    c=copy.deepcopy(read_json(A/'single_TR207_configuration_03/configuration_pending_ROOT_execution_review.json'))
    for value in c['source_bindings'].values():require(pin(value['path'])==value,'Exact previously reviewed dependency')
    harness=R/'run_ASR18_single_full_hard_TR207_memory_v3.py'
    files={k:Path(v['path']) for k,v in c['source_bindings'].items()};files.update(harness=harness,configuration_builder=Path(__file__))
    queue=[harness]
    while queue:
        f=queue.pop()
        for n in ast.walk(ast.parse(f.read_text(encoding='utf-8-sig'))):
            names=([n.module] if isinstance(n,ast.ImportFrom) and n.module else [v.name for v in n.names] if isinstance(n,ast.Import) else [])
            for name in names:
                path=S/(name.split('.')[0]+'.py')
                if path.exists() and path not in files.values():files[name.split('.')[0]]=path;queue.append(path)
    for f in files.values():compile(f.read_text(encoding='utf-8-sig'),str(f),'exec')
    c['source_bindings']={k:pin(v) for k,v in files.items()}
    source=A/'hard_U_rows_decode_01/decoded/report.json';actual=A/'hard_U_rows_decode_01/actual_owned_CLOSED0.json'
    decode=read_json(source);item=decode['rows'][0];row=item['row']
    require(decode['passed'] and decode['heldout_PCM_check_performed'] and row['encoder_frames']==1991 and row['target_tokens']==574 and len(row['tr_metadata_labels'])==574,'Exact full original maximum T/U row and independent heldout proof')
    require(read_json(actual)['actual_exit_code']==0 and read_json(actual)['report']==pin(source),'Genuine prior original PCM actual0')
    require(pin(item['PCM_cache']['path'])==item['PCM_cache'],'Exact retained whole NPY without decoding/copy')
    prior=A/'full_hard_2D_native_01/full_hard_native_validation.json';history=read_json(prior)['history'][0]
    require(history['full_T']==1991 and history['full_U']==574 and history['native_memory']['private_peak_bytes']==5_726_646_272,'Exact historical full native shape peak, not old undercounted forecast')
    peak=history['native_memory']['private_peak_bytes'];forecast=peak+750_000_000+64_000_000
    require(forecast==6_540_646_272 and forecast<8_000_000_000,'Retained750M/write and original8GB maximum; no CNN-credit subtraction')
    from tdt_full_native_allocation_budget_v1 import allocation_budget
    joint=allocation_budget(1991,574,projected_elements=(1991+574+1)*640,head_elements=8716*641,hidden_width=640)
    require(joint['total_budget_bytes']==791_167_592 and joint['total_budget_bytes']<1_700_000_000,'Correct v7 entire explicit DP/joint union')
    graph={'conservative32_boundary_inputs_actual24layers_FP32':32*1991*1024*4,'one_recompute_block64_width_equivalents_estimate':64*1991*1024*4,
        'two_relative_BD_arrays_FP32':2*8*1991*(2*1991-1)*4,'two_attention_arrays_FP32':2*8*1991*1991*4,
        'decoder_and_input_staging':32_000_000,'fullCNN_stitch_and_local_union':150_000_000,'shared_positions_masks':64_000_000}
    cuda_candidate=sum(graph.values())+joint['total_budget_bytes']+4*81_459_760+1_073_741_824
    require(cuda_candidate<7_500_000_000,'Separate conservative CUDA candidate under explicit observed cap, not host accounting credit')
    population=B/'ASR18_final_two_pass_population_metadata_with_declared_geometry_v1.json';meta=read_json(population)
    require(meta['complete'] and meta['population_mean_U']==101470757/2631115,'Exact full BPE TRAIN population denominator')
    c.update(kind='ASR18_single_full_hard_TR207_memory_diagnostic_v1',selected_row=row,PCM_array_sha256='6cfe986bb9ddef2f4d8a98a20b0d4c8bdb9af242be7e2af80e6992df4b98899a',population_mean_U=meta['population_mean_U'],
        maximum_stage_seconds={'prepare':180,'train':600,'recover':180},input_scope='ONE whole159.24s retainedTRAIN source1051133/T1991/U574, effective1/physical1, no teacher/replay/augmentation/samplercommit, source BPE64250+LoRA62250 original207 matureAdam; unqualified diagnostic.')
    c['evidence'].update(hard_PCM_reference=pin(source),hard_PCM_actual0=pin(actual),hard_retained_PCM_NPY=pin(item['PCM_cache']['path']),prior_full_hard_native=pin(prior),
        prior_full_hard_actual0=pin(A/'full_hard_2D_native_01/actual_owned_CLOSED0.json'),full_population_metadata=pin(population))
    c['resource_policy'].update(training_private_hard_cap_bytes=forecast,training_whole_family_hard_cap_bytes=forecast+192_000_000,
        CUDA_peak_allocated_ceiling_bytes=7_500_000_000,bootstrap_whole_family_forecast_bytes=forecast+192_000_000,
        full_hard_native_empirical_peak_bytes=peak,unchanged_variation_allowance_bytes=750_000_000,checkpoint_write_increment_bytes=64_000_000,
        CPU_frontend_private_cap_bytes=1_800_000_000,CPU_frontend_full_family_forecast_bytes=1_992_000_000,CPU_frontend_incremental_fresh_admission_bytes=1_500_000_000,
        initial_native_admission='Actual all-setup ready<=2.1GB then full declared wholeworker6.540646272 minus actualcurrent plus exact unused128Mparent/64MALLshim; fullforecast also admitted before startup.',
        maximum_remaining_parent_and_shim_prospective_allowance_bytes=192_000_000)
    # Remove inherited short-row-only accounting fields, which this full-shape harness never consumes.
    for key in ('post_ready_full_native_increment_bytes','all_logical_graph_gradient_optimizer_staging_bytes',
                'remaining_parent_and_shim_prospective_allowance_bytes','first_native_requires_fresh_measured_ready_plus_entire_remaining_increment'):
        c['resource_policy'].pop(key,None)
    c['resource_policy']['all_joint_DP_logical_bytes']=joint['total_budget_bytes']
    c['resource_policy']['separate_encoder_graph_and_CUDA_candidate_bytes']=cuda_candidate
    c['resource_policy']['bootstrap_whole_family_forecast_bytes']=3_192_000_000
    c['resource_policy']['full_training_family_admitted_before_process_launch_bytes']=forecast+192_000_000
    rawpath=B/'ASR18_final_data_population_for_bounded_preparation_v1.json';raw=read_json(rawpath);bankpin=raw['bindings']['heldout_PCM']
    require(pin(bankpin['path'])==bankpin,'Exact final complete heldout bank')
    bank=read_json(bankpin['path']);require(bank['passed'] is True and bank['complete'] is True and bank['reference_clips']==len(bank['PCM_i16_sha256'])==12273 and bank['metadata_binding']==raw['bindings']['heldout_metadata'],'Complete12273 source-bound PCM bank')
    require(item['reader_evidence']['PCM_i16_sha256']=='ebeb085f96a560b05d92b8eded2e356e9b495ffb32b8594adc25c973ca40fecc' and item['reader_evidence']['PCM_i16_sha256'] not in bank['PCM_i16_sha256'],'Known immutable actual PCM_i16 absent; metadata-only no redecode/scan')
    c['evidence'].update(full_heldout_PCM_bank=bankpin,current_final_DATA_raw=pin(rawpath),current_heldout_metadata=raw['bindings']['heldout_metadata'])
    c['resource_policy']['unchanged_end_cache_reserved_ceiling_bytes']=3_000_000_000
    folder=A/'single_full_hard_TR207_configuration_04';folder.mkdir(exist_ok=False)
    path=folder/'configuration_pending_ROOT_execution_review.json';write_json(path,c)
    proposal={'kind':'ASR18_single_full_hard_TR207_resource_proposal_v1','accepted':False,'configuration':pin(path),'source_bindings':c['source_bindings'],'resource_policy':c['resource_policy'],
        'empirical_host_basis':{'prior_actual_full_T1991_U574_native':peak,'unchanged_variation':750_000_000,'separate_persistence':64_000_000,'total_worker':forecast,'family':forecast+192_000_000,'CNN_component_credit_subtracted':0},
        'prior_v6_allocation_forecast_is_not_reused':True,'current_v7_full_allocation_budget':joint,'encoder_checkpoint_graph_candidate_parts':graph,
        'graph_is_source_estimate_not_formal_allocator_proof':True,'CUDA_candidate_with_separate1GiB_workspace':cuda_candidate,'CUDA_actual_peak_ceiling':7_500_000_000,'CUDA_allocator_unchanged8GB':True,
        'CPU_scope':'Original pinned AutoFeatureExtractor on already decoded complete float32 NPY; no source audio re-decode. Whole leaf1.8GB/family1.992GB and fresh1.5GB incremental guard; physical15925/valid15924/T1991/full574 targets mandatory.',
        'runtime_hard_joint_limit_remains_1_7GB':True,'source_checkpoint_seed':'Original source BPE64250/LoRA62250/207 weights + mature3Adam64250/2000/62250, never any diagnostic output.',
        'write_and_recovery':'PREsave full207/3Adam/all4RNG fingerprint; streamedwrite within64MB and postwriteOS gate; genuine native0/dead then freshly admitted3.35GB worker with entire350MB exactrestore, fullsource snapshot unconsumed.',
        'selection_is_one_extremal_shape_not_population_training':True,'original_CNN_byte_equivalence_on_this_actual_row_claimed':False,'production_enabled':False}
    write_json(folder/'resource_proposal.json',proposal);write_json(folder/'handoff.json',{'configuration':pin(path),'resource_proposal':pin(folder/'resource_proposal.json'),'source_bindings':c['source_bindings'],'execution_admitted':False,'production_enabled':False})
    print(json.dumps({'config':pin(path),'proposal':pin(folder/'resource_proposal.json'),'handoff':pin(folder/'handoff.json'),'sources':len(files),'CUDA_candidate':cuda_candidate}))

if __name__=='__main__':main()
