"""One bounded stdlib configuration seal; no ML imports or component execution."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
E=Path(r'E:\gpu_training_scratchpad');A=E/'analysis/ASR18_memory_diagnosis_v1'
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_production_state_v1 import require

def record(path):
    path=Path(path).resolve();require(path.is_relative_to(P) or path.is_relative_to(E),'Project/E only')
    st=path.stat();h=hashlib.sha256()
    with path.open('rb') as stream:
        while b:=stream.read(1024**2):h.update(b)
    require((st.st_size,st.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns),'Stable pin')
    return {'path':str(path),'bytes':st.st_size,'sha256':h.hexdigest()}

def main():
    require(read_json(E/'storage_config.json')['storage_root']==str(E),'Required E')
    require(os.environ['CUDA_VISIBLE_DEVICES']=='-1','No GPU configuration job')
    folder=A/'frozen_CNN_pair_configuration_06';folder.mkdir(exist_ok=False)
    harness=P/'work/research/run_ASR18_frozen_CNN_lifetime_pair_v6.py'
    original=P/'work/training-env/Lib/site-packages/transformers/models/parakeet/modeling_parakeet.py'
    files={'harness':harness,'original_CNN':original,'configuration_builder':Path(__file__),
        'original_CNN_configuration':original.with_name('configuration_parakeet.py')}
    # Complete active project-local import closure; installed runtime packages are
    # separately fixed to the existing exact coherent runtime, never installed here.
    queue=[harness]
    while queue:
        path=queue.pop();tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            names=([node.module] if isinstance(node,ast.ImportFrom) and node.module else
                [item.name for item in node.names] if isinstance(node,ast.Import) else [])
            for name in names:
                candidate=S/(name.split('.')[0]+'.py')
                if candidate.exists() and candidate not in files.values():files[name.split('.')[0]]=candidate;queue.append(candidate)
    for path in files.values():compile(path.read_text(encoding='utf-8-sig'),str(path),'exec')
    pins={key:record(path) for key,path in files.items()}
    bank_path=A/'frozen_weight_bank_v1/report.json';bank=read_json(bank_path)
    require(bank['passed'] is True and bank['GPU_used'] is False and bank['matrix_count']==294,'Actual immutable original bank')
    base=Path(bank['source']['path']);st=base.stat();base_signature={'path':str(base),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
    matrices={name[len('encoder.subsampling.'):]:value for name,value in bank['matrices'].items() if name.startswith('encoder.subsampling.')}
    require(len(matrices)==6,'Six CNN matrices')
    biases={};raw_read=0
    with base.open('rb') as stream:
        length=struct.unpack('<Q',stream.read(8))[0];require(length<4*1024**2,'Bounded original header')
        raw_header=stream.read(length);header=json.loads(raw_header);start=8+length
        for name,info in header.items():
            if name.startswith('encoder.subsampling.') and name.endswith('.bias'):
                low,high=info['data_offsets'];require(info['dtype']=='F32' and high-low<65536,'Bounded six raw original biases')
                stream.seek(start+low);raw=stream.read(high-low);require(len(raw)==high-low,'Exact bias length')
                raw_read+=len(raw);biases[name[len('encoder.subsampling.'):]]={'descriptor':info,'raw_sha256':hashlib.sha256(raw).hexdigest()}
    require(len(biases)==6 and raw_read==9216,'Exactly 9216 original bias bytes; no model deserialization')
    rows=[]
    for name,f,v in [('left_edge7',7,6),('exact8',8,8),('physical_extra9',9,8),('masked_suffix33',33,16),
        ('one_second101',101,100),('seam257',257,256),('seam273',273,272),('measured_replay1519',1519,1518),
        ('measured_TR8914',8914,8913),('full_hard15925',15925,15924)]:
        rows.append({'name':name,'physical_frames':f,'valid_frames':v,'physical_output_frames':(f+7)//8,'valid_encoder_frames':(v+7)//8})
    f=15925;a1=256*((f+1)//2)*64*2;a2=256*((f+3)//4)*32*2;a3=256*((f+7)//8)*16*2
    live_pairs={'conv0':f*128*4+a1,'relu1':2*a1,'depthwise2':a1+a2,'pointwise3':2*a2,
        'relu4':2*a2,'depthwise5':a2+a3,'pointwise6':2*a3,'relu7':2*a3,
        'flatten_linear':2*a3+8_388_608+((f+7)//8)*1024*2}
    logical=max(live_pairs.values())+2*f*128*4+2*a3+2*((f+7)//8)*1024*2+8_388_608
    require(logical<=650_000_000,'Sequential synchronized live tensor envelope')
    evidence=A/'feature_window_native20_01/stages.jsonl'
    with evidence.open(encoding='utf-8') as stream:startup=[json.loads(next(stream)) for _ in range(2)]
    require([v['phase'] for v in startup]==['production_prep_runtime_ready','production_prep_narrow_native_ready'],'Actual owned import stages')
    policy={'worker_private_forecast_bytes':4_100_000_000,'parent_private_cap_bytes':128_000_000,
        'launcher_private_reserve_bytes':64_000_000,'whole_owned_family_forecast_bytes':4_292_000_000,
        'maximum_CNN_ready_private_bytes':1_500_000_000,'logical_tensor_union_ceiling_bytes':650_000_000,
        'backend_workspace_allowance_bytes':1_073_741_824,'unchanged_variation_allowance_bytes':750_000_000,
        'CPU_staging_hash_output_allowance_bytes':64_000_000,'CUDA_tensor_and_workspace_peak_ceiling_bytes':1_723_741_824,
        'CUDA_allocator_cap_bytes':8_000_000_000,'global_moderate_fraction':.90,'global_critical_fraction':.95,
        'minimum_system_headroom_bytes':2_000_000_000,'minimum_E_free_bytes':100_000_000_000,
        'three_fresh_projected_samples_before_each_arm_and_case':True,
        'whole_worker_component_sum_bytes':4_037_741_824,
        'backend_workspace_bound_is_measured_candidate_not_formal_cuDNN_workspace_cap':True}
    previous=A/'frozen_CNN_lifetime_pair_actual_02'
    saved={name:record(previous/relative) for name,relative in {
        'original_report':'original/worker_report.json','inplace_report':'inplace_relu/worker_report.json',
        'original_process0':'original/actual_process0.json','inplace_process0':'inplace_relu/actual_process0.json',
        'actual_paired_summary':'actual_paired_summary_v1.json','parent_guard':'guard_closure.json'}.items()}
    c={'kind':'ASR18_frozen_CNN_tiled_saved_baseline_component_v1','production_enabled':False,'actual_training_updates':0,
        'arms':['temporal_tiled'],'GPU_python':str(P/'work/nemo-native-env-v2/Scripts/python.exe'),
        'maximum_worker_seconds':300,'resource_policy':policy,'source_bindings':pins,'cases':rows,
        'GPU_python_binary':record(P/'work/nemo-native-env-v2/Scripts/python.exe'),
        'bank_report':record(bank_path),'base_archive':bank['source'],'base_archive_signature':base_signature,
        'base_header_sha256':hashlib.sha256(raw_header).hexdigest(),'base_config':record(base.with_name('config.json')),
        'CNN_matrices':matrices,'CNN_biases':biases,
        'saved_baseline':saved,
        'numerical_policy':{'autocast':'BF16','weights':'original readonly BF16 matrix bank + original FP32 biases',
            'matmul_TF32':False,'cuDNN_TF32':True,'cuDNN_benchmark':False,'cuDNN_deterministic':False,
            'encoder_eval':True,'input_requires_grad':False,'all4_RNG_gate':'byte-exact before/after each forward',
            'output_gate':'all full physical BF16 bytes exact; complete both case lists before numeric verdict'},
        'scope':'Frozen CNN only; no full encoder/LoRA/Adam/teacher/audio/native training or checkpoint state'}
    path=folder/'configuration_pending_ROOT_execution_review.json';write_json(path,c)
    proposal={'kind':'ASR18_frozen_CNN_component_resource_proposal_v1','accepted':False,'execution_admitted':False,
        'configuration':record(path),'source_bindings':pins,'resource_policy':policy,
        'logical_arithmetic':{'F':f,'BF16_conv1':a1,'BF16_conv2_pointwise':a2,'BF16_conv3_pointwise':a3,
            'formula':'max(per_layer_input_plus_output)+2*F*128*4+2*A3+2*ceil(F/8)*1024*2+8388608',
            'per_layer_input_plus_output_bytes':live_pairs,'synchronized_sequential_tensor_envelope_bytes':logical,
            'miscellaneous_tensor_slack_bytes':650_000_000-logical,
            'lifetime_basis':'Original loop overwrites hidden_states after each layer; symmetric hooks synchronize every layer and retain only numeric snapshots, no tensors. Masks inplace; one matrix stream at a time. Additional flatten/output/input allowances counted even where not simultaneous.'},
        'previous_actual_admission_failed_no_GPU':record(A/'frozen_CNN_lifetime_pair_actual_01/original_fresh_admission.json'),
        'supported_workload_specific_change':'Replace all-layers output/copy union with source-derived synchronous sequential liveness; tighten component CUDAallocated ceiling and current/lifetime cap accordingly. All workspace/startup/variation/staging/family margins unchanged.',
        'startup_evidence':record(evidence),'startup_observations':startup,
        'startup_actual_private_max_bytes':max(x['worker_private_memory']['private_bytes'] for x in startup),
        'CNN_ready_1_5GB_is_actual_prerequisite_not_proved_new_runtime':True,
        'workspace1GiB_is_separate_candidate_measured_against_CUDA_peak_not_formal_backend_allocator_bound':True,
        'variation_750MB_preserved':True,'denied_training_forecast_borrowed_or_reduced':False,
        'host_saving_assumed':False,'fresh_three_paired_before_arm_and_each_case_required':True,
        'new_original_bias_bytes_read':raw_read,'original_header_bytes_read':len(raw_header)+8,
        'no_ML_imports_or_model_instantiation':not any(x in sys.modules for x in ('torch','numpy','transformers')),
        'time_target':'First causal comparison within focus window; ROOT exact source/config/resource review still mandatory'}
    write_json(folder/'resource_proposal.json',proposal)
    write_json(folder/'handoff.json',{'configuration':record(path),'resource_proposal':record(folder/'resource_proposal.json'),
        'harness':record(harness),'builder':record(__file__),'source_bindings':pins,'production_enabled':False,'execution_admitted':False})
    print(json.dumps({'folder':str(folder),'configuration':record(path),'proposal':record(folder/'resource_proposal.json'),
        'handoff':record(folder/'handoff.json'),'harness':record(harness),'source_count':len(pins),'logical_union':logical}))

if __name__=='__main__':main()
