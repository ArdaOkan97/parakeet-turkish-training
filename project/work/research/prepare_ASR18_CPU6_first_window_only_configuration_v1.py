"""Thin CPU-only metadata assembly; ROOT resource rebind required for config."""
import argparse,ast,copy,hashlib,json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
E=Path(r'E:\gpu_training_scratchpad');A=E/'analysis/ASR18_memory_diagnosis_v1'
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
def function(path,name):
    text=Path(path).read_text(encoding='utf-8-sig');node=next(v for v in ast.parse(text).body if isinstance(v,ast.FunctionDef) and v.name==name)
    return ast.dump(node,include_attributes=False)
def main(resource_review):
    require(os.environ['CUDA_VISIBLE_DEVICES']=='-1' and read_json(E/'storage_config.json')['storage_root']==str(E),'Metadata E-only')
    wrapper=P/'work/research/run_ASR18_CPU6_first_window_only_v1.py';parent=P/'work/research/run_ASR18_full_DATA_multiwindow_v9.py'
    require(function(wrapper,'run_CPU')==function(parent,'run_CPU'),'Copied run_CPU EXACT unchanged AST, truthful new __file__ identity')
    original_folder=A/'full_DATA_two_window_configuration_CPU_actual_05';original_path=original_folder/'finite_two_window_configuration_pending_execution_review.json'
    old=read_json(original_path);oldreview=read_json(old['actual_ROOT_CPU_resource_review']['path']);manifest=read_json(old['source_manifest']['path'])
    files={'harness':wrapper,'builder':Path(__file__),'copied_callable_original_parent':parent,
        'telemetry_helper':P/'work/research/run_ASR18_frozen_CNN_lifetime_pair_v8.py',
        'CPU_controller':P/'work/research/prepare_FastMSS_phase_window_CPU_v6.py','repreparation_comparator':S/'asr18_reprepared_array_equivalence_v1.py'}
    for p in files.values():compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
    pins={k:record(v) for k,v in files.items()}
    policy={'parent_private_cap_bytes':64_000_000,'CPU_child_family_allowance_bytes':1_934_217_728,
        'whole_CPU_job_allowance_bytes':1_998_217_728,'maximum_CPU_seconds':900,
        'unchanged_inner_controller_leaf_shim_resource_review':oldreview['resource_review'],
        'unchanged_raw_lifetime_review':oldreview['raw_lifetime_review'],
        'unchanged_global90_threefresh_95critical_headroom2GB_E100GB':True,'GPU_admission_claimed':False}
    proposal={'kind':'ASR18_CPU6_first_window_only_source_resource_proposal_v1','accepted':False,'production_enabled':False,
        'outer_parent_source':pins['harness'],'resource_policy':policy,'unchanged_CPU_owner_bundle':manifest['CPU_owner_bundle'],
        'old_numeric_resource_review':old['actual_ROOT_CPU_resource_review'],'old_caller_source_manifest':old['source_manifest'],
        'original_caller_configuration':record(original_path),'first_plan':record(original_folder/'original_first_window_plan.json'),
        'source_bindings':pins,'run_CPU_AST_sha256':hashlib.sha256(function(wrapper,'run_CPU').encode()).hexdigest(),
        'scope':'ONE CPU-only original20update feature window, all40 actual reference/frontend children + sink4 stdlib finalization then exact640NPY/320rows/20payload comparison. No GPUService import/construction/native worker/Adam/draw; original failedcontroller1 remains1.',
        'source_identity':'Actual outer wrapper argv/PID/ctime/source are reported separately from copied unchanged parent9 run_CPU implementation provenance. Contract and ROOT resource receipt must bind actual wrapper.',
        'next_gate':'ROOT source/resource rebind to wrapper, then metadata config, exact ROOT CPU-only execution review and fresh1,998,217,728B whole-job admission. CPU proof never satisfies GPU budget.'}
    if resource_review is None:
        folder=A/'CPU6_first_window_only_source_proposal_v1';folder.mkdir(exist_ok=False);write_json(folder/'handoff.json',proposal)
        print(json.dumps({'proposal':record(folder/'handoff.json'),'harness':pins['harness'],'builder':pins['builder'],'execution_admitted':False}));return
    rrpin=record(resource_review);rr=read_json(resource_review)
    require(rr['accepted'] is True and rr['reviewer']=='/root' and rr['scope']=='CPU_preparation_only'
        and rr['outer_parent_source']==pins['harness'] and rr['CPU_source_manifest']==manifest['CPU_owner_bundle']
        and rr['resource_review']==oldreview['resource_review'] and rr['raw_lifetime_review']==oldreview['raw_lifetime_review'],
        'Genuine ROOT source/resource rebound, never restamp prior permission')
    folder=A/'CPU6_first_window_only_configuration_01';folder.mkdir(exist_ok=False)
    session=read_json(old['session']['path']);contract=read_json(session['CPU_preparation_contract']['path'])
    contract.update(outer_parent_source=pins['harness'],actual_ROOT_CPU_resource_review=rrpin)
    cp=folder/'reviewed_CPU_preparation_contract.json';write_json(cp,contract);session['CPU_preparation_contract']=record(cp)
    sp=folder/'CPU_only_session.json';write_json(sp,session)
    keys=('CPU_python','CPU_guard_source','CPU_controller_source','CPU_owned_allowance_bytes','CPU_inner_owned_allowance_bytes',
        'CPU_outer_controller_launcher_reserve_bytes','planning_arguments','first_window_repreparation_comparison')
    config={k:copy.deepcopy(old[k]) for k in keys};config.update(kind='ASR18_CPU6_first_window_only_configuration_v1',GPU_execution_enabled=False,
        actual_native_updates=0,production_enabled=False,maximum_CPU_phase_seconds=900,source_bindings=pins,CPU_only_resource_policy=policy,
        original_caller_configuration=record(original_path),actual_ROOT_CPU_resource_review=rrpin,session=record(sp),
        first_plan=proposal['first_plan'],copied_run_CPU_provenance={'source':record(parent),'exact_function_AST_sha256':proposal['run_CPU_AST_sha256']})
    conf=folder/'CPU_only_configuration_pending_execution_review.json';write_json(conf,config)
    result={'configuration':record(conf),'session':record(sp),'CPU_contract':record(cp),'first_plan':proposal['first_plan'],
        'source_bindings':pins,'resource_policy':policy,'execution_admitted':False,'production_enabled':False}
    write_json(folder/'handoff.json',result);print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ROOT-CPU-review');a=p.parse_args();main(a.ROOT_CPU_review)
