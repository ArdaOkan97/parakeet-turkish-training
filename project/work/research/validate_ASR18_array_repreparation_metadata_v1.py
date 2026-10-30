"""New source/metadata gate controls; no actual arrays, cache writes or audio."""
import argparse,ast,copy,importlib.util,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from fastmss_row_serial_cpu_v6 import environment,resource_guard
from asr18_reprepared_array_equivalence_v1 import ROW_KEYS,exact,validate_retirement
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')


def method(path,cls,name):
    tree=ast.parse(path.read_bytes());obj=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls) if cls else tree
    return ast.dump(next(x for x in obj.body if isinstance(x,ast.FunctionDef) and x.name==name))


def main(folder):
    environment();sample=resource_guard();m=sample['system_memory'];budget=64_000_000
    require(m['commit_used_bytes']+budget<.9*m['commit_limit_bytes'] and m['commit_available_bytes']-budget>2_000_000_000,'Fresh64MB metadata forecast')
    require(folder.resolve().is_relative_to(A.resolve()) and not folder.exists(),'Fresh E-only metadata fixture')
    folder.mkdir();write_json(folder/'fresh64MB_admission.json',{'passed':True,'increment_bytes':budget,'sample':sample})
    checks=[]
    def check(name,value):require(value,name);checks.append(name)
    def reject(name,call):
        try:call()
        except (ValueError,KeyError):checks.append(name);return
        raise AssertionError('Expected fail-closed: '+name)
    p=R/'run_ASR18_full_DATA_multiwindow_v9.py';old=R/'run_ASR18_full_DATA_multiwindow_v8.py'
    for cls,names in [('ParentGuard',('check','_run','close')),('GPUService',('__init__','_watch','check','finish','abort'))]:
        for name in names:check(cls+'.'+name+' unchanged',method(p,cls,name)==method(old,cls,name))
    check('Actual CPU family wait/drain/admission unchanged',method(p,None,'run_CPU')==method(old,None,'run_CPU'))
    before=(S/'asr18_full_DATA_native_window_v3.py').read_text().replace('fastmss_phase_aware_feature_window_v3','fastmss_phase_aware_feature_window_v4')
    check('Entire native engine changes ONLY sink import; all math/state/phase operations unchanged',ast.dump(ast.parse(before))==ast.dump(ast.parse((S/'asr18_full_DATA_native_window_v4.py').read_bytes())))
    spec=importlib.util.spec_from_file_location('source_only_parent9',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    check('New parent imports without ML/numerical/audio runtime',not any(x in sys.modules for x in ('torch','numpy','scipy','av','soundfile','pyarrow','transformers')))
    template=read_json(S/'asr18_full_DATA_multiwindow.disabled.v9.json')
    check('Worker path remains exact STRING',type(template['worker_driver_source']) is str and template['worker_driver_source']==str(p.resolve()) and template['parent_driver_source']==record(p))
    check('Unarmed finite40 only',template['TEMPLATE_NOT_LAUNCHABLE'] is True and template['finite_diagnostic_execution_enabled'] is False and template['maximum_total_native_updates']==40 and template['production_enabled'] is False)
    check('Resident/recovery/native margins unchanged',template['entire_worker_forecast']==read_json(S/'asr18_full_DATA_multiwindow.disabled.v8.json')['entire_worker_forecast'] and template['final_recovery_worker_forecast']==read_json(S/'asr18_full_DATA_multiwindow.disabled.v8.json')['final_recovery_worker_forecast'])
    check('No fabricated future cleanup proof in template',template['first_window_repreparation_comparison']['prior_owned_retirement_actual0'] is None)
    # Read only small actual closed metadata, never a NPY or source waveform.
    prior=A/'full_DATA_two_window_native_02';failure=record(prior/'actual_owned_CLOSED1_v1.json')
    inventory=read_json(prior/'preserved_original_feature_inventory_after_failure_v1.json')
    check('Exactly640 durable original array hashes are present',len([x for x in inventory['actual_files'] if x['path'].endswith('.npy')])==640)
    rows=0
    for i in range(20):
        proof=read_json(prior/('CPU_window_0/payload/frontend_closed_%02d.json'%i))
        pin=proof['update_record'];check('Actual old update metadata signature%d'%i,record(pin['path'])==pin)
        value=read_json(pin['path']);require(len(value['rows'])==16,'Full old14TR+2RP schema')
        for row in value['rows']:
            require(set(ROW_KEYS).issubset(row) and type(row['full_target_ids']) is list
                and type(row['target_tokens']) is int and len(row['full_target_ids'])==row['target_tokens'],
                'Comparator full row field names match actual old saved schema')
            rows+=1
    check('All320 actual row metadata schemas are compatible',rows==320)
    reject('Typed row identity drift bool/int rejected',lambda:exact({'draw':False},{'draw':0},'typed drift'))
    reject('Full target mutation rejected',lambda:exact([1,2,8710],[1,2,3],'target drift'))
    reject('Ordered contexts differ even with duplicate sourceID',lambda:exact([{'source':'x','draw':0},{'source':'x','draw':1}],[{'source':'x','draw':1},{'source':'x','draw':0}],'order drift'))
    wrong=folder/'explicitly_negative_cleanup_kind.json';write_json(wrong,{'kind':'explicitly_negative_not_an_actual_cleanup'})
    reject('Missing genuine cleanup TOOL0 blocks source replay admission',lambda:validate_retirement(record(wrong),prior_failure=failure,guard=resource_guard))
    wrong2=folder/'explicitly_negative_boolean_cleanup_code.json';write_json(wrong2,{'kind':'ASR18_actual_exact_owned_failed_window_discard_TOOL0_v2','owner':'/root/memory_diagnosis','actual_exit_code':False})
    reject('Boolean cleanup exitFalse cannot pass integer0 gate',lambda:validate_retirement(record(wrong2),prior_failure=failure,guard=resource_guard))
    check('No numerical/audio import after every new control',not any(x in sys.modules for x in ('torch','numpy','scipy','av','soundfile','pyarrow','transformers')))
    result={'kind':'ASR18_new_finalizer_caller_array_equivalence_metadata_controls_v1','passed':True,'checks':checks,'check_count':len(checks),
        'source':record(__file__),'parent':record(p),'engine':record(S/'asr18_full_DATA_native_window_v4.py'),
        'comparison':record(S/'asr18_reprepared_array_equivalence_v1.py'),'actual_prior_row_metadata_read':320,
        'actual_arrays_or_audio_read':False,'existing_cache_files_changed':False,'fake_positive_process_or_cleanup_proof_created':False,
        'actual_repreparation_or_native_comparison_pass_claimed':False,'actual_GPU_or_Adam_or_draws':0,'production_enabled':False}
    write_json(folder/'report.json',result);print(json.dumps({'passed':True,'checks':len(checks),'old_actual_metadata_rows':rows,'no_arrays_audio_GPU':True}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folder',required=True,type=Path);main(parser.parse_args().folder)
