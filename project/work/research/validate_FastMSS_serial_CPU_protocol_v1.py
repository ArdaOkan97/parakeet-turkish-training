"""Small metadata/owned-Popen controls; no real DATA/audio/frontend admission."""
import argparse
from copy import deepcopy
import json,os,subprocess,sys,time
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
import fastmss_row_serial_cpu_v1 as core
import fastmss_owned_process_closure_v1 as proc

def child(request,output):
    core.environment();guard=core.resource_guard
    value={'kind':'SYNTHETIC_owned_process_metadata_NOT_DATA_or_frontend','passed':True,
        'producer':proc.identity(),'worker_source':core.record(__file__,guard),
        'parent_source':core.record(__file__,guard),'raw_PCM_files_written':0,
        'actual_optimizer_updates':0,'sampler_journal_frontier_advanced':False}
    core.write_new(output,value)

def main(folder):
    core.environment();guard=core.resource_guard
    folder=core.scoped(folder);folder.mkdir(parents=True,exist_ok=False)
    before=list(core.sink.CACHE.iterdir()) if core.sink.CACHE.exists() else []
    checks=[]
    def check(name,action):action();checks.append(name)
    def reject(name,action):
        try:action()
        except (ValueError,KeyError,TypeError,ProcessLookupError):checks.append(name);return
        raise AssertionError('Expected rejection: '+name)
    sources=[Path(core.__file__),Path(proc.__file__),Path(core.bridge.__file__),Path(core.sink.__file__),
        PROJECT/'work/research/produce_FastMSS_independent_rows_CPU_v1.py',
        PROJECT/'work/research/produce_FastMSS_serial_features_CPU_v1.py',
        PROJECT/'work/research/prepare_FastMSS_phase_window_CPU_v1.py']
    import ast
    for path in sources:
        ast.parse(path.read_text());checks.append('source_parse_'+path.name)
    old_bridge=(PROJECT/'outputs/parakeet_turkish/fastmss_phase_aware_plan_bridge_v3.py').read_text()
    new_bridge=Path(core.bridge.__file__).read_text()
    a=next(n for n in ast.parse(old_bridge).body if isinstance(n,ast.FunctionDef) and n.name=='actual0')
    b=next(n for n in ast.parse(new_bridge).body if isinstance(n,ast.FunctionDef) and n.name=='actual0')
    assert ast.dump(a)==ast.dump(b);checks.append('final_DATA_actual0_AST_unchanged')
    assert not any(k=='torch' or k.startswith(('torch.','transformers','scipy','pyarrow','librosa')) for k in sys.modules)
    checks.append('parent_imports_no_numerical_frontend_or_model')
    for n,physical,valid,t in ((160,2,1,1),(319,2,1,1),(1280,9,8,1),(16000,101,100,13),(263040,1645,1644,206)):
        assert core.geometry(n)=={'clean16k_samples':n,'physical_feature_frames':physical,'valid_mel_frames':valid,'encoder_frames':t}
        checks.append('physical_vs_valid_scalar_'+str(n))
    reject('bool_samples_rejected',lambda:core.geometry(True))
    reject('incomplete_DATA_still_rejected',lambda:core.bridge.validate_DATA({},guard))
    monitor=folder/'child_guard';report=folder/'child_report.json';log=folder/'child.log'
    script=core.record(__file__,guard);guard_source=core.record(PROJECT/'work/research/ASR18_bounded_CPU_phase_v3.py',guard)
    arguments=[str(core.CPU_ENV/'Scripts/python.exe'),'-B',guard_source['path'],'--folder',str(monitor),
        '--seconds','20','--script',script['path'],'--','--child','--output',str(report)]
    with log.open('xb') as stream:
        p=subprocess.Popen(arguments,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        launcher=proc.identity(p.pid);code=p.wait(timeout=25)
    assert code==0
    pin=proc.observe(p,kind='ASR18_owned_CPU_process_completion_v1',owner='/root/memory_diagnosis',
        report=core.record(report,guard),worker_source=script,parent_source=script,guard_source=guard_source,
        guard_closure=core.record(monitor/'closure.json',guard),launcher_identity=launcher,
        parent_guard_observation=guard(),output=folder/'actual_process0.json',guard=guard)
    proof=core.read_bound(pin,guard)
    check('real_owned_Popen_guard_exit0',lambda:proc.verify(proof,kind=proof['kind'],owner='/root/memory_diagnosis',guard=guard))
    check('dedicated_reference_process0',lambda:core.bridge.actual_reference0(pin,core.record(report,guard),guard))
    check('dedicated_owned_cleanup_process0',lambda:core.bridge.observed_cleanup0(proof,guard))
    reject('final_DATA_refuses_process0_without_tool',lambda:core.bridge.actual0(pin,core.record(report,guard),guard,owner='/root/memory_diagnosis'))
    reject('GPU_process_validator_refuses_CPU_kind',lambda:core.bridge.observed_native0(proof,guard))
    def mutation(key,value):
        altered=deepcopy(proof);altered[key]=value
        return lambda:proc.verify(altered,kind=proof['kind'],owner='/root/memory_diagnosis',guard=guard)
    for key,value in (('actual_Popen_wait_code',True),('actual_native_exit_code',1),('parent_observed_wait',False),
        ('producer_dead_after_wait',False),('owner','/root/other'),('actual_tool_chunk_id','fabricated'),
        ('worker_source',guard_source),('parent_source',guard_source),('report',guard_source)):
        reject('reject_'+key,mutation(key,value))
    altered=deepcopy(proof);altered['launcher']['ppid']+=1
    reject('reject_foreign_Popen_parent',lambda:proc.verify(altered,kind=proof['kind'],owner='/root/memory_diagnosis',guard=guard))
    altered=deepcopy(proof);altered['producer']=proc.identity()
    reject('reject_live_producer_as_exit',lambda:proc.verify(altered,kind=proof['kind'],owner='/root/memory_diagnosis',guard=guard))
    altered=deepcopy(proof);altered['parent_guard_observation']['system_memory']['commit_available_bytes']=1
    reject('reject_insufficient_parent_headroom',lambda:proc.verify(altered,kind=proof['kind'],owner='/root/memory_diagnosis',guard=guard))
    wrong_guard=core.read_bound(proof['guard_closure'],guard);wrong_guard['observed']['samples']=2
    path=folder/'invalid_two_sample_guard.json';core.write_new(path,wrong_guard)
    reject('reject_less_than_three_child_samples',mutation('guard_closure',core.record(path,guard)))
    reject('no_fake_Popen_object',lambda:proc.observe(object(),kind=proof['kind'],owner='/root/memory_diagnosis',report=proof['report'],worker_source=script,parent_source=script,guard_source=guard_source,guard_closure=proof['guard_closure'],launcher_identity=launcher,parent_guard_observation=guard(),output=folder/'never_written.json',guard=guard))
    assert (list(core.sink.CACHE.iterdir()) if core.sink.CACHE.exists() else [])==before
    checks.append('actual_shared_CACHE_untouched')
    core.write_new(folder/'report.json',{'kind':'FastMSS_row_serial_CPU_metadata_and_process_protocol_fixture_v1','passed':True,
        'checks':checks,'check_count':len(checks),'actual_child_process0':pin,'actual_shared_CACHE_unchanged':True,
        'genuine_final_DATA_positive_tested':False,'original_audio_decoded':False,'original_CPU_frontend_executed':False,
        'actual_native_updates':0,'actual_production_enabled':False,'sources':[core.record(p,guard) for p in sources]})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder');p.add_argument('--child',action='store_true');p.add_argument('--output')
    a=p.parse_args();child(None,a.output) if a.child else main(a.folder)
