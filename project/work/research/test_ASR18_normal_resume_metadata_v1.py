"""Focused readonly metadata negatives after a genuine normal first4/owner0.

Uses actual source-bound pause/request/proofs as the positive control. Declared
in-memory fault injection exercises semantic gates; no original file is edited,
no fake process/review artifact is produced and no heavy child is created.
"""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import argparse,json,sys
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
import asr18_full_run_parent_controller_v13 as caller
import fastmss_owned_process_closure_v2 as process
from fastmss_phase_aware_plan_bridge_v7 import read_bound
from fastmss_row_serial_cpu_v8 import environment,resource_guard
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
ROOT=Path(r'E:\gpu_training_scratchpad')

def main(config_path,request_path,output):
    environment();guard=resource_guard;config=record(config_path);request_pin=record(request_path)
    request=read_bound(request_pin,guard);snapshot_pin=request['normal_pause_snapshot'];snapshot=read_bound(snapshot_pin,guard)
    c=read_bound(config,guard);parent=c['parent_driver_source']
    # Private protocol fixture fields consumed by checked_normal_resume_request;
    # this object is not an accepted ROOT execution review and is never saved as one.
    fixture={'resume_request':request_pin,'invocation':'normal_boundary_resume'}
    def operation():return caller.checked_normal_resume_request(config,fixture,request_pin,parent_source=parent,guard=guard)
    require(operation()==snapshot,'Genuine normally closed actual first4 positive metadata control')
    checks=[{'name':'genuine_normal_owner0_current_recovery0_ACK_exact_prefix','passed':True}]
    original_bound=caller.read_bound;original_read=caller.read_json
    def reject(name,overrides=None,json_overrides=None,extra=None):
        overrides={} if overrides is None else overrides;json_overrides={} if json_overrides is None else json_overrides
        def bound(pin,callback):
            value=original_bound(pin,callback)
            return deepcopy(overrides.get(pin['path'],value))
        def read(path):
            value=original_read(path);return deepcopy(json_overrides.get(str(path),value))
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch.object(caller,'read_bound',bound));stack.enter_context(patch.object(caller,'read_json',read))
            if extra is not None:stack.enter_context(extra)
            try:operation()
            except (ValueError,KeyError,TypeError,OSError,AssertionError) as error:
                checks.append({'name':name,'passed':True,'expected_rejection':type(error).__name__,'message':str(error)[:240]});return
        raise ValueError('Missing expected metadata rejection: '+name)
    def changed_snapshot(name,change):
        value=deepcopy(snapshot);change(value);reject(name,{snapshot_pin['path']:value})
    value=deepcopy(request);value['configuration']={**config,'sha256':'0'*64}
    reject('wrong_immutable_configuration',{request_pin['path']:value})
    value=deepcopy(request);value['controller_source']={**record(caller.__file__),'sha256':'0'*64}
    reject('wrong_frozen_controller_source',{request_pin['path']:value})
    reject('still_live_prior_owner',extra=patch.object(process,'dead',lambda producer:False))
    changed_snapshot('stale_current_ACK',lambda x:x['current_ack'].update(sha256='0'*64))
    latest_path=Path(c['run_namespace'])/'durable/latest.json';latest=original_read(latest_path)
    stale=deepcopy(latest);stale['head']={**latest['head'],'sha256':'0'*64}
    reject('newer_or_stale_durable_head',json_overrides={str(latest_path):stale})
    state_path=Path(c['checkpoint_storage_namespace'])/'state.json';state=original_read(state_path)
    pending=deepcopy(state);pending['pending_publication']={'private_fault':True}
    reject('pending_publication_or_storage_mutation',json_overrides={str(state_path):pending})
    changed_snapshot('sampler_counter_mismatch',lambda x:x.update(actual_optimizer_updates=x['actual_optimizer_updates']+1))
    changed_snapshot('next_packet_mismatch',lambda x:x.update(next_draw_packet_sha256='0'*64))
    changed_snapshot('response_release_nonce_mismatch',lambda x:x['current_response'].update(nonce='0'*32))
    changed_snapshot('pool_registry_quota_state_mismatch',lambda x:x['worker_control_pool_state'].update(sha256='0'*64))
    # Actual source journal is never touched. Only its observed stat for this
    # exact path is injected to represent a preserved unknown trailing suffix.
    journal=Path(snapshot['journal_prefix']['path']);original_stat=Path.stat
    def stat(path,*args,**kwargs):
        value=original_stat(path,*args,**kwargs)
        if path==journal:
            from types import SimpleNamespace
            return SimpleNamespace(**{key:getattr(value,key) for key in dir(value) if key.startswith('st_') and key!='st_size'},st_size=value.st_size+1)
        return value
    reject('unknown_journal_suffix_preserved_and_rejected',extra=patch.object(Path,'stat',stat))
    # The sole SQLite query is source-local preparing-workspace metadata. A
    # private result injection exercises the exact guard without creating any workspace.
    import sqlite3
    class Query:
        def fetchone(self):return ('private_preparing_workspace',)
    class Database:
        def execute(self,query):require("status='preparing'" in query,'Only exact private query injection');return Query()
        def close(self):pass
    reject('inflight_preparing_workspace',extra=patch.object(sqlite3,'connect',lambda *a,**k:Database()))
    require(operation()==snapshot,'Original metadata/prefix/state unchanged after every injected negative')
    require('torch' not in sys.modules and 'numpy' not in sys.modules,'No heavy imports/native/model/audio')
    require(output.is_relative_to(ROOT) and not output.exists(),'Fresh E-only negative receipt')
    write_json(output,{'kind':'ASR18_genuine_normal_pause_readonly_metadata_controls_v1','passed':True,
        'source':record(__file__),'configuration':config,'request':request_pin,'checks':checks,
        'fault_injection':'declared in-memory semantic/stat/query observations only; original bytes untouched',
        'actual_GPU_audio_model_or_Popen_calls_here':0,'private_fixture_is_not_ROOT_execution_authority':True})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('config','resume-request','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();main(a.config,a.resume_request,Path(a.output).resolve())
