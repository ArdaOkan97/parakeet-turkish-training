"""Real readonly configure + full reserved-p4 bridge API, no FullOwner ctor/audio/GPU."""
import argparse,ctypes,json,os,sys,time,threading,traceback
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace
from ctypes import wintypes
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
sys.path.insert(0,str(S));sys.path.insert(0,str(R))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from run_ASR18_BPE9223_full_run_owner_v2 import fresh_admit,private_lifetime_limit
from run_ASR18_frozen_CNN_lifetime_pair_v8 import Watch,pid_identity,storage
ROOT=Path(r'E:/gpu_training_scratchpad')


def main(a):
    storage();folder=Path(a.folder).resolve();require(folder.is_relative_to(ROOT) and not folder.exists(),'One fresh E metadata output')
    cpin=record(a.config);rpin=record(a.request);review=read_json(a.ROOT_review)
    require(review['accepted'] is True and review['reviewer']=='/root'
        and review['scope']=='metadata_BPE9223_ACKed60_stable_readonly_and_real_reserved_p4_bridge_v1'
        and review['source']==record(__file__) and review['configuration']==cpin and review['resource_resume_request']==rpin
        and review['production_approval_claimed'] is False,'Exact ROOT readonly API/source/resource authority')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=wintypes.HANDLE
    kernel.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.SetPriorityClass.restype=wintypes.BOOL
    require(kernel.SetPriorityClass(kernel.GetCurrentProcess(),0x4000),'BelowNormal metadata API process')
    admissions=[]
    for _ in range(3):
        value=fresh_admit(2_000_000_000);require(value['passed'],'Fresh whole2GB metadata family');admissions.append(value);time.sleep(.2)
    folder.mkdir();write_json(folder/'admission.json',admissions);write_json(folder/'boot.json',{'producer':pid_identity(),'source':record(__file__)})
    watcher=Watch(folder,512_000_000,worker=True);watcher.phase='real_readonly_configure_and_bridge_p4'
    timer=threading.Timer(180,lambda:write_json(folder/'stop.json',{'reason':'reviewed180smetadata'}));timer.daemon=True;timer.start();code=1
    import psutil
    process=psutil.Process();began=time.monotonic()
    def guard():
        watcher.check();private_lifetime_limit(512_000_000)
        io=process.io_counters();require(io.read_bytes+io.write_bytes<=16*1024**3,'Bounded whole metadata process IO16GiB')
    @contextmanager
    def phase(name):
        require(name=='read_index','Only readonly existing metadata');guard();yield;guard()
    try:
        c=read_json(a.config);request=read_json(a.request)
        from asr18_BPE9223_full_run_controller_v17 import FullOwner,checked_ACKed60_session_resume_request
        require(c['full_owner_controller_source']==record(sys.modules[FullOwner.__module__].__file__),'Actual controller17 source')
        snapshot=checked_ACKed60_session_resume_request(cpin,{'resource_resume_request':rpin},rpin,guard=guard)
        failed=read_json(c['prior_acknowledged_committed60_stop']['path'])
        frozen={k:failed[k] for k in ('storage_state','durable_latest','current_registered_journal','pool_registry',
            'pool_index','pool_archive_prefix','storage_operation_witnesses')}
        frozen.update(failed['reserved_unconsumed_preparation']['files'])
        require(all(record(pin['path'])==pin for pin in frozen.values()),'Exact actual60/p4 before metadata API')
        reserved=snapshot['reserved_preparation'];workspace=Path(reserved['workspace']['workspace'])
        require(not(workspace/'CPU').exists() and not(workspace/'native').exists(),'No p4 CPU/native or producer')
        import asr18_BPE9223_operational_metadata_v2 as metadata
        authority=c['process_readonly_metadata_authority'];session=read_json(c['session']['path'])
        require(authority==session['operational_metadata'] and authority['operational_compatibility']!=session['operational_compatibility'],
            'Exact stable comp2 reader with distinct current mutable session12 authority')
        spools_before={str(path) for path in Path(authority['control_root']).glob('owned_member_*.tmp')}
        reader=metadata.configure(**authority,guard=guard,metadata_phase=phase)
        from asr18_BPE9223_indexed_checkpoint_storage_v2 import IndexedOwnedFutureCheckpointStore
        mutable=IndexedOwnedFutureCheckpointStore(c['operational_storage_owner'],guard=guard,
            operational_compatibility=c['operational_compatibility'])
        require(mutable.operational_compatibility==c['operational_compatibility'],'Actual current mutable store remains separate')
        from fastmss_BPE9223_resource_CPU_boundary_v2 import configure_session
        require(configure_session(session,guard) is True
            and metadata.configure(**authority,guard=guard,metadata_phase=phase) is reader,'Repeated exact CPU session uses SAME resolver')
        # Readonly real owner11 field view. Execute unchanged production boundary
        # construction and callback without FullOwner ctor, reserves or writes.
        schedule=SimpleNamespace(committed_updates=lambda:60,state_dict=lambda:snapshot['sampler_state'])
        view=SimpleNamespace(check=guard,parent=SimpleNamespace(check=guard),config=c,schedule=schedule,gpu=None,
            resource_closure=c['current_native_resource_closure'],durable=SimpleNamespace(head_pin=snapshot['durable_head']))
        for key in ('current_capture','current_packet','current_bridge','current_cleanup0','current_cleanup',
            'current_ack','current_recovery_report','closed_native','closed_recovery'):
            setattr(view,key,snapshot[key])
        FullOwner._refresh_boundary(view)
        boundary=view.boundary_state
        require(boundary['optimizer_updates']==60 and boundary['committed_window_updates']==20
            and boundary['actual_native_closed']==snapshot['closed_recovery']==failed['actual_normal_fresh_recovery0']
            and boundary['durable_head']==snapshot['durable_head'],'Genuine fresh60 normal authority, no oldaborted native0')
        boundary_path=folder/'actual_prior_boundary_p4.json';write_json(boundary_path,boundary);view.prior_boundary=record(boundary_path)
        callback=FullOwner.verify_prior_boundary.__get__(view,type(view))
        import inspect
        require(record(inspect.getsourcefile(callback))==c['full_owner_controller_source'],'Actual production callback source')
        require(callback(boundary) is True,'Actual callback validates real60/head7 boundary')
        from fastmss_BPE9223_plan_bridge_v9 import bridge_plan
        from asr18_BPE9223_source_context_v6 import verify_CPU_prior_boundary
        require(verify_CPU_prior_boundary(boundary,guard) is True,'Independent actual normal-fresh60/full9223 metadata boundary')
        bridge=bridge_plan(reserved['plan'],DATA_evidence=c['planning_arguments']['DATA_evidence'],session_binding=c['session'],
            window_index=3,prior_window_boundary=view.prior_boundary,verify_prior_boundary=callback,guard=guard)
        require(bridge['session']==c['session'] and bridge['coordinator_plan']==reserved['plan']
            and bridge['initial_committed_updates']==60 and len(bridge['updates'])==20
            and sum(len(row['rows']) for row in bridge['updates'])==320
            and bridge['actual_optimizer_updates']==bridge['actual_authoritative_draws_consumed']==0,
            'Complete real p4 source60-to80/320fullrows/targets; zero audio/consumed draws')
        bridge_path=folder/'actual_readonly_bridge_p4.json';write_json(bridge_path,bridge)
        from asr18_BPE9223_CPU_window_configuration_v8 import derive_contract,derive_configuration
        from fastmss_BPE9223_feature_protocol_v9 import verify_source_contract
        contract=derive_contract(c['CPU_template_configuration'],record(bridge_path),guard)
        contract_path=folder/'actual_CPU_source_contract.json';write_json(contract_path,contract)
        verify_source_contract(record(contract_path),guard=guard)
        proposal=derive_configuration(c['CPU_template_configuration'],record(contract_path),3,'p000000004',str(workspace/'CPU/payload'),guard)
        write_json(folder/'actual_CPU_configuration.json',proposal)
        from asr18_BPE9223_owned_GPU_runtime_v22 import verify_closed_journal_source_authority
        runtime_config={**c,'source_native_process0':snapshot['closed_native'],'source_checkpoint_capture':snapshot['current_capture']}
        saved=metadata.read_bound(snapshot['current_capture']['pointer'],guard)['journal_prefix']
        require(verify_closed_journal_source_authority(runtime_config,snapshot['closed_native'],saved,guard=guard) is True,
            'Real runtime22 accepts exact historical fresh60/runtime21-parent18 journal authority')
        from copy import deepcopy
        negatives=0
        for changed in ('pointer','journal','source'):
            bad=deepcopy(runtime_config);bad_saved=deepcopy(saved)
            if changed=='pointer':bad['source_checkpoint_capture']['pointer']['sha256']='0'*64
            elif changed=='journal':bad_saved['sha256']='0'*64
            else:bad['parent_driver_source']['sha256']='0'*64
            try:verify_closed_journal_source_authority(bad,snapshot['closed_native'],bad_saved,guard=guard)
            except (ValueError,KeyError,AssertionError):negatives+=1
            else:raise ValueError('Changed historical '+changed+' must reject')
        require(negatives==3,'Three real-source runtime gate negatives')
        require(metadata.configure(**authority,guard=guard,metadata_phase=phase) is reader,'Parent and subsequent configure one immutable resolver')
        require(all(record(pin['path'])==pin for pin in frozen.values()) and record(c['session']['path'])==c['session']
            and not(workspace/'CPU').exists() and not(workspace/'native').exists(),
            'All actual state/journal/head/pool/13p4 bytes unchanged; output only new analysisfolder')
        require({str(path) for path in Path(authority['control_root']).glob('owned_member_*.tmp')}==spools_before,
            'All owned readspools closed, no residual selected-member spool')
        guard();io=process.io_counters();mutable.close()
        report={'kind':'ASR18_actual_ACKed60_stable_session_configure_and_reserved_p4_bridge_preflight_v1','passed':True,
            'source':record(__file__),'producer':pid_identity(),'configuration':cpin,'resource_resume_request':rpin,
            'actual_bridge':record(bridge_path),'actual_bridge_semantic_sha256':digest(bridge),'source_controller_callback':c['full_owner_controller_source'],
            'actual_original_callback_on_readonly_owner11_fields_executed':True,'FullOwner_constructor_invoked':False,
            'same_single_readonly_resolver_before_after_CPU_session_calls':True,'frozen_CPU_session_rewritten':False,
            'readonly_authority':authority,'mutable_authority':c['operational_compatibility'],'mutable_store_operation_invoked':False,
            'real_p4_updates':20,'real_full_rows':320,'real_p4_initial_optimizer_updates':60,'runtime22_real_historical60_positive':True,'runtime22_real_source_negatives':negatives,'protocol9_source_contract_and_factory8_derived':True,
            'actual_audio_GPU_native_or_optimizer_draws':0,'authoritative_storage_journal_pool_p4_mutation':False,'resolver_owned_read_spools_allowed_and_closed':True,
            'existing_storage_verifier_may_stream_payload_SHA':True,'tensor_payload_loaded_into_ML':False,
            'process_IO_read_bytes':io.read_bytes,'process_IO_write_bytes':io.write_bytes,'wall_seconds':time.monotonic()-began,
            'private':private_lifetime_limit(512_000_000),'all_authoritative_state_pins_unchanged':True}
        write_json(folder/'report.json',report);code=0
        print(json.dumps({'passed':True,'report':record(folder/'report.json'),'bridge':record(bridge_path)}),flush=True)
    except BaseException:
        with (folder/'traceback.log').open('a',encoding='utf8') as stream:traceback.print_exc(file=stream)
        raise
    finally:
        timer.cancel();watcher.close(code)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--request',required=True)
    p.add_argument('--ROOT-review',required=True);p.add_argument('--folder',required=True);main(p.parse_args())
