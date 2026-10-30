"""Exact failed-RUN06 feature retirement; never a successful native closure."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import psutil
P=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(P/'outputs/parakeet_turkish'),str(P/'work/research')]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from fastmss_owned_process_closure_v2 import identity,dead
import fastmss_BPE9223_feature_protocol_v11 as public
import fastmss_BPE9223_feature_engine_v11 as engine
from run_ASR18_BPE9223_full_run_owner_v19 import storage,sample_memory

def main():
    a=argparse.ArgumentParser();a.add_argument('--request',required=True);a.add_argument('--folder',required=True)
    args=a.parse_args();storage();started=time.monotonic();initial_io=psutil.Process().io_counters()
    require(sys.flags.utf8_mode==1 and os.environ.get('PYTHONUTF8')=='1','UTF8 at process entry')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1','CPU-only retirement')
    out=Path(args.folder).resolve();require(out.is_relative_to(Path(r'E:\gpu_training_scratchpad')),'E-only evidence')
    req=read_json(args.request);review=read_json(req['ROOT_review']['path'])
    require(record(req['ROOT_review']['path'])==req['ROOT_review'] and review['accepted'] is True
        and review['reviewer']=='/root' and review['scope']=='exact_failed_RUN06_owned_features_only'
        and review['source']==record(__file__) and review['failure']==req['failure']
        and review['owner']==req['owner'] and review['manifest']==req['manifest'], 'Exact ROOT failure/source/owned-target review')
    failure=read_json(req['failure']['path']);require(record(req['failure']['path'])==req['failure'],'Exact sealed actual failure')
    require(failure['kind']=='ASR18_actual_fresh_RUN06_committed20_cleanup_protocol_source_failure_TOOL1_v1'
        and failure['actual_tool_exit_code']==failure['actual_exec_tool_exit_code']==1
        and failure['actual_tool_chunk_id']=='99ea92' and failure['preserved_production_updates']==20
        and failure['all_captured_identities_dead'] is True and len(failure['all_captured_identities'])==7
        and failure['actual_native_normal_exit0_claimed'] is False,'Honest abnormal closed owner20; no native exit number invented')
    def guard():
        require(time.monotonic()-started<180,'Bounded retirement180 seconds')
        require(all(dead(v) for v in failure['all_captured_identities']),'Every exact captured production/cleanup PID+ctime remains dead')
        sm=sample_memory(os.getpid(),'failed_RUN06_readonly_validation_then_feature_retirement')
        require(sm['paired_sampling_within_one_second'] and sm['worker_private_memory']['available']
            and sm['worker_private_memory']['private_bytes']<=512000000,'512MB owned metadata worker')
        system=sm['system_memory'];require(system['commit_fraction']<.95 and system['commit_available_bytes']>2000000000,
            'Unchanged critical system/headroom guards')
        io=psutil.Process().io_counters();require(io.read_bytes+io.write_bytes-initial_io.read_bytes-initial_io.write_bytes<=16*1024**3,
            'Bounded16GiB total process IO')
    guard()
    completion=read_json(failure['current_completion']['path'])
    require(record(failure['current_completion']['path'])==failure['current_completion']
        and completion['global_new_updates']==20 and completion['feature_owner']==req['owner']
        and completion['feature_manifest']==req['manifest'],'Genuine exact completed window owner/manifest')
    config=read_json(failure['configuration']['path'])
    require(record(failure['configuration']['path'])==failure['configuration'],'Exact failed production configuration')
    protected=[failure['current_checkpoint_capture'][k] for k in ('pointer','checkpoint','pre_save_fingerprint','next_draw_packet')]
    pointer=read_json(protected[0]['path']);journal=Path(pointer['journal_prefix']['path'])
    protected += [record(journal)]
    run=Path(config['run_namespace'])
    for rel in ('checkpoints/state.json','checkpoints/catalog.sqlite','durable/latest.json'):
        protected.append(record(run/rel))
    before=[record(p['path']) for p in protected];require(before==protected,'Protected saved20 metadata/journal exact')
    owner,folder,plan=public.owner_folder(req['owner'],guard)
    require(owner['source']==record(public.__file__) and owner['file_engine_source']==record(engine.__file__)
        and owner['session']==config['session'] and owner['updates']==20 and owner['window_index']==0,
        'Real protocol11 owner_folder/source/session/plan constructor validation')
    CPU_report=read_json(run/'checkpoints/transient/p000000001/CPU/payload/report.json')
    public.verify_feature_window(CPU_report['actual_frontend_window_closure'],req['owner'],guard=guard)
    manifest=public.read_bound(req['manifest'],guard)
    require(manifest['owner']==req['owner'] and len(manifest['updates'])==20,'Real exact manifest owner20')
    reader=public.FeatureWindowReader(req['owner'],req['manifest'],guard=guard)
    for index,update in enumerate(plan['updates']):
        public._validate_update(req['owner'],public.read_bound(reader.update_record(index),guard),update,reader.store)
    store=engine._Store(folder,owner,guard,public.capacity_lock)
    actual=list(folder.iterdir());store.inventory()
    require(len(actual)==682 and sum(p.suffix=='.npy' for p in actual)==640 and sum(p.suffix=='.json' for p in actual)==42,
        'Exactly the genuine682 owned feature files before retirement')
    snapshot={p.name:public.record(p,guard) for p in actual}
    archive=out/'failed_owned_feature_metadata_archive.json'
    boundary=dict(kind='ASR18_truthful_abnormal_owner_TOOL1_and_all_captured_dead_feature_retirement_v1',
        actual_failure=req['failure'],actual_owner_TOOL_exit_code=1,actual_native_normal_exit0_claimed=False,
        native_exit_numeric_code_observed=False,all7_captured_identities_dead=True,preserved_checkpoint_updates=20,
        window_session=owner['session'],owner_sha256=req['owner']['sha256'])
    removed,archive_pin=engine._retire_files(store,archive=dict(kind='ASR18_failed_RUN06_exact_owned_feature_metadata_archive_v1',
        failure=req['failure'],owner=req['owner'],manifest=req['manifest'],boundary=boundary,
        source=record(__file__),engine_source=record(engine.__file__),complete_original_owned_file_hash_inventory=snapshot),evidence_path=archive)
    require(len(removed)==682 and not folder.exists(),'Exact failed feature owner retired after archive')
    require([record(p['path']) for p in protected]==before,'Saved weights metadata/PRE/nextdraw/journal/storage/durable untouched')
    result=dict(kind='ASR18_failed_RUN06_exact_owned_feature_retirement_report_v1',passed=True,producer=identity(),
        worker_source=record(__file__),parent_source=req['parent_source'],failure=req['failure'],owner=req['owner'],
        manifest=req['manifest'],preserved_metadata=archive_pin,removed_exact_owned_files=len(removed),
        complete_original_owned_hash_inventory_entries=len(snapshot),target_absent=True,CACHE_members=len(list(public.CACHE.iterdir())),
        actual_native_normal_exit0_claimed=False,native_exit_numeric_code_observed=False,
        preserved_checkpoint_updates=20,protected_metadata_unchanged=before,physical_feature_retirement_performed_here=True,
        no_checkpoint_journal_control_or_original_audio_deletion=True,wall_seconds=time.monotonic()-started)
    guard();write_json(out/'report.json',result);print(json.dumps({'passed':True,'report':record(out/'report.json')}),flush=True)

if __name__=='__main__':main()
