"""Bounded protocol/frontend/cleanup fixture; no model, GPU, Adam or draw commit."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

P = Path(__file__).resolve().parents[2]; S = P / "outputs/parakeet_turkish"
ROOT = Path(r"E:\gpu_training_scratchpad"); AREA = ROOT / "analysis/ASR18_memory_diagnosis_v1"
CPU_PYTHON = ROOT / "scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe"
sys.path.insert(0,str(S))
from fastmss_feature_window_v1 import *
from fastmss_ordered_draw_binding_v1 import AugmentationDrawContext as Draw, FastMSSOrderedDrawBinding
from fastmss_train_augmentation_v1 import AugmentationPolicy, TrainWaveformAugmenter


def cpu_numpy():
    training_site=P/'work/training-env/Lib/site-packages'
    sys.path.insert(0,str(training_site))
    import numpy as np
    sys.path.remove(str(training_site))
    return np


def main(output):
    out = scoped(output); out.mkdir(parents=True,exist_ok=False)
    guard = lambda: None  # Entire process and each child run under pinned CPUguard3.
    np=cpu_numpy()
    checks = {}; child_resources=[]
    def check(name,value):
        require(bool(value),name);checks[name]=True
    def rejects(name,call):
        try:call()
        except (ValueError,TypeError,KeyError,FileNotFoundError):checks[name]=True
        else:raise AssertionError(name)
    source = AREA/'hard_U_rows_decode_01/decoded/report.json'
    original = read_bound(record(source),guard)
    check('two_full_hard_rows_available',len(original['rows'])==2)
    source_records=[v['PCM_cache'] for v in original['rows']]
    policy=S/'fastmss_broad_optin_v1.json'
    contexts=[];updates=[];raws=[]
    for i,item in enumerate(original['rows']):
        row=item['row'];raw=deepcopy(row['broad_original']);raw['BPE']={'target_ids':row['tr_metadata_labels']}
        raws.append(raw)
    for update in (1,2):
        draws=[{'ordinal':r['ordinal'],'epoch':update-1,'draw_index':(update-1)*2+i} for i,r in enumerate(raws)]
        ctx=[asdict(Draw(v['row']['source_id'],'tr',update-1,(update-1)*2+i)) for i,v in enumerate(original['rows'])]
        updates.append({'update':update,'batch_id':f'update_{update}','TR_draws':draws,
            'TR_identity':[{'ordinal':r['ordinal'],'source_id':r['unique_id'],'language':'tr'} for r in raws],
            'replay_rows':[],'contexts':ctx,'virtual_only':True,
            'expected_native_transaction_sha256':digest({'fixture_only':True,'update':update}),
            'expected_before_sampler_sha256':digest({'fixture_only_frontier':update-1}),
            'expected_after_sampler_sha256':digest({'fixture_only_frontier':update})})
    plan={'kind':'FastMSS_feature_window_transport_fixture_v1','diagnostic_only':True,'production_enabled':False,
        'phase':'weak_label_continued_training','actual_optimizer_updates':0,'actual_broad_draws_consumed':0,
        'actual_replay_frontiers_advanced':False,'maximum_feature_window_bytes':CAPACITY,
        'TR_rows':raws,'updates':updates,'augmentation_policy':record(policy)}
    plan_path=out/'plan.json';write_new(plan_path,plan);plan_binding=record(plan_path)
    frontend_base=Path((P/'work/base_model_path.txt').read_text().strip())
    frontend={'source':record(P/'work/training-env/Lib/site-packages/transformers/models/parakeet/feature_extraction_parakeet.py'),
        'base_path':str(frontend_base),'processor_config':record(frontend_base/'processor_config.json'),
        'arguments':FRONTEND_ARGS,'Torch':'2.7.1+cpu','torch_threads':4,
        'CPU_GPU_byte_equivalence_proof':record(AREA/'original_parent_frontend_compare_01/frontend_byte_equivalence.json')}
    actual_calls=[]
    class Huge:
        nbytes=CAPACITY
    def launch(name,script,arguments,*,expected=0):
        folder=out/(name+'_guard');log=out/(name+'.log')
        command=[str(CPU_PYTHON),'-X','utf8','-B',str(P/'work/research/ASR18_bounded_CPU_phase_v3.py'),
            '--folder',str(folder),'--seconds','120','--script',str(script),'--',*arguments]
        with log.open('x',encoding='utf-8') as stream:
            child=subprocess.Popen(command,cwd=P,env=os.environ.copy(),stdout=stream,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
            code=child.wait(timeout=130)
        require(code==expected,f'{name} actual exit {code}, see{log}')
        closure=read_bound(record(folder/'closure.json'),guard);child_resources.append(closure)
        check(name+'_actual_owned_exit',closure['actual_exit_code']==expected)
        actual_calls.append({'name':name,'handle_PID':child.pid,'actual_PID':closure['pid'],'actual_exit_code':code})
        return child
    owner_binding=None;closures=[];PCM_cleanup=[]
    for index in (0,1):
        request={'kind':'fixture_PCM_prepare','source_report':record(source),'plan':plan_binding,'update_index':index,
            'policy':str(policy),'job_id':out.name+f'_PCM_{index}','result':str(out/f'PCM_{index}_report.json')}
        request_path=out/f'PCM_{index}_request.json';write_new(request_path,request)
        pcm_child=launch(f'PCM_{index}',Path(__file__),['--prepare-request',str(request_path)])
        prepared=read_bound(record(out/f'PCM_{index}_report.json'),guard);transport=prepared['augmentation_transport']
        from fastmss_cpu_prepared_transport_v1 import observe_closed_child,cleanup_owned_cache
        pcm_closure=observe_closed_child(pcm_child,owner_record=transport['owner'],manifest_record=transport['manifest'],
            proof_path=out/f'PCM_{index}_actual_closure.json',guard=guard)
        if owner_binding is None:
            owner_binding=create_window(plan_binding,window_id=out.name,augmentation_identity=prepared['binding_identity'],frontend=frontend,guard=guard)
            temporary_owner,temporary_folder=owner_folder(owner_binding,guard)
            rejects('hardcapacity_rejected_before_large_allocation_or_write',lambda:save_array(temporary_owner,temporary_folder,'u00_r00_features.npy',Huge(),guard))
        require(prepared['binding_identity']==read_bound(owner_binding)['augmentation_identity'],'Prepared identity drift')
        feature_request={'kind':'FastMSS_CPU_feature_update_request_v1','owner':owner_binding,'update_index':index,
            'prior_closures':closures,'decoder_report':record(out/f'PCM_{index}_report.json'),'PCM_actual_closure':pcm_closure}
        feature_request_path=out/f'feature_{index}_request.json';write_new(feature_request_path,feature_request)
        feature_child=launch(f'feature_{index}',P/'work/research/produce_fastmss_feature_update_CPU_v1.py',
            ['--request',str(feature_request_path),'--output',str(out/f'feature_{index}_result.json')])
        proof=observe_feature_child(feature_child,owner_binding=owner_binding,index=index,
            proof_path=out/f'feature_{index}_actual_closure.json',guard=guard)
        closures.append(proof)
        closed=verify_feature_closure(proof,owner_binding,success=True,guard=guard)
        update_record=read_bound(closed['update_record'],guard)
        check(f'feature_{index}_prepared_payload_not_committed',not update_record['prepared_is_committed'] and
            not update_record['sampler_journal_frontier_advanced'])
        # The closed feature output preserves the whole original small PCM
        # manifest/owner/closure before a provisional/discard handoff removes it.
        boundary={'kind':'discard_uncommitted_preparation','batch_id':f'update_{index+1}',
            'owner_sha256':transport['owner']['sha256'],'feature_update':closed['update_record'],'feature_closure':proof}
        cleanup=cleanup_owned_cache(owner_record=transport['owner'],manifest_record=transport['manifest'],closure_record=pcm_closure,
            boundary=boundary,verify_boundary=lambda value:value==boundary and read_bound(closed['update_record'])['prepared_is_committed'] is False,guard=guard)
        PCM_cleanup.append(cleanup)
    rejects('missing_update_closure_rejected',lambda:finalize_window(owner_binding,closures[:-1],guard=guard))
    manifest_binding=finalize_window(owner_binding,closures,guard=guard);manifest=read_bound(manifest_binding)
    owner,folder=owner_folder(owner_binding,guard)
    check('hard512MiB_actual_file_bound',inventory(owner,folder)==manifest['aggregate_files_bytes']<CAPACITY)
    reader=FeatureWindowReader(owner_binding,manifest_binding,guard=guard)
    rejects('incomplete_feature_update_payload_rejected',lambda:reader.prepared_payload(0))
    expected=read_bound(record(AREA/'CPU_frontend_actual_02/report.json'))
    for index,update in enumerate(updates):
        for j,context in enumerate(update['contexts']):
            values=reader.take_row(index,j,context=Draw(**context),native_transaction_sha256=update['expected_native_transaction_sha256'],
                before_sampler_sha256=update['expected_before_sampler_sha256'])
            for key in ('input_features','attention_mask'):
                check(f'full_row_{index}_{j}_{key}_BYTEEXACT_original_CPU_GPU_proof',
                    hashlib.sha256(memoryview(values[key]).cast('B')).hexdigest()==expected['rows'][j]['inputs'][key]['tensor_bytes_sha256'])
            del values
        payload=reader.prepared_payload(index)
        check(f'prepared_payload_{index}_copied_exact',payload==read_bound(manifest['updates'][index]['update'])['prepared_payload'])
    bad=FeatureWindowReader(owner_binding,manifest_binding,guard=guard)
    rejects('out_of_order_feature_rejected',lambda:bad.take_row(0,1,context=Draw(**updates[0]['contexts'][1]),
        native_transaction_sha256=updates[0]['expected_native_transaction_sha256'],before_sampler_sha256=updates[0]['expected_before_sampler_sha256']))
    rejects('failed_reader_poisoned',lambda:bad.take_row(0,0,context=Draw(**updates[0]['contexts'][0]),
        native_transaction_sha256=updates[0]['expected_native_transaction_sha256'],before_sampler_sha256=updates[0]['expected_before_sampler_sha256']))
    bad=FeatureWindowReader(owner_binding,manifest_binding,guard=guard)
    rejects('wrong_native_transaction_rejected',lambda:bad.take_row(0,0,context=Draw(**updates[0]['contexts'][0]),
        native_transaction_sha256='0'*64,before_sampler_sha256=updates[0]['expected_before_sampler_sha256']))
    fixture_checkpoint=out/'fixture_NOT_NATIVE_checkpoint.json';write_new(fixture_checkpoint,{'fixture_only':True,'actual_Adam_updates':0})
    resume={'checkpoint':record(fixture_checkpoint),'committed_window_updates':1}
    rejects('blind_skip_rejected',lambda:FeatureWindowReader(owner_binding,manifest_binding,guard=guard,resume=resume))
    rejects('wrong_checkpoint_verifier_rejected',lambda:FeatureWindowReader(owner_binding,manifest_binding,guard=guard,resume=resume,verify_resume=lambda value:True))
    # Protocol test double only; NOT a native restore or committed sampler proof.
    expected_resume={'passed':True,'checkpoint':resume['checkpoint'],'feature_window_manifest':manifest_binding,
        'committed_window_updates':1,'committed_transaction_sha256':updates[0]['expected_native_transaction_sha256'],
        'committed_after_sampler_sha256':updates[0]['expected_after_sampler_sha256'],
        'next_before_sampler_sha256':updates[1]['expected_before_sampler_sha256'],'augmentation_identity':owner['augmentation_identity'],
        'native_checkpoint_state_verified':True,'actual_restored_sampler_verified':True,'committed_journal_prefix_verified':True}
    resumed=FeatureWindowReader(owner_binding,manifest_binding,guard=guard,resume=resume,verify_resume=lambda value:expected_resume)
    check('verified_resume_protocol_starts_next_complete_update',resumed.index==1 and resumed.row_index==0)
    values=resumed.take_row(1,0,context=Draw(**updates[1]['contexts'][0]),native_transaction_sha256=updates[1]['expected_native_transaction_sha256'],
        before_sampler_sha256=updates[1]['expected_before_sampler_sha256']);del values
    boundary={'kind':'coherent_checkpoint_boundary','owner_sha256':owner_binding['sha256'],'committed_window_updates':1,
        'window_manifest_sha256':manifest_binding['sha256'],'checkpoint':record(fixture_checkpoint)}
    rejects('intermediate_checkpoint_cannot_delete_future_features',lambda:cleanup_window(owner_binding,closures=closures,
        manifest_binding=manifest_binding,boundary=boundary,verify_boundary=lambda value:True,evidence_path=out/'forbidden_cleanup.json',guard=guard))
    rejects('existing_feature_file_rewrite_rejected',lambda:save_array(owner,folder,'u00_r00_features.npy',Huge(),guard))
    unknown=folder/'UNKNOWN_fixture_sentinel.txt';unknown.write_text('created only by this fixture')
    discard={'kind':'discard_uncommitted_feature_window','owner_sha256':owner_binding['sha256']}
    rejects('unknown_file_preserves_everything',lambda:cleanup_window(owner_binding,closures=closures,manifest_binding=manifest_binding,
        boundary=discard,verify_boundary=lambda value:value==discard,evidence_path=out/'unknown_cleanup.json',guard=guard))
    require(unknown.resolve().parent==folder and unknown.read_text()=='created only by this fixture','Fixture sentinel identity before unlink');unknown.unlink()
    cleanup=cleanup_window(owner_binding,closures=closures,manifest_binding=manifest_binding,boundary=discard,
        verify_boundary=lambda value:value==discard,evidence_path=out/'durable_feature_metadata.json',guard=guard)
    check('all_small_evidence_preserved_before_owned_feature_eviction',Path(cleanup['preserved_metadata']['path']).is_file() and not folder.exists())
    partial_owner=create_window(plan_binding,window_id=out.name+'_partial',augmentation_identity=owner['augmentation_identity'],frontend=frontend,guard=guard)
    partial_request=out/'partial_failure_request.json';write_new(partial_request,{'owner':partial_owner})
    failed_child=launch('intentional_partial_failure',Path(__file__),['--failure-request',str(partial_request)],expected=13)
    failed_closure=observe_feature_child(failed_child,owner_binding=partial_owner,index=0,proof_path=out/'partial_actual_closure.json',guard=guard)
    rejects('failed_feature_producer_cannot_be_admitted',lambda:verify_feature_closure(failed_closure,partial_owner,success=True,guard=guard))
    partial_discard={'kind':'discard_uncommitted_feature_window','owner_sha256':partial_owner['sha256']}
    rejects('missing_actual_failed_child_closure_preserves_partial',lambda:cleanup_window(partial_owner,closures=[],manifest_binding=None,
        boundary=partial_discard,verify_boundary=lambda value:True,evidence_path=out/'forbidden_partial_cleanup.json',guard=guard))
    partial_cleanup=cleanup_window(partial_owner,closures=[failed_closure],manifest_binding=None,boundary=partial_discard,
        verify_boundary=lambda value:value==partial_discard,evidence_path=out/'partial_failure_durable_evidence.json',guard=guard)
    check('actual_failed_partial_child_only_owned_cache_discarded',len(partial_cleanup['removed_exact_owned_files'])==3)
    check('original_full_PCM_files_unchanged',all(record(v['path'])==v for v in source_records))
    check('no_GPU_model_or_Adam_imported_in_coordinator','torch' not in sys.modules)
    write_new(out/'validation.json',{'passed':True,'meaningful_checks':len(checks),'checks':checks,'actual_child_calls':actual_calls,
        'child_resources':child_resources,'source':file_binding(__file__),'feature_helper':file_binding(S/'fastmss_feature_window_v1.py'),
        'frontend_producer':file_binding(P/'work/research/produce_fastmss_feature_update_CPU_v1.py'),
        'native_plan':plan_binding,'feature_manifest':manifest_binding,'feature_manifest_contents':manifest,'cleanup':cleanup,
        'PCM_provisional_handoff_cleanup':PCM_cleanup,'all_originals_preserved':True,'actual_optimizer_updates':0,
        'intentional_failed_feature_child_exit':13,'partial_feature_cleanup':partial_cleanup,
        'authoritative_draws_committed':0,'resume_native_verifier_test_double_only':True,'native_state_stability_claimed':False,
        'two_complete_full_hard_audio_frontends_byte_exact':True,'production_enabled':False})
    print(json.dumps({'passed':True,'checks':len(checks),'actual_optimizer_updates':0}),flush=True)


def prepare_child(request_path):
    np=cpu_numpy()
    from fastmss_cpu_prepared_transport_v1 import prepare_TR_transport
    request=json.loads(Path(request_path).read_bytes());original=read_bound(request['source_report'])
    plan=load_plan(request['plan'],lambda:None);update=plan['updates'][request['update_index']]
    augmenter=TrainWaveformAugmenter(AugmentationPolicy.from_json(request['policy']),guard=lambda:None)
    binding=FastMSSOrderedDrawBinding(augmenter,policy_path=request['policy'],adapter_path=S/'fastmss_train_augmentation_v1.py')
    outputs=[]
    for source,draw in zip(original['rows'],update['TR_draws']):
        row=deepcopy(source['row']);row.update(augmentation_role='student_tr',durable_draw=draw)
        outputs.append({**source,'row':row})
    contexts=[Draw(**value) for value in update['contexts']]
    def read(row):
        item=next(v for v in outputs if v['row']['source_id']==row['source_id'])
        require(record(item['PCM_cache']['path'])==item['PCM_cache'],'Original full hard PCM changed')
        return np.load(item['PCM_cache']['path'],allow_pickle=False)
    prepared=prepare_TR_transport(binding,[v['row'] for v in outputs],contexts,batch_id=update['batch_id'],
        job_id=request['job_id'],clean_reader=read,guard=lambda:None)
    write_new(Path(request['result']),{'passed':True,'rows':outputs,'augmentation_transport':prepared,'binding_identity':binding.identity,
        'fixture_reuses_original_full_PCM_only':True,'sampler_or_journal_advanced':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output');parser.add_argument('--prepare-request');parser.add_argument('--failure-request')
    args=parser.parse_args()
    if args.failure_request:
        np=cpu_numpy()
        request=json.loads(Path(args.failure_request).read_bytes())
        owner,folder,update,active=begin_update(request['owner'],0,[],lambda:None)
        save_array(owner,folder,'u00_r00_features.npy',np.zeros((1,2,128),dtype=np.float32),lambda:None)
        raise SystemExit(13)
    prepare_child(args.prepare_request) if args.prepare_request else main(args.output)
