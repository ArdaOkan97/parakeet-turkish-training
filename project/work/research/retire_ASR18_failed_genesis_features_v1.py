"""Archive-first retirement of one failed genesis' uncommitted CPU window.

The native exit code was not independently persisted. This adapter requires the
real outer TOOL1, captured dead identities, zero-commit marker and genuine CPU0;
it never represents this as a successful commit or production checkpoint.
"""
import argparse, json, stat, sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
import fastmss_BPE9223_feature_protocol_v3 as public
from asr18_finite_BPE9223_checkpoint_evidence_v2 import identity_dead
from fastmss_BPE9223_cpu_common_v3 import environment, resource_guard
from fastmss_owned_process_closure_v2 import verify
from bpe_broad_audio_v1 import record, require
from bpe_broad_production_state_v1 import digest
N=public.ROOT/'analysis/BPE9223_training_launch_v1'
OLD_RUN=public.ROOT/'runs/BPE9223_weak2_clean1_20261018_01'
FAILED=N/'full_training_genesis_actual_01'
MARKER=FAILED/'zero_authoritative_commit_marker_v1.json'
MARKER_SHA='6023484852bb881e9da36482f936a5607400b78768406e8fd43328bd1d011cdc'
OUTER_SHA='50dee1a01c38cdcde05adf01e9e1b4cb39582734d7480e6145502958dce32b27'
OWNED=public.CACHE/'p000000001'
CPU_REPORT=OLD_RUN/'checkpoints/transient/p000000001/CPU/payload/report.json'


def checked_failure(guard):
    pin=record(MARKER);require(pin['sha256']==MARKER_SHA,'Exact sealed zero-authoritative-commit marker')
    m=public.read_bound(pin,guard);f=public.read_bound(m['outer_TOOL1'],guard)
    require(m['kind']=='ASR18_BPE9223_failed_genesis_zero_authoritative_commit_marker_v1' and m['owner']=='/root/memory_diagnosis'
        and m['outer_TOOL1']==m['native_failure_and_captured_dead'] and m['outer_TOOL1']['sha256']==OUTER_SHA
        and f['kind']=='ASR18_BPE9223_actual_failed_genesis_zero_commit_evidence_v1' and f['owner']=='/root/memory_diagnosis'
        and all(type(f[k]) is int and f[k]==1 for k in ('actual_tool_exit_code','actual_exec_tool_exit_code'))
        and f['actual_tool_chunk_id']=='d188a3' and f['owned_tool_session']==84986
        and f['native_exit_code_not_independently_persisted'] is True and f['production_success_claimed'] is False,
        'Actual failed owner TOOL1, no invented native exit or success')
    require(all(type(m[k]) is int and m[k]==0 for k in ('authoritative_optimizer_updates','authoritative_TR_draws','authoritative_replay_updates','checkpoint_writes','normal_publications'))
        and type(f['zero_optimizer_updates']) is int and f['zero_optimizer_updates']==0 and f['zero_authoritative_commits'] is True
        and m['latest_checkpoint'] is None and m['native_worker_ready_existed'] is False and m['native_command_existed'] is False
        and f['all_captured_identities_dead'] is True and len(f['all_captured_identities'])==4
        and f['producer'] in f['all_captured_identities'],'Typed zero authoritative updates/draws/replay/publication; all captured owners')
    for item in f['all_captured_identities']:require(identity_dead(item) is True,'Captured old owner/launcher/shim/native actually closed')
    for key in ('runtime_source','parent_source','configuration','worker_log'):public.read_bound(m[key],guard) if key=='configuration' else require(record(m[key]['path'])==m[key],'Source/log pin unchanged')
    require(m['parent_source']==f['parent_source'] and m['configuration']==f['configuration'] and m['worker_log']==f['native_worker_log']
        and m['runtime_source']['sha256']=='f4c71c31e6c0de063e32cc6996f8272a4c440b9d4324eedc314252c710cd375a'
        and m['parent_source']['sha256']=='78d7d97a4719a7d95bd54ea6060f5339f7643daa451eb7c222b6529bdc6f82d8','Exact failed runtime3/parent2/config/source joins')
    boot=public.read_bound(f['native_boot'],guard);g=public.read_bound(f['owner_guard'],guard)
    require(boot['producer'] in f['all_captured_identities'] and boot['source']==m['runtime_source'] and boot['parent_source']==m['parent_source']
        and g['producer']==f['producer'] and g['source']==m['parent_source'] and g['monitor_clean'] is True and g['monitor_joined'] is True
        and type(g['samples']) is int and g['samples']>=3,'Actual boot identity and clean joined outer watchdog')
    native=public.read_bound(boot['configuration'],guard);log=Path(m['worker_log']['path']).read_text(encoding='utf-8')
    require('collect_GPU_timeline' not in native and "KeyError: 'collect_GPU_timeline'" in log
        and not (FAILED/'native_000001/worker_ready.json').exists() and not (FAILED/'native_000001/command.json').exists(),'Pinned pre-ready/command startup failure')
    journal=m['registered_authoritative_journal'];state=public.read_bound(m['storage_state'],guard)
    require(journal==f['registered_journal']==record(OLD_RUN/'journals/commits_000000.jsonl') and journal['bytes']==0
        and m['storage_state']==f['storage_state'] and all(type(state[k]) is int and state[k]==0 for k in ('publications','normal_publications','emergency_publications'))
        and state['latest'] is None and state['catalog']['publications']==0,'Unchanged registered zero journal and no checkpoint publication')
    require(m['CPU_report']==f['CPU_report']==record(CPU_REPORT) and m['CPU_actual0']==f['CPU_actual_process0']
        and m['feature_owner']==f['prepared_feature_owner'] and m['feature_manifest']==f['prepared_feature_manifest'],'Failed source binds this exact prepared CPU window')
    cpu=public.read_bound(m['CPU_report'],guard);proof=public.read_bound(m['CPU_actual0'],guard)
    verify(proof,kind='ASR18_owned_CPU_process_completion_v1',owner='/root/memory_diagnosis',report=m['CPU_report'],guard=guard)
    require(proof['parent_source']==m['parent_source'] and proof['observer']==f['producer'] and cpu['owner']==m['feature_owner']
        and cpu['manifest']==m['feature_manifest'] and cpu['actual_frontend_window_closure']==f['prepared_feature_frontend_closure']
        and cpu['updates']==20 and cpu['full_rows']==320 and cpu['individual_array_raw_hashes']==640 and cpu['real_actual_child0_count']==2
        and cpu['actual_optimizer_updates']==0,'Genuine CPU aggregate0 and exactly 20 prepared, zero native-committed updates')
    return m,f,cpu


def inventory(folder):
    require(folder.resolve()==OWNED.resolve() and not folder.is_symlink(),'Only literal failed-genesis transient directory')
    result=[]
    for path in sorted(folder.iterdir()):
        s=path.stat(follow_symlinks=False)
        require(stat.S_ISREG(s.st_mode) and not s.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT,'Ordinary exact owned files only')
        result.append({'name':path.name,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'device':s.st_dev,'inode':s.st_ino})
    require(len(result)==682 and sum(x['name'].endswith('.npy') for x in result)==640,'Exact20/320row/640array/42JSON inventory')
    return result


def prepare(folder):
    environment();resource_guard();folder=Path(folder).resolve()
    require(folder.is_relative_to(N.resolve()) and not folder.exists(),'Fresh E proposal/archive namespace')
    m,f,cpu=checked_failure(resource_guard);owner=public.read_bound(cpu['owner'],resource_guard);manifest=public.read_bound(cpu['manifest'],resource_guard)
    require(Path(cpu['owner']['path']).resolve()==OWNED/'owner.json' and owner['source']==record(public.__file__) and manifest['owner']==cpu['owner'],'Exact failed owner/manifest/source')
    rows=[row for part in manifest['updates'] for row in public.read_bound(part['update'],resource_guard)['rows']]
    arrays=[row[key] for row in rows for key in ('input_features','attention_mask')];values=inventory(OWNED)
    require(len(rows)==320 and len(arrays)==640 and len({x['file']['path'] for x in arrays})==640 and {x['name'] for x in values}==set(owner['owned_names']),'Complete ordered rows, array metadata and owned inventory')
    folder.mkdir();proposal={'kind':'ASR18_failed_genesis_exact_owned_feature_retirement_proposal_v1','owner':'/root/cpu_data_preparation',
        'source':record(__file__),'failed_zero_commit_marker':record(MARKER),'failed_owner_TOOL1':m['outer_TOOL1'],'CPU_report':m['CPU_report'],'window_owner':cpu['owner'],
        'manifest':cpu['manifest'],'frontend_closure':cpu['actual_frontend_window_closure'],'inventory':values,'array_hash_metadata':arrays,'rows':320,'arrays':640,
        'archive_path':str(folder/'complete_window_metadata_archive.json'),'result_path':str(folder/'retirement_result.json'),
        'protected_originals_checkpoints_journals_untouched':True,'prepare_only_no_feature_or_checkpoint_payload_hash_or_deletion':True}
    public.write_new(folder/'proposal.json',proposal);return record(folder/'proposal.json')


def retire(proposal_path,ROOT_review_path):
    environment();guard=resource_guard;guard();pin=record(proposal_path);proposal=public.read_bound(pin,guard);review=public.read_bound(record(ROOT_review_path),guard)
    require(review['kind']=='ROOT_failed_genesis_exact_owned_feature_retirement_review_v1' and review['accepted'] is True and review['reviewer']=='/root'
        and review['proposal']==pin and review['source']==record(__file__) and review['window_owner']==proposal['window_owner']
        and review['inventory_sha256']==digest(proposal['inventory']),'Exact ROOT source+owner+inventory retirement review')
    require(proposal['source']==record(__file__) and proposal['failed_zero_commit_marker']==record(MARKER) and inventory(OWNED)==proposal['inventory'],'Unchanged reviewed failure and literal inventory')
    m,f,cpu=checked_failure(guard);require(proposal['CPU_report']==m['CPU_report'] and proposal['window_owner']==cpu['owner'] and proposal['manifest']==cpu['manifest'],'Exact reviewed failure/CPU joins')
    owner,folder,plan=public.owner_folder(proposal['window_owner'],guard)
    require(folder==OWNED.resolve() and owner['updates']==20 and owner['owner_validation_only'] is False and type(plan['initial_committed_updates']) is int
        and plan['initial_committed_updates']==0 and plan['actual_optimizer_updates']==0,'Prepared production window begins at zero; no committed-window claim')
    public.verify_feature_window(proposal['frontend_closure'],proposal['window_owner'],guard=guard)
    ref=cpu['reference_actual_aggregate0'];public.verify_reference_window(ref['report'],ref['actual0'],contract_pin=owner['source_contract'],guard=guard)
    reader=public.FeatureWindowReader(proposal['window_owner'],proposal['manifest'],guard=guard);arrays=[]
    for index,update in enumerate(plan['updates']):
        item=public.read_bound(reader.update_record(index),guard);public._validate_update(proposal['window_owner'],item,update,reader.store)
        arrays.extend(row[key] for row in item['rows'] for key in ('input_features','attention_mask'))
    require(arrays==proposal['array_hash_metadata'] and len(arrays)==640 and inventory(folder)==proposal['inventory'],'Every actual feature/mask hash and reviewed inventory exact')
    archive=Path(proposal['archive_path']).resolve();result=Path(proposal['result_path']).resolve()
    require(archive.parent==Path(proposal_path).resolve().parent==result.parent and archive.parent.is_relative_to(N.resolve()) and not archive.exists() and not result.exists(),'Fresh durable metadata outside cache, no overwrite')
    boundary={'kind':'ASR18_failed_genesis_uncommitted_window_retirement_boundary_v1','failed_zero_commit_marker':record(MARKER),'actual_owner_TOOL1':m['outer_TOOL1'],
        'actual_CPU_process0':m['CPU_actual0'],'authoritative_optimizer_updates':0,'authoritative_commits':0,'native_exit_code_not_independently_persisted':True,
        'successful_commit_or_production_pointer_claimed':False}
    removed,preserved=public.engine._retire_files(reader.store,archive={'kind':'ASR18_failed_genesis_full_window_metadata_archive_v1','proposal':pin,'ROOT_review':record(ROOT_review_path),
        'boundary':boundary,'owner_binding':proposal['window_owner'],'manifest_binding':proposal['manifest'],'all640_array_hash_metadata':arrays,'all320_ordered_row_metadata_preserved_in_update_JSON':True},evidence_path=archive)
    require(len(removed)==682 and not OWNED.exists(),'Only complete exact reviewed transient inventory retired')
    public.write_new(result,{'kind':'ASR18_failed_genesis_exact_owned_retirement_v1','passed':True,'proposal':pin,'archive':preserved,'removed_exact_owned_files':removed,
        'original_audio_checkpoints_journals_untouched':True,'new_draws_or_native_updates':0,'failed_uncommitted_window':True});return record(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-folder');p.add_argument('--proposal');p.add_argument('--ROOT-review');a=p.parse_args()
    require(bool(a.prepare_folder)!=bool(a.proposal and a.ROOT_review),'Choose prepare-only or exact reviewed retirement')
    print(json.dumps(prepare(a.prepare_folder) if a.prepare_folder else retire(a.proposal,a.ROOT_review)),flush=True)
