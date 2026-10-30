"""Narrow hard1 retirement; archive first, existing exact engine deletion only.

No production pointer/discard fiction. Originals/checkpoints/journals and failed
hard-window outputs are outside the one literal source-owned transient folder.
"""
import argparse,json,os,stat,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
import fastmss_BPE9223_feature_protocol_v5 as public
import asr18_completed_hard_checkpoint_evidence_v1 as finite
from fastmss_BPE9223_cpu_common_v5 import environment,resource_guard
from fastmss_owned_process_closure_v2 import verify
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
N=public.ROOT/'analysis/BPE9223_training_launch_v1'
OWNED=public.CACHE/'hard_01'
CPU_REPORT=N/'hard_CPU_window_actual_02/report.json'
require(record(finite.__file__)['sha256']=='dae6ec0d63bb52de32bd858eddc3d6fde1ac215446d0a06578cee4b5427b7571','Exact frozen hard checkpoint evidence helper')


def inventory(folder):
    require(folder.resolve()==OWNED.resolve() and not folder.is_symlink(),'Only literal completed_hard1 transient directory')
    result=[]
    for path in sorted(folder.iterdir()):
        s=path.stat(follow_symlinks=False)
        require(stat.S_ISREG(s.st_mode) and not s.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT,'Ordinary exact owned files only')
        result.append({'name':path.name,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'device':s.st_dev,'inode':s.st_ino})
    require(len(result)==36 and sum(x['name'].endswith('.npy') for x in result)==32,'Exact1/16row/32array/4JSON inventory')
    return result


def prepare(packet_path,folder):
    environment();resource_guard();folder=Path(folder).resolve()
    require(folder.is_relative_to(N.resolve()) and not folder.exists(),'Fresh E proposal/archive namespace')
    packet=finite.metadata_binding(record(packet_path));cpu=public.read_bound(record(CPU_REPORT),resource_guard)
    owner=public.read_bound(cpu['owner'],resource_guard);manifest=public.read_bound(cpu['manifest'],resource_guard)
    require(packet['kind']==finite.KIND and packet['optimizer_updates']==1 and cpu['updates']==1 and cpu['full_rows']==16
        and cpu['individual_array_raw_hashes']==32 and Path(cpu['owner']['path']).resolve()==OWNED/'owner.json'
        and owner['source']==record(public.__file__) and manifest['owner']==cpu['owner'],'Exact completed hard1 owner/manifest/source')
    rows=[row for part in manifest['updates'] for row in public.read_bound(part['update'],resource_guard)['rows']]
    arrays=[row[key] for row in rows for key in ('input_features','attention_mask')]
    require(len(rows)==16 and len(arrays)==32 and len({x['file']['path'] for x in arrays})==32,'Complete ordered rows/array hash metadata')
    values=inventory(OWNED);require({x['name'] for x in values}==set(owner['owned_names']),'No unknown or missing transient member')
    folder.mkdir();proposal={'kind':'ASR18_hard1_exact_owned_feature_retirement_proposal_v1','owner':'/root/cpu_data_preparation',
        'source':record(__file__),'evidence_source':record(finite.__file__),'finite_packet':record(packet_path),'CPU_report':record(CPU_REPORT),'window_owner':cpu['owner'],
        'manifest':cpu['manifest'],'frontend_closure':cpu['actual_frontend_window_closure'],'inventory':values,'array_hash_metadata':arrays,
        'rows':16,'arrays':32,'archive_path':str(folder/'complete_window_metadata_archive.json'),'result_path':str(folder/'retirement_result.json'),
        'protected_originals_checkpoints_journals_untouched':True,'prepare_only_no_feature_or_checkpoint_payload_hash_or_deletion':True}
    public.write_new(folder/'proposal.json',proposal);return record(folder/'proposal.json')


def retire(proposal_path,ROOT_review_path):
    environment();guard=resource_guard;guard();pin=record(proposal_path);proposal=public.read_bound(pin,guard);review=public.read_bound(record(ROOT_review_path),guard)
    require(review['kind']=='ROOT_hard1_exact_owned_feature_retirement_review_v1' and review['accepted'] is True and review['reviewer']=='/root'
        and review['proposal']==pin and review['source']==record(__file__) and review['window_owner']==proposal['window_owner']
        and review['inventory_sha256']==digest(proposal['inventory']),'Exact ROOT source+owner+inventory retirement review')
    require(proposal['source']==record(__file__) and proposal['CPU_report']==record(CPU_REPORT) and inventory(OWNED)==proposal['inventory'],'Unchanged reviewed literal inventory')
    packet,metadata,sampler=finite.checked_binding(proposal['finite_packet']);cpu=public.read_bound(proposal['CPU_report'],guard)
    overall=public.read_bound(packet['overall_result'],guard);shared=public.read_bound(overall['shared_features'],guard)
    cpupin=overall['actual_CPU_process0'];cpuproof=public.read_bound(cpupin,guard)
    verify(cpuproof,kind='ASR18_owned_CPU_process_completion_v1',owner='/root/memory_diagnosis',report=proposal['CPU_report'],guard=guard)
    require(cpuproof['parent_source']==packet['native_source'] and shared['actual_CPU_process0']==cpupin
        and shared['CPU_report']==proposal['CPU_report'] and shared['owner']==proposal['window_owner']==cpu['owner']
        and shared['manifest']==proposal['manifest']==cpu['manifest'] and packet['identity']['shared_feature_owner']==cpu['owner'], 'Actual finite native/fresh/CPU source-owner lifetime joins')
    owner,folder,plan=public.owner_folder(proposal['window_owner'],guard)
    require(folder==OWNED.resolve() and owner['updates']==1 and owner['owner_validation_only'] is True and packet['sampler_state_sha256']==plan['final_sampler_sha256'],'The actual hard1 native commit consumed this exact complete CPU window')
    public.verify_feature_window(proposal['frontend_closure'],proposal['window_owner'],guard=guard)
    ref=cpu['reference_actual_aggregate0'];public.verify_reference_window(ref['report'],ref['actual0'],contract_pin=owner['source_contract'],guard=guard)
    reader=public.FeatureWindowReader(proposal['window_owner'],proposal['manifest'],guard=guard);arrays=[]
    for index,update in enumerate(plan['updates']):
        item=public.read_bound(reader.update_record(index),guard);public._validate_update(proposal['window_owner'],item,update,reader.store)
        arrays.extend(row[key] for row in item['rows'] for key in ('input_features','attention_mask'))
    require(arrays==proposal['array_hash_metadata'] and len(arrays)==32 and inventory(folder)==proposal['inventory'],'Every actual feature/mask hash and reviewed inventory remains exact')
    archive=Path(proposal['archive_path']).resolve();result=Path(proposal['result_path']).resolve()
    require(archive.parent==Path(proposal_path).resolve().parent==result.parent and archive.parent.is_relative_to(N.resolve())
        and not archive.exists() and not result.exists(),'Fresh durable metadata outside cache, no overwrite')
    boundary={'kind':'ASR18_completed_hard1_recovered_window_retirement_boundary_v1','finite_packet':proposal['finite_packet'],
        'actual_CPU_process0':cpupin,'actual_native_process0':packet['native_process0'],'actual_fresh_process0':packet['recovery_process0'],'actual_owner_TOOL0':packet['owner_tool_actual0'],'committed_hard_updates':1,'production_pointer_or_discard_claimed':False}
    removed,preserved=public.engine._retire_files(reader.store,archive={'kind':'ASR18_completed_hard1_full_window_metadata_archive_v1',
        'proposal':pin,'ROOT_review':record(ROOT_review_path),'boundary':boundary,'owner_binding':proposal['window_owner'],
        'manifest_binding':proposal['manifest'],'all32_array_hash_metadata':arrays,'all16_ordered_row_metadata_preserved_in_update_JSON':True},evidence_path=archive)
    require(len(removed)==36 and not OWNED.exists(),'Only complete exact reviewed transient inventory retired')
    public.write_new(result,{'kind':'ASR18_completed_hard1_exact_owned_retirement_v1','passed':True,'proposal':pin,'archive':preserved,
        'removed_exact_owned_files':removed,'original_audio_checkpoints_journals_untouched':True,'new_draws_or_native_updates':0});return record(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-packet');p.add_argument('--folder');p.add_argument('--proposal');p.add_argument('--ROOT-review');a=p.parse_args()
    require(bool(a.prepare_packet and a.folder)!=bool(a.proposal and a.ROOT_review),'Choose prepare-only or exact reviewed retirement')
    print(json.dumps(prepare(a.prepare_packet,a.folder) if a.prepare_packet else retire(a.proposal,a.ROOT_review)),flush=True)
