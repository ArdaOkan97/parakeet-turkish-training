"""Source-bound hard1 recovered checkpoint, never full-run or finite20 authority."""
import hashlib
from pathlib import Path
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from bpe_streaming_checkpoint_semantics_v2 import decode_scalar_tree,select,verify_saved_metadata
from asr18_full_run_serial_DEV_9223_v2 import checkpoint_format_view
from asr18_finite_BPE9223_checkpoint_evidence_v2 import checked_owner,checked_process,metadata_fingerprint,normalize_fingerprint,read_bound
ROOT=Path(r'E:\gpu_training_scratchpad');N=ROOT/'analysis/BPE9223_training_launch_v1'
KIND='ASR18_completed_hard1_recovered_checkpoint_evidence_v1'
SOURCE_SHA='b7cc39f45458cd516fcee9b97ecd5715e5df2df5242f5d219670efa269e46b95'
CONFIG_SHA='866a44bc7b7b13e9c87b8e6f716654659d9dfa1665d418becf14dd4a8a3fc20f'
PLAN_SHA='7110e796688b09923bff292d59b5c820921f8a72ba43e7845e534bbb28dd2ea5'


def same(a,b):return digest(a)==digest(b)
def integer(v,n):require(type(v) is int and v==n,'Exact nonboolean hard1 scalar');return v


def metadata_binding(outer_pin):
    """Small real metadata only; actual closure validation remains checked_binding."""
    outer=read_bound(outer_pin);overall=read_bound(outer['report'])
    native0=read_bound(overall['actual_native_process0']);fresh0=read_bound(overall['actual_fresh_recovery_process0'])
    native=read_bound(native0['report']);metadata=read_bound(native['checkpoint'])
    return {'kind':KIND,'optimizer_updates':1,'owner_tool_actual0':outer_pin,'overall_result':outer['report'],
        'configuration':outer['configuration'],'native_source':outer['source'],'native_report':native0['report'],
        'native_process0':overall['actual_native_process0'],'recovery_report':fresh0['report'],
        'recovery_process0':overall['actual_fresh_recovery_process0'],'checkpoint':native['checkpoint'],
        'payload':metadata['payload'],'PRE_save_fingerprint':native['PRE_save_fingerprint'],'persistence':native['persistence'],
        'next_draw':native['next_draw'],'journal_prefix':native['journal_prefix'],'sampler_state_sha256':native['sampler_sha256'],
        'identity':decode_scalar_tree(select(metadata['tree'],'identity'))}


def checked_binding(outer_pin):
    v=metadata_binding(outer_pin);source=v['native_source'];configpin=v['configuration'];config=read_bound(configpin)
    require(source==record(Path(__file__).resolve().parents[2]/'work/research/run_ASR18_BPE9223_hard_v3.py')
        and source['sha256']==SOURCE_SHA and configpin['sha256']==CONFIG_SHA and config['plan']['sha256']==PLAN_SHA,
        'Exact ROOT-reviewed hard source/config/whole full-row plan')
    require(config['kind']=='ASR18_BPE9223_hard_original_source_configuration_v1' and config['production_enabled'] is False
        and config['owner_validation_only'] is True and config['checkpoint_diagnostic_only'] is True
        and integer(config['vocabulary'],9223) and integer(config['native_updates'],1) and integer(config['window_count'],1)
        and config['harness_source']==source,'Hard qualification only; no source/clock reset or production adoption')
    overall=read_bound(v['overall_result']);require(overall['passed'] is True and overall['production_enabled'] is False
        and overall['arm']=='candidate' and integer(overall['actual_native_updates'],1) and overall['source']==source
        and overall['configuration']==configpin,'Actual source-bound hard overall0')
    folder=Path(v['overall_result']['path']).resolve().parent
    require(folder==N/'hard_CPU_native_actual_02' and Path(v['native_report']['path'])==folder/'train/training_result.json'
        and Path(v['recovery_report']['path'])==folder/'recover/recovery_result.json','Exact actual hard train/fresh namespaces')
    checked_owner(read_bound(outer_pin),source=source,configuration=configpin,report=v['overall_result'])
    native0=checked_process(read_bound(v['native_process0']),source=source,configuration=configpin,report=v['native_report'])
    fresh0=checked_process(read_bound(v['recovery_process0']),source=source,configuration=configpin,report=v['recovery_report'])
    require(native0['native']!=fresh0['native'],'Separate actual fresh hard restoration')
    native=read_bound(v['native_report']);fresh=read_bound(v['recovery_report']);shared=read_bound(overall['shared_features'])
    require(native['passed'] is True and native['production_enabled'] is False and native['production_source_eligible'] is False
        and integer(native['actual_optimizer_updates'],1) and len(native['history'])==1 and integer(native['history'][0]['update'],1)
        and native['configuration']==configpin and native['source']==source and native['arm']=='candidate'
        and fresh['passed'] is True and fresh['production_enabled'] is False and fresh['arm']=='candidate'
        and fresh['source']==source and integer(fresh['actual_native_updates'],0)==0
        and fresh['all207_threeAdam_all4RNG_sampler_nextdraw_exact'] is True,'Actual one hard native update plus exact zero-update fresh restoration')
    expected={'kind':'ASR18_original_source_BPE9223_hard_unqualified_checkpoint_identity_v1','config':configpin,'arm':'candidate',
        'shared_feature_owner':shared['owner'],'BPE9223_population':config['planning_arguments']['DATA_evidence']['BPE9223_population'],
        'row_migration_plan':config['row_migration_plan'],'production_enabled':False,'production_source_eligible':False}
    require(same(v['identity'],expected),'Actual hard checkpoint identity is not a production pointer')
    metadata=read_bound(v['checkpoint']);checkpoint_format_view(metadata)
    require(Path(v['payload']['path'])==Path(v['checkpoint']['path']).with_suffix('.tensors.bin')
        and record(v['payload']['path'])==v['payload'],'One complete hard checkpoint payload SHA before retirement')
    scalar=lambda *p:decode_scalar_tree(select(metadata['tree'],*p));sampler=scalar('sampler')
    semantic=verify_saved_metadata(metadata,identity=expected,optimizer_updates=1,sampler_state=sampler,journal_prefix=v['journal_prefix'],diagnostic_only=True)
    require(scalar('optimizer','kind')=='BPE9223_broad_full207_three_AdamW_mixed_fresh_clock_v1'
        and integer(scalar('optimizer','appended512_Adam_clock'),1) and scalar('optimizer','clocks')==[64251,2001,62251]
        and digest(sampler)==v['sampler_state_sha256'],'Full hard sampler/three mature clocks/new appended clock1')
    actual=metadata_fingerprint(metadata);pre=normalize_fingerprint(read_bound(v['PRE_save_fingerprint']))
    require(actual['state_sha256']==digest(actual['state'])==pre['state_sha256']==digest(pre['state']),'All854/215/207/all4RNG saved canonical PRE equality')
    persistence=read_bound(v['persistence'])
    require(persistence['kind']=='ASR18_actual_streamed_saved_state_matches_independent_pre_save_v1'
        and persistence['passed'] is True and persistence['checkpoint']==v['checkpoint'] and persistence['payload']==v['payload']
        and persistence['independent_pre_save_state_sha256']==pre['state_sha256']==persistence['saved_state_sha256']
        and integer(persistence['optimizer_updates'],1) and integer(persistence['weights_verified'],207)
        and integer(persistence['Adam_states_verified'],215) and integer(persistence['named_tensors_verified'],854)
        and persistence['sampler_state_sha256']==v['sampler_state_sha256'] and persistence['journal_prefix']==v['journal_prefix']
        and same(persistence['saved_semantic_state_verified'],semantic) and persistence['actual_GPU_restore_performed'] is False,
        'Actual saved-byte/PRE persistence remains separate from fresh GPU restore')
    nextdraw=read_bound(v['next_draw']);require(digest(nextdraw)==native['next_draw_sha256']
        and nextdraw['kind']=='ASR18_completed_hard_effective16_diagnostic_no_further_draw_packet_v1'
        and integer(nextdraw['optimizer_updates'],1) and nextdraw['sampler_state_sha256']==digest(sampler)
        and nextdraw['carried_replay_state_sha256']==digest(sampler['carried_replay'])
        and nextdraw['unchanged_original_fullpopulation_ancestor_sha256']==digest(sampler['continued_full_population_schedule'])
        and nextdraw['further_native_or_authoritative_draws_allowed'] is False and nextdraw['production_source_eligible'] is False
        and nextdraw['production_phase_advance_claimed'] is False,'Terminal hard diagnostic packet, no fabricated production next draw')
    plan=read_bound(config['plan']);require(len(plan['updates'])==1 and plan['updates'][0]['after_sampler_sha256']==digest(sampler),'Actual full16 hard plan ends at same sampler')
    hard=plan['updates'][0]['rows'][13]
    require(len(plan['updates'][0]['rows'])==16 and hard['source_id']=='kiraat:src01792-00990'
        and hard['encoder_frames']==1991 and hard['target_tokens']==len(hard['tr_metadata_labels'])==517
        and hard['declared_geometry']['clean16k_samples']==2547840,'Whole retained longest hard source/targets/geometry')
    prefix=v['journal_prefix'];branch=fresh['branch'];require(branch['kind']=='ASR18_preserved_tail_checkpoint_branch_v1'
        and branch['original_checkpoint_prefix']==prefix and branch['old_journal_modified'] is False and branch['tail_replayed_or_sampler_advanced'] is False
        and branch['branch_checkpoint_prefix']['offset']==prefix['offset'] and branch['branch_checkpoint_prefix']['sha256']==prefix['sha256']
        and Path(branch['branch_checkpoint_prefix']['path'])==folder/'recover/restored_journal_prefix.jsonl','Actual fresh hard prefix branch')
    require(type(prefix['offset']) is int and 0<=prefix['offset']<=16*1024**2 and Path(prefix['path']).resolve()==folder/'train/actual_commits.jsonl','Exact bounded hard committed journal')
    with Path(prefix['path']).open('rb') as stream:raw=stream.read(prefix['offset'])
    require(len(raw)==prefix['offset'] and hashlib.sha256(raw).hexdigest()==prefix['sha256'],'Actual original committed hard prefix SHA')
    return v,metadata,sampler
