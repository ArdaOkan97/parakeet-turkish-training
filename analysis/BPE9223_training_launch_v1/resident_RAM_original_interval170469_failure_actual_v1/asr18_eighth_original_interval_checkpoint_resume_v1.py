"""INACTIVE typed source31 interval failure -> source32 saved170400 continuation.
Compact metadata and the preserved69 COPY only; original journal is STAT-only.
No model, audio, NumPy, restore, copy, native, sampler or kernel-credit operation.
"""
import copy,json
from pathlib import Path
import psutil
from atomic_io_retry_v1 import read_json
from bpe_broad_audio_v1 import record,require
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish'
F=Path(__file__).resolve().parent
BASE=F.parent/'resident_RAM_recurrent_metadata_slowdown_fix_v1/production_configuration_source31_v1/configuration.json'
CLOSURE=F/'actual_failure_closure.json'
ROOT_FAILURE=F/'ROOT_source31_original_interval170469_saved170400_compact_failure_and_candidate_review_v1.json'
ROOT_CPU=F/'ROOT_eighth_alias9_CPU13_actual02_original_ingress_frontend_CPU_review_v1.json'
ROOT_TRANSPORT=F/'ROOT_CPU13_overlay_import_only_transport_compatibility_successor_v1.json'
CERTIFICATE=F/'eighth_original_full_clip_geometry_certificate_v1.json'
OPERATIONAL_FIELDS=('original_clip_geometry_overlay','prior_resident_RAM_original_interval_failure',
    'resident_RAM_resource_scheduling_policy','resident_RAM_packet_credit_policy')


def exact_dead(who):
    try:return abs(psutil.Process(who['pid']).create_time()-who['create_time'])>=.001
    except psutil.NoSuchProcess:return True


def checked_eighth_original_policy(base):
    old=read_json(BASE);restored=copy.deepcopy(base)
    for key in OPERATIONAL_FIELDS:restored[key]=copy.deepcopy(old[key])
    require(restored==old,'Only four exact eighth-file operational fields differ from proper source31 base')
    overlay=base['original_clip_geometry_overlay'];proof=record(ROOT_CPU)
    require(proof['sha256']=='c482dd1eb7e2cc1353d9a1fbbb7405ef3098cc244ec49362db4bb07695a4d34a'
        and overlay=={'prior_overlay':old['original_clip_geometry_overlay'],
            'source':record(S/'fastmss_original_clip_geometry_overlay_v9.py'),
            'certificate':record(CERTIFICATE),'ROOT_review':proof},
        'Exact corrected alias9 eighth certificate and genuine completed ingress/frontend ROOT gate')
    review=read_json(ROOT_CPU)
    require(review['accepted']is True and review['reviewer']=='/root'
        and review['scope']=='ASR18_source_bound_eighth_original_full_clip_geometry_overlay_v1'
        and review['overlay_source']==overlay['source']and review['certificate']==overlay['certificate']
        and review['CPU_source']==record(S/'fastmss_resident_feature_stream_v13.py')
        and record(review['actual_CPU_control_report']['path'])==review['actual_CPU_control_report']
        and review['all_exact_metadata_identities_dead']is True,
        'Actual full99, original frontend/mask/finite, strict negatives and exact closure accepted; no new transport claim')
    resource=copy.deepcopy(old['resident_RAM_resource_scheduling_policy'])
    resource['CPU_protocol_source']=record(S/'fastmss_resident_feature_stream_v13.py')
    credit=copy.deepcopy(old['resident_RAM_packet_credit_policy'])
    credit['actual_CPU_control_review']=record(ROOT_TRANSPORT)
    require(base['resident_RAM_resource_scheduling_policy']==resource
        and base['resident_RAM_packet_credit_policy']==credit
        and record(ROOT_TRANSPORT)['sha256']=='7b1f87b098b0248f183faf348d3328312858b9bf77139243e95c6559371c2e1b',
        'Only CPU13 provenance and genuine derived import-only transport successor; all original byte/math/225/MAX2 values exact')
    require(base['prior_resident_RAM_original_interval_failure']==record(CLOSURE),
        'Exact actual source31 failed owner, no historical failure alias')
    from fastmss_resident_feature_stream_v13 import checked_resource_profile,checked_packet_credit_policy
    checked_resource_profile(base);checked_packet_credit_policy(base)
    return overlay


def checked_eighth_original_resume(pin,*,base,folder,pointer_pin):
    checked_eighth_original_policy(base);folder=Path(folder)
    require(pin==base['prior_resident_RAM_original_interval_failure']==record(CLOSURE)
        and pin['sha256']=='e2e1e2aadf1cc423067a24bba0531f5bad67995e6c54031ab882dcdcddac215e',
        'Exact sealed actual170469 failure, never intentional stop or normal0')
    value=read_json(pin['path']);e=value['evidence']
    require(value['kind']=='ASR18_source31_original_interval170469_saved170400_actual_failure_closure_v1'
        and value['actual_outer_owned_session']==2901 and value['actual_outer_TOOL_exit']==1
        and value['closed_session_never_consume_again']is True
        and value['separate_native_CPU_wrapper_numeric_OS_exits']=='UNKNOWN; not inferred from outerTOOL1 or absence',
        'Actual TOOL1 and honest unknown separate OS exits; closed owner cannot be consumed or reused')
    root_pin=record(ROOT_FAILURE)
    require(root_pin['sha256']=='b2762e2dc58294ca002119e97ec64e32e172d84778be0428335d61d35b433ed0'
        and read_json(ROOT_FAILURE)['accepted']is True,'Closed independent actual failure/candidate ROOT acceptance')
    compact=('actual_failure','actual_CPU_failure','originalMonitor_completion','private_metadata_worker_completion',
        'CPU_ready','boot','runtime','current_child','saved_pointer','saved_persistence','owned2901_actual_TOOL')
    for key in compact:require(record(e[key]['path'])==e[key],'Exact compact failed source31 evidence '+key)
    failure=read_json(e['actual_failure']['path']);cpu=read_json(e['actual_CPU_failure']['path'])
    monitor=read_json(e['originalMonitor_completion']['path']);private=read_json(e['private_metadata_worker_completion']['path'])
    require(failure==value['failure']and failure['actual_optimizer_updates']==170469 and failure['last_checkpoint']==170400
        and failure['error']=="ValueError('Never release credit from errored transport')"
        and cpu==value['CPU_failure']and cpu['completed_updates']==2469
        and cpu['message']=="ValueError('Original interval beyond source unresolved')"
        and monitor==value['originalMonitor']and monitor['joined']is True and monitor['clean']is True and monitor['error']is None
        and private['joined']is True and private['worker_connections_closed_on_worker']is True,
        'Actual original interval failure with clean original Monitor and closed scalar worker, not an added-buffer denial')
    old_driver=record(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v31.py')
    require(read_json(e['boot']['path'])['source']==old_driver
        and read_json(e['CPU_ready']['path'])['source']['protocol']==record(S/'fastmss_resident_feature_stream_v11.py')
        and Path(e['runtime']['path'])==folder/'invocations/owner_000023/configuration.json',
        'Failed historical source31/CPU11 remains exact while next source uses CPU13')
    deaths=value['known_exact_production_deaths']
    expected=[(2456,1792752131.0863705),(41672,1792752131.0791),(34044,1792752130.7049348),
        (38964,1792753467.9316857),(24620,1792753467.9153557)]
    require([(x['identity']['pid'],x['identity']['create_time'])for x in deaths]==expected
        and all(x['exact_dead']is True and x['separate_numeric_OS_exit']=='UNKNOWN'
            and exact_dead(x['identity'])for x in deaths),'All five exact old runtime identities dead before new lease/boot/GPU writes')
    require(e['saved_pointer']==pointer_pin and pointer_pin==value['saved_full_checkpoint']['pointer']
        and pointer_pin['sha256']=='d05d494e5618e625fedfd54806c2e50c289c96a6cc0c99720596e55de0466aa5',
        'Genuine saved170400 pointer only')
    pointer=read_json(pointer_pin['path']);persistence=read_json(e['saved_persistence']['path'])
    require(pointer['committed_updates']==170400 and read_json(folder/'latest_checkpoint.json')['pointer']==pointer_pin
        and persistence['passed']is True and persistence['optimizer_updates']==170400
        and persistence['named_tensors_verified']==854 and persistence['weights_verified']==207
        and persistence['Adam_states_verified']==215 and persistence['saved_semantic_state_verified']['passed']is True
        and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256'],
        'Full854/207/215 semantic PRE-SAVED publication; actual allstate restore remains required before native')
    journal=value['journal'];saved=journal['saved_prefix'];stat=journal['original_STAT_only'];tail=journal['preserved_suffix']
    require(saved==pointer['journal_prefix']==persistence['journal_prefix']and saved['offset']==344297457
        and saved['sha256']=='20d7e414019f2963b6f31d7e09399cb684255f4fa94e2529319188d450393dfe'
        and Path(saved['path'])==folder/'journals/commits_000023.jsonl'
        and stat['path']==saved['path']and stat['bytes']==344433090 and stat['mtime_ns']==1792757832000000000
        and stat['bytes']-saved['offset']==tail['bytes']==135633 and record(tail['path'])==tail
        and Path(tail['path'])==F/'preserved69_unsaved_positive_journal_suffix.jsonl',
        'Exact published saved prefix and compact preserved69 COPY; original23 STAT only')
    current=Path(stat['path']).stat()
    require(current.st_size==stat['bytes']and current.st_mtime_ns==stat['mtime_ns'],
        'Original failed whole23 remains untouched; no prefix/whole read or hash')
    events=[json.loads(line)for line in Path(tail['path']).read_bytes().splitlines()]
    require(len(events)==69 and journal['all69_require_genuine_redo_after170400restore']is True
        and all(x['optimizer_updates']==x['native']['update']==x['prepared_evidence']['actual_positive_native_update']==u
            for u,x in zip(range(170401,170470),events)),
        'All69 COPY events remain historical; no replay, promotion or fabricated new positive')
    require(Path(read_json(folder/'active_journal.json')['journal_prefix']['path'])==Path(saved['path']),
        'Old23 remains active until genuine full restore and savedprefix-only new branch')
    return {**value,'saved_optimizer_updates':170400,'saved_journal_prefix':saved,
        'whole_journal_including_preserved_tail':stat,'preserved_unsaved_tail':tail,
        'preserved_tail_positive_updates':list(range(170401,170470))}
