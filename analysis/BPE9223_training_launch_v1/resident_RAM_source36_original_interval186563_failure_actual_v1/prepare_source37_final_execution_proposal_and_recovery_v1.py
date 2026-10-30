"""Prepare final launch proposal and future metadata recovery source only; never activate or import it."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import psutil

F=Path(__file__).parent
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
EVENT=F.parent/'resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1'
OLD=EVENT/'seal_source36_dynamic_natural_recovery_v1.py'
ARCHIVE_SOURCE=F.parent/'resident_RAM_original_interval170469_failure_actual_v1/seal_source33_saved170600_eighth_recovery_v1.py'
NEW=F/'seal_source37_saved186600_ninth_recovery_v1.py'
LEDGER=F/'source37_ninth_recovery_sealer_full_RAW_derivation_v1.json'
HANDOFF=F/'final_source37_ninth_saved186400_execution_handoff_PROPOSAL_v1.json'


def require(v,m):
    if not v:raise ValueError(m)


def pin(p):
    p=Path(p).resolve();require(p.stat().st_size<=8388608,'Source/compact metadata only');b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p,v):
    with p.open('x',encoding='utf-8',newline='\n')as h:json.dump(v,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())


def main():
    me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    who=lambda p:{'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
    print(json.dumps({'prep_identity':who(me),'prep_launcher':who(me.parent())}),flush=True)
    sample=sample_memory(me.pid,'source37_final_proposal_source_only_preparation')
    write(F/'source37_final_proposal_preparation_initial_numeric_sample_v1.json',sample)
    require(sample['paired_sampling_within_one_second']and sample['worker_private_memory']['private_bytes']<536870912
        and sample['system_memory']['commit_available_bytes']>=5055306368
        and sample['system_memory']['physical_available_bytes']>=2305306368
        and shutil.disk_usage('E:/').free>=100000000000,'Original metadata full225/fullprospective/fullprefetch/floors/E guards')
    require(all(not p.exists()for p in (NEW,LEDGER,HANDOFF)),'Fresh immutable proposal/source targets')
    raw=OLD.read_bytes();text=raw.decode('utf-8');records=[]
    def rep(old,new,count=1):
        nonlocal text
        require(text.count(old)==count,'Exact recovery source literal '+repr(old[:90]))
        records.append({'old_literal':old,'new_literal':new,'expected_count':count});text=text.replace(old,new)
    rep('source36','source37',text.count('source36'))
    rep('training_v36.py','training_v37.py',text.count('training_v36.py'))
    rep('Only new25 saved-offset..old actual+1 events; original24 STAT only. No archive,',
        'Only new26 saved-offset..old actual+1 events; original25 STAT only. One new bounded archive,')
    rep('One new bounded archive,\ntensor, audio, model, native/GC operation or source/reader/environment mutation.',
        'One new bounded archive.\nNo tensor, audio, model, native/GC operation or source/reader/environment mutation.')
    rep('import argparse,hashlib,json,math,os,shutil,sys,time','import argparse,gzip,hashlib,io,json,math,os,shutil,sys,time')
    rep('from resource_watchdog_v2 import sample_memory','from resource_watchdog_v2 import sample_memory\nfrom bpe_broad_production_state_v1 import digest')
    rep("I=B/'invocations/owner_000025';A=F.parent.parent/'resident_RAM_training_actual_26'", "I=B/'invocations/owner_000026';A=F.parent/'resident_RAM_training_actual_27'")
    rep(" require(s['paired_sampling_within_one_second'] and PEAK<536870912", 
        " if admission:write(F/'source37_recovery_initial_numeric_before_predicate_v1.json',s)\n require(s['paired_sampling_within_one_second'] and PEAK<536870912")
    rep(" require(root_boot.get('accepted',root_boot.get('passed')) is True,'Reuse separately accepted actual boot/runtime7/restore-order ROOT receipt')", 
        " require(root_boot.get('accepted')is True and root_boot.get('reviewer')=='/root','Reuse separately accepted actual boot/runtime7/restore-order ROOT receipt')")
    rep(" closure=read(review['gc_refresh_resume']['path']);source=pin(R/'run_ASR18_BPE9223_resident_RAM_training_v37.py')", 
        " require(review['failure_ROOT_review']==pin(F/'ROOT_source36_failure186563_saved186400_actual_independent_verification_v1.json'),'Closed actual source36 failure ROOT metadata, no old tail or closure re-audit')\n closure=read(review['failure_ROOT_review']['path']);source=pin(R/'run_ASR18_BPE9223_resident_RAM_training_v37.py')")
    rep("old_actual=closure['actual_optimizer_updates']", "old_actual=closure['actual_optimizer']")
    rep("restored==closure['last_fully_saved_optimizer_updates']", "restored==closure['saved']==186400")
    rep("branch['predecessor']==review['gc_refresh_resume']", "branch['predecessor']==review['ninth_original_resume']")
    rep("closure['published_journal_optimizer_updates']", "closure['published']")
    rep("B/'journals/commits_000025.jsonl'", "B/'journals/commits_000026.jsonl'")
    rep("'Only exact new25 active branch'", "'Only exact new26 active branch'")
    rep("old=closure['closed_old_journal_STAT']", "old=closure['original_journal_STAT_ONLY']")
    rep("'Original24 STAT-only untouched; never prefix/whole/tail reread'", "'Original25 STAT-only untouched; never prefix/whole/tail reread'")
    rep("'Actual unchanged CPU13/MAX2/helper identity'", "'Actual CPU14/MAX2/helper identity; unchanged transport/math/guards'")
    rep("'Genuine ordered nativepositive finiteobjective/grad/CPU13/16IDs14TR2replay/absolute cursors'", "'Genuine ordered nativepositive finiteobjective/grad/CPU14/16IDs14TR2replay/absolute cursors'")
    rep(" require(len({x['prepared_evidence']['token'] for x in events})==count,'Unique actual new prepared tokens')", 
        " require(len({x['prepared_evidence']['token'] for x in events})==count and len({x['transaction_sha256']for x in events})==count,'Unique actual new prepared tokens and transactions')\n require(events[-1]['native']['source_ids'][4]=='alania:a280321c852958fbb53e9d7a2044d1a6_00000','Repaired186564 row5 original source reached actual native')")
    # Reuse the exact already source-reviewed archive parsing/known-routekey method.
    archive_raw=ARCHIVE_SOURCE.read_bytes();archive_text=archive_raw.decode('utf-8')
    begin=archive_text.index(' # One NEW standard170600 metadata archive, never audio or tensor data.')
    end=archive_text.index(" require('numpy'not in sys.modules",begin)
    oldblock=archive_text[begin:end];block=oldblock;block_records=[]
    def brep(old,new):
        nonlocal block
        count=block.count(old);require(count>0,'Declared archive-block literal '+old)
        block_records.append({'old_literal':old,'new_literal':new,'expected_count':count});block=block.replace(old,new)
    for old,new in [('170600','186600'),('170470','186564'),('170401','186401'),('170601','186601'),
                    ('prepared[69]','prepared[163]'),("metadata['rows'][5]","metadata['rows'][4]"),
                    ("metadata['same_pass_clean_evidence'][5]","metadata['same_pass_clean_evidence'][4]"),
                    ("native['source_ids'][5]","native['source_ids'][4]"),("row['target_tokens']==99","row['target_tokens']==66"),
                    ("row['samples']==273624","row['samples']==210280"),("row['encoder_frames']==214","row['encoder_frames']==165"),
                    ('3a9e9240d7348caf1b2a26f53b98d362836ec6cad088a747cc31008a0cf9659b','9bf1307f8c93030c8b09fcc263b2e21879d47fe03a900a96ffcd71fb05ab72af'),
                    ('1711','1315'),('820896','630864'),('273632','210288'),("'archive_index':69","'archive_index':163"),
                    ("'consumed_row_index':6","'consumed_row_index':5"),('332560','156134'),('row6','row5'),('full99','full66'),
                    ('clean273624','clean210280'),('eighth certificate/alias9','ninth certificate/alias10')]:brep(old,new)
    # Dynamic observer uses cp and pin/write helpers; all file reads remain metadata.
    brep("archive=checkpoint/", "archive=cp/")
    brep('read_json(certificate_pin', 'read(certificate_pin')
    brep('write_json(out/', 'write(out/')
    binverse=block
    for r in reversed(block_records):
        require(binverse.count(r['new_literal'])==r['expected_count'],'Archive block RAW inverse unambiguous')
        binverse=binverse.replace(r['new_literal'],r['old_literal'])
    require(binverse==oldblock,'Entire original archive extraction block reconstructed exactly')
    inheritance=""" inherited=read(I/'actual_inherited_consumed180000_refresh_opportunity.json')
 require(inherited['passed']is True and inherited['historical_event_optimizer_updates']==180000
  and inherited['restored_optimizer_updates']==restored and inherited['opportunity_consumed_before_native_loop']is True
  and inherited['new_owner_attempted']is False and inherited['new_owner_completed']is False
  and inherited['new_owner_operations_completed']==[]and inherited['new_eligible_or_ineligible_event_capture_GC_sync_calls']==0
  and inherited['policy']==configuration['inherited_one_postDEV_refresh_opportunity_policy']
  and not(I/'actual_one_postDEV_gc_refresh.json').exists()and not(I/'actual_one_postDEV_gc_refresh_ineligible.json').exists(),
  'Actual inherited consumed180k receipt before loop; no second eligible/ineligible/capture/GC/sync event')
"""
    anchor=" reader={}\n for name in ('current_status.json','mirror_state.json','owner.json'):"
    rep(anchor,block+inheritance+anchor)
    rep("'old24_STAT_only':old", "'old25_STAT_only':old")
    rep("'no_old_journal_prefix_whole_tail_archive_tensor_audio_GC_GPU_or_production_mutation':True", 
        "'no_old_journal_prefix_whole_tail_archive_tensor_audio_GC_GPU_or_production_mutation':True,\n  'new186600_archive':archive_pin,'repaired186564_row5_production_evidence':corrected,\n  'inherited_consumed180k_receipt':pin(I/'actual_inherited_consumed180000_refresh_opportunity.json'),\n  'new_owner_completed_false_zero_second_event':True")
    inverse=text
    for r in reversed(records):
        require(inverse.count(r['new_literal'])==r['expected_count'],'Entire recovery RAW inverse unambiguous')
        inverse=inverse.replace(r['new_literal'],r['old_literal'])
    require(inverse.encode('utf-8')==raw,'Full old recovery observer bytes reconstructed')
    ast.parse(text)
    with NEW.open('xb')as h:h.write(text.encode('utf-8'));h.flush();os.fsync(h.fileno())
    write(LEDGER,{'kind':'source37_ninth_recovery_source_full_RAW_derivation_v1','old_source':pin(OLD),'new_source':pin(NEW),
        'replacement_records':records,'whole_RAW_forward_inverse_exact':True,'newline_normalization':'NONE',
        'inserted_archive_source':pin(ARCHIVE_SOURCE),'original_archive_block':oldblock,'new_archive_block':block,
        'archive_block_replacement_records':block_records,'entire_archive_block_inverse_exact':True,
        'no_observer_import_or_execution':True})
    source=read(F/'source37_ninth_inactive_source_handoff_v1.json')
    typed_scope=read(F/'ROOT_source37_ninth_entire_selector_metadata_exact_ONCE_execution_scope_v1.json')
    old_final=read(EVENT/'ROOT_source36_one_postDEV_gc_refresh_saved178800_final_execution_review_v1.json')
    cfg=read(source['candidate_configuration']['path']);future7=read(source['future_runtime7']['path'])['future_runtime_values']
    typed_ROOT=F/'ROOT_source37_ninth_actual_typed_metadata_independent_verification_v1.json'
    require(pin(typed_ROOT)['sha256']=='1e6d62bb21bf9e81c814d3cb8225e219073b55138af2a0c5ff67d2d27758395a','Genuine actual typed ROOT gate')
    final_path=F/'ROOT_source37_ninth_saved186400_final_ONE_training_execution_review_v1.json'
    prefix=[str(P/'work/nemo-native-env-v2/Scripts/python.exe'),'-Xutf8','-B',source['wrapper_source']['path']]
    args=['--configuration',source['candidate_configuration']['path'],'--pointer',source['initial_checkpoint']['path'],
        '--latest-failure',old_final['latest_failure']['path'],'--branch',str(B),'--review',str(final_path),
        '--output',str(F.parent/'resident_RAM_training_actual_27'),'--start-invocation','26','--queue-bytes','268435456',
        '--ninth-original-resume',source['ninth_original_resume']['path']]
    require(len(args)==18,'Exact18 wrapper strings')
    proposal={'kind':'source37_ninth_saved186400_final_execution_handoff_PROPOSAL_v1',
        'production_execution_admitted':False,'proposed_final_ROOT_review_path':str(final_path),
        'driver_source':source['driver_source'],'wrapper_source':source['wrapper_source'],
        'RAM_native_source':pin(S/'asr18_BPE9223_grouped_RAM_native_v1.py'),
        'CPU_source':pin(S/'fastmss_resident_feature_stream_v14.py'),'DEV_helper_source':pin(S/'asr18_resident_serial350_DEV_v2.py'),
        'source_configuration':source['candidate_configuration'],'initial_checkpoint':source['initial_checkpoint'],
        'latest_failure':old_final['latest_failure'],'ninth_original_resume':source['ninth_original_resume'],
        'ninth_original_transition_validator':source['transition_validator'],
        'original_clip_geometry_overlay':cfg['original_clip_geometry_overlay'],'packet_credit_policy':cfg['resident_RAM_packet_credit_policy'],
        'inherited_one_postDEV_refresh_opportunity_policy':cfg['inherited_one_postDEV_refresh_opportunity_policy'],
        'one_postDEV_gc_refresh_policy':cfg['one_postDEV_gc_refresh_policy'],
        'private_scalar_prefetch_source':cfg['private_scalar_prefetch_policy']['source'],
        'private_scalar_prefetch_actual_CPU_review':cfg['private_scalar_prefetch_policy']['actual_CPU_control_review'],
        'additional_prefetch_host_reservation_bytes':268435456,'periodic_DEV_CPU_quiescence_policy':cfg['periodic_DEV_CPU_quiescence_policy'],
        'fresh_metadata_signature_and_collect_observation_policy':cfg['fresh_metadata_signature_and_collect_observation_policy'],
        'prepared_metadata_archive_representation_policy':cfg['prepared_metadata_archive_representation_policy'],
        'native_update_commit_pressure_policy':cfg['native_update_commit_pressure_policy'],
        'bounded_RAM_queue_bytes':268435456,'periodic_DEV_interval_updates':4000,'initial_owner_invocation':26,
        'future_seven_runtime_bindings':future7,'runtime7':source['future_runtime7'],
        'source_semantic_ROOT_review':typed_scope['source_semantic_review'],'actual_typed_ROOT_review':pin(typed_ROOT),
        'failure_ROOT_review':pin(F/'ROOT_source36_failure186563_saved186400_actual_independent_verification_v1.json'),
        'saved186400_ROOT_review':pin(EVENT/'ROOT_source36_standard186400_metadata_and_PRE_SAVED_equality_verification_v1.json'),
        'actual_ninth_ingress_ROOT_review':pin(F/'ROOT_ninth_CPU14_actual01_original_ingress_frontend_independent_verification_v1.json'),
        'derived_CPU14_transport_ROOT_review':pin(F/'ROOT_CPU14_one_overlay_import_only_transport_compatibility_successor_v1.json'),
        'exact_initial_launch_command_prefix':prefix,'exact_initial_wrapper_arguments':args,'exact_future_wrapper_arguments':args,
        'recovery_observer_source':pin(NEW),'recovery_observer_RAW_derivation':pin(LEDGER),
        'recovery_observer_future_arguments':['--execution-review',str(final_path),'--ROOT-boot-review','PENDING_SEPARATELY_ACCEPTED_ACTUAL_SOURCE37_BOOT_ROOT_RECEIPT'],
        'required_actual_recovery':{'all854_weights207_3Adam_4RNG_bothsamplers_nextdraw_restore186400_before_native_and_copy':True,
            'saved25_prefix_only':{'offset':375720854,'sha256':'c658c5c1711f0e43b3b764e0a8875e90cdd5137f7d27469dd28936802fd02e80'},
            'copy_after_restore_zero_native_draws_old25_STAT_only':True,'old163_historical_require_genuine_redo':True,
            'ONLY_new164_range':[186401,186564],'direct_ROOT_seek_count':1,'owner_range_bounded_snapshot_once':True,
            'actual186564_row5_CPU14_certificate9_source10_full66_T165_evidence':True,
            'ONE_new186600_archive_compressed_cap':8388608,'ONE_new186600_archive_uncompressed_cap':33554432,
            'natural_full_saved':186600,'postpositive_at_least':186620,
            'current5_exact_alive_and_healthy_reader26_catchup':True,
            'inherited180k_before_loop_newowner_completed_false_no_second_event':True},
        'prelaunch_original_guards_and_storage':{'read_E_storage_config_dot_source_processlocal':True,
            'production_CUDA_visible_devices':'0','all4math_caps':1,'UTF8_B_BelowNormal':True,
            'no_diagnostic_CPU_hidden_environment_leak':True,'full_target':401072,'administrative_stop':None,
            'same225_MAX2_LIVE1_packet_queue_FULLprefetch_and16MiB_bookkeeping_reservation':True,
            'native_commit':4518435456,'native_physical':1768435456,'CPU_prospective':2500000000,'CPU_actual':1500000000,
            'CPU_private':8000000000,'E_free':100000000000,'allocation_and_packet_wait':300,
            'actual_original_production_admission_required_no_freed_credit':True},
        'fresh_namespace_requirements':[str(B/'invocations/owner_000026'),str(B/'journals/commits_000026.jsonl'),str(F.parent/'resident_RAM_training_actual_27')],
        'last_known_training5_exact_dead_reader2_alive':typed_scope['protected_runtime_identities'],
        'source_and_config_bytes_unchanged_from_actual_typed_proof':True,'no_more_CPU_proof_or_original_tail_audit_required':True,
        'human_recurrent_speed_fix_OPEN_no_permanent_speed_fit_saving_claim':True,
        'no_timing538f_instrumentation_no_GC_LR_science_guard_reader_change':True,
        'input_pins':typed_scope['input_pins']+[pin(typed_ROOT),pin(NEW),pin(LEDGER),pin(OLD),pin(ARCHIVE_SOURCE)],
        'prep_identity':who(me),'prep_launcher_identity':who(me.parent())}
    write(HANDOFF,proposal)
    print(json.dumps({'recovery_source':pin(NEW),'recovery_RAW':pin(LEDGER),'final_handoff_proposal':pin(HANDOFF),
        'exact_prefix':prefix,'exact18_arguments':args,'prep_identity':who(me),'prep_launcher_identity':who(me.parent())}),flush=True)


if __name__=='__main__':main()
