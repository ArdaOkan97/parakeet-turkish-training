"""One actual fixed350 quality and resumed-training observation."""
import copy,json,sys
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
N=Path(__file__).parent;B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01')
entry=[v for v in read_json(B/'periodic_DEV.json')['completed_evaluations']if v['optimizer_updates']==68000]
require(len(entry)==1,'One actual periodic68000 result');entry=entry[0]
report=read_json(entry['DEV']['path']);preserved=read_json(entry['frontier_preservation']['path'])
for key in ('DEV','frontier_preservation','checkpoint_pointer'):require(record(entry[key]['path'])==entry[key],'Exact actual report/state/pointer pins')
require(report['passed'] is True and report['optimizer_updates']==68000 and report['actual_native_updates_in_DEV']==0
 and report['all207_threeAdam_all4RNG_unchanged'] is True and report['TEST_used'] is False and report['retention_early_stop'] is False
 and report['official_saved_text_score']==entry['score'],'Actual full350 report/official score/preservation')
require(set(report['predictions'])=={'tr','en','es','it','fr','de','pt'} and all(report['scores']['validation_'+lang]['asr']['utterances']==50 for lang in report['predictions']),'Actual original seven50-clip DEV sets')
for key in ('all207_threeAdam_all4RNG_unchanged','both_cursors_journal_and_nextdraw_unchanged','fullpass_and_winner_state_unchanged_here'):require(preserved[key] is True,'Actual state/nextdraw/pending-plan preservation')
require(preserved['actual_native_updates_in_DEV']==0 and preserved['actual_sampler_or_replay_advance'] is False and preserved['actual_phase_private']['joined'] is True and preserved['actual_phase_private']['error'] is None,'Real idle DEV/phase monitor closure')
require(entry['periodic_record_advances_full_pass_or_winner'] is False and record(preserved['pending_private_plan_retained']['path'])==preserved['pending_private_plan_retained'],'Original pending private plan exact')
raw=(B/'progress/latest.json').read_bytes();progress=json.loads(raw);who=progress['producer']
require(who['pid']==8252 and who['create_time']==1792486098.8274934 and abs(psutil.Process(who['pid']).create_time()-who['create_time'])<.001
 and progress['actual_optimizer_updates']>68000 and progress['last_saved_updates']>=68000 and not progress['optimizer_poisoned'] and not progress['schedule_poisoned'],'Same actual healthy source10 and positive resumed updates')
observed=N/'observed_post68000_DEV_positive_progress.json';require(not observed.exists(),'Unique immutable positive observation');observed.write_bytes(raw)
persistence=record(B/'checkpoints/update_000068000/persistence_verified.json');pv=read_json(persistence['path']);require(pv['named_tensors_verified']==854 and pv['Adam_states_verified']==215,'Actual full checkpoint persistence')
prior=N.parent/'resident_RAM_training_actual_10/actual_periodic64000_and_resumed_training_status.json';prior_value=read_json(prior);old_errors=prior_value['periodic_result']['score']['word_errors']['tr'];score=entry['score']
failed={lang:{'actual':score['word_errors'][lang],'limit':limit,'excess':score['word_errors'][lang]-limit} for lang,limit in score['retention_limits'].items()if score['word_errors'][lang]>limit}
status={'kind':'ASR18_actual_periodic68000_full350_and_resumed_live_training_v1','actual_owned_session':62574,'producer':who,'producer_observed_live':True,'source':progress['source'],'source_configuration':record(N.parent/'resident_RAM_original_interval_post47727_failure_actual_v1/production_configuration_v1/configuration.json'),'ROOT_execution_review':record(N.parent/'ROOT_third_original_full_clip_saved47600_execution_review_v1.json'),'periodic_result':entry,'checkpoint_persistence':persistence,'DEV_wall_seconds':preserved['wall_seconds'],'DEV_private_peak_bytes_native_owner':preserved['actual_phase_private']['private_peak_bytes'],'CUDA_after_allocated_bytes':preserved['CUDA_after']['cuda']['allocated_bytes'],'CUDA_lifetime_peak_after_bytes':preserved['CUDA_lifetime_peak_allocated_bytes_after'],'CUDA_lifetime_peak_is_not_DEV_only':True,'all207_threeAdam_all4RNG_and_cursors_journal_nextdraw_preserved':True,'pending_private_plan_retained':preserved['pending_private_plan_retained'],'prior64000_result':record(prior),'prior64000_Turkish_errors':old_errors,'Turkish_word_errors_reduction_since64000':old_errors-score['word_errors']['tr'],'Turkish_WER_percentage_points_improvement_since64000':(old_errors-score['word_errors']['tr'])/825*100,'failed_retention_languages':failed,'observed_post_DEV_positive_progress':record(observed),'actual_optimizer_updates_at_observation':progress['actual_optimizer_updates'],'same_active_journal_path':progress['journal_prefix']['path'],'full401072_automatic_continuation_active':True,'next_periodic_frontier':72000,'periodic_fullpass_or_winner_advancement':False,'source_environment_or_training_child_changed_during_DEV':False,'private_scalar_prefetch_applied':True,'private_scalar_prefetch_changed_scientific_schedule_or_numerics':False,'cache_candidate_applied':False,'duplicate_milestone_or_notification_recorded':False}
path=N/'actual_periodic68000_and_resumed_training_status.json';require(not path.exists(),'One new milestone receipt');write_json(path,status);print(json.dumps({'receipt':record(path),'score':score,'actual_updates':progress['actual_optimizer_updates']}),flush=True)
