"""Bounded scalar status maintenance for the owned head-LR control."""
import argparse
import datetime
import hashlib
import json
import os
import re
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = root / 'outputs/parakeet_turkish'
p = argparse.ArgumentParser()
p.add_argument('--DEV-step', type=int, required=True)
p.add_argument('--milestone', required=True)
args = p.parse_args()
assert args.milestone.replace('_', '').isalnum()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path, value):
    path = Path(path); temp = path.with_suffix('.metadata.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
def record(path):
    path = Path(path).resolve()
    return {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes':path.stat().st_size}
lease_path = root / 'work/gpu_training_owner.json'
lease = read(lease_path)
run_id = lease['target_run_id']
assert run_id == 'adaptive_20261013_TR_uniform_headlr3e5_cosine_01'
status = read(Path(lease['actual_controller_loop']) / 'loop_status.json')
assert status['status'] == 'running'
with Path(status['paths']['log']).open('rb') as stream:
    stream.seek(max(0, stream.seek(0, 2) - 1_000_000))
    lines = stream.read().decode('utf-8', errors='replace').splitlines()
    rows = [json.loads(line) for line in lines if line.startswith('{"step":')]
last = rows[-1]
assert last['step'] >= args.DEV_step > 25109
summary_path = out / 'results' / (run_id + '_step_' + str(args.DEV_step)) / 'summary.json'
summary = read(summary_path)
counts = {lang:summary['validation_' + lang]['asr'] for lang in ('tr','en','es','it','fr','de','pt')}
refs = {'tr':825,'en':1045,'es':1289,'it':1158,'fr':1072,'de':1004,'pt':1095}
limits = {'en':69,'es':52,'it':30,'fr':74,'de':70,'pt':66}
assert all(counts[l]['reference_words'] == refs[l] and counts[l]['utterances'] == 50 for l in refs)
failed = [l for l in limits if counts[l]['word_errors'] > limits[l]]
best_log = next((line for line in reversed(lines) if re.match(r'^dev step\d+ .* retention (True|False) best\d+$', line)), None)
best = None
if best_log:
    best_step = int(best_log.rsplit('best', 1)[1])
    if best_step > 25109:
        best_path = out / 'results' / (run_id + '_step_' + str(best_step)) / 'summary.json'
        best_summary = read(best_path)
        assert all(best_summary['validation_' + l]['asr']['reference_words'] == refs[l] and best_summary['validation_' + l]['asr']['word_errors'] <= limits[l] for l in limits)
        best = {'step':best_step, 'TR_WER':best_summary['validation_tr']['asr']['wer'], 'summary':record(best_path), 'save_completion_stdout':best_log, 'checkpoint_tensor_read':False}
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
receipt = {'kind':'TSC_uniform_head_LR_actual_live_milestone_v1','observed_at_utc':now,'run_id':run_id,'actual_owned_session':lease['actual_controller_session_id'],'step':last['step'],'local_updates':last['step']-25109,'matched_cycle':last['Turkish_domain_coverage']['matched_cycle'],'applied_head_lr':last['head_lr'],'applied_encoder_lr':last['lora_lr'],'last_complete_DEV_step':args.DEV_step,'counts':counts,'failed_retention_languages':failed,'summary':record(summary_path),'memory':status['system_and_owned_private_memory'],'durable_checkpoint_claimed':False,'TEST_used':False}
receipt['actual_branch_qualified_BEST'] = best
receipt_path = out / ('matched_headlr3e5_cosine_' + args.milestone + '_v1.json')
assert not receipt_path.exists()
write(receipt_path, receipt)
lease.update(updated_at_utc=now,actual_completed_step=last['step'],actual_encoder_schedule_updates=last['step']-25109,current_matched_cycle_unique=receipt['matched_cycle']['cycle_unique'],actual_applied_encoder_lr=last['lora_lr'],actual_last_complete_DEV_step=args.DEV_step,current_complete_DEV_all_six_passed=not failed,actual_current_milestone=record(receipt_path))
if best:
    lease['current_head_qualified_BEST'] = best
write(lease_path, lease)
plan_path = out / 'experiment_plan.json'; plan = read(plan_path)
for key in ('current_training','actual_current_training'):
    current = plan[key]
    assert current['run_id'] == run_id
    current.update(current_frontier=last['step'],local_encoder_schedule_updates=last['step']-25109,cycle2_unique_including_carried=receipt['matched_cycle']['cycle_unique'],applied_encoder_lr=last['lora_lr'],last_complete_DEV_step=args.DEV_step,last_DEV_counts=counts,last_complete_DEV_TR_WER=counts['tr']['wer'],last_DEV_TR_WER=counts['tr']['wer'],last_DEV_TR_word_errors=counts['tr']['word_errors'],last_DEV_all_six_passed=not failed,last_DEV_scope='Actual complete fixed7x50 DEV',actual_live_milestone=record(receipt_path))
    if best:
        current.update(current_branch_qualified_BEST_step=best['step'],current_branch_qualified_BEST_TR_WER=best['TR_WER'],branch_best_saved_completion_log=best['save_completion_stdout'])
plan['updated_at_utc'] = now
plan['status_authority'] = f'Head-LR3e-5/cosine control is actually running at{last["step"]:,} (+{last["step"]-25109:,} from25,109), owned{lease["actual_controller_session_id"]}; current-cycle unique{receipt["matched_cycle"]["cycle_unique"]:,}/253,798. Complete DEV{args.DEV_step:,}: TR{counts["tr"]["word_errors"]}/825; strict failures{failed}. Full18,129-update matched cycle continues monitor-only with unchanged guards. Both encoder controls completed; selected cosine qualified BEST42,500/TR279 remains preserved. TEST frozen.'
write(plan_path, plan)
print(json.dumps({'passed':True,'receipt':record(receipt_path),'step':last['step'],'local_updates':last['step']-25109,'DEV':args.DEV_step,'TR_errors':counts['tr']['word_errors'],'failed':failed,'commit_fraction':status['system_and_owned_private_memory']['system_memory']['commit_fraction']}))
