"""Bounded actual progress bookkeeping; no weights/history arrays or GPU."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
run='adaptive_20261012_TR_uniform_majority_PT_focus_02';directory=HERE/'runs'/run
def record(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
queue=json.loads((HERE/'experiment_queue_state.json').read_text(encoding='utf-8'))
assert queue['current_run_id']==run and queue['status']=='running' and queue['published_step']>8000
path=directory/'training_history.json'
with path.open('rb') as stream:
    stream.seek(max(0,path.stat().st_size-262144));tail=stream.read().decode('utf-8',errors='ignore')
marker='\n  {\n    "step":';position=tail.rfind(marker)
assert position>=0
raw=tail[position+3:].strip();assert raw.endswith(']');row=json.loads(raw[:-1].rstrip())
assert row['step']>8000 and row['PT_focus_updates']==row['step']-8000
coverage=row['Turkish_domain_coverage'];focus=coverage['PT_focus_transition'];local=coverage['uniform_pass']
assert focus['Portuguese_draws_after_transition']==focus['old_five_draws_after_transition']==row['PT_focus_updates']
assert local['draws']==14274+14*row['PT_focus_updates'] and local['unique_clips_seen']==local['draws'] and local['repeat_visits']==0
assert row['lora_lr']==1e-4 and row['head_lr']==1e-5 and row['PT_replay_clips_per_update']==row['old_five_replay_clips_per_update']==1
proof=json.loads((directory/'majority_initial_source_equivalence.json').read_text(encoding='utf-8'))
assert proof['passed'] and proof['actual_GPU_decode'] and proof['compared_fixed_DEV_utterances']==350 and proof['source_selected_step']==8000
assert not proof['all_six_retention_gates_passed']
now=datetime.now(timezone.utc).isoformat()
actual={'schema':1,'kind':'PT_focus02_actual_source350_positive_updates_v1','passed':True,'observed_at_utc':now,
    'actual_session_id':51956,'actual_controller_pid':queue['pid'],'actual_native_launcher_pid':33432,'actual_worker_pid':5728,
    'run_id':run,'completed_steps':row['step'],'PT_focus_updates':row['PT_focus_updates'],'new_uniform_unique_TR_clips':local['unique_clips_seen'],
    'uniform_full_TR_pool':253798,'repeat_visits':0,'configured_end_step':25109,'objective_complete':False,
    'source350':record(directory/'majority_initial_source_equivalence.json'),'source_all_six_qualified':False,
    'source_PT_word_errors':72,'PT_allowed_word_errors':66,'parent_failure_counter_carried':4,
    'actual_native_positive_updates':True,'head_lr':row['head_lr'],'lora_lr':row['lora_lr'],
    'actual_language_weights':row['preservation_coefficients_by_language'],'phase_cold_throughput_not_steady_claimed':True,
    'last_actual_history_row_canonical_sha256':hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),
    'current_gpu_peak_allocated_gb':row['peak_allocated_vram_gb'],'actual_new_DEV_pending':True,'source_points_are_not_new_quality_gains':True}
p=HERE/'PT_focus02_actual_source350_positive_updates_v1.json'
with p.open('x',encoding='utf-8') as stream:json.dump(actual,stream,indent=2)
lease_path=ROOT/'work/gpu_training_owner.json';lease=json.loads(lease_path.read_text(encoding='utf-8'))
lease.update(status='actual_sole_GPU_PT_focus02_positive_updates',updated_at_utc=now,actual_completed_step=row['step'],
    actual_PT_focus_updates=row['PT_focus_updates'],actual_new_uniform_unique_TR_clips=local['unique_clips_seen'],
    source350_and_first_optimizer_pending=False,actual_source350_passed=True,actual_positive_update_receipt=record(p))
temp=lease_path.with_suffix('.tmp');temp.write_text(json.dumps(lease,indent=2),encoding='utf-8');temp.replace(lease_path)
plan_path=HERE/'experiment_plan.json';plan=json.loads(plan_path.read_text(encoding='utf-8'))
plan['status_authority']=f'Actual soleGPU owner benchmark_design session51956 LIVE at{row["step"]}/PTfocused{row["PT_focus_updates"]} updates after real source350 equality; {local["unique_clips_seen"]}/253798 uniform unique/no repeats. Full207 state from8000; next completeDEV authoritative; no goal completion.'
plan['actual_current_training']={**actual,'actual_GPU_launched':True,'worker':'train_lora_batched_tsc_uniform_majority_ptfocus_v2.py',
    'controller':'run_TSC_uniform_majority_ptfocus_v2.py','logger':'track_continuation_v14.py','same_Trackio_project':'parakeet-turkish'}
item=next(item for item in plan['ordered_experiments'] if item['id']=='04')
item.update(status='actual_PT_focused_same_fullpass_running',prepared_worker='train_lora_batched_tsc_uniform_majority_ptfocus_v2.py',
    prepared_boundary='tsc_uniform_majority_ptfocus_transition_v1.py',prepared_sampler='tsc_uniform_majority_ptfocus_sampler_v1.py',
    actual_current_run=run,actual_completed_steps=row['step'],actual_new_uniform_unique_TR_clips=local['unique_clips_seen'],
    active_configured_end_step=25109,actual_PT_focus_updates=row['PT_focus_updates'],source_step6500_original_champion_retained=True)
item['recipe']='Full207 latest8000 state/Adam/RNG/history and existing uniform queue carry; LRhead1e-5/LoRA1e-4/r8alpha16/native4/B8x2;14TR+1PT+1uniformoldfive replay. PT KL40/duration2/features4; oldfive coefficients multiply5/3. No renewedgrace or counterreset; strict allsix BEST/guards, sameend25109/full253798unique.'
item['prior_PT_focus01_attempt']={'actual_closed_session':92124,'actual_exit_code':1,'tool_chunk':'77b922','optimizer_updates':0,
    'failure':'missing logging context key lora_boundary_step','repair':'newv2 reads validated parentconfig.source_step7750; same8000 weights/state/science'}
plan['automatic_execution_policy']['PT_focused_replay']='Healthy exclusive owner51956: preserve14TRprimary and fullpassfrontier; onePT+oneuniformoldfive, PTcoeff40/2/4 andoldfive5/3 compensation. AwaitcompleteDEV8250/allstrictgates. Budget-only owncomplete0 mayautoexactcontinue; resource/retention/manual require evidence review, no unchangedretry/reset.'
temp=plan_path.with_suffix('.tmp');temp.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(plan_path)
notes=f'Actual PT-focused full-state continuation {run} is LIVE under soleGPU owner benchmark_design session51956/controllerPID{queue["pid"]}/workerPID5728. Source8000 actual350 raw/token/duration/count equality passed with PT72>66 honestly failed; complete207 initialstate saved and native positive updates reached{row["step"]} (new{row["PT_focus_updates"]}). Uniform unique{local["unique_clips_seen"]}/253798,0repeat; end25109/no reset/no renewedPTgrace/failure4 carried. RP onePT+oneuniformoldfive; PT40/2/4,oldfive5/3 compensation; LR1e-4/head1e-5. NewDEV8250 pending; no new quality gain/goal claim. Prior01 actuallyclosed1/77b922 at0updates due missing logging field, preserved and corrected in v2. V13 actualROOTclosed0/86b3ef; soleV14 logger66719 retains every historical metric in sameproject. Evidence: PT_focus02_actual_source350_positive_updates_v1.json.'
for task in ('ASR-15','ASR-15.1'):
    subprocess.run([str(HERE/'tools/backlog.exe'),'task','edit',task,'--status','In Progress','--append-notes',notes],cwd=HERE,check=True,capture_output=True,text=True,encoding='utf-8')
print(json.dumps({'actual_step':row['step'],'PT_focus_updates':row['PT_focus_updates'],'unique_TR':local['unique_clips_seen'],'receipt':record(p),'plan_Backlog_lease_updated':True}))
