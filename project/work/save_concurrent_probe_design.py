"""Record a CPU-only concurrency design; never launch or initialize CUDA."""
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/parakeet_turkish'
RUN='adaptive_20261010_recovery_b8_01_b8_quality'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
history=read(OUT/'runs'/RUN/'training_history.json')
row=history[-1]
torch_memory_source=Path(importlib.util.find_spec('torch').origin).parent/'cuda/memory.py'
speed=read(OUT/'large_batch_speed_report.json')
b1=[{k:r[k] for k in ('family','audio_min_per_wall_min','peak_allocated_vram_gb','peak_reserved_vram_gb')} for r in speed['table'] if r['batch_size']==1]
state=read(OUT/'runs/adaptive_20261010_recovery_b8_01/loop_status.json')
report={
 'schema':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'CPU-only design and read-only live history; no GPU initialization, concurrency launch, source/report/model mutation',
 'decision':'do_not_launch_secondary_now',
 'reason':f"Primary allocator reported{row['current_reserved_vram_gb']:.3f}GB reserved at step{row['step']} in real B8 training. Frozen matched B1 second-job evidence needs3.21GB typical/3.45GB long reserved, before additional CUDA context/workspace/desktop memory. These allocator snapshots do not establish a conservative fit within16GiB nominal physical VRAM. Windows driver-free/committed-memory observations are still required; neither tensor allocation nor allocator reservation alone guarantees coexistence.",
 'primary':{'run_id':RUN,'controller_sha256':sha(OUT/'run_experiment_loop_recovery_v2.py'),'original_deadline_at_utc':state['deadline_at_utc'],'last_published_history_row':{k:row[k] for k in ['step','audio_seen_seconds','train_wall_seconds','process_wall_seconds','audio_minutes_per_wall_minute','peak_allocated_vram_gb','current_allocated_vram_gb','current_reserved_vram_gb']},'note':'Live early-run cumulative training rate excludes DEV/startup/checkpoint/history serialization; not a concurrent or full-run throughput claim.'},
 'secondary_solo_evidence':{'large_batch_speed_report_sha256':sha(OUT/'large_batch_speed_report.json'),'fixed_train_effective16_B1_cases':b1,'limitation':'Historical solo figures are capacity/workload references, never summed to estimate concurrent performance.'},
 'existing_profiler_constraints':{'path':str(OUT/'profile_large_batches.py'),'sha256':sha(OUT/'profile_large_batches.py'),'requires_gpu_context_solo':True,'writes_solo_gpu_operator_declared_true':True,'concurrent_use':'Cannot use its unchanged CLI or bypass/monkeypatch the guard. Any concurrent probe must be a new explicitly concurrent worker/report.'},
 'conditional_future_probe':{
  'launch_prerequisites':['Parent observes sustained compute headroom on real B8 windows and stable actual driver-free memory; low-utilization startup/DEV/checkpoint periods do not qualify.','A conservative primary peak envelope across the upcoming long TRAIN shapes plus secondary allocation/context/workspace and driver/desktop reserve fits physical VRAM. Existing sampled long data do not bound every expanded-pool shape.','Parent-owned five-minute total probe budget does not encroach on the original finalization reserve; abort rather than run through a DEV boundary.','Current primary process/model/helpers remain unchanged and no cache-release or allocator patch is injected.'],
  'worker':'NEW standalone concurrent_training_probe_v1.py, if later authorized; import frozen row selection, collator, objective and initialization functions, never call frozen solo profiler main.',
  'workload':'B1 with accumulation16, exact previously audited balanced16 typical TRAIN clips, immutable pilot1200 weights, fresh Adam/objective coefficients/row multiplier. Fixed TRAIN only; secondary weights disposable and no DEV/TEST inference or model selection.',
  'memory_controls':{'starting_secondary_allocator_cap_gb':3.6,'enforcement':'torch.cuda.set_per_process_memory_fraction before secondary model initialization, with secondary catching allocator OOM and exiting; additional driver/context/workspace allowance must be budgeted separately.','caveat':'This is a per-process caching-allocator ceiling, not a global reservation or primary OOM guarantee. External watchdog/termination cannot preempt an instantaneous primary allocation spike.','watchdog':'External CPU monitor driver-free VRAM at100–250ms; cease/terminate secondary only if free reserve is crossed, primary progress stalls, or a DEV boundary appears. No primary stop/empty_cache/restart.'},
  'initialization_and_warmup':'Load CPU assets outside measured windows. GPU initialization and two warmup updates count as concurrent overhead; record separately and monitor primary slowdown, not silently exclude the interference.',
  'window_schedule':'A/B/A, each60–90seconds and >=50 completed primary updates when possible: primary alone, primary+secondary, primary alone after secondary GPU process exits. Total<=5minutes including bounded secondary startup/warmup. All windows use one parent monotonic clock; no independent-timing sum.',
  'boundary_methodology':'Without editing the primary, align each measured window to immediate observation of a fresh10-step history publication. Record receipt times and file mtime, begin after published step s0, finish at first chosen later publication s1, and count exact primary sampled-audio deltas. Secondary emits synchronized update completion monotonic timestamps and exact fixed-WAV PCM seconds; count only completions within the same observed [T0,T1]. Poll at<=100ms and explicitly report primary publication/serialization boundary uncertainty; cannot claim exact kernel-level timing.',
  'metrics':{'primary_only_rate':'primary completed audio seconds/(T1-T0)','concurrent_primary_rate':'primary completed audio seconds/(T1-T0)','concurrent_secondary_rate':'secondary completed audio seconds/(T1-T0)','concurrent_total_rate':'(primary+secondary completed audio seconds)/(same T1-T0)','units':'audio minutes per wall minute (seconds/seconds)','comparison':'Compare concurrent total to the bracketing solo-primary rate; also report primary retention ratio and effective elapsed GPU wall overhead. Never add the historical independently measured solo rates.'},
  'record':['All window start/end UTC and monotonic timestamps, process inventory and overlap, exact primary step/source/duration ranges and mean/p90T/U, secondary WAV/cache/weights/code hashes and completed clip counts.','Actual device utilization/power/free-memory samples and process allocated/reserved peaks; actual secondary allocator cap and driver-scope limitations.','Primary rate before/during/after, total joint rate, slowdown, error/OOM/abort and window contamination (DEV/checkpoints/history writes).','Declare descriptive changing-primary-data evidence; adjacent windows can differ in clip lengths and padding. A causal matched-throughput comparison requires a later solo-controlled repeat.'],
  'acceptance_policy':'No second production experiment from this probe. Continue secondary work only if measured total improves materially while primary throughput loss and memory risk meet parent-chosen bounds; any CUDA/model error is a failed probe, not pressure to loosen numerical or quality gates.'
 },
 'useful_parallel_work_now':['CPU-only provenance/auditor/data analysis and dashboards already run concurrently, with low thread counts and no large cache scans during timed training.','A future trainer version can overlap CPU WAV/frontend/cache prefetch with the same GPU training stream, retaining source/order and objective; it needs its own matched throughput/parity measurement.','Independent second GPU model/training is deferred until the primary budget or allocator design allows a controlled test.'],
 'registration_verification':{'path':str(OUT/'trackio_b8_readonly_verification.json'),'sha256':sha(OUT/'trackio_b8_readonly_verification.json'),'passed':True,'checks':read(OUT/'trackio_b8_readonly_verification.json')['checks_passed']},
 'primary_sources':{'pytorch27_memory_management':'https://docs.pytorch.org/docs/2.7/notes/cuda.html#memory-management','local_pytorch27_allocator_api':str(torch_memory_source),'local_allocator_api_sha256':sha(torch_memory_source)}
}
destination=OUT/'concurrent_training_probe_plan.json'
destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
text='''Concurrent GPU throughput probe: deferred for the current live B8 run.

The live B8 allocator reservation grew from14.93GB to above17GB; the audited fixed B1 secondary needs3.21GB typical or3.45GB long reserved before driver/context memory. Windows driver-free/committed-memory readings are still needed; low current tensor allocation does not imply another model can fit. No secondary job was launched.

The frozen profile_large_batches.py requires solo GPU and declares solo timing; preserve it. If a later parent-owned window has conservative memory/compute headroom, use a NEW explicitly concurrent worker importing its frozen functions. Start with physicalB1/accum16 on the same audited balanced16 TRAIN clips and a3.6GB secondary allocator cap. The cap bounds its PyTorch allocator only and cannot guarantee the primary's future peak.

Bound the entire probe to5minutes. Measure60–90s primary-alone / joint / primary-alone windows on a single parent clock, aligned to primary history publication boundaries. Include exact completed primary+secondary audio in the same joint window; compare this combined rate against the bracketing primary-alone rate and show primary slowdown separately. Record publication-time uncertainty, input-length/padding differences, GPU utilization/free memory, startup/warmup interference, and every abort/OOM. Do not sum existing solo-throughput numbers.

No current GPU probe is needed to keep CPU provenance checks, dashboard mirroring and light analysis parallel. CPU frontend/cache prefetch in a future trainer is a more direct way to fill CPU/GPU gaps and needs its own measured matched test. Preserve current training and the original23:41:22.889692UTC finalization deadline.

PyTorch caching-allocator documentation:
https://docs.pytorch.org/docs/2.7/notes/cuda.html#memory-management
'''
(OUT/'concurrent_training_probe_plan.txt').write_text(text,encoding='utf-8')
print(json.dumps({'saved':str(destination),'live_step':row['step'],'primary_reserved_gb':row['current_reserved_vram_gb'],'decision':report['decision']}))
