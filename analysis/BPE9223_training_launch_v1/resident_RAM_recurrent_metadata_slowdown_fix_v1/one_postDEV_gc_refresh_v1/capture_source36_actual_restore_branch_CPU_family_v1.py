"""Once-only compact actual restore/branch/CPU-ready capture; no data/payload read."""
import hashlib,json,os,shutil,sys
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent;N=F.parent.parent;A=N/'resident_RAM_training_actual_26'
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000025'
T=Path('E:/gpu_training_scratchpad/analysis/BPE9223_Trackio_live_v1');OUT=A/'actual_source36_restore_branch_CPU_ready_family_v1.json'
def pin(p):
 p=Path(p).resolve();assert p.stat().st_size<=8*1024**2;b=p.read_bytes()
 return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):
 p=Path(p);assert p.stat().st_size<=8*1024**2;return json.loads(p.read_text(encoding='utf-8-sig'))
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def alive(w):
 try:return abs(psutil.Process(w['pid']).create_time()-w['create_time'])<.001
 except psutil.NoSuchProcess:return False
assert not OUT.exists() and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
samples=[sample_memory(p.pid,'source36_actual_restore_branch_CPU_compact_capture') for p in (me,me.parent())]
assert all(s['paired_sampling_within_one_second'] for s in samples) and sum(s['worker_private_memory']['private_bytes'] for s in samples)<=536870912
m=samples[-1]['system_memory'];assert m['commit_available_bytes']>=5055306368 and m['physical_available_bytes']>=2305306368
assert shutil.disk_usage('E:/').free>=100000000000
review_path=F/'ROOT_source36_one_postDEV_gc_refresh_saved178800_final_execution_review_v1.json';r=read(review_path)
assert pin(review_path)['sha256']=='2a23e3143e8e91586f46214e5af1c650f6057887a43e169352fd2be6c99b04d2'
assert not(I/'actual_failure.json').exists()
restore=read(I/'actual_parent_restore.json');branch=read(I/'actual_journal_branch.json');ready=read(I/'CPU/actual_CPU_ready.json')
boot=read(I/'boot.json');child=read(A/'current_child.json');configuration=read(I/'configuration.json')
expected=read(r['source_configuration']['path']);expected.update(r['future_seven_runtime_bindings']);assert configuration==expected
assert restore['passed'] is True and restore['all207_threeAdam_all4RNG_match_actual_pre_save'] is True
assert restore['next_draw_packet_match'] is True and restore['restore_occurred_only_at_startup_before_native'] is True
saved=read(r['initial_checkpoint']['path'])['journal_prefix']
assert saved['offset']==360796716 and saved['sha256']=='ea1f1ee763d18c09bdcaa37830f49ae26ca3ad843ce835cbf977e755c4619ef8'
assert branch['predecessor']==r['gc_refresh_resume'] and branch['saved_prefix']==saved
assert branch['new_active_journal']=={'path':str(B/'journals/commits_000025.jsonl'),'offset':saved['offset'],'sha256':saved['sha256']}
assert branch['historical_unsaved_tail_updates']==list(range(178801,178821)) and branch['native_updates_or_draws_in_copy']==0
assert ready['source']['protocol']==r['CPU_source'] and ready['packet_credit']['logical_packet_maximum']==2
assert ready['packet_credit']['helper_source']==r['packet_credit_policy']['helper_source']
cpu=ready['producer_identity'];launcher=child['launcher']
family=[('native',boot['producer']),('native_interpreter_launcher',launcher),('wrapper',who(psutil.Process(launcher['ppid']))),
 ('CPU',cpu),('CPU_interpreter_launcher',who(psutil.Process(cpu['ppid'])))]
assert all(alive(w) for _,w in family)
progress=read(B/'progress/latest.json');reader={name:read(T/name) for name in ('current_status.json','mirror_state.json','owner.json')}
assert alive({'pid':26892,'create_time':1792652904.3359847})
value={'kind':'ASR18_actual_source36_restore_branch_CPU_ready_family_v1','owned_session':42997,'initial_exec_chunk':'a4bdd3',
 'ROOT_execution_review':pin(review_path),'initial_owned_boot_receipt':pin(A/'actual_owned_source36_boot_session.json'),
 'boot':pin(I/'boot.json'),'configuration':pin(I/'configuration.json'),'restore':pin(I/'actual_parent_restore.json'),
 'branch':pin(I/'actual_journal_branch.json'),'CPU_ready':pin(I/'CPU/actual_CPU_ready.json'),'current_child':pin(A/'current_child.json'),
 'all7_and_every_remaining_configuration_value_exact':True,'full207_threeAdam_fourRNG_bothcursors_nextdraw_restored_before_native':True,
 'saved_prefix_only_copied_AFTER_restore_with_zero_native_or_draw':True,'historical20_preserved_must_genuinely_redo':True,
 'CPU13_MAX2_and_original_source_helper_bound':True,'actual_five_family':[{'role':role,**w,'exact_alive':True} for role,w in family],
 'current_native_progress':progress,'reader_unchanged_exact_identity_alive':True,'bounded_reader_current_records':reader,
 'capture_identity':who(me),'capture_launcher_identity':who(me.parent()),'metadata_resources':samples,
 'actual_journal_events_read':0,'old24_journal_prefix_whole_or_tail_or_tensor_audio_read':False,
 'no_probe_GC_scientific_control_or_training_reader_environment_change':True,'initial_recovery_and_speed_still_unaccepted':True}
with OUT.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'stage_receipt':pin(OUT),'CPU_ready':value['CPU_ready'],'actual_five':value['actual_five_family'],
 'progress':progress['actual_optimizer_updates'],'saved':progress['last_saved_updates']}),flush=True)
