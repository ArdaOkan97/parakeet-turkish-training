"""Seal already observed actual0; no new training/fixture or tensor payload scan."""
import json,sys
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from fastmss_owned_process_closure_v2 import verify
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');D=A/'hard_effective16_native_actual_01'
r=read_json(D/'report.json');require(r['passed'] is True and r['actual_native_updates']==r['actual_global_updates']==1,'Observed finite actual1 pass')
for pin in r['actual_process_completions']:
 proof=read_json(pin['path']);require(record(pin['path'])==pin,'Immutable actualprocess proof')
 verify(proof,kind=proof['kind'],owner='/root/memory_diagnosis',guard=lambda:None)
require(r['all_deferred_recovery_obligations_discharged'] is True and r['final_mature_clocks']==[64251,2001,62251],'Complete original-source update/freshrestore')
def stages(folder):
 values=[json.loads(line) for line in (folder/'stages.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
 return {'actual_current_private_peak_bytes':max(v['worker_private_memory']['private_bytes'] for v in values),
  'OS_lifetime_private_peak_bytes':max(v['worker_private_peak_bytes_lifetime'] for v in values),
  'CUDA_peak_allocated_bytes':max(v.get('cuda',{}).get('peak_allocated_bytes',0) for v in values),
  'CUDA_peak_reserved_bytes':max(v.get('cuda',{}).get('peak_reserved_bytes',0) for v in values),'stage_count':len(values)}
native=stages(D/'GPU_process_0');recovery=stages(D/'GPU_recovery_process')
cpu=read_json(D/'CPU_window_0/owned_CPU_family_completion.json');pointer=read_json(r['current_checkpoint_pointer']['path'])
actual={'kind':'ASR18_hard_effective16_operator_observed_TOOL_CLOSED0_v1','owner':'/root/memory_diagnosis',
 'actual_owned_session':58706,'actual_tool_chunk_id':'d69391','actual_tool_exit_code':0,'actual_tool_completion_observed':True,
 'report':record(D/'report.json'),'configuration':record(A/'hard_effective16_configuration_CPU_actual_02/configuration_pending_execution_review.json'),
 'execution_review':record(A/'ROOT_hard_effective16_execution_configuration_review_v1.json'),
 'source_manifest':record(A/'hard_effective16_full_caller_source_review_manifest_v3.json'),
 'actual_process_completions':r['actual_process_completions'],'actual_native_updates':1,'production_enabled':False}
ap=D/'actual_owned_CLOSED0_v1.json';require(not ap.exists(),'Fresh observedactual receipt');write_json(ap,actual)
summary={'kind':'ASR18_hard_effective16_CPU8_native_fresh_recovery_actual_summary_v1','passed':True,'actual0':record(ap),
 'actual_owned_session':58706,'actual_tool_chunk_id':'d69391','actual_tool_exit_code':0,
 'actual_CPU_controller_PID':30948,'actual_native_PID':32100,'actual_recovery_PID':37204,
 'actual_new_updates':1,'final_mature_clocks':[64251,2001,62251],
 'scope':'ONE original-source diagnostic update:13 exact retained realTR +completehard lastTR +2 exact original carriedreplay; currentCPU8 fullraw/header/frontend then precomputed physical1 effective16, real sampler/journal commit, PRE/save/persistence/native0/cleanup0/fresh allstate1 recovery0',
 'hard_geometry':{'original_ordinal':1051133,'source_id':'kiraat:src01792-00990','audio_samples':2547840,'seconds':159.24,'physical_feature_frames':15925,'valid_feature_frames':15924,'T':1991,'U':574},
 'actual_languages':r['actual_languages'],'CPU_family_peak_private_bytes':cpu['measured_family_peak_private_bytes'],
 'CPU_parent_phase_wall_seconds':cpu['actual_parent_phase_wall_seconds'],'native':native,'recovery':recovery,
 'whole_wall_seconds':r['wall_seconds'],'native_window_phase_seconds':r['windows'][0]['phase_seconds'],
 'checkpoint_pointer':r['current_checkpoint_pointer'],'checkpoint_metadata':pointer['checkpoint'],'payload':pointer['payload'],
 'payload_SHA_scope':'Exact native writer/persistence and fresh restoration already verified; this summary does not reread payload bytes',
 'all207_threeAdam_all4RNG_bothsamplers_journal_prefix_nextdraw_exact_fresh_recovery':True,
 'actual_CPU_native_cleanup_recovery_process0_all_captured_identities_dead':True,
 'source_raw_audio_and_full_targets_preserved':True,'source_schedule_immutable_ancestry':True,
 'production_phase_advance_claimed':False,'diagnostic_only':True,'production_source_eligible':False,'qualified_checkpoint':False,
 'all_shape_long_loop_fit_claimed':False,'production_enabled':False,'report':record(D/'report.json'),
 'still_required':['Concrete source-bound full-run4-update validation with indexed AUX/currentlatest recovery and streamed durable prefix','Coherent emergency partial-window resource-stop closure','Final resource/storage/performance/launch review']}
sp=D/'actual_hard_effective16_end_to_end_summary_v1.json';require(not sp.exists(),'Fresh summary');write_json(sp,summary)
now=datetime.now(timezone.utc).isoformat();old=A/'launch_readiness_v80.json';v=read_json(old)
v.update(updated_at_utc=now,status='actual_hard_effective16_CPU8_native_and_fresh_recovery_closed0_fullowner_pending',actual_owned_session=None,
 actual_CPU_controller_PID=None,actual_CPU_reference_leaf_PID=None,actual_GPU_running=False,actual_owned_GPU_PID=None,actual_new_native_updates=1,
 latest_hard_effective16_summary=record(sp),historical_readiness=record(old))
vp=A/'launch_readiness_v81.json';require(not vp.exists(),'Fresh readiness');write_json(vp,v)
c=read_json(A/'current_launch_readiness.json');c.update(updated_at_utc=now,current_readiness=record(vp),current_stage=v['status'],actual_owned_session=None,
 actual_GPU_running=False,actual_owned_GPU_PID=None,latest_hard_effective16_summary=record(sp),
 current_actual_run={'owned_session':58706,'actual_tool_chunk_id':'d69391','actual_exit_code':0,'folder':str(D),'actual_new_native_updates':1,'summary':record(sp),'production_enabled':False})
write_json(A/'current_launch_readiness.json',c)
lease=P/'work/gpu_training_owner.json';l=read_json(lease);require(l['owner']=='/root/memory_diagnosis' and not l['actual_GPU_running'] and not l.get('actual_owned_worker_pid'),'Actual owned idle lease after allclosures')
l.update(status=v['status'],actual_training_optimizer_running=False,updated_at_utc=now,latest_hard_effective16_summary=record(sp));write_json(lease,l)
print(json.dumps({'summary':record(sp),'actual0':record(ap),'readiness':record(vp),'native':native,'recovery':recovery,'CPU':cpu,'wall_seconds':r['wall_seconds']}))
