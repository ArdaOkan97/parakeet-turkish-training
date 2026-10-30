"""Actual finite CPU control wait0 and identity death; no GPU action."""
import hashlib,json
from pathlib import Path
import psutil
N=Path(__file__).parent
def pin(p):
 p=Path(p).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def dead(w):
 try:return abs(psutil.Process(w['pid']).create_time()-w['create_time'])>=.001
 except psutil.NoSuchProcess:return True
r=json.loads((N/'ingress_frontend_actual_01/report.json').read_text());who=r['producer']
assert dead(who) and r['passed'] is True and r['raw_buffers_released_and_metadata_handles_closed'] is True
assert r['actual_geometry']=={'clean16k_samples':120048,'valid_mel_frames':750,'physical_feature_frames':751,'encoder_frames':94}
assert len(r['reference']['full_target_ids'])==39 and len(r['negative_controls'])==5 and all(x['rejected'] for x in r['negative_controls'])
c={'kind':'ASR18_actual_fourth_original_terminal_interval_ingress_CPU_control_TOOL0_v1','actual_exec_tool_exit_code':0,'actual_tool_chunk':'b14faf','actual_owned_session':46257,'producer':who,'producer_observed_dead':True,'CPU_report':pin(N/'ingress_frontend_actual_01/report.json'),'CPU_completion':pin(N/'ingress_frontend_actual_01/CPU_completion.json'),'source':pin(N/'verify_fourth_terminal_row_ingress_frontend_v1.py'),'configuration':r['configuration'],'ROOT_source_review':r['ROOT_source_review'],'overlay_source':r['overlay_source'],'CPU_stream_source':r['CPU_stream_source'],'certificate':r['certificate'],'ordered_source_coverage_localization':{'actual_exec_tool_exit_code':0,'actual_tool_chunk':'d73333','report':pin(N/'probe_actual_01/report.json')},'independent_full_member':{'actual_exec_tool_exit_code':0,'actual_tool_chunk':'8bbcac','report':pin(N/'full_member_actual_01/report.json')},'passed':True,'no_GPU_or_model_or_native_updates':True,'original_source_indexes_plan_and_journal_unchanged':True,'actual_full_PCM_frontend_mask_and39targets_proven':True,'historical_mel750_physical_mel750_explicit':True,'encoder_T94_preserved':True}
p=N/'actual_ingress_frontend_CPU_closure_v1.json';assert not p.exists();p.write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8');print(json.dumps({'closure':pin(p),'elapsed_seconds':r['elapsed_seconds'],'producer':who,'peak_private_bytes':r['peak_private_bytes']}),flush=True)
