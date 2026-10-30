"""Seal actual finite control wait0 and observed process death; no GPU action."""
import hashlib,json
from pathlib import Path
import psutil
N=Path(__file__).parent
def pin(p):
 p=Path(p).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
r=json.loads((N/'ingress_frontend_actual_01/report.json').read_text());who=r['producer']
try:dead=abs(psutil.Process(who['pid']).create_time()-who['create_time'])>=.001
except psutil.NoSuchProcess:dead=True
assert dead and r['passed'] is True and r['raw_buffers_released_and_metadata_handles_closed'] is True
assert r['actual_geometry']=={'clean16k_samples':235358,'valid_mel_frames':1470,'physical_feature_frames':1471,'encoder_frames':184}
assert len(r['reference']['full_target_ids'])==97 and len(r['negative_controls'])==5 and all(x['rejected'] for x in r['negative_controls'])
c={'kind':'ASR18_actual_second_original_full_clip_ingress_CPU_control_TOOL0_v1','actual_exec_tool_exit_code':0,'actual_tool_chunk':'68b08f','actual_owned_session':46416,'producer':who,'producer_observed_dead':dead,'CPU_report':pin(N/'ingress_frontend_actual_01/report.json'),'CPU_completion':pin(N/'ingress_frontend_actual_01/CPU_completion.json'),'source':pin(N/'verify_second_row_ingress_frontend_v2.py'),'configuration':r['configuration'],'ROOT_source_review':r['ROOT_source_review'],'overlay_source':r['overlay_source'],'CPU_stream_source':r['CPU_stream_source'],'certificate':r['certificate'],'prior_exact_four_row_probe':{'actual_exec_tool_exit_code':0,'actual_tool_chunk':'0ccf0f','report':pin(N/'probe_actual_01/report.json')},'independent_full_member':{'actual_exec_tool_exit_code':0,'actual_tool_chunk':'40f4eb','report':pin(N/'full_member_actual_01/report.json')},'preserved_unexecuted_fixture_typo_source':pin(N/'verify_second_row_ingress_frontend_v1.py'),'fixture_typo_repaired_before_actual_execution':True,'passed':True,'no_GPU_or_model_or_native_updates':True,'original_source_indexes_plan_and_journal_unchanged':True,'actual_full_PCM_frontend_mask_and97targets_proven':True,'historical_mel1471_physical_mel1470_explicit':True,'encoder_T184_preserved':True}
p=N/'actual_ingress_frontend_CPU_closure_v1.json';assert not p.exists();p.write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8');print(json.dumps({'closure':pin(p),'elapsed_seconds':r['elapsed_seconds'],'producer':who,'peak_private_bytes':r['peak_private_bytes']}),flush=True)
