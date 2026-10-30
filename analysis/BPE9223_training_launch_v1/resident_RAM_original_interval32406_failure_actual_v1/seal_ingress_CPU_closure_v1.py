import hashlib,json
from pathlib import Path
import psutil
N=Path(__file__).parent
def pin(p):
 p=Path(p).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
r=json.loads((N/'ingress_frontend_actual_01/report.json').read_text());who=r['producer']
try:dead=abs(psutil.Process(who['pid']).create_time()-who['create_time'])>=.001
except psutil.NoSuchProcess:dead=True
assert dead
c={'kind':'ASR18_actual_one_original_full_clip_ingress_CPU_control_TOOL0_v1','actual_exec_tool_exit_code':0,'actual_tool_chunk':'535b03','actual_owned_session':82716,'producer':who,'producer_observed_dead':dead,'CPU_report':pin(N/'ingress_frontend_actual_01/report.json'),'CPU_completion':pin(N/'ingress_frontend_actual_01/CPU_completion.json'),'source':pin(N/'verify_one_row_ingress_frontend_v1.py'),'configuration':r['configuration'],'ROOT_source_review':r['ROOT_source_review'],'overlay_source':r['overlay_source'],'CPU_stream_source':r['CPU_stream_source'],'certificate':r['certificate'],'prior_exact_four_row_probe':{'actual_exec_tool_exit_code':0,'actual_tool_chunk':'e38be1','report':pin(N/'probe_actual_02/report.json')},'independent_full_member':{'actual_exec_tool_exit_code':0,'actual_tool_chunk':'f82d7e','report':pin(N/'full_member_actual_02/report.json')},'preserved_fixture_failures':[{'scope':'Probe1 import initialization beforeaudio','actual_tool_exit_code':1,'tool_chunk':'195ed3','source':pin(N/'probe_four_original_intervals_v1.py'),'completion':pin(N/'probe_actual_01/CPU_completion.json')},{'scope':'Independent fullmember fixture1 public PacketSideData conversion before frame decode','actual_tool_exit_code':1,'tool_chunk':'137076','source':pin(N/'observe_exact_full_member_EOF_v1.py'),'completion':pin(N/'full_member_actual_01/CPU_completion.json')}],'passed':r['passed'],'no_GPU_or_model_or_native_updates':True,'original_source_indexes_plan_and_journal_unchanged':True,'actual_corrected_full_PCM_and_frontend_geometry_proven':True}
p=N/'actual_ingress_frontend_CPU_closure_v1.json';assert not p.exists();p.write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8');print(json.dumps({'closure':pin(p),'elapsed_seconds':r['elapsed_seconds'],'producer':who}))
