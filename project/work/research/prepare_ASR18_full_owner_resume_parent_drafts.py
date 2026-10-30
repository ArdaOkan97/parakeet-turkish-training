"""NEW normal-resume CLI/guard evidence drafts; pool API binding still pending."""
import ast,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parents[2];R=P/'work/research'
def emit(p,t):
    assert not p.exists();ast.parse(t);compile(t,str(p),'exec');p.write_text(t,encoding='utf-8',newline='\n')
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
text=(R/'run_ASR18_full_run_owner_v13.py').read_text(encoding='utf-8-sig')
text=text.replace('def controller(config_path,review_path,folder):','def controller(config_path,review_path,folder,resume_request=None):',1)
text=text.replace('from asr18_full_run_parent_controller_v11 import run','from asr18_full_run_parent_controller_v12 import run',1)
text=text.replace('native_boundary_DEV=native_boundary_DEV,install_owner=install_owner)',
    'native_boundary_DEV=native_boundary_DEV,install_owner=install_owner,resume_request=resume_request)',1)
text=text.replace("p.add_argument('--cleanup',action='store_true');p.add_argument('--request')",
    "p.add_argument('--cleanup',action='store_true');p.add_argument('--request');p.add_argument('--resume-request')",1)
text=text.replace('else:controller(a.config,a.review,folder)',
    "else:controller(a.config,a.review,folder,None if a.resume_request is None else read_json(a.resume_request))",1)
before="        require(not self.thread.is_alive() and self.error is None and self.count>=3,'Clean joined parent guard')"
after=before+"""
        from fastmss_owned_process_closure_v2 import identity
        write_json(self.folder/'owner_guard_completion.json',{
            'kind':'ASR18_actual_outer_owner_guard_joined_completion_v1','producer':identity(),
            'source':record(__file__),'monitor_joined':True,'monitor_clean':True,'samples':self.count,
            'last_observation':self.last,'actual_owner_TOOL_exit_still_unobserved_inside_process':True})"""
assert text.count(before)==1;text=text.replace(before,after)
parent=emit(R/'run_ASR18_full_run_owner_v14.py',text)
text=(R/'prepare_ASR18_full_owner_configuration_v8.py').read_text(encoding='utf-8-sig')
text=text.replace('from asr18_full_run_parent_controller_v11 import FullOwner','from asr18_full_run_parent_controller_v12 import FullOwner',1)
before="        and template['full_run_launch_enabled'] is False,'No metadata action enables a long run')"
after="""        and template['full_run_launch_enabled'] is False
        and template['normal_boundary_resume_enabled'] is True and template['owner_validation_pause_after_updates']==4,
        'No metadata action enables a long run; exact resume-enabled genesis and bounded split4/4')"""
assert text.count(before)==1;text=text.replace(before,after)
builder=emit(R/'prepare_ASR18_full_owner_configuration_v9.py',text)
print(json.dumps({'draft_parent':parent,'draft_builder':builder,'worker_control_pool_API_pending':True,'no_target_import_or_job':True}))
