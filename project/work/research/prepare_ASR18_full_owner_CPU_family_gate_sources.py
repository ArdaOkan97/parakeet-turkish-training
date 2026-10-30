"""NEW exact prospective unused-family gate before each CPU/cleanup child."""
from pathlib import Path
import ast,hashlib
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
old=S/'asr18_full_run_parent_controller_v6.py';new=S/'asr18_full_run_parent_controller_v7.py'
assert not new.exists();text=old.read_text(encoding='utf-8-sig')
before='    def _prepare(self,plan,index):'
after='''    def _admit_CPU_family(self,output):
        require(self.parent.phase=='base','CPU launch remains in actual base-parent domain')
        admissions=[]
        for _ in range(3):
            own=self.parent.check()['worker_private_memory']['private_bytes']
            if self.gpu is None:
                remaining=max(128000000-own,0)+64000000
                context={'GPU_absent':True,'actual_parent_private_bytes':own,'remaining_family_bytes':remaining}
            else:
                self.gpu.check()
                require(self.gpu.command_state=='coherent_idle_CPU_preparation','No native forward/Adam during CPU preparation')
                context=self.gpu._family_guard()
                require(time.monotonic()-context['observed_monotonic']<=1,'Exact fresh available resident-family context')
                remaining=context['remaining_parent_and_allshim_allowance_bytes']
            admission=self.fresh_admit(self.config['CPU_owned_allowance_bytes']+remaining)
            require(admission['passed'],'Fresh wholeCPU plus unused parent/allGPUshims does not fit; no child launch')
            admissions.append({'admission':admission,'exact_family_context':context});time.sleep(.1)
        self._new(output,{'kind':'ASR18_actual_pre_CPU_full_family_admission_v1','passed':True,
            'configuration':self.config_pin,'source':record(__file__),'CPU_whole_allowance_bytes':1934217728,
            'resident_actual_memory_already_in_global_baseline':True,'samples':admissions})

    def _prepare(self,plan,index):'''
assert text.count(before)==1;text=text.replace(before,after)
before="        cpu=root/'CPU';proof=self.run_CPU("
after="        self._admit_CPU_family(root/'CPU_remaining_family_admission.json')\n        cpu=root/'CPU';proof=self.run_CPU("
assert text.count(before)==1;text=text.replace(before,after)
before="        proof=self.run_CPU(self.config,self.cleanup_script,"
after="        self._admit_CPU_family(root/'cleanup_remaining_family_admission.json')\n        proof=self.run_CPU(self.config,self.cleanup_script,"
assert text.count(before)==1;text=text.replace(before,after)
ast.parse(text);new.write_text(text,encoding='utf-8',newline='\n')
raw=new.read_bytes();print({'path':str(new),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'AST_only':True})
old=R/'run_ASR18_full_run_owner_v8.py';new=R/'run_ASR18_full_run_owner_v9.py'
assert not new.exists();text=old.read_text(encoding='utf-8-sig').replace('from asr18_full_run_parent_controller_v6 import run','from asr18_full_run_parent_controller_v7 import run')
ast.parse(text);new.write_text(text,encoding='utf-8',newline='\n')
raw=new.read_bytes();print({'path':str(new),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'AST_only':True})
old=R/'prepare_ASR18_full_owner_configuration_v2.py';new=R/'prepare_ASR18_full_owner_configuration_v3.py'
assert not new.exists();text=old.read_text(encoding='utf-8-sig').replace('from asr18_full_run_parent_controller_v6 import FullOwner','from asr18_full_run_parent_controller_v7 import FullOwner')
ast.parse(text);new.write_text(text,encoding='utf-8',newline='\n')
raw=new.read_bytes();print({'path':str(new),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'AST_only':True})
# Static source/identity successor. This reads only known small metadata/source,
# reuses already closed evidence bindings, and creates NO runtime namespace.
text=(R/'seal_ASR18_full_owner_static_identity_v1.py').read_text(encoding='utf-8-sig')
text=text.replace('prepare_ASR18_full_owner_configuration_v1.py','prepare_ASR18_full_owner_configuration_v3.py')
text=text.replace('asr18_full_run_parent_controller_v6.py','asr18_full_run_parent_controller_v7.py')
text=text.replace('run_ASR18_full_run_owner_v8.py','run_ASR18_full_run_owner_v9.py')
for name in ('full_owner_source_contract','full_owner_validation_identity','full_owner_external_protection_plan','full_owner_exact_ROOT_review_inputs'):
    text=text.replace(name+'_v1.json',name+'_v3.json')
before=" 'required_roles':['source','history','DEV','qualified_winner'],"
assert text.count(before)==1;text=text.replace(before," 'required_roles':{")
before=" 'qualified_winner':[qualified_summary],'adopt_external_sources_as_owned':False}"
assert text.count(before)==1;text=text.replace(before," 'qualified_winner':[qualified_summary]},'adopt_external_sources_as_owned':False}")
ast.parse(text);exec(compile(text,str(R/'seal_ASR18_full_owner_static_identity_v1.py'),'exec'),
 {'__name__':'__main__','__file__':str(R/'seal_ASR18_full_owner_static_identity_v1.py')})
