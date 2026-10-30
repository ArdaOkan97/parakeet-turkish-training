"""NEW prior archive verification BEFORE any preparing workspace reservation."""
from pathlib import Path
import ast,hashlib,json
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def emit(path,text):
    assert not path.exists();ast.parse(text);compile(text,str(path),'exec');path.write_text(text,encoding='utf-8',newline='\n')
    raw=path.read_bytes();print({'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'AST_only':True})
text=(S/'asr18_full_run_parent_controller_v8.py').read_text(encoding='utf-8-sig')
before="    def _prepare(self,plan,index):\n        self.check();intent=self.store.reserve_publication(publication_role='completed_window_end');self.current_intent=intent"
after='''    def _verify_prior_prefix_before_reservation(self):
        self._refresh_boundary();boundary=deepcopy(self.boundary_state)
        require(boundary['durable_head']==self.durable.head_pin and boundary['coherent_checkpoint_pointer']==self.current_capture['pointer'],
                'Same actual current durable/capture before reservation')
        io=_Metadata(self.identity,self.resolve_member,self.parent.check);journal=_JournalPrefixes(self.parent.check)
        try:
            prior_plan=io.read(self.current_original_plan)
            pointer,saved=_capture(self.current_capture,self.current_capture['pointer'],io,
                identity_sha=digest(self.identity),previous_sampler=prior_plan['initial_sampler_sha256'],
                previous_updates=prior_plan['initial_committed_updates'],journal=journal)
            require(digest(saved)==digest(self.schedule.state_dict())==boundary['sampler_state_sha256'],
                    'Independent prior PRE/persistence/sampler/journal while store has NO preparing workspace')
        finally:journal.close()
        return {'boundary':boundary,'pointer':pointer,'sampler_sha256':digest(saved),'phase':prior_plan['phase']}

    def _prepare(self,plan,index):
        self.check()
        verified=self._verify_prior_prefix_before_reservation() if index>0 else None
        intent=self.store.reserve_publication(publication_role='completed_window_end');self.current_intent=intent'''
assert text.count(before)==1;text=text.replace(before,after)
a=text.index('            io=_Metadata(',text.index('    def _prepare('))
b=text.index('            if self.gpu is not None:',a)
text=text[:a]+'''            require(boundary==verified['boundary'] and self.durable.head_pin==boundary['durable_head']
                and digest(self.schedule.state_dict())==verified['sampler_sha256'],
                'After new bridge, recheck SAME previously verified old head/capture/sampler')
            pointer=read_bound(self.current_capture['pointer'],self.parent.check)
            require(pointer==verified['pointer'],'No prior pointer mutation between archive verification and CPU handoff')
''' +text[b:]
text=text.replace("phase=boundary['phase'] if 'phase' in boundary else prior_plan['phase'],","phase=verified['phase'],")
text=text.replace("'saved_sampler_sha256':digest(saved),'historical_fresh_GPU_restore_claimed':False}",
                  "'saved_sampler_sha256':verified['sampler_sha256'],'historical_fresh_GPU_restore_claimed':False}")
emit(S/'asr18_full_run_parent_controller_v9.py',text)
text=(R/'run_ASR18_full_run_owner_v10.py').read_text(encoding='utf-8-sig')
text=text.replace('from asr18_full_run_parent_controller_v8 import run','from asr18_full_run_parent_controller_v9 import run')
before="require(('.atomic-v1-' in name and name.endswith('.tmp')) or '.proof.' in name or name.endswith('.plan.json'),"
after="require(('.atomic-v1-' in name and name.endswith('.tmp')) or (name.startswith('owned_member_') and name.endswith('.tmp')) or '.proof.' in name or name.endswith('.plan.json'),"
assert text.count(before)==1;text=text.replace(before,after)
emit(R/'run_ASR18_full_run_owner_v11.py',text)
text=(R/'prepare_ASR18_full_owner_configuration_v4.py').read_text(encoding='utf-8-sig').replace('from asr18_full_run_parent_controller_v8 import FullOwner','from asr18_full_run_parent_controller_v9 import FullOwner')
emit(R/'prepare_ASR18_full_owner_configuration_v5.py',text)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    path=Path(path).resolve();raw=path.read_bytes();return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(path,value):
    assert not path.exists();path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n');return pin(path)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
mapping={str(S/'asr18_full_run_parent_controller_v8.py'):pin(S/'asr18_full_run_parent_controller_v9.py'),
 str(R/'run_ASR18_full_run_owner_v10.py'):pin(R/'run_ASR18_full_run_owner_v11.py'),
 str(R/'prepare_ASR18_full_owner_configuration_v4.py'):pin(R/'prepare_ASR18_full_owner_configuration_v5.py')}
manifest=read(A/'full_owner_source_contract_v5.json')
manifest['bindings']=[mapping.get(item['path'],item) for item in manifest['bindings']]
for old,new in mapping.items():manifest['direct_sources'].pop(Path(old).name,None);manifest['direct_sources'][Path(new['path']).name]=new
manifest_pin=write(A/'full_owner_source_contract_v6.json',manifest)
identity=read(A/'full_owner_validation_identity_v5.json');identity['full_run_source_contract']=manifest_pin
controller=pin(S/'asr18_full_run_parent_controller_v9.py');parent=pin(R/'run_ASR18_full_run_owner_v11.py')
identity['persistent_prefix_sources'].update(metadata_resolver=controller,native_parent=parent,recovery_parent=parent)
identity_pin=write(A/'full_owner_validation_identity_v6.json',identity)
inputs=read(A/'full_owner_exact_ROOT_review_inputs_v6.json')
inputs.update(source_manifest=manifest_pin,identity=identity,identity_binding=identity_pin,run_identity_sha256=digest(identity),
 owner_verifier_source=controller,parent_source=parent,metadata_builder_source=pin(R/'prepare_ASR18_full_owner_configuration_v5.py'))
inputs['lifecycle_callback_sources']={name:controller for name in inputs['lifecycle_callback_sources']}
inputs['proof_callback_sources'].update(verify_pending_window=controller,verify_writer_authority=controller,verify_durable_resume=controller)
inputs['DEV_execution_sources'].update(parent_source=parent,guard_source=parent)
inputs['DEV_native_boundary_sources'].update(boundary_verifier=parent,native_parent=parent,recovery_parent=parent)
inputs['DEV_source_bindings'].pop('asr18_full_run_parent_controller_v8.py',None)
inputs['DEV_source_bindings']['asr18_full_run_parent_controller_v9.py']=controller
input_pin=write(A/'full_owner_exact_ROOT_review_inputs_v7.json',inputs)
print(json.dumps({'source_contract':manifest_pin,'identity':identity_pin,'ROOT_inputs':input_pin,'bindings':len(manifest['bindings'])}))
