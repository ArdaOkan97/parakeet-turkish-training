"""NEW actual prior-window CPU8 handoff; no target imports or execution."""
from pathlib import Path
import ast,hashlib,json
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def source(old,new,before,after):
    assert not new.exists();text=old.read_text(encoding='utf-8-sig');assert text.count(before)==1
    text=text.replace(before,after);ast.parse(text);compile(text,str(new),'exec');new.write_text(text,encoding='utf-8',newline='\n')
    raw=new.read_bytes();print({'path':str(new),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'AST_only':True})
before="""        self._admit_CPU_family(root/'CPU_remaining_family_admission.json')
        cpu=root/'CPU';proof=self.run_CPU(self.config,self.config['CPU_controller_source']['path'],['--bridge',str(ba),'--output',str(cpu/'payload')],
            cpu,guard=self.parent.check if self.gpu is None else self.gpu.check,report_relative='payload/report.json')"""
after="""        cpu=root/'CPU';arguments=['--bridge',str(ba),'--output',str(cpu/'payload')]
        if index>0:
            boundary=read_bound(self.prior_boundary,self.parent.check)
            self.verify_prior_boundary(boundary)
            io=_Metadata(self.identity,self.resolve_member,self.parent.check);journal=_JournalPrefixes(self.parent.check)
            try:
                # The original control plan was retired only after exact archive
                # verification. Resolve its complete immutable original identity.
                prior_plan=io.read(self.current_original_plan)
                pointer,saved=_capture(self.current_capture,self.current_capture['pointer'],io,
                    identity_sha=digest(self.identity),previous_sampler=prior_plan['initial_sampler_sha256'],
                    previous_updates=prior_plan['initial_committed_updates'],journal=journal)
                require(digest(saved)==digest(self.schedule.state_dict())==boundary['sampler_state_sha256']
                    and boundary['coherent_checkpoint_pointer']==self.current_capture['pointer'],
                    'Independent actual PRE/persistence/full-sampler/journal prefix before next CPU')
            finally:journal.close()
            if self.gpu is not None:
                self.gpu.check()
                verify_resident_idle(read_bound(self.current_idle,self.parent.check),pointer=pointer,
                    proof_binding=self.current_idle,phase=boundary['phase'] if 'phase' in boundary else prior_plan['phase'],
                    guard=self.parent.check)
            else:
                require(self.current_ack is not None,'Closed native handoff requires actual current recovery ACK')
                self._verify_closed(self.closed_native);self._verify_closed(self.closed_recovery)
            attestation={'kind':'FastMSS_native_owner_prior_window_CPU_handoff_v1','owner':OWNER,'passed':True,
                'boundary_sha256':digest(boundary),'actual_all_state_prefix_checkpoint_verified':True,
                'bridge':record(bp),'session':self.config['session'],'parent_source':self.parent_source,
                'durable_head':self.durable.head_pin,'current_durable_owner_head_and_persistence_prefix_verified':True,
                'current_checkpoint_capture':self.current_capture,'source':record(__file__),
                'saved_sampler_sha256':digest(saved),'historical_fresh_GPU_restore_claimed':False}
            ap=root/'prior_boundary_verification.json';self._new(ap,attestation)
            ab=root/'prior_boundary_verification_binding.json';self._new(ab,record(ap))
            arguments.extend(['--prior-boundary-verification',str(ab)])
        self._admit_CPU_family(root/'CPU_remaining_family_admission.json')
        proof=self.run_CPU(self.config,self.config['CPU_controller_source']['path'],arguments,
            cpu,guard=self.parent.check if self.gpu is None else self.gpu.check,report_relative='payload/report.json')"""
source(S/'asr18_full_run_parent_controller_v7.py',S/'asr18_full_run_parent_controller_v8.py',before,after)
source(R/'run_ASR18_full_run_owner_v9.py',R/'run_ASR18_full_run_owner_v10.py',
 'from asr18_full_run_parent_controller_v7 import run','from asr18_full_run_parent_controller_v8 import run')
source(R/'prepare_ASR18_full_owner_configuration_v3.py',R/'prepare_ASR18_full_owner_configuration_v4.py',
 'from asr18_full_run_parent_controller_v7 import FullOwner','from asr18_full_run_parent_controller_v8 import FullOwner')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    path=Path(path).resolve();raw=path.read_bytes();return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(path,value):
    assert not path.exists();path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n');return pin(path)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
mapping={str(S/'asr18_full_run_parent_controller_v7.py'):pin(S/'asr18_full_run_parent_controller_v8.py'),
 str(R/'run_ASR18_full_run_owner_v9.py'):pin(R/'run_ASR18_full_run_owner_v10.py'),
 str(R/'prepare_ASR18_full_owner_configuration_v3.py'):pin(R/'prepare_ASR18_full_owner_configuration_v4.py')}
manifest=read(A/'full_owner_source_contract_v4.json')
manifest['bindings']=[mapping.get(item['path'],item) for item in manifest['bindings']]
for old,new in mapping.items():manifest['direct_sources'].pop(Path(old).name,None);manifest['direct_sources'][Path(new['path']).name]=new
manifest_pin=write(A/'full_owner_source_contract_v5.json',manifest)
identity=read(A/'full_owner_validation_identity_v4.json');identity['full_run_source_contract']=manifest_pin
controller=pin(S/'asr18_full_run_parent_controller_v8.py');parent=pin(R/'run_ASR18_full_run_owner_v10.py')
identity['persistent_prefix_sources'].update(metadata_resolver=controller,native_parent=parent,recovery_parent=parent)
identity_pin=write(A/'full_owner_validation_identity_v5.json',identity)
inputs=read(A/'full_owner_exact_ROOT_review_inputs_v5.json')
inputs.update(source_manifest=manifest_pin,identity=identity,identity_binding=identity_pin,run_identity_sha256=digest(identity),
 owner_verifier_source=controller,parent_source=parent,metadata_builder_source=pin(R/'prepare_ASR18_full_owner_configuration_v4.py'))
inputs['lifecycle_callback_sources']={name:controller for name in inputs['lifecycle_callback_sources']}
inputs['proof_callback_sources'].update(verify_pending_window=controller,verify_writer_authority=controller,verify_durable_resume=controller)
inputs['DEV_execution_sources'].update(parent_source=parent,guard_source=parent)
inputs['DEV_native_boundary_sources'].update(boundary_verifier=parent,native_parent=parent,recovery_parent=parent)
inputs['DEV_source_bindings'].pop('asr18_full_run_parent_controller_v7.py',None)
inputs['DEV_source_bindings']['asr18_full_run_parent_controller_v8.py']=controller
input_pin=write(A/'full_owner_exact_ROOT_review_inputs_v6.json',inputs)
print(json.dumps({'source_contract':manifest_pin,'identity':identity_pin,'ROOT_inputs':input_pin,'bindings':len(manifest['bindings'])}))
