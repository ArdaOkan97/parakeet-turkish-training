"""NEW exact closed-producer empty-log sealing; frozen trial01 stays untouched."""
import ast,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def emit(p,t):
    assert not p.exists();ast.parse(t);compile(t,str(p),'exec');p.write_text(t,encoding='utf-8',newline='\n');return pin(p)
def write(p,v):
    assert not p.exists();p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n');return pin(p)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
old=S/'asr18_full_run_parent_controller_v9.py';new=S/'asr18_full_run_parent_controller_v10.py'
text=old.read_text(encoding='utf-8-sig')
method='''    def _seal_closed_empty_CPU_logs(self,plan):
        # Only logs opened by the exact actually CLOSED0 CPU producers. Empty
        # content is retained verbatim as SHA(empty)/size0 in immutable receipts.
        import stat
        root=Path(self.current_workspace['workspace'])
        report=read_bound(self.current_CPU_report,self.parent.check)
        contract=read_bound(report['contract'],self.parent.check)
        require(report['bridge']==self.current_bridge and report['all_children_actually_closed0'] is True
            and report['parent_source']==self.parent_source,'Same actual closed CPU8 window before empty-log sealing')
        sources=contract['sources'];allowed=[]
        def add(path,proof_pin,worker,parent):
            proof=self._verify_closed(proof_pin)
            require(proof['kind']=='ASR18_owned_CPU_process_completion_v1'
                and proof['worker_source']==worker and proof['parent_source']==parent,
                'Exact source-bound CLOSED0 producer of this optional log')
            path=Path(path)
            require(path.is_relative_to(root) and path.name=='worker.log','Only this new registered AUX optional worker.log')
            require(Path(proof_pin['path'])==path.parent/'actual_owned_process0.json',
                'Log and actual producer completion share the exact owned directory')
            for ancestor in (path,*path.parents):
                info=ancestor.lstat()
                require(not (getattr(info,'st_file_attributes',0)&0x400) and not ancestor.is_symlink(),
                    'No reparse/symlink log or ancestor')
                if ancestor==ROOT:break
            info=path.lstat();require(stat.S_ISREG(info.st_mode),'Exact regular owned optional log')
            if info.st_size==0:
                original=record(path)
                require(original['bytes']==0 and original['sha256']==hashlib.sha256(b'').hexdigest(),
                    'Only genuinely empty bytes, never discard output')
                allowed.append({'original':original,'actual_process0':proof_pin,
                    'producer':proof['producer'],'worker_source':worker,'parent_source':parent})
        add(root/'CPU/worker.log',self.current_CPU0,self.config['CPU_controller_source'],self.parent_source)
        add(root/'cleanup/worker.log',self.current_cleanup0,self.parent_source,self.parent_source)
        require(len(report['frontend_proofs'])==len(report['independent_reference_proofs'])==len(plan['updates']),
                'One exact closed frontend/reference pair per prepared update')
        for i in range(len(plan['updates'])):
            for kind,key,source_key in (('frontend','frontend_proofs','feature_worker'),
                                        ('reference','independent_reference_proofs','reference_worker')):
                item=report[key][i]
                add(root/'CPU/payload'/('%s_child_%02d'%(kind,i))/'worker.log',item['actual0'],
                    sources[source_key],self.config['CPU_controller_source'])
        require(len(allowed)<=42,'At most two optional leaf logs per20 plus controller/cleanup')
        require(self.gpu is not None,'Empty-log sealing only at actual resident idle, no fabricated closed boundary')
        self.gpu.check();pointer=read_bound(self.current_capture['pointer'],self.parent.check)
        verify_resident_idle(read_bound(self.current_idle,self.parent.check),pointer=pointer,
            proof_binding=self.current_idle,phase=plan['phase'],guard=self.parent.check)
        release=read_bound(self.current_release,self.parent.check)
        require(release['producer']==self.gpu.producer and release['worker_source']==self.config['worker_driver_source']
            and release['request_nonce']==self.current_response['nonce'] and release['writer_intent']==self.current_intent
            and release['checkpoint_pointer']==self.current_capture['pointer']
            and all(release[k] is True for k in ('writer_context_returned','persistence_reader_context_returned',
                                                'startup_restore_context_returned')),
            'Actual native idle and returned candidate contexts before optional empty-log unlink')
        intent=self._new(root/'empty_CPU_log_retirement_intent.json',{
            'kind':'ASR18_exact_closed_empty_CPU_log_retirement_intent_v1','owner':OWNER,
            'source':record(__file__),'configuration':self.config_pin,'workspace':self.current_workspace,
            'native_idle':self.current_idle,'native_release_ack':self.current_release,
            'empty_logs':allowed,'nonempty_log_content_retirement_permitted':False})
        for row in allowed:
            path=Path(row['original']['path']);info=path.lstat()
            require(stat.S_ISREG(info.st_mode) and not (getattr(info,'st_file_attributes',0)&0x400)
                and record(path)==row['original'] and info.st_size==0,'Same exact regular empty log immediately before unlink')
            require(record(intent['path'])==intent,'Durable complete empty-byte/source/closure receipt before unlink')
            path.unlink();require(not path.exists(),'Exact new optional empty log retired')
        return self._new(root/'empty_CPU_log_retirement_completed.json',{
            'kind':'ASR18_exact_closed_empty_CPU_log_retirement_completed_v1','passed':True,
            'intent':intent,'retired_exact_empty_logs':len(allowed),'discarded_content_bytes':0,
            'all_original_empty_path_SHA_and_producer_proofs_retained':True})

'''
assert text.count('    def _archive_and_publish(self,plan,completion):')==1
text=text.replace('    def _archive_and_publish(self,plan,completion):',method+'    def _archive_and_publish(self,plan,completion):')
before="        workspace=Path(self.current_workspace['workspace']);rows=[]\n"
assert text.count(before)==1;text=text.replace(before,before+"        self._seal_closed_empty_CPU_logs(plan)\n")
controller=emit(new,text)
oldparent=R/'run_ASR18_full_run_owner_v11.py';parent=R/'run_ASR18_full_run_owner_v12.py'
parent_pin=emit(parent,oldparent.read_text(encoding='utf-8-sig').replace('from asr18_full_run_parent_controller_v9 import run','from asr18_full_run_parent_controller_v10 import run'))
oldbuilder=R/'prepare_ASR18_full_owner_configuration_v6.py';builder=R/'prepare_ASR18_full_owner_configuration_v7.py'
builder_pin=emit(builder,oldbuilder.read_text(encoding='utf-8-sig').replace('from asr18_full_run_parent_controller_v9 import FullOwner','from asr18_full_run_parent_controller_v10 import FullOwner'))
mapping={str(old):controller,str(oldparent):parent_pin,str(oldbuilder):builder_pin}
manifest=read(A/'full_owner_source_contract_v7.json')
manifest['bindings']=[mapping.get(x['path'],x) for x in manifest['bindings']]
for oldpath,newpin in mapping.items():manifest['direct_sources'].pop(Path(oldpath).name,None);manifest['direct_sources'][Path(newpin['path']).name]=newpin
manifest_pin=write(A/'full_owner_source_contract_v8.json',manifest)
identity=read(A/'full_owner_validation_identity_v7.json');identity['full_run_source_contract']=manifest_pin
identity['persistent_prefix_sources'].update(metadata_resolver=controller,native_parent=parent_pin,recovery_parent=parent_pin)
identity_pin=write(A/'full_owner_validation_identity_v8.json',identity)
inputs=read(A/'full_owner_exact_ROOT_review_inputs_v8.json')
inputs.update(source_manifest=manifest_pin,identity=identity,identity_binding=identity_pin,run_identity_sha256=digest(identity),
    owner_verifier_source=controller,parent_source=parent_pin,metadata_builder_source=builder_pin,
    run_namespace=str(Path(r'E:\gpu_training_scratchpad\runs\ASR18_full_owner_validation_08_02')),
    actual_folder=str(A/'full_owner_eight_update_actual_02'),configuration_output=str(A/'full_owner_eight_update_configuration_03'),
    preserved_native_trial01_actual1=pin(A/'full_owner_eight_update_actual_01/actual_owned_CLOSED1_v1.json'))
inputs['storage_namespace']=str(Path(inputs['run_namespace'])/'checkpoint_storage');inputs['durable_namespace']=str(Path(inputs['run_namespace'])/'durable')
inputs['lifecycle_callback_sources']={k:controller for k in inputs['lifecycle_callback_sources']}
inputs['proof_callback_sources'].update(verify_pending_window=controller,verify_writer_authority=controller,verify_durable_resume=controller)
inputs['DEV_execution_sources'].update(parent_source=parent_pin,guard_source=parent_pin)
inputs['DEV_native_boundary_sources'].update(boundary_verifier=parent_pin,native_parent=parent_pin,recovery_parent=parent_pin)
inputs['DEV_source_bindings'].pop('asr18_full_run_parent_controller_v9.py',None);inputs['DEV_source_bindings'][new.name]=controller
inputs_pin=write(A/'full_owner_exact_ROOT_review_inputs_v9.json',inputs)
print(json.dumps({'controller':controller,'parent':parent_pin,'builder':builder_pin,'source_manifest':manifest_pin,
    'identity':identity_pin,'run_identity_sha256':digest(identity),'ROOT_inputs':inputs_pin}))
