"""Fresh controller from proved merged/UTF8 services; no RUN02 mutation."""
import ast,difflib,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
old=S/'asr18_BPE9223_full_run_controller_v18.py';s=old.read_text(encoding='utf8')
def change(a,b,count=1):
 global s
 assert s.count(a)==count,(a[:90],s.count(a),count);s=s.replace(a,b)
def replace_function(name,value,cls=None):
 global s
 tree=ast.parse(s);nodes=tree.body if cls is None else next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name==cls).body
 n=next(x for x in nodes if isinstance(x,ast.FunctionDef)and x.name==name)
 lines=s.splitlines(True);s=''.join(lines[:n.lineno-1])+value.rstrip()+'\n'+''.join(lines[n.end_lineno:])

# Historic abnormal receipt routines remain frozen in controller18, not imported
# or called by this explicitly authorized new genesis/normal-continuation lane.
tree=ast.parse(s);lines=s.splitlines(True)
remove=[(n.lineno,n.end_lineno)for n in tree.body if isinstance(n,ast.FunctionDef)and n.name not in ('checked_normal_resume_request','_original_proof_sequence','run','main')]
for a,b in reversed(remove):lines[a-1:b]=[]
s=''.join(lines)
dictionary=(N/'witness80_dictionary_v2_actual_01/dictionary_v2.bin').read_bytes()
pos=s.index('\n',s.index('WITNESS_ZDICT='))+1
s=s[:pos]+"WITNESS_ZDICT_V2="+repr(dictionary)+'''

def witness_decoder(packed):
    require(type(packed)is bytes and len(packed)>=6,'Complete dictionary zlib header')
    cmf,flg=packed[0],packed[1]
    require(cmf&15==8 and cmf>>4<=7 and (cmf*256+flg)%31==0 and flg&32,'Source dictionary witness stream')
    dictionary_id=int.from_bytes(packed[2:6],'big')
    dictionaries={zlib.adler32(WITNESS_ZDICT):WITNESS_ZDICT,zlib.adler32(WITNESS_ZDICT_V2):WITNESS_ZDICT_V2}
    require(len(dictionaries)==2 and dictionary_id in dictionaries,'Only explicit legacy/V2 generic schema DICTIDs')
    return zlib.decompressobj(zdict=dictionaries[dictionary_id])

'''+s[pos:]
# Only the normal resume frame reader, sequence reader and writer remain active.
change('packed=stream.read(packed_size);decoder=zlib.decompressobj(zdict=WITNESS_ZDICT)',
 'packed=stream.read(packed_size);decoder=witness_decoder(packed)')
change('raw=canonical(event);encoder=zlib.compressobj(level=9,zdict=WITNESS_ZDICT)',
 'raw=canonical(event);encoder=zlib.compressobj(level=9,zdict=WITNESS_ZDICT_V2)')
change('decoder=zlib.decompressobj(zdict=WITNESS_ZDICT)\n        decoded=decoder.decompress(actual[40:],65537)',
 'decoder=witness_decoder(actual[40:])\n        decoded=decoder.decompress(actual[40:],65537)')
s=s.replace('from asr18_BPE9223_durable_owner_adapter_v8 import','from asr18_BPE9223_durable_owner_adapter_v10 import')
s=s.replace('from asr18_BPE9223_indexed_checkpoint_storage_v2 import IndexedOwnedFutureCheckpointStore',
 'from asr18_BPE9223_indexed_checkpoint_storage_v3 import IndexedOwnedFutureCheckpointStore')
# The exact immutable readonly reader remains indexed2/resolver3, distinct from
# mutable fresh3. This also resolves archived stubs in a new normal owner.
change("    local_store=IndexedOwnedFutureCheckpointStore(c['operational_storage_owner'],guard=guard,\n        operational_compatibility=c['operational_compatibility'])", 
 "    from asr18_BPE9223_indexed_checkpoint_storage_v2 import IndexedOwnedFutureCheckpointStore as ReadonlyStore\n    readonly=c['process_readonly_metadata_authority']\n    local_store=ReadonlyStore(c['operational_storage_owner'],guard=guard,\n        operational_compatibility=readonly['operational_compatibility'])")
change("metadata_phase=phase,identity=c['full_run_identity'],operational_compatibility=c['operational_compatibility'],", 
 "metadata_phase=phase,identity=c['full_run_identity'],operational_compatibility=readonly['operational_compatibility'],")
change("        if Path(pin['path']).exists():return physical_or_configured(pin,guard)",
 "        if Path(pin['path']).is_file() and Path(pin['path']).stat().st_size==pin['bytes']:return physical_or_configured(pin,guard)")

# Keep the successful current services/numerics; restore only original genesis
# constructor choice and reject every abnormal/adoption path in this new run.
start=s.index("        require(resume_request is not None,'Operational owner requires")
end=s.index("        administrative=r.get('administrative_maximum_new_updates')",start)
s=s[:start]+'''        self.committed60_initial=self.prepared60_initial=self.ACKed60_initial=False
        self.reserved_committed60=self.ACKed40_initial=self.pending40_initial=self.pending40_finalization=False
        self.resource_initial_pending=self.acknowledged_resource_pending=False
        self.resource_closure=None;self.reserved_preparation=None;self.prepared_CPU_resume=None;self.pending40_inputs=None
        checked_compatibility(c['operational_compatibility'],self.identity)
        if resume_request is not None:
            request=read_bound(resume_request)
            require(request['kind']=='ASR18_exact_normal_boundary_owner_resume_request_v1','Fresh RUN03 supports genuine normal continuation only')
            self.resume_snapshot=checked_normal_resume_request(self.config_pin,r,resume_request,parent_source=parent_source,guard=lambda:None)
            require(self.root.is_dir() and self.control.is_dir(),'Only exact normally closed same namespace may reopen')
        else:
            require(r['invocation']=='genesis_authorized_production' and r['fresh_genesis_authorized'] is True
                and r['fresh_genesis_namespace']==str(self.root) and not self.root.exists(),
                'Explicit new original-seed genesis; never reset/adopt RUN02')
'''+s[end:]
start=s.index('            snapshot=(checked_prepared60_UTF8_resume_request(')
end=s.index("            self.schedule.load_state_dict",start)
s=s[:start]+'''            snapshot=checked_normal_resume_request(self.config_pin,r,resume_request,parent_source=parent_source,guard=self.parent.check)
'''+s[end:]
change("proofs=proofs,create=False,\n                resume_proof=None if self.resource_initial_pending or self.committed60_initial else self.current_ack,", 
 "proofs=proofs,create=not resuming,\n                resume_proof=None if not resuming else self.current_ack,")
start=s.index("                resource_resume=self.resource_closure if self.resource_initial_pending else None,")
end=s.index("            require(self.durable.owner",start)
s=s[:start]+s[start:end].splitlines(True)[0][:0]+s[end:] if False else s
# Keep the operational_compatibility preceding these three old keyword lines.
change("                operational_compatibility=c['operational_compatibility'],\n                resource_resume=self.resource_closure if self.resource_initial_pending else None,\n                committed60_CPU_resume=c['prior_committed60_CPU_memory_failure'] if self.committed60_initial else None,\n                acknowledged_committed60_resume=c['prior_acknowledged_committed60_stop'] if self.ACKed60_initial else None)",
 "                operational_compatibility=c['operational_compatibility'])")
change("protected=c['protected_external'],guard=self.parent.check))", "protected=c['protected_external'],guard=self.parent.check,operational_compatibility=c['operational_compatibility']))")
start=s.index('        if self.acknowledged_resource_pending:\n')
end=s.index('        self.pool_verify_protection=',start)
s=s[:start]+s[end:]

replace_function('_worker_control_protection','''    def _worker_control_protection(self):
        state=read_bound(self.parent.control_pool.state_pin,self.parent.check);folders=set()
        if self.gpu is not None:folders.add(str(self.gpu.folder))
        for key in ('closed_native','closed_recovery'):
            pin=getattr(self,key,None)
            if pin is not None:folders.add(str(Path(pin['path']).parent))
        folders.update(x['folder']for x in state['live']if Path(x['folder']).name.startswith('DEV_'))
        folders&={x['folder']for x in state['live']}
        files=[self.config_pin,self.config['session'],self.config['operational_compatibility']]
        files.extend(x['configuration']for x in state['live']if x['folder']in folders)
        if self.resume_request is not None:
            request=read_bound(self.resume_request,self.parent.check)
            require(request['kind']=='ASR18_exact_normal_boundary_owner_resume_request_v1','No abnormal-source continuation in fresh RUN03')
            files.append(self.resume_request)
            files.extend(request[k]for k in ('normal_pause_snapshot','prior_owner_report','prior_owner_actual_TOOL0','prior_owner_guard'))
        unique={x['path']:x for x in files}
        return sorted(folders),[unique[key]for key in sorted(unique)]
''','FullOwner')
replace_function('verify_durable_resume','''    def verify_durable_resume(self,request):
        self.check();require(self.gpu is None and self.current_ack is not None and request['restore_proof']==self.current_ack
            and request['identity']==self.identity and request['namespace']==str(self.root/'durable')
            and request.get('native_replay_or_sampler_rollback_permitted') is False,'Exact current recovery before same-run normal reopen')
        value=read_bound(self.current_ack,self.parent.check)
        require(request['state']['last_recovery']==self.current_ack and request['state']['pending']['publications']==0
            and value['final_checkpoint_pointer']==self.current_capture['pointer'],'No rollback or abnormal owner0 fiction')
        self._verify_closed(self.closed_native);self._verify_closed(self.closed_recovery);return True
''','FullOwner')
replace_function('_verify_closed','''    def _verify_closed(self,pin):
        proof=read_bound(pin,self.parent.check)
        verify(proof,kind=proof['kind'],owner=OWNER,guard=self.parent.check);return proof
''','FullOwner')
replace_function('_current_capture_writer_source',"    def _current_capture_writer_source(self):\n        return self.config['worker_driver_source']\n",'FullOwner')
change("'resource_journal_handoff':self.closed_native==self.resource_closure", "'resource_journal_handoff':False")
# These whole old recovery implementations cannot be entered by this new lane.
tree=ast.parse(s);lines=s.splitlines(True);cl=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name=='FullOwner')
remove_names={'_verify_pending40_quiescence','_finish_pending40','_pending40_producer_table','_seal_reconciled_empty_CPU_logs',
 '_recover_committed60','_recover_resource_stop','_reuse_reserved_resource_metadata','_reuse_reserved_ACKed40_metadata','_reuse_reserved_committed60_metadata'}
for n in reversed(cl.body):
 if isinstance(n,ast.FunctionDef)and n.name in remove_names:lines[n.lineno-1:n.end_lineno]=[]
s=''.join(lines)
change("        if self.committed60_initial:self._recover_committed60()\n        if self.pending40_initial:self._finish_pending40()\n        if self.resource_initial_pending:self._recover_resource_stop()\n",'')
# Entry is truthful new controller code calling frozen GPU/CPU/guard services.
change("parser.add_argument('--resume-request',required=True)","parser.add_argument('--resume-request')")
change('ParentGuard=frozen.ParentGuard,','ParentGuard=FreshParentGuard,')
change('resume_request=record(a.resume_request))','resume_request=None if a.resume_request is None else record(a.resume_request))')
change("'Actual controller18 CLI entry distinct", "'Actual fresh controller19 CLI entry distinct")
change("    return run(a.config,a.review,Path(a.folder).resolve()",'''    class FreshParentGuard(frozen.ParentGuard):
        def configure_control_quota(self,control,policy,*,identity,create,operational_compatibility):
            require(policy==frozen.CONTROL_POLICY,'Same exact control quotas')
            self.control=Path(control).resolve()
            require(self.control.is_relative_to(ROOT/'runs'),'Exact new E namespace')
            from asr18_worker_control_pool_v3 import WorkerControlPool
            self.control_pool=WorkerControlPool(self.control,identity=identity,policy=policy,
                authority_source=record(frozen.frozen_parent.__file__),guard=self.pool_guard,metadata_phase=self.metadata_phase,
                admit_control=self.control_guard,operational_compatibility=operational_compatibility,create=create)
            self.worker_slots=list(self.control_pool.live_folders());self.control_guard()
    return run(a.config,a.review,Path(a.folder).resolve()''')
new=S/'asr18_BPE9223_full_run_controller_v19.py';assert not new.exists();ast.parse(s)
assert len(s.encode('utf8'))<=262144
new.write_text(s,encoding='utf8',newline='\n')
area=N/'fresh_RUN03_source_v1';area.mkdir(exist_ok=False)
(area/'controller18_to19.diff.txt').write_text(''.join(difflib.unified_diff(old.read_text(encoding='utf8').splitlines(True),s.splitlines(True),fromfile=str(old),tofile=str(new))),encoding='utf8')
def pin(p):
 raw=p.read_bytes();return {'path':str(p.resolve()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
handoff={'controller':pin(new),'parent_unchanged':pin(R/'run_ASR18_BPE9223_full_run_owner_v19.py'),
 'runtime_unchanged':pin(R/'asr18_BPE9223_owned_GPU_runtime_v23.py'),'native_unchanged':pin(S/'asr18_BPE9223_full_run_native_v9.py'),
 'source_scope':'explicit_new_original_seed_genesis_and_normal_continuations_only','RUN02_mutated':False,
 'witness_limits':[65536,2048,805306368],'dictionary_measurement':pin(N/'witness80_dictionary_v2_actual_01/report.json')}
(area/'source_handoff_draft.json').write_text(json.dumps(handoff,indent=2)+'\n',encoding='utf8')
print(json.dumps(handoff))
