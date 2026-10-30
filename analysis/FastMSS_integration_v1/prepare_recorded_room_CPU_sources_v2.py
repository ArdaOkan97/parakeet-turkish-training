"""Immutable source successors: full-stage admission and fresh frontend admission."""
import ast,copy,hashlib,json
from pathlib import Path
P=Path(r'C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
F=Path(r'E:/gpu_training_scratchpad/analysis/FastMSS_integration_v1')
def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
def source(path,text):
    ast.parse(text)
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:f.write(text)
adapter=S/'fastmss_recorded_room_train_v2.py';protocol=S/'fastmss_recorded_room_feature_protocol_v2.py'
reference=R/'produce_FastMSS_recorded_room_reference_CPU_v2.py'
frontend=R/'produce_FastMSS_recorded_room_frontend_CPU_v2.py'
controller=R/'run_ASR18_recorded_room_CPU_window_v2.py'
for path in (adapter,protocol,reference,frontend,controller):assert not path.exists()
t=(S/'fastmss_recorded_room_train_v1.py').read_text()
old="            extra=new_raw+plan['compressed_media_cap_bytes']+workspace+review['numerical_runtime_reserve_bytes']"
assert old in t
t=t.replace(old,"            whole_workspace=max(workspace,*plan['workspace_incremental_envelopes'].values())\n"
    "            extra=new_raw+plan['compressed_media_cap_bytes']+whole_workspace+review['numerical_runtime_reserve_bytes']")
t=t.replace("'raw_bytes':new_raw,'FFT_workspace_bytes':workspace,", "'raw_bytes':new_raw,'FFT_workspace_bytes':workspace,'whole_stage_workspace_bytes':whole_workspace,")
t=t.replace("'raw_reserved_bytes':new_raw,'FFT_workspace_bytes':workspace,'adapter':self.identity['adapter']})",
    "'raw_reserved_bytes':new_raw,'FFT_workspace_bytes':workspace,'whole_stage_workspace_bytes':whole_workspace,\n"
    "                'prospective_incremental_bytes':extra,'adapter':self.identity['adapter']})")
old='    def _observe(self,array,name):'
assert old in t
t=t.replace(old,'''    def frontend(self,operation,wave):
        # Transform libraries/caches may have grown since the original decode
        # admission. Re-admit the complete current raw/media/MAX(all stages)
        # domain immediately before the unchanged original frontend allocates.
        key=self.reader.active_job
        mixer.require(key is not None and not self.reader.ledger.jobs[key]['stages'],
                      'Fresh quiescent raw stage before original frontend')
        plan=self.reader.ledger.jobs[key]['plan'];review=self.reader.raw_contract
        workspace=max(plan['workspace_incremental_envelopes'].values())
        extra=plan['raw_peak_bytes']+plan['compressed_media_cap_bytes']+workspace+review['numerical_runtime_reserve_bytes']
        value=self.guard()
        mixer.require(value['worker_private_memory']['private_bytes']+extra<=self.reader.private_limit,
                      'Post-transform current private plus complete frontend/raw/media/workspace domain exceeds leaf cap')
        import fastmss_row_serial_cpu_v8 as core
        admitted=core.admit(extra,self.guard)
        self.reader.ledger.events.append({'event':'recorded_room_fresh_full_domain_before_frontend','job':key,
            'raw_reserved_bytes':plan['raw_peak_bytes'],'whole_stage_workspace_bytes':workspace,
            'prospective_incremental_bytes':extra,'adapter':self.identity['adapter']})
        return self.reader.frontend(operation,wave)

    def _observe(self,array,name):''')
source(adapter,t)

renames={'fastmss_recorded_room_train_v1':'fastmss_recorded_room_train_v2',
    'fastmss_recorded_room_feature_protocol_v1':'fastmss_recorded_room_feature_protocol_v2',
    'produce_FastMSS_recorded_room_reference_CPU_v1':'produce_FastMSS_recorded_room_reference_CPU_v2',
    'produce_FastMSS_recorded_room_frontend_CPU_v1':'produce_FastMSS_recorded_room_frontend_CPU_v2',
    'run_ASR18_recorded_room_CPU_window_v1':'run_ASR18_recorded_room_CPU_window_v2'}
for old_path,new_path in ((S/'fastmss_recorded_room_feature_protocol_v1.py',protocol),
    (R/'produce_FastMSS_recorded_room_reference_CPU_v1.py',reference),
    (R/'produce_FastMSS_recorded_room_frontend_CPU_v1.py',frontend),
    (R/'run_ASR18_recorded_room_CPU_window_v1.py',controller)):
    t=old_path.read_text()
    for old,new in renames.items():t=t.replace(old,new)
    if new_path==frontend:
        assert 'return reader.frontend(original_frontend, wave)' in t
        t=t.replace('return reader.frontend(original_frontend, wave)','return augmenter.frontend(original_frontend, wave)')
    source(new_path,t)

contract=copy.deepcopy(load(F/'recorded_room_CPU_source_contract_v1.json'))
contract['sources'].update(augmentation_adapter=pin(adapter),protocol=pin(protocol),reference=pin(reference),
    frontend=pin(frontend),controller=pin(controller))
contract['augmentation_identity']['adapter']=pin(adapter)
contract['raw_extension_full_stage_admission']=True
contract['fresh_post_transform_frontend_full_domain_admission']=True
cp=F/'recorded_room_CPU_source_contract_v2.json';write(cp,contract)
cfg=copy.deepcopy(load(F/'recorded_room_CPU_configuration_v2.json'))
cfg['source_contract']=pin(cp)
cfg['sources'].update(augmentation_adapter=pin(adapter),protocol=pin(protocol),reference=pin(reference),
    frontend=pin(frontend),parent=pin(controller))
cfg_path=F/'recorded_room_CPU_configuration_v3.json'
cfg['execution'].update(script=pin(controller),arguments=['--config',str(cfg_path),'--ROOT-review','EXTERNAL_ROOT_CPU_REVIEW_PATH','--folder',cfg['output_root']])
cfg['scope'].append('New raw reservation includes MAX(all original stage workspaces,new RIR FFT) plus raw/media/runtime; fresh full-domain private/global admission immediately before original frontend after transform growth.')
write(cfg_path,cfg)
m=copy.deepcopy(load(F/'recorded_room_CPU_source_interface_manifest_v2.json'))
m.update(kind='FastMSS_recorded_room_CPU_source_interface_manifest_v3',sources=contract['sources'],
    configuration=pin(cfg_path),source_contract=pin(cp),
    sole_successor_delta='Adapter2 full MAX stage domain admission +fresh frontend admission; other source changes only explicit import/path pins and frontend delegates through that new admission.')
mp=F/'recorded_room_CPU_source_interface_manifest_v3.json';write(mp,m)
proposal=copy.deepcopy(load(F/'recorded_room_CPU_source_resource_proposal_v2.json'))
proposal.update(kind='FastMSS_recorded_room_CPU_first4_source_resource_proposal_v3',configuration=pin(cfg_path),
    source_contract=pin(cp),source_manifest=pin(mp),sources=contract['sources'],sole_successor_delta=m['sole_successor_delta'])
proposal['new_raw_extension'].update(private_preallocation_formula='newraw+compressed+MAX(all original stageworkspaces,newFFT)+runtime',
    fresh_post_transform_frontend_full_domain_admission=True,
    original_frontend_spectral_allowance_bytes=469762048)
pp=F/'recorded_room_CPU_source_resource_proposal_v3.json';write(pp,proposal)
h=copy.deepcopy(load(F/'recorded_room_CPU_source_handoff_v2.json'))
h.update(kind='FastMSS_recorded_room_CPU_source_handoff_v3',controller=pin(controller),reader=pin(protocol),
    configuration=pin(cfg_path),source_contract=pin(cp),manifest=pin(mp),proposal=pin(pp),CLI=cfg['execution']['arguments'])
hp=F/'recorded_room_CPU_source_handoff_v3.json';write(hp,h)
print(json.dumps({'handoff':pin(hp),'adapter':pin(adapter),'controller':pin(controller),'reader':pin(protocol),
    'config':pin(cfg_path),'contract':pin(cp),'manifest':pin(mp),'proposal':pin(pp)},indent=2))
