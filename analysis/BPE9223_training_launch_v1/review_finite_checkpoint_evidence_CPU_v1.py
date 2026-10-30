"""Independent finite evidence metadata equations/joins; no payload or live worker."""
import copy,importlib,json,random,sys
from pathlib import Path
L=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1');sys.path.insert(0,str(L))
m=importlib.import_module('audit_and_fit_append_only_v5')
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
O=N/'independent_finite_checkpoint_evidence_review_v1';O.mkdir(exist_ok=True);m.AREA=O
S=m.PROJECT/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
def rec(path):return {k:v for k,v in m.record(path).items() if k in ('path','bytes','sha256')}
def main():
 g=m.Guard();g.phase='finite_checkpoint_evidence_metadata_only'
 h=N/'finite_serial_DEV_source_v1/source_handoff_v1.json';assert m.sha(h)=='71138c0cbe44915a61df9d5050b236787ae9bcf5899bf91206a95ac6d8d54233'
 handoff=json.loads(h.read_text());sources=handoff['sources']
 for item in sources.values():assert rec(item['path'])==item
 import asr18_finite_BPE9223_checkpoint_evidence_v1 as f
 from asr18_full_run_serial_DEV_9223_v2 import checkpoint_format_view
 tensors=[]
 def tensor(dtype='float32',shape=None):
  shape=[] if shape is None else shape;index=len(tensors);size=__import__('math').prod(shape)*(1 if dtype=='uint8' else 4)
  offset=((tensors[-1]['offset']+tensors[-1]['bytes']+63)//64*64) if tensors else 0
  tensors.append({'index':index,'dtype':dtype,'shape':shape,'offset':offset,'bytes':size,'sha256':format(index,'064x')})
  return ['tensor',index]
 weights={f'weight_{i:03}':tensor(shape=[1]) for i in range(207)};parts=[]
 for count in (17,6,192):
  states={i:{key:tensor() for key in ('step','exp_avg','exp_avg_sq')} for i in range(count)}
  parts.append({'param_groups':[{'params':list(range(count)),'lr':.00001}],'state':states})
 torch_rng=tensor('uint8',[16]);cuda_rng=tensor('uint8',[16]);assert len(tensors)==854
 def tag(value):
  if type(value) is list and len(value)==2 and value[0]=='tensor' and type(value[1]) is int:return value
  if type(value) is dict:return ['dict',[[tag(k),tag(v)] for k,v in value.items()]]
  if type(value) is tuple:return ['tuple',[tag(v) for v in value]]
  if type(value) is list:return ['list',[tag(v) for v in value]]
  return ['scalar',value]
 journal={'path':str(O/'IN_MEMORY_ONLY.journal'),'offset':0,'sha256':'0'*64}
 saved={'kind':'ASR18_broad_caller_state_v1','diagnostic_only':True,'identity':{},'source_provenance':{},'weights':weights,
  'optimizer':{'kind':'BPE9223_broad_full207_three_AdamW_mixed_fresh_clock_v1','updates':20,'identity':{},'clocks':[64270,2020,62270],'appended512_Adam_clock':20,'optimizers':parts},
  'sampler':{'fixture':True},'augmentation':{'committed_journal':journal},'python_rng':random.Random(123).getstate(),
  'numpy_rng':('MT19937',[0]*624,0,0,0.),'torch_rng':torch_rng,'cuda_rng':[cuda_rng]}
 payload={'path':str(O/'IN_MEMORY_ONLY.tensors.bin'),'bytes':tensors[-1]['offset']+tensors[-1]['bytes'],'sha256':'f'*64}
 metadata={'kind':'ASR18_streaming_named_tensor_checkpoint_v1','schema_version':1,'endianness':'little','source':rec(S/'bpe_streaming_checkpoint_v1.py'),
  'maximum_CPU_tensor_slice_bytes':4*1024**2,'payload':payload,'tensors':tensors,'tree':tag(saved),'dynamic_module_serialization':False}
 format_result=checkpoint_format_view(metadata);assert format_result['weight_count']==207 and format_result['optimizer_updates']==20
 fp=f.metadata_fingerprint(metadata);assert fp['state']['updates']==20 and fp['state']['clocks']==[64270,2020,62270]
 assert len(fp['state']['weights'])==207 and sum(len(p['state']) for p in fp['state']['Adam_partitions'])==215
 assert set(fp['state']['RNG'])=={'python','numpy','torch','CUDA'}
 assert f.normalize_fingerprint(json.loads(json.dumps(fp)))==fp
 def independent_decode(node):
  kind,value=node
  if kind=='tensor':
   e=tensors[value];return {'shape':e['shape'],'dtype':'torch.'+e['dtype'],'bytes':e['bytes'],'sha256':e['sha256']}
  if kind=='scalar':return value
  if kind=='dict':return {independent_decode(k):independent_decode(v) for k,v in value}
  items=[independent_decode(x) for x in value];return tuple(items) if kind=='tuple' else items
 decoded=independent_decode(metadata['tree']);opt=decoded['optimizer']
 expected={'kind':'ASR18_actual_live207_threeAdam_all4RNG_fingerprint_v1','updates':20,'identity':opt['identity'],'clocks':opt['clocks'],'weights':decoded['weights'],
  'Adam_partitions':[{'param_groups':p['param_groups'],'state':p['state']} for p in opt['optimizers']],
  'RNG':{'python':decoded['python_rng'],'numpy':decoded['numpy_rng'],'torch':decoded['torch_rng'],'CUDA':decoded['cuda_rng']},'maximum_transfer_slice_bytes':4*1024**2}
 assert f.same(fp,{'state_sha256':f.digest(expected),'state':expected})
 negatives=[]
 def reject(name,operation):
  try:operation()
  except (KeyError,ValueError,AssertionError) as error:negatives.append({'name':name,'rejected':True,'reason':str(error)})
  else:raise AssertionError('Negative finite evidence fixture accepted: '+name)
 for mode in ('missing_tensor','duplicate_index','shape_bytes_mismatch','unreferenced_tensor'):
  bad=copy.deepcopy(metadata)
  if mode=='missing_tensor':bad['tensors'].pop()
  elif mode=='duplicate_index':bad['tensors'][-1]['index']=852
  elif mode=='shape_bytes_mismatch':bad['tensors'][0]['shape']=[2]
  else:
   tree=copy.deepcopy(saved);tree['torch_rng']=['tensor',853];bad['tree']=tag(tree)
  reject(mode,lambda b=bad:f.metadata_fingerprint(b))
 for mode in ('old_kind','bool_clock','wrong_clock','missing_clock'):
  bad=copy.deepcopy(saved)
  if mode=='old_kind':bad['optimizer']['kind']='old8711'
  elif mode=='bool_clock':bad['optimizer']['appended512_Adam_clock']=True
  elif mode=='wrong_clock':bad['optimizer']['clocks'][1]-=1
  else:del bad['optimizer']['appended512_Adam_clock']
  meta=copy.deepcopy(metadata);meta['tree']=tag(bad);reject(mode,lambda b=meta:checkpoint_format_view(b))
 for mode in ('weight_count206','four_Adam_partitions','duplicate_Adam_index','missing_Adam_moment'):
  bad=copy.deepcopy(saved)
  if mode=='weight_count206':bad['weights'].pop('weight_206')
  elif mode=='four_Adam_partitions':bad['optimizer']['optimizers'].append(copy.deepcopy(parts[0]))
  elif mode=='duplicate_Adam_index':bad['optimizer']['optimizers'][0]['param_groups'][0]['params'].append(0)
  else:del bad['optimizer']['optimizers'][0]['state'][0]['exp_avg_sq']
  meta=copy.deepcopy(metadata);meta['tree']=tag(bad);reject(mode,lambda b=meta:f.metadata_fingerprint(b))
 source=rec(f.RUNNER);config={'path':str(O/'IN_MEMORY_ONLY.config.json'),'bytes':1,'sha256':'1'*64};report_pin={'path':str(O/'IN_MEMORY_ONLY.report.json'),'bytes':1,'sha256':'2'*64}
 identity={'pid':1234,'ppid':1233,'create_time':1.};guard_pin={'path':str(O/'IN_MEMORY_ONLY.guard.json'),'bytes':1,'sha256':'3'*64}
 guard={'pid':1234,'actual_exit_code':0,'monitor_clean':True,'monitor_joined':True,'error':None,'samples':3}
 report={'passed':True,'producer':identity,'source':source}
 def read(pin):return copy.deepcopy(guard if pin==guard_pin else report)
 proof={'kind':f.PROCESS_KIND,'actual_Popen_wait_code':0,'all_captured_identities_dead':True,'source':source,'configuration':config,'report':report_pin,
  'captured_identities':[identity],'native':identity,'guard':guard_pin}
 assert f.checked_process(proof,source=source,configuration=config,report=report_pin,read_artifact=read,check_dead=lambda v:True) is proof
 owner={'kind':f.TOOL_KIND,'owner':f.OWNER,'actual_native_exit_code':0,'actual_exec_tool_exit_code':0,'actual_tool_chunk_id':'synthetic_scope_not_actual_TOOL0','source':source,'configuration':config,'report':report_pin,'native_PID':1234,'producer':identity,'guard_closure':guard_pin}
 assert f.checked_owner(owner,source=source,configuration=config,report=report_pin,read_artifact=read,check_dead=lambda v:True) is owner
 for key,bad_value in (('kind','ASR18_owned_GPU_process_completion_v1'),('actual_Popen_wait_code',False),('actual_Popen_wait_code',1),('all_captured_identities_dead',False),('configuration',report_pin)):
  bad=copy.deepcopy(proof);bad[key]=bad_value;reject('process_'+key,lambda b=bad:f.checked_process(b,source=source,configuration=config,report=report_pin,read_artifact=read,check_dead=lambda v:True))
 reject('captured_identity_still_live',lambda:f.checked_process(proof,source=source,configuration=config,report=report_pin,read_artifact=read,check_dead=lambda v:False))
 for key,bad_value in (('actual_native_exit_code',False),('actual_exec_tool_exit_code',1),('actual_tool_chunk_id',''),('owner','/root/other')):
  bad=copy.deepcopy(owner);bad[key]=bad_value;reject('owner_'+key,lambda b=bad:f.checked_owner(b,source=source,configuration=config,report=report_pin,read_artifact=read,check_dead=lambda v:True))
 for key,bad_value in (('monitor_clean',False),('error','dirty'),('samples',2),('actual_exit_code',False)):
  bad=copy.deepcopy(guard);bad[key]=bad_value;reject('guard_'+key,lambda b=bad:f.checked_guard(b,1234))
 finite={'kind':f.KIND,'owner':f.OWNER,'diagnostic_only':True,'qualified':False,'production_enabled':False,'optimizer_updates':20}
 def unreachable(*a,**kw):raise AssertionError('Metadata IO must not be reached for invalid flags')
 for key,bad_value in (('kind','ASR18_production_window_completed_checkpoint_pointer_v2'),('diagnostic_only',False),('diagnostic_only',1),('qualified',True),('production_enabled',True),('optimizer_updates',True),('optimizer_updates',19)):
  bad=copy.deepcopy(finite);bad[key]=bad_value
  reject('finite_'+key+'_'+repr(bad_value),lambda b=bad:f._checked_binding(b,read_artifact=unreachable,check_process=unreachable,check_owner=unreachable,check_format=unreachable,check_payload=unreachable,check_prefix=unreachable))
 assert not any(k in sys.modules for k in ('torch','numpy','transformers','nemo'));g.check()
 controls={'passed':True,'negative_count':len(negatives),'negative_controls':negatives,'synthetic854_descriptor_PRE_equations_and_215_states_checked':True,
  'actual_source_format_checker_used':True,'all4_RNG_descriptor_families_checked':True,'physical_IO_liveness_payload_injected_or_omitted':True,'actual_audio_model_payload_GPU_jobs':0}
 (O/'actual_metadata_controls_v1.json').write_text(json.dumps(controls,indent=2)+'\n')
 result={'kind':'ASR18_independent_finite_BPE9223_checkpoint_evidence_source_review_v1','passed':True,'reviewer':'/root/label_bpe_audit','source_handoff':rec(h),'sources':sources,
  'scope':'finite honest evidence helper plus exact actual runner2/process/persistence/format API; no native or resource admission','controls':rec(O/'actual_metadata_controls_v1.json'),
  'findings':['Finite KIND is distinct; strict diagnosticTrue/qualifiedFalse/productionFalse and exactly20 required','Runner2 real component process KIND and H.Watch error/samples/monitor closure fields match helper','Same source/config/report/native identities joined for20 history, native0, separate fresh recovery0 and owner TOOL0','Metadata854 descriptor reconstruction matches saved_fingerprint equations; all207/215Adam/all4RNG normalized PRE comparison plus newclock20/sourceclocks enforced','Complete actual payload hash and saved journal prefix are mandatory in nonfixture checked_binding; physical IO intentionally not executed in this review','Provisional nextdraw consumes0/advancesno replay/audio; fresh branch prefix/source/planfrontier bound','No fabricated production pointer, durable head/ACK or full-pass ancestry; no winner/promotion authority'],
  'new_concrete_source_blockers':[],'reuse_prior_DEV_structure_format_mapper_review':True,'actual_model_audio_GPU_payload_jobs':0,'execution_admitted_here':False}
 (O/'source_findings_v1.json').write_text(json.dumps(result,indent=2)+'\n');g.publish(completed=True,findings=rec(O/'source_findings_v1.json'))
 print(json.dumps({'passed':True,'negative_controls':len(negatives),'findings':rec(O/'source_findings_v1.json'),'controls':rec(O/'actual_metadata_controls_v1.json')}))
if __name__=='__main__':main()
