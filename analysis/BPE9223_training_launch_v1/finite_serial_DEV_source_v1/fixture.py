"""Explicit synthetic metadata controls; no real process0 or payload claims."""
import ast,copy,hashlib,json,os,random,sys,types
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish'
sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import record
from bpe_broad_production_state_v1 import digest
from bpe_streaming_checkpoint_semantics_v2 import verify_saved_metadata
import asr18_finite_BPE9223_checkpoint_evidence_v1 as e
from asr18_full_run_serial_DEV_9223_v2 import checkpoint_format_view

OUT=Path(__file__).parent
F=OUT/'synthetic_not_actual_process_or_payload'
def pin(path):return {'path':str(path),'bytes':1,'sha256':hashlib.sha256(str(path).encode()).hexdigest()}
source=record(e.RUNNER);configpin=pin(F/'config.json');overallpin=pin(F/'overall_result.json')
nativepin=pin(F/'train/training_result.json');recoverpin=pin(F/'recover/recovery_result.json')
n0=pin(F/'train/actual_process0.json');r0=pin(F/'recover/actual_process0.json')
ownerpin=pin(F/'owner_tool0.json');cppin=pin(F/'train/after20.state.json');payload=pin(F/'train/after20.state.tensors.bin')
prepin=pin(F/'train/PRE_save_fingerprint.json');persistpin=pin(F/'train/persistence_verified.json')
nextpin=pin(F/'train/next_draw.json');sharedpin=pin(F/'shared_features.json');planpin=pin(F/'plan.json')
prefix={'path':str(F/'train/actual_commits.jsonl'),'offset':0,'sha256':hashlib.sha256(b'').hexdigest()}
identities=[{'pid':900001+i,'ppid':900000,'create_time':1000.+i} for i in range(3)]
data={'BPE9223_population':pin(F/'complete_new_labels.json')};migration=pin(F/'migration.json');featureowner=pin(F/'CPU/owner.json')
identity={'kind':'ASR18_original_first4_throughput_unqualified_checkpoint_identity_v1','config':configpin,
    'arm':'candidate','shared_feature_owner':featureowner,'BPE9223_population':data['BPE9223_population'],
    'row_migration_plan':migration,'production_enabled':False,'production_source_eligible':False}
sampler={'carried_replay':{'updates':20},'synthetic_full_sampler_marker':'not an actual source state'}
entries=[]
def tensor(shape=(1,),dtype='float32'):
    i=len(entries);width={'float32':4,'uint8':1}[dtype];size=width
    entries.append({'index':i,'dtype':dtype,'shape':list(shape),'offset':i*64,'bytes':size,'sha256':hashlib.sha256(bytes(size)).hexdigest()})
    return ('tensor',i)
weights={f'synthetic_parameter_{i}':tensor() for i in range(207)}
parts=[]
for count in (200,8,7):
    states={i:{'step':tensor((), 'float32'),'exp_avg':tensor(),'exp_avg_sq':tensor()} for i in range(count)}
    parts.append({'param_groups':[{'params':list(range(count))}],'state':states})
torch_rng=tensor((1,),'uint8');cuda_rng=[tensor((1,),'uint8')]
assert len(entries)==854
opt={'kind':'BPE9223_broad_full207_three_AdamW_mixed_fresh_clock_v1','updates':20,
    'appended512_Adam_clock':20,'clocks':[64270,2020,62270],'identity':{'synthetic_optimizer':'only'},'optimizers':parts}
saved={'kind':'ASR18_broad_caller_state_v1','diagnostic_only':True,'identity':identity,'source_provenance':{},
    'weights':weights,'optimizer':opt,'sampler':sampler,'augmentation':{'committed_journal':prefix},
    'python_rng':random.Random(17).getstate(),'numpy_rng':('MT19937',[0]*624,0,0,0.),'torch_rng':torch_rng,'cuda_rng':cuda_rng}
def encode(v):
    if type(v) is tuple and len(v)==2 and v[0]=='tensor':return ['tensor',v[1]]
    if type(v) is dict:return ['dict',[[['scalar',k],encode(x)] for k,x in v.items()]]
    if type(v) in (tuple,list):return ['tuple' if type(v) is tuple else 'list',[encode(x) for x in v]]
    return ['scalar',v]
payload['bytes']=entries[-1]['offset']+entries[-1]['bytes']
metadata={'kind':'ASR18_streaming_named_tensor_checkpoint_v1','schema_version':1,'endianness':'little',
    'source':record(S/'bpe_streaming_checkpoint_v1.py'),'maximum_CPU_tensor_slice_bytes':4*1024**2,
    'payload':payload,'tensors':entries,'tree':encode(saved),'dynamic_module_serialization':False}
pre=e.metadata_fingerprint(metadata)
semantic=verify_saved_metadata(metadata,identity=identity,optimizer_updates=20,sampler_state=sampler,journal_prefix=prefix,diagnostic_only=True)
persistence={'kind':'ASR18_actual_streamed_saved_state_matches_independent_pre_save_v1','passed':True,
    'checkpoint':cppin,'payload':payload,'independent_pre_save_state_sha256':pre['state_sha256'],
    'saved_state_sha256':pre['state_sha256'],'optimizer_updates':20,'weights_verified':207,'Adam_states_verified':215,
    'named_tensors_verified':854,'sampler_state_sha256':digest(sampler),'journal_prefix':prefix,
    'saved_semantic_state_verified':semantic,'actual_GPU_restore_performed':False,
    'live_model_optimizer_or_RNG_mutated_here':False,'independent_post_check_live_fingerprint_comparison_required':True,
    'full_fresh_GPU_restore_required_before_overall_diagnostic_pass':True,
    'source':record(S/'asr18_streamed_persistence_fingerprint_v1.py')}
nextdraw={'kind':'ASR18_actual_checkpoint_provisional_next_draw_packet_v1','optimizer_updates':20,
    'before_sampler_sha256':digest(sampler),'draws_consumed':0,'replay_frontier_advanced':False,'audio_or_augmentation_applied':False}
native={'passed':True,'source':source,'producer':identities[0],'configuration':configpin,'arm':'candidate',
    'production_source_eligible':False,'production_enabled':False,'actual_optimizer_updates':20,
    'history':[{'update':i+1} for i in range(20)],'checkpoint':cppin,'PRE_save_fingerprint':prepin,'persistence':persistpin,
    'next_draw':nextpin,'next_draw_sha256':digest(nextdraw),'journal_prefix':prefix,'sampler_sha256':digest(sampler)}
branch={'kind':'ASR18_preserved_tail_checkpoint_branch_v1','original_checkpoint_prefix':prefix,
    'old_journal_modified':False,'tail_replayed_or_sampler_advanced':False,
    'branch_checkpoint_prefix':{**prefix,'path':str(F/'recover/restored_journal_prefix.jsonl')}}
recovery={'passed':True,'source':source,'producer':identities[1],'arm':'candidate','production_enabled':False,
    'all207_threeAdam_all4RNG_sampler_nextdraw_exact':True,'actual_native_updates':0,'branch':branch}
def guard(pid):return {'pid':pid,'actual_exit_code':0,'monitor_clean':True,'monitor_joined':True,'samples':3,'error':None}
ng=pin(F/'train/guard_closure.json');rg=pin(F/'recover/guard_closure.json');og=pin(F/'guard_closure.json')
def proc(identity,report,g):return {'kind':e.PROCESS_KIND,'actual_Popen_wait_code':0,'all_captured_identities_dead':True,
    'captured_identities':[identity],'native':identity,'source':source,'configuration':configpin,'report':report,'guard':g}
owner={'kind':e.TOOL_KIND,'owner':e.OWNER,'actual_native_exit_code':0,'actual_exec_tool_exit_code':0,
    'actual_tool_chunk_id':'EXPLICIT_SYNTHETIC_FIXTURE_NOT_ACTUAL_TOOL','native_PID':identities[2]['pid'],
    'producer':identities[2],'source':source,'configuration':configpin,'report':overallpin,'guard_closure':og}
overall={'passed':True,'production_enabled':False,'arm':'candidate','actual_native_updates':20,'source':source,
    'configuration':configpin,'actual_native_process0':n0,'actual_fresh_recovery_process0':r0,'shared_features':sharedpin}
config={'kind':'ASR18_BPE9223_sustained_original_source_configuration_v1','production_enabled':False,
    'owner_validation_only':True,'checkpoint_diagnostic_only':True,'vocabulary':9223,'native_updates':20,'window_count':1,
    'harness_source':source,'planning_arguments':{'DATA_evidence':data},'row_migration_plan':migration,'plan':planpin}
packet={'kind':e.KIND,'owner':e.OWNER,'diagnostic_only':True,'qualified':False,'production_enabled':False,
    'optimizer_updates':20,'native_source':source,'configuration':configpin,'overall_result':overallpin,'owner_tool_actual0':ownerpin,
    'native_process0':n0,'native_report':nativepin,'recovery_process0':r0,'recovery_report':recoverpin,
    'checkpoint':cppin,'payload':payload,'PRE_save_fingerprint':prepin,'persistence':persistpin,'next_draw':nextpin,
    'journal_prefix':prefix,'sampler_state_sha256':digest(sampler),'identity':identity,'saved_semantic_state_verified':semantic}
objects={configpin['path']:config,overallpin['path']:overall,ownerpin['path']:owner,nativepin['path']:native,
    recoverpin['path']:recovery,n0['path']:proc(identities[0],nativepin,ng),r0['path']:proc(identities[1],recoverpin,rg),
    ng['path']:guard(identities[0]['pid']),rg['path']:guard(identities[1]['pid']),og['path']:guard(identities[2]['pid']),
    sharedpin['path']:{'owner':featureowner},cppin['path']:metadata,prepin['path']:pre,persistpin['path']:persistence,
    nextpin['path']:nextdraw,planpin['path']:{'updates':[{'after_sampler_sha256':digest(sampler)} for _ in range(20)]}}
def run(p,o):
    read=lambda v:o[v['path']]
    def dead(v):return v in identities
    def process(v,**kw):return e.checked_process(v,read_artifact=read,check_dead=dead,**kw)
    def own(v,**kw):return e.checked_owner(v,read_artifact=read,check_dead=dead,**kw)
    def payloadcheck(c,v):assert c==cppin and v==payload
    def prefixcheck(v):assert v==prefix
    return e._checked_binding(p,read_artifact=read,check_process=process,check_owner=own,
        check_format=checkpoint_format_view,check_payload=payloadcheck,check_prefix=prefixcheck)
checks=[]
assert run(copy.deepcopy(packet),copy.deepcopy(objects))[2]==sampler
checks.append('synthetic complete finite20 source/PRE/854tensor/215Adam/process joins accepted')
def reject(name,change):
    p,o=copy.deepcopy(packet),copy.deepcopy(objects);change(p,o)
    try:run(p,o)
    except (ValueError,AssertionError,KeyError,TypeError):checks.append(name);return
    raise AssertionError('Control incorrectly accepted: '+name)
reject('production pointer kind rejected',lambda p,o:p.update(kind='ASR18_production_window_completed_checkpoint_pointer_v2'))
for flag,bad in [('diagnostic_only',False),('diagnostic_only',1),('qualified',True),('production_enabled',True),('optimizer_updates',True),('optimizer_updates',19)]:
    reject(f'finite packet {flag}={bad!r} rejected',lambda p,o,k=flag,v=bad:p.update({k:v}))
reject('changed runner source rejected',lambda p,o:p['native_source'].update(sha256='0'*64))
reject('production config rejected',lambda p,o:o[configpin['path']].update(owner_validation_only=False))
reject('8711 config rejected',lambda p,o:o[configpin['path']].update(vocabulary=8711))
reject('wrong overall native0 rejected',lambda p,o:o[overallpin['path']].update(actual_native_process0=r0))
reject('outer TOOL1 rejected',lambda p,o:o[ownerpin['path']].update(actual_exec_tool_exit_code=1))
reject('boolean owner exit rejected',lambda p,o:o[ownerpin['path']].update(actual_native_exit_code=False))
reject('empty actual chunk rejected',lambda p,o:o[ownerpin['path']].update(actual_tool_chunk_id=''))
reject('live/unknown captured owner rejected',lambda p,o:o[ownerpin['path']]['producer'].update(create_time=77.))
reject('component Popen1 rejected',lambda p,o:o[n0['path']].update(actual_Popen_wait_code=1))
reject('boolean component wait rejected',lambda p,o:o[n0['path']].update(actual_Popen_wait_code=False))
reject('live/unknown captured native rejected',lambda p,o:o[n0['path']]['native'].update(create_time=77.))
reject('dirty guard rejected',lambda p,o:o[rg['path']].update(monitor_clean=False))
reject('two guard samples rejected',lambda p,o:o[ng['path']].update(samples=2))
reject('recovery native update rejected',lambda p,o:o[recoverpin['path']].update(actual_native_updates=1))
reject('wrong recovery prefix rejected',lambda p,o:o[recoverpin['path']]['branch']['original_checkpoint_prefix'].update(offset=1))
reject('stale native history rejected',lambda p,o:o[nativepin['path']]['history'][-1].update(update=19))
reject('actual nextdraw disagreement rejected',lambda p,o:o[nextpin['path']].update(draws_consumed=1))
reject('wrong nextdraw sampler rejected',lambda p,o:o[nextpin['path']].update(before_sampler_sha256='0'*64))
reject('plan end sampler disagreement rejected',lambda p,o:o[planpin['path']]['updates'][-1].update(after_sampler_sha256='0'*64))
reject('PRE descriptor drift rejected',lambda p,o:o[cppin['path']]['tensors'][0].update(sha256='1'*64))
reject('legacy207 Adam count rejected',lambda p,o:o[persistpin['path']].update(Adam_states_verified=207))
reject('missing named tensor rejected',lambda p,o:o[cppin['path']]['tensors'].pop())
reject('fake persistenceGPU restore rejected',lambda p,o:o[persistpin['path']].update(actual_GPU_restore_performed=True))
# Compare COMPLETE unchanged numerical/runtime bodies after declared namespace substitutions.
old=(S/'asr18_diagnostic_serial_DEV_9223_v1.py').read_text();new=(S/'asr18_finite_BPE9223_serial_DEV_v1.py').read_text()
old=old.replace('asr18_diagnostic_DEV_runtime_9223_v1','finite_BPE9223_DEV_runtime_v1').replace("tag = 'ASR18_diagnostic_9223_DEV_'","tag = 'ASR18_finite_9223_DEV_'")
functions=lambda text:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
a,b=functions(old),functions(new)
unchanged=['serial','same','integer','sources','new_E','write_new','canonical_DEV_paths','_saved_dictionary',
    'checkpoint_format_view','_tensor_hash','_capture_RNG','_restore_RNG','check_optimizer_free_report_protocol',
    'evaluate_boundary_checkpoint','verified_completed_DEV','seal_completed_DEV','diagnostic_saved_text_rescore']
for name in unchanged:assert a[name]==b[name],name
checks.append('17 complete evaluator/restore/RNG/report/scorer/helper function AST bodies unchanged')
oldruntime=(S/'asr18_diagnostic_DEV_runtime_9223_v1.py').read_text().replace('asr18_diagnostic_serial_DEV_9223_v1','asr18_finite_BPE9223_serial_DEV_v1').replace('asr18_diagnostic_DEV_runtime_9223_v1','finite_BPE9223_DEV_runtime_v1')
assert ast.dump(ast.parse(oldruntime),include_attributes=False)==ast.dump(ast.parse((S/'finite_BPE9223_DEV_runtime_v1.py').read_text()),include_attributes=False)
checks.append('entire runtime AST unchanged except two explicit module substitutions')
import asr18_full_run_serial_DEV_9223_v2 as protected
namespace=dict(protected.checked_checkpoint.__globals__);namespace['read_bound']=lambda pin:packet
f=types.FunctionType(protected.checked_checkpoint.__code__,namespace)
try:f({},identity={},expected_updates=20,boundary={})
except (ValueError,KeyError):checks.append('production checker still rejects finite packet')
else:raise AssertionError('finite packet admitted by production checker')
assert all(k not in sys.modules for k in ('torch','numpy','soundfile','pyarrow','av'))
checks.append('no numerical/audio/model import and no real tensor payload read')
report={'kind':'ASR18_finite_BPE9223_serialDEV_stdlib_source_controls_v1','passed':True,'PID':os.getpid(),
    'checks':checks,'count':len(checks),'synthetic_process_and_payload_I_O_only':True,
    'actual_native_or_DEV_jobs':0,'actual_audio_or_payload_reads':0,'production_admission_claimed':False,
    'sources':{n:record(S/n) for n in ('asr18_finite_BPE9223_serial_DEV_v1.py','finite_BPE9223_DEV_runtime_v1.py','asr18_finite_BPE9223_checkpoint_evidence_v1.py')},
    'fixture_source':record(__file__)}
(OUT/'fixture_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
