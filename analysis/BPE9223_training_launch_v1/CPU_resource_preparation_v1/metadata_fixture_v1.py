"""Synthetic protocol negatives + real source import/hash closure, no data jobs."""
import ast,copy,hashlib,importlib,json,re,sys
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
sys.path[:0]=[str(S),str(R)]
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from fastmss_owned_process_closure_v2 import identity
import fastmss_BPE9223_resource_CPU_boundary_v1 as h
modules=['fastmss_BPE9223_plan_bridge','fastmss_BPE9223_feature_engine','fastmss_BPE9223_feature_protocol','fastmss_BPE9223_cpu_common','fastmss_BPE9223_recorded_room_train','run_ASR18_BPE9223_CPU_window','produce_FastMSS_BPE9223_reference_CPU','produce_FastMSS_BPE9223_frontend_CPU']
loaded={name:importlib.import_module(name+'_v6') for name in modules}
context=importlib.import_module('asr18_BPE9223_source_context_v3');factory=importlib.import_module('asr18_BPE9223_CPU_window_configuration_v5')
require(loaded['fastmss_BPE9223_feature_engine'].BRIDGE_SHA==record(loaded['fastmss_BPE9223_plan_bridge'].__file__)['sha256'],'Real current engine->bridge hash')
require(factory.PROTOCOL_SHA256==record(loaded['fastmss_BPE9223_feature_protocol'].__file__)['sha256'],'Real current factory->protocol hash')
mapping={name+'_v3':name+'_v6' for name in modules};mapping['asr18_BPE9223_source_context_v2']='asr18_BPE9223_source_context_v3'
unchanged=[]
for name in modules[1:]:
    new=Path(loaded[name].__file__);old=new.with_name(name+'_v3.py');text=old.read_text(encoding='utf-8')
    for a,b in mapping.items():text=text.replace(a,b)
    if 'feature_engine' in name:text=re.sub(r'BRIDGE_SHA = "[0-9a-f]{64}"','BRIDGE_SHA = "'+record(loaded['fastmss_BPE9223_plan_bridge'].__file__)['sha256']+'"',text,count=1)
    require(text==new.read_text(encoding='utf-8'),'Exact import/path/pin-only numerical successor: '+name);unchanged.append(name)
require('fastmss_BPE9223_feature_engine_v6'==loaded['fastmss_BPE9223_feature_protocol'].engine.__name__,'Live protocol->engine import')
require('fastmss_BPE9223_plan_bridge_v6'==loaded['fastmss_BPE9223_cpu_common'].bridge.__name__,'Live core->bridge import')
require('fastmss_BPE9223_feature_protocol_v6'==factory.protocol.__name__,'Live factory->protocol import')
require(not any(k=='torch' or k.startswith('torch.') for k in sys.modules),'No Torch/audio/model payload imported')
checks=['real_engine_bridge_SHA','real_factory_protocol_SHA','live_import_closure','seven_numerical_import_pin_only_successors','no_Torch']

def pin(name):return {'path':'E:\\gpu_training_scratchpad\\analysis\\synthetic_'+name+'.json','bytes':1,'sha256':digest(name)}
objects={}
def put(name,value):p=pin(name);objects[p['path']]=value;return p
identity_value={'immutable_science':'synthetic metadata fixture only'}
op={'runtime':pin('runtime'),'parent':pin('parent'),'current_recovery':pin('current_recovery'),'prefix':pin('prefix'),'CPU_resource_boundary':record(h.__file__)}
compat=put('compat',{'operational_sources':op,'original_configuration':pin('config')})
keys={'full_run_identity_sha256':digest(identity_value),'initial_optimizer_updates':0,'initial_sampler_sha256':digest('initial'),
    'label_population_contract':pin('population'),'DATA_evidence_sha256':digest('DATA'),'augmentation_identity':{'frozen':True},
    'scientific_source_state':pin('source0'),'durable_owner':pin('durable_owner'),'planned_native_updates':401072,
    'owner_validation_only':False,'checkpoint_diagnostic_only':False,'production_source_eligible':True}
original_session=put('oldsession',copy.deepcopy(keys));objects[pin('config')['path']]={'session':original_session}
session={**keys,'original_session':original_session,'operational_compatibility':compat,
    'operational_metadata':{'storage_owner':pin('store'),'control_root':'E:\\gpu_training_scratchpad\\runs\\synthetic_control',
        'identity':identity_value,'operational_compatibility':compat,'current_native_resource_closure':None}}
pointer={'identity':{'caller_identity':{'full_run_lifecycle_identity':identity_value}},'committed_updates':20,'sampler_state_sha256':digest('sampler20'),'checkpoint':pin('checkpoint')}
pointer_pin=put('pointer',pointer);resource_pin=pin('actual_resource');process_pin=pin('fresh0');report_pin=pin('report');ack_pin=pin('ACK');coverage_pin=pin('coverage');evidence_pin=pin('evidence');recovered_pin=pin('recovered')
capture={'pointer':pointer_pin,'pre_save_fingerprint':pin('PRE'),'next_draw_packet':pin('next')}
stopped={'current_checkpoint_pointer':pointer_pin,'all_captured_identities':[{'pid':1,'ppid':2,'create_time':3.0}]}
objects[process_pin['path']]={'worker_source':op['runtime'],'parent_source':op['parent'],'producer':{'pid':4,'ppid':5,'create_time':6.0},'actual_Popen_wait_code':0,'report':report_pin}
objects[report_pin['path']]={'kind':'ASR18_full_run_actual_CURRENT_checkpoint_recovered_after_resource_stop_v1','passed':True,'source':op['current_recovery'],
    'diagnostic_only':False,'current_latest_checkpoint_recovered':True,'historical_fresh_GPU_restores_claimed':False,'run_identity_sha256':digest(identity_value),
    'current_native_resource_closure':resource_pin,'operational_compatibility':compat,'actual_native_updates_in_recovery_process':0,
    'actual_draws_consumed_in_recovery':0,'recovered_checkpoints':[recovered_pin]}
objects[ack_pin['path']]={'kind':'ASR18_resource_stop_current_latest_recovery_and_verified_persistence_prefix_v1','diagnostic_only':False,
    'run_identity_sha256':digest(identity_value),'operational_compatibility':compat,'actual_resource_quiescence_proven':True,'actual_native_normal_exit0_claimed':False,
    'actual_closed_native_index':resource_pin,'actual_closed_recovery_index':process_pin,'final_checkpoint_pointer':pointer_pin,'fresh_recovery_scope':'current_latest_only',
    'all_persistence_verified_through_pending_prefix':True,'all_historical_checkpoints_freshly_restored':False,
    'current_latest_all207_threeAdam_all4RNG_sampler_prefix_nextdraw_match_actual_pre_save':True,'actual_recovery_native_updates':0,
    'actual_recovery_draws_consumed':0,'current_recovery_evidence':evidence_pin,'persistence_prefix_coverage_index':coverage_pin,'pending_head':pin('pending'),'current_latest_exact_restore':recovered_pin}
objects[evidence_pin['path']]={'kind':'ASR18_full_run_resource_stop_current_recovery_prefix_inputs_v1','current_native_resource_closure':resource_pin,
    'actual_recovery_process0':process_pin,'recovery_report':report_pin,'operational_compatibility':compat,'current_checkpoint_capture':capture}
objects[coverage_pin['path']]={'passed':True,'source':op['prefix'],'current_recovery_evidence':evidence_pin,'final_checkpoint_pointer':pointer_pin,'pending_head':pin('pending'),
    'all_completed_window_PRE_saved_live_persistence_verified':True,'native_updates_consumed_in_verification':0,'historical_fresh_GPU_restores_claimed':False,'current_latest_exact_restore':recovered_pin}
objects[recovered_pin['path']]={'passed':True,'pointer':pointer_pin,'checkpoint':pointer['checkpoint'],'pre_save_fingerprint':capture['pre_save_fingerprint'],
    'next_draw_packet':capture['next_draw_packet'],'all207_threeAdam_all4RNG_match_actual_pre_save':True,'both_samplers_journal_prefix_and_next_draw_exact':True,
    'separate_fresh_GPU_recovery':True,'native_updates_performed':0,'draws_consumed':0}
boundary={'actual_native_resource_closure':resource_pin,'actual_fresh_recovery_process0':process_pin,'resource_recovery_report':report_pin,'resource_recovery_ack':ack_pin,
    'operational_compatibility':compat,'coherent_checkpoint_pointer':pointer_pin,'optimizer_updates':20,'sampler_state_sha256':digest('sampler20')}
saved=copy.deepcopy(objects)
originals=(h.metadata.read_bound,h.metadata.configure,h.checked_compatibility,h.checked_resource_stop,h.verify)
def fake_read(p,g=lambda:None):return copy.deepcopy(objects[p['path']])
def fake_compat(p,i,g=lambda:None):require(p==compat and i==identity_value,'Synthetic compatibility exact');return fake_read(p)
def fake_stop(p,**kw):require(p==resource_pin and kw['identity']==identity_value and kw['compatibility']==compat,'Synthetic stop exact');return copy.deepcopy(stopped)
def fake_verify(v,**kw):require(type(v['actual_Popen_wait_code']) is int and v['actual_Popen_wait_code']==0 and v['report']==kw['report'],'Synthetic actual0 marker negative')
try:
    h.metadata.read_bound=fake_read;h.metadata.configure=lambda **kw:None;h.checked_compatibility=fake_compat;h.checked_resource_stop=fake_stop;h.verify=fake_verify
    require(h.verify_resource_boundary(boundary,pointer=pointer,session=session,guard=lambda:None) is True,'Complete synthetic joins accepted');checks.append('synthetic_complete_resource_joins')
    mutations=[('missing_ACK',lambda b,o:b.pop('resource_recovery_ack')),('resource_plus_normal0',lambda b,o:b.update(actual_native_closed=pin('false0'))),
        ('resource_plus_live',lambda b,o:b.update(resident_native_boundary=pin('false_live'))),('wrong_stop',lambda b,o:b.update(actual_native_resource_closure=pin('wrong_stop'))),
        ('not_fresh',lambda b,o:o[process_pin['path']].update(producer=stopped['all_captured_identities'][0])),
        ('actual_process1',lambda b,o:o[process_pin['path']].update(actual_Popen_wait_code=1)),('normal0_claim',lambda b,o:o[ack_pin['path']].update(actual_native_normal_exit0_claimed=True)),
        ('quiescence_false',lambda b,o:o[ack_pin['path']].update(actual_resource_quiescence_proven=False)),('ACK_other_stop',lambda b,o:o[ack_pin['path']].update(actual_closed_native_index=pin('wrong'))),
        ('ACK_other_fresh0',lambda b,o:o[ack_pin['path']].update(actual_closed_recovery_index=pin('wrong'))),('ACK_other_pointer',lambda b,o:o[ack_pin['path']].update(final_checkpoint_pointer=pin('wrong'))),
        ('report_bool_zero',lambda b,o:o[report_pin['path']].update(actual_native_updates_in_recovery_process=False)),('report_other_source',lambda b,o:o[report_pin['path']].update(source=pin('wrong'))),
        ('coverage_other_source',lambda b,o:o[coverage_pin['path']].update(source=pin('wrong'))),('recovered_RNG_false',lambda b,o:o[recovered_pin['path']].update(all207_threeAdam_all4RNG_match_actual_pre_save=False)),
        ('next_draw_drift',lambda b,o:o[recovered_pin['path']].update(next_draw_packet=pin('wrong')))]
    for name,change in mutations:
        objects.clear();objects.update(copy.deepcopy(saved));b=copy.deepcopy(boundary);change(b,objects)
        try:h.verify_resource_boundary(b,pointer=pointer,session=session,guard=lambda:None)
        except (ValueError,KeyError):checks.append('reject_'+name)
        else:raise ValueError('Negative accepted: '+name)
finally:h.metadata.read_bound,h.metadata.configure,h.checked_compatibility,h.checked_resource_stop,h.verify=originals
sources=[record(m.__file__) for m in loaded.values()]+[record(context.__file__),record(factory.__file__),record(h.__file__)]
out=Path(__file__).parent/'report.json';require(not out.exists(),'Fresh fixture receipt')
out.write_text(json.dumps({'kind':'ASR18_CPU_resource_boundary_metadata_fixture_v1','passed':True,'producer':identity(),'source':record(__file__),'sources':sources,'checks':checks,
    'numerical_import_pin_only_clones':unchanged,'synthetic_protocol_joins_only':True,'actual_GPU_recovery_or_CPU_audio_admission_proven':False,'audio_GPU_model_or_cache_jobs':0},indent=2),encoding='utf-8')
print(json.dumps(record(out)),flush=True)
