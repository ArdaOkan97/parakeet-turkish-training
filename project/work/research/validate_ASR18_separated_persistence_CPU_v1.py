"""Bounded read-only checkpoint transport and synthetic persistence controls."""
from copy import deepcopy
import argparse, hashlib, json, math, os, random, struct, sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import write_json,read_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from bpe_streaming_checkpoint_semantics_v2 import select,decode_scalar_tree
import asr18_streamed_persistence_fingerprint_v1 as persistence
import asr18_separated_worker_forecast_v1 as policy
from fastmss_row_serial_cpu_v4 import environment,resource_guard


class TensorSpec:
    def __init__(self,dtype,shape,data):
        self.dtype=dtype;self.shape=shape;self.data=data
    def descriptor(self):
        return {'shape':self.shape,'dtype':'torch.'+self.dtype,'bytes':len(self.data),
                'sha256':hashlib.sha256(self.data).hexdigest()}


def synthetic(folder,*,corrupt=False):
    folder.mkdir();weights={};expected_weights={};partitions=[];expected_parts=[]
    for i in range(207):
        value=TensorSpec('float32',[1],struct.pack('<f',i+.25))
        weights['parameter_'+str(i)]=value;expected_weights['parameter_'+str(i)]=value.descriptor()
    for group in range(3):
        states={};expected_states={}
        for i in range(69):
            tensors={'step':TensorSpec('float32',[],struct.pack('<f',float((64250,2000,62250)[group]+1))),
                'exp_avg':TensorSpec('float32',[1],struct.pack('<f',i/64)),
                'exp_avg_sq':TensorSpec('float32',[1],struct.pack('<f',i/128))}
            states[i]=tensors;expected_states[i]={k:v.descriptor() for k,v in tensors.items()}
        groups=[{'params':list(range(69)),'lr':1e-5,'betas':(.9,.999),'eps':1e-8,'weight_decay':.01}]
        partitions.append({'param_groups':groups,'state':states});expected_parts.append({'param_groups':groups,'state':expected_states})
    torch_rng=TensorSpec('uint8',[8],b'01234567');cuda_rng=[TensorSpec('uint8',[8],b'76543210')]
    python=random.getstate();numpy=('MT19937',[0]*624,624,0,0.);identity={'synthetic':True,'scope':'transport_only'}
    sampler={'broad_cursor':{'draws':14},'carried_replay':{'updates':1}};prefix={'path':str(folder/'journal.jsonl'),'offset':0,'sha256':hashlib.sha256(b'').hexdigest()}
    value={'kind':'ASR18_broad_caller_state_v1','diagnostic_only':True,'identity':identity,'source_provenance':{},
        'weights':weights,'optimizer':{'kind':'synthetic_only','identity':identity,'updates':1,'clocks':[64251,2001,62251],'optimizers':partitions},
        'sampler':sampler,'augmentation':{'committed_journal':prefix},'python_rng':python,'numpy_rng':numpy,
        'torch_rng':torch_rng,'cuda_rng':cuda_rng}
    fp={'kind':'ASR18_actual_live207_threeAdam_all4RNG_fingerprint_v1','updates':1,'identity':identity,
        'clocks':[64251,2001,62251],'weights':{k:expected_weights[k] for k in sorted(expected_weights)},
        'Adam_partitions':expected_parts,'RNG':{'python':python,'numpy':numpy,'torch':torch_rng.descriptor(),
        'CUDA':[v.descriptor() for v in cuda_rng]},'maximum_transfer_slice_bytes':4*1024**2}
    expected={'state_sha256':digest(fp),'state':fp};entries=[];payload=bytearray()
    def encode(item):
        if isinstance(item,TensorSpec):
            padding=(-len(payload))%64;payload.extend(bytes(padding));offset=len(payload);payload.extend(item.data)
            index=len(entries);entries.append({'index':index,'dtype':item.dtype,'shape':item.shape,'offset':offset,
                'bytes':len(item.data),'sha256':hashlib.sha256(item.data).hexdigest()});return ['tensor',index]
        if type(item) is dict:return ['dict',[[encode(k),encode(v)] for k,v in item.items()]]
        if type(item) in (list,tuple):return ['list' if type(item) is list else 'tuple',[encode(v) for v in item]]
        return ['scalar',item]
    tree=encode(value);path=folder/'checkpoint.state.json';data=path.with_suffix('.tensors.bin')
    normal_sha=hashlib.sha256(payload).hexdigest()
    if corrupt:payload[0]^=1
    data.write_bytes(payload)
    metadata={'kind':'ASR18_streaming_named_tensor_checkpoint_v1','schema_version':1,'endianness':'little',
        'source':record(S/'bpe_streaming_checkpoint_v1.py'),'maximum_CPU_tensor_slice_bytes':4*1024**2,
        'payload':{'path':str(data),'bytes':len(payload),'sha256':normal_sha},'tensors':entries,'tree':tree,
        'dynamic_module_serialization':False}
    write_json(path,metadata)
    return record(path),expected,identity,sampler,prefix


def main(folder):
    environment();require(folder.resolve().is_relative_to(Path(r'E:\gpu_training_scratchpad').resolve()) and not folder.exists(),'Fresh E-only controls');folder.mkdir()
    before=resource_guard();v=before['system_memory'];budget=128_000_000
    require(v['commit_used_bytes']+budget<.9*v['commit_limit_bytes'] and v['commit_available_bytes']-budget>2_000_000_000,'Fresh128MB metadata/stream admission')
    write_json(folder/'fresh128MB_admission.json',{'passed':True,'sample':before,'incremental_bytes':budget});checks=[]
    def check(name,value):require(value,name);checks.append(name)
    def reject(name,call):
        try:call()
        except ValueError:checks.append(name);return
        raise AssertionError('Expected rejection: '+name)
    pin,expected,identity,sampler,prefix=synthetic(folder/'synthetic')
    proof=persistence.verify_persistence(pin,expected_pre_save=expected,identity=identity,sampler_state=sampler,
        journal_prefix=prefix,diagnostic_only=True,guard=resource_guard)
    check('Independent synthetic207/threeAdam/all4RNG bytes match saved transport',proof['weights_verified']==207 and proof['Adam_states_verified']==207)
    check('Read-only persistence never claims actual GPU restore',proof['actual_GPU_restore_performed'] is False and proof['full_fresh_GPU_restore_required_before_overall_diagnostic_pass'] is True)
    transported=json.loads(json.dumps(expected));normalized=persistence.normalize_fingerprint(transported)
    check('Integer-keyed Adam fingerprint survives exact JSON transport',normalized['state_sha256']==expected['state_sha256'])
    bad=deepcopy(expected);bad['state']['Adam_partitions'][0]['state'][True]=bad['state']['Adam_partitions'][0]['state'].pop(1)
    reject('Boolean Adam map index rejected',lambda:persistence.normalize_fingerprint(bad))
    bad=json.loads(json.dumps(expected));bad['state']['Adam_partitions'][0]['state']['01']=bad['state']['Adam_partitions'][0]['state'].pop('1')
    reject('Noncanonical Adam map index rejected',lambda:persistence.normalize_fingerprint(bad))
    bad=deepcopy(expected);bad['state']['weights']['parameter_0']['sha256']='0'*64;bad['state_sha256']=digest(bad['state'])
    reject('Independent PRE-save tensor mismatch rejected',lambda:persistence.verify_persistence(pin,expected_pre_save=bad,identity=identity,sampler_state=sampler,journal_prefix=prefix,diagnostic_only=True,guard=resource_guard))
    corrupt,exp,ident,state,journal=synthetic(folder/'synthetic_corrupt',corrupt=True)
    reject('Actual owned payload byte corruption rejected',lambda:persistence.verify_persistence(corrupt,expected_pre_save=exp,identity=ident,sampler_state=state,journal_prefix=journal,diagnostic_only=True,guard=resource_guard))
    reject('Stale sampler rejected',lambda:persistence.verify_persistence(pin,expected_pre_save=expected,identity=identity,sampler_state={'carried_replay':{'updates':0}},journal_prefix=prefix,diagnostic_only=True,guard=resource_guard))
    reject('Typed identity drift rejected',lambda:persistence.verify_persistence(pin,expected_pre_save=expected,identity={'synthetic':1,'scope':'transport_only'},sampler_state=sampler,journal_prefix=prefix,diagnostic_only=True,guard=resource_guard))
    native={'kind':'ASR18_finite_measured_geometry_native_forecast_v6','maximum_native_updates_in_process':40,
        'prospective_native_success_claimed':False,'production_forecast_approved':False,'absolute_private_forecast_bytes':5560813440}
    resident=policy.resident_worker_forecast(native);restore=policy.restore_only_worker_forecast()
    check('Resident forecast retains native ceiling plus full64MB write',resident['entire_worker_private_forecast_bytes']==5624813440)
    check('Separate recovery retains3GB startup plus full350MB restore',restore['entire_worker_private_forecast_bytes']==3350000000)
    policy.phase_projection(resident,phase='native',current_private_bytes=3_000_000_000,increment_bytes=2560813440)
    check('Original native increment fits actual separated phase accounting',True)
    reject('In-residency restore is forbidden by phase policy',lambda:policy.phase_projection(resident,phase='checkpoint_restore',current_private_bytes=3_000_000_000,increment_bytes=350_000_000))
    reject('Startup restore beyond3GB pre-stage is forbidden',lambda:policy.phase_projection(resident,phase='startup_restore',current_private_bytes=3_000_000_001,increment_bytes=350_000_000))
    reject('Reduced write increment rejected',lambda:policy.phase_projection(resident,phase='checkpoint_write',current_private_bytes=3_000_000_000,increment_bytes=1))
    # New transport reader applied to immutable actual historical bytes. This
    # proves its format/hash compatibility only, not an independent live-state
    # match: no historical PRE-save fingerprint in the new format is invented.
    historical=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1\feature_window_native20_01\after_update20.state.json')
    historical_pin=record(historical);metadata=read_json(historical);payload_before=record(metadata['payload']['path'])
    view=persistence.saved_fingerprint(historical_pin,guard=resource_guard)
    check('Actual historical244MB payload full hashes and named207 format verified',len(view['state']['weights'])==207 and view['state']['updates']==20)
    check('Actual historical source metadata/payload remain unchanged',record(historical)==historical_pin and record(metadata['payload']['path'])==payload_before)
    check('No numerical/model/audio/frontend imports',not any(n in sys.modules for n in ('torch','numpy','scipy','transformers','av','soundfile','pyarrow')))
    result={'kind':'ASR18_separated_persistence_metadata_and_rawbyte_controls_v1','passed':True,'check_count':len(checks),'checks':checks,
        'source':record(__file__),'persistence_source':record(S/'asr18_streamed_persistence_fingerprint_v1.py'),
        'policy_source':record(S/'asr18_separated_worker_forecast_v1.py'),'historical_checkpoint':historical_pin,
        'historical_payload':payload_before,'historical_comparison_scope':'saved transport hashes and format only; no independent historical PRE-save equality claimed',
        'independent_PREsave_positive_scope':'synthetic known scalar/tensor bytes only','actual_live_GPU_or_native_trial_started':False,
        'actual_original_audio_frontend_jobs':0,'shared_feature_CACHE_changed':False,'production_enabled':False,
        'resident_forecast':resident,'restore_only_forecast':restore}
    write_json(folder/'report.json',result);print(json.dumps({'passed':True,'checks':len(checks),'historical_payload_verified_bytes':payload_before['bytes'],'no_GPU_or_native_trial':True}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True,type=Path);a=p.parse_args();main(a.folder)
