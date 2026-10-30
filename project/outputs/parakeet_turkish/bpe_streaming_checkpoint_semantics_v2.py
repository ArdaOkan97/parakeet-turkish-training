"""Inspect the trusted streaming writer's saved scalar state without Torch.

This verifies snapshot identity/frontiers and prefix, not tensor values against
live GPU memory. The caller separately hashes the complete returned payload.
"""
import math,sys
from pathlib import Path
from bpe_broad_production_state_v1 import require,digest
from bpe_broad_audio_v1 import record

def _node(value):
    require(type(value) is list and len(value)==2 and type(value[0]) is str,'Exact tagged checkpoint node required')
    return value[0],value[1]

def _items(value):
    tag,pairs=_node(value);require(tag=='dict' and type(pairs) is list and len(pairs)<=100000,'Bounded saved dictionary required')
    result={}
    for pair in pairs:
        require(type(pair) is list and len(pair)==2,'Exact saved dictionary pair required')
        kind,key=_node(pair[0]);require(kind=='scalar' and type(key) in (str,int) and key not in result,'Unique exact saved scalar dictionary keys required')
        result[key]=pair[1]
    return result

def select(tree,*path):
    node=tree
    for key in path:
        values=_items(node);require(key in values,'Required saved semantic checkpoint field missing: '+str(key));node=values[key]
    return node

def decode_scalar_tree(node,*,_depth=0,_budget=None):
    budget=[0] if _budget is None else _budget;budget[0]+=1
    require(_depth<=64 and budget[0]<=2_000_000,'Bounded scalar checkpoint tree required')
    tag,value=_node(node)
    if tag=='scalar':
        require(value is None or type(value) in (str,int,float,bool),'Exact saved scalar required')
        require(type(value) is not float or math.isfinite(value),'Finite scalar checkpoint values required')
        return value
    if tag=='dict':return {key:decode_scalar_tree(child,_depth=_depth+1,_budget=budget) for key,child in _items(node).items()}
    require(tag in ('list','tuple') and type(value) is list,'Semantic snapshot fields cannot contain tensor references or unknown tags')
    values=[decode_scalar_tree(child,_depth=_depth+1,_budget=budget) for child in value]
    return tuple(values) if tag=='tuple' else values

def verify_saved_metadata(metadata,*,identity,optimizer_updates,sampler_state,journal_prefix,diagnostic_only):
    require(type(diagnostic_only) is bool and type(optimizer_updates) is int and optimizer_updates>0,'Explicit qualification scope and actual positive optimizer frontier required')
    require(type(metadata) is dict and metadata['kind']=='ASR18_streaming_named_tensor_checkpoint_v1'
        and type(metadata['schema_version']) is int and metadata['schema_version']==1 and metadata['endianness']==sys.byteorder=='little'
        and metadata['dynamic_module_serialization'] is False and metadata['maximum_CPU_tensor_slice_bytes']==4*1024**2
        and metadata['source']==record(Path(__file__).with_name('bpe_streaming_checkpoint_v1.py'))
        and type(metadata['tensors']) is list and 0<len(metadata['tensors'])<=4096,'Exact trusted bounded named-state writer contract required')
    tree=metadata['tree'];read=lambda *path:decode_scalar_tree(select(tree,*path))
    require(read('kind')=='ASR18_broad_caller_state_v1' and type(read('diagnostic_only')) is bool and read('diagnostic_only')==diagnostic_only,
        'Saved scientific/diagnostic qualification scope differs from current checkpoint')
    saved_identity=read('identity');saved_updates=read('optimizer','updates');saved_sampler=read('sampler');saved_prefix=read('augmentation','committed_journal')
    require(saved_identity==identity and digest(saved_identity)==digest(identity),'Returned checkpoint saved identity differs from immutable pre-writer identity')
    require(type(saved_updates) is int and saved_updates==optimizer_updates,'Returned checkpoint saved Adam update count is stale/different')
    require(saved_sampler==sampler_state and digest(saved_sampler)==digest(sampler_state),'Returned checkpoint saved sampler/frontiers/RNG differ from live pre-writer snapshot')
    require(type(saved_prefix) is dict and set(saved_prefix)=={'path','offset','sha256'} and type(saved_prefix['offset']) is int and saved_prefix['offset']>=0
        and saved_prefix==journal_prefix,'Returned checkpoint saved augmentation journal path/prefix differs from live pre-writer snapshot')
    require(type(saved_sampler['carried_replay']['updates']) is int and saved_sampler['carried_replay']['updates']==optimizer_updates,
        'Saved carried replay frontier differs from actual Adam update count')
    return {'kind':'ASR18_saved_streamed_checkpoint_semantics_match_live_v1','passed':True,'identity_sha256':digest(saved_identity),
        'optimizer_updates':saved_updates,'sampler_state_sha256':digest(saved_sampler),'journal_prefix':saved_prefix,'diagnostic_only':diagnostic_only,
        'saved_identity_frontier_sampler_and_journal_verified_before_publication':True,'whole_payload_hash_required_separately':True,
        'tensor_values_reloaded_or_live_tensor_equality_claimed':False,'verifier':record(__file__)}
