"""Prepare new semantic publication controls; preserve old26/control history."""
from pathlib import Path
import sys,json
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import record,require
def main():
    old=P/'work/research/validate_ASR18_production_window_v2.py';target=P/'work/research/validate_ASR18_production_window_v3.py';require(not target.exists(),'Fresh publication-control successor')
    text=old.read_text().replace('bpe_production_window_coordinator_v2','bpe_production_window_coordinator_v3')
    start=text.index('    def writer(state,identity_value):');end=text.index("    pointer=f/'latest.json'",start)
    writer='''    def encode(value):
        if type(value) is dict:return ['dict',[[encode(k),encode(v)] for k,v in value.items()]]
        if type(value) in (list,tuple):return ['tuple' if type(value) is tuple else 'list',[encode(v) for v in value]]
        require(value is None or type(value) in (str,int,float,bool),'Fixture scalar metadata only')
        return ['scalar',value]
    def writer(state,identity_value):
        payload=f/('state_'+str(opt.updates)+'.bin');payload.write_bytes(b'FIXTURE_NOT_NATIVE_CHECKPOINT')
        metadata=f/('state_'+str(opt.updates)+'.json')
        value={'kind':'ASR18_broad_caller_state_v1','diagnostic_only':True,'identity':identity_value,
            'optimizer':{'updates':opt.updates},'sampler':state,'augmentation':{'committed_journal':journal.binding()}}
        tree=encode(value);tree[1].append([['scalar','torch_rng'],['tensor',0]])
        payload_binding=record(payload)
        write_json(metadata,{'kind':'ASR18_streaming_named_tensor_checkpoint_v1','schema_version':1,'endianness':'little',
            'source':record(S/'bpe_streaming_checkpoint_v1.py'),'dynamic_module_serialization':False,
            'maximum_CPU_tensor_slice_bytes':4194304,'payload':payload_binding,'tree':tree,
            'tensors':[{'index':0,'dtype':'uint8','shape':[payload_binding['bytes']],'offset':0,'bytes':payload_binding['bytes'],'sha256':payload_binding['sha256']}]})
        return record(metadata)
'''
    text=text[:start]+writer+text[end:]
    text=text.replace("checkpoint_identity={'FIXTURE_ONLY':True})","checkpoint_identity={'FIXTURE_ONLY':True},diagnostic_only=True)")
    text=text.replace('checkpoint_identity={})','checkpoint_identity={},diagnostic_only=True)')
    old_loop="    for _ in range(3):co.consume(prepare_evidence=evidence,native_call=native,verify_preparation=verify)"
    new_loop="""    co.consume(prepare_evidence=evidence,native_call=native,verify_preparation=verify)
    before_stale=pointer.read_bytes();live_state=copy.deepcopy(s.state_dict());live_prefix=journal.binding()
    reject(lambda:co.publish_checkpoint(write_coherent_checkpoint=lambda state,identity:published['checkpoint'],pointer_path=pointer,
        checkpoint_identity={'FIXTURE_ONLY':True},diagnostic_only=True),'Returned old checkpoint1 cannot publish live update2')
    check(pointer.read_bytes()==before_stale and s.state_dict()==live_state and journal.binding()==live_prefix and opt.updates==2,
          'Stale-return rejection preserves previous latest and every live frontier')
    reject(lambda:co.publish_checkpoint(write_coherent_checkpoint=writer,pointer_path=pointer,checkpoint_identity={},diagnostic_only=0),
          'Integer cannot masquerade as explicit diagnostic qualification scope')
    for _ in range(2):co.consume(prepare_evidence=evidence,native_call=native,verify_preparation=verify)"""
    require(text.count(old_loop)==1,'Exact stale-return extension');text=text.replace(old_loop,new_loop)
    insert="""    # Independent saved scalar inspection of the ACTUAL closed cleaner21.
    # No tensors/audio/model are loaded. Source/latest/journal are separate
    # expected runtime evidence, not decoded out of the candidate metadata.
    from bpe_streaming_checkpoint_semantics_v1 import verify_saved_metadata,select,decode_scalar_tree
    actual_folder=A/'cleaner_precomputed_after20_01';latest=json.loads((actual_folder/'latest_coherent_checkpoint.json').read_text())
    candidate=json.loads(Path(latest['checkpoint']['path']).read_text());declaration=json.loads((actual_folder/'declaration.json').read_text())
    expected_identity={'kind':'ASR18_cleaner_hook_checkpoint_identity_v1','config':record(S/'asr18_cleaner_hook_v2.disabled.json'),
        'probe':record(S/'asr18_cleaner_precomputed_probe_v2.py'),'native_consumer':record(S/'asr18_precomputed_native_v2.py'),
        'declaration':declaration,'parent_checkpoint':declaration['parent_checkpoint']}
    expected_prefix=CommitJournal(actual_folder/'from_actual20_authoritative_prefix.jsonl').binding()
    def semantic(value):return verify_saved_metadata(value,identity=expected_identity,optimizer_updates=21,
        sampler_state=latest['sampler'],journal_prefix=expected_prefix,diagnostic_only=True)
    semantic_proof=semantic(candidate)
    check(semantic_proof['passed'] and semantic_proof['optimizer_updates']==21,'Actual closed21 saved identity/frontier/sampler/journal matches independent runtime evidence without tensors')
    stale=json.loads((A/'feature_window_native20_01/after_update20.state.json').read_text())
    reject(lambda:semantic(stale),'Actual older20 metadata rejected against actual21 live identity/frontier')
    for name,path,replacement in (
        ('Stale saved Adam updates',('optimizer','updates'),['scalar',20]),
        ('Boolean saved Adam updates',('optimizer','updates'),['scalar',True]),
        ('Wrong saved sampler',('sampler',),['scalar',None]),
        ('Wrong saved journal path',('augmentation','committed_journal','path'),['scalar','wrong_prefix_path']),
        ('Wrong saved journal prefix hash',('augmentation','committed_journal','sha256'),['scalar','0'*64]),
        ('Boolean saved journal offset',('augmentation','committed_journal','offset'),['scalar',False]),
        ('Tensor reference forbidden in scalar identity',('identity',),['tensor',0]),
        ('Wrong saved qualification',('diagnostic_only',),['scalar',False])):
        broken=copy.deepcopy(candidate);select(broken['tree'],*path)[:]=replacement
        reject(lambda broken=broken:semantic(broken),name+' rejected before publication')
    duplicate=copy.deepcopy(candidate);duplicate['tree'][1].append(copy.deepcopy(duplicate['tree'][1][0]))
    reject(lambda:semantic(duplicate),'Duplicate serialized semantic dictionary key rejected')
    missing=copy.deepcopy(candidate);optimizer_node=select(missing['tree'],'optimizer')
    optimizer_node[1]=[pair for pair in optimizer_node[1] if pair[0]!=['scalar','updates']]
    reject(lambda:semantic(missing),'Missing saved optimizer update field rejected')
    reject(lambda:decode_scalar_tree(['unknown_tag',0]),'Unknown saved scalar tag fails closed')
"""
    point="    check(not any(n in sys.modules for n in ('torch','numpy','scipy','transformers','pyarrow'))"
    at=text.index(point);text=text[:at]+insert+text[at:]
    text=text.replace("'actual_cleaner_audio_or_frontend_proved':False,'production_ready':False}",
        "'actual_cleaner_audio_or_frontend_proved':False,'production_ready':False,'semantic_verifier':record(S/'bpe_streaming_checkpoint_semantics_v1.py'),\n        'actual21_scalar_metadata_proof':semantic_proof,'actual21_source':record(actual_folder/'actual_cleaner21_summary_v1.json'),\n        'fixture_writer_is_synthetic_not_actual_tensor_writer':True,'actual_native_tensor_payload_reloaded':False}")
    target.write_text(text);print(json.dumps({'source':record(target),'production_ready':False}))
if __name__=='__main__':main()
