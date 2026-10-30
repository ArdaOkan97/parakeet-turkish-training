"""Synthetic new E payloads only; no real checkpoints/model/GPU adoption."""
import argparse
from copy import deepcopy
import gzip,hashlib,json,os,sqlite3,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
import bpe_owned_checkpoint_storage_v2 as component
from fastmss_row_serial_cpu_v2 import environment,resource_guard,admit
from fastmss_owned_process_closure_v1 import identity

OWNER_CALLS=[]

def current_synthetic_owner(request):
    proof=component.read(request['proof'])
    OWNER_CALLS.append(request['operation'])
    return proof['synthetic_current_owner_ack'] is True and proof['nonce']==request['expected_nonce'] if 'expected_nonce' in request else proof['synthetic_current_owner_ack'] is True


def synthetic_verifier(pointer_pin,expected):
    pointer=component.read(pointer_pin);metadata=component.read(pointer['checkpoint'])
    component.require(pointer['kind']=='SYNTHETIC_ONLY_coherent_pointer' and metadata['kind']=='SYNTHETIC_ONLY_streamed_pair_metadata',
                      'Never spoof trusted streamed checkpoint/writer kinds in controls')
    component.bound(pointer['payload'])
    component.require(pointer['expected_sha256']==component.digest(expected) and metadata['payload']==pointer['payload'], 'Synthetic full expected state/payload binding')
    return {'pointer':pointer,'metadata':metadata,'expectation':deepcopy(expected),'semantic_proof':{'synthetic_only':True}}


def main(output):
    environment();resource_guard();CPU=admit(80_000_000)
    output=component.safe(output);component.require(not output.exists(),'Fresh E synthetic fixture only');output.mkdir()
    checks=[]
    def check(name,value):resource_guard();component.require(value,name);checks.append(name)
    def rejects(name,function):
        try:function()
        except (ValueError,KeyError,TypeError,OSError):checks.append(name);return
        raise ValueError('Expected rejection: '+name)
    policy={'kind':'ASR18_synthetic_owned_checkpoint_policy_v1','production_enabled':False,'retire_existing_or_original_artifacts':False,
        'retire_owned_metadata_or_pointer_files':False,'publication_cadence':'at_most_one_payload_per_actual_positive_optimizer_update',
        'maximum_payload_bytes':65536,'maximum_metadata_bytes':16384,'maximum_pointer_bytes':16384,'maximum_archive_record_bytes':16384,
        'maximum_publications':12,'planned_native_updates':12,'maximum_archive_bytes':15*16384,
        'maximum_retained_metadata_bytes':12*32768,'maximum_control_bytes':8*1024**2,'maximum_history_record_bytes':2048,
        'maximum_catalog_bytes':1024**2,'steady_live_pairs':4,'maximum_inflight_publications':1,'hard_live_pair_peak':5}
    policy_pin=component.new_json(output/'synthetic_policy.json',policy)
    original=output/'synthetic_external_original.bin';original.write_bytes(b'never owned, never removed');original_pin=component.record(original)
    role_pins=[]
    for role in ('source','history','DEV','qualified_winner'):
        path=output/('synthetic_protected_'+role+'.json');path.write_text(json.dumps({'synthetic_only':True,'role':role}))
        role_pins.append(component.record(path))
    protection={'kind':'ASR18_future_owned_checkpoint_protection_plan_v1','adopt_external_sources_as_owned':False,
        'required_roles':{role:[pin] for role,pin in zip(('source','history','DEV','qualified_winner'),role_pins)}}
    protection_pin=component.new_json(output/'synthetic_protection_plan.json',protection)
    check('Exact nonempty reviewed role plan matches source pins',component.protection_plan(protection_pin,role_pins)==role_pins)
    rejects('Empty caller protected list rejected',lambda:component.protection_plan(protection_pin,[]))
    rejects('Omitted protected source pin rejected',lambda:component.protection_plan(protection_pin,role_pins[1:]))
    rejects('Changed protected list cannot inherit review',lambda:component.protection_plan(protection_pin,[original_pin]+role_pins[1:]))
    rejects('Reordered protected list cannot inherit review',lambda:component.protection_plan(protection_pin,list(reversed(role_pins))))
    changed=deepcopy(protection);changed['required_roles'].pop('DEV')
    changed_pin=component.new_json(output/'synthetic_missing_DEV_plan.json',changed)
    rejects('Required DEV role cannot be omitted',lambda:component.protection_plan(changed_pin,role_pins))
    changed=deepcopy(protection);changed['required_roles']['qualified_winner']=[]
    changed_pin=component.new_json(output/'synthetic_empty_winner_plan.json',changed)
    rejects('Qualified initial winner role cannot be empty',lambda:component.protection_plan(changed_pin,role_pins[:3]))
    journal=output/'synthetic_authoritative_journal.jsonl';journal.write_bytes(b'original-prefix\n')
    prefix={'path':str(journal),'offset':journal.stat().st_size,'sha256':hashlib.sha256(journal.read_bytes()).hexdigest()}
    verifier_source=component.record(__file__);run={'kind':'SYNTHETIC_ONLY_not_full_DATA_or_ROOT_review','scientific_clocks':[64250,2000,62250]}
    def store(name):return component._StorageProtocol.initialize(output/name,policy_pin,run_identity=run,verifier_source=verifier_source,
        protected=[original_pin],checkpoint_verifier=synthetic_verifier,guard=resource_guard,synthetic=True)
    serial=0
    def proof(store,**extra):
        nonlocal serial
        state,entries,_=store._load();serial+=1
        value={'kind':'ASR18_owned_checkpoint_mapping_release_quiescence_v1','owner':component.OWNER,'ownership':store.marker_pin,
            'state_revision':state['revision'],'catalog':state['catalog'],'nonce':('a'*32)+str(serial),
            'coherent_latest_pointer':entries[state['latest']]['pointer'] if state['latest'] else None,
            'protected_owned_winners':deepcopy(state['winners']),
            'native_work_quiescent':True,'checkpoint_mappings_released':True,'candidate_payload_handles_released':True,
            'checkpoint_write_inflight':False,'restore_in_progress':False,'synthetic_current_owner_ack':True,**extra}
        return component.new_json(output/f'synthetic_owner_proof_{serial:03}.json',value)
    def prepare(store,step):
        intent=store.reserve_publication();payload=Path(intent['payload_path']);payload.write_bytes(bytes([step])*4096)
        metadata={'kind':'SYNTHETIC_ONLY_streamed_pair_metadata','payload':component.record(payload),'no_model_tensors':True}
        mp=component.new_json(intent['metadata_path'],metadata)
        expected={'run_identity':run,'optimizer_updates':step,'identity':{'synthetic_only':True,'step':step},
            'sampler_state':{'synthetic_only':True,'updates':step},'journal_prefix':deepcopy(prefix),
            'source_history_and_DEV_links':[original_pin]}
        pointer={'kind':'SYNTHETIC_ONLY_coherent_pointer','checkpoint':mp,'payload':component.record(payload),'expected_sha256':component.digest(expected)}
        pp=component.new_json(intent['pointer_path'],pointer)
        return intent,pp,expected
    def publish(store,step):
        intent,pp,expected=prepare(store,step)
        return store.complete_publication(intent,pp,expected,proof(store),current_synthetic_owner)
    s=store('owned_synthetic_01');check('Fresh separate synthetic owner kind',s.owner['kind']!='ASR18_future_owned_checkpoint_storage_namespace_v1')
    first=publish(s,1);check('Publication verifies bytes before latest',s._load()[0]['latest']==first['id'])
    archive=json.loads(gzip.decompress(Path(first['archive']['path']).read_bytes()))
    check('Fsynced archive preserves full metadata and authoritative prefix',archive['metadata']['payload']==first['payload'] and archive['actual_live_expectation']['journal_prefix']==prefix)
    check('External original winner/source never adopted',s.owner['existing_source_checkpoints_adopted_as_owned'] is False and s._load()[2]['live_pairs']==1)
    second=publish(s,2)
    qp=proof(s,strict_all_six_qualified=True,qualified_pointer=first['pointer'])
    s.protect_qualified_winner(first['id'],qp,current_synthetic_owner)
    check('Actual qualification proof archived before winner protection','qualification' in s._load()[1][first['id']])
    rejects('Qualified winner payload never retired',lambda:s.retire_obsolete_payloads([first['id']],proof(s,retire_checkpoint_ids=[first['id']]),current_synthetic_owner))
    rejects('Coherent latest payload never retired',lambda:s.retire_obsolete_payloads([second['id']],proof(s,retire_checkpoint_ids=[second['id']]),current_synthetic_owner))
    rejects('External original ID cannot be retired',lambda:s.retire_obsolete_payloads([str(original)],proof(s,retire_checkpoint_ids=[str(original)]),current_synthetic_owner))
    third=publish(s,3);before_metadata=deepcopy(second['metadata']);retire=proof(s,retire_checkpoint_ids=[second['id']])
    result=s.retire_obsolete_payloads([second['id']],retire,current_synthetic_owner)
    check('Only explicit obsolete future-owned payload removed',not Path(second['payload']['path']).exists() and Path(first['payload']['path']).exists() and Path(third['payload']['path']).exists())
    check('Retired metadata/pointer/archives remain exact',component.record(before_metadata['path'])==before_metadata and component.record(second['pointer']['path'])==second['pointer'] and component.record(second['archive']['path'])==second['archive'])
    check('Protected current/original byte source unchanged',component.record(original)==original_pin)
    rejects('Duplicate retired payload retirement rejected',lambda:s.retire_obsolete_payloads([second['id']],proof(s,retire_checkpoint_ids=[second['id']]),current_synthetic_owner))
    bad=proof(s,retire_checkpoint_ids=[first['id']],restore_in_progress=True)
    rejects('Active restore/mapping conflict blocks cleanup',lambda:s.retire_obsolete_payloads([first['id']],bad,current_synthetic_owner))
    bad=proof(s,retire_checkpoint_ids=[first['id']],checkpoint_mappings_released=False)
    rejects('Live resident idle does not prove mapping release',lambda:s.retire_obsolete_payloads([first['id']],bad,current_synthetic_owner))
    bad=proof(s,retire_checkpoint_ids=[first['id']],synthetic_current_owner_ack=False)
    rejects('Current owner callback must acknowledge nonce/state',lambda:s.retire_obsolete_payloads([first['id']],bad,current_synthetic_owner))
    fourth=publish(s,4);fifth=publish(s,5);sixth=publish(s,6)
    check('One publication may reach exact hard5 pair peak',s._load()[2]['live_pairs']==5)
    rejects('No sixth live payload/inflight window',lambda:s.reserve_publication())
    s.retire_obsolete_payloads([third['id'],fourth['id']],proof(s,retire_checkpoint_ids=[third['id'],fourth['id']]),current_synthetic_owner)
    check('Explicit retirement restores bounded steady state',s._load()[2]['live_pairs']==3)
    # Separate namespaces retain each intentional failure evidence unchanged.
    stale=store('stale_return_01');publish(stale,1);intent,pp,expected=prepare(stale,1)
    rejects('Valid older checkpoint cannot advance current latest',lambda:stale.complete_publication(intent,pp,expected,proof(stale),current_synthetic_owner))
    check('Stale rejection preserves previous latest',stale._load()[0]['latest']=='p000000001')
    broken=store('unknown_file_01');publish(broken,1);(broken.folder/'unowned.bin').write_bytes(b'unknown')
    rejects('Unknown file stops before any broad cleanup',lambda:broken.reserve_publication())
    conflict=store('payload_corruption_01');a=publish(conflict,1);Path(a['payload']['path']).write_bytes(b'changed')
    rejects('Changed owned payload hash fails closed',lambda:conflict.reserve_publication())
    pending=store('unfinished_write_01');intent=pending.reserve_publication()
    rejects('Second publication cannot adopt unfinished writer files',lambda:pending.reserve_publication())
    check('Exact original writer payload suffix retained',Path(intent['metadata_path']).with_suffix('.tensors.bin')==Path(intent['payload_path']))
    bad=deepcopy(policy);bad['maximum_archive_bytes']=1
    rejects('Full-horizon archive count cannot use minimum20k windows',lambda:component.policy_values(bad,synthetic=True))
    bad=deepcopy(policy);bad['maximum_publications']=1
    rejects('All planned native-update publication worst count mandatory',lambda:component.policy_values(bad,synthetic=True))
    bad=deepcopy(policy);bad['retire_existing_or_original_artifacts']=True
    rejects('No original/current adoption or cleanup authority',lambda:component.policy_values(bad,synthetic=True))
    rejects('Path traversal cannot escape E',lambda:component.safe(r'C:\not-authorized-new-storage'))
    rejects('Public store cannot open synthetic/proposal owner',lambda:component.OwnedFutureCheckpointStore(s.marker_pin,checkpoint_verifier=component.verify_streamed_pointer,guard=resource_guard))
    altered=store('owner_marker_mismatch_01');changed_owner=deepcopy(altered.owner);changed_owner['protected_external']=[]
    (altered.folder/'ownership.json').write_bytes(component.canonical(changed_owner))
    rejects('Changed owner-marker protected list cannot inherit pinned ownership',lambda:altered.reserve_publication())
    # Typed streamed FORMAT-only scalar fixture. No actual streamed tensor
    # writer/model ran; it is never adopted by a public/ROOT-reviewed store.
    # This exercises the frozen semantic verifier rather than the separate
    # synthetic file engine's checkpoint callback.
    def encode(value):
        if type(value) is dict:return ['dict',[[encode(k),encode(v)] for k,v in value.items()]]
        if type(value) is list:return ['list',[encode(v) for v in value]]
        if type(value) is tuple:return ['tuple',[encode(v) for v in value]]
        return ['scalar',value]
    format_dir=output/'synthetic_stream_format_only';format_dir.mkdir()
    raw=format_dir/'scalar-format.tensors.bin';raw.write_bytes(b'FORMAT_ONLY_NOT_NATIVE_MODEL_STATE')
    expected={'identity':{'kind':'synthetic-format-only','nested_integer':1},'optimizer_updates':2,
        'sampler_state':{'carried_replay':{'updates':2},'synthetic_only':True},'journal_prefix':prefix,'run_identity':run}
    saved={'kind':'ASR18_broad_caller_state_v1','diagnostic_only':False,'identity':expected['identity'],
        'optimizer':{'updates':2},'sampler':expected['sampler_state'],'augmentation':{'committed_journal':prefix}}
    metadata={'kind':'ASR18_streaming_named_tensor_checkpoint_v1','schema_version':1,'endianness':'little',
        'dynamic_module_serialization':False,'maximum_CPU_tensor_slice_bytes':4*1024**2,
        'source':component.record(PROJECT/'outputs/parakeet_turkish/bpe_streaming_checkpoint_v1.py'),
        'payload':component.record(raw),'tensors':[{'index':0,'dtype':'uint8','shape':[raw.stat().st_size],
            'offset':0,'bytes':raw.stat().st_size,'sha256':component.record(raw)['sha256']}],
        'tree':encode(saved),'synthetic_format_fixture_only':True,'actual_streamed_writer_or_native_model_run':False}
    mp=component.new_json(format_dir/'scalar-format.json',metadata)
    semantic=component.verify_saved_metadata(metadata,identity=expected['identity'],optimizer_updates=2,
        sampler_state=expected['sampler_state'],journal_prefix=prefix,diagnostic_only=False)
    value={'kind':'ASR18_production_window_completed_checkpoint_pointer_v2','checkpoint':mp,'payload':component.record(raw),
        'identity':expected['identity'],'committed_updates':2,'sampler_state_sha256':component.digest(expected['sampler_state']),
        'journal_prefix':prefix,'saved_semantic_state_verified':semantic,'writer_returned_and_payload_verified':True,
        'all_window_updates_committed':True,'diagnostic_only':False,'synthetic_format_fixture_only':True}
    pp=component.new_json(format_dir/'scalar-format.pointer.json',value)
    check('Frozen real scalar semantic verifier exercised on explicitly synthetic FORMAT',component.verify_streamed_pointer(pp,expected)['semantic_proof']['passed'] is True)
    bad=deepcopy(expected);bad['identity']['nested_integer']=True
    rejects('Canonical identity rejects nested boolean versus integer',lambda:component.verify_streamed_pointer(pp,bad))
    bad=deepcopy(expected);bad['optimizer_updates']=3
    rejects('Saved optimizer frontier cannot lag live snapshot',lambda:component.verify_streamed_pointer(pp,bad))
    bad=deepcopy(expected);bad['sampler_state']['carried_replay']['updates']=3
    rejects('Saved replay sampler cannot lag live snapshot',lambda:component.verify_streamed_pointer(pp,bad))
    bad=deepcopy(expected);bad['journal_prefix']['sha256']='0'*64
    rejects('Saved authoritative journal prefix cannot drift',lambda:component.verify_streamed_pointer(pp,bad))
    other=deepcopy(value);other['diagnostic_only']=True
    diagnostic=component.new_json(format_dir/'diagnostic.pointer.json',other)
    rejects('Diagnostic checkpoint cannot enter production owned latest',lambda:component.verify_streamed_pointer(diagnostic,expected))
    check('Owner callbacks are rechecked at every unlink',sum(v.startswith('unlink_exact') for v in OWNER_CALLS)>=3)
    check('No Torch/model/frontend/GPU imported',not any(k=='torch' or k.startswith(('torch.','numpy','scipy','transformers','pyarrow')) for k in sys.modules))
    report={'kind':'ASR18_owned_future_checkpoint_storage_synthetic_controls_v1','passed':True,'synthetic_only':True,
        'producer':identity(),'worker_source':component.record(__file__),'parent_source':component.record(__file__),
        'component_source':component.record(component.__file__),'check_count':len(checks),'checks':checks,'CPU_admission':CPU,
        'protected_original_fixture':original_pin,'original_fixture_still_exact':component.record(original)==original_pin,
        'real_current_or_original_checkpoints_changed':False,'actual_model_or_GPU_or_native_updates':False,
        'actual_full_run_storage_throughput_or_publication_restore_proved':False,'production_enabled':False}
    report['scalar_stream_format_test_is_explicitly_synthetic_no_actual_writer_or_model']=True
    component.new_json(output/'report.json',report);print(json.dumps({'passed':True,'checks':len(checks),'synthetic_only':True}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args();main(args.output)
