"""Synthetic new E payloads only; no real checkpoints/model/GPU adoption."""
import argparse
from copy import deepcopy
import gzip,hashlib,json,os,sqlite3,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
import bpe_owned_checkpoint_storage_v1 as component
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
    check('Owner callbacks are rechecked at every unlink',sum(v.startswith('unlink_exact') for v in OWNER_CALLS)>=3)
    check('No Torch/model/frontend/GPU imported',not any(k=='torch' or k.startswith(('torch.','numpy','scipy','transformers','pyarrow')) for k in sys.modules))
    report={'kind':'ASR18_owned_future_checkpoint_storage_synthetic_controls_v1','passed':True,'synthetic_only':True,
        'producer':identity(),'worker_source':component.record(__file__),'parent_source':component.record(__file__),
        'component_source':component.record(component.__file__),'check_count':len(checks),'checks':checks,'CPU_admission':CPU,
        'protected_original_fixture':original_pin,'original_fixture_still_exact':component.record(original)==original_pin,
        'real_current_or_original_checkpoints_changed':False,'actual_model_or_GPU_or_native_updates':False,
        'actual_full_run_storage_throughput_or_publication_restore_proved':False,'production_enabled':False}
    component.new_json(output/'report.json',report);print(json.dumps({'passed':True,'checks':len(checks),'synthetic_only':True}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args();main(args.output)
