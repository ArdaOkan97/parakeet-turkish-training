"""ROOT-reviewed metadata derivatives; never launches jobs or mutates run state."""
import copy, datetime, hashlib, json
from pathlib import Path

N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
    p=Path(p); b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def save(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists(),p
    p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return pin(p)
def main():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    hand=load(N/'ACKed60_session_reentry_source_v1/source_handoff.json')
    oldpath=N/'ROOT_BPE9223_resource_stop_operational_compatibility_review_v11.json'
    old=load(oldpath); c=load(hand['proposal']['path'])
    manifest=load(c['operational_source_manifest']['path'])
    original=load(c['original_source_manifest']['path'])
    assert len(manifest['bindings'])==464 and len(original['bindings'])==432
    actual={x['path']:x for x in manifest['bindings']}
    assert all(actual.get(x['path'])==x for x in original['bindings'])
    for x in manifest['bindings']: assert pin(x['path'])==x,x['path']
    c.update(accepted=True,reviewer='/root',proposal_only_not_ROOT_authority=False,at_utc=now,
        exact_current_source_handoff=pin(N/'ACKed60_session_reentry_source_v1/source_handoff.json'),
        source_only_ACKed60_session_authority_derivative=True,
        reviewed_scope='Already genuine recovered/ACKed60; exact p4 reentry, stable original readonly metadata authority, unchanged numerical/resource policy. No duplicate fresh60 recovery; future native/recovery evidence still required.')
    comp=save(N/'ROOT_BPE9223_resource_stop_operational_compatibility_review_v12.json',c)
    mappings={}
    for role,before in old['operational_sources'].items():
        after=c['operational_sources'][role]
        if before!=after:mappings[before['path']]=(before,after)
    mappings[str(oldpath)]=(pin(oldpath),comp)
    def replace(o):
        if isinstance(o,dict):
            if set(o)=={'path','bytes','sha256'} and o['path'] in mappings:
                before,after=mappings[o['path']];assert o==before,o
                return copy.deepcopy(after)
            result={}
            for k,v in o.items():
                nk=k
                if isinstance(v,dict) and set(v)=={'path','bytes','sha256'} and v['path'] in mappings:
                    before,after=mappings[v['path']]
                    if k==Path(before['path']).name:nk=Path(after['path']).name
                assert nk not in result
                result[nk]=replace(v)
            return result
        if isinstance(o,list):return [replace(v) for v in o]
        return copy.deepcopy(o)
    cpu_old=N/'operational_CPU_configuration_v5'; cpu_new=N/'operational_CPU_configuration_v6'
    oldh=load(cpu_old/'handoff.json')
    for name in ('CPU_numerical_contract.json','operational_CPU_session.json','CPU_window_template.json','ROOT_static_CPU_execution_resource_review.json'):
        before=cpu_old/name;o=replace(load(before))
        if name=='operational_CPU_session.json':
            prior=load(before)
            assert o['operational_metadata']==prior['operational_metadata']==c['process_readonly_metadata_authority']
            assert o['prior_operational_session']==prior['prior_operational_session']
        if name.startswith('ROOT_'):
            o.update(prior_operational_CPU_review=pin(before),at_utc=now,
                source_only_ACKed60_session_authority_derivative=True,current_derivative_audio_execution_claimed=False)
        out=save(cpu_new/name,o);mappings[str(before)]=(pin(before),out)
    ch=replace(oldh);ch.update(kind='ROOT_operational_CPU_configuration_authorized_handoff_v6',at_utc=now,
        prior_handoff=pin(cpu_old/'handoff.json'),numerical_policy_and_all_resource_numbers_unchanged_from_v5=True,
        source_only_ACKed60_session_authority_derivative=True)
    cpu_handoff=save(cpu_new/'handoff.json',ch)
    dev_old=N/'operational_DEV_authorities_v11';dev_new=N/'operational_DEV_authorities_v12'
    dh=load(dev_old/'handoff.json')
    for name in ('ROOT_DEV_policy_review.json','ROOT_DEV_resource_review.json','ROOT_DEV_static_execution_review.json'):
        before=dev_old/name;o=replace(load(before))
        o.update(prior_operational_authority=pin(before),at_utc=now,
            source_only_ACKed60_session_authority_derivative=True,accepted=True,reviewer='/root')
        out=save(dev_new/name,o);mappings[str(before)]=(pin(before),out)
    dh=replace(dh);dh.update(kind='ROOT_BPE9223_operational_DEV_authority_derivatives_handoff_v12',at_utc=now,
        prior_handoff=pin(dev_old/'handoff.json'),source_only_ACKed60_session_authority_derivative=True)
    dev_handoff=save(dev_new/'handoff.json',dh)
    source=Path(__file__).parent/'prepare_ASR18_BPE9223_operational_configuration_v16.py'
    review=dict(kind='ROOT_BPE9223_operational_configuration_metadata_source_read_review_v1',accepted=True,reviewer='/root',
        scope='metadata_BPE9223_committed60_merged_CPU_configuration_v1',source=pin(source),operational_compatibility=comp,
        production_approval_claimed=False,at_utc=now,prior_source_scope=pin(N/'ROOT_operational_configuration_metadata_source_review_v12.json'),
        reviewed_delta='Actual ACKed60 failed-owner11 snapshot and reserved p4; original60 resource proof preserved separately; no repeated GPU recovery. Stdlib config/request assembly and prospective DEV constructor only.',
        actual_run_mutation_or_GPU_admitted=False)
    rp=save(N/'ROOT_operational_configuration_metadata_source_review_v13.json',review)
    print(json.dumps(dict(compatibility=comp,CPU_handoff=cpu_handoff,DEV_handoff=dev_handoff,metadata_source_review=rp)))
if __name__=='__main__':main()
