"""Inactive import/typed saved134400 entry; original scheduling and numerics retained."""
import ast,copy,difflib,hashlib,json
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research';N=Path(__file__).parent
def pin(p):
    p=Path(p);r=p.read_bytes();return {'path':str(p.resolve()),'bytes':len(r),'sha256':hashlib.sha256(r).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):assert not p.exists();p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
cohort=[]
def source(old,new,changes):
    original=old.read_text(encoding='utf-8');text=original
    for before,after,count in changes:assert text.count(before)==count,before;text=text.replace(before,after)
    ast.parse(text);inverse=text
    for before,after,count in reversed(changes):assert inverse.count(after)==count;inverse=inverse.replace(after,before)
    assert inverse==original and not new.exists();new.write_text(text,encoding='utf-8')
    delta=N/(new.stem+'.diff.txt');delta.write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(old),tofile=str(new))),encoding='utf-8')
    cohort.append({'old':pin(old),'new':pin(new),'delta':pin(delta),'declared_changes':changes,'exact_inverse_full_source':True})
source(S/'fastmss_resident_feature_stream_v6.py',S/'fastmss_resident_feature_stream_v7.py',[
    ('from fastmss_original_clip_geometry_overlay_v4 import OriginalRows','from fastmss_original_clip_geometry_overlay_v5 import OriginalRows',1)])
old=R/'run_ASR18_BPE9223_resident_RAM_training_v15.py';oldtext=old.read_text(encoding='utf-8')
before=oldtext[oldtext.index('def checked_original_interval_resume('):oldtext.index('\n\ndef private_successor_request(')]
after='''def checked_original_interval_resume(pin,*,base,folder,pointer_pin):
    """Exact EOF failure after134529, saved134400 and129 preserved positive attempts."""
    import psutil
    require(pin==base['prior_resident_RAM_original_interval_failure']and record(pin['path'])==pin,'Exact ROOT-bound actual EOF failure')
    value=read_json(pin['path']);e=value['source_evidence']
    require(value['kind']=='ASR18_actual_resident_RAM_original_interval_post134529_saved134400_abnormal_closure_v1'
        and value['actual_owned_session']==60255 and value['actual_TOOL_exit_code']==1 and value['actual_TOOL_chunk_id']=='4c0093'
        and value['actual_failed_optimizer_updates']==134529 and value['saved_optimizer_updates']==134400
        and value['CPU_completed_updates_this_owner']==1529 and value['normal_completion_claimed']is False
        and value['native_OS_exit_code']is None and value['native_numeric_OS_exit_not_observed']is True
        and value['actual_wrapper_wait_code_was_nonzero']is True and value['CPU_OS_exit_code']is None
        and value['original_failure_was_source_coverage_not_resource_guard_or_normal0']is True,
        'Truthful original coverage TOOL1, unknown OS codes and preserved129 attempts; never normal0')
    require(e['saved_pointer']==pointer_pin and Path(pointer_pin['path']).resolve().is_relative_to(folder/'checkpoints')
        and e['driver_source']==record(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v15.py')
        and e['wrapper_source']==record(P/'work/research/run_ASR18_BPE9223_resident_RAM_autocontinue_v15.py')
        and e['CPU_source']==record(S/'fastmss_resident_feature_stream_v6.py'),'Exact failed source15/CPU6 and original branch')
    for item in e.values():require(record(item['path'])==item,'Exact small frozen failed-owner/source/checkpoint evidence')
    require(record(value['owned_process_deaths']['path'])==value['owned_process_deaths'],'Exact observed owned deaths')
    deaths=read_json(value['owned_process_deaths']['path'])['identities']
    require(len(deaths)==5 and {x['label']for x in deaths}=={'native','native interpreter launcher','resident CPU','CPU interpreter launcher','outer wrapper'},'Only five actually known identities')
    for item in deaths:
        who=item['original_identity']
        try:now=psutil.Process(who['pid']).create_time()
        except psutil.NoSuchProcess:continue
        require(who['create_time']is not None and abs(now-who['create_time'])>=.001,'Original owned identity actually dead, no fabricated timestamp')
    failure=read_json(e['actual_failure']['path']);monitor=read_json(e['monitor_completion']['path']);CPU=read_json(e['CPU_failure_frame']['path'])
    require('Original interval beyond source unresolved'in failure['error']
        and 'Original interval beyond source unresolved'in Path(e['CPU_stderr']['path']).read_text(encoding='utf-8')
        and CPU['completed_updates']==1529 and failure['actual_optimizer_updates']==134529 and failure['last_checkpoint']==134400
        and monitor['joined']is True and monitor['clean']is True and monitor['error']is None,
        'Actual direct failed coverage guard, not a resource or masked duplicate-plan failure')
    remaining_pin=value['ordered_current_plan_remaining_metadata'];require(record(remaining_pin['path'])==remaining_pin,'Actual remaining ordered plan')
    remaining=read_json(remaining_pin['path']);plan_pin=remaining['source_plan'];future_pin=remaining['later_private_plan_not_authoritative']
    require(record(plan_pin['path'])==plan_pin and record(future_pin['path'])==future_pin,'Exact current and later private plan bytes')
    plan=read_json(plan_pin['path']);future=read_json(future_pin['path'])
    require(plan['initial_committed_updates']==134520 and len(plan['updates'])==20
        and future['initial_committed_updates']==134540 and len(future['updates'])==20
        and remaining=={'source_plan':plan_pin,'initial_updates':134520,'first_unconsumed_update_index':9,
            'candidate_global_updates':134530,'updates':plan['updates'][9:],'not_assumed_native_plus_one':True,
            'CPU_actual_completed_updates':1529,'native_actual_completed_updates_since133000':1529,
            'later_private_plan_not_authoritative':future_pin},'Actual ordered plan/CPU counts, not native+1 alone or fictitious future consumption')
    saved={k:value['saved_journal_prefix'][k]for k in ('path','offset','sha256')};whole=value['whole_old_journal']
    require(saved=={'path':str(folder/'journals/commits_000011.jsonl'),'offset':273582680,
        'sha256':'49149d8ad291482561dca0398a11a0379119b0ed54380c6eb67b93fb83fac0ac'}
        and read_json(pointer_pin['path'])['journal_prefix']==saved and whole['path']==saved['path']
        and Path(whole['path']).stat().st_size==whole['bytes'],'Exact saved134400 prefix and unchanged whole failed journal')
    tail={k:value['preserved_unsaved_positive_tail'][k]for k in ('path','bytes','sha256')}
    require(record(tail['path'])==tail and whole['bytes']-saved['offset']==tail['bytes'],'Exact retained positive suffix, no truncation/replay')
    events=[__import__('json').loads(line)for line in Path(tail['path']).read_bytes().splitlines()]
    require(len(events)==129 and all(x['optimizer_updates']==x['native']['update']==x['prepared_evidence']['actual_positive_native_update']==u
        for u,x in zip(range(134401,134530),events)),'All129 ordered positive attempts remain historical')
    return {**value,'saved_journal_prefix':saved,'whole_journal_including_preserved_tail':whole,
        'preserved_unsaved_tail':tail,'preserved_tail_positive_updates':list(range(134401,134530))}
'''
source(old,R/'run_ASR18_BPE9223_resident_RAM_training_v16.py',[(before,after,1),
    ('fastmss_resident_feature_stream_v6','fastmss_resident_feature_stream_v7',2),
    ("elif interval_failure is not None:require(a.prior_process is None and a.invocation==10,","elif interval_failure is not None:require(a.prior_process is None and a.invocation==12,",1),
    ('ASR18_readonly_saved88400_prefix_after_original_interval_failure_v1','ASR18_readonly_saved134400_prefix_after_original_interval_failure_v1',1)])
source(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v15.py',R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v16.py',[
    ('run_ASR18_BPE9223_resident_RAM_training_v15','run_ASR18_BPE9223_resident_RAM_training_v16',2),
    ('fastmss_resident_feature_stream_v6.py','fastmss_resident_feature_stream_v7.py',1),
    ("and review['original_clip_geometry_overlay']==configuration['original_clip_geometry_overlay'],",
     "and review['original_clip_geometry_overlay']==configuration['original_clip_geometry_overlay']\n            and a.start_invocation==review['initial_owner_invocation']==12,",1),
    ('actual abnormal88400 predecessor','actual abnormal134400 predecessor',1)])
basepin=N.parent/'resident_RAM_single_packet_resource_source_v1/configuration_candidate.json';base=read(basepin);candidate=copy.deepcopy(base)
candidate['original_clip_geometry_overlay']={'source':pin(S/'fastmss_original_clip_geometry_overlay_v5.py'),
    'certificate':pin(N/'fifth_original_full_clip_geometry_certificate_v1.json'),'ROOT_review':None,
    'prior_overlay':copy.deepcopy(base['original_clip_geometry_overlay'])}
candidate['prior_resident_RAM_original_interval_failure']=pin(N/'actual_failure_closure.json')
candidate['resident_RAM_resource_scheduling_policy']['CPU_protocol_source']=pin(S/'fastmss_resident_feature_stream_v7.py')
inverse=copy.deepcopy(candidate)
for key in ('original_clip_geometry_overlay','prior_resident_RAM_original_interval_failure','resident_RAM_resource_scheduling_policy'):inverse[key]=base[key]
assert inverse==base
for k,v in base['resident_RAM_resource_scheduling_policy'].items():
    assert k=='CPU_protocol_source'or candidate['resident_RAM_resource_scheduling_policy'][k]==v
proposal=N/'configuration_proposal_template.json';write(proposal,candidate)
handoff={'kind':'ASR18_INACTIVE_saved134400_fifth_clip_owner16_source_handoff_v1','production_admitted':False,
    'sources':{'driver':pin(R/'run_ASR18_BPE9223_resident_RAM_training_v16.py'),'wrapper':pin(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v16.py'),
        'CPU':pin(S/'fastmss_resident_feature_stream_v7.py'),'overlay':pin(S/'fastmss_original_clip_geometry_overlay_v5.py')},
    'certificate':pin(N/'fifth_original_full_clip_geometry_certificate_v1.json'),'configuration_proposal':pin(proposal),'prior_configuration':pin(basepin),
    'source_deltas':cohort,'exact_base_config_except_overlay_failure_and_CPU_protocol_pin':True,
    'prior_four_overlay_policy_exact':True,'source15_resource_policy_values_unchanged':True,'native_commit_DEV_prefetch_numerical_bodies_unchanged':True,
    'future_invocation':12,'future_journal':'commits_000012.jsonl','future_output':'resident_RAM_training_actual_13',
    'typed_saved_frontier':134400,'historical_positive_tail':129,'actual_CPU_ingress_control_pending':True}
out=N/'owner16_stream7_inactive_source_handoff_v1.json';write(out,handoff);print(json.dumps({'handoff':pin(out),'sources':handoff['sources'],'proposal':pin(proposal)}),flush=True)
