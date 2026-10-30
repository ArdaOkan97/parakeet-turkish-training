"""INACTIVE metadata materializer AFTER actual admitted stop seal; no job/restore.
Replaces only candidate actual-transition None with its real immutable closure pin.
"""
import argparse,copy,hashlib,json
from pathlib import Path
F=Path(__file__).resolve().parent
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
def require(v,m):
    if not v:raise ValueError(m)
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,indent=2);f.write('\n')
def main(a):
    closure_pin=pin(a.actual_closure);closure=read(a.actual_closure)
    require(closure['kind']=='ASR18_actual_approved_source27_checkpoint_lookahead_transition_v1'
        and closure['actual_outer_TOOL_exit_code']==1 and closure['all_five_exact_dead']is True,
        'Only real admitted old owner TOOL1/dead seal can fill final actual-transition field')
    candidate_pin=pin(F/'candidate_configuration_completed_CPU_pending_actual_transition_v1.json');candidate=read(candidate_pin['path'])
    require(candidate['prior_resident_RAM_approved_lookahead_checkpoint_transition']is None,
        'Previously reviewed candidate retains exactly one actual-closure None slot')
    final=copy.deepcopy(candidate);final['prior_resident_RAM_approved_lookahead_checkpoint_transition']=closure_pin
    fresh=F/'production_configuration_v1';require(not fresh.exists(),'One immutable final config');fresh.mkdir()
    final_path=fresh/'configuration.json';write(final_path,final)
    restored=copy.deepcopy(final);restored['prior_resident_RAM_approved_lookahead_checkpoint_transition']=None
    require(restored==candidate,'Only None to genuine actual closure pin')
    source_base=F.parent/'resident_RAM_source26_postrestore_selector_failure_actual_v1/inactive_selector_fix_v2/production_configuration_v1/configuration.json'
    previous=read(source_base);inverse=copy.deepcopy(final);del inverse['prior_resident_RAM_approved_lookahead_checkpoint_transition']
    inverse['resident_RAM_packet_credit_policy']=previous['resident_RAM_packet_credit_policy']
    inverse['resident_RAM_resource_scheduling_policy']['CPU_protocol_source']=previous['resident_RAM_resource_scheduling_policy']['CPU_protocol_source']
    require(inverse==previous,'Every source27 byte/floor/science value exact after three operational fields')
    pointer_pin=closure['evidence']['pointer'];pointer=read(pointer_pin['path']);driver=pin(R/'run_ASR18_BPE9223_resident_RAM_training_v29.py')
    wrapper=pin(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v29.py');final_pin=pin(final_path)
    future={'worker_driver_source':driver,'parent_driver_source':driver,'resident_RAM_stream_configuration':final_pin,
        'resident_branch_namespace':str(B),'checkpoint_interval_updates':200,'source_checkpoint':pointer_pin,'source_configuration':final_pin}
    args=['--configuration',str(final_path),'--pointer',pointer_pin['path'],'--latest-failure',
        str(F.parent/'pipeline_overlap_RUN07_actual_01/actual_control_namespace_failure_TOOL1_v1.json'),
        '--branch',str(B),'--output',str(F.parent/'resident_RAM_training_actual_22'),
        '--queue-bytes','268435456','--start-invocation','21','--lookahead-resume',closure_pin['path'],
        '--review','<ROOT exact final execution binding required>']
    require(len(args)==18 and not(B/'invocations/owner_000021').exists()and not(B/'journals/commits_000021.jsonl').exists()
        and not(F.parent/'resident_RAM_training_actual_22').exists(),'Fresh reviewed future21/actual22 names, no launch')
    facts={'source_base':pin(source_base),'candidate':candidate_pin,'final_configuration':final_pin,
        'candidate_delta':'ONLY prior_resident_RAM_approved_lookahead_checkpoint_transition None to genuine closure pin',
        'source27_structural_inverse':'ONLY CPU protocol pointer, approved packet credit policy, actual transition provenance',
        'original_seven_future_runtime_bindings':future,'saved_optimizer_updates':pointer['committed_updates'],
        'actual_optimizer_updates_at_stop':closure['actual_optimizer_updates'],
        'published_journal_optimizer_updates_at_stop':closure['published_journal_optimizer_updates'],
        'published_unsaved_tail_count':closure['unsaved_positive_tail_count'],
        'optimizer_vs_journal_frontier':closure['optimizer_vs_journal_frontier'],
        'saved_prefix_only_after_restore':pointer['journal_prefix'],'numerical_resource_science_cadence_old_values_exact':True}
    write(F/'final_configuration_inverse_and_future7_runtime_bindings_v1.json',facts)
    handoff={'kind':'ASR18_final_INACTIVE_source29_approved_lookahead_saved_checkpoint_handoff_v1',
        'driver_source':driver,'wrapper_source':wrapper,'CPU_source':pin(S/'fastmss_resident_feature_stream_v11.py'),
        'helper_source':pin(F/'asr18_owned_packet_credit_v2.py'),
        'lookahead_transition_validator':pin(S/'asr18_approved_lookahead_checkpoint_transition_v3.py'),
        'source_configuration':final_pin,'lookahead_resume':closure_pin,'initial_checkpoint':pointer_pin,
        'latest_failure':pin(args[5]),'initial_owner_invocation':21,'output':args[9],'branch':str(B),
        'actual_CPU_review':final['resident_RAM_packet_credit_policy']['actual_CPU_control_review'],
        'packet_credit_policy':final['resident_RAM_packet_credit_policy'],
        'native_update_commit_pressure_policy':final['native_update_commit_pressure_policy'],
        'human_approval':final['resident_RAM_packet_credit_policy']['human_approval'],
        'configuration_and_runtime_bindings':pin(F/'final_configuration_inverse_and_future7_runtime_bindings_v1.json'),
        'typed_fixture_source':pin(F/'verify_source29_actual_checkpoint_transition_entire_selectors_v1.py'),
        'typed_external_launcher_source':pin(F/'launch_one_reviewed_source29_selector_metadata_CPU_v2.py'),
        'future_wrapper_arguments_NOT_EXECUTABLE_WITHOUT_ROOT_FINAL_BINDING':args,
        'full_native_goal':401072,'administrative_stop':None,'production_launch_admitted':False,
        'new_journal':'commits_000021.jsonl','genuine_tail_redo_required':True,
        'actual_postlaunch_speed_observation':'Fixed contiguous400 poststartup updates including all natural save gaps; full original timing rows/native/audio/pipeline/gate counters; no separate benchmark or finite stop'}
    write(F/'final_source29_actual_checkpoint_transition_execution_handoff_v1.json',handoff)
    print(json.dumps({'configuration':final_pin,'handoff':pin(F/'final_source29_actual_checkpoint_transition_execution_handoff_v1.json'),
        'runtime_bindings':pin(F/'final_configuration_inverse_and_future7_runtime_bindings_v1.json'),'no_production_launch':True}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--actual-closure',required=True);main(p.parse_args())
