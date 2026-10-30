"""Source-only exact failed-update1 continuation seal; no ML/audio import."""
import argparse,copy,hashlib,json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
E=Path(r'E:\gpu_training_scratchpad');A=E/'analysis/ASR18_memory_diagnosis_v1';R=P/'work/research'
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_production_state_v1 import require
from asr18_streamed_persistence_fingerprint_v1 import normalize_fingerprint

def pin(path):
    path=Path(path).resolve();require(path.is_relative_to(P) or path.is_relative_to(E),'P/E only');st=path.stat();h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(1024**2):h.update(b)
    require((st.st_size,st.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns),'Stable source')
    return {'path':str(path),'bytes':st.st_size,'sha256':h.hexdigest()}

def main():
    args=argparse.ArgumentParser();args.add_argument('--observed-tool-chunk',required=True);args.add_argument('--observed-session',type=int,required=True);a=args.parse_args()
    require(a.observed_tool_chunk=='7d0aae' and a.observed_session==86866,'Exact operator-observed consumed TOOL1/session; never inferred from files')
    require(os.environ['CUDA_VISIBLE_DEVICES']=='-1' and read_json(E/'storage_config.json')['storage_root']==str(E),'Metadata only/E policy')
    old=A/'fixed_TR_replay207_actual_03';train=old/'train';source_config=A/'fixed_TR_replay207_configuration_02/configuration_pending_ROOT_execution_review.json'
    context=read_json(train/'owned_family_context.json');guard=read_json(train/'guard_closure.json');parent_guard=read_json(old/'guard_closure.json')
    import psutil
    ids={v['pid']:v for v in [context['observer'],context['native'],context['launcher'],*context['all_live_shims']]}
    for v in ids.values():
        try:live=abs(psutil.Process(v['pid']).create_time()-v['create_time'])<.001
        except psutil.NoSuchProcess:live=False
        require(not live,'Every actual source identity dead')
    require(guard['pid']==context['native']['pid'] and parent_guard['pid']==context['observer']['pid'],'Exact source guard identities')
    require(all(g['actual_exit_code']==1 and g['monitor_clean'] and g['monitor_joined'] and g['error'] is None and g['samples']>=3 for g in (guard,parent_guard)),'Actual clean monitored failure1, not success')
    require('Observed native high-water plus unchanged750MB cannot exceed4.9GB wholeworker' in (train/'worker.log').read_text(),'Exact update2 forecast stop')
    progress=read_json(train/'native_progress.json');require(progress['actual_updates']==1 and len(progress['history'])==1,'Exactly one actual committed optimizer update')
    stages=[json.loads(v) for v in (train/'stages.jsonl').read_text().splitlines()]
    post=next(v for v in stages if v['phase']=='after_checkpoint_write_and_persistence_1');peak=post['private_high_water']['lifetime_private_peak_bytes']
    require(peak==4_214_124_544,'Exact actual postwrite OS peak')
    source_pre=normalize_fingerprint(read_json(train/'PRE_save_1.json'));require(source_pre['state']['updates']==1,'Canonical actual PRE1 fingerprint')
    checkpoint=read_json(train/'after_1.state.json');require(pin(checkpoint['payload']['path'])==checkpoint['payload'],'Actual completed full payload signature')
    failure={'kind':'ASR18_operator_observed_fixed20_after_update1_actual_TOOL1_v1','owner':'/root/memory_diagnosis','owned_session':a.observed_session,
        'actual_tool_chunk_id':a.observed_tool_chunk,'actual_tool_exit_code':1,'actual_optimizer_updates':1,'actual_sampler_draws':0,
        'owned_identities':list(ids.values()),'all_owned_identities_dead':True,'native_guard':pin(train/'guard_closure.json'),
        'parent_guard':pin(old/'guard_closure.json'),'source_context':pin(train/'owned_family_context.json'),'native_progress':pin(train/'native_progress.json'),
        'checkpoint1':pin(train/'after_1.state.json'),'payload1':checkpoint['payload'],'PRE1':pin(train/'PRE_save_1.json'),'postwrite_OS_lifetime_peak_bytes':peak,
        'worker_log':pin(train/'worker.log'),'source_configuration':pin(source_config),'stop_before_update2':True,'recovery_not_yet_claimed':True,'production_enabled':False}
    failure_path=old/'actual_owned_CLOSED1_summary_v1.json';require(failure_path.exists() and read_json(failure_path)==failure,'Preserve exact alreadysealed actual failure evidence')
    folder=A/'fixed_TR_replay207_resume19_configuration_02';folder.mkdir(exist_ok=False)
    c=copy.deepcopy(read_json(source_config))
    for binding in c['source_bindings'].values():require(pin(binding['path'])==binding,'Original frozen source current')
    c['source_bindings'].update(harness=pin(R/'run_ASR18_fixed_TR_replay207_resume19_v2.py'),configuration_builder=pin(__file__),fixed_operations=pin(R/'asr18_fixed_TR_replay207_resume19_operations_v2.py'))
    c['source_bindings']['prior_harness']=pin(R/'run_ASR18_fixed_TR_replay207_memory_v2.py');c['source_bindings']['prior_operations']=pin(R/'asr18_fixed_TR_replay207_operations_v2.py')
    c.update(kind='ASR18_fixed20_resume_checkpoint1_remaining19_diagnostic_v1',new_optimizer_updates=19,native_updates=20)
    prefix={'path':str(train/'unconsumed_source_journal.jsonl'),'offset':0,'sha256':hashlib.sha256(b'').hexdigest()}
    require(pin(prefix['path'])['bytes']==0,'Original authoritative source prefix remains empty/unconsumed')
    c['continuation']={'failed_actual_tool1':pin(failure_path),'checkpoint1':pin(train/'after_1.state.json'),'PRE_save1':pin(train/'PRE_save_1.json'),
        'native_progress':pin(train/'native_progress.json'),'source_configuration':pin(source_config),'source_guard':pin(train/'guard_closure.json'),
        'source_parent_guard':pin(old/'guard_closure.json'),'source_stages':pin(train/'stages.jsonl'),'prepared':pin(old/'prepare/prepared.json'),
        'prepared_actual_process0':pin(old/'prepare/actual_process0.json'),'source_journal_prefix':prefix,'postwrite_source_OS_lifetime_peak_bytes':peak}
    c['resource_policy'].update(training_private_hard_cap_bytes=5_624_813_440,training_whole_family_hard_cap_bytes=5_816_813_440,
        unchanged_variation_allowance_bytes=750_000_000,prior_genuine_native20_peak_bytes=4_682_813_440,replay_runtime_extension_bytes=128_000_000,
        checkpoint_write_increment_bytes=64_000_000,initial_ancestor_restore_increment_bytes=350_000_000,initial_ancestor_restore_worker_cap_bytes=3_350_000_000)
    c['resource_policy']['postwrite_and_native_running_peak_plus_750M_plus_64M_must_fit']=True
    c['evidence']['failed_positive_update1_tool1']=pin(failure_path)
    for v in c['source_bindings'].values():compile(Path(v['path']).read_text(encoding='utf-8-sig'),v['path'],'exec')
    path=folder/'configuration_pending_ROOT_execution_review.json';write_json(path,c)
    proposal={'kind':'ASR18_exact_checkpoint1_resume19_resource_proposal_v1','accepted':False,'configuration':pin(path),'source_bindings':c['source_bindings'],
        'resource_policy':c['resource_policy'],'actual_failed_source':pin(failure_path),'continuation':c['continuation'],
        'worker_forecast_arithmetic':{'prior_genuine_native20_peak':4_682_813_440,'unchanged_variation':750_000_000,'retained_replay_extension':128_000_000,'persistence':64_000_000,'total':5_624_813_440},
        'whole_family_total_bytes':5_816_813_440,'source_workload_coverage':'Older true native20 T1115/U419, replayU75 empirical peak retained without CNN credit; selected fixed T208/U95 and replayT183/U67 remain bounded by original fixed input contracts.',
        'no_old_failure_relabel_or_source_restart':True,'history_update1_and_original_journal_ancestor_preserved':True,
        'restore_before_any_update2':'Fresh350MB/entire3.35GB startup; exact canonical PRE1 all207/3Adam/all4RNG/sampler and copied original journal prefix; adopt restored branch as continuing journal.',
        'final_recovery':'Only after genuine resumed native0/dead; enumerate old1 plus new10/20 and independently fresh350MB before each exact restore. Total20 includes old1+new19.',
        'CPU_audio_frontend_repeated':False,'CNN_savings_subtracted':False,'original8GB_CUDA_allocator_unchanged':True,'production_enabled':False}
    write_json(folder/'resource_proposal.json',proposal)
    handoff={'configuration':pin(path),'resource_proposal':pin(folder/'resource_proposal.json'),'source_bindings':c['source_bindings'],'source_actual_tool1':pin(failure_path),'execution_admitted':False,'production_enabled':False}
    write_json(folder/'handoff.json',handoff)
    print(json.dumps({'config':pin(path),'proposal':pin(folder/'resource_proposal.json'),'handoff':pin(folder/'handoff.json'),'source_count':len(c['source_bindings'])}))

if __name__=='__main__':main()
