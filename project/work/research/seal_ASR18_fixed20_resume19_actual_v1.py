"""Compact source-bound observed actual0 receipt, no numerical imports."""
import argparse,hashlib,json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
E=Path(r'E:\gpu_training_scratchpad');A=E/'analysis/ASR18_memory_diagnosis_v1'
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_production_state_v1 import require
from asr18_streamed_persistence_fingerprint_v1 import normalize_fingerprint
import importlib.util
spec=importlib.util.spec_from_file_location('guard',P/'work/research/run_ASR18_frozen_CNN_lifetime_pair_v8.py');H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
pin=H.record

def main():
    p=argparse.ArgumentParser();p.add_argument('--actual-tool-chunk',required=True);p.add_argument('--owned-session',type=int,required=True);args=p.parse_args()
    require(args.actual_tool_chunk=='dc905b' and args.owned_session==59114,'Actual own consumed outer TOOL0 only; not inferred')
    run=A/'fixed_TR_replay207_resume19_actual_01';overall=read_json(run/'overall_result.json');train=read_json(run/'train/training_result.json');recovery=read_json(run/'recover/recovery_result.json')
    require(overall['passed'] is True and overall['actual_updates']==20 and overall['actual_new_optimizer_updates']==19,'Exact old1+new19 total20, not20sameprocess')
    require(train['passed'] and train['actual_optimizer_updates']==20 and train['actual_new_optimizer_updates']==19 and [v['update'] for v in train['history']]==list(range(1,21)),'Contiguous original1/new2..20 history')
    require(recovery['passed'] and [v['update'] for v in recovery['recovered']]==[1,10,20] and all(v['all207_threeAdam_all4RNG_PREsave_exact'] is True for v in recovery['recovered']),'Every saved capture freshly exactly recovered')
    observations={};phases={};captures=[]
    for phase in ('train','recover'):
        actual=read_json(run/phase/'actual_process0.json');guard=read_json(run/phase/'guard_closure.json')
        require(actual['actual_Popen_wait_code']==0 and actual['all_captured_identities_dead'] and not any(H.alive(v) for v in actual['captured_identities']),'Genuine actual0 and all captured identities dead')
        require(pin(actual['report']['path'])==actual['report'] and pin(actual['guard']['path'])==actual['guard'],'Exact report/guard bindings')
        require(guard['actual_exit_code']==0 and guard['monitor_clean'] and guard['monitor_joined'] and guard['error'] is None and guard['pid']==actual['native']['pid'],'Own clean guard0')
        stages=[json.loads(v) for v in (run/phase/'stages.jsonl').read_text().splitlines()]
        observations[phase]={'actual_process0':pin(run/phase/'actual_process0.json'),'guard0':pin(run/phase/'guard_closure.json'),'native':actual['native'],
            'sampled_private_peak_bytes':guard['peak_private_bytes'],'OS_lifetime_private_peak_bytes':max(v['private_high_water']['lifetime_private_peak_bytes'] for v in stages),
            'CUDA_peak_allocated_bytes':max(v['cuda']['peak_allocated'] for v in stages),'global_peak_fraction':guard['system_commit_peak_fraction'],'samples':guard['samples']}
        phases[phase]=[{'phase':v['phase'],'current_private_bytes':v['private_high_water']['current_private_bytes'],'OS_lifetime_private_peak_bytes':v['private_high_water']['lifetime_private_peak_bytes'],'CUDA_peak_allocated_bytes':v['cuda']['peak_allocated']} for v in stages if v['phase'].startswith(('actual_','after_actual_','before_checkpoint','after_checkpoint','before_restore','after_exact_restore','after_exact_checkpoint1','before_actual_checkpoint1'))]
    for item in train['captures']:
        require(pin(item['checkpoint']['path'])==item['checkpoint'] and pin(item['PRE_save_fingerprint']['path'])==item['PRE_save_fingerprint'],'Exact independent checkpoint/PRE bindings')
        document=read_json(item['checkpoint']['path']);require(pin(document['payload']['path'])==document['payload'],'Exact complete saved payload signature')
        pre=normalize_fingerprint(read_json(item['PRE_save_fingerprint']['path']));require(pre['state']['updates']==item['update'] and pre['state']['clocks']==[64250+item['update'],2000+item['update'],62250+item['update']],'Actual canonical PRE full optimizer frontier')
        captures.append({'update':item['update'],'checkpoint':item['checkpoint'],'payload':document['payload'],'PRE_save':item['PRE_save_fingerprint'],'actual_clocks':pre['state']['clocks'],'identity':item['identity'],'all207_threeAdam_all4RNG_fresh_recovery_exact':True})
    require([v['update'] for v in captures]==[1,10,20] and captures[0]['identity']!=captures[1]['identity']==captures[2]['identity'],'Old1 and new10/20 distinct exact lineage identities')
    outer=read_json(run/'guard_closure.json');require(outer['actual_exit_code']==0 and outer['monitor_clean'] and outer['monitor_joined'],'Actual clean outer0')
    summary={'kind':'ASR18_fixed20_checkpoint1_plus_resume19_actual_CLOSED0_summary_v1','owner':'/root/memory_diagnosis','owned_session':args.owned_session,
        'actual_tool_exit_code':0,'actual_tool_chunk_id':args.actual_tool_chunk,'overall':pin(run/'overall_result.json'),'actual_new_updates':19,'actual_prior_partial_updates':1,
        'actual_total_contiguous_updates':20,'same_process_new_updates':19,'twenty_same_process_updates_claimed':False,'final_original_source_clocks':[64270,2020,62270],
        'old_actual_failure1_preserved':overall['source_update1'],'initial_ancestor_exact_recovery':pin(run/'train/initial_ancestor_recovery.json'),
        'training_result':pin(run/'train/training_result.json'),'recovery_result':pin(run/'recover/recovery_result.json'),'captures':captures,'observations':observations,'phase_observations':phases,
        'outer_guard0':pin(run/'guard_closure.json'),'all_source_samplers_and_authoritative_draws_unconsumed':True,'sampler_draws_consumed':0,
        'scope':'Fixed8 original rows:2TR(T13/U6,T208/U95)+first genuine DE/EN/ES/FR/IT/PT replay; physical1 serial14TR+2RP effective16 perupdate. No wholecorpus/hardT1991/replaymax/quality/retention/production proof.',
        'production_enabled':False,'diagnostic_checkpoint_promotion':False}
    target=run/'actual_fixed20_continuation_summary_v1.json';require(not target.exists(),'Preserve immutable summary');write_json(target,summary)
    print(json.dumps({'summary':pin(target),'observations':observations,'final_clocks':summary['final_original_source_clocks']}))

if __name__=='__main__':main()
