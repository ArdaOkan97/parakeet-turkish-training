"""Readonly bounded exact paired-profile comparison; no ML/tensor imports."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
E=Path(r'E:\gpu_training_scratchpad')
from bpe_broad_production_state_v1 import digest,require
from asr18_streamed_persistence_fingerprint_v1 import normalize_fingerprint


def read(path):
    path=Path(path).resolve();require(path.is_relative_to(E) and 0<path.stat().st_size<=16*1024**2,'Bounded E metadata only')
    return json.loads(path.read_text(encoding='utf-8'))


def pin(path):
    path=Path(path).resolve();before=path.stat();h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(4*1024**2):h.update(block)
    require(before.st_size==path.stat().st_size and before.st_mtime_ns==path.stat().st_mtime_ns,'Artifact changed during exact hash')
    return {'path':str(path),'bytes':before.st_size,'sha256':h.hexdigest()}


def bound(value):
    require(pin(value['path'])==value,'Exact source-bound pair artifact');return read(value['path'])


def closed(folder):
    overall=read(folder/'overall_result.json');outer=read(folder/'guard_closure.json')
    require(overall['passed'] and outer['actual_exit_code']==0 and outer['monitor_clean'] and outer['monitor_joined'],'Genuine whole parent normal0')
    values=[]
    for key in ('actual_native_process0','actual_fresh_recovery_process0'):
        proof=bound(overall[key]);require(proof['actual_Popen_wait_code']==0 and proof['all_captured_identities_dead'],'Actual native/recovery handle0/death')
        import psutil
        for old in proof['captured_identities']:
            try:require(psutil.Process(old['pid']).create_time()!=old['create_time'],'Pair producer still alive')
            except psutil.NoSuchProcess:pass
        guard=bound(proof['guard']);require(guard['actual_exit_code']==0 and guard['monitor_clean'] and guard['monitor_joined'],'Exact clean joined native/recovery guards')
        values.append(bound(proof['report']))
    require(values[0]['actual_optimizer_updates'] in (1,4) and values[1]['all207_threeAdam_all4RNG_sampler_nextdraw_exact'] is True,'Each pair restored all scientific state')
    return overall,values[0]


def compare(baseline,candidate,output):
    b,bnative=closed(baseline);c,cnative=closed(candidate)
    N=bnative['actual_optimizer_updates'];require(N==cnative['actual_optimizer_updates'],'Same declared original-prefix update count')
    bi=bound(bnative['initial_state']);ci=bound(cnative['initial_state'])
    checks={}
    def exact(name,left,right):
        checks[name]=digest(left)==digest(right);require(checks[name],'Strict paired inequality: '+name)
    exact('same_new_owned_immutable_CPU_window',bound(b['shared_features']),bound(c['shared_features']))
    exact('all207_threeAdam_all4RNG_initial',normalize_fingerprint(bi['full_state']),normalize_fingerprint(ci['full_state']))
    exact('actual_initial_all4_RNG_bytes',bi['all4_rng'],ci['all4_rng']);exact('unchanged_teacher_initial',bi['teacher'],ci['teacher'])
    exact('initial_original_sampler',bi['original_schedule_sha256'],ci['original_schedule_sha256'])
    bp=normalize_fingerprint(bound(bnative['PRE_save_fingerprint']));cp=normalize_fingerprint(bound(cnative['PRE_save_fingerprint']))
    exact('all207_threeAdam_all4RNG_final',bp,cp)
    exact('full_original_final_sampler',bnative['sampler_sha256'],cnative['sampler_sha256'])
    exact('exact_nextdraw_packet',bound(bnative['next_draw']),bound(cnative['next_draw']))
    bm=bound(bnative['checkpoint']);cm=bound(cnative['checkpoint'])
    exact('all845_tensor_dtype_shape_offset_full_byte_hashes',bm['tensors'],cm['tensors'])
    require(len(bm['tensors'])==845,'Complete weights/moments/clocks/Torch/CUDA checkpoint tensor table')
    bpayload=pin(bm['payload']['path']);cpayload=pin(cm['payload']['path'])
    require(bpayload==bm['payload'] and cpayload==cm['payload'],'Actual full payload file bytes hashed with bounded4MiB buffer')
    exact('complete_payload_bytes_sha256',{k:bpayload[k] for k in ('bytes','sha256')},{k:cpayload[k] for k in ('bytes','sha256')})
    journals=[]
    for folder,native in ((baseline,bnative),(candidate,cnative)):
        path=Path(native['journal_prefix']['path']);require(path.stat().st_size==native['journal_prefix']['offset'],'No unknown journal suffix')
        require(pin(path)['sha256']==native['journal_prefix']['sha256'],'Actual entire committed journal prefix bytes')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        require(len(rows)==N,'All declared original native and sampler transactions')
        for row in rows:row['native'].pop('graph_boundary_memory_observations')
        journals.append(rows)
    exact('all_chronological_scientific_journal_fields',*journals)
    profiles=[]
    for folder,native in ((baseline,bnative),(candidate,cnative)):
        profile=bound(native['profile']);profiles.append(profile)
    if all('all207_gradient_fingerprints' in v for v in profiles):
        exact('all207_pre_post_original_clip_gradients_all_declared_updates',profiles[0]['all207_gradient_fingerprints'],profiles[1]['all207_gradient_fingerprints'])
        for update in range(1,N+1):
            left=read(baseline/'train'/f'profile_update_{update}.json');right=read(candidate/'train'/f'profile_update_{update}.json')
            l=[x for x in left['full_row_output_fingerprints'] if x['label'].startswith('row_')]
            r=[x for x in right['full_row_output_fingerprints'] if x['label'].startswith('row_')]
            require(len(l)==len(r)==16,'All16 rows per declared update, no samples')
            exact('full_encoder_mask_objective_outputs_update_'+str(update),l,r)
    else:
        require(any(v.get('profile_enabled') is False for v in profiles),'Only reviewed matched profiler-off final timing may omit intrusive row/gradient observers')
        require(any(v.get('kind')=='ASR18_matched_profiler_off_native_wall_observation_v1' and n.get('grouped_oracle_observers_enabled') is False
            for v,n in zip(profiles,(bnative,cnative))),'Only explicit reviewed no-oracle timing may omit grouped row/gradient readbacks')
    value={'kind':'ASR18_actual_exact_native_throughput_pair_comparison_v1','passed':True,'source':pin(__file__),
        'baseline_overall':pin(baseline/'overall_result.json'),'candidate_overall':pin(candidate/'overall_result.json'),
        'checks':checks,'actual_native_updates_per_arm':N,'original_rows_consumed_per_arm':16*N,'Transformer_batched_trajectory_not_serial_oracle':True,
        'full845_tensor_payload_bytes':bpayload['bytes'],'original_initial_all4_RNG_snapshot_shared_not_arbitrary_seed':True,
        'normalized_fields':['Only journal.native.graph_boundary_memory_observations (explicit performance/resource/GC observations)'],
        'weights_Adam_RNG_sampler_loss_norm_payload_or_metadata_policy_exclusions':False,'tolerance_widened':False,
        'all207_gradient_and_all_declared_row_output_observers_present':all('all207_gradient_fingerprints' in v for v in profiles),
        'production_enabled':False}
    output=output.resolve();require(output.is_relative_to(E) and not output.exists(),'One fresh E comparison receipt')
    output.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');print(json.dumps({'passed':True,'receipt':pin(output)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();compare(Path(a.baseline).resolve(),Path(a.candidate).resolve(),Path(a.output))
