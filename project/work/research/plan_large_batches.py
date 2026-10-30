"""CPU-only tensor-size/padding estimates from previously audited TRAIN lengths."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'outputs/parakeet_turkish'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def quantile(values, fraction):
    values = sorted(values)
    return values[round(fraction*(len(values)-1))]


def cell_cost(row):
    return row['encoder_frames']*(row['target_tokens']+1)


def bucket(row):
    return (int(math.log2(row['encoder_frames'])), int(math.log2(row['target_tokens']+1)))


def describe(pool):
    return {'clips':len(pool), 'hours':math.fsum(row['duration'] for row in pool)/3600,
            'encoder_frames':{str(q):quantile([r['encoder_frames'] for r in pool],q) for q in (0,.25,.5,.9,.95,1)},
            'target_tokens':{str(q):quantile([r['target_tokens'] for r in pool],q) for q in (0,.25,.5,.9,.95,1)}}


def tensor_sizes(rows, chunk=16, vocab=8199, durations=5, joint_hidden=640, encoder_hidden=1024):
    batch=len(rows);frames=max(r['encoder_frames'] for r in rows);targets=max(r['target_tokens'] for r in rows)
    decoder=targets+1;time_chunk=min(chunk,frames);cells=batch*time_chunk*decoder
    padded=batch*frames*decoder;valid=sum(cell_cost(r) for r in rows)
    # FP32 tokens.float() is a view of FP32 logits; BF16 tokens.float() allocates.
    bf16_logits=cells*(vocab+durations)*2;fp32_logits=cells*(vocab+durations)*4
    token_copy=cells*vocab*4;normalizer=cells*4
    bf16_joint_hidden=cells*joint_hidden*2;fp32_joint_hidden=cells*joint_hidden*4
    retained=4*batch*frames*(decoder+(decoder-1)+decoder*durations)
    return {'physical_batch':batch,'max_encoder_frames':frames,'max_target_tokens':targets,
            'padded_lattice_cells':padded,'valid_lattice_cells':valid,'lattice_padding_factor':padded/valid,
            'frame_padding_factor':batch*frames/sum(r['encoder_frames'] for r in rows),
            'chunk_time_frames':time_chunk,'chunk_cells':cells,
            'bf16_joint_logits_bytes':bf16_logits,'bf16_fp32_token_copy_bytes':token_copy,
            'bf16_minimum_joint_chunk_bytes':bf16_logits+token_copy+normalizer+bf16_joint_hidden,
            'fp32_minimum_joint_chunk_bytes':fp32_logits+normalizer+fp32_joint_hidden,
            'retained_reduced_probability_bytes':retained,'dp_alpha_bytes':4*padded,
            'original_teacher_padded_features_bytes':4*batch*frames*encoder_hidden,
            'checkpoint_encoder_input_bytes_if_fp32':24*4*batch*frames*encoder_hidden,
            'math_attention_one_score_matrix_bytes_if_fp32':4*batch*8*frames*frames,
            'subsampling_first_conv_output_bytes_if_fp32_approx':4*batch*256*64*math.ceil(8*frames/2),
            'one_dp_index_plan_bytes_upper_bound':233*frames*decoder+64,
            'four_cached_dp_index_plans_bytes_upper_bound':4*(233*frames*decoder+64)}


def configuration(tr, replay, batch_size):
    effective=len(tr)+len(replay)
    if batch_size==effective:
        micros=[tr+replay]
        composition='mixed TR/replay microbatch'
    else:
        micros=[stream[start:start+batch_size] for stream in (tr,replay)
                for start in range(0,len(stream),batch_size)]
        composition='separate TR and replay microbatches, homogeneous length bucket'
    assert all(len(m)==batch_size for m in micros)
    lengths=[tensor_sizes(m) for m in micros]
    valid=sum(r['valid_lattice_cells'] for r in lengths);padded=sum(r['padded_lattice_cells'] for r in lengths)
    return {'physical_batch_size':batch_size,'accumulation':len(micros),'effective_optimizer_batch':effective,
            'tr_clips_per_update':len(tr),'replay_clips_per_update':len(replay),'composition':composition,
            'optimizer_batch_change_vs_current_loop':effective!=4,
            'physical_batch_only_change_within_this_comparison_family':True,
            'aggregate_lattice_padding_factor':padded/valid,
            'max_bf16_minimum_joint_chunk_gib':max(r['bf16_minimum_joint_chunk_bytes'] for r in lengths)/2**30,
            'max_fp32_minimum_joint_chunk_gib':max(r['fp32_minimum_joint_chunk_bytes'] for r in lengths)/2**30,
            'max_retained_probability_mib':max(r['retained_reduced_probability_bytes'] for r in lengths)/2**20,
            'max_math_attention_score_mib_if_fp32':max(r['math_attention_one_score_matrix_bytes_if_fp32'] for r in lengths)/2**20,
            'microbatch_tensor_estimates':lengths}


def main():
    audit_path=OUTPUT/'lora_training_data_verification.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    assert audit['passed'] is True
    streams=audit['streams']
    tr_cv=streams['turkish_commonvoice']['accepted_source_ids_and_labels']
    tr_fleurs=streams['turkish_fleurs_train']['accepted_source_ids_and_labels']
    replay=streams['retention_fleurs_train']['accepted_source_ids_and_labels']
    all_rows=tr_cv+tr_fleurs+replay
    assert len({r['source_id'] for r in all_rows})==len(all_rows)==3719
    assert all(isinstance(r['encoder_frames'],int) and r['encoder_frames']>0 and 0<r['target_tokens']<=180 for r in all_rows)
    bins={}
    for name,pool in (('tr',tr_cv),('replay',replay)):
        current=defaultdict(list)
        for row in pool:current[bucket(row)].append(row)
        bins[name]=current
    common=[key for key in bins['tr'] if len(bins['tr'][key])>=16 and len(bins['replay'].get(key,[]))>=16]
    common.sort(key=lambda key:2**sum(key))
    assert common
    median_tr=sorted(tr_cv,key=lambda r:(cell_cost(r),r['source_id']))
    median_replay=sorted(replay,key=lambda r:(cell_cost(r),r['source_id']))
    cases=[('typical_stream_medians',None,median_tr[(len(median_tr)-16)//2:(len(median_tr)-16)//2+16],
            median_replay[(len(median_replay)-16)//2:(len(median_replay)-16)//2+16])]
    for index,key in enumerate(common):
        name='short_homogeneous' if index==0 else 'long_homogeneous' if index==len(common)-1 else f'middle_homogeneous_{index}'
        tr=sorted(bins['tr'][key],key=lambda r:(cell_cost(r),r['source_id']))
        rp=sorted(bins['replay'][key],key=lambda r:(cell_cost(r),r['source_id']))
        tr=tr[(len(tr)-16)//2:(len(tr)-16)//2+16];rp=rp[(len(rp)-16)//2:(len(rp)-16)//2+16]
        cases.append((name,key,tr,rp))
    cases.append(('long_train_stress',None,sorted(tr_fleurs,key=cell_cost)[-16:],sorted(replay,key=cell_cost)[-16:]))
    results=[]
    for name,key,tr,rp in cases:
        fixed16=tr[:8]+rp[:8];fixed32=tr+rp
        results.append({'name':name,'log2_length_bucket':key,
                        'fixed_effective16_clip_metadata':fixed16,
                        'fixed_effective16_source_ids_sha256':hashlib.sha256('\n'.join(r['source_id'] for r in fixed16).encode()).hexdigest(),
                        'effective16_same_clips_configurations':[configuration(tr[:8],rp[:8],b) for b in (1,2,4,8,16)],
                        'effective32_clip_metadata':fixed32,
                        'effective32_is_separate_optimizer_batch_experiment':True,
                        'effective32_controls':[configuration(tr,rp,b) for b in (1,2,4,8,16)]})
    plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'CPU size arithmetic on independently verified TRAIN metadata; no GPU, model load, raw test predictions or timings',
          'status':'prepared; future physical-batch GPU profiling not launched',
          'provenance':{'metadata_audit_path':str(audit_path),'metadata_audit_sha256':sha(audit_path),'script_path':str(Path(__file__)),'script_sha256':sha(Path(__file__)),
                        'base_revision':audit['base_revision'],'tokenizer_semantic_sha256':audit['extended_tokenizer_semantic_sha256'],
                        'code_sha256':{n:sha(OUTPUT/n) for n in ('run_experiment_loop.py','train_lora_batched.py','batched_lora.py','efficient_tdt.py','efficient_tdt_v2.py')}},
          'train_metadata_summary':{'turkish_commonvoice':describe(tr_cv),'turkish_fleurs_train':describe(tr_fleurs),'retention_fleurs_train':describe(replay)},
          'expanded_pool_limit':'Uses already audited pilot TRAIN lengths. Expanded25,628-clip cache is still being built; rerun on the completed training_pool for population estimates. CV cases reflect the dominant expanded source; FLEURS long stress is deliberately oversampled.',
          'memory_assumptions':{'extended_token_vocabulary':8199,'duration_rows':5,'joint_hidden':640,'encoder_hidden':1024,'encoder_layers':24,'attention_heads':8,'joint_time_chunk':16,
              'bf16_storage':'2-byte logits/hidden plus separate 4-byte token normalization copy',
              'fp32_storage':'4-byte logits/hidden; tokens.float() aliases the logits slice',
              'limits':'Listed tensor storage is a lower bound for chunk core and illustrative encoder/DP sizes, not a predicted CUDA peak. Backward temporaries, checkpoint recomputation, allocator reservation, cuDNN/attention workspaces, model, teacher, gradients and Adam states must be measured.',
              'capacity':'RTX 5060 Ti has 16 GB nominal VRAM. Existing B1 measured peak around 3.05 GB is evidence of available capacity; do not multiply the persistent model/Adam allocation by physical batch.'},
          'configured_limit_tensor_arithmetic':{'scope':'Theoretical arithmetic only, not a sampled recording or an observed peak. Approximately 376 encoder frames at 30 seconds; exact frontend/cache lengths remain authoritative.',
              'bf16_B16_chunk_gib':tensor_sizes([{'encoder_frames':376,'target_tokens':180}]*16)['bf16_minimum_joint_chunk_bytes']/2**30,
              'fp32_B16_chunk_gib':tensor_sizes([{'encoder_frames':376,'target_tokens':180}]*16)['fp32_minimum_joint_chunk_bytes']/2**30},
          'matched_comparison_families':{'effective16':'Same8TR+8replay recordings, checkpoint, optimizer state, per-clip mean objective, precision/attention backend; B1x16/B2x8/B4x4/B8x2/B16x1. B16x1 is mixed; smaller batches separate streams.',
              'effective32':'Separate16 TR + 16 replay optimizer experiment. B16x2 is homogeneous by stream and doubles the optimizer batch versus effective16. B1x32/B2x16/B4x8/B8x4 controls are provided. All planned effective16/32 settings also change current loop effective4.'},
          'numerical_findings':{'fp32_default_sdpa':'Shape-dependent B2 encoder drift found in the full 24-layer model.',
              'fp32_forced_math':'New diagnostic reports final B2 encoder relative L2 around 1.3e-6 to 1.5e-6, equal-length TR clips bit-exact; full objective/gradient verification still underway.',
              'bf16':'Convolution/projection batch-shape rounding persists even with MATH/cuDNNdisabled: ~0.29% subsampling and 1-1.6% final encoder relativeL2. Report and evaluate these numerical differences separately; do not claim unchanged gradients from available VRAM.',
              'timing':'The diagnostic overlaps active cache only for correctness. Those runs provide no valid throughput estimate.'},
          'independent_source_review':{'reviewer':'benchmark_design', 'joint_chunk_core_formula_confirmed':True,
              'checks':'Matches reduced_joint_probs: 8,204 output rows, separate BF16-to-FP32 token copy for 8,199 token rows, 640 joint hidden width; FP32 tokens.float() aliases logits.',
              'limits':'Independent source review confirms arithmetic only; actual peak memory and larger-batch throughput remain unmeasured.'},
          'future_profile_rules':['First validate full objective, teacher losses, masking, per-parameter gradients and actual Adam/new-token-row updates against singleton FP32+MATH on identical clips.',
              'For BF16, report per-layer/encoder/gradient drift against BF16B1 and FP32 reference, finite gradients and an explicitly reviewed numerical gate. Keep precision/backend identical in each speed comparison.',
              'Give the GPU exclusively to each timing run: no active teacher cache, training, development/final inference, other profiler or CUDA application compute. Synchronize timing, warm up, then measure matched 20 updates.',
              'Record clips/s, actual audio-minutes/wall-minute, per-update median/p90, allocated/reserved peakVRAM, data/frontend/H2D/forward/backward/optimizer timing boundary, padding, precision/backend and full input/code/checkpoint hashes.',
              'Test median and long train cases; stop configurations on OOM/NaN or memory pressure, retain failures, and choose measured throughput rather than largest batch.',
              'Adopt a qualified setting only in a new explicit DEV-selected training trial; current frozen 12-hour B1 loop continues unchanged. Retention/Turkish gates remain mandatory; held-out predictions never tune batch or LR.'],
          'optional_finite_profile_queue':{'proposed_budget_minutes':45,'launched':False,'scheduled':False,
              'start_condition':'Parent coordinates a free GPU after current loop, finalizer, strong TEST and audit finish; budget must be explicitly assigned to this finite profiler. Check terminal status and child process exit rather than only an idle GPU snapshot.',
              'termination':'One finite matrix, bounded runtime and memory failures; no recurring automation or silent extension of current 12-hour window.'},
          'cases':results}
    target=OUTPUT/'large_batch_plan.json';target.write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'path':str(target),'cases':len(results),'common_length_buckets':len(common),
        'effective16_max_minimum_joint_chunk_gib_bf16':max(c['max_bf16_minimum_joint_chunk_gib'] for case in results for c in case['effective16_same_clips_configurations']),
        'effective16_max_minimum_joint_chunk_gib_fp32':max(c['max_fp32_minimum_joint_chunk_gib'] for case in results for c in case['effective16_same_clips_configurations'])}))


if __name__=='__main__':main()
