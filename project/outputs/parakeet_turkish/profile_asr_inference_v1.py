"""Bounded selected1871 inference throughput on the same sixteen FLEURS DEV clips.

CPU preparation never initializes CUDA. Actual inference requires the parent's
solo GPU authorization; it never trains, samples TEST, changes original sources,
or writes an existing output. Model loading and warmup are excluded from timing.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import time

import soundfile as sf
import torch

import capture_nemo_train_logits_v1 as capture
from metrics import score_transcriptions
from resource_watchdog_v2 import sample_memory
from runtime import ROOT, OUTPUT, read_audio

EXPECTED_WEIGHTS_SHA = '4790b0a53c0e99ba6ea1a2b8bd9bbdd1d9787ca80b9a38ea6107701c098b98ed'
EXPECTED_LOADER_SHA = '698c5394567fa0721222b49e2d8b67f9b76422fede4ef8bf00af22e897dc79cb'
require = capture.require
sha = capture.sha


def selected_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    selected = []
    for language in ('tr', 'en'):
        group = [row for row in rows if row['language'] == language][:8]
        require(len(group) == 8, 'Eight fixed development clips per language required')
        selected.extend(group)
    require(len({row['source_id'] for row in selected}) == 16, 'Duplicate development source')
    for row in selected:
        require(row['dataset'] == 'google/fleurs' and row['dataset_split'] == 'validation', 'FLEURS DEV only')
        require(sha(row['audio_filepath']) == row['wav_sha256'], 'Development WAV changed')
        info = sf.info(row['audio_filepath'])
        require(info.samplerate == 16000 and info.channels == 1 and
                abs(info.duration-row['duration']) < 1e-5 and 0 < info.duration <= 30,
                'Finite mono16k full development WAV required')
    return selected


def groups(order, batch_size):
    require(batch_size in (1, 4, 8, 16) and sorted(order) == list(range(16)), 'Fixed-set batch grouping')
    return [order[index:index+batch_size] for index in range(0, len(order), batch_size)]


def transcript_comparison(reference, candidate, rows):
    require(len(reference) == len(candidate) == len(rows) == 16, 'Sixteen aligned hypotheses required')
    differences = [{'source_id':row['source_id'], 'language':row['language'],
                    'before':before, 'after':after}
                   for row, before, after in zip(rows, reference, candidate) if before != after]
    return {'exact_transcripts_equal':not differences, 'changed_utterances':len(differences),
            'differences':differences}


def scores(texts, rows):
    return {language:score_transcriptions([
        {**row, 'hypothesis':text} for row, text in zip(rows, texts) if row['language'] == language
    ], language=language) for language in ('tr', 'en')}


def memory_event(phase):
    try:
        return {'phase':phase, **sample_memory(os.getpid(), phase)}
    except Exception as error:
        return {'phase':phase, 'available':False, 'error':str(error)}


def critical_commit(event):
    system = event.get('system_memory', {})
    process = event.get('calling_process_commit_memory', {})
    process_headroom = process.get('ullAvailPageFile_bytes')
    if process.get('available') and process_headroom is not None and process_headroom < 2e9:
        return True
    if not system.get('available'):
        return False
    fraction, headroom = system.get('commit_fraction'), system.get('commit_available_bytes')
    return ((fraction is not None and fraction >= .95) or
            (headroom is not None and headroom < 2e9))


def infer_pass(model, processor, signals, batch_size, order):
    hypotheses = [None]*16
    batch_metadata = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for indices in groups(order, batch_size):
            inputs = dict(processor(audio=[signals[index] for index in indices],
                                    sampling_rate=16000, return_tensors='pt', padding='longest'))
            mask = inputs['attention_mask'].bool()
            inputs['input_features'] = inputs['input_features'].masked_fill(~mask.unsqueeze(-1), 0)
            valid_frames = mask.sum(-1).tolist()
            batch_metadata.append({'indices':indices, 'valid_frontend_frames':valid_frames,
                'padded_frontend_frames':int(mask.shape[1]),
                'frontend_padding_ratio':mask.numel()/max(1, int(mask.sum()))})
            inputs = {name:value.cuda() for name, value in inputs.items()}
            generated = model.generate(**inputs, do_sample=False, num_beams=1)
            sequences = generated.sequences.detach().cpu()
            texts = processor.batch_decode(sequences, skip_special_tokens=True)
            require(len(texts) == len(indices), 'Decoder batch cardinality differs')
            for index, text in zip(indices, texts):
                require(isinstance(text, str), 'Invalid decoded hypothesis')
                hypotheses[index] = text
    require(all(isinstance(value, str) for value in hypotheses), 'Unassigned inference source')
    return hypotheses, batch_metadata


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=ROOT/'work'/(capture.RUN+'_best.pt'))
    parser.add_argument('--weights-sha256', default=EXPECTED_WEIGHTS_SHA)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--gpu-window-authorized', action='store_true')
    parser.add_argument('--include-b16', action='store_true', help='Additional guarded case; not required')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    args = parser.parse_args()
    require(args.output.resolve().is_relative_to(ROOT) and not args.output.exists(), 'NEW workspace output required')
    require(args.repeats >= 3 and args.warmup >= 1, 'At least three repeats and one warmup pass')
    require(args.repeats <= 5 and args.warmup <= 2, 'Bounded inference window required')
    require(args.weights_sha256 == EXPECTED_WEIGHTS_SHA and sha(args.weights) == args.weights_sha256, 'Exact1871 weights')
    require(sha(capture.__file__) == EXPECTED_LOADER_SHA, 'Frozen capture/architecture loader changed')
    config, sources = capture.input_sources()
    data_path = ROOT/'work/data/validation.jsonl'
    rows = selected_rows(data_path)
    audio_seconds = sum(row['duration'] for row in rows)
    sources.update(data_path=str(data_path), data_sha256=sha(data_path),
                   shared_loader_path=str(Path(capture.__file__).resolve()), shared_loader_sha256=sha(capture.__file__))
    report = {'schema':1, 'kind':'selected1871_fixed_development_inference_throughput',
        'run_id':capture.RUN, 'checkpoint_step':1871, 'checkpoint_sha256':args.weights_sha256,
        'producer_sha256':sha(__file__), 'sources':sources, 'split':'FLEURS validation only',
        'rows':rows, 'utterances_per_pass':16, 'unpadded_audio_seconds_per_pass':audio_seconds,
        'precision':'FP32 resident weights, BF16 autocast, default model SDPA implementation',
        'generation':{'do_sample':False, 'num_beams':1, 'confidence_hooks':False,
                      'other_settings':'Unmodified pretrained generation configuration'},
        'timing_scope':'Each complete16-clip pass includes CPU frontend, zero-padding masks, H2D, encoder, greedy decoder, D2H and text decode. Raw WAV arrays prefetched once; excludes file reads, checkpoint/model loading, warmup and memory telemetry.',
        'warmup_complete_passes_per_case':args.warmup, 'timed_complete_passes_per_case':args.repeats,
        'scope_limits':['Sixteen fixed DEV recordings; descriptive repeated workload, not general production throughput.',
                        'TR and EN grouped separately for B1/4/8; optionalB16 mixes languages and has more padding.',
                        'Allocator cache is retained between cases; timed peak counters reset after each case warmup.',
                        'A transcript difference is retained; speed does not imply numerical or quality equivalence.'],
        'cases':{}, 'memory_events':[]}
    if args.prepare_only:
        require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA')
        report.update(passed=True, cuda_initialized=False, scope='CPU source/row preparation only; no inference')
        save(args.output, report)
        print(json.dumps({'prepared':True, 'audio_seconds':audio_seconds, 'utterances':16, 'output':str(args.output)}))
        return
    require(args.gpu_window_authorized, 'Parent-owned solo GPU window required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    total = torch.cuda.get_device_properties(0).total_memory
    require(15e9 <= total <= 18e9, 'Declared16GB-class local GPU required')
    cap = 9_000_000_000
    torch.cuda.set_per_process_memory_fraction(cap/total, device=0)
    report['gpu'] = {'name':torch.cuda.get_device_name(0), 'total_memory_bytes':total,
                     'allocator_cap_bytes':cap, 'cuda_matmul_allow_tf32':False,
                     'cudnn_allow_tf32':True, 'cudnn_benchmark':False}
    report['memory_events'].append(memory_event('before_model_load'))
    require(not critical_commit(report['memory_events'][-1]), 'Critical memory before model loading; no model allocated')
    signals = [read_audio(row) for row in rows]
    require(abs(sum(len(signal)/16000 for signal in signals)-audio_seconds) < 1e-5, 'Unpadded seconds changed')
    model, processor, model_evidence = capture.load_evaluation_model(args.weights, args.weights_sha256)
    torch.set_num_threads(2)
    report['model_evidence'] = model_evidence
    report['gpu']['attention_implementation'] = model.config._attn_implementation
    report['generation_config'] = model.generation_config.to_dict()
    report['base_allocated_bytes'] = torch.cuda.memory_allocated()
    order = list(range(16))
    for batch_size in ((1,4,8,16) if args.include_b16 else (1,4,8)):
        event = memory_event('before_batch_'+str(batch_size)); report['memory_events'].append(event)
        if critical_commit(event):
            report['cases'][str(batch_size)] = {'status':'skipped', 'reason':'Critical system commitment; no next case launched'}
            break
        if batch_size == 16:
            free, _ = torch.cuda.mem_get_info()
            previous_peak = report['cases']['8']['peak_allocated_bytes']
            estimated_increment = max(0, previous_peak-report['base_allocated_bytes'])*2.25
            if free < 2e9 or torch.cuda.memory_allocated()+estimated_increment >= cap-500e6:
                report['cases']['16'] = {'status':'skipped', 'reason':'ConservativeB8-derived allocator/device headroom bound',
                    'device_free_bytes':free, 'estimated_increment_bytes':estimated_increment}
                continue
        try:
            for _ in range(args.warmup): infer_pass(model, processor, signals, batch_size, order)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            elapsed = []; reference = None; repeated = []
            for repeat in range(args.repeats):
                torch.cuda.synchronize(); start = time.perf_counter()
                texts, metadata = infer_pass(model, processor, signals, batch_size, order)
                torch.cuda.synchronize(); seconds = time.perf_counter()-start
                require(seconds > 0, 'Nonpositive inference time')
                elapsed.append(seconds)
                if reference is None: reference = texts
                else: repeated.append(transcript_comparison(reference, texts, rows))
            case = {'status':'measured', 'physical_batch_size':batch_size, 'elapsed_seconds_per_pass':elapsed,
                'median_seconds_per_pass':statistics.median(elapsed),
                'audio_minutes_per_wall_minute':audio_seconds/statistics.median(elapsed),
                'utterances_per_second':16/statistics.median(elapsed),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(), 'batch_metadata':metadata,
                'hypotheses':reference, 'repeat_comparisons':repeated, 'development_scores':scores(reference, rows)}
            if batch_size != 1:
                case['versus_b1'] = transcript_comparison(report['cases']['1']['hypotheses'], reference, rows)
                case['speedup_vs_b1'] = report['cases']['1']['median_seconds_per_pass']/case['median_seconds_per_pass']
            report['cases'][str(batch_size)] = case
            print(json.dumps({'batch_size':batch_size, 'audio_min_per_min':case['audio_minutes_per_wall_minute'],
                              'seconds':elapsed, 'max_allocated_GB':case['peak_allocated_bytes']/1e9}), flush=True)
        except torch.OutOfMemoryError as error:
            report['cases'][str(batch_size)] = {'status':'out_of_memory', 'error':str(error)}
            gc.collect(); torch.cuda.empty_cache()
            break
        report['memory_events'].append(memory_event('after_batch_'+str(batch_size)))
    if report['cases'].get('8',{}).get('status') == 'measured' and not critical_commit(report['memory_events'][-1]):
        permuted, _ = infer_pass(model, processor, signals, 8, list(reversed(order)))
        torch.cuda.synchronize()
        report['batch8_reversed_order_comparison'] = transcript_comparison(report['cases']['8']['hypotheses'], permuted, rows)
    require(sha(args.weights) == args.weights_sha256 and sha(data_path) == sources['data_sha256'], 'Checkpoint/input changed')
    for name, digest in sources['code_sha256'].items(): require(sha(OUTPUT/name) == digest, 'Frozen source changed:'+name)
    require(sha(capture.__file__) == EXPECTED_LOADER_SHA, 'Shared loader changed')
    report['integrity_passed'] = True
    report['required_cases_completed'] = all(report['cases'].get(str(size),{}).get('status') == 'measured' for size in (1,4,8))
    report['scope'] = 'Actual selected1871 localGPU inference; speed and transcript equality reported separately'
    save(args.output, report)
    print(json.dumps({'completed':report['required_cases_completed'], 'output':str(args.output), 'sha256':sha(args.output)}))


if __name__ == '__main__': main()
