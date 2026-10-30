"""Build a CPU-only source/evidence review; this does not import or train NeMo."""
import hashlib
import json
from pathlib import Path
from collections import Counter

import yaml

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PIN = ROOT / 'work/research/nemo_reference/pinned_e4a34fd25f89'
COMMIT = 'e4a34fd25f895ecd7f64f0e56bc288165e078da9'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def evidence(path):
    return {'path': str(path.resolve()), 'sha256': sha(path)}


def main():
    receipts = [PIN/'source_receipt.json', PIN/'extra_source_receipt.json', PIN/'decoder_source_receipt.json']
    sources = {}
    for receipt_path in receipts:
        receipt = read(receipt_path)
        assert receipt['commit'] == COMMIT
        for item in receipt['files']:
            assert sha(item['path']) == item['sha256'], item['repo_path']
            sources[item['repo_path']] = evidence(Path(item['path']))
    greedy = PIN/'nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py'
    assert sha(greedy) == '2684ad07f189b1ce98abfa4d405720018351d7242ac6d14a3029218e2fda9b8e'
    sources['nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py'] = evidence(greedy)
    native_path = ROOT/'work/research/nemo_reference/parakeet_v3_native_model_config.yaml'
    assert sha(native_path) == 'eadae985ef97451402be2faef3b240c2cae7d9cc0362c5bd00e64dfc4265ed09'
    native = yaml.safe_load(native_path.read_text(encoding='utf-8'))
    loss_path = OUT/'nemo_tdt_reference_cpu_verification_v2.json'
    frontend_path = OUT/'nemo_frontend_reference_cpu_verification_v1.json'
    loss = read(loss_path)
    frontend = read(frontend_path)
    assert loss['passed'] and not loss['cuda_initialized']
    assert frontend['passed_real_audio_matched_settings'] and not frontend['cuda_initialized']
    run_id = 'adaptive_20261010_continuation_02_resume_B8'
    dev_dir = OUT/'results'/(run_id+'_step_1821')
    decoder_observations = {}
    for path in sorted(dev_dir.glob('validation_*.jsonl')):
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        maxima = [max(Counter(t['start_frame'] for t in row['confidence']['tokens']).values(), default=0) for row in rows]
        decoder_observations[path.stem] = {
            'evidence': evidence(path), 'clips': len(rows),
            'maximum_nonblank_tokens_at_one_frame': max(maxima, default=0),
            'clips_with_more_than_10_nonblank_tokens_at_one_frame': sum(x > 10 for x in maxima),
            'nonblank_zero_duration_tokens': sum(t['duration_frames'] == 0 for row in rows for t in row['confidence']['tokens'])}
    assert len(decoder_observations) == 6 and sum(x['clips'] for x in decoder_observations.values()) == 300
    report = {
        'schema': 1, 'kind': 'pinned_official_nemo_training_source_review',
        'passed_source_evidence_checks': True, 'official_commit': COMMIT,
        'sources': sources, 'native_v3_configuration': evidence(native_path),
        'native_v3_configuration_prefix_receipt': evidence(ROOT/'work/research/nemo_reference/parakeet_v3_native_config_prefix_receipt.json'),
        'released_native_nemo_version': native['nemo_version'],
        'review_source_version_caveat': 'Pinned current Speech sources are newer than released model nemo_version; no native full-model equivalence has been executed.',
        'executed_proofs': {'tdt_reference': evidence(loss_path), 'frontend_reference': evidence(frontend_path)},
        'proof_limits': [
            'Earlier oracles used the Transformers TDT loss, not NVIDIA NeMo.',
            'New nine-case official-reference audit extracts the pinned PyTorch recurrence and changes only two CUDA allocation calls to the input device; it uses synthetic FP32 CPU tensors.',
            'No native NeMo module/typechecker, CUDA Numba TDT kernel, mixed-precision native kernel, full encoder/decoder, or complete NeMo training loop has been executed.',
            'The official PyTorch reference has no omega RNNT branch or FastEmit; passing it does not validate those training settings.',
            'Frontend audit covers four fixed real FLEURS DEV clips in singleton/padded B4 with dither disabled; no complete native encoder proof.'],
        'recipe_comparison': {
            'released': {'loss': native['loss'], 'reduction': native['rnnt_reduction'],
                'optimizer': native['optim'], 'decoding': native['decoding'],
                'preprocessor': native['preprocessor'], 'spec_augment': native['spec_augment'],
                'fused_batch_size': native['joint']['fused_batch_size'],
                'fuse_loss_wer': native['joint']['fuse_loss_wer'], 'target': native['target']},
            'current_custom': {'sigma': 0.0, 'omega': 0.0, 'fastemit_lambda': 0.0,
                'reduction': 'mean: each raw clip NLL divided by its target length, then equal clip mean',
                'optimizer': 'AdamW default betas(.9,.999); heads weight_decay .01, LoRA weight_decay0; separate head/LoRA learning rates; warmup25 then constant',
                'spec_augment': 'disabled', 'dither': 0.0,
                'retention': '50% multilingual replay plus token KD5, duration KD.5, feature KD1, five held-out DEV retention gates',
                'encoder': 'all24 attention q/k/v/output rank8 adapters; frozen base encoder kept eval; trainable heads'},
            'interpretation': 'Matching recurrence math does not mean matching released training objective. sigma .02 and omega .1 are justified official-recipe experiment candidates; FastEmit .001 and Noam LR5 belong to a generic from-scratch config and should not be copied.'},
        'source_level_risks_to_verify_before_migration': [
            {'topic': 'vocabulary transplant', 'finding': 'Generic fine-tune update_tokenizer restores decoder/joint only when vocabulary sizes match. A changed size leaves newly initialized prediction/joint modules. Preserve old token IDs and explicitly transplant embedding/output/duration rows; assert the ID-to-piece map even when sizes match.'},
            {'topic': 'TDT loss after tokenizer change', 'finding': 'Pinned RNNT BPE change_vocabulary reconstructs RNNTLoss(num_classes=joint.num_classes_with_blank-1) without TDT loss name/kwargs, while the initial constructor subtracts num_extra_outputs for TDT. Explicitly verify/recreate the TDT loss and blank ID after resizing.'},
            {'topic': 'SpecAug override attribute', 'finding': 'Generic fine-tune writes spec_augment; native RNNT forward reads spec_augmentation. Verify whether the loaded model has an alias; otherwise set/verify the actual module used by forward.'},
            {'topic': 'adapter type', 'finding': 'Official ASR adaptation example trains enabled bottleneck/Houlsby/attention adapters, not a drop-in equivalent of current all-layer attention LoRA. Freeze base weights and verify BN/dropout modes explicitly.'},
            {'topic': 'CTC config', 'finding': 'The released target is EncDecRNNTBPEModel, not a hybrid RNNT-CTC class. The aux_ctc .3 YAML entry alone does not show an active CTC training loss.'},
            {'topic': 'decoder safeguard', 'finding': 'Pinned singleton GreedyTDTInfer limits joint steps per frame. Native greedy_batch label-looping also forces time advance after max_symbols nonblank emissions at one timestamp; HF TDT skips this guard. Native greedy_batch and HF both force blank duration0 to1. Native maps duration argmax through configured durations; HF uses the index directly, equivalent for current durations0..4. No native full decoder execution/parity has been measured.'}],
        'decoder_saved_dev_observations': decoder_observations,
        'decoder_observation_limit': 'Saved token metadata excludes blank steps. No >10 nonblank emissions per frame was seen; this does not prove full step-by-step native decoder equivalence or every max-symbol guard condition.',
        'frontend_silence_difference': {
            'finding': 'NeMo reference-centered per-feature mean yields exact zero for constant log-mel features; HF reduction leaves roundoff which is amplified by near-zero standard deviation.',
            'all_zero_1sec_hf_max_abs': 0.1600533276796341,
            'all_zero_4_21sec_hf_max_abs': 0.2760425806045532,
            'native_max_abs': 0.0,
            'limit': 'Measured synthetic silence edge; not evidence it explains current Turkish WER. Any frontend change requires a new cache namespace and consistent teacher/student handling.'},
        'reuse_plan': [
            'First isolate a versioned sigma .02 experiment from the same DEV-selected weights, unchanged data/replay and retention gates; compare actual DEV and calibration. Do not change an exact-resume objective silently.',
            'For full released-objective comparison implement/test omega .1 standard-RNNT mixing and mean_volume with an explicitly matched effective-batch policy. In native NeMo, use its official TDT kernel; verify raw per-clip loss and gradients on the same saved joint logits before training.',
            'Try AdamW betas(.9,.98), conservative LR decay and modest augmentation as separate DEV-controlled changes. Keep separate head/adapter/new-row rates and teacher preservation; do not transplant generic LR5/5000warmup into the small pilot.',
            'Use official dynamic duration/length batching and fused batch4 joint processing as reference memory engineering, with measured time/host commit and exact full-vocabulary loss semantics.',
            'A native migration restores the actual0.6B .nemo model, explicitly preserves old rows and TDT settings, then validates deterministic frontend/encoder/transcripts before training. Port replay/KD and all six DEV evaluation splits; freeze winner on DEV before FLEURS TEST.',
            'Native Lightning resume is distinct from loading .nemo/pretrained weights. Custom207-parameter Adam/sampler/RNG state is not directly a Lightning checkpoint; migration is a declared warm-start trial, not an exact resume.'],
        'native_environment': {
            'status': 'No installation attempted by this audit; parent completed a successful resolver-only native Windows probe.',
            'normal_resolver_plan': evidence(ROOT/'work/research/nemo_reference/nemo_native_windows_dependency_plan_v3.json'),
            'normal_resolver_log': evidence(ROOT/'work/research/nemo_reference/nemo_native_windows_dependency_probe_v3.log'),
            'source_facts': ['Pinned package requires Python>=3.10 and core Torch>=2.7; current Python3.11/Torch2.7.1 meet those core version constraints.',
                'Official README recommends NVIDIA Linux containers; pinned uv source rules route Windows Torch to CPU. That uv policy is not proof pip/NeMo CUDA cannot work on Windows.',
                'NeMo ASR needs Lightning/Hydra/OmegaConf/Lhotse and CUDA Numba dependencies; actual native CUDA loss import/JIT must be tested in an isolated environment before recommending migration.'],
            'measured_probe_limits': ['First resolver failed sandbox temporary-directory permissions.',
                'Binary-only resolver rejected source-only pure-Python antlr4-python3-runtime4.9; this is not a native CUDA blocker.',
                'Normal prefer-binary native Windows dependency resolution succeeded, with no Torch replacement in the plan; this is not an import/CUDA kernel execution test.',
                'The resolved plan includes pytorch-lightning2.6.6 alongside lightning2.4.0; constrain both to2.4.0 for a consistent isolated trial. Preserve current training environment and Torch build.']},
        'code_sha256': sha(__file__)}
    output = OUT/'nemo_training_review_v1.json'
    assert not output.exists(), 'Review output already exists'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    text = """Official NeMo review (pinned e4a34fd25f89)

The custom trainer matches the official PyTorch TDT recurrence in nine synthetic CPU FP32 cases, including sigma .02 and all loss reductions. This is an extracted-reference test with two device-allocation edits, not a native NeMo CUDA/full-model test. Earlier parity tests used Transformers as the oracle.

The released v3 .nemo config differs from our recipe: sigma .02, omega .1 and mean_volume reduction; ours uses sigma0, no RNNT mixture, and equal length-normalized clip weighting. Released FastEmit is0. Native AdamW uses betas(.9,.98), weight_decay .001 and cosine decay; ours uses(.9,.999), separate head/adapter groups and constant LR after warmup. These are concrete matched DEV experiment candidates, not a demonstrated explanation of current accuracy.

The native frontend is close on four real DEV clips (max relative L2 5.53e-6 with matched dither0), but on synthetic all-zero audio HF produces up to .276 absolute normalized-feature magnitude while the official stable-mean calculation gives exactly0. A versioned frontend change needs fresh student/teacher cache fingerprints. The native singleton decoder has a per-frame symbol guard which HF TDT skips; saved latest300 DEV clips show at most5 nonblank tokens at one frame, so no observed runaway nonblank guard trigger.

Do not use the generic fine-tune tokenizer replacement unmodified for retention: it reinitializes prediction/joint modules when vocabulary size changes. Explicit old-row/token-ID transplant is required. Pinned change_vocabulary also rebuilds the loss without its original TDT options; validate/recreate that loss. The SpecAug override name and the actual RNNT forward attribute differ in the inspected sources and need a runtime check. The released model target is RNNT BPE, so aux_ctc .3 in YAML does not itself show an active CTC objective.

Use the released0.6B architecture/config, controlled sigma/omega/reduction experiments, careful LR decay, duration/length buckets and fused joint memory processing. Keep multilingual replay, KD and all five retention DEV gates. Official adapter examples are not a drop-in replacement for current attention LoRA. Native migration must validate the CUDA kernel on identical saved logits, compare deterministic frontend/encoder/decoding, then run a declared warm-start trial; custom Adam/sampler checkpoints cannot be described as native Lightning exact resumes.

Native Windows dependency resolution succeeded without a planned Torch replacement. Imports and CUDA kernel execution remain untested. The core Python/Torch constraints are satisfied; Linux container guidance and Windows uv CPU routing are source facts, not a demonstrated native pip/CUDA failure. Prepare an isolated environment and pin both Lightning distributions to2.4.0. No NeMo package was installed by this review.
"""
    (OUT/'nemo_training_review_v1.txt').write_text(text, encoding='utf-8')
    print(json.dumps({'passed': True, 'report': str(output), 'report_sha256': sha(output), 'code_sha256': sha(__file__)}))


if __name__ == '__main__':
    main()
