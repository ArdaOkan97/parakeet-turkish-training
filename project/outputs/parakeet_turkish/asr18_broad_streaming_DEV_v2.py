"""Optimizer-free serial fixed350 DEV of a closed streamed diagnostic state."""
import gc, json, random, threading
from pathlib import Path
from atomic_io_retry_v1 import write_json
from bpe_streaming_checkpoint_v1 import MappedCheckpoint, record, require
from bpe_broad_production_v3 import frozen_parameter, prohibit_model_transforms

def evaluate_checkpoint(model, processor, folder, config, *, torch, stage, admit, stop_check):
    require(config['production_enabled'] is False and threading.current_thread() is threading.main_thread(),
            'Finite serial diagnostic DEV only')
    updates=config['DEV_expected_native_updates']
    require(type(updates) is int and 1<=updates<=20,'Explicit finite actual native frontier required')
    expected_clocks=[c+updates for c in (64250,2000,62250)]
    binding = config['DEV_checkpoint']
    report_binding = config['DEV_training_report']
    closure_binding = config['DEV_actual_tool_closure']
    require(record(report_binding['path']) == report_binding and record(closure_binding['path']) == closure_binding,
            'Actual closed native proof changed')
    training = json.loads(Path(report_binding['path']).read_text())
    closure = json.loads(Path(closure_binding['path']).read_text())
    require(training['passed'] and training['actual_exit_code'] == 0 and training['completion']['checkpoint'] == binding
            and closure['kind'] == 'ASR18_owned_actual_closed_GPU_child_v1' and closure['actual_tool_exit_code'] == 0
            and closure['actual_native_exit_code'] == 0 and closure['report'] == report_binding
            and closure['actual_tool_chunk_id'] and closure['owner'] == '/root/memory_diagnosis',
            'Passed actual-owned closed checkpoint proof required')
    require(model.joint._ASR18_fp32_joint_policy['kind'] == 'ASR18_joint_canonical_T1_U128_FP32_v2', 'Exact new joint recipe required')
    prohibit_model_transforms(model, torch)
    admit('stream_checkpoint_weights_only_restore_for_DEV', 350_000_000)
    mapped = MappedCheckpoint(binding, torch=torch, guard=stop_check)
    try:
        def restore(saved):
            from bpe_broad_full207_optimizer_v1 import bits_equal
            require(saved['diagnostic_only'] is True and len(saved['weights']) == 207
                    and saved['optimizer']['clocks'] == expected_clocks
                    and saved['sampler']['broad_cursor']['draws'] == 14*updates, 'Exact actual unqualified diagnostic frontier required')
            result = model.load_state_dict(saved['weights'], strict=False)
            actual = dict(model.named_parameters())
            require(not result.unexpected_keys and all(name in actual and bits_equal(torch, actual[name], value)
                    for name, value in saved['weights'].items()), 'All207 DEV weights must restore byte exactly')
            random.setstate(saved['python_rng']); torch.set_rng_state(saved['torch_rng'])
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        restore(mapped.value)
    finally:
        mapped.close()
    gc.collect(); model.requires_grad_(False); model.eval(); torch.cuda.empty_cache()
    require(all((parameter := frozen_parameter(model, name)).device.type == 'cpu'
                and not parameter.requires_grad and parameter.data_ptr() == backing.data_ptr()
                for name, backing in model._diagnostic_frozen_backing.items()), 'Readonly frozen backing changed during DEV restore')
    stage('new_joint_optimizer_free_serial350DEV_ready', checkpoint=binding)
    # Prior actual fixed350 DEV rose1.8343GB above its ready baseline.
    # Reserve that measured increase plus250MB, rounded up; guards unchanged.
    admit('new_joint_optimizer_free_serial_full350DEV', 2_100_000_000)
    from train_lora_batched_tsc_bpe_full_decoder_v1 import evaluate, gate
    import benchmark
    tag = 'ASR18_'+folder.name
    require(not (Path(r'E:\gpu_training_scratchpad')/'results'/tag).exists(), 'Fresh evaluation result namespace required')
    original = benchmark.inputs_for
    def inputs(processor, row):
        stop_check()
        return original(processor, row)
    benchmark.inputs_for = inputs
    try:
        scores = evaluate(model, processor, tag)
    finally:
        benchmark.inputs_for = original
    value = gate(scores)
    require(value['retention_allowed_word_errors'] == {'en': 69, 'es': 52, 'it': 30, 'fr': 74, 'de': 70, 'pt': 66}, 'Exact strict integer retention gates required')
    require(model._diagnostic_frozen_metrics['logical_current_matrix_bytes'] == 0, 'In-flight frozen hook at serial DEV end')
    report = {'passed': True, 'full350_DEV_complete': True, 'serial_optimizer_free': True,
        'checkpoint': binding, 'training_report': report_binding, 'training_actual_closure': closure_binding,
        'all207_weight_restore_byte_exact': True, 'optimizer_histories_loaded_for_DEV': False,
        'clocks_preserved_in_checkpoint': expected_clocks, 'actual_native_frontier':updates, 'new_joint_policy': model.joint._ASR18_fp32_joint_policy,
        'DEV': value, 'diagnostic_checkpoint_unqualified': True, 'quality_gain_or_winner_claimed': False,
        'full_broad_corpus_passes': 0, 'TEST_used': False, 'result_tag': tag}
    write_json(folder/'final_DEV_validation.json', report)
    stage('new_joint_full350DEV_complete', DEV=value)
    return report
