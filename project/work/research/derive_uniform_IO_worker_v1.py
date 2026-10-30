from pathlib import Path
import ast, hashlib

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'outputs/parakeet_turkish'
parent = HERE / 'train_lora_batched_tsc_uniform_full_epoch_v1.py'
assert hashlib.sha256(parent.read_bytes()).hexdigest() == '1c33cbde6933c55d0839d0ebf82253fb3b673dde7f5a105adc3905e14f7013c3'
text = parent.read_text(encoding='utf-8')
text = text.replace('tsc_uniform_full_epoch_transition_v1', 'tsc_uniform_io_transition_v1')
text = text.replace('run_TSC_uniform_full_epoch_v1.py', 'run_TSC_uniform_io_v1.py')
text = text.replace('tsc_uniform_majority_ptfocus_sampler_v1', 'tsc_uniform_io_sampler_v1')
text = text.replace('TSCUniformMajorityPTFocusSampler', 'TSCUniformIOSampler')
text = text.replace('proof_path, ptfocus_declaration, restore_parent_sampler, full_epoch_declaration)',
                    'proof_path, ptfocus_declaration, restore_parent_sampler, full_epoch_declaration, io_declaration)')
text = text.replace('"declared": full_epoch_declaration(context),', '"declared": io_declaration(context,args.persistence_mode),')
text = text.replace('"kind": "TSC_uniform_USER_full_epoch_monitor_only_full_state_v1"',
                    '"kind": "TSC_uniform_matched_IO_full_state_v1"')
old = '    sampler_transition=restore_parent_sampler(sampler,source,context)'
assert text.count(old) == 1
text = text.replace(old, old + '\n    sampler_transition={"strict_parent_restore":sampler_transition,\n        "technical_extension":sampler.begin_technical_extension(context)}')
text = text.replace('"full_epoch_policy_transition":transition_identity(args,context,sampler_transition),',
                    '"full_epoch_policy_transition":context["parent_identity"]["full_epoch_policy_transition"],\n        "IO_control_transition":transition_identity(args,context,sampler_transition),')
text = text.replace('"full_epoch_policy_transition":config_identity["full_epoch_policy_transition"],',
                    '"full_epoch_policy_transition":config_identity["full_epoch_policy_transition"],\n        "IO_control_transition":config_identity["IO_control_transition"],\n        "persistence_control":{"source_step":source_step,"updates":100,"mode":args.persistence_mode,\n            "original_full_pass_end_step":25109,"technical_end_step":25209,"quality_recipe_changed":False},')
text = text.replace('"phase":"HF_TSC_8199_uniform_Turkish_majority_PT_focused_v1"',
                    '"phase":"HF_TSC_8199_uniform_IO_control_v1"')
text = text.replace('"retention-stop-policy', '"retention-stop-policy')  # No recipe changes.
start = text.index('        if (source_step!=9000 or start_step!=9000')
end = text.index('    random.setstate(resume["python_rng"])', start)
text = text[:start] + '''        if (source_step!=25109 or start_step!=25109 or retention_failures!=23 or selected_step!=19750
                or best!=inherited_gate["turkish_wer"] or not inherited_gate["retention_gate_passed"]
                or source_gate["retention_gate_passed"] or not equality["passed"]):
            raise ValueError("Matched IO source must retain coherent25109/counter23/separate qualified BEST19750")
''' + text[end:]
text = text.replace('"portuguese_token_kl_multiplier","retention_stop_policy")',
                    '"portuguese_token_kl_multiplier","retention_stop_policy","persistence_mode","persistence_benchmark_updates")')
text = text.replace('"native_tdt_checkpointed_v4.py","lazy_training_rows_v1.py"):',
                    '"native_tdt_checkpointed_v4.py","lazy_training_rows_v1.py","atomic_paired_history_v1.py"):')
text = text.replace('"tsc_uniform_io_sampler_v1.py","lazy_training_rows_v1.py","native_tdt_checkpointed_v4.py"):',
                    '"tsc_uniform_io_sampler_v1.py","lazy_training_rows_v1.py","native_tdt_checkpointed_v4.py","atomic_paired_history_v1.py"):')
# A and B use this same worker/update loop; only these paired publications differ.
marker = '    def save(path,step,reason=None):'
assert text.count(marker) == 1
publisher = '''    from atomic_paired_history_v1 import PairedHistoryPublisher
    pair_publisher=PairedHistoryPublisher(run_dir/"training_history.json",OUTPUT/"training_history.json")
    pair_events=[]
    def publish_history(label):
        before=time.perf_counter()
        if args.persistence_mode=="single_serialize_two_commits":
            detail=pair_publisher.publish(history)
        else:
            first=time.perf_counter();write_json(run_dir/"training_history.json",history)
            first_elapsed=time.perf_counter()-first
            second=time.perf_counter();write_json(OUTPUT/"training_history.json",history)
            detail={"mode":"double_json","first_original_write_seconds":first_elapsed,
                    "second_original_write_seconds":time.perf_counter()-second}
        pair_events.append({"label":label,"step":history[-1]["step"],
            "pair_total_observed_seconds":time.perf_counter()-before,"detail":detail})
        write_json(run_dir/"persistence_timing.json",{"kind":"matched_history_persistence_actual_timing_v1",
            "run_id":args.run_id,"mode":args.persistence_mode,"source_step":source_step,
            "original_format_and_frequency":True,"events":pair_events,
            "publisher_counters":pair_publisher.counters_snapshot(),
            "no_training_math_change":True})
'''
text = text.replace(marker, publisher + marker)
pairs = [
    ('    write_json(run_dir/"training_history.json",history);write_json(OUTPUT/"training_history.json",history)', '    publish_history("initial")'),
    ('        write_json(run_dir / "training_history.json", history)\n        write_json(OUTPUT / "training_history.json", history)', '        publish_history("DEV")'),
    ('            write_json(run_dir / "training_history.json", history)\n            write_json(OUTPUT / "training_history.json", history)', '            publish_history("periodic")'),
    ('    write_json(run_dir / "training_history.json", history)\n    write_json(OUTPUT / "training_history.json", history)', '    publish_history("final")'),
]
for old, new in pairs:
    assert text.count(old) == 1, (old, text.count(old))
    text = text.replace(old, new)
text = text.replace('"full_epoch_continuation_updates":step-source_step,',
                    '"full_epoch_continuation_updates":step-9000,"persistence_control_updates":step-source_step,"persistence_mode":args.persistence_mode,')
text = text.replace('                 "sampling_identity":sampler.identity,',
                    '                 "sampling_identity":sampler.identity,"persistence_control_updates":completed-source_step,\n                 "persistence_mode":args.persistence_mode,"IO_control_transition":config_identity["IO_control_transition"],')
text = text.replace('    return p\n\n\ndef validate_arguments',
                    '    p.add_argument("--persistence-mode",choices=("double_json","single_serialize_two_commits"),required=True)\n    p.add_argument("--persistence-benchmark-updates",type=int,default=100)\n    return p\n\n\ndef validate_arguments')
text = text.replace('    majority_preflight(args)\n    return args',
                    '    if args.persistence_benchmark_updates!=100:raise ValueError("Matched technical control requires exactly100 updates")\n    majority_preflight(args)\n    return args')
text = text.replace('"kind": "TSC_uniform_USER_full_epoch_monitor_only_full_state_v1"', '"kind": "TSC_uniform_matched_IO_full_state_v1"')
ast.parse(text)
target=HERE/'train_lora_batched_tsc_uniform_io_v1.py'
assert not target.exists()
target.write_text(text,encoding='utf-8')
sync=[n for n,line in enumerate(text.splitlines(),1) if line.strip()=='torch.cuda.synchronize()']
print({'worker':str(target),'SHA':hashlib.sha256(target.read_bytes()).hexdigest(),'synchronizes':sync})
