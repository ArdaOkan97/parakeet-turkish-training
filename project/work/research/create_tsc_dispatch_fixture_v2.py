"""Create the narrow v2 sibling fixture without modifying frozen v1 evidence."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
source = (ROOT / "work/research/check_tsc_runtime_dispatch_v1.py").read_text(encoding="utf-8")
for old, new in (
    ("_tsc_v1", "_tsc_v2"),
    ("tsc_training_inputs_v1", "tsc_training_inputs_v2"),
    ("tsc_supplemental_data_policy_v1", "tsc_supplemental_data_policy_v2"),
    ("TSC_supplemental_TR_frame_target_inputs_v1", "TSC_supplemental_TR_frame_target_inputs_v2"),
    ("root_owned_TSC_closed_source_v1", "root_owned_TSC_closed_source_v2"),
    ("isolated_TSC_runtime_mocked_CPU_dispatch_verification_v1", "isolated_TSC_runtime_mocked_CPU_dispatch_verification_v2"),
):
    source = source.replace(old, new)
source = source.replace('import tsc_training_inputs_v2 as inputs',
    'import tsc_training_inputs_v2 as inputs\nimport launch_with_memory_guard_TSC_resume_v2 as exact_memory')
source = source.replace('"domain30_runtime_chain_pins_v1.json"', '"TSC_runtime_chain_pins_v1.json"')
source = source.replace('("train_lora_batched_tsc_v2.py", "tsc_training_inputs_v2.py", "tsc_supplemental_data_policy_v2.py")',
    '("train_lora_batched_tsc_v2.py", "tsc_training_inputs_v2.py", "tsc_supplemental_data_policy_v2.py", "prepare_TSC_quarantine_v2.py")')
begin = source.index('    science = {name: getattr(options, name)')
end = source.index('    source_run = ', begin)
source = source[:begin] + '''    science = {name: getattr(options, name) for name in inputs.COMMON_TRAINING_KEYS}
    science.update(epochs=3., steps=28083, sampling_mode="domain30", fleurs_fraction=.3, replay_seed=2026101101)
    sampling_identity = {"sampling_mode": "domain30", "replay_seed": 2026101101,
        "turkish_seed": science["seed"], "replay_rng_independent_of_turkish": True,
        "fleurs_fraction_numerator": 3, "fleurs_fraction_denominator": 10,
        "batch_size": 8, "accumulation": 2, "replay_per_update": 8}
    for name, digest in inputs.DOMAIN_SOURCES.items():
        check("actual_frozen_domain_source_"+name, sha(OUT/name) == digest)
        (scratch/name).write_bytes((OUT/name).read_bytes())
''' + source[end:]
source = source.replace('"config_identity": {"training_arguments": science,',
    '"code_sha256": inputs.DOMAIN_SOURCES,\n         "config_identity": {"training_arguments": science, "domain_sampling_identity": sampling_identity,')
source = source.replace('"child_exit_code": 0, "source_run_id": source_run}',
    '"child_exit_code": 0, "source_run_id": source_run,\n             "allow_source_sampler_transition_to_combined_uniform": True, "source_sampling_metadata": sampling_identity}')
needle = '    save(asset["closure"], proof)\n'
assert source.count(needle) == 1
source = source.replace(needle, needle + '''    declared = inputs.source_sampling_transition(json.loads(source_config.read_text(encoding="utf-8")), proof)
    check("actual_domain_to_shared_transition_declared", declared["source_mode"] == "domain30"
        and declared["source_replay_RNG_independent"] is True and declared["target_replay_RNG_independent"] is False
        and declared["replay_order_bitwise_identity_claimed"] is False)
    denied = dict(proof); denied["allow_source_sampler_transition_to_combined_uniform"] = False
    try: inputs.common_source_recipe(json.loads(source_config.read_text(encoding="utf-8")), denied)
    except ValueError: check("undeclared_domain_sampler_transition_rejected", True)
    else: raise AssertionError("Domain sampler transition accepted without ROOT approval")
    import_proof = scratch/"synthetic_import.json"; quarantine_proof = scratch/"synthetic_quarantine.json"
    save(import_proof, {"fixture_only": True}); save(quarantine_proof, {"fixture_only": True})
    approval = {"kind": "root_owned_TSC_TRAIN_candidate_approval_v2", "passed": True,
        "allow_separate_TRAIN_candidate_preparation": True,
        "import_proof": inputs.file_record(import_proof), "quarantine_proof": inputs.file_record(quarantine_proof)}
    asset["training_approval"].write_text(json.dumps(approval), encoding="utf-8")
    check("actual_approval_quarantine_binding", inputs.approve(asset["training_approval"], import_proof,
        quarantine_proof_path=quarantine_proof) == approval)
    try: inputs.approve(asset["training_approval"], import_proof)
    except ValueError: check("missing_quarantine_approval_binding_rejected", True)
    else: raise AssertionError("No-quarantine approval accepted")
''')
needle = '        outer.main(outer_args)\n'
assert source.count(needle) == 1
source = source.replace(needle, '''        exact_flags = list(flags); exact_flags[exact_flags.index("--resume-weights")] = "--resume-exact"
        original_memory_tree = ast.parse((OUT/"launch_with_memory_guard_tsc_v2.py").read_text(encoding="utf-8"))
        exact_memory_tree = ast.parse((OUT/"launch_with_memory_guard_TSC_resume_v2.py").read_text(encoding="utf-8"))
        original_nodes = {n.name:n for n in original_memory_tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        exact_nodes = {n.name:n for n in exact_memory_tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        check("exact_admission_preserves_capacity_and_telemetry_AST", all(ast.dump(original_nodes[name], include_attributes=False)
            == ast.dump(exact_nodes[name], include_attributes=False)
            for name in ("validate_capacity","argument_identity","LiveTelemetrySampler","UpdateTelemetry","safe_sample")))
        check("exact_admission_accepts_same_assets_and_opaque_state", exact_memory.tsc_argument_records(exact_flags)["resume_exact"]
            == {"path": str(checkpoint.resolve()), "sha256": sha(checkpoint)})
        try: exact_memory.tsc_argument_records(flags)
        except ValueError: check("exact_admission_rejects_model_only_warmstart", True)
        else: raise AssertionError("Warm weights admitted by exact-only route")
        try: memory.tsc_argument_records(exact_flags)
        except ValueError: check("initial_admission_rejects_exact_state", True)
        else: raise AssertionError("Exact state admitted by first-only route")
        outer.main(outer_args)
''')
needle = '        check("actual_memory_callback_and_cap8",'
assert source.count(needle) == 1
source = source.replace(needle, '''        check("quarantine_and_frozen_parent_policy_identity_pins", identity["TSC_quarantine_preparer_sha256"]
            == dependencies["prepare_TSC_quarantine_v2.py"]
            and identity["TSC_parent_supplemental_policy_sha256"] == inputs.policy.PARENT_POLICY_SHA)
''' + needle)
source = source.replace('"source_sha256": {**new_before, **dependencies},',
    '"source_sha256": {**new_before, **dependencies}, "caller_line": line,\n'
    '              "exact_memory_launcher": {"path": str(OUT / "launch_with_memory_guard_TSC_resume_v2.py"), "sha256": sha(OUT / "launch_with_memory_guard_TSC_resume_v2.py")},\n'
    '              "exact_admission_checked": True, "quarantine_source_pin_checked": True,\n'
    '              "fixture_script_path": str(Path(__file__).resolve()),')
source = source.replace('synthetic approval/source/baseline and all ML/CUDA/telemetry are explicit mocks',
    'synthetic approved domain source/quarantine/baseline, initial nested route and exact-only admission; all ML/CUDA/telemetry are explicit mocks')
destination = ROOT / "work/research/check_tsc_runtime_dispatch_v2.py"
compile(source, str(destination), "exec")
with destination.open("x", encoding="utf-8", newline="\n") as stream:
    stream.write(source)
print(str(destination))
