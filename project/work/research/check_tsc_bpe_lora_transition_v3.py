"""Small stdlib-only schema/path fixture. No actual phase1 admission or tensors."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0, str(OUT))
import tsc_bpe_lora_transition_v2 as helper
from metrics import score_transcriptions


def run():
    assertions = 0
    rejections = 0
    def check(value):
        nonlocal assertions
        assert value
        assertions += 1
    def rejected(call):
        nonlocal assertions, rejections
        try: call()
        except (ValueError, KeyError, AttributeError):
            assertions += 1; rejections += 1
        else: raise AssertionError("Invalid synthetic contract accepted")
    receipt = {"schema": 1, "kind": helper.RECEIPT_KIND, "status": "actual_CLOSED_qualified_phase1",
        "source_run_id": "synthetic_phase1_fixture", "target_run_id": "synthetic_phase2_fixture",
        "allow_new_rows_plus_encoder_LoRA": True, "declared_adaptation": helper.adaptation_declaration(),
        "original_step": 10, "acquisition_updates": 2, "chosen_step": 12, "completed_steps": 12,
        "chosen_checkpoint_role": "latest", "retention_gate_passed_at_chosen_step": True,
        "owned_actual_exits": [{"role": "trainer_controller", "exit_code": 0, "session_id": 999,
            "chunk_id": "SYNTHETIC_NOT_EXECUTION_EVIDENCE", "observed_at_utc": "synthetic_fixture_only"}],
        "artifacts": {"chosen_checkpoint": {"path": str(ROOT / "work/research/unused_chosen_fixture.pt"),
            "sha256": "0" * 64, "bytes": 1}},
        "original_artifact_paths": {"chosen_checkpoint": str(ROOT / "work/research/unused_original_fixture.pt")}}
    check(helper.validate_receipt(receipt, "synthetic_phase2_fixture") is receipt)
    for key, bad in (("kind", "wrong"), ("status", "running"), ("acquisition_updates", 0),
                     ("chosen_step", 11), ("chosen_checkpoint_role", "warm_weights"),
                     ("retention_gate_passed_at_chosen_step", False), ("allow_new_rows_plus_encoder_LoRA", False)):
        altered = copy.deepcopy(receipt); altered[key] = bad
        rejected(lambda: helper.validate_receipt(altered, "synthetic_phase2_fixture"))
    altered = copy.deepcopy(receipt); altered["owned_actual_exits"][0]["exit_code"] = 1
    rejected(lambda: helper.validate_receipt(altered, "synthetic_phase2_fixture"))
    altered = copy.deepcopy(receipt); altered["declared_adaptation"]["encoder_LoRA_learning_rate"] = 1e-4
    rejected(lambda: helper.validate_receipt(altered, "synthetic_phase2_fixture"))
    rejected(lambda: helper.validate_receipt(receipt, receipt["source_run_id"]))
    identity = {"phase": helper.PHASE1,
        "extended_tokenizer_sha256": "10b747471a6613f818445f4c5697d04bb64ca45c48eb1bde7184d33e28e68607"}
    receipt["scientific_config_identity_sha256"] = helper.canonical(identity)
    config = {"run_id": receipt["source_run_id"], "phase": helper.PHASE1,
        "source_selected_step": 10, "config_identity": identity}
    selection = {"run_id": receipt["source_run_id"], "phase": helper.PHASE1, "accepted": True,
        "heldout_tests_evaluated": False, "selection_split": "FLEURS validation only",
        "completed_steps": 12, "source_selected_step": 10, "selected_step": 10,
        "latest_weights": receipt["original_artifact_paths"]["chosen_checkpoint"],
        "weights": str(ROOT / "work/research/unused_best_fixture.pt"), "turkish_validation_wer": .1}
    gate = {"retention_gate_passed": True, "turkish_wer": .2}
    history = [{"step": i} for i in range(1, 13)]
    for i in (11, 12): history[i-1]["acquisition_updates_completed"] = i - 10
    history[-1]["validation"] = gate
    summary = {"validation_"+language: {"asr": {"utterances": 50, "wer": .2 if language == "tr" else .01}}
               for language in helper.LANGUAGES}
    old_read = helper.read
    helper.read = lambda path: {"validation_"+language: {"asr": {"wer": .1}} for language in helper.LANGUAGES}
    try:
        # Qualified acquired LATEST can have worse TR than localBEST. This is
        # schema behavior only, not a fabricated actual source or WER result.
        check(helper.validate_chosen_selection(receipt, config, selection, history, summary) == gate)
        altered = copy.deepcopy(selection); altered["latest_weights"] = altered["weights"]
        rejected(lambda: helper.validate_chosen_selection(receipt, config, altered, history, summary))
        altered = copy.deepcopy(history); altered[-1]["acquisition_updates_completed"] = 1
        rejected(lambda: helper.validate_chosen_selection(receipt, config, selection, altered, summary))
        altered = copy.deepcopy(history); altered[-1]["validation"]["retention_gate_passed"] = False
        rejected(lambda: helper.validate_chosen_selection(receipt, config, selection, altered, summary))
        altered = copy.deepcopy(summary); altered["validation_de"]["asr"]["wer"] = .2
        rejected(lambda: helper.validate_chosen_selection(receipt, config, selection, history, altered))
    finally: helper.read = old_read

    # Execute the actual phase1 worker serializer AST with opaque state/Torch stubs,
    # preserving the producer's Python tuple then its genuine JSON representation.
    import ast
    from types import SimpleNamespace
    typed_identity={**identity,"lora_layout":{"synthetic_layout_only":True},
        "domain_sampling_identity":{"synthetic_sampler_only":True},
        "source_sha256":{"frozen_fixture_source":"abc"},
        "acquisition_optimizer":{"declared_optimizer_group":{"betas":(.9,.999)}}}
    json_identity=json.loads(json.dumps(typed_identity))
    context={"receipt":receipt,"parent_identity":json_identity,"source_step":12,"original_step":10,
        "acquisition_updates":2,"source_history_prefix_sha256":helper.canonical(history),"chosen_gate":gate}
    class FakePath:
        def __init__(self,value):self.value=str(value)
        def __str__(self):return self.value
        def replace(self,target):pass
    captured={}
    def phase_state(*,runtime_state):
        return {"schema":1,"kind":helper.PHASE1_STATE_KIND,"acquisition_optimizer_updates":2,
            "runtime_state":copy.deepcopy(runtime_state),"lora_metadata":typed_identity["lora_layout"]}
    fake_torch=SimpleNamespace(get_rng_state=lambda:"CPU_OPAQUE_NOT_TENSOR",
        cuda=SimpleNamespace(get_rng_state_all=lambda:["CUDA_OPAQUE_NOT_TENSOR"]),
        save=lambda state,path:captured.update(state=state))
    tree=ast.parse((OUT/'train_lora_batched_tsc_bpe_acquire_v1.py').read_text(encoding='utf-8'))
    serializer=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='save_training_state')
    env={"Path":FakePath,"torch":fake_torch,"random":SimpleNamespace(getstate=lambda:"PYTHON_OPAQUE")}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[serializer],type_ignores=[])),str(OUT/'train_lora_batched_tsc_bpe_acquire_v1.py'),'exec'),env)
    env['save_training_state'](FakePath('opaque_never_written.pt'),None,SimpleNamespace(state_dict=phase_state),
        SimpleNamespace(state_dict=lambda:{"identity":typed_identity["domain_sampling_identity"],"updates":12}),
        run_id=receipt['source_run_id'],step=12,config_identity=typed_identity,history=history,best_score=.1,
        selected_step=10,patience=0,retention_failures=0,next_eval_epoch=.25,train_wall_seconds=1.,
        audio_seconds=1.,language_audio_seconds={'tr':1.},evaluation_wall_seconds=0.,
        initial_gate={'retention_gate_passed':True},source_selected_step=10)
    producer=captured['state']
    check(producer['config_identity']==producer['runtime_state']['config_identity']
        and producer['config_identity']!=json_identity and helper.canonical(producer['config_identity'])==helper.canonical(json_identity))
    check(helper.validate_phase1_state(producer,context)['row_Adam_clock']==2)
    import tsc_bpe_lora_transition_v1 as old_helper
    rejected(lambda:old_helper.validate_phase1_state(producer,context))
    for value in ((.999,.9),(.9,.998)):
        bad=copy.deepcopy(producer)
        for data in (bad['config_identity'],bad['runtime_state']['config_identity']):
            data['acquisition_optimizer']['declared_optimizer_group']['betas']=value
        rejected(lambda bad=bad:helper.validate_phase1_state(bad,context))
    bad=copy.deepcopy(producer)
    bad['runtime_state']['config_identity']=json_identity
    rejected(lambda:helper.validate_phase1_state(bad,context))
    for key,value in (('step',11),('run_id','wrongsource'),('source_selected_step',9),('acquisition_optimizer_updates',1)):
        bad=copy.deepcopy(producer);bad[key]=value
        rejected(lambda bad=bad:helper.validate_phase1_state(bad,context))
    bad=copy.deepcopy(producer);bad['runtime_state']['history'][-1]['validation']['retention_gate_passed']=False
    rejected(lambda:helper.validate_phase1_state(bad,context))
    bad=copy.deepcopy(producer)
    for data in (bad['config_identity'],bad['runtime_state']['config_identity']):data['source_sha256']['frozen_fixture_source']='changed'
    rejected(lambda:helper.validate_phase1_state(bad,context))

    for language in helper.LANGUAGES:
        rows=[]
        for i in range(50):
            row = {key: "fixture_"+key for key in helper.IDENTITY_KEYS}
            row.update(source_id=f"fixture_{language}_{i}", sentence_id=i, language=language, split="validation",
                reference="a", hypothesis="a", duration=1., confidence={"text": "a", "blank_steps": 1,
                "total_steps": 2, "tokens": [{"token_id": 8710, "duration_id": 1, "piece": "a",
                "start_frame": 1, "duration_frames": 1}]})
            rows.append(row)
        asr = {key:value for key,value in score_transcriptions(rows, language=language).items() if key != "details"}
        check(helper.compare_development(rows, copy.deepcopy(rows), asr, asr, language)["tokens_compared"] == 50)
        for mutation in ("hypothesis", "reference", "token_id", "duration_id"):
            bad = copy.deepcopy(rows)
            if mutation in ("hypothesis", "reference"): bad[0][mutation] = "b"
            else: bad[0]["confidence"]["tokens"][0][mutation] = 8192 if mutation == "token_id" else 2
            rejected(lambda: helper.compare_development(rows, bad, asr, asr, language))
    check("torch" not in sys.modules and "transformers" not in sys.modules and "nemo" not in sys.modules)
    report = {"schema": 1, "kind": "inactive_TSC_BPE_LoRA_boundary_stdlib_schema_verification_v3",
        "passed": True, "CPU_only": True, "assertions": assertions, "rejections": rejections,
        "source": {"path": str(OUT / "tsc_bpe_lora_transition_v2.py"),
            "sha256": hashlib.sha256((OUT / "tsc_bpe_lora_transition_v2.py").read_bytes()).hexdigest()},
        "fixture": {"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "scope": "Synthetic schema/chosen-pointer/localclock/retention/8711 token-path fixtures only; no actual source fabricated/admitted.",
        "actual_CLOSED_phase1_source_admitted": False, "actual_tensor_restoration_executed": False,
        "actual_GPU_decode_or_training_executed": False, "full_corpus_loaded": False,
        "remaining_activation_gates": ["Actual ROOT CLOSED main phase1 chosen qualifying own-state receipt",
            "Deferred injected-Torch tensor/state/clock fixture", "Actual unsuppressed chosen8711 source300 equality",
            "Actual protected rows and restored192 LoRA whole-update GPU smoke/resource review"]}
    destination = OUT / "TSC_BPE_LoRA_boundary_stdlib_verification_v3.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"report": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "assertions": assertions, "rejections": rejections, "actual_tensor_or_GPU_execution": False}))


if __name__ == "__main__": run()
