"""Small stdlib-only isolated TSC metadata/recipe/source guard fixture."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import wave

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0, str(HERE))
import tsc_training_inputs_v1 as helper


def main(args):
    report = args.output.resolve()
    assert report.is_relative_to(ROOT) and not report.exists()
    scratch = report.parent / (report.stem+"_fixture")
    scratch.mkdir(parents=True, exist_ok=False)
    checks = []
    def check(value, name):
        assert value, name
        checks.append(name)
    def reject(fn, name):
        try:
            fn()
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError(name)
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False)
        return helper.file_record(path)

    # No worker import: prove the unchanged actual update/sampler/gate AST.
    old = ast.parse((HERE/"train_lora_batched_data_v11.py").read_text(encoding="utf-8"))
    new = ast.parse((HERE/"train_lora_batched_tsc_v1.py").read_text(encoding="utf-8"))
    for name in ("BucketCycler", "BalancedBucketSampler", "lr_factor", "next_evaluation_epoch", "development_gate",
                 "validate_trainable_checkpoint", "annotate_cache_lengths", "save_training_state", "native_loss_function", "native_loss_identity"):
        a = next(node for node in old.body if getattr(node,"name",None)==name)
        b = next(node for node in new.body if getattr(node,"name",None)==name)
        check(ast.dump(a)==ast.dump(b), "unchanged_science_AST_"+name)
    old_run = next(node for node in old.body if getattr(node,"name",None)=="run")
    new_run = next(node for node in new.body if getattr(node,"name",None)=="run")
    update = lambda node: next(item for item in ast.walk(node) if isinstance(item,ast.For)
                              and ast.unparse(item.target)=="step" and isinstance(item.iter,ast.Call))
    check(ast.dump(update(old_run))==ast.dump(update(new_run)), "entire_optimizer_update_loop_unchanged")
    sync = [node for node in ast.walk(new_run) if isinstance(node,ast.Call) and ast.unparse(node.func)=="torch.cuda.synchronize"]
    check(len(sync)==1 and sync[0].lineno==545, "sole_exact_run_synchronize545")
    check("confirm_replay_after_cache" in ast.unparse(new_run), "replay_complete_rows_guard_present")

    # Genuine small PCM headers; labels are explicit fixture IDs, never features.
    wav = scratch/"metadata.wav"
    with wave.open(str(wav),"wb") as stream:
        stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(16000);stream.writeframes(bytes(16000))
    labels=[1,2,3]
    label_sha=hashlib.sha256(json.dumps(labels,sort_keys=True).encode()).hexdigest()
    row={"source_id":"fixture-tr","language":"tr","audio_filepath":str(wav),"duration":.5,
         "target_tokens":3,"label_ids_sha256":label_sha,"text":"fixture"}
    item=helper.metadata_item(row,labels)
    check((item["actual_PCM_frames"],item["valid_mel_frames"],item["encoder_frames"])==(8000,50,7), "actual_header_frame_formula")
    reject(lambda:helper.metadata_item({**row,"duration":.6},labels), "header_duration_mismatch_rejected")
    reject(lambda:helper.metadata_item({**row,"target_tokens":1},labels), "label_count_mismatch_rejected")
    reject(lambda:helper.metadata_item(row,[8192]), "blank_target_rejected")
    reject(lambda:helper.metadata_item(row,[8199]), "wrong_vocab_target_rejected")

    # Actual original accepted-source metadata only: no audio/checkpoints imported.
    with (ROOT/"work/data/tr_public_v3_train.jsonl").open(encoding="utf-8") as stream:
        actual_prefix=[json.loads(next(stream)) for _ in range(3)]
    for actual in actual_prefix:
        helper.policy.validate_original_source(actual)
    check(len(actual_prefix)==3, "three_real_original_prefix_source_records_valid")

    # Synthetic closed-source records (not actual closure/GPU/model evidence).
    current=json.loads((HERE/"runs/adaptive_20261011_hf_quality_window_01_stronger_kd/run_config.json").read_text(encoding="utf-8"))
    source_id="fixture_closed_uniform"
    source={"run_id":source_id,"turkish_clips_used":74883,"replay_clips_used":1595,"config_identity":current["config_identity"]}
    fixture_here=scratch/"outputs"
    helper.HERE=fixture_here
    baseline={"validation_"+lang:{"asr":{"wer":.1,"utterances":50}} for lang in helper.policy.CONFIGS}
    save(fixture_here/"results/base_v3/summary.json",baseline)
    dev={key:{"asr":dict(value["asr"])} for key,value in baseline.items()};dev["validation_tr"]["asr"]["wer"]=.5
    dev_record=save(fixture_here/"results"/(source_id+"_step_1")/"summary.json",dev)
    checkpoint=scratch/"NOT_A_MODEL_fixture_bytes.pt";checkpoint.write_bytes(b"fixture checkpoint bytes only; no tensors")
    selection={"run_id":source_id,"accepted":True,"heldout_tests_evaluated":False,"selection_split":"FLEURS validation only",
               "turkish_validation_wer":.5,"weights":str(checkpoint),"selected_step":1,"completed_steps":1}
    history=[{"step":1,"validation":{"retention_gate_passed":True,"turkish_wer":.5}}]
    closure={"kind":"root_owned_TSC_closed_source_v1","passed":True,"root_observed_owned_session_completed":True,
             "child_exit_code":0,"source_run_id":source_id,"source_checkpoint":helper.file_record(checkpoint),
             "source_selection":save(scratch/"selection.json",selection),"source_run_config":save(scratch/"config.json",source),
             "source_training_history":save(scratch/"history.json",history),"source_development_summary":dev_record}
    closure_record=save(scratch/"closure.json",closure)
    preparation={"clips":74884,"original_TR_source_ids_sha256":source["config_identity"]["tr_pool_sha256"]}
    prep=save(scratch/"inputs.json",preparation)
    options=dict(source["config_identity"]["training_arguments"])
    options.update(epochs=1.,steps=9361,source_closure=Path(closure_record["path"]),tsc_input_proof=Path(prep["path"]),
                   resume_exact=None,resume_weights=checkpoint,resume_runtime_migration=None,supplement=None,reuse_input_audit=None,
                   native_max_joint_bytes=source["config_identity"]["native_loss"]["max_joint_bytes"])
    arm=SimpleNamespace(**options)
    result=helper.closed_source_identity(arm)
    check(result["initial_optimizer_sampler_RNG_history_counters"]=="fresh", "data_arm_is_fresh_optimizer_not_exact")
    check(len(result["declared_scientific_changes"])==2,"declared_data_plus_one_pass_horizon")
    arm.steps=28082;reject(lambda:helper.closed_source_identity(arm),"old_horizon_rejected");arm.steps=9361
    arm.replay_kl=5;reject(lambda:helper.closed_source_identity(arm),"silent_KD_change_rejected");arm.replay_kl=10
    arm.resume_runtime_migration=Path("unused");reject(lambda:helper.closed_source_identity(arm),"legacy_migration_rejected");arm.resume_runtime_migration=None
    changed={**source,"turkish_clips_used":74884}
    save(scratch/"bad_config.json",changed)
    bad_closure={**closure,"source_run_config":helper.file_record(scratch/"bad_config.json")}
    save(scratch/"bad_closure.json",bad_closure)
    arm.source_closure=scratch/"bad_closure.json"
    reject(lambda:helper.closed_source_identity(arm),"same_shaped_alternate_source_pool_rejected")
    arm.source_closure=Path(closure_record["path"])
    reject(lambda:helper.approve(scratch/"closure.json",scratch/"inputs.json"),"nonapproval_receipt_rejected")
    report.parent.mkdir(parents=True,exist_ok=True)
    with report.open("x",encoding="utf-8") as stream:
        json.dump({"schema":1,"kind":"TSC_isolated_training_stdlib_source_metadata_guards_v1","passed":True,
                   "checks":len(checks),"check_names":checks,"worker_sha256":helper.policy.sha256(HERE/"train_lora_batched_tsc_v1.py"),
                   "input_helper_sha256":helper.policy.sha256(HERE/"tsc_training_inputs_v1.py"),
                   "fixture_sha256":helper.policy.sha256(__file__),"actual_TSC_inputs_generated":False,
                   "actual_closed_source_checkpoint_read":False,"Torch_NeMo_GPU_network_or_current_audio_used":False},stream,indent=2)
    print(json.dumps({"passed":True,"checks":len(checks),"report":str(report)}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    main(p.parse_args())
