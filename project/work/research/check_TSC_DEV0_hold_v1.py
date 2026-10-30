"""Tiny stdlib synthetic checks; never Popen/import ML/read actual data."""
from pathlib import Path
import argparse
import copy
import json
import sys
import uuid
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0, str(HERE))
import run_hf_TSC_DEV0_hold_v1 as hold
import verify_TSC_CLOSED_DEV0_v1 as verifier
C = hold.controller


def main(args):
    output = C.scoped(args.output)
    assert not output.exists()
    scratch = ROOT / "work/research" / ("TSC_DEV0_hold_fixture_" + uuid.uuid4().hex)
    scratch.mkdir(); checks = 0

    def check(ok):
        nonlocal checks
        assert ok; checks += 1

    def rejects(call):
        nonlocal checks
        try: call()
        except (ValueError, FileExistsError, RuntimeError): checks += 1
        else: raise AssertionError("Expected rejection")

    hold.frozen_sources(); check(True)
    check(not any(name in sys.modules for name in ("torch", "nemo", "transformers", "tokenizers")))
    original_here = C.HERE; original_hold_here = hold.HERE
    original_prepare, original_run = C.prepare, C.run_arm
    original_awake = C.guards.prior.keep_awake
    C.HERE = hold.HERE = scratch
    original_frozen = hold.frozen_sources
    hold.frozen_sources = lambda: None  # synthetic paths; actual hashes checked above
    C.guards.prior.keep_awake = lambda value: {"synthetic": True, "value": value}
    base = {"kind": "isolated_HF_TSC_guarded_quality_arm_v3", "controller_sha256": hold.CONTROLLER_SHA,
            "loop_id": "hold_before_dispatch", "run_id": "hold_before_dispatch_TSC",
            "source_identity": {"closure": {"path": "synthetic", "sha256": "0"*64},
                                "source_checkpoint": {"path": "synthetic.pt", "sha256": "1"*64}},
            "worker_arguments": ["--run-id", "hold_before_dispatch_TSC"],
            "paths": {"stop": str(scratch / "runs/hold_before_dispatch/stop.txt")}}
    C.prepare = lambda args: copy.deepcopy(base)
    dispatches = []

    def fake_run(state, deadline, args):
        # This is the position of unchanged run_arm/Popen, not a worker simulation.
        request = C.read(state["ROOT_DEV0_hold"]["path"], {})
        check(Path(state["paths"]["stop"]).read_text(encoding="utf-8") == hold.HOLD_TEXT)
        check(request["hold_created_before_run_arm_dispatch"] is True)
        check(request["original_worker_arguments_sha256"] == C.guards.argument_sha(base["worker_arguments"]))
        check(C.read(scratch / "runs/hold_before_dispatch/loop_status.json", {})["ROOT_DEV0_hold"] == state["ROOT_DEV0_hold"])
        dispatches.append(state["run_id"])
        state.update(status="quality_window_ended", child_exit_code=0, completed_steps=0, selected_step=0,
                     stop_early_reason="controller_stop_file", stop_category="controller_stop_file",
                     resource_review_required=False)
        return 0

    C.run_arm = fake_run
    args0 = SimpleNamespace(loop_id="hold_before_dispatch", window_hours=3., validate_only=False,
                            gpu_window_authorized=True, preflight_output=None)
    try:
        check(hold.main(args0) == 0)
        check(C.run_arm is fake_run and dispatches == [base["run_id"]])
        state = C.read(scratch / "runs/hold_before_dispatch/loop_status.json", {})
        hold.require_closed_zero(state); check(True)
        # A resource guard may attempt open('x') on the deliberately existing hold.
        # Preserve that exact failure and original main's failed state.
        base["loop_id"] = "resource_collision"; base["run_id"] = "resource_collision_TSC"
        base["paths"]["stop"] = str(scratch / "runs/resource_collision/stop.txt")
        def collision(state, deadline, args):
            with Path(state["paths"]["stop"]).open("x"): pass
        C.run_arm = collision; args0.loop_id = "resource_collision"
        rejects(lambda: hold.main(args0))
        check(C.run_arm is collision)
        check(C.read(scratch / "runs/resource_collision/loop_status.json", {})["status"] == "failed")
        # Validate-only executes original preflight and never creates stop/loop.
        base["loop_id"] = "validate_only"; base["run_id"] = "validate_only_TSC"
        base["paths"]["stop"] = str(scratch / "runs/validate_only/stop.txt")
        args0.loop_id = "validate_only"; args0.validate_only = True
        check(hold.main(args0) == 0 and not (scratch / "runs/validate_only").exists())
        check(C.run_arm is collision)
        for field, value in (("completed_steps",1),("child_exit_code",1),("status","failed"),
                             ("resource_review_required",True),("cooperative_stop_reason","resource")):
            changed = copy.deepcopy(state); changed[field] = value
            rejects(lambda changed=changed: hold.require_closed_zero(changed))
    finally:
        C.prepare, C.run_arm = original_prepare, original_run
        C.HERE, hold.HERE = original_here, original_hold_here
        hold.frozen_sources = original_frozen
        C.guards.prior.keep_awake = original_awake

    def sample(language, i):
        return {"source_id": f"synthetic:{language}:{i}", "sentence_id": i, "language": language,
            "split": "validation", "reference": "A test.", "hypothesis": "A test.",
            "dataset_revision": "fixture", "audio_sha256": "2"*64, "wav_sha256": "3"*64,
            "duration": 2., "confidence": {"text": "A test.", "tokens": [
                {"token_id": 17, "duration_id": 2, "piece": "test", "start_frame": 1, "duration_frames": 2}]}}
    summaries = {}
    for language in verifier.LANGUAGES:
        source = [sample(language,i) for i in range(50)]; candidate = copy.deepcopy(source)
        score = verifier.score_transcriptions(source,language=language); score.pop("details")
        summaries["validation_"+language] = {"asr":score}
        actual = verifier.compare_language(source,candidate,score,score,language)
        check(actual["exact_transcripts"] == 50 and actual["exact_emitted_paths"] == 50)
    source = [sample("tr",i) for i in range(50)]
    score = verifier.score_transcriptions(source,language="tr"); score.pop("details")
    for key,value in (("hypothesis","A TEST."),("source_id","wrong"),("reference","different"),("split","test")):
        candidate = copy.deepcopy(source); candidate[0][key] = value
        rejects(lambda candidate=candidate: verifier.compare_language(source,candidate,score,score,"tr"))
    candidate = copy.deepcopy(source); candidate[0]["confidence"]["tokens"][0]["duration_id"] = 3
    candidate[0]["confidence"]["tokens"][0]["duration_frames"] = 3
    rejects(lambda: verifier.compare_language(source,candidate,score,score,"tr"))
    candidate = copy.deepcopy(source); candidate[0]["confidence"]["tokens"][0]["token_id"] = 8192
    rejects(lambda: verifier.compare_language(source,candidate,score,score,"tr"))
    badscore = copy.deepcopy(score); badscore["wer"] = .1
    rejects(lambda: verifier.compare_language(source,source,score,badscore,"tr"))
    check(all(g["passed"] for g in verifier.retention(summaries,summaries).values()))
    changed = copy.deepcopy(summaries); changed["validation_es"]["asr"]["wer"] = .01
    rejects(lambda: verifier.retention(changed,summaries))
    rejects(lambda: verifier.verify(SimpleNamespace(root_owned_controller_exit_confirmed=False)))
    check(not any(name in sys.modules for name in ("torch", "nemo", "transformers", "tokenizers")))
    result = {"kind":"TSC_DEV0_hold_before_dispatch_and_equality_tiny_CPU_verification_v1",
        "passed":True,"checks":checks,"actual_saved_DEV_predictions_compared":False,
        "synthetic_utterance_comparisons":300,"actual_controller_main_exercised_with_mock_run_arm":True,
        "actual_GPU_or_process_execution":False,"ML_imported":False,
        "source_sha256": {name:C.sha(HERE/name) for name in
            ("run_hf_TSC_DEV0_hold_v1.py","verify_TSC_CLOSED_DEV0_v1.py","run_hf_TSC_arm_v3.py")},
        "fixture":C.record(__file__),"scratch":str(scratch)}
    with output.open("x",encoding="utf-8") as stream:json.dump(result,stream,indent=2)
    print(json.dumps({"passed":True,"checks":checks,"output":str(output)}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args())
