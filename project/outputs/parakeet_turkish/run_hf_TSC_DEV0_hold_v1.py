"""ROOT-only TSC DEV0 hold through the unchanged, pinned first-arm controller.

Use exactly run_hf_TSC_arm_v3's CLI. The stop file is created before its
run_arm dispatch, so the frozen worker evaluates/saves DEV0 and exits before
the first training batch. A successful hold is NOT the source equality gate;
run verify_TSC_CLOSED_DEV0_v1.py after ROOT observes this controller exit0.
Resource errors, including an existing-stop-file collision, remain failures.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

import run_hf_TSC_arm_v3 as controller

HERE = Path(__file__).resolve().parent
CONTROLLER_SHA = "7eccccd31c294e04bc962eed7cc3d0e385768386607b491064cbe21955cfc24b"
CHAIN_SHA = "bd692e2e67815cf4f8ab197d269e92b1c523ddd852d59742dad4517f2fcb234d"
HOLD_TEXT = "ROOT DEV0 hold: evaluate/save0 and exit before the first optimizer update.\n"
HOLD_NAME = "ROOT_DEV0_hold_receipt_v1.json"


def frozen_sources():
    controller.require(controller.sha(HERE / "run_hf_TSC_arm_v3.py") == CONTROLLER_SHA,
                       "Frozen TSC first controller changed")
    controller.require(controller.sha(controller.CHAIN_PATH) == CHAIN_SHA,
                       "Frozen TSC routing receipt changed")


def dispatch_hold(state, deadline, args, original_run_arm):
    """Only the dispatch boundary changes; no worker arguments are modified."""
    frozen_sources()
    loop = HERE / "runs" / state["loop_id"]
    stop = controller.scoped(state["paths"]["stop"])
    controller.require(loop.is_dir() and stop.parent == loop and not stop.exists(),
                       "Original controller must create the fresh loop; stop must be absent")
    receipt_path = loop / HOLD_NAME
    controller.require(not receipt_path.exists(), "Hold receipt must be new")
    with stop.open("x", encoding="utf-8") as stream:
        stream.write(HOLD_TEXT)
    receipt = {
        "schema": 1, "kind": "root_requested_TSC_DEV0_before_dispatch_hold_v1",
        "loop_id": state["loop_id"], "run_id": state["run_id"],
        "wrapper": controller.record(__file__), "original_controller_sha256": CONTROLLER_SHA,
        "routing_receipt": controller.record(controller.CHAIN_PATH),
        "source_closure": state["source_identity"]["closure"],
        "source_checkpoint": state["source_identity"]["source_checkpoint"],
        "original_worker_arguments_sha256": controller.guards.argument_sha(state["worker_arguments"]),
        "stop_file": controller.record(stop), "expected_completed_updates": 0,
        "hold_created_before_run_arm_dispatch": True, "created_at_utc": controller.now(),
        "wrapper_pid": os.getpid(), "worker_or_runtime_source_changed": False,
        "equality_verified": False,
        "resource_stop_file_collision_policy": "Propagate original controller failure; never mask or waive resource guards",
    }
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2); stream.write("\n")
    state["ROOT_DEV0_hold"] = controller.record(receipt_path)
    # Persist the request before original Popen; original main writes final state.
    controller.write(loop / "loop_status.json", state)
    controller.require(stop.read_text(encoding="utf-8") == HOLD_TEXT,
                       "DEV0 hold must exist BEFORE unchanged worker dispatch")
    return original_run_arm(state, deadline, args)


def main(args):
    frozen_sources()
    original = controller.run_arm
    controller.run_arm = lambda state, deadline, incoming: dispatch_hold(state, deadline, incoming, original)
    try:
        result = controller.main(args)
    finally:
        controller.run_arm = original
    if not args.validate_only and result == 0:
        state = controller.read(HERE / "runs" / args.loop_id / "loop_status.json", {})
        require_closed_zero(state)
        print(json.dumps({"DEV0_hold_closed": True, "completed_updates": 0,
                          "source_equality_verified": False,
                          "next_action": "ROOT observe exit0, then run external CLOSED DEV0 equality verifier"}), flush=True)
    return result


def require_closed_zero(state):
    controller.require(state.get("kind") == "isolated_HF_TSC_guarded_quality_arm_v3"
        and state.get("controller_sha256") == CONTROLLER_SHA
        and state.get("status") == "quality_window_ended" and state.get("child_exit_code") == 0
        and state.get("completed_steps") == 0 and state.get("selected_step") == 0
        and state.get("stop_early_reason") == "controller_stop_file"
        and state.get("stop_category") == "controller_stop_file"
        and state.get("resource_review_required") is False
        and not state.get("cooperative_stop_reason") and state.get("objective_complete") is False,
        "Successful DEV0 hold requires actual closed child0 / selected0 / no resource or training updates")


if __name__ == "__main__":
    parser = controller.parser(); parser.description = __doc__
    sys.exit(main(parser.parse_args()))
