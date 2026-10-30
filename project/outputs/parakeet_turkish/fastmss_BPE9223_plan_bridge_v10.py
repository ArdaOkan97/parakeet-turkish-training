"""Inactive source-typed full-run or one-hard-update CPU plan bridge.

This is a NEW producer protocol, not an alias for either fixed diagnostic
feature helper. No audio/runtime/cache/sampler/optimizer operation lives here.
DATA completeness is independent of native memory admission. A bridged plan
is still provisional, and cannot make PCM/features or a committed frontier.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import threading

ROOT = Path(r"E:\gpu_training_scratchpad")
PROJECT = Path(__file__).resolve().parents[2]
WEAK = "weak_label_continued_training"
CLEANER = "cleaner_data_fine_tuning"
KIND = "FastMSS_BPE9223_phase_aware_feature_window_bridge_v1"
FULL_MODE = "full_two_broad_one_cleaner"
HARD_MODE = "one_hard_envelope_diagnostic"
FULL_PURPOSE = "reviewed_full_two_broad_one_cleaner_production_CPU_preparation"
HARD_PURPOSE = "bounded_one_full_hard_envelope_diagnostic"
FULL_SESSION_KIND = "FastMSS_full_run_CPU_preparation_session_v1"
HARD_SESSION_KIND = "FastMSS_one_full_hard_envelope_diagnostic_CPU_session_v1"
PLANNED_UPDATES = 401072
INPUT_KIND = "ASR18_phase_aware_provisional_production_window_plan_v1"
DATA_KIND = "ASR18_validated_final_DATA_population_for_bounded_preparation_v1"
CAPACITY = 512 * 1024**2
METADATA = 8 * 1024**2
PCM_CAPACITY = 128 * 1024**2
PINS = {
    "hard_context": ("asr18_hard_envelope_schedule_context_v1.py", "24cb88c113800c5eede05863d51eaf33ace7a57c4fdcd3e1dd38a4e6c7f60b59"),
    "schedule_state": ("bpe_broad_production_state_v1.py", "04c4d3a1becaa74b0b6dfbfad716c8ebb6c1d542a086f804650c96410d4d0a4a"),
    "coordinator": ("bpe_production_window_coordinator_v4.py", "527ddd91b17d447269a3253dd5d3c4ad1762a1f79a93d3199ac989d127170b1a"),
    "DATA_verifier": ("bpe_final_DATA_population_v1.py", "f8536cb791e1f2e66a5a4f7565c3b569bda067e01f77f296f908ed7d12e050ee"),
    "adapter": ("fastmss_train_augmentation_v1.py", "1cbab261592846e971e1539781b1852bf9e328ef43fcded1af11691abd715ef2"),
    "draw_helper": ("fastmss_ordered_draw_binding_v1.py", "cef78a6986918ec0b542a2f4d34f4f2f5aeadead358c823203846077891e7cd5"),
    "policy": ("fastmss_broad_optin_v1.json", "07a7e671f47ff5ceceaa906a5371c7677ab21cc0079a4f164f661af114d3cbf7"),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def same(left, right):
    return left == right and digest(left) == digest(right)


def integer(value, minimum=0, maximum=None):
    require(type(value) is int and value >= minimum and (maximum is None or value <= maximum), "Exact bounded integer required")
    return value


def sha(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def purpose(mode):
    require(mode in (FULL_MODE, HARD_MODE), "Explicit source-typed mode required")
    return FULL_PURPOSE if mode == FULL_MODE else HARD_PURPOSE


def validate_session_scope(session):
    mode = session["mode"]
    require(session["purpose"] == purpose(mode) and session["production_enabled"] is False,
            "CPU preparation is unarmed; mode and purpose must agree")
    require(type(session['owner_validation_only']) is bool and type(session['checkpoint_diagnostic_only']) is bool
            and type(session['production_source_eligible']) is bool
            and session['checkpoint_diagnostic_only'] == session['owner_validation_only']
            and session['production_source_eligible'] is (not session['owner_validation_only'])
            and type(session['diagnostic_only']) is bool
            and session['diagnostic_only'] == session['checkpoint_diagnostic_only']
            and (mode != HARD_MODE or session['owner_validation_only'] is True),
            'Explicit qualification checkpoint namespace; no diagnostic-to-production promotion')
    if mode == FULL_MODE:
        require(session["kind"] == FULL_SESSION_KIND and type(session["planned_native_updates"]) is int
                and session["planned_native_updates"] == PLANNED_UPDATES
                and type(session["initial_optimizer_updates"]) is int and session["initial_optimizer_updates"] == 0
                and sha(session["initial_sampler_sha256"])
                and "maximum_windows" not in session and "maximum_native_updates" not in session
                and not any("diagnostic_context" in key or key == "hard_context_verifier_source" for key in session),
                "One immutable original0 full session; resume retains its identity/index, never renumbers a diagnostic frontier")
        require(sha(session["full_run_identity_sha256"]), "Exact durable full-run identity required")
    else:
        require(session["kind"] == HARD_SESSION_KIND and type(session["maximum_windows"]) is int
                and session["maximum_windows"] == 1 and type(session["maximum_native_updates"]) is int
                and session["maximum_native_updates"] == 1 and session["production_source_eligible"] is False
                and "targeted_cleaner_diagnostic_context" not in session and "planned_native_updates" not in session
                and "full_run_identity_sha256" not in session,
                "One hard diagnostic cannot become a full-population run or a cleaner diagnostic")
    return mode


def validate_bridge_scope(value):
    mode = value["mode"]
    require(value["kind"] == KIND and value["purpose"] == purpose(mode)
            and type(value['owner_validation_only']) is bool and type(value['checkpoint_diagnostic_only']) is bool
            and value['checkpoint_diagnostic_only'] == value['owner_validation_only']
            and type(value["diagnostic_only"]) is bool and value['diagnostic_only'] == value['checkpoint_diagnostic_only']
            and (mode != HARD_MODE or value['owner_validation_only'] is True)
            and value["production_enabled"] is False, "Exact unarmed source-typed bridge scope")
    return mode


def scoped(path, *, source=False):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT.resolve()) or (source and path.is_relative_to(PROJECT)), "E-only artifacts; existing project sources only")
    return path


def verify_file(pin, guard=lambda: None, *, maximum=None):
    require(type(pin) is dict and set(pin) in ({"path", "bytes", "sha256"}, {"path", "bytes", "sha256", "mtime_ns"}) and sha(pin["sha256"]), "Exact file binding required")
    integer(pin["bytes"], 1)
    if maximum is not None:
        require(pin["bytes"] <= maximum, "Bounded metadata/source required")
    path = scoped(pin["path"], source=True)
    before = path.stat()
    require(str(path) == pin["path"] and before.st_size == pin["bytes"], "Pinned file path/size differs")
    if "mtime_ns" in pin:
        require(type(pin["mtime_ns"]) is int and before.st_mtime_ns == pin["mtime_ns"], "Pinned file timestamp differs")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            guard()
            value.update(block)
    after = path.stat()
    require(value.hexdigest() == pin["sha256"] and (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Pinned file changed during verification")
    return path


def _direct_read_bound(pin, guard=lambda: None):
    path = verify_file(pin, guard, maximum=2 * METADATA)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate metadata key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON value: " + value)
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique, parse_constant=invalid)


def read_bound(pin, guard=lambda: None):
    if Path(pin['path']).exists():
        return _direct_read_bound(pin, guard)
    from asr18_BPE9223_operational_metadata_v2 import read_bound as archived_read
    return archived_read(pin, guard)


def pin_source(pin, key, guard):
    expected_name, expected_sha = PINS[key]
    require(Path(pin["path"]).resolve() == Path(__file__).with_name(expected_name).resolve() and pin["sha256"] == expected_sha, "Frozen interface source differs: " + key)
    verify_file(pin, guard, maximum=256 * 1024)


def frozen_augmentation(identity, guard):
    """Metadata-only exact NEW recorded-noise/16RIR policy, no old active pilot."""
    import fastmss_BPE9223_recorded_room_train_v3 as augmentation
    _, actual = augmentation.policy_identity(identity['policy_file'], guard)
    require(same(actual, identity), 'Exact NEW9223 recorded-room policy/runtime/source identity')
    return deepcopy(identity)

def actual0(pin, report, guard, *, owner):
    proof = read_bound(pin, guard)
    require(proof.get("owner") == owner, "Actual tool closure owner differs")
    if proof.get("kind") == "ASR18_source_bound_actual_tool_exit_v1":
        require(type(proof["actual_exit_code"]) is int and proof["actual_exit_code"] == 0 and same(proof["published_marker"], report), "Actual producer0 must bind exact published metadata")
        pid = proof["native_PID"]
    else:
        require(proof.get("kind") == "ASR18_guarded_CPU_tool_completion_v1" and same(proof["report"], report), "Exact guarded metadata tool closure required")
        require(type(proof["actual_native_exit_code"]) is int and proof["actual_native_exit_code"] == 0 and type(proof["actual_exec_tool_exit_code"]) is int and proof["actual_exec_tool_exit_code"] == 0, "Native and tool must actually close0")
        monitor = read_bound(proof["guard_closure"], guard)
        pid = proof["native_PID"]
        require(monitor["pid"] == pid and type(monitor["actual_exit_code"]) is int and monitor["actual_exit_code"] == 0 and monitor["monitor_clean"] is True and monitor["monitor_joined"] is True and monitor["observed"]["monitor_error"] is None, "Actual clean joined producer monitor required")
        integer(monitor["observed"]["samples"], 3)
    integer(pid, 1)
    require(type(proof.get("actual_tool_chunk_id")) is str and proof["actual_tool_chunk_id"], "Observed tool chunk required; marker alone is insufficient")
    return proof


def observed_native0(value, guard=lambda: None):
    """Typed observed process or outer-tool result; scientific state verified separately."""
    if value.get("kind") == "ASR18_owned_GPU_process_completion_v1":
        from fastmss_owned_process_closure_v2 import verify
        return verify(value, kind=value["kind"], owner="/root/memory_diagnosis", guard=guard)
    require(value.get("owner") == "/root/memory_diagnosis" and type(value.get("actual_native_exit_code")) is int and value["actual_native_exit_code"] == 0, "Actual native-owner exit0 required before next window")
    tool = value.get("actual_tool_exit_code", value.get("actual_exec_tool_exit_code"))
    require(type(tool) is int and tool == 0 and type(value.get("actual_tool_chunk_id")) is str and value["actual_tool_chunk_id"], "Actual observed tool0 required; status marker insufficient")
    integer(value.get("native_PID"), 1)


def journal_prefix(value):
    require(type(value) is dict and set(value) == {"path", "offset", "sha256"} and sha(value["sha256"]), "Exact caller-owned committed journal prefix required")
    scoped(value["path"])
    integer(value["offset"])
    return value


def observed_cleanup0(value, guard):
    if value.get("kind") == "ASR18_owned_CPU_process_completion_v1":
        from fastmss_owned_process_closure_v2 import verify
        require(value.get("owner") in ("/root/memory_diagnosis", "/root/fastmss_integration"), "Owned cleanup parent required")
        return verify(value, kind=value["kind"], owner=value["owner"], guard=guard)
    if value.get("kind") == "ASR18_guarded_CPU_tool_completion_v1":
        observed_native0(value)
        monitor = read_bound(value["guard_closure"], guard)
        pid = value["native_PID"]
    else:
        require(value.get("owner") in ("/root/memory_diagnosis", "/root/fastmss_integration") and type(value.get("actual_tool_exit_code")) is int and value["actual_tool_exit_code"] == 0 and type(value.get("actual_tool_chunk_id")) is str and value["actual_tool_chunk_id"], "Actual owned cleanup tool0 required")
        monitor = read_bound(value["actual_guard"], guard)
        pid = integer(value["actual_native_pid"], 1)
    require(monitor["pid"] == pid and type(monitor["actual_exit_code"]) is int and monitor["actual_exit_code"] == 0 and monitor["monitor_joined"] is True and monitor["monitor_clean"] is True and monitor["observed"]["monitor_error"] is None, "Clean joined actual cleanup closure required")
    integer(monitor["observed"]["samples"], 3)


def validate_DATA(evidence, guard):
    """NEW9223 population audit; original source/heldout facts remain bound."""
    from asr18_BPE9223_source_context_v7 import validate_CPU_population
    data = validate_CPU_population(evidence, guard=guard)
    require(type(data) is dict and data['retained_rows'] == 2631115
        and data['total_target_tokens'] == 90746310 and data['population_mean_U'] == 90746310 / 2631115,
        'Complete9223 broad denominator, no partial/full8711 substitution')
    require(data['vocabulary'] == 9223 and data['blank_id'] == 8192,
        'Explicit9223 TR labels; replay retains original8193 cached IDs')
    return data

def full_targets(row):
    language = row["language"]
    ids = row["tr_metadata_labels"] if language == "tr" else row["clean_target_ids"]
    require(type(ids) is list and ids and all(type(v) is int and 0 <= v < (9223 if language == "tr" else 8193) and v != 8192 for v in ids), "Every original target ID required; no cap or prefix")
    require(type(row["target_tokens"]) is int and row["target_tokens"] == len(ids), "Full row target count differs")
    return ids, hashlib.sha256(b"".join(struct.pack("<I", v) for v in ids)).hexdigest()


def normalize_updates(plan, *, population, guard=lambda: None, mode=FULL_MODE,
                      hard_context_binding=None, hard_context_source=None):
    """Pure metadata transformation; not DATA validation or preparation admission."""
    integer(population, 1)
    require(plan["kind"] == INPUT_KIND and plan["production_enabled"] is False and plan["phase"] in (WEAK, CLEANER), "Exact inactive phase-aware coordinator plan required")
    phase = plan["phase"]
    for key in ("actual_optimizer_updates", "actual_authoritative_draws_consumed"):
        require(type(plan[key]) is int and plan[key] == 0, "Prepared plan cannot claim a commit")
    require(plan["actual_replay_frontiers_advanced"] is False and plan["source_optimizer_or_RNG_reset"] is False and plan["capacity_bytes"] == CAPACITY and plan["metadata_reserve_bytes"] == METADATA, "Unchanged clocks/RNG and hard capacities required")
    updates = plan["updates"]
    require(type(updates) is list and 1 <= len(updates) <= 20 and sha(plan["initial_sampler_sha256"]), "One bounded nonempty window required")
    initial = integer(plan["initial_committed_updates"])
    final = plan["final_private_planning_sampler"]
    purpose(mode)
    if mode == HARD_MODE:
        require(phase == WEAK and initial == 0 and len(updates) == 1
                and same(plan.get("hard_envelope_diagnostic_context"), hard_context_binding)
                and hard_context_binding is not None and hard_context_source is not None,
                "Exactly one explicitly bound original-source hard diagnostic before any audio")
        pin_source(hard_context_source, "hard_context", guard)
        from asr18_hard_envelope_schedule_context_v1 import verify_bridge_state
        checked = verify_bridge_state(plan, hard_context_binding, population=population, guard=guard)
        require(same(checked, {"passed": True, "mode": HARD_MODE, "production_source_eligible": False,
                "maximum_windows": 1, "maximum_native_updates": 1, "TR_rows": 14, "replay_rows": 2,
                "population_pass_or_DEV_barrier_claimed": False}), "Exact hard-context verifier protocol")
    else:
        require(hard_context_binding is None and hard_context_source is None
                and not any("diagnostic_context" in key for key in plan),
                "Full-population production branch rejects all diagnostic context pins")
        require(final["kind"] == "ASR18_two_broad_then_one_cleaner_schedule_v1" and final["broad_cursor"]["population"] == population and type(final["broad_cursor"]["population"]) is int and final["broad_cursor"]["passes"] == 2 and type(final["broad_cursor"]["passes"]) is int and final["cleaner_cursor"]["population"] == 352768 and type(final["cleaner_cursor"]["population"]) is int and final["cleaner_cursor"]["passes"] == 1 and type(final["cleaner_cursor"]["passes"]) is int, "Private plan must preserve full broad and exact original cleaner populations/passes")
        integer(final["broad_cursor"]["draws"], 0, 2 * population)
        integer(final["cleaner_cursor"]["draws"], 0, 352768)
        name, expected_sha = PINS["schedule_state"]
        source = Path(__file__).with_name(name)
        pin_source({"path": str(source.resolve()), "bytes": source.stat().st_size, "sha256": expected_sha}, "schedule_state", guard)
        from bpe_broad_production_state_v1 import permutation
    require(type(final["population_mean_U"]) is dict and set(final["population_mean_U"]) == {WEAK, CLEANER} and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in final["population_mean_U"].values()), "Exact positive typed phase means required; no bool mean")
    if mode == FULL_MODE:
        require(final["cleaner_cursor"]["draws"] == 0 or final["broad_cursor"]["draws"] == 2 * population, "No cleaner phase before both actual broad pass frontiers")
    require(type(final["carried_replay"]["updates"]) is int and final["carried_replay"]["updates"] == initial + len(updates) and final["population_mean_U"][CLEANER] == 10621773 / 352768, "Carried replay clock/full cleaner target denominator differs")
    previous, result, total = plan["initial_sampler_sha256"], [], 0
    first_epoch, next_TR_draw = None, None
    for index, original in enumerate(updates):
        guard()
        require(original["local_update_index"] == index and type(original["local_update_index"]) is int and type(original["global_update"]) is int and original["global_update"] == initial + index + 1 and original["phase"] == phase and original["prepared_is_committed"] is False, "Ordered phase/local/global update identity differs")
        require(type(original["batch_id"]) is str and original["batch_id"] == "update_" + str(original["global_update"]), "Actual caller global batch ID required")
        require(all(sha(original[k]) for k in ("native_transaction_sha256", "before_sampler_sha256", "after_sampler_sha256")) and original["before_sampler_sha256"] == previous, "Contiguous exact sampler/transaction hash chain required")
        tr, replay, rows, contexts, draws = (original[k] for k in ("TR_draws", "replay_rows", "rows", "contexts", "draws"))
        require(all(type(v) is list for v in (tr, replay, rows, contexts, draws)) and 1 <= len(tr) <= 14 and len(replay) == 2 and len(rows) == len(contexts) == len(draws) == len(tr) + 2 and len(original["TR_identity"]) == len(tr), "Full1..14 TR tail plus exactly2 original carried replay required")
        require([v["language"] for v in rows[:-2]] == ["tr"] * len(tr) and rows[-2]["language"] in ("en", "es", "it", "fr", "de") and rows[-1]["language"] == "pt", "Protected old-five/PT replay ordering differs")
        if mode == FULL_MODE:
            cursor = final["broad_cursor"] if phase == WEAK else final["cleaner_cursor"]
            start = integer(tr[0]["draw_index"])
            epoch = start // cursor["population"]
            if first_epoch is None:
                first_epoch, next_TR_draw = epoch, start
            count = min(14, cursor["population"] - start % cursor["population"])
            expected = [{"ordinal": permutation((start + j) % cursor["population"], cursor["population"], [cursor["seed"], epoch]) + 1,
                         "epoch": epoch, "draw_index": start + j} for j in range(count)]
            require(start == next_TR_draw and epoch == first_epoch and same(tr, expected),
                    "Exact original permutation/full14-or-tail, no within-window broad-pass crossing or dropped draws")
            next_TR_draw += count
        require(len({(v["epoch"], v["draw_index"]) for v in contexts}) == len(contexts), "Duplicate per-update draw context")
        entries = []
        for row_index, (row, context, draw) in enumerate(zip(rows, contexts, draws)):
            require(type(row) is dict and type(context) is dict and set(context) == {"source_id", "language", "epoch", "draw_index", "phase", "split"} and context["phase"] == phase and context["split"] == "train", "Explicit original TRAIN phase/context required; no DEV/TEST/teacher routing")
            integer(context["epoch"]); integer(context["draw_index"])
            require(type(row["source_id"]) is str and row["source_id"] and row["source_id"] == context["source_id"] == draw["source_id"] and row["language"] == context["language"] == draw["language"] and all(same(draw[k], context[k]) for k in ("epoch", "draw_index", "phase")), "Original row/source/language/order/draw differs")
            require(type(row["text"]) is str, "Original supplied transcript required")
            ids, ids_sha = full_targets(row)
            is_tr = row_index < len(tr)
            require(row.get("augmentation_role", "student_tr" if is_tr else "replay") == ("student_tr" if is_tr else "replay"), "Teacher/evaluation rows cannot enter an augmentation/student input route")
            if "audio_draw_context" in row:
                require(same(row["audio_draw_context"], context), "Existing immutable row context differs; never overwrite it")
            if is_tr:
                identity, source_draw = original["TR_identity"][row_index], tr[row_index]
                integer(source_draw["ordinal"], 1, population if phase == WEAK else 352768)
                require(identity["source_id"] == row["source_id"] and identity["language"] == "tr" and identity["ordinal"] == source_draw["ordinal"] == draw["ordinal"] and all(same(source_draw[k], context[k]) for k in ("epoch", "draw_index")), "Exact full-population TR ordinal/draw identity differs")
                require(context["epoch"] in ((0, 1) if phase == WEAK else (0,)), "No extra corpus pass")
                if row_index:
                    require(context["epoch"] == contexts[row_index - 1]["epoch"] and context["draw_index"] == contexts[row_index - 1]["draw_index"] + 1, "A native update cannot cross or drop a pass tail")
                if phase == CLEANER:
                    require(type(row.get("cleaner_original_ordinal")) is int and row["cleaner_original_ordinal"] == draw["ordinal"], "Original cleaner ordinal required")
            else:
                require(draw["ordinal"] is None and row["source_id"] == replay[row_index - len(tr)]["source_id"] and row["language"] == replay[row_index - len(tr)]["language"] and context["epoch"] == tr[-1]["epoch"] and context["draw_index"] == tr[-1]["draw_index"] + 1 + row_index - len(tr), "Original carried replay cannot be selected/reseeded by bridge")
            entries.append({"row_index": row_index, "original_row_sha256": digest(row), "source_id": row["source_id"], "language": row["language"], "context": deepcopy(context), "full_target_ids": deepcopy(ids), "target_ids_u32_sha256": ids_sha, "raw_text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(), "role": "student_tr" if is_tr else "replay", "route": "frozen_weak_TR_augmentation_CPU_only" if is_tr and phase == WEAK else "direct_original_clean", "independent_clean_PCM_still_required": True})
        size = integer(original["feature_bytes_upper_bound"], 1, CAPACITY - METADATA)
        total += size
        stage_population = population if phase == WEAK else 352768
        ends_pass = mode == FULL_MODE and (tr[-1]["draw_index"] + 1) % stage_population == 0
        require(not ends_pass or index == len(updates) - 1,
                "Actual population pass tail ends this window before preparation; next pass waits coherent checkpoint and complete serial DEV")
        result.append({"update_index": index, "global_update": original["global_update"], "phase": phase, "batch_id": original["batch_id"], "coordinator_update": deepcopy(original), "coordinator_update_sha256": digest(original), "rows": entries, "feature_bytes_upper_bound": size, "prepared_is_committed": False, "ends_population_pass": ends_pass, "mandatory_fixed_DEV_checkpoint_barrier_after_update": ends_pass})
        previous = original["after_sampler_sha256"]
    require(total == plan["planned_payload_upper_bound_bytes"] and total + METADATA <= CAPACITY and digest(plan["final_private_planning_sampler"]) == previous, "Full planned feature capacity/final sampler mismatch")
    if mode == FULL_MODE:
        cursor = final["broad_cursor"] if phase == WEAK else final["cleaner_cursor"]
        require(cursor["draws"] == next_TR_draw, "Final active-cursor frontier differs from all ordered TR draws")
        committed = 0
        for cursor in (final["broad_cursor"], final["cleaner_cursor"]):
            passes, prefix = divmod(cursor["draws"], cursor["population"])
            require(prefix % 14 == 0, "Actual full/tail optimizer frontier required")
            committed += passes * ((cursor["population"] + 13) // 14) + prefix // 14
        require(committed == initial + len(updates) <= PLANNED_UPDATES,
                "Exact two-broad/one-cleaner optimizer clocks, not finite-budget resets")
    require(len(canonical(result)) <= METADATA, "Bounded copied full metadata; never truncate or exclude a row")
    return result


def read_durable_head(pin, guard):
    """One immutable <=64KiB range; never hash the growing durable stream."""
    require(type(pin) is dict and set(pin) == {"kind", "path", "offset", "framed_bytes", "sha256", "sequence"}
            and pin["kind"] == "ASR18_logical_immutable_durable_head_descriptor_v2" and sha(pin["sha256"]),
            "Source-bound packed durable head descriptor required")
    offset = integer(pin["offset"]); framed = integer(pin["framed_bytes"], 41, 65536)
    sequence = integer(pin["sequence"], 1)
    with scoped(pin["path"]).open("rb") as stream:
        stream.seek(offset); header = stream.read(40); raw = stream.read(framed - 40)
    guard()
    require(len(header) == 40 and len(raw) == framed - 40 and struct.unpack("<Q", header[:8])[0] + 40 == framed
            and hashlib.sha256(raw).digest() == header[8:] and hashlib.sha256(raw).hexdigest() == pin["sha256"],
            "Exact complete logical frame and hash")
    value = json.loads(raw)
    head = value["head"]
    require(value["kind"] == "ASR18_packed_durable_event_and_head_record_v2"
            and type(value["sequence"]) is int and value["sequence"] == sequence
            and head["kind"] == "ASR18_packed_durable_full_run_lifecycle_head_v2"
            and type(head["sequence"]) is int and head["sequence"] == sequence
            and same(head["owner"], value["owner"]) and head["identity_sha256"] == value["identity_sha256"]
            and head["production_enabled"] is False and head["TEST_complete"] is False and head["export_complete"] is False
            and type(head["total_event_bytes"]) is int and head["total_event_bytes"] == offset + framed,
            "Exact owner/identity/frontier and untouched TEST/export")
    return value


def bridge_plan(coordinator_plan_binding, *, DATA_evidence, session_binding, window_index, prior_window_boundary=None, verify_prior_boundary=None, guard=lambda: None):
    require(threading.current_thread() is threading.main_thread(), "Serial immutable bridge only")
    session = read_bound(session_binding, guard)
    mode = validate_session_scope(session)
    from fastmss_BPE9223_resource_CPU_boundary_v2 import configure_session, verify_resource_boundary
    configure_session(session, guard)
    integer(window_index, 0, 0 if mode == HARD_MODE else PLANNED_UPDATES - 1)
    pin_source(session["coordinator_source"], "coordinator", guard)
    context_source = session['source_context_source']
    require(Path(context_source['path']).resolve() == Path(__file__).with_name('asr18_BPE9223_source_context_v7.py').resolve(), 'Exact NEW9223 context source path')
    verify_file(context_source, guard, maximum=256*1024)
    require(session['vocabulary'] == 9223 and same(session['label_population_contract'], DATA_evidence['BPE9223_population']), 'Session and9223 label population binding differ')
    augmentation_identity = frozen_augmentation(session["augmentation_identity"], guard)
    read_bound(session["scientific_source_state"], guard)
    require(session["preserve_original_all207_threeAdam_allRNG_and_replay_lineage"] is True, "Native source-state carry is mandatory; bridge never initializes or resets it")
    require(session["DATA_evidence_sha256"] == digest(DATA_evidence), "Session bound to another DATA result")
    data = validate_DATA(DATA_evidence, guard)
    plan = read_bound(coordinator_plan_binding, guard)
    require(same(plan["finalized_population"], DATA_evidence["BPE9223_population"]), "Coordinator and DATA verifier use different populations")
    hard_context = plan.get("hard_envelope_diagnostic_context")
    hard_source = session.get("hard_context_verifier_source")
    if mode == HARD_MODE:
        require(window_index == 0 and prior_window_boundary is None and hard_context is not None
                and same(hard_context, session["hard_envelope_diagnostic_context"]),
                "Same plan/session one-window diagnostic context; never a production state alias")
    else:
        require(hard_context is None and hard_source is None and data["retained_rows"] == 2631115,
                "Full original two-broad/one-cleaner population without diagnostic contexts")
    normalized = normalize_updates(plan, population=data["retained_rows"], guard=guard, mode=mode,
                                   hard_context_binding=hard_context, hard_context_source=hard_source)
    require(plan["final_private_planning_sampler"]["population_mean_U"][WEAK] == data["population_mean_U"], "Plan uses a partial or different broad target denominator")
    initial = integer(session["initial_optimizer_updates"])
    require(sha(session["initial_sampler_sha256"]) and plan["initial_committed_updates"] >= initial
            and plan["initial_committed_updates"] + len(normalized) <= (1 if mode == HARD_MODE else PLANNED_UPDATES),
            "Declared original-source absolute native clock budget, no resets or finite-session extension")
    require(mode != HARD_MODE or initial == 0, "Hard diagnostic starts at original scientific source0")
    if window_index == 0:
        require(prior_window_boundary is None and plan["initial_committed_updates"] == initial and plan["initial_sampler_sha256"] == session["initial_sampler_sha256"], "First window preserves exact original source state")
    else:
        require(mode == FULL_MODE and prior_window_boundary is not None and callable(verify_prior_boundary), "Every later full-run window requires actual current coherent checkpoint and retired predecessor")
        boundary = read_bound(prior_window_boundary, guard)
        require(boundary["kind"] == "FastMSS_actual_coherent_full_run_CPU_window_boundary_v2" and same(boundary["session"], session_binding) and boundary["completed_window_index"] == window_index - 1 and type(boundary["completed_window_index"]) is int, "Exact scalar prior same-session full-run window boundary required")
        prior = read_bound(boundary["bridge"], guard)
        resource = 'actual_native_resource_closure' in boundary
        require(resource or not any(key in boundary for key in ('actual_fresh_recovery_process0','resource_recovery_report','resource_recovery_ack')),
                'Resource recovery evidence cannot decorate a normal/live boundary')
        prior_session = session['original_session'] if resource and window_index == 1 else session_binding
        merged_transition = window_index == 3 and plan['initial_committed_updates'] == 60 and not same(prior['session'], session_binding)
        if merged_transition:
            previous_pin = session['prior_operational_session']
            require(previous_pin['sha256'] == '7891b2f228fb1d88ff13955e2bc4b69491218b05d30f98fef1d45bf751fe9a89'
                and Path(previous_pin['path']).resolve() == ROOT / 'analysis/BPE9223_training_launch_v1/operational_CPU_configuration_v2/operational_CPU_session.json',
                'Only the genuine session2 predecessor at exact source60/window3')
            previous_session = read_bound(previous_pin, guard)
            validate_session_scope(previous_session)
            for key in ('full_run_identity_sha256', 'initial_optimizer_updates', 'initial_sampler_sha256',
                    'label_population_contract', 'DATA_evidence_sha256', 'augmentation_identity',
                    'scientific_source_state', 'durable_owner', 'planned_native_updates',
                    'owner_validation_only', 'checkpoint_diagnostic_only', 'production_source_eligible',
                    'original_session', 'operational_metadata'):
                require(same(previous_session[key], session[key]), 'Merged-role transition cannot change science/read authority: ' + key)
            require(prior['bridge_source_sha256'] == 'ee85f8af3ed68a6aa3ad2930b42b9fb59ddb1aaa15bf3c509d63a4c2faaa833a',
                'Prior source60 bridge keeps its actual original7 source identity')
            prior_session = previous_pin
        require(not resource or window_index == 1 and boundary['optimizer_updates'] == 20,
                'Resource transition is only the explicit stopped p1/current20 handoff')
        require(prior["kind"] == KIND and prior["mode"] == FULL_MODE and prior["window_index"] == window_index - 1 and type(prior["window_index"]) is int and same(prior["session"], prior_session) and same(prior["DATA_evidence"], DATA_evidence), "No switched DATA/session/phase provenance; only reviewed original-session resource lineage")
        require(boundary["committed_window_updates"] == len(prior["updates"]) and type(boundary["committed_window_updates"]) is int and boundary["optimizer_updates"] == plan["initial_committed_updates"] == prior["initial_committed_updates"] + len(prior["updates"]) and boundary["sampler_state_sha256"] == plan["initial_sampler_sha256"] == prior["final_sampler_sha256"], "Window2 must continue exact actual prior prefix")
        pointer = read_bound(boundary["coherent_checkpoint_pointer"], guard)
        require(type(pointer['diagnostic_only']) is bool
                and pointer['diagnostic_only'] == session['checkpoint_diagnostic_only'],
                'Same qualification/production checkpoint scope; never promote a diagnostic restore')
        durable = read_durable_head(boundary["durable_head"], guard)
        head = durable["head"]
        require(head["identity_sha256"] == session["full_run_identity_sha256"]
                and same(head["owner"], session["durable_owner"])
                and same(head["lifecycle"]["last_pointer"], boundary["coherent_checkpoint_pointer"])
                and head["lifecycle"]["last_sampler_sha256"] == plan["initial_sampler_sha256"],
                "Current source-bound durable logical head and actual prior checkpoint/sampler must agree")
        read_bound(session["durable_owner"], guard)
        if prior["updates"][-1]["ends_population_pass"]:
            completed = head["lifecycle"]["completed_boundaries"]
            expected_count = prior["updates"][-1]["rows"][0]["context"]["epoch"] + 1 if prior["phase"] == WEAK else 3
            require(type(completed) is list and len(completed) == expected_count
                    and same(completed[-1]["checkpoint_pointer"], boundary["coherent_checkpoint_pointer"])
                    and head["last_recovery"] is not None and type(head["pending"]["publications"]) is int
                    and head["pending"]["publications"] == 0 and type(head["pending"]["windows"]) is int
                    and head["pending"]["windows"] == 0,
                    "Actual pass-tail current recovery and complete serial350DEV durable acknowledgement before next-pass preparation")
        require(sum(key in boundary for key in ('actual_native_closed', 'resident_native_boundary', 'actual_native_resource_closure')) == 1,
                'Exactly one truthful normal exit, live idle or explicit abnormal recovered boundary')
        if resource:
            require(verify_resource_boundary(boundary, pointer=pointer, session=session, guard=guard) is True
                    and same(head['last_recovery'], boundary['resource_recovery_ack'])
                    and type(head['pending']['publications']) is int and head['pending']['publications'] == 0
                    and type(head['pending']['windows']) is int and head['pending']['windows'] == 0,
                    'Actual resource quiescence + fresh current recovery + full acknowledged durable prefix')
        elif "resident_native_boundary" in boundary:
            from fastmss_owned_process_closure_v2 import verify_resident_idle
            verify_resident_idle(read_bound(boundary["resident_native_boundary"], guard), pointer=pointer, proof_binding=boundary["resident_native_boundary"], phase=prior["phase"], guard=guard)
        else:
            observed_native0(read_bound(boundary["actual_native_closed"], guard), guard)
        require(pointer["kind"] == "ASR18_production_window_completed_checkpoint_pointer_v2" and pointer["all_window_updates_committed"] is True and pointer["committed_updates"] == boundary["optimizer_updates"] and type(pointer["committed_updates"]) is int and pointer["committed_window_updates"] == len(prior["updates"]) and type(pointer["committed_window_updates"]) is int and pointer["writer_returned_and_payload_verified"] is True, "Only final whole-window returned coherent checkpoint may release next window")
        scientific = pointer["identity"]["window"]
        require(scientific["plan_sha256"] == prior["coordinator_plan_sha256"] and scientific["last_transaction_sha256"] == prior["updates"][-1]["coordinator_update"]["native_transaction_sha256"] and scientific["sampler_state_sha256"] == boundary["sampler_state_sha256"] == pointer["sampler_state_sha256"], "Final checkpoint identity must bind exact prior plan/last transaction/sampler")
        semantic = pointer["saved_semantic_state_verified"]
        require(semantic["passed"] is True and semantic["saved_identity_frontier_sampler_and_journal_verified_before_publication"] is True and semantic["identity_sha256"] == digest(pointer["identity"]) and semantic["optimizer_updates"] == pointer["committed_updates"] and semantic["sampler_state_sha256"] == pointer["sampler_state_sha256"] and same(semantic["journal_prefix"], pointer["journal_prefix"]), "Saved actual semantic proof required, not pointer counters alone")
        metadata = read_bound(pointer["checkpoint"], guard)
        require(same(metadata["payload"], pointer["payload"]), "Final pointer and immutable returned checkpoint payload differ")
        cleanup_actual = read_bound(boundary["owned_cleanup_actual_closed"], guard)
        observed_cleanup0(cleanup_actual, guard)
        cleanup = read_bound(boundary["owned_cleanup_receipt"], guard)
        archived = read_bound(boundary["preserved_metadata"], guard)
        require(cleanup["passed"] is True and cleanup["feature_files_only_deleted"] is True and cleanup["original_audio_source_checkpoint_journal_untouched"] is True and same(cleanup["preserved_metadata"], boundary["preserved_metadata"]) and same(archived["owner_binding"], cleanup["owner"]) and same(archived["manifest_binding"], cleanup["manifest"]), "Actual exact owned retirement with durable original metadata required")
        require(type(cleanup["removed_exact_owned_files"]) is list and 0 < len(cleanup["removed_exact_owned_files"]) <= 704, "Exact finite owned retirement list required")
        require(boundary["originals_checkpoints_and_journals_untouched"] is True and boundary["owned_feature_window_retired"] is True and verify_prior_boundary(deepcopy(boundary)) is True, "Caller must verify actual source/state/closure/cleanup; existence or a dormant supervisor is insufficient")
    result = {"kind": KIND, "purpose": purpose(mode), "mode": mode, "production_enabled": False, "diagnostic_only": session["checkpoint_diagnostic_only"], "owner_validation_only": session["owner_validation_only"], "checkpoint_diagnostic_only": session["checkpoint_diagnostic_only"], "bridge_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "coordinator_plan": coordinator_plan_binding, "coordinator_plan_sha256": digest(plan), "session": session_binding, "window_index": window_index, "DATA_evidence": deepcopy(DATA_evidence), "DATA_rows": data["retained_rows"], "DATA_full_target_tokens": data["total_target_tokens"], "DATA_population_mean_U": data["population_mean_U"], "augmentation_identity": augmentation_identity, "scientific_source_state": session["scientific_source_state"], "phase": plan["phase"], "initial_committed_updates": plan["initial_committed_updates"], "initial_sampler_sha256": plan["initial_sampler_sha256"], "final_sampler_sha256": normalized[-1]["coordinator_update"]["after_sampler_sha256"], "updates": normalized, "prior_window_boundary": prior_window_boundary, "maximum_feature_window_bytes": CAPACITY, "metadata_reserve_bytes": METADATA, "raw_PCM_capacity_bytes": PCM_CAPACITY, "raw_PCM_maximum_jobs": 2, "feature_window_lock_required": True, "actual_optimizer_updates": 0, "actual_authoritative_draws_consumed": 0, "actual_replay_frontiers_advanced": False, "actual_clean_PCM_or_features_ready": False, "native_forecast_approved": False, "lifecycle_contract": {"NEW_phase_aware_sink_required": True, "must_not_spoof_fixed_diagnostic_plan_kind": True, "actual_closed0_before_GPU": True, "consume_once_ordered_rows": True, "prepared_never_commits": True, "intermediate_checkpoint_keeps_future_features": True, "cleanup_only_final_coherent_or_explicit_discard_with_actual_children_closed": True, "archive_small_metadata_before_exact_owned_deletion": True, "no_original_or_checkpoint_or_journal_deletion": True, "teacher_replay_DEV_TEST_clean": True, "over120s_retained_clean_via_frozen_policy": True, "runtime_whole_audio_geometry_heldout_check_before_consumed_draw": True}}
    result['lifecycle_contract']['persistence_idle_is_not_GPU_restore_or_process_exit'] = True
    result['lifecycle_contract']['recovery_scope'] = ('EVERY_DIAGNOSTIC_CAPTURE_BEFORE_OVERALL_PASS' if mode == HARD_MODE
        else 'CURRENT_LATEST_ONLY_WITH_VERIFIED_DURABLE_PERSISTENCE_PREFIX_BEFORE_PASS_DEV_WINNER_ADVANCE')
    require(len(canonical(result)) <= 2 * METADATA, "Bounded bridge metadata; no whole-corpus order or audio pool")
    return result


def actual_reference0(pin, report, guard):
    """Independent PCM alone may use genuine inner owned-Popen completion."""
    proof = read_bound(pin, guard)
    if proof.get("kind") == "ASR18_owned_CPU_process_completion_v1":
        from fastmss_owned_process_closure_v2 import verify
        return verify(proof, kind=proof["kind"], owner="/root/memory_diagnosis", report=report, guard=guard)
    return actual0(pin, report, guard, owner="/root/memory_diagnosis")

def bind_verified_update_inputs(bridge_binding, update_index, *, independent_PCM_proof, actual_closed0, guard=lambda: None):
    """Attach independent original clean PCM/full-target proofs; no audio read."""
    bridge = read_bound(bridge_binding, guard)
    validate_bridge_scope(bridge)
    require(bridge["kind"] == KIND and bridge["purpose"] == purpose(bridge["mode"]) and bridge["production_enabled"] is False and bridge["bridge_source_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "Exact unarmed source-bound phase-aware bridge required")
    integer(update_index, 0, len(bridge["updates"]) - 1)
    update = bridge["updates"][update_index]
    proof = read_bound(independent_PCM_proof, guard)
    actual_reference0(actual_closed0, independent_PCM_proof, guard)
    require(proof["kind"] == "FastMSS_phase_aware_independent_clean_PCM_references_v1" and proof["passed"] is True and same(proof["bridge"], bridge_binding) and type(proof["update_index"]) is int and proof["update_index"] == update_index and proof["phase"] == bridge["phase"] and proof["actual_optimizer_updates"] == 0 and type(proof["actual_optimizer_updates"]) is int and proof["sampler_journal_frontier_advanced"] is False, "Complete independently actual0 provisional PCM proof required")
    references = proof["references"]
    require(type(references) is list and len(references) == len(update["rows"]), "No missing/extra/duplicate or truncated source proof")
    for entry, reference in zip(update["rows"], references):
        require(reference["original_row_sha256"] == entry["original_row_sha256"] and same(reference["context"], entry["context"]) and same(reference["full_target_ids"], entry["full_target_ids"]) and reference["target_ids_u32_sha256"] == entry["target_ids_u32_sha256"], "Independent ordered source/fulltarget/phase proof differs")
        audio = reference["expected_audio"]
        require(type(audio) is dict and set(audio) == {"sample_rate", "samples", "waveform_sha256"} and type(audio["sample_rate"]) is int and audio["sample_rate"] == 16000 and sha(audio["waveform_sha256"]), "Exact whole original mono float32 16k PCM identity required")
        integer(audio["samples"], 1)
        require(reference["whole_original_clip"] is True and reference["augmentation_applied"] is False and reference["actual_geometry_and_heldout_checked"] is True, "Original full audio/heldout/declared-geometry proof required, never metadata duration inference")
        sources = reference["original_source_files"]
        require(type(sources) is list and 1 <= len(sources) <= 4, "Independent whole original source file identities required")
        for source in sources:
            require(type(source) is dict and set(source) in ({"path", "bytes", "sha256"}, {"path", "bytes", "sha256", "mtime_ns"}) and sha(source["sha256"]), "Exact independently verified source file binding required")
            integer(source["bytes"], 1)
            path = scoped(source["path"], source=True)
            require(path.stat().st_size == source["bytes"] and ("mtime_ns" not in source or type(source["mtime_ns"]) is int and path.stat().st_mtime_ns == source["mtime_ns"]), "Original source signature changed after independent PCM proof")
        if entry["role"] == "replay":
            cache = reference["genuine_teacher_feature_cache"]
            require(type(cache) is dict and cache["source_id"] == entry["source_id"] and cache["target_ids_u32_sha256"] == entry["target_ids_u32_sha256"] and cache["clean_waveform_sha256"] == audio["waveform_sha256"] and sha(cache["genuine_feature_sha256"]) and reference["teacher_or_replay_waveform_changed"] is False, "Original genuine carried replay cache/targets stay clean")
    return {"kind": "FastMSS_phase_aware_feature_update_input_binding_v1", "bridge": bridge_binding, "update_index": update_index, "coordinator_update_sha256": update["coordinator_update_sha256"], "independent_clean_PCM": independent_PCM_proof, "independent_actual0": actual_closed0, "phase": bridge["phase"], "prepared_is_committed": False, "actual_features_ready": False, "sampler_journal_frontier_advanced": False}


def validate_resume(bridge_binding, resume_binding, *, verify_checkpoint, guard=lambda: None):
    """No arbitrary skip: native caller independently attests coherent restore."""
    bridge = read_bound(bridge_binding, guard)
    validate_bridge_scope(bridge)
    resume = read_bound(resume_binding, guard)
    require(bridge["kind"] == KIND and bridge["purpose"] == purpose(bridge["mode"]) and bridge["production_enabled"] is False and bridge["bridge_source_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest() and resume["kind"] == "FastMSS_phase_aware_feature_prefix_resume_v1" and same(resume["bridge"], bridge_binding) and resume["phase"] == bridge["phase"] and callable(verify_checkpoint), "Explicit phase/source/checkpoint-bound prefix resume required")
    count = integer(resume["committed_window_updates"], 1, len(bridge["updates"]))
    update = bridge["updates"][count - 1]["coordinator_update"]
    require(resume["optimizer_updates"] == bridge["initial_committed_updates"] + count and type(resume["optimizer_updates"]) is int and resume["last_transaction_sha256"] == update["native_transaction_sha256"] and resume["sampler_state_sha256"] == update["after_sampler_sha256"], "No stale/wrong-phase/native-prefix resume")
    prefix = journal_prefix(resume["journal_prefix"])
    pointer = read_bound(resume["coherent_checkpoint_pointer"], guard)
    require(type(pointer["diagnostic_only"]) is bool and pointer["diagnostic_only"] == bridge["diagnostic_only"],
            "Resume cannot promote a diagnostic checkpoint or switch source-typed eligibility")
    require(pointer["kind"] == "ASR18_production_window_completed_checkpoint_pointer_v2" and pointer["writer_returned_and_payload_verified"] is True and pointer["committed_updates"] == resume["optimizer_updates"] and type(pointer["committed_updates"]) is int and pointer["committed_window_updates"] == count and type(pointer["committed_window_updates"]) is int and same(pointer["checkpoint"], resume["checkpoint"]) and same(pointer["journal_prefix"], prefix), "Prefix resume requires exact returned coherent pointer, not arbitrary checkpoint existence")
    window = pointer["identity"]["window"]
    require(window["plan_sha256"] == bridge["coordinator_plan_sha256"] and window["committed_window_updates"] == count and type(window["committed_window_updates"]) is int and window["last_transaction_sha256"] == resume["last_transaction_sha256"] and window["sampler_state_sha256"] == resume["sampler_state_sha256"], "Resume identity binds another plan/window/phase/source prefix")
    semantic = pointer["saved_semantic_state_verified"]
    require(semantic["passed"] is True and semantic["identity_sha256"] == digest(pointer["identity"]) and semantic["optimizer_updates"] == resume["optimizer_updates"] and semantic["sampler_state_sha256"] == resume["sampler_state_sha256"] and same(semantic["journal_prefix"], prefix), "Saved identity/state/journal semantic verification required before resume")
    metadata = read_bound(resume["checkpoint"], guard)
    require(same(metadata["payload"], pointer["payload"]), "Exact retained coherent checkpoint payload binding required")
    require(verify_checkpoint(deepcopy(resume)) is True, "Actual native owner must verify restored all207/Adam/RNG/journal/source identity")
    return {"next_update_index": count, "phase": bridge["phase"], "retain_all_window_files": True, "sampler_journal_frontier_advanced": False}
