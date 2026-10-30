"""Synthetic metadata mapping/negative controls only; no final DATA/PCM claim."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT / "outputs/parakeet_turkish"), str(Path(__file__).parent)]
import fastmss_phase_aware_plan_bridge_v1 as bridge
from bpe_training_schedule_v1 import TrainingSchedule, CLEANER, WEAK
from bpe_production_window_coordinator_v4 import make_plan
from validate_ASR18_training_schedule_v1 import ReplayFixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()
    folder = Path(args.folder).resolve()
    bridge.require(folder.is_relative_to(bridge.ROOT.resolve()), "Fixture artifacts E-only")
    folder.mkdir(exist_ok=False)
    checks = []

    def check(value, message):
        bridge.require(value, message)
        checks.append(message)

    def reject(callback, message):
        try:
            callback()
        except (ValueError, KeyError, TypeError, OSError):
            checks.append(message)
            return
        raise ValueError("Expected fail closed: " + message)

    def schedule():
        return TrainingSchedule(broad_population=15, broad_identity={"SYNTHETIC_NOT_DATA": True},
            broad_mean_U=8.5, cleaner_population=352768, cleaner_identity={"SYNTHETIC_CONTRACT": True},
            cleaner_mean_U=11972886 / 352768, source_turkish_rng=("SYNTHETIC", 34), replay=ReplayFixture())

    def identity(phase, ordinal):
        return {"source_id": phase + ":fixture:" + str(ordinal), "language": "tr", "ordinal": ordinal}

    def rows(phase, draws, replay):
        result = []
        for draw in draws:
            row = identity(phase, draw["ordinal"])
            row.update(text="Raw English/Turkish source label unchanged", tr_metadata_labels=[3] * 600 + [8710],
                target_tokens=601, raw_language_flag="historical alarm preserved", duration=159.12,
                encoder_frames=1989, augmentation_role="student_tr")
            if phase == CLEANER:
                row["cleaner_original_ordinal"] = draw["ordinal"]
            result.append(row)
        result.extend({**row, "text": "Original replay label", "clean_target_ids": [7, 9],
            "target_tokens": 2, "augmentation_role": "replay"} for row in replay)
        return result

    artifact = folder / "synthetic_not_final_DATA.json"
    artifact.write_bytes(bridge.canonical({"SYNTHETIC_ONLY": True}))
    binding = {"path": str(artifact), "bytes": artifact.stat().st_size, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
    s = schedule()
    initial = deepcopy(s.state_dict())

    def plan(value, maximum=20):
        return make_plan(value, clone_schedule=schedule, resolve_identity=identity, resolve_rows=rows,
            feature_bytes_upper_bound=lambda phase, row: 500000, finalized_population=binding,
            verify_finalized_population=lambda pin: pin == binding, maximum_updates=maximum)

    original = plan(s)
    before = bridge.canonical(original)
    mapped = bridge.normalize_updates(original, population=15)
    check([len(v["coordinator_update"]["TR_draws"]) for v in mapped] == [14, 1, 14, 1], "All two-pass14/1 tails preserved")
    check(s.state_dict() == initial and bridge.canonical(original) == before, "Mapping never mutates plan/cursors/replay/RNG")
    check(all(v["phase"] == WEAK for v in mapped), "A window never spans a weak-to-clean phase switch")
    check(all(bridge.same(v["coordinator_update"], original["updates"][i]) for i, v in enumerate(mapped)), "Every full raw update/rows/draws/contexts/hash retained")
    check(mapped[0]["rows"][0]["full_target_ids"] == original["updates"][0]["rows"][0]["tr_metadata_labels"] and len(mapped[0]["rows"][0]["full_target_ids"]) == 601, "No long target cap/quality filter/language filter")
    check(mapped[0]["rows"][0]["route"] == "frozen_weak_TR_augmentation_CPU_only" and all(v["route"] == "direct_original_clean" for v in mapped[0]["rows"][-2:]), "Weak TR only; both original carried replay routes clean")
    check(mapped[0]["coordinator_update"]["rows"][0]["raw_language_flag"] == "historical alarm preserved", "Raw text/flags retained without correction or filtering")
    clean_schedule = schedule()
    clean_schedule.broad.draws = 30
    clean_schedule.replay.updates = 4
    cleaner = plan(clean_schedule, 2)
    clean_mapped = bridge.normalize_updates(cleaner, population=15)
    check(all(v["route"] == "direct_original_clean" for update in clean_mapped for v in update["rows"]), "Explicit cleaner TR and replay fully bypass augmentation")
    check(clean_mapped[0]["global_update"] == 5 and clean_mapped[0]["update_index"] == 0, "Local feature index distinct from carried optimizer count")
    check(all(row["independent_clean_PCM_still_required"] for update in mapped for row in update["rows"]), "Metadata mapping never claims actual PCM/features ready")

    def bad(change, value=original):
        copy = deepcopy(value)
        change(copy)
        return lambda: bridge.normalize_updates(copy, population=15)

    reject(bad(lambda p: p.update(production_enabled=True)), "Production arming rejected")
    reject(bad(lambda p: p.update(actual_optimizer_updates=True)), "Boolean fake commit rejected")
    reject(bad(lambda p: p["updates"][0]["rows"][0].update(target_tokens=600)), "Incomplete full target metadata rejected")
    reject(bad(lambda p: p["updates"][0]["rows"][0]["tr_metadata_labels"].__setitem__(0, 8192)), "Blank target forbidden without dropping row")
    reject(bad(lambda p: p["updates"][0]["contexts"][0].update(split="test")), "Evaluation context rejected")
    reject(bad(lambda p: p["updates"][0]["contexts"][0].update(phase=CLEANER)), "Wrong phase rejected")
    reject(bad(lambda p: p["updates"][0]["rows"][0].update(augmentation_role="teacher")), "Teacher row cannot enter student augmentation route")
    reject(bad(lambda p: p["updates"][0]["contexts"][1].update(draw_index=p["updates"][0]["contexts"][0]["draw_index"])), "Duplicate ordered draw rejected")
    reject(bad(lambda p: p["updates"][0]["rows"].reverse()), "Out-of-order source rows rejected")
    reject(bad(lambda p: p["updates"][0]["contexts"][-1].update(source_id="replacement_replay")), "Replacement replay identity rejected")
    reject(bad(lambda p: p["updates"][1].update(before_sampler_sha256="0" * 64)), "Stale before/after sampler chain rejected")
    reject(bad(lambda p: p.update(planned_payload_upper_bound_bytes=1)), "Declared capacity inconsistency rejected")
    reject(bad(lambda p: p["final_private_planning_sampler"]["broad_cursor"].update(population=14)), "Partial population cannot masquerade as full DATA")
    reject(bad(lambda p: p["final_private_planning_sampler"]["carried_replay"].update(updates=True)), "Boolean or reset carried replay frontier rejected")
    reject(bad(lambda p: p["updates"][0]["rows"][0].update(cleaner_original_ordinal=0), cleaner), "Wrong cleaner ordinal rejected")
    reject(lambda: bridge.validate_DATA({}, lambda: None), "Missing actual final DATA artifacts fail closed")
    reject(lambda: bridge.bridge_plan(binding, DATA_evidence={}, session_binding=binding, window_index=0), "Synthetic fixture cannot generate actual bridged session")
    check(not any(v in sys.modules for v in ("torch", "numpy", "scipy", "pyarrow", "transformers", "av", "soundfile")), "No numerical/audio/frontend/model imports")
    report = {"kind": "FastMSS_synthetic_phase_plan_bridge_metadata_controls_v1", "passed": True,
        "checks": checks, "check_count": len(checks), "actual_native_updates": 0, "actual_source_audio_or_features_prepared": False,
        "actual_final_DATA_positive_path_exercised": False, "synthetic_mapping_only": True, "production_enabled": False,
        "bridge_source_sha256": hashlib.sha256(Path(bridge.__file__).read_bytes()).hexdigest()}
    (folder / "report.json").write_bytes(bridge.canonical(report))
    print(json.dumps({"passed": True, "checks": len(checks), "source_only_synthetic_metadata": True}))


if __name__ == "__main__":
    main()
