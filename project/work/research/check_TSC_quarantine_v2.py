"""Small repeatable stdlib fixtures; no corpus/archive/PCM/checkpoint/model reads."""
from __future__ import annotations
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0, str(HERE))
import prepare_TSC_quarantine_v2 as quarantine
import tsc_supplemental_data_policy_v2 as policy
import tsc_training_inputs_v2 as inputs


def main(args):
    checks = []
    scratch = ROOT / "work/research" / ("TSC_quarantine_fixture_" + uuid.uuid4().hex)
    scratch.mkdir(exist_ok=False)
    def check(value, name):
        assert value, name
        checks.append(name)
    def reject(call, name):
        try:
            call()
        except (ValueError, KeyError, AssertionError):
            checks.append(name)
        else:
            raise AssertionError(name)
    def write_json(name, value):
        path = scratch / name
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    cases, raw_record = quarantine.review_cases(HERE / "TSC_TRAIN_Whisper_actual_prediction_ROOT_review_v1.json")
    check(set(cases) == set(quarantine.QUARANTINE_IDS), "actual bounded18 TRAIN review supplies all3 exact risk IDs")
    source = scratch / "accepted_train.jsonl"
    def row(stem, case=None):
        wav = f"ISSAI_TSC_218/Train/{stem}.wav"
        text = case["reference"] if case else "örnek konuşma"
        duration = case["duration"] if case else 2.0
        digest = case["canonical_pcm_sha256"] if case else "a" * 64
        return dict(dataset=policy.TSC_REPO, dataset_revision=policy.TSC_REVISION, dataset_config="default",
            dataset_split="train", training_role="train", language="tr", dataset_file="ISSAI_TSC_218.tar.gz",
            archive_sha256=policy.ARCHIVE_SHA, data_expansion_role="tsc_train", sample_rate=16000,
            speaker_id=None, speaker_identity_verified=False, sentence_id=None,
            original_audio_path=wav, original_transcript_path=wav[:-4]+".txt",
            source_id=f"{policy.TSC_REPO}:default:train:None:{wav}", reference_raw_text=text, text=text,
            reference=text, reference_normalized=quarantine.parent.normalize_text(text,"tr"),
            audio_sha256=digest, wav_sha256=digest, canonical_pcm_sha256=digest, source_transcript_sha256="b"*64,
            label_ids_sha256="c"*64, target_tokens=2, target_max_id=200, duration=duration, source_duration=duration,
            audio_filepath=str(scratch/"audio"/(hashlib.sha256(f"{policy.TSC_REVISION}:{stem}".encode()).hexdigest()[:24]+".wav")))

    # Only one listed ID is present. The two absent IDs must never be reported removed.
    removed_row = row("188561", cases[quarantine.QUARANTINE_IDS[0]])
    kept_rows = [row("999001"), row("999002")]
    rows = [kept_rows[0], removed_row, kept_rows[1]]
    raw_lines = [(json.dumps(item, ensure_ascii=False, indent=None) + "\r\n").encode() for item in rows]
    original_bytes = b"\n" + b"".join(raw_lines)
    source.write_bytes(original_bytes)
    destination = scratch / "quarantined_accepted_train.jsonl"
    result = quarantine.filter_manifest(source, destination, cases)
    check(source.read_bytes() == original_bytes, "original manifest bytes unchanged")
    check(destination.read_bytes() == b"\n" + raw_lines[0] + raw_lines[2], "retained bytes/order/CRLF and labels exact")
    check((result["original_clips"], result["retained_clips"], result["quarantined_clips"]) == (3,2,1), "actual present count1 not requested count3")
    check(result["quarantined_records"] == [removed_row], "excluded metadata includes untouched original label")
    check(result["retained_ordered_rows_sha256"] == policy.ordered_rows_sha(kept_rows), "ordered retained metadata digest matches")
    check(result["original_manifest_sha256"] == policy.sha256(source), "streamed original SHA matches actual metadata bytes")
    check(result["retained_manifest_sha256"] == policy.sha256(destination), "streamed retained SHA matches actual metadata bytes")
    check(result["retained_actual_PCM_hours"] == 4.0/3600, "retained actual duration sum")
    check(abs(result["original_actual_PCM_hours"]-result["retained_actual_PCM_hours"]-result["quarantined_actual_PCM_hours"])<1e-12, "duration sums close without cropping")
    altered = copy.deepcopy(removed_row); altered["canonical_pcm_sha256"] = "d"*64
    reject(lambda: quarantine.keep_row(altered, cases), "name-only case with differing PCM rejected")
    altered = copy.deepcopy(removed_row); altered["text"] += "!"
    reject(lambda: quarantine.keep_row(altered, cases), "teacher-changed label rejected")
    check(quarantine.keep_row(kept_rows[0], cases), "arbitrary nonlisted ID retained")
    subset = scratch/"absent_source.jsonl"; subset.write_bytes(raw_lines[0]+raw_lines[2])
    # Use the same source-root for canonical path validation; it is metadata only.
    all_absent = quarantine.filter_manifest(subset, scratch/"all_absent_output.jsonl", cases)
    check(all_absent["quarantined_clips"] == 0 and all_absent["retained_clips"] == 2, "already excluded risk IDs remove zero rows")

    proof = {"accepted_manifest_sha256":policy.sha256(source), "full_Train_candidate_attempted":True}
    proof_path = write_json("full_import_fixture.json", proof)
    closure_path = write_json("closure_fixture.json", {"kind":"root_owned_TSC_full_import_closure_v2", "passed":True,
        "root_observed_owned_session_completed":True, "child_exit_code":0,
        "import_proof":quarantine.record(proof_path), "accepted_manifest":quarantine.record(source)})
    receipt = {"kind":"TSC_TRAIN_uncertainty_quarantine_v2", "passed":True, "CPU_only":True,
        "helper_sha256":policy.sha256(HERE/"prepare_TSC_quarantine_v2.py"),
        "references":{"full_import_proof":quarantine.record(proof_path), "import_closure":quarantine.record(closure_path),
            "ROOT_Whisper_review":quarantine.record(HERE/"TSC_TRAIN_Whisper_actual_prediction_ROOT_review_v1.json"),
            "actual_Whisper_TRAIN_sanity":raw_record, "original_accepted_manifest":quarantine.record(source)},
        "retained_manifest":quarantine.record(destination), **result,
        "approved_uncertainty_source_ids":list(quarantine.QUARANTINE_IDS),
        "present_and_removed_source_ids":[removed_row["source_id"]],
        "reviewed_cases_absent_from_full_import_accepted":sorted(set(quarantine.QUARANTINE_IDS)-{removed_row["source_id"]}),
        "preserved_original_TR_clips":74883, "preserved_original_replay_clips":1595,
        "combined_TR_clips":74883+len(kept_rows),
        "combined_uniform_one_pass_optimizer_updates":math.ceil((74883+len(kept_rows))/8),
        "original_labels_of_retained_rows_preserved":True, "kept_manifest_rows_byte_identical":True,
        **{key:False for key in ("teacher_relabeling_performed", "original_import_proof_and_files_modified",
            "audio_cropped_or_decoded", "archive_rehashed", "new_data_added_to_training", "human_audio_audition_performed", "corpus_noise_rate_estimated")}}
    context = SimpleNamespace(proof_path=proof_path, proof_sha256=policy.sha256(proof_path), proof=proof,
        accepted_manifest_path=source, tsc_rows=rows)
    # policy requires a workspace data path, so move only the synthetic kept file there.
    data_scratch = ROOT/"work/data"/("TSC_quarantine_fixture_"+uuid.uuid4().hex)
    data_scratch.mkdir(exist_ok=False)
    actual_retained = data_scratch/destination.name; actual_retained.write_bytes(destination.read_bytes())
    receipt["retained_manifest"] = quarantine.record(actual_retained)
    check(policy.validate_quarantine(receipt, context) == kept_rows, "v2 receipt admits only deterministic reviewed subset")
    tampered = copy.deepcopy(receipt); tampered["quarantined_clips"] = 3
    reject(lambda: policy.validate_quarantine(tampered, context), "requested count substituted for actual removed count rejected")
    tampered = copy.deepcopy(receipt); tampered["reviewed_cases_absent_from_full_import_accepted"] = []
    reject(lambda: policy.validate_quarantine(tampered, context), "missing absent-ID provenance rejected")
    tampered = copy.deepcopy(receipt); tampered["teacher_relabeling_performed"] = True
    reject(lambda: policy.validate_quarantine(tampered, context), "teacher relabel claim rejected")

    actual_source = json.loads((HERE/"runs/adaptive_20261011_hf_domain30_01_domain30/run_config.json").read_text(encoding="utf-8"))
    permission = {"allow_source_sampler_transition_to_combined_uniform":True,
                  "source_sampling_metadata":actual_source["config_identity"]["domain_sampling_identity"]}
    common = inputs.common_source_recipe(actual_source, permission)
    check(set(common)==inputs.COMMON_TRAINING_KEYS and not set(common)&inputs.DOMAIN_TRAINING_KEYS, "actual domain config common recipe split")
    check(all(value==actual_source["config_identity"]["training_arguments"][key] for key,value in common.items()), "common LR/KD/gates/cadence unchanged")
    transition = inputs.source_sampling_transition(actual_source, permission)
    check(transition["source_mode"]=="domain30" and transition["source_replay_RNG_independent"] is True
          and transition["target_mode"]=="combined_uniform" and transition["target_replay_RNG_independent"] is False
          and transition["replay_order_bitwise_identity_claimed"] is False, "domain/RP RNG nuisance truthfully declared")
    reject(lambda: inputs.common_source_recipe(actual_source, {}), "domain transition without ROOT permission rejected")
    altered = copy.deepcopy(actual_source); altered["config_identity"]["training_arguments"]["unexpected_argument"] = 1
    reject(lambda: inputs.common_source_recipe(altered, permission), "unreviewed hidden scientific key rejected")
    altered = copy.deepcopy(actual_source); altered["config_identity"]["domain_sampling_identity"]["replay_rng_independent_of_turkish"] = False
    reject(lambda: inputs.common_source_recipe(altered, permission), "different original replay sampler rejected")
    altered = copy.deepcopy(actual_source); altered["code_sha256"]["turkish_domain_sampler_v2.py"] = "f"*64
    reject(lambda: inputs.common_source_recipe(altered, permission), "different domain implementation rejected")

    old_tree = ast.parse((HERE/"train_lora_batched_tsc_v1.py").read_text(encoding="utf-8"))
    new_tree = ast.parse((HERE/"train_lora_batched_tsc_v2.py").read_text(encoding="utf-8"))
    class Normalize(ast.NodeTransformer):
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Name) and t.id=="source_names" for t in node.targets):
                node.value=ast.Constant("source_map_only")
            return self.generic_visit(node)
        def visit_Constant(self,node):
            if isinstance(node.value,str):
                node.value=node.value.replace("train_lora_batched_tsc_v1.py","train_lora_batched_tsc_v2.py")
            return node
    old_defs={node.name:node for node in old_tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    new_defs={node.name:node for node in new_tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    check(set(old_defs)==set(new_defs), "worker function set unchanged")
    check(all(ast.dump(Normalize().visit(copy.deepcopy(old_defs[name])))==ast.dump(new_defs[name]) if name!="run"
              else ast.dump(Normalize().visit(copy.deepcopy(old_defs[name])))==ast.dump(Normalize().visit(copy.deepcopy(new_defs[name])))
              for name in old_defs), "all worker functions including update/save/gate AST unchanged except routing IDs")
    calls=[node.lineno for node in ast.walk(new_defs["run"]) if isinstance(node,ast.Call)
           and ast.unparse(node.func)=="torch.cuda.synchronize"]
    check(calls==[545], "source-backed unique optimizer-end synchronize run545")
    check(inputs.SUPPLEMENT_POLICY_SHA==policy.sha256(HERE/"tsc_supplemental_data_policy_v2.py")
          and policy.QUARANTINE_HELPER_SHA==policy.sha256(HERE/"prepare_TSC_quarantine_v2.py"), "new internal source pins exact")
    for name,expected in (("train_lora_batched_tsc_v1.py","1bf121e4d25b39112c8a05412b856ac4d7c8315c92819f001d7183146d2a0f8b"),
                          ("tsc_training_inputs_v1.py","7123bd6978294af4fe15c6c475699db825af98d245f1940d827ee68f961f3f7c"),
                          ("tsc_supplemental_data_policy_v1.py",policy.PARENT_POLICY_SHA)):
        check(policy.sha256(HERE/name)==expected, "frozen v1 unchanged: "+name)
    output = args.output.resolve() if args.output else HERE/("TSC_quarantine_v2_CPU_fixture_"+uuid.uuid4().hex+".json")
    assert output.is_relative_to(HERE) and not output.exists()
    report={"kind":"TSC_quarantine_and_domain_source_admission_CPU_verification_v2", "passed":True,
        "created_at_utc":datetime.now(timezone.utc).isoformat(), "checks":len(checks), "named_checks":checks,
        "fixture_script_sha256":policy.sha256(__file__), "scratch_dir":str(scratch),
        "source_sha256":{name:policy.sha256(HERE/name) for name in ("prepare_TSC_quarantine_v2.py",
            "tsc_supplemental_data_policy_v2.py","tsc_training_inputs_v2.py","train_lora_batched_tsc_v2.py")},
        "actual_source_config_metadata_read":True,"actual_closed_source_or_import_execution_claimed":False,
        "full_corpus_audio_archive_checkpoint_model_GPU_network_read_or_launch":False}
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"passed":True,"checks":len(checks),"report":str(output)}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path)
    main(parser.parse_args())
