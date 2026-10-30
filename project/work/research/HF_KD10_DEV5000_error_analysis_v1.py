"""One saved-DEV error diagnosis, not a new benchmark/inference or verifier."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/parakeet_turkish"
sys.path.insert(0,str(OUT))
from analyze_turkish_dev_errors_v1 import alignment, boundary_oracle, FOLD, top
from metrics import normalize_text, edit_distance, score_transcriptions, paired_bootstrap_delta

RUN="adaptive_20261011_hf_quality_window_01_stronger_kd"
STEPS=(0,2500,5000)
target=OUT/"HF_KD10_DEV5000_error_evidence_v1.json"
if target.exists():raise FileExistsError("Preserve prior diagnosis")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:return [json.loads(line) for line in stream if line.strip()]


fixed=read_rows(ROOT/"work/data/validation_tr.jsonl")
config_path=OUT/"runs"/RUN/"run_config.json"
config=json.loads(config_path.read_text(encoding="utf-8"))
vocabulary_path=OUT/"runs"/RUN/"vocabulary_audit.json"
vocabulary=json.loads(vocabulary_path.read_text(encoding="utf-8"))
six={item["id"]:item["token"] for item in vocabulary["added_characters"]}
report={"schema":1,"kind":"HF_KD10_saved_DEV0_2500_5000_error_diagnosis_v1",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),"run_id":RUN,
        "scope":"Same fixed50 Turkish DEV recordings; saved hypotheses only, no TEST/audio/model/GPU/network",
        "run_config_path":str(config_path),"run_config_sha256":sha(config_path),
        "vocabulary_audit_path":str(vocabulary_path),"vocabulary_audit_sha256":sha(vocabulary_path),
        "fixed_DEV_manifest_sha256":sha(ROOT/"work/data/validation_tr.jsonl"),"models":{},"pairs":{}}
predictions={}
details={}
for step in STEPS:
    folder=OUT/"results"/f"{RUN}_step_{step}"
    path=folder/"validation_tr.jsonl"
    rows=read_rows(path)
    assert len(rows)==len(fixed)==50
    for row,reference in zip(rows,fixed):
        assert row["source_id"]==reference["source_id"] and row["reference"]==reference["text"]
        assert all(row[key]==reference[key] for key in ("sentence_id","dataset_revision","audio_sha256","wav_sha256","duration"))
    score=score_transcriptions(rows,language="tr")
    stored=json.loads((folder/"summary.json").read_text(encoding="utf-8"))["validation_tr"]["asr"]
    assert all(score[key]==stored[key] for key in ("wer","cer","word_errors","reference_words","character_errors","reference_characters"))
    details[step]=score.pop("details")
    aggregate_saved=Counter()
    word_ops=Counter();char_ops=Counter();subs=Counter();char_subs=Counter();numeric=[];nonnumeric=[]
    char_ref=Counter();char_correct=Counter();char_deleted=Counter();char_substituted=Counter();char_hyp=Counter()
    diagnostic=Counter();new_piece_emissions=Counter();raw_six_ref=Counter();raw_six_hyp=Counter()
    boundary_examples=[];word_error_examples=[]
    correct_words=set();emitted_words=Counter();reference_word_types=set()
    for index,(row,detail) in enumerate(zip(rows,details[step])):
        reference,hypothesis=detail["reference_normalized"],detail["hypothesis_normalized"]
        for key in ("hits","substitutions","deletions","insertions"):aggregate_saved[key]+=row["alignment"][key]
        assert sum(row["alignment"][key] for key in ("substitutions","deletions","insertions"))==detail["word_errors"]
        assert row["alignment"]["reference_word_count"]==detail["reference_words"]
        ops=alignment(reference.split(),hypothesis.split())
        assert sum(op!="equal" for op,a,b in ops)==detail["word_errors"]
        ref_index=0
        for op,a,b in ops:
            word_ops[op]+=1
            if op=="equal":correct_words.add((index,ref_index))
            if op!="ins":ref_index+=1
            if op=="sub":subs[a,b]+=1
        char_ref.update(reference);char_hyp.update(hypothesis)
        for op,a,b in alignment(reference,hypothesis):
            char_ops[op]+=1
            if op=="equal":char_correct[a]+=1
            if op=="del":char_deleted[a]+=1
            if op=="sub":
                char_substituted[a]+=1;char_subs[a,b]+=1
                if a.translate(FOLD)==b.translate(FOLD):char_ops["diacritic_only_substitutions"]+=1
            if op!="equal" and " " in (a,b):char_ops["space_operations"]+=1
        for token in row["confidence"]["tokens"]:
            if token["token_id"] in six:new_piece_emissions[six[token["token_id"]]]+=1
        for token in six.values():
            raw_six_ref[token]+=row["reference"].count(token);raw_six_hyp[token]+=row["hypothesis"].count(token)
        diagnostic["folded_word_errors"]+=edit_distance(reference.translate(FOLD).split(),hypothesis.translate(FOLD).split())
        diagnostic["folded_character_errors"]+=edit_distance(reference.translate(FOLD),hypothesis.translate(FOLD))
        diagnostic["no_space_character_errors"]+=edit_distance(reference.replace(" ",""),hypothesis.replace(" ",""))
        diagnostic["no_space_reference_characters"]+=len(reference.replace(" ",""))
        oracle=boundary_oracle(reference.split(),hypothesis.split())
        diagnostic["boundary_oracle_word_errors"]+=oracle
        if detail["word_errors"]-oracle>=2:
            boundary_examples.append({"source_id":row["source_id"],"reference":row["reference"],"hypothesis":row["hypothesis"],
                "word_errors":detail["word_errors"],"exact_concat_boundary_oracle_errors":oracle,
                "spacing_only_edits_forgiven":detail["word_errors"]-oracle})
        (numeric if any(character.isdigit() for character in reference) else nonnumeric).append(row)
        emitted_words.update(hypothesis.split());reference_word_types.update(reference.split())
        word_error_examples.append({"source_id":row["source_id"],"reference":row["reference"],"hypothesis":row["hypothesis"],
                                   "word_errors":detail["word_errors"],"reference_words":detail["reference_words"]})
    assert aggregate_saved["substitutions"]+aggregate_saved["deletions"]+aggregate_saved["insertions"]==score["word_errors"]
    assert char_ops["sub"]+char_ops["del"]+char_ops["ins"]==score["character_errors"]
    diagnostic.update(folded_wer=diagnostic["folded_word_errors"]/score["reference_words"],
                      folded_cer=diagnostic["folded_character_errors"]/score["reference_characters"],
                      no_space_cer=diagnostic["no_space_character_errors"]/diagnostic["no_space_reference_characters"],
                      boundary_oracle_wer=diagnostic["boundary_oracle_word_errors"]/score["reference_words"],
                      exact_boundary_forgiven_word_edits=score["word_errors"]-diagnostic["boundary_oracle_word_errors"])
    def lean_score(subset):
        value=score_transcriptions(subset,language="tr");value.pop("details");return value
    model={"step":step,"predictions_path":str(path),"predictions_sha256":sha(path),"summary_sha256":sha(folder/"summary.json"),
           "standard_metrics":score,"saved_word_alignment_counts":dict(aggregate_saved),
           "diagnostic_word_alignment_counts":dict(word_ops),"character_alignment_counts":dict(char_ops),
           "normalized_core_new_characters":{char:{"reference_occurrences":char_ref[char],"correct_in_one_minimum_traceback":char_correct[char],
                "substituted":char_substituted[char],"deleted":char_deleted[char],"hypothesis_occurrences":char_hyp[char],
                "aligned_recall":char_correct[char]/char_ref[char] if char_ref[char] else None} for char in "ğış"},
           "six_added_token_rows":{char:{"id":token,"emitted_token_count":new_piece_emissions[char],
                "raw_reference_character_occurrences":raw_six_ref[char],"raw_hypothesis_character_occurrences":raw_six_hyp[char]} for token,char in six.items()},
           "diagnostic_only":dict(diagnostic),"digit_reference_subset":lean_score(numeric),"no_digit_reference_subset":lean_score(nonnumeric),
           "top_word_substitutions":top(subs,16),"top_character_substitutions":top(char_subs,16),
           "exact_boundary_examples":sorted(boundary_examples,key=lambda row:-row["spacing_only_edits_forgiven"])[:5],
           "largest_residual_word_error_examples":sorted(word_error_examples,key=lambda row:-row["word_errors"])[:5],
           "reference_word_types":len(reference_word_types),"observed_correct_reference_word_occurrences":len(correct_words)}
    report["models"][str(step)]=model
    predictions[step]=rows
    model["_correct_words"]=correct_words

for before,after in ((0,2500),(2500,5000),(0,5000)):
    a,b=report["models"][str(before)],report["models"][str(after)]
    before_correct,after_correct=a["_correct_words"],b["_correct_words"]
    differences=[left["word_errors"]-right["word_errors"] for left,right in zip(details[before],details[after])]
    example_indices=sorted(range(50),key=lambda index:-differences[index])[:5]
    pair={"before":before,"after":after,"WER_delta_percentage_points":100*(b["standard_metrics"]["wer"]-a["standard_metrics"]["wer"]),
          "CER_delta_percentage_points":100*(b["standard_metrics"]["cer"]-a["standard_metrics"]["cer"]),
          "word_errors_removed":sum(differences),"utterances_improved":sum(value>0 for value in differences),
          "utterances_worsened":sum(value<0 for value in differences),"utterances_unchanged":sum(value==0 for value in differences),
          "newly_correct_reference_word_occurrences_one_traceback":len(after_correct-before_correct),
          "lost_correct_reference_word_occurrences_one_traceback":len(before_correct-after_correct),
          "largest_improvements":[{"source_id":fixed[index]["source_id"],"reference":fixed[index]["text"],
              "before_hypothesis":predictions[before][index]["hypothesis"],"after_hypothesis":predictions[after][index]["hypothesis"],
              "word_errors_before":details[before][index]["word_errors"],"word_errors_after":details[after][index]["word_errors"]} for index in example_indices]}
    report["pairs"][f"{before}_to_{after}"]=pair
    if before==0 and after==5000:
        pair["paired_sentence_bootstrap_descriptive_only"]=paired_bootstrap_delta(
            [row["text"] for row in fixed],[row["hypothesis"] for row in predictions[before]],
            [row["hypothesis"] for row in predictions[after]],groups=[row["sentence_id"] for row in fixed],resamples=2000)
for model in report["models"].values():model.pop("_correct_words")

log_path=OUT/"runs"/"adaptive_20261011_hf_quality_window_01"/(RUN+".log")
exposure={}
for line in log_path.open(encoding="utf-8",errors="replace"):
    if line.startswith('{"step": 2500,') or line.startswith('{"step": 5000,'):
        row=json.loads(line)
        if row["step"] in (2500,5000):
            fields=("step","turkish_epoch","turkish_clips_seen","turkish_unique_clips_seen","turkish_audio_seen_seconds", "audio_seen_seconds",
                    "train_wall_seconds","language_audio_seen_seconds","replay_language_clip_counts")
            value={key:row[key] for key in fields}
            value.update(exact_published_log_record_sha256=hashlib.sha256(line.encode()).hexdigest(),
                         TR_audio_hours_seen=row["turkish_audio_seen_seconds"]/3600,
                         TR_fraction_of_all_audio=row["turkish_audio_seen_seconds"]/row["audio_seen_seconds"],
                         TR_audio_minutes_per_train_wall_minute=row["turkish_audio_seen_seconds"]/row["train_wall_seconds"],
                         unique_TR_clip_coverage=row["turkish_unique_clips_seen"]/config["turkish_clips_used"])
            exposure[str(row["step"])]=value
report["exposure"]={"mutable_log_path":str(log_path),"immutable_record_excerpts":exposure,
                    "pool_clips":config["turkish_clips_used"],"pool_actual_TR_hours":config["turkish_unique_pool_hours"],
                    "TR_clips_per_update":config["tr_clips_per_update"],"replay_clips_per_update":config["replay_clips_per_update"],
                    "source_domain_exposure_not_recounted":True}
report["limits"]=["Only50 DEV clips/825 reference words; these points were reused for model selection. Bootstrap is descriptive, not an untouched-test/generalization CI.",
                  "One minimum-edit traceback determines substitution/deletion/character categories; tied alignments can change categories without changing WER/CER.",
                  "Folded, space-free and concatenation-oracle results are diagnostics only, not replacement ASR benchmark scores; categories overlap.",
                  "Correct core new-character positions are alignment-based, not token-level acoustic causality. Uppercase case is normalized away in benchmark scoring.",
                  "Exposure grows with optimizer updates; no matched coverage ablation establishes that coverage caused gains. Source-domain exposure was not independently recounted."]
with target.open("x",encoding="utf-8") as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
print(json.dumps({"path":str(target),"sha256":sha(target),"models":{step:{key:value for key,value in model.items() if key in
    ("standard_metrics","saved_word_alignment_counts","normalized_core_new_characters","six_added_token_rows","diagnostic_only","digit_reference_subset","no_digit_reference_subset")} for step,model in report["models"].items()},
    "pairs":{key:{k:v for k,v in value.items() if k not in ("largest_improvements","paired_sentence_bootstrap_descriptive_only")} for key,value in report["pairs"].items()},
    "exposure":exposure},ensure_ascii=True))
