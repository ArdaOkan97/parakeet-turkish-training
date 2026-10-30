"""Narrow source/saved-DEV review; no PCM/archive/model/checkpoint/inference reads."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/"outputs/parakeet_turkish"
sys.path.insert(0,str(HERE))
from atomic_io_retry_v1 import read_json
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
run="adaptive_20261011_hf_domain30_01_domain30"
config_path=HERE/"runs"/run/"run_config.json"
config=read_json(config_path,{})
qpath=HERE/"TSC_TRAIN_uncertainty_quarantine_v2.json"
quarantine=read_json(qpath,{})
assert quarantine["retained_clips"]==178915 and quarantine["quarantined_clips"]==3
assert quarantine["preserved_original_TR_clips"]==74883 and quarantine["preserved_original_replay_clips"]==1595
domain=config["sampling"]["initial_pool"]["domain"]
fleurs=domain["fleurs"]["pool_clips"]; cv=domain["cv"]["pool_clips"]
original_n=config["turkish_clips_used"]; tsc_n=quarantine["retained_clips"]
combined_n=original_n+tsc_n
original_hours=config["turkish_unique_pool_hours"]; tsc_hours=quarantine["retained_actual_PCM_hours"]
baseline=read_json(HERE/"results/base_v3/summary.json",{})
points=[]
for step in (0,250,348,500,750,1000):
    path=HERE/"results"/(run+"_step_"+str(step))/"summary.json"
    if not path.exists():continue
    summary=read_json(path,{})
    if not all("validation_"+lang in summary for lang in ("tr","en","es","it","fr","de")):continue
    gates={}
    for lang in ("en","es","it","fr","de"):
        b=baseline["validation_"+lang]["asr"]; m=summary["validation_"+lang]["asr"]
        limit=b["wer"]+max(.005,.1*b["wer"])
        gates[lang]={"word_errors":m["word_errors"],"reference_words":m["reference_words"],"wer":m["wer"],
            "gate_wer_limit":limit,"maximum_integer_word_errors":math.floor(limit*m["reference_words"]+1e-10),"passed":m["wer"]<=limit}
    points.append({"step":step,"TR_asr":summary["validation_tr"]["asr"],"retention":gates,
        "all_five_passed":all(item["passed"] for item in gates.values()),"summary_path":str(path),"summary_sha256":sha(path)})
history=read_json(HERE/"runs"/run/"training_history.json",[])
tail=history[-100:]
last=history[-1]
loss={key:math.fsum(row[key] for row in tail)/len(tail) for key in
    ("loss","tdt_loss","replay_kl","replay_duration_kl","replay_feature_distillation","gradient_norm")}
loss["weighted_token_KD_term_in_total"]=10*loss["replay_kl"]
loss["weighted_duration_KD_term_in_total"]=.5*loss["replay_duration_kl"]
loss["updates_above_gradient_clip1_fraction"]=sum(row["gradient_norm"]>1 for row in tail)/len(tail)
pool={"original_TR":original_n,"original_CV":cv,"original_FLEURS":fleurs,"TSC_retained":tsc_n,
    "combined_TR":combined_n,"unchanged_replay":1595,"one_uniform_pass_updates":math.ceil(combined_n/8),
    "original_TR_actual_PCM_hours":original_hours,"TSC_actual_PCM_hours":tsc_hours,"combined_TR_actual_PCM_hours":original_hours+tsc_hours,
    "uniform_TR_clip_fractions":{"CV":cv/combined_n,"FLEURS":fleurs/combined_n,"TSC":tsc_n/combined_n},
    "TSC_fraction_of_TR_audio_hours":tsc_hours/(original_hours+tsc_hours),
    "initial_250_update_expected_draws":{"total_TR":2000,"TSC":2000*tsc_n/combined_n,"FLEURS":2000*fleurs/combined_n,"CV":2000*cv/combined_n,"replay":2000},
    "domain30_250_update_FLEURS_visits":600,"FLEURS_exposure_ratio_domain30_to_combined_uniform":.3/(fleurs/combined_n),
    "250_update_uniform_pool_fraction":2000/combined_n,
    "one_pass_replay_draws":math.ceil(combined_n/8)*8,"mean_replay_visits_per_source_at_one_pass":math.ceil(combined_n/8)*8/1595,
    "last_batch_boundary_TR_repeat_count":math.ceil(combined_n/8)*8-combined_n}
sources=("train_lora_batched_tsc_v2.py","tsc_training_inputs_v2.py","tsc_supplemental_data_policy_v2.py",
    "tsc_supplemental_data_policy_v1.py","prepare_TSC_quarantine_v2.py","train_lora_batched.py","batched_lora.py",
    "batched_lora_tr_metadata_v1.py","native_tdt_checkpointed_v3.py","encoder_lora.py","run_hf_TSC_arm_v2.py")
report={"kind":"TSC_training_design_source_and_DEV_review_v2","created_at_utc":datetime.now(timezone.utc).isoformat(),
 "scope":"Source, completed actual quarantine metadata and saved DEV summaries/history only; no actual TSC training or causal ablation",
 "source_sha256":{name:sha(HERE/name) for name in sources},
 "provenance":{"quarantine":{"path":str(qpath),"sha256":sha(qpath)},"domain_run_config":{"path":str(config_path),"sha256":sha(config_path)},
   "history_snapshot_last_step":last["step"],"saved_DEV_points":points},"pool_and_expected_exposure":pool,
 "domain_recent_objective":{"steps":[tail[0]["step"],tail[-1]["step"]],"means":loss,
   "warning":"Scalar loss contributions and aggregate norm do not measure TR/replay gradient conflict or prove a KD bottleneck"},
 "source_findings":[
   {"status":"no_concrete_math_bug_found","evidence":"batched_lora.py:181-281, trainer:502-545","finding":"Every clip contributes raw native TDT NLL divided by its own target count, then equal clip mean; two B8 microbatches divided by accumulation2. Turkish preservation terms are literal zero; replay alone gets original-token KL/duration KL/cached-feature cosine."},
   {"status":"correct_geometry_and_acquisition","evidence":"trainer:285-335,529-541; encoder_lora.py","finding":"Teacher captured before vocabulary extension/warmstart, original8193 inclblank8192; student8199 lexical+5duration, learned six8193:8199 get actual postAdam delta6x without blank/duration scaling. All24 attention q/k/v/o rank8 adapters carry gradients despite encoder.eval; no checkpoint row reset during warmstart."},
   {"status":"real_domain_exposure_change_not_leakage","finding":"Combined-uniform replaces domain30: FLEURS2037/253798 draws, while TSC178915 dominates. Prior domain gains can regress through changed exposure independently of TSC label quality. Source closure explicitly declares this and independent->shared replay RNG nuisance."},
   {"status":"coverage_clock_is_sound_but_not_guaranteed","finding":"BucketCycler uniformly shuffles full pool, length-sorts only128-row reservoirs; original prefix is not forced to train first. One-pass31725 covers all253798 unique TR then repeats2 at final update. FreshAdam begins new pool clock; budget exact resumes preserve frontier, but internal two-retention failures can still stop well before full pass."},
   {"status":"metadata_leakage_guard_present","evidence":"policy_v1:245-271 + v2 derivative","finding":"Source IDs, global FLEURSsentence IDs, normalized fixed heldout refs, embedded/WAV/PCM hashes excluded; exact TRAIN audio duplicates excluded, distinct-audio repeated TRAIN transcripts retained. Only TSCTrain imported. Model DEV calls are validation-only, all6x50; TEST metadata used for isolation, never predictions for selection."},
   {"status":"unproven_label_and_speaker_quality","finding":"Three actual reviewed cases quarantined, original labels retained.18 purpose-selected Whisper transcripts are not a representative noise estimate or human audition. Metadata/hash checks cannot verify semantic alignment, speaker/video/perceptual near-duplicates or guarantee learning quality."},
   {"status":"preservation_strength_is_hypothesis_not_demonstrated_bottleneck","finding":"50% clips plusKD10 puts substantial old-task objective on shared parameters, but TR has no teacher penalty. UniformKD10 gained through adequate exposure and domain30 still gains atsameKD10. Earlier25%-replay short screens did not establish improved TR. PreserveKD/RP in firstTSCarm; reduce only in a separate matched experiment if TR learning stalls with ample retention margin."},
   {"status":"selected_DEV_reuse_limits_generalization","finding":"50 TR clips/825words and five50-clip retention sets have guided many model choices. No added training overlap was found in the scoped contract, but repeated DEV selection creates selection bias; all5 pass does not establish whole-language retention or meet competitive11.94% target."}],
 "recommendations_priority":[
   "Finish the currently improving domain arm and initialize firstTSCtrial only from a CLOSED qualifying globalBEST. Keep recipe unchanged initially, compare candidate to its exact source0 and preserve globalBEST if no gain; do not expect250updates to evaluate209h of new speech.",
   "If combined-uniform TSC regresses/stalls, next isolated sampler arm should keep30% FLEURS TR and splitremaining70% between CV/TSC; preserve replay50/KD10 and declare new fullcoverage clock. Match TR/audio exposure and retain same replay RNG across its uniform comparator. This directly addresses the measured37.38x FLEURS dilution before changing capacity/KD.",
   "Use multiple bounded same-run budget windows to retain Adam/frontier; distinguish internal earlystop from budget closure. If two marginal retention misses repeatedly prevent adequate exposure, separately declare an exploration stopping-policy change, while strict all5 selection and resource guards remain unchanged.",
   "Only after an exposure-matched plateau with safe retention margin test one preservation change: KD10->5 atsame replay50 OR replay50->25 atsameKD10. Keep source/sampler/LRs/vocabulary fixed and recordclip/audio/control exposure; do not reduce both together. CurrentES/DE margins make an unconditional weakening unjustified.",
   "If both data/sampling arms plateau, the alreadyisolated BPE512 arm is a useful lexical/sequence-length test with mandatory unsuppressed source-vs-candidate DEV equality. Existing error evidence does not justify a stronger6-character boost or rank16 as the first intervention; glyphs alreadyemit and broad substitutions dominate."],
 "no_model_GPU_checkpoint_PCM_archive_TEST_predictions_or_network_read":True,"frozen_sources_modified":False,
 "actual_TSC_model_quality_or_training_launch_claimed":False}
path=HERE/"TSC_training_design_source_review_v2.json"
with path.open("x",encoding="utf-8") as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
print(json.dumps({"report":str(path),"sha256":sha(path),"pool":pool,
    "DEV":[{"step":x["step"],"TR_WER":x["TR_asr"]["wer"],"all5":x["all_five_passed"]} for x in points],"loss":loss}))
