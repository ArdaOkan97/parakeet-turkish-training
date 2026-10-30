"""Inactive metadata policy for a separately verified TSC TRAIN supplement.

No archive, audio, tokenizer/model, network or GPU APIs. Actual canonical PCM
and target roundtrips are reused ONLY from the approved importer and unchanged
parent proofs. A future trainer must explicitly call this policy and prepare
new acoustic/target metadata; the frozen v3 policy remains unchanged.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import unicodedata

from metrics import normalize_text

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parent.parent
DATA = ROOT / "work/data"
TSC_REPO = "issai/Turkish_Speech_Corpus"
TSC_REVISION = "d79ea01392882322e8f46657cb635236af25b6d4"
ARCHIVE_SHA = "3feaba1f911542b587622d49c968bef971cfd8a4b3908d73090b8b914d2b694f"
ARCHIVE_BYTES = 21366175820
APPROVED_IMPORTER_SHA = "08785c2446229ae6716feee5e9103d77c468faf0eaf68a602e0477bc2d36063e"
V3_POLICY_SHA = "08eee3896bf8571c9e97dae1be688ecef7766816c019741dd7631d70f640ce41"
PARENT_PROOF_SHA = "5c220d7fa4fa554b741afa7c565afe3f1500b3de8bfa99de0b1040a450cf4980"
PARENT_LABEL_SHA = "42aa606057ba78232ef930f7317675ef318dddb034bc302e4de2247eedd04e3a"
TOKENIZER_FILE_SHA = "9c9f537bf7190ae2f0e30448a6915c4f607fa4d011b4f62b6f1849064ef8afcd"
TOKENIZER_SEMANTIC_SHA = "b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75"
NORMALIZER_SHA = "433c31d9cbc4975c42d3a0fe659cd0bbd6c1caf2e1eb3d85169baafec7f61047"
REPLAY_SOURCE_SET_SHA = "c165e0a2e2021b5c69e5b43b837b1276be1dc0abb7f0f69b8d9623284a87fd2b"
CV_REPO = "ysdede/commonvoice_17_tr_fixed"
CV_REVISION = "9d8a025f1ae7c5a2f5c43298c7c4910b03995a4b"
FL_REPO = "google/fleurs"
FL_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
CONFIGS = {"en":"en_us", "es":"es_419", "it":"it_it", "fr":"fr_fr", "de":"de_de", "tr":"tr_tr"}
POLICY = {"kind":"public_turkish_v3_plus_explicit_TSC_TRAIN_v1", "vocabulary_size":8199,
          "blank_id":8192, "max_target_tokens":180, "duration_bounds_seconds":[.5,30.],
          "original_TR_ordered_prefix_unchanged":True, "original_1595_replay_unchanged":True,
          "distinct_audio_repeated_TRAIN_text_allowed":True, "TSC_Dev_Test_ineligible":True,
          "TSC_no_speaker_disjointness_claim":True, "TDT_tokenizer_grouping":False}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_digest(value):
    return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def ids_sha(rows):
    return hashlib.sha256("\n".join(sorted(row["source_id"] for row in rows)).encode()).hexdigest()


def ordered_rows_sha(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",",":"), ensure_ascii=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def verified_path(path, expected, *, workspace=True):
    path = Path(path).resolve()
    require(not workspace or path.is_relative_to(ROOT), "Workspace-contained provenance required")
    require(valid_digest(expected) and sha256(path)==expected, "Proof-bound file changed: "+str(path))
    return path


def validate_import_proof(proof, parent, *, require_full_train_attempt=False):
    require(proof.get("kind")=="TSC_TRAIN_candidate_import_v1" and proof.get("passed") is True,
            "Completed approved TSC candidate import required")
    expected = {"importer_sha256":APPROVED_IMPORTER_SHA, "archive_revision":TSC_REVISION,
                "archive_sha256":ARCHIVE_SHA, "archive_size_bytes":ARCHIVE_BYTES,
                "full_archive_hash_verified":True, "archive_stat_unchanged_after_import":True,
                "tokenizer_semantic_sha256":TOKENIZER_SEMANTIC_SHA, "max_target_tokens":180,
                "blank_id":8192, "TDT_decode_grouping":False, "duration_bounds_seconds":[.5,30.],
                "parent_preparation_sha256":PARENT_PROOF_SHA, "parent_labels_sha256":PARENT_LABEL_SHA,
                "existing_TRAIN_manifests_modified":False, "Dev_Test_archive_audio_or_text_imported":False,
                "fixed_heldout_predictions_read":False, "new_data_added_to_training":False,
                "GPU_or_model_used":False}
    require(all(proof.get(key)==value for key,value in expected.items()), "Pinned TSC import/parent/tokenizer policy differs")
    observed, selected, accepted, prepared = (proof.get(key) for key in
        ("Train_pairs_observed","Train_pairs_selected_for_candidate","accepted_clips","prepared_clips"))
    require(all(type(value) is int for value in (observed,selected,accepted,prepared))
            and 0<accepted<=prepared<=selected<=observed, "Positive complete pair/accepted counts required")
    require(proof.get("archive_split_member_counts",{}).get("Train")==2*observed,
            "Full archive metadata must prove complete TRAIN WAV/TXT pairs")
    staging=proof.get("staging",{})
    require(staging.get("Train_pairs_staged")==selected and staging.get("raw_Train_files_staged")==2*selected
            and type(staging.get("raw_Train_bytes_staged")) is int and staging["raw_Train_bytes_staged"]>0
            and staging.get("metadata_only_first_pass") is True and staging.get("tar_member_cache_discarded") is True
            and staging.get("archive_stream_passes")==2, "Bounded selected-pair staging evidence differs")
    require(type(proof.get("full_Train_candidate_attempted")) is bool
            and (not proof["full_Train_candidate_attempted"] or selected==observed)
            and (not require_full_train_attempt or proof["full_Train_candidate_attempted"]),
            "Full TRAIN attempt explicitly required or inconsistent")
    protected=proof.get("protected_existing_manifests_sha256",{})
    required={str(Path(parent["CPU_tokenizer_accepted_manifest_path"]).resolve()):parent["CPU_tokenizer_accepted_manifest_sha256"],
              **{str(Path(path).resolve()):digest for path,digest in parent["heldout_manifests_sha256"].items()}}
    require(all(protected.get(path)==digest for path,digest in required.items())
            and str((DATA/"replay_train.jsonl").resolve()) in protected,
            "Exact original TR/replay/fixed heldout exclusion inputs required")


def validate_tsc_row(row, accepted_manifest_path):
    wav, txt = row.get("original_audio_path"), row.get("original_transcript_path")
    require(isinstance(wav,str) and isinstance(txt,str) and "\\" not in wav and "\\" not in txt,
            "Flat paired TSC TRAIN provenance required")
    paths=[PurePosixPath(value) for value in (wav,txt)]
    require(all(len(path.parts)==3 and path.parts[:2]==("ISSAI_TSC_218","Train")
                and not any(part in ("..", ".") or ":" in part for part in path.parts) for path in paths)
            and paths[0].suffix==".wav" and paths[1].suffix==".txt" and paths[0].stem==paths[1].stem,
            "TSC only paired flat TRAIN WAV/TXT; Dev/Test/unsafe paths ineligible")
    expected={"dataset":TSC_REPO,"dataset_revision":TSC_REVISION,"dataset_config":"default",
              "dataset_split":"train","training_role":"train","language":"tr",
              "dataset_file":"ISSAI_TSC_218.tar.gz","archive_sha256":ARCHIVE_SHA,
              "data_expansion_role":"tsc_train","sample_rate":16000,
              "speaker_id":None,"speaker_identity_verified":False,"sentence_id":None}
    require(all(key in row and row[key]==value for key,value in expected.items()), "TSC source/split/audio/unknown-speaker identity differs")
    require(row.get("source_id")==f"{TSC_REPO}:default:train:None:{wav}", "Exact original TSC source ID required")
    raw=row.get("reference_raw_text")
    require(isinstance(raw,str) and row.get("text")==unicodedata.normalize("NFC",raw).strip()
            and row.get("reference")==row["text"] and row.get("reference_normalized")==normalize_text(row["text"],"tr")
            and bool(row["reference_normalized"]), "Raw/NFC/current Turkish normalization identity differs")
    require(all(valid_digest(row.get(key)) for key in
                ("audio_sha256","wav_sha256","canonical_pcm_sha256","source_transcript_sha256","label_ids_sha256")),
            "Approved source/WAV/canonical PCM/transcript/label hashes required")
    require(type(row.get("target_tokens")) is int and 1<=row["target_tokens"]<=180
            and type(row.get("target_max_id")) is int and 0<=row["target_max_id"]<8199,
            "Current HF positive representable target proof required")
    require(all(type(row.get(key)) in (int,float) and math.isfinite(row[key]) and .5<=row[key]<=30.
                for key in ("duration","source_duration")), "Uncropped whole-recording duration bound required")
    expected_audio=Path(accepted_manifest_path).parent/"audio"/(hashlib.sha256(f"{TSC_REVISION}:{paths[0].stem}".encode()).hexdigest()[:24]+".wav")
    require(Path(row["audio_filepath"]).resolve()==expected_audio.resolve(), "Approved staged canonical WAV path differs")


@dataclass(frozen=True)
class SupplementContext:
    proof_path: Path
    proof_sha256: str
    proof: dict
    accepted_manifest_path: Path
    original_tr_manifest_path: Path
    original_tr_rows: list
    tsc_rows: list
    original_replay_rows: list
    heldout_rows: list
    parent_pcm_by_source: dict
    replay_reference: dict


def load_supplement_context(proof_path, *, replay_pool_path, replay_pool_sha256,
                            replay_config_path, replay_config_sha256, require_full_train_attempt=False):
    """Future metadata read/hashing only; consumes real ROOT-approved proof paths."""
    for name,expected in (("prepare_TSC_TRAIN_v1.py",APPROVED_IMPORTER_SHA),("public_turkish_data_policy_v3.py",V3_POLICY_SHA),
                          ("metrics.py",NORMALIZER_SHA),("tr_public_v3_preparation.json",PARENT_PROOF_SHA),
                          ("tr_public_v3_tokenizer_acceptance.json",PARENT_LABEL_SHA),("checkpoint/tokenizer.json",TOKENIZER_FILE_SHA)):
        verified_path(OUTPUT/name,expected)
    parent=read_json(OUTPUT/"tr_public_v3_preparation.json")
    labels=read_json(OUTPUT/"tr_public_v3_tokenizer_acceptance.json")
    require(labels.get("passed") is True and labels.get("group_tokens") is False
            and labels.get("tokenizer_semantic_sha256")==TOKENIZER_SEMANTIC_SHA,
            "Completed current ungrouped HF tokenizer/canonical PCM proof required")
    proof_path=Path(proof_path).resolve()
    require(proof_path.is_relative_to(OUTPUT), "Separate workspace import proof required")
    proof=read_json(proof_path)
    validate_import_proof(proof,parent,require_full_train_attempt=require_full_train_attempt)
    for path,digest in proof["protected_existing_manifests_sha256"].items():
        verified_path(path,digest)
    original_path=verified_path(parent["CPU_tokenizer_accepted_manifest_path"],parent["CPU_tokenizer_accepted_manifest_sha256"])
    accepted=verified_path(proof["accepted_manifest_path"],proof["accepted_manifest_sha256"])
    require(accepted.is_relative_to(DATA) and accepted.name=="accepted_train.jsonl"
            and accepted.parent!=original_path.parent, "Fresh separate TSC accepted manifest required")
    verified_path(proof["prepared_manifest_path"],proof["prepared_manifest_sha256"])
    replay_pool_path=verified_path(replay_pool_path,replay_pool_sha256)
    replay_config_path=verified_path(replay_config_path,replay_config_sha256)
    replay_config=read_json(replay_config_path)
    original_replay=read_json(replay_pool_path)["replay"]
    require(replay_config.get("replay_clips_used")==1595
            and replay_config.get("config_identity",{}).get("replay_pool_sha256")==REPLAY_SOURCE_SET_SHA
            and len(original_replay)==1595 and ids_sha(original_replay)==REPLAY_SOURCE_SET_SHA,
            "Explicit prior1595 accepted replay pool/config required")
    original=read_rows(original_path); supplemental=read_rows(accepted)
    require(len(original)==parent["CPU_tokenizer_accepted_clips"]==74883
            and len(supplemental)==proof["accepted_clips"], "Proof-bound original and TSC accepted counts differ")
    require(abs(math.fsum(row["duration"] for row in supplemental)/3600-proof["accepted_actual_PCM_hours"])<=1e-10,
            "TSC proof-bound accepted PCM duration sum differs")
    parent_pcm={row["source_id"]:row["canonical_pcm_sha256"] for row in labels["accepted_source_labels_and_audio"]}
    require(set(parent_pcm)=={row["source_id"] for row in original}, "Exact original canonical PCM source set required")
    heldout={row["source_id"]:row for path in parent["heldout_manifests_sha256"] for row in read_rows(path)}
    require(all(row.get("dataset_split") in ("validation","test") for row in heldout.values()), "Fixed heldout splits differ")
    return SupplementContext(proof_path,sha256(proof_path),proof,accepted,original_path,original,supplemental,
                             original_replay,list(heldout.values()),parent_pcm,
                             {"pool_path":str(replay_pool_path),"pool_sha256":replay_pool_sha256,
                              "config_path":str(replay_config_path),"config_sha256":replay_config_sha256})


def validate_original_source(row):
    repo,revision,split=row.get("dataset"),row.get("dataset_revision"),row.get("dataset_split")
    if repo==CV_REPO:
        valid=(revision==CV_REVISION and row.get("language")=="tr" and row.get("dataset_config")=="default"
               and ((split=="train" and row.get("dataset_file")=="data/train-00000-of-00001.parquet")
                    or (split=="validated" and row.get("training_role")=="train"
                        and row.get("dataset_file")=="data/validated-00000-of-00001.parquet")))
    elif repo==FL_REPO:
        config=CONFIGS.get(row.get("language"))
        valid=(revision==FL_REVISION and split=="train" and config is not None
               and row.get("dataset_config")==config
               and row.get("dataset_file")==f"parquet-data/{config}/train-00000-of-00001.parquet")
    else:
        valid=False
    require(valid and row["source_id"].startswith(f'{repo}:{row.get("dataset_config")}:{split}:'),
            "Unchanged v3 CV/FLEURS eligibility required: "+row["source_id"])


def audit_combined_rows(combined_tr, replay, context):
    """Validate an explicit original-prefix+supplement pool, without audio reads."""
    prefix=context.original_tr_rows
    require(len(combined_tr)==len(prefix)+len(context.tsc_rows)
            and combined_tr[:len(prefix)]==prefix and combined_tr[len(prefix):]==context.tsc_rows,
            "Combined TR must preserve every original row/order and append exactly the approved TSC manifest")
    require(replay==context.original_replay_rows and len(replay)==1595 and ids_sha(replay)==REPLAY_SOURCE_SET_SHA,
            "Original1595 accepted replay rows/order/cache/labels must remain exact")
    for row in prefix+replay:
        validate_original_source(row)
    heldout_ids={row["source_id"] for row in context.heldout_rows}
    heldout_sentences={row["sentence_id"] for row in context.heldout_rows if row.get("dataset")==FL_REPO and row.get("sentence_id") is not None}
    heldout_hashes={row[key] for row in context.heldout_rows for key in ("audio_sha256","wav_sha256","canonical_pcm_sha256") if row.get(key)}
    references={(row["language"],normalize_text(row["text"],row["language"])) for row in context.heldout_rows}
    tr_references={normalize_text(row["text"],language) for row in context.heldout_rows for language in (row["language"],"tr")}
    all_rows=combined_tr+replay
    require(len({row["source_id"] for row in all_rows})==len(all_rows), "Combined/replay source IDs must be unique")
    for row in all_rows:
        require(not (row["source_id"] in heldout_ids
                     or (row.get("dataset")==FL_REPO and row.get("sentence_id") in heldout_sentences)
                     or any(row.get(key) in heldout_hashes for key in ("audio_sha256","wav_sha256","canonical_pcm_sha256") if row.get(key))
                     or (row["language"],normalize_text(row["text"],row["language"])) in references
                     or (row["language"]=="tr" and normalize_text(row["text"],"tr") in tr_references)),
                "Fixed heldout source/sentence/reference/audio overlap: "+row["source_id"])
    seen_hashes={row[key] for row in prefix+replay for key in ("audio_sha256","wav_sha256","canonical_pcm_sha256") if row.get(key)}
    seen_hashes.update(context.parent_pcm_by_source.values())
    for row in context.tsc_rows:
        validate_tsc_row(row,context.accepted_manifest_path)
        hashes={row[key] for key in ("audio_sha256","wav_sha256","canonical_pcm_sha256")}
        require(not hashes & seen_hashes, "TSC exact audio/PCM duplicates existing TRAIN or earlier TSC row")
        seen_hashes.update(hashes)
    identity={"policy":POLICY,"policy_source_sha256":sha256(__file__),"unchanged_v3_policy_sha256":V3_POLICY_SHA,
              "parent_preparation_sha256":PARENT_PROOF_SHA,"parent_labels_sha256":PARENT_LABEL_SHA,
              "TSC_import_proof_path":str(context.proof_path),"TSC_import_proof_sha256":context.proof_sha256,
              "TSC_accepted_manifest_sha256":context.proof["accepted_manifest_sha256"],
              "TSC_archive_sha256":ARCHIVE_SHA,"TSC_revision":TSC_REVISION,
              "tokenizer_semantic_sha256":TOKENIZER_SEMANTIC_SHA,"original_TR_clips":len(prefix),
              "TSC_clips":len(context.tsc_rows),"combined_TR_clips":len(combined_tr),
              "combined_TR_actual_PCM_hours":math.fsum(row["duration"] for row in combined_tr)/3600,
              "combined_TR_ordered_rows_sha256":ordered_rows_sha(combined_tr),
              "combined_TR_source_ids_sha256":ids_sha(combined_tr),
              "replay_ordered_rows_sha256":ordered_rows_sha(replay),"replay_source_ids_sha256":REPLAY_SOURCE_SET_SHA,
              "replay_reference":context.replay_reference,
              "source_role_counts":dict(Counter(row.get("data_expansion_role","preserved_existing") for row in combined_tr))}
    return {"schema":1,"kind":"TSC_supplemental_metadata_input_audit_v1","passed":True,"identity":identity,
            "actual_audio_read_in_this_audit":False,"canonical_PCM_and_tokenizer_evidence":"Approved full-hash import plus unchanged parent canonical/target proofs",
            "independent_metadata_heldout_and_exact_hash_checks":True,
            "complete_importer_heldout_PCM_checks_reused":True,
            "current_WAV_mutation_not_independently_checked":True,
            "new_TR_frame_target_metadata_acceptance_pending":True,"optimizer_exposure_claimed":False,
            "no_model_GPU_or_heldout_predictions_used":True,
            "speaker_video_and_perceptual_near_duplicate_disjointness_not_proven":True}


def audit_combined_manifest(manifest, replay, context):
    manifest=Path(manifest).resolve()
    protected={Path(path).resolve() for path in context.proof["protected_existing_manifests_sha256"]}
    require(manifest.is_relative_to(DATA) and manifest!=context.accepted_manifest_path
            and manifest!=context.original_tr_manifest_path and manifest not in protected,
            "Explicit fresh combined manifest required")
    result=audit_combined_rows(read_rows(manifest),replay,context)
    result["identity"].update(combined_TR_manifest_path=str(manifest),combined_TR_manifest_sha256=sha256(manifest))
    return result
