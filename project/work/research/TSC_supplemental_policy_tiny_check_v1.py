"""Tiny synthetic metadata checks; no real manifests/archive/audio are read."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/"outputs/parakeet_turkish"
source=OUTPUT/"tsc_supplemental_data_policy_v1.py"
sys.path.insert(0,str(OUTPUT))
spec=importlib.util.spec_from_file_location("tsc_policy_fixture",source)
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
checks=[]


def accepted(name,fn):
    fn();checks.append(name)


def rejected(name,fn):
    try:fn()
    except (ValueError,KeyError):checks.append(name)
    else:raise AssertionError(name)


digest="a"*64
parent={"CPU_tokenizer_accepted_manifest_path":str(module.DATA/"synthetic_original.jsonl"),
        "CPU_tokenizer_accepted_manifest_sha256":digest,
        "heldout_manifests_sha256":{str(module.DATA/"synthetic_validation_tr.jsonl"):digest}}
proof={"kind":"TSC_TRAIN_candidate_import_v1","passed":True,"importer_sha256":module.APPROVED_IMPORTER_SHA,
       "archive_revision":module.TSC_REVISION,"archive_sha256":module.ARCHIVE_SHA,"archive_size_bytes":module.ARCHIVE_BYTES,
       "full_archive_hash_verified":True,"archive_stat_unchanged_after_import":True,
       "tokenizer_semantic_sha256":module.TOKENIZER_SEMANTIC_SHA,"max_target_tokens":180,"blank_id":8192,
       "TDT_decode_grouping":False,"duration_bounds_seconds":[.5,30.],"parent_preparation_sha256":module.PARENT_PROOF_SHA,
       "parent_labels_sha256":module.PARENT_LABEL_SHA,"existing_TRAIN_manifests_modified":False,
       "Dev_Test_archive_audio_or_text_imported":False,"fixed_heldout_predictions_read":False,
       "new_data_added_to_training":False,"GPU_or_model_used":False,
       "Train_pairs_observed":7,"Train_pairs_selected_for_candidate":3,"prepared_clips":2,"accepted_clips":2,
       "archive_split_member_counts":{"Train":14,"Dev":2,"Test":2},"full_Train_candidate_attempted":False,
       "staging":{"Train_pairs_staged":3,"raw_Train_files_staged":6,"raw_Train_bytes_staged":300,
                  "metadata_only_first_pass":True,"tar_member_cache_discarded":True,"archive_stream_passes":2},
       "protected_existing_manifests_sha256":{parent["CPU_tokenizer_accepted_manifest_path"]:digest,
           **parent["heldout_manifests_sha256"],str((module.DATA/"replay_train.jsonl").resolve()):digest}}
accepted("Dynamic future selected/accepted counts accepted without invented corpus counts",lambda:module.validate_import_proof(proof,parent))
for key,value in (("importer_sha256","b"*64),("archive_sha256","b"*64),("full_archive_hash_verified",False),
                  ("TDT_decode_grouping",True),("Dev_Test_archive_audio_or_text_imported",True)):
    changed=copy.deepcopy(proof);changed[key]=value
    rejected("Reject changed proof "+key,lambda changed=changed:module.validate_import_proof(changed,parent))
changed=copy.deepcopy(proof);changed["staging"]["raw_Train_files_staged"]=14
rejected("Reject all-pool staging disguised as limited pilot",lambda:module.validate_import_proof(changed,parent))
rejected("Full attempt required rejects subset pilot honestly",lambda:module.validate_import_proof(proof,parent,require_full_train_attempt=True))
changed=copy.deepcopy(proof);changed["protected_existing_manifests_sha256"].pop(next(iter(parent["heldout_manifests_sha256"])))
rejected("Reject missing protected heldout manifest",lambda:module.validate_import_proof(changed,parent))

manifest=module.DATA/"synthetic_tsc_candidate"/"accepted_train.jsonl"
stem="fixture"
wav=f"ISSAI_TSC_218/Train/{stem}.wav"
row={"original_audio_path":wav,"original_transcript_path":f"ISSAI_TSC_218/Train/{stem}.txt",
     "dataset":module.TSC_REPO,"dataset_revision":module.TSC_REVISION,"dataset_config":"default",
     "dataset_split":"train","training_role":"train","language":"tr","dataset_file":"ISSAI_TSC_218.tar.gz",
     "archive_sha256":module.ARCHIVE_SHA,"data_expansion_role":"tsc_train","sample_rate":16000,
     "speaker_id":None,"speaker_identity_verified":False,"sentence_id":None,
     "source_id":f"{module.TSC_REPO}:default:train:None:{wav}",
     "reference_raw_text":" IĞDIR 123 İŞ\n","text":"IĞDIR 123 İŞ","reference":"IĞDIR 123 İŞ",
     "reference_normalized":"ığdır 123 iş","audio_sha256":digest,"wav_sha256":digest,
     "canonical_pcm_sha256":digest,"source_transcript_sha256":digest,"label_ids_sha256":digest,
     "target_tokens":8,"target_max_id":8198,"duration":2.,"source_duration":2.,
     "audio_filepath":str(manifest.parent/"audio"/(hashlib.sha256(f"{module.TSC_REVISION}:{stem}".encode()).hexdigest()[:24]+".wav"))}
accepted("Exact synthetic TRAIN pair/raw Turkish text/row provenance accepted",lambda:module.validate_tsc_row(row,manifest))
for key,value in (("training_role","validation"),("original_audio_path","ISSAI_TSC_218/Dev/fixture.wav"),
                  ("original_transcript_path","ISSAI_TSC_218/Train/other.txt"),("text","igdir is"),
                  ("canonical_pcm_sha256","not-a-hash"),("target_tokens",181),("target_max_id",8199),
                  ("duration",31.),("speaker_identity_verified",True),("audio_filepath",str(module.DATA/"outside.wav"))):
    changed=copy.deepcopy(row);changed[key]=value
    rejected("Reject invalid TSC row "+key,lambda changed=changed:module.validate_tsc_row(changed,manifest))
cv={"dataset":module.CV_REPO,"dataset_revision":module.CV_REVISION,"dataset_config":"default",
    "dataset_split":"validated","training_role":"train","language":"tr",
    "dataset_file":"data/validated-00000-of-00001.parquet","source_id":f"{module.CV_REPO}:default:validated:fixture"}
accepted("Existing validated TRAIN-role policy retained",lambda:module.validate_original_source(cv))
changed=dict(cv,training_role="validation")
rejected("Existing validated role cannot become implicit evaluation training",lambda:module.validate_original_source(changed))
imports={alias.name.split('.')[0] for node in ast.walk(ast.parse(source.read_text())) if isinstance(node,ast.Import) for alias in node.names}
imports.update(node.module.split('.')[0] for node in ast.walk(ast.parse(source.read_text())) if isinstance(node,ast.ImportFrom) and node.module)
assert not imports&{"torch","transformers","nemo","numpy","scipy","soundfile","tokenizers","requests","urllib"}
checks.append("No archive/audio/ML/network runtime imports")
report={"schema":1,"kind":"TSC_supplemental_policy_tiny_metadata_verification_v1","passed":True,
        "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"check_count":len(checks),"checks":checks,
        "synthetic_metadata_only":True,"real_manifest_archive_audio_reads":False,"model_GPU_network_used":False,
        "actual_TSC_import_and_combined_pool_validation_pending":True,"future_counts_not_claimed":True}
target=Path(sys.argv[1]).resolve() if len(sys.argv)==2 else OUTPUT/"TSC_supplemental_policy_tiny_verification_v1.json"
assert len(sys.argv)<=2 and target.resolve().is_relative_to(OUTPUT.resolve())
with target.open("x",encoding="utf-8") as stream:json.dump(report,stream,indent=2)
print(json.dumps({"passed":True,"checks":len(checks),"source_sha256":report["source_sha256"]}))
