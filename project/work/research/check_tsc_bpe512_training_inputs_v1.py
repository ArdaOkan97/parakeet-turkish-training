"""Small stdlib-only consumer fixture; no whole runtime pool or ML/audio import."""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
MODULE = OUT / "tsc_bpe512_training_inputs_v1.py"
spec = importlib.util.spec_from_file_location("tsc_bpe512_training_inputs_v1", MODULE)
reader = importlib.util.module_from_spec(spec); spec.loader.exec_module(reader)
checks = []


def check(condition, name):
    if not condition: raise AssertionError(name)
    checks.append(name)


def rejects(call, name):
    try: call()
    except (ValueError, KeyError, TypeError): checks.append(name); return
    raise AssertionError("Not rejected: " + name)


class Encoder:
    def __init__(self, labels): self.labels = labels
    def encode(self, text, *, add_special_tokens):
        check(add_special_tokens is False, "No inserted special tokens")
        return self.labels[text].copy()


def dump_lines(path, items):
    raw = b"".join(reader.canonical(item)+b"\n" for item in items)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def fixtures(directory):
    # Tiny synthetic label streams; dimensions/provenance schema follow genuine headers.
    rows = []
    for index, group in enumerate(("cv", "fleurs", "tsc")):
        dataset = {"cv":"ysdede/commonvoice_17_tr_fixed", "fleurs":"google/fleurs", "tsc":"issai/Turkish_Speech_Corpus"}[group]
        revision = {"cv":"9d8a025f1ae7c5a2f5c43298c7c4910b03995a4b", "fleurs":"70bb2e84b976b7e960aa89f1c648e09c59f894dd", "tsc":"d79ea01392882322e8f46657cb635236af25b6d4"}[group]
        rows.append({"source_id":("issai/Turkish_Speech_Corpus:ISSAI_TSC_218/Train/00001.wav" if group=="tsc" else dataset+":train:"+str(index)),
            "language":"tr", "dataset":dataset, "dataset_revision":revision,
            "dataset_config":"tr_tr" if group=="fleurs" else "default", "dataset_split":"train", "training_role":"train",
            "audio_filepath":str(directory/(str(index)+".wav")), "text":"fixture "+str(index),
            "reference":"fixture "+str(index), "reference_normalized":"fixture "+str(index),
            "duration":1.0, "sample_rate":16000, "audio_sha256":"a"*64, "wav_sha256":"b"*64})
    old = {row["text"]:[10+index,20+index,30+index] for index,row in enumerate(rows)}
    new = {row["text"]:[10+index,8199+index] for index,row in enumerate(rows)}
    parents = [{"source_id":row["source_id"], "actual_PCM_frames":16000, "valid_mel_frames":100,
        "encoder_frames":13, "target_tokens":3, "label_ids_sha256":reader.label_sha(old[row["text"]]),
        "teacher_feature_values_required_for_TR_preservation":False} for row in rows]
    parents[2]["canonical_WAV_stat_at_preparation"]={"size_bytes":32044,"mtime_ns":123456}
    items = []
    for index,(row,parent) in enumerate(zip(rows,parents)):
        labels=new[row["text"]]
        item={**parent, "kind":reader.CANDIDATE_KIND, "manifest_row_index":index,
            "target_tokens":len(labels), "label_ids_sha256":reader.label_sha(labels), "label_ids":labels,
            "baseline_target_tokens":3,"baseline_label_ids_sha256":parent["label_ids_sha256"],
            "training_manifest_row_sha256":hashlib.sha256(reader.canonical(row)).hexdigest(),
            "text_sha256":hashlib.sha256(row["text"].encode()).hexdigest()}
        item.update({key:row[key] for key in reader.SOURCE_KEYS if key in row}); items.append(item)
    paths=tuple(directory/name for name in ("manifest.jsonl","parent.jsonl","candidate.jsonl"))
    digests=[dump_lines(path,data) for path,data in zip(paths,(rows,parents,items))]
    report={"clips":3,"prefix_clips":2,"group_counts":{"cv":1,"fleurs":1,"tsc":1},
        "manifest":{"sha256":digests[0]},"parent_metadata":{"sha256":digests[1]},"metadata_sha256":digests[2],
        "ordered_source_ids_newline_sha256":hashlib.sha256("".join(row["source_id"]+"\n" for row in rows).encode()).hexdigest(),
        "canonical_manifest_rows_sha256":hashlib.sha256(b"".join(reader.canonical(row)+b"\n" for row in rows)).hexdigest(),
        "native_joint_guard":{"maximum_candidate_bytes":13*3*8716*4}}
    return rows,parents,items,Encoder(old),Encoder(new),paths,report


report=reader.preflight()
check(report["clips"]==253798 and report["metadata_sha256"]==reader.METADATA_SHA, "Actual immutable preparation report and small asset hashes checked")
check(report["counts"]["candidate_roundtrip_passed"]==253798, "Prepared full count, not original74883-only")
with reader.MANIFEST_PATH.open("rb") as a, reader.PARENT_PATH.open("rb") as b, reader.METADATA_PATH.open("rb") as c:
    raw_headers=[stream.readline() for stream in (a,b,c)]
header,parent,item=map(json.loads,raw_headers)
check(header["source_id"]==parent["source_id"]==item["source_id"] and item["manifest_row_index"]==0, "Actual first header ordered source/index agreement")
check(hashlib.sha256(reader.canonical(header)).hexdigest()==item["training_manifest_row_sha256"], "Actual header canonical raw-manifest hash")
check(reader.label_sha(item["label_ids"])==item["label_ids_sha256"] and len(item["label_ids"])==item["target_tokens"]==6
    and parent["target_tokens"]==item["baseline_target_tokens"]==9
    and parent["label_ids_sha256"]==item["baseline_label_ids_sha256"], "Actual header candidate IDs/hash and parent originalU/hash")
check(all(item[k]==v for k,v in parent.items() if k not in ("target_tokens","label_ids_sha256")), "Actual copied header non-target/frame values")
rejects(lambda:reader.read_pinned_json(reader.REPORT_PATH,"0"*64), "Changed preparation bytes rejected")
rejects(lambda:reader.preflight(reader.PARENT_PATH,reader.REPORT_PATH), "Original metadata cannot substitute candidate")

with tempfile.TemporaryDirectory(prefix="tsc_bpe512_fixture_",dir=ROOT/"work/research") as temp:
    directory=Path(temp)
    rows,parents,items,old,new,paths,fixture=fixtures(directory)
    ids=[id(row) for row in rows]; list_id=id(rows); opened=[]; original_open=Path.open
    def count_open(path,*args,**kwargs):
        if path in paths: opened.append((path,args[0] if args else kwargs.get("mode","r")))
        return original_open(path,*args,**kwargs)
    with patch.object(Path,"open",count_open):
        result=reader._admit_stream(rows,old,new,fixture,*paths,1_700_000_000)
    check(id(result)==list_id and [id(row) for row in result]==ids, "Same accepted pool list and row dictionaries, no duplicate array")
    check(opened==[(path,"rb") for path in paths], "Each candidate/parent/manifest opened once for binary streaming")
    check(all(row["bpe512_TR_admission_complete"] for row in rows), "Admission markers only after complete stream hashes/counts")
    check(all(row["training_input_type"]=="TR_frame_target_metadata_v1" and row["target_tokens"]==2
        and row["tr_metadata_labels"]==new.labels[row["text"]] and row["tr_metadata_labels"][1]>=8199 for row in rows),
        "Original physical carrier interface holds genuine candidate new IDs")
    check(all("cache_path" not in row and "features" not in row for row in rows), "TR creates no fake teacher features/cache")
    # The existing loader's actual function is extracted by AST; fake tensorizer only.
    tree=ast.parse((OUT/"tr_frame_metadata_v1.py").read_text(encoding="utf-8"))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="load_training_item")
    class Tensorizer:
        long="int64"
        @staticmethod
        def tensor(value,*,dtype): return {"ids":value.copy(),"dtype":dtype}
    namespace={"KIND":reader.KIND,"torch":Tensorizer,"Path":Path}
    exec(compile(ast.Module(body=[function],type_ignores=[]),"actual_original_loader_AST","exec"),namespace)
    loaded=namespace["load_training_item"](rows[0])
    check(loaded["kind"]==reader.KIND and loaded["labels"]["ids"]==rows[0]["tr_metadata_labels"]
        and loaded["encoder_frames"]==13, "Actual frozen loader function accepts unchanged carrier with newID tensorization stub")
    before_replay={"language":"de","source_id":"external-replay","cache_path":"original.pt"}
    replay=copy.deepcopy(before_replay)
    rejects(lambda:reader.validate_training_item(replay), "Replay rejected by candidate-only validation")
    check(replay==before_replay, "External replay dictionary remains unchanged")
    rows,parents,items,old,new,paths,fixture=fixtures(directory)
    def validate(r=None,p=None,i=None,bound=1_700_000_000):
        return reader.validate_row(rows[0] if r is None else r,rows[0],parents[0] if p is None else p,
            items[0] if i is None else i,0,old,new,bound)
    for field,value in (("text","changed text"),("reference","changed reference"),("audio_filepath","changed.wav"),
                        ("source_id","changedsource"),("duration",1.01)):
        r=copy.deepcopy(rows[0]);r[field]=value;rejects(lambda r=r:validate(r=r),"Reject accepted "+field+" alteration")
    for field,value in (("manifest_row_index",1),("kind","old74883kind"),("baseline_target_tokens",4),
        ("baseline_label_ids_sha256","0"*64),("target_tokens",3),("label_ids_sha256","0"*64),
        ("training_manifest_row_sha256","0"*64),("text_sha256","0"*64),("encoder_frames",14)):
        i=copy.deepcopy(items[0]);i[field]=value;rejects(lambda i=i:validate(i=i),"Reject candidate "+field+" alteration")
    for labels in ([8192],[8711],[True],[1]*181,[]):
        rejects(lambda labels=labels:reader._validate_ids(labels,8711),"Reject invalid labels "+str(labels[:2])+"/length"+str(len(labels)))
    reject=copy.deepcopy(items[0]);reject["label_ids"]=[12,8200];reject["label_ids_sha256"]=reader.label_sha(reject["label_ids"])
    rejects(lambda:validate(i=reject), "Actual runtime candidate IDs differ even with self-consistent hash")
    reject=copy.deepcopy(parents[0]);reject["label_ids_sha256"]="0"*64
    rejects(lambda:validate(p=reject), "Actual baseline re-encoding disagrees with parenthash")
    rejects(lambda:validate(bound=1000), "Native full joint arithmetic overbound aborts, no dropping")
    rejects(lambda:reader._admit_stream(rows[:2],old,new,fixture,*paths,1_700_000_000), "Cropped pool rejected before stream")
    bad_fixture=copy.deepcopy(fixture);bad_fixture["metadata_sha256"]="0"*64
    rejects(lambda:reader._admit_stream(rows,old,new,bad_fixture,*paths,1_700_000_000), "Full streamed file hash mutation rejected")
    check(all(row["bpe512_TR_admission_complete"] is False for row in rows), "Failed final digest cannot publish completed admission markers")

check(not any(name in sys.modules for name in ("torch","transformers","nemo","soundfile","torchaudio","tokenizers")),
    "No ML, tokenizer, audio, model or GPU libraries imported")
payload={"schema":1,"kind":"TSC_BPE512_training_inputs_stdlib_tiny_verification_v1","passed":True,
    "created_at_utc":datetime.now(timezone.utc).isoformat(),"checks":len(checks),"check_labels":checks,
    "source":{"path":str(MODULE),"sha256":hashlib.sha256(MODULE.read_bytes()).hexdigest()},
    "fixture":{"path":str(Path(__file__).resolve()),"sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    "actual_preparation_report":{"path":str(reader.REPORT_PATH),"sha256":reader.REPORT_SHA},
    "actual_header_rows_read":1,"actual_candidate_metadata_whole_file_hash_or_pool_admission_executed":False,
    "actual_full_report_and_tokenizer_asset_hashes_checked":True,
    "tiny_stream_rows":3,"tiny_labels":"Synthetic known lists; not runtime tokenizer encoding/model evidence.",
    "physical_loader_check":"Frozen function AST; tensorization stub; no Torch/model/audio import.",
    "replay_modified":False,"GPU_or_ASR_activation":False,"whole_runtime_pool_loaded":False,
    "integration_limits":["Future worker must invoke approved original corpus/heldout/replay audit first, then consumer before updates.",
        "Future config must bind candidate report/assets and explicit target-dependent sampler transition; same-science claim is false.",
        "Existing original TR loader/carrier can carry8711 labels, but frozen74883 BPE collator/worker admission cannot consume combined pool.",
        "No real optimizer/model row extension, full runtime consumer, initial300 DEV equality or measured GPU-fit proof in this check."]}
output=OUT/"TSC_BPE512_training_inputs_CPU_verification_v1.json"
with output.open("x",encoding="utf-8") as target:json.dump(payload,target,ensure_ascii=False,indent=2);target.write("\n")
print(json.dumps({"passed":True,"checks":len(checks),"source_sha256":payload["source"]["sha256"],"output":str(output)}))
