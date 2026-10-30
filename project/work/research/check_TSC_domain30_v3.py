"""Small stdlib source/sampler/streamed metadata fixtures, never ML/data inference."""
import argparse
import ast
import copy
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/"outputs/parakeet_turkish"
sys.path.insert(0,str(HERE))
import tsc_domain_sampler_v3 as new
import turkish_domain_sampler_v2 as old
import tsc_domain_training_identity_v3 as identity
import tsc_training_inputs_v2 as inputs


def main(args):
    checks=[]
    def check(value,name):
        assert value,name
        checks.append(name)
    def reject(call,name):
        try:call()
        except (ValueError,KeyError,AssertionError):checks.append(name)
        else:raise AssertionError(name)
    def tree(name):return ast.parse((HERE/name).read_text(encoding="utf-8"))
    def functions(name):return {n.name:n for n in tree(name).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    original=functions("train_lora_batched_data_v11.py")
    ns={"math":math,"ids_sha":lambda rows:hashlib.sha256(json.dumps(sorted(r["source_id"] for r in rows)).encode()).hexdigest()}
    exec(compile(ast.Module(body=[original["BucketCycler"]],type_ignores=[]),"<frozen BucketCycler AST>","exec"),ns)
    cycler=ns["BucketCycler"]
    def row(sid,group,i):
        dataset={"fleurs":"google/fleurs","cv":"ysdede/commonvoice_17_tr_fixed","tsc":new.TSC_REPO}[group]
        revision={"fleurs":"70bb2e84b976b7e960aa89f1c648e09c59f894dd","cv":"9d8a025f1ae7c5a2f5c43298c7c4910b03995a4b","tsc":new.TSC_REVISION}[group]
        return dict(source_id=sid,dataset=dataset,dataset_revision=revision,dataset_config="tr_tr" if group=="fleurs" else "default",
            dataset_split="train",training_role="train",language="tr",encoder_frames=20+i%7,target_tokens=2+i%3,
            duration=1.+i/10,text="örnek",audio_sha256="a"*64,label_ids_sha256="b"*64)
    original_tr=[row("f"+str(i),"fleurs",i) for i in range(7)]+[row("c"+str(i),"cv",i) for i in range(19)]
    expanded=original_tr+[row("t"+str(i),"tsc",i) for i in range(23)]
    replay=[dict(row(lang+str(i),"fleurs",i),language=lang) for lang in old.LANGUAGES for i in range(8)]
    common=dict(seed=17,replay_seed=23,sampling_mode="domain30",batch_size=8,accumulation=2,replay_fraction=.5,
        fleurs_fraction=.3,bucket_size=8,cycler_factory=cycler,cycler_source_sha256=digest(HERE/"train_lora_batched_data_v11.py"))
    def make(rows=expanded):return new.TSCDomainBalancedBucketSampler(rows,replay,**common)
    before=old.DomainBalancedBucketSampler(original_tr,replay,**common);same=make(original_tr);larger=make()
    sequences=[]
    for step in range(1,35):
        b=before.next_batches();s=same.next_batches();e=larger.next_batches()
        check([[r["source_id"] for r in group] for group in b]==[[r["source_id"] for r in group] for group in s],f"original pool same TR/replay sequence update{step}")
        check([r["source_id"] for r in b[1]]==[r["source_id"] for r in e[1]],f"expanded TR leaves independent replay order unchanged update{step}")
        coverage=larger.coverage()
        check(coverage["domain"]["fleurs"]["draws"]==step*8*3//10 and coverage["domain"]["cv_tsc"]["draws"]==step*8-step*8*3//10,f"exact cumulative30/70 quota update{step}")
        sequences.extend(r["source_id"] for r in e[0] if r["source_id"][0] in "ct")
    cycle=make().turkish_streams["cv_tsc"]
    check(len({cycle.next()["source_id"] for _ in range(42)})==42,"first other-domain shuffled cycle without replacement")
    check(set(larger.coverage()["source_groups"])=={"fleurs","cv","tsc"},"three source groups counted honestly")
    check(new.domain_horizon(2037,251761)==44958 and new.domain_horizon(2037,72846)==13009,"exact actual combined/original horizons")
    h=new.domain_horizon(2037,251761)
    check((h-1)*8-(h-1)*8*3//10==251760 and h*8-h*8*3//10==251765,"prior horizon misses one; final horizon4 other repeats")
    check(larger.updates_for_epochs(1)==new.domain_horizon(7,42),"class horizon uses actual domain pools")
    saved=copy.deepcopy(larger.state_dict());restored=make();restored.load_state_dict(saved)
    check(saved==restored.state_dict(),"complete sampler/RNG/source-group state restored")
    for step in range(20):
        check(larger.next_batches()==restored.next_batches(),f"exact next batch after resume{step}")
    check(larger.state_dict()==restored.state_dict(),"resumed all counters/audio/RNG exact")
    for key in ("updates","domain_draws","source_group_draws","source_group_seen","identity"):
        bad=copy.deepcopy(saved)
        if key=="updates":bad[key]+=1
        elif key=="source_group_seen":bad[key]["tsc"]=[]
        elif key=="identity":bad[key]["replay_seed"]+=1
        else:bad[key][next(iter(bad[key]))]+=1
        reject(lambda:make().load_state_dict(bad),"tampered "+key+" rejected")
    bad=row("bad","tsc",0);bad["dataset_split"]="test"
    reject(lambda:make(original_tr+[bad]),"nonTRAIN TSC never enters sampler")

    source=json.loads((HERE/"runs/adaptive_20261011_hf_domain30_01_domain30/run_config.json").read_text(encoding="utf-8"))
    closure={"allow_source_domain30_expansion":True,"source_sampling_metadata":source["config_identity"]["domain_sampling_identity"]}
    check(identity.approved_source_recipe(source,closure)==source["config_identity"]["training_arguments"],"actual source35 recipe fields preserved")
    reject(lambda:identity.approved_source_recipe(source,{**closure,"allow_source_domain30_expansion":False}),"unapproved sampler expansion rejected")
    altered=copy.deepcopy(source);altered["config_identity"]["training_arguments"]["fleurs_fraction"]=.2
    reject(lambda:identity.approved_source_recipe(altered,closure),"source fraction change rejected")

    scratch=ROOT/"work/research"/("TSC_stream_fixture_"+uuid.uuid4().hex);scratch.mkdir(exist_ok=False)
    token_json={"fixture":True};semantic=hashlib.sha256(json.dumps(token_json,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    class Tokenizer:
        _tokenizer=SimpleNamespace(to_str=lambda:json.dumps(token_json))
        def __call__(self,text):return {"input_ids":[3,4]}
    processor=SimpleNamespace(tokenizer=Tokenizer());raw=[row("x"+str(i),"tsc",i) for i in range(3)]
    metadata=[]
    for r in raw:
        audio=scratch/(r["source_id"]+".synthetic_stat");audio.write_bytes(b"no audio decoded")
        r["audio_filepath"]=str(audio);pcm=int(round(r["duration"]*16000));stat=audio.stat()
        metadata.append(dict(source_id=r["source_id"],actual_PCM_frames=pcm,valid_mel_frames=pcm//160,encoder_frames=(pcm//160+7)//8,
            target_tokens=2,label_ids_sha256=hashlib.sha256(json.dumps([3,4],sort_keys=True).encode()).hexdigest(),
            canonical_WAV_stat_at_preparation={"size_bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns}))
    metadata_path=scratch/"metadata.jsonl";proof_path=scratch/"proof.json";raw_path=scratch/"raw.jsonl"
    raw_path.write_text("".join(json.dumps(r)+"\n" for r in raw),encoding="utf-8");raw_hash=digest(raw_path)
    def write_metadata(items):
        metadata_path.write_text("".join(json.dumps(i)+"\n" for i in items),encoding="utf-8")
        proof={"kind":"TSC_supplemental_TR_frame_target_inputs_v2","passed":True,"helper_sha256":digest(HERE/"tsc_training_inputs_v2.py"),
            "clips":3,"metadata_sha256":digest(metadata_path),"files":{"TR_metadata":inputs.file_record(metadata_path)}}
        proof_path.write_text(json.dumps(proof),encoding="utf-8")
    semantic_original=inputs.policy.TOKENIZER_SEMANTIC_SHA;inputs.policy.TOKENIZER_SEMANTIC_SHA=semantic
    try:
        write_metadata(metadata);old_rows=copy.deepcopy(raw);new_rows=copy.deepcopy(raw)
        expected=inputs.annotate_tr_frame_metadata(processor,old_rows,metadata_path,proof_path,180)
        actual=identity.annotate_tr_frame_metadata(processor,new_rows,metadata_path,proof_path,180)
        check(actual==expected,"streamed annotations equal frozen v2 field-for-field")
        check(actual is new_rows and actual[0] is new_rows[0],"no second augmented row dictionary array retained")
        check(old_rows==raw and digest(raw_path)==raw_hash,"independent protected source objects/file unchanged")
        for field,value in (("source_id","wrong"),("label_ids_sha256","0"*64),("encoder_frames",1)):
            bad=copy.deepcopy(metadata);bad[0][field]=value;write_metadata(bad)
            reject(lambda:identity.annotate_tr_frame_metadata(processor,copy.deepcopy(raw),metadata_path,proof_path,180),"changed "+field+" rejected")
        bad=copy.deepcopy(metadata);bad[0]["canonical_WAV_stat_at_preparation"]["size_bytes"]+=1;write_metadata(bad)
        reject(lambda:identity.annotate_tr_frame_metadata(processor,copy.deepcopy(raw),metadata_path,proof_path,180),"changed canonical WAV stat rejected")
        write_metadata(metadata+[metadata[0]])
        reject(lambda:identity.annotate_tr_frame_metadata(processor,copy.deepcopy(raw),metadata_path,proof_path,180),"extra streamed metadata row rejected")
    finally:inputs.policy.TOKENIZER_SEMANTIC_SHA=semantic_original
    v2=functions("train_lora_batched_tsc_v2.py");v3=functions("train_lora_batched_tsc_v3.py")
    for name in ("BucketCycler","development_gate","lr_factor","save_training_state","validate_trainable_checkpoint","annotate_cache_lengths","next_evaluation_epoch","native_loss_function"):
        check(ast.dump(v2[name],include_attributes=False)==ast.dump(v3[name],include_attributes=False),"unchanged math/state function "+name)
    micro=lambda f:next(n for n in ast.walk(f["run"]) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=="group")
    check(ast.dump(micro(v2),include_attributes=False)==ast.dump(micro(v3),include_attributes=False),"whole microbatch forward/backward/reclaim math unchanged")
    check(ast.dump(v3["BucketCycler"],include_attributes=False)==ast.dump(original["BucketCycler"],include_attributes=False),"source cycler AST provenance exact")
    sync=[n.lineno for n in ast.walk(v3["run"]) if isinstance(n,ast.Call) and ast.unparse(n.func)=="torch.cuda.synchronize"]
    check(len(sync)==1,"one source-bound optimizer end synchronize")
    run_source=ast.unparse(v3["run"])
    check(run_source.index("audit_tsc_training_inputs(args)")<run_source.index("load_model()")<run_source.index("source_isolation(tr, replay, heldout, public_data_audit)")<run_source.index("annotate_tr_frame_metadata("),"raw audit/isolation precede own transient annotation")
    check("shared RNG is retained" not in run_source and "target_replay_rng_independent_of_turkish" in run_source,"new independent sampling metadata overrides only v2 observations")
    check(not any(n in sys.modules for n in ("torch","nemo","transformers","tokenizers")),"no ML/tokenizer/model imports")
    report={"schema":1,"kind":"TSC_domain30_v3_sampler_streaming_source_CPU_verification","passed":True,
        "created_at_utc":datetime.now(timezone.utc).isoformat(),"checks_passed":len(checks),"checks":checks,
        "source_sha256":{n:digest(HERE/n) for n in ("tsc_domain_sampler_v3.py","tsc_domain_training_identity_v3.py","train_lora_batched_tsc_v3.py","tsc_training_inputs_v2.py","train_lora_batched_tsc_v2.py","turkish_domain_sampler_v2.py")},
        "optimizer_end_synchronize_line":sync[0],"actual_combined_domain_horizon":44958,"actual_corpus_or_GPU_execution":False,
        "scope":"Synthetic sampler/metadata/stat, bounded current config and AST; full data/closure/actual GPU adoption remains ROOT-owned"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ("passed","checks_passed","optimizer_end_synchronize_line","source_sha256")}))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);main(parser.parse_args())
