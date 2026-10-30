"""Inactive boundary regression plus narrow reviewed resource stop; no ML/audio/checkpoint access."""
import copy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/"outputs/parakeet_turkish"
sys.path.insert(0,str(OUT))
import tsc_bpe_acquisition_transition_v1 as boundary
checks=[]
def check(value,label):
    if not value:raise AssertionError(label)
    checks.append(label)
def rejects(call,label):
    try:call()
    except (ValueError,KeyError,TypeError):checks.append(label);return
    raise AssertionError("Not rejected: "+label)

# Actual source/config text is small and read-only; no live source state is admitted.
config=json.loads((OUT/"runs"/boundary.PARENT_RUN/"run_config.json").read_text(encoding="utf-8"))
identity=config["config_identity"];boundary.validate_layout(identity["lora_layout"])
checks.append("Actual current read-only source layout remains rank8/alpha16/all24QKVO")
candidate,donors=boundary.candidate_assets()
check(len(donors)==512 and all(d["HF_id"]==8199+i and all(0<=leaf<8199 and leaf!=8192 for leaf in d["original_leaf_donor_ids"])
    and d["bias_penalty_applied_once"]==2 for i,d in enumerate(donors)),"Actual frozen512 originalleaf donor plans/one penalty checked without tensors")
check(len(candidate["model"]["vocab"])==8711,"Actual candidate geometry8711")
receipt={"schema":1,"kind":"ROOT_owned_closed_TSC_best_for_BPE512_acquisition_v1","status":"actual_CLOSED_qualified_best",
    "source_run_id":boundary.PARENT_RUN,"target_run_id":"inactive_fixture_BPE512","allow_BPE512_row_only_acquisition":True,
    "selected_step":2500,"completed_steps":3000,"retention_gate_passed_at_selected_step":True,
    "declared_acquisition":boundary.declared_acquisition(),
    "owned_actual_exits":[{"session_id":1,"role":"trainer_controller","exit_code":0,"tool_chunk":"FIXTURE_ONLY_NOT_ACTUAL_CLOSURE"}]}
check(boundary.validate_receipt(receipt,receipt["target_run_id"]) is receipt,"Dynamic2500BEST schema admits declaredsyntheticclosure only")
for field,value in (("source_run_id","otherlineage"),("selected_step",3001),("status","running"),
    ("retention_gate_passed_at_selected_step",False),("allow_BPE512_row_only_acquisition",False)):
    bad=copy.deepcopy(receipt);bad[field]=value;rejects(lambda bad=bad:boundary.validate_receipt(bad,receipt["target_run_id"]),"Rejectreceipt "+field)
bad=copy.deepcopy(receipt);bad["declared_acquisition"]["learning_rate"]=.0003
rejects(lambda:boundary.validate_receipt(bad,receipt["target_run_id"]),"Rejectunapproved activeLR")
bad=copy.deepcopy(receipt);bad["owned_actual_exits"][0]["exit_code"]=1
rejects(lambda:boundary.validate_receipt(bad,receipt["target_run_id"]),"Rejectresource/internal parentexit")
bad=copy.deepcopy(receipt);bad["owned_actual_exits"][0].pop("tool_chunk")
rejects(lambda:boundary.validate_receipt(bad,receipt["target_run_id"]),"Rejectinferred/missing actualownerexit")

# Pure shape/schema placeholders cannot establish any tensor/state preservation.
history=[{"step":i,"validation":{"retention_gate_passed":True,"turkish_wer":.4}} for i in range(1,3)]
context={"source_selected_step":2,"parent_identity":identity,"source_history_prefix_sha256":boundary.canonical(history),
    "selection":{"turkish_validation_wer":.4},"source_recipe":identity["training_arguments"],"config":config,
    "receipt":{"artifacts":{"best_checkpoint":{"path":str(ROOT/"work/inactive_fixture_BEST.pt")}}}}
state={"schema":2,"run_id":boundary.PARENT_RUN,"step":2,"selected_step":2,"config_identity":identity,
    "weights":{str(i):"schema_placeholder_not_tensor" for i in range(207)},
    "optimizer":{"state":{i:{"step":2,"exp_avg":"placeholder","exp_avg_sq":"placeholder"} for i in range(207)},
        "param_groups":[{"params":list(range(15)),"lr":1e-5},{"params":list(range(15,207)),"lr":3e-4}]},
    "sampler":{"identity":identity["domain_sampling_identity"],"updates":2},"lora_metadata":identity["lora_layout"],
    "history":history,"best_score":.4,"retention_failures":0}
for key in ("python_rng","torch_rng","cuda_rng","patience","next_eval_epoch","train_wall_seconds","audio_seconds",
    "language_audio_seconds","evaluation_wall_seconds","initial_gate"):state[key]="schema_placeholder"
check(boundary.validate_parent_state(state,context)["first_acquisition_update"]==3,"BESTdynamicboundary/schema history usesselected+1")
before=copy.deepcopy(state);boundary.validate_parent_state(state,context)
check(state==before,"Pure schema admission doesnot rewrite dormantsource state")
for field,value in (("step",3),("selected_step",1),("retention_failures",1),("run_id","wrong")):
    bad=copy.deepcopy(state);bad[field]=value;rejects(lambda bad=bad:boundary.validate_parent_state(bad,context),"Rejectstate "+field)
bad=copy.deepcopy(state);bad["optimizer"]["state"].pop(0)
rejects(lambda:boundary.validate_parent_state(bad,context),"Rejectmissing dormantAdamstate")
bad=copy.deepcopy(state);bad["sampler"]["updates"]=3
rejects(lambda:boundary.validate_parent_state(bad,context),"Rejectlatest sampler mixed intoBEST")
bad=copy.deepcopy(state);bad["history"][0]["step"]=2
rejects(lambda:boundary.validate_parent_state(bad,context),"Rejecthistoryprefixrewrite")

# Injected tensor-like stdlib fixtures exercise the new audit's exact branches;
# they are not actual Torch tensors and do not establish any real state audit.
class FakeMask:
    def __init__(self,value):self.value=value
    def all(self):return self.value
class FakeTensor:
    def __init__(self,shape=(2,),value=0.,dtype="float32",finite=True):
        self.shape=shape;self.value=value;self.dtype=dtype;self.finite=finite;self.ndim=len(shape)
    def item(self):return self.value
    def __ge__(self,other):return FakeMask(self.value>=other)
class FakeTorch:
    float32="float32"
    @staticmethod
    def is_tensor(value):return isinstance(value,FakeTensor)
    @staticmethod
    def isfinite(value):return FakeMask(value.finite)
expected={**{"head"+str(i):FakeTensor() for i in range(15)},
          **{"encoder.adapter"+str(i)+".lora_A":FakeTensor() for i in range(192)}}
adam=copy.deepcopy(state)
for group,decay in zip(adam["optimizer"]["param_groups"],(.01,0.)):
    group.update(betas=(.9,.999),eps=1e-8,weight_decay=decay,amsgrad=False,
        maximize=False,capturable=False,differentiable=False,foreach=None,fused=None)
adam["optimizer"]["state"]={i:{"step":FakeTensor((),2),"exp_avg":FakeTensor(),"exp_avg_sq":FakeTensor(value=.1)} for i in range(207)}
adam_context={**context,"source_selected_step":2,"source_recipe":{**context["source_recipe"],"warmup_steps":1}}
audit=boundary.audit_source_optimizer(adam,adam_context,expected,torch_module=FakeTorch)
check(audit["source_Adam_states_checked"]==207 and audit["source_Adam_clock"]==2,
      "Injected stdlib tensor-like audit covers207 named moments and dynamic clock; not real Torch state")
for key,value in (("step",FakeTensor((),3)),("step",FakeTensor((1,),2)),
                  ("exp_avg",FakeTensor((3,))), ("exp_avg",FakeTensor(dtype="float16")),
                  ("exp_avg",FakeTensor(finite=False)),("exp_avg_sq",FakeTensor(value=-.1))):
    bad=copy.deepcopy(adam);bad["optimizer"]["state"][0][key]=value
    rejects(lambda bad=bad:boundary.audit_source_optimizer(bad,adam_context,expected,torch_module=FakeTorch),
        "Injected audit rejects moment "+key+" "+repr((value.shape,value.value,value.dtype,value.finite)))
bad=copy.deepcopy(adam);bad["optimizer"]["state"][0]["max_exp_avg_sq"]=FakeTensor()
rejects(lambda:boundary.audit_source_optimizer(bad,adam_context,expected,torch_module=FakeTorch),"Injected audit rejects extra moment key")
for key,value in (("lr",.00005),("betas",(.8,.999)),("eps",1e-6),("weight_decay",0.)):
    bad=copy.deepcopy(adam);bad["optimizer"]["param_groups"][0][key]=value
    rejects(lambda bad=bad:boundary.audit_source_optimizer(bad,adam_context,expected,torch_module=FakeTorch),"Injected audit rejects group "+key)
bad=copy.deepcopy(adam);bad["optimizer"]["param_groups"][0]["params"][0:2]=[1,0]
rejects(lambda:boundary.audit_source_optimizer(bad,adam_context,expected,torch_module=FakeTorch),"Injected audit rejects original serialized parameter order")

args=SimpleNamespace(**identity["training_arguments"],run_id="inactive_fixture_BPE512",resume_acquisition=ROOT/"work/inactive_fixture_BEST.pt",
    acquisition_source_receipt=ROOT/"work/inactive_fixture_receipt.json",resume_exact=None,resume_weights=None,
    resume_runtime_migration=None,resume_optimization=None,acquisition_lr=.0001,acquisition_max_updates=500,smoke_updates=None,
    native_max_joint_bytes=1850000000,candidate_dir=ROOT/"work/hf_turkish_bpe_candidate_512_v1",
    candidate_tr_metadata=ROOT/"work/data/TSC_BPE512_TRAIN_metadata_v1/candidate_TR_frame_target_metadata.jsonl",
    candidate_tr_metadata_proof=OUT/"TSC_BPE512_TRAIN_target_metadata_preparation_v1.json")
for key in ("tsc_input_proof","tsc_training_approval","source_closure","tr_manifest","replay_manifest","data_v3_proof",
    "tr_frame_metadata","tr_frame_metadata_proof","reuse_input_audit","supplement"):
    if not hasattr(args,key):setattr(args,key,config.get(key))
with patch.object(boundary,"sha",lambda p:"fixture-not-actual-receipt-hash"),patch.object(boundary,"_context",lambda *a:context):
    check(boundary.preflight(args) is context,"Actualpreflightargumentchecks afterexplicit mockedCLOSED context; noactualclosure claim")
    smoke=copy.copy(args);smoke.smoke_updates=1;check(boundary.preflight(smoke) is context,"Oneupdatesmoke keepssource44958 horizon")
    for field,value in (("head_lr",.00005),("lora_lr",.0001),("native_max_joint_bytes",1700000000),
        ("replay_seed",1),("acquisition_lr",.0003),("acquisition_max_updates",0),("smoke_updates",2),
        ("resume_exact",ROOT/"work/other.pt"),("candidate_tr_metadata",ROOT/"work/data/wrong.jsonl")):
        bad=copy.copy(args);setattr(bad,field,value);rejects(lambda bad=bad:boundary.preflight(bad),"Rejectscientific/phase option "+field)

# Saved CLOSED pre-headLR selected1500 source predictions test the true row schema.
# This is a self-comparison only, not the future actual GPU expanded-source gate.
folder=OUT/"results/adaptive_20261011_hf_TSC_domain30_01_TSC_step_1500"
summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
for language in boundary.LANGUAGES:
    rows=[json.loads(line) for line in (folder/("validation_"+language+".jsonl")).read_text(encoding="utf-8").splitlines() if line.strip()]
    asr=summary["validation_"+language]["asr"]
    result=boundary.compare_development(rows,copy.deepcopy(rows),asr,asr,language)
    check(result["utterances"]==50,"Actualsaved50 "+language+" fullschema/path/ASRselfcomparison")
    if language=="tr":
        for kind in ("hypothesis","reference","duration","token_id"):
            bad=copy.deepcopy(rows)
            if kind=="token_id":bad[0]["confidence"]["tokens"][0]["token_id"]=8200
            elif kind=="duration":bad[0]["duration"]+=.01
            else:bad[0][kind]+=" altered"
            rejects(lambda bad=bad:boundary.compare_development(rows,bad,asr,asr,language),"RejectactualDEV "+kind+" perturbation")


# Tiny synthetic JSON records exercise the exact reviewed-resource admission.
# These are deliberately not actual ROOT closure/state evidence.
import ast
import tempfile
with tempfile.TemporaryDirectory(dir=ROOT/"work/research",prefix="resource_review_boundary_fixture_") as directory:
    temp=Path(directory)
    status={"status":"resource_review_required","stop_category":"resource","resource_review_required":True,
        "child_exit_code":0,"cooperative_stop_reason":"system_commit_at_or_above95percent",
        "run_id":boundary.PARENT_RUN,"controller_sha256":boundary.PARENT_SOURCES["run_TSC_headlr_fork_v1.py"],
        "selected_step":2500,"completed_steps":3000}
    def write_record(path,value):
        path.write_text(json.dumps(value),encoding="utf-8");return boundary.record(path)
    original=temp/"original_status.json";archive=temp/"archived_status.json";review_path=temp/"root_review.json"
    status_record=write_record(original,status);archived_record=write_record(archive,status)
    owned={"session_id":34624,"exit_code":1,"chunk_id":"b84373","observed_at_utc":"2026-10-11T22:00:03Z"}
    review={"kind":"ROOT_headlr_memory_resource_stop_review_v1","passed":True,"source_run_id":boundary.PARENT_RUN,
        "source_controller_sha256":status["controller_sha256"],"owned_actual_exit":owned,
        "reviewed_status_artifact":status_record,"allow_BPE512_row_only_acquisition":True,"guards_retained":True}
    resource=copy.deepcopy(receipt)
    resource["artifacts"]={"controller_status":archived_record,"fresh_runtime":{"sha256":"fixture_runtime_hash"}}
    resource["original_artifact_paths"]={"controller_status":str(original)}
    resource["owned_actual_exits"]=[{**owned,"role":"trainer_controller"}]
    resource["resource_stop_review"]=write_record(review_path,review)
    completion={"kind":"TSC_headlr_fork_native_completion_v1","status":"complete","run_id":boundary.PARENT_RUN,
        "sources_unchanged":True,"fresh_runtime_sha256":"fixture_runtime_hash",
        "actual_dispatch_source_sha256":status["controller_sha256"],"pid":99}
    check(boundary.validate_receipt(resource,resource["target_run_id"]) is resource,"Explicit synthetic reviewed exit1 stays truthful1")
    check(boundary.validate_closed_runtime(resource,status,completion,{"pid":99}) is None,"Reviewed child0 nativecomplete terminal admitted by actual context guard function")
    check(resource["owned_actual_exits"][0]["exit_code"]==1 and json.loads(archive.read_text())["status"]=="resource_review_required",
        "Admission leaves archivedstatus and actualexit1 unchanged")
    bad=copy.deepcopy(resource);bad.pop("resource_stop_review")
    rejects(lambda:boundary.validate_receipt(bad,bad["target_run_id"]),"Exit1 without explicit review rejects")
    bad=copy.deepcopy(resource);bad["resource_stop_review"]["sha256"]="0"*64
    rejects(lambda:boundary.validate_receipt(bad,bad["target_run_id"]),"Changed review hash rejects")
    for field,value in (("passed",False),("kind","broad_resource_waiver"),("source_run_id","wrong"),
        ("source_controller_sha256","0"*64),("guards_retained",False),("allow_BPE512_row_only_acquisition",False)):
        changed=copy.deepcopy(review);changed[field]=value;bad=copy.deepcopy(resource)
        bad["resource_stop_review"]=write_record(temp/(field+".json"),changed)
        rejects(lambda bad=bad:boundary.validate_receipt(bad,bad["target_run_id"]),"Review rejects "+field)
    for field,value in (("session_id",1),("exit_code",0),("chunk_id","stale"),("observed_at_utc","2026-10-11T21:00:03Z")):
        changed=copy.deepcopy(review);changed["owned_actual_exit"][field]=value;bad=copy.deepcopy(resource)
        bad["resource_stop_review"]=write_record(temp/("exit_"+field+".json"),changed)
        rejects(lambda bad=bad:boundary.validate_receipt(bad,bad["target_run_id"]),"Review rejects ownedexit "+field)
    bad=copy.deepcopy(resource);bad["owned_actual_exits"][0]["exit_code"]=0
    rejects(lambda:boundary.validate_receipt(bad,bad["target_run_id"]),"Cannot rewrite reviewed controllerexit to0")
    for field,value in (("status","train_window_ended"),("stop_category","budget"),("resource_review_required",False),
        ("cooperative_stop_reason","actual_worker_process_commit_headroom_below2GB"),("child_exit_code",1)):
        changed=copy.deepcopy(status);changed[field]=value
        changed_record=write_record(temp/("status_"+field+".json"),changed)
        changed_review=copy.deepcopy(review);changed_review["reviewed_status_artifact"]=changed_record
        bad=copy.deepcopy(resource);bad["artifacts"]["controller_status"]=changed_record
        bad["original_artifact_paths"]["controller_status"]=changed_record["path"]
        bad["resource_stop_review"]=write_record(temp/("review_status_"+field+".json"),changed_review)
        rejects(lambda bad=bad:boundary.validate_receipt(bad,bad["target_run_id"]),"Reviewedstatus rejects "+field)
    bad=copy.deepcopy(resource);bad["artifacts"]["controller_status"]["sha256"]="0"*64
    rejects(lambda:boundary.validate_receipt(bad,bad["target_run_id"]),"Archivedstatus hash mismatch rejects")
    bad_complete=copy.deepcopy(completion);bad_complete["status"]="failed"
    rejects(lambda:boundary.validate_closed_runtime(resource,status,bad_complete,{"pid":99}),"Reviewed exit1 does not waive native completion")
    tree=ast.parse((OUT/"tsc_bpe_acquisition_transition_v1.py").read_text(encoding="utf-8"))
    context_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_context")
    check(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="validate_closed_runtime" for n in ast.walk(context_node)),
        "Actual context calls exact tested terminal/native guard")

check(not any(name in sys.modules for name in ("torch","transformers","nemo","tokenizers","soundfile","torchaudio")),"NoML/tokenizer/audio/GPU imports")
payload={"schema":1,"kind":"inactive_TSC_BPE512_acquisition_boundary_stdlib_verification_v3","passed":True,
    "created_at_utc":datetime.now(timezone.utc).isoformat(),"checks":len(checks),"check_labels":checks,
    "source":{"path":str(OUT/"tsc_bpe_acquisition_transition_v1.py"),"sha256":boundary.sha(OUT/"tsc_bpe_acquisition_transition_v1.py")},
    "fixture":{"path":str(Path(__file__).resolve()),"sha256":boundary.sha(__file__)},
    "actual_saved_DEV_schema_selfcomparisons":300,"actual_source_predictions_folder":str(folder),
    "actual_CURRENT_CLOSED_receipt_available_or_admitted":False,"actual_checkpoint_weights_read_or_expanded":False,
    "actual_source_tensor_Adam_sampler_RNG_integrity_verified":False,"actual_runtime_pool_loaded":False,
    "actual_GPU_source_candidate_equality_executed":False,"activation":False,
    "Adam_runtime_audit_test_uses_stdlib_tensor_like_mocks_not_Torch":True,
    "scope":"Puredeclaredrecipe/closure/fullstate schema rejection + injected tensor-like Adam branch checks + actualsmall savedDEVselfcomparison +frozen512donorplans only; not runtime state/model/ASR/GPU proof."}
target=OUT/"TSC_BPE512_acquisition_boundary_CPU_verification_v3.json"
with target.open("x",encoding="utf-8") as stream:json.dump(payload,stream,ensure_ascii=False,indent=2);stream.write("\n")
print(json.dumps({"passed":True,"checks":len(checks),"source_sha256":payload["source"]["sha256"],"output":str(target)}))
