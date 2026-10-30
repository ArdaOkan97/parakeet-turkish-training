"""Small source/API guards; no model, Torch, audio, pool scan or GPU imports."""
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/parakeet_turkish"
sys.path.insert(0,str(OUT))
import hf_bpe512_training_v1 as helper


def reject(call):
    try:call()
    except (ValueError,KeyError):return
    raise AssertionError("Corruption accepted")


def functions(path):
    tree=ast.parse(path.read_text(encoding="utf-8"))
    return {node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}


def main():
    candidate,additions=helper.candidate_assets(ROOT/"work/hf_turkish_bpe_candidate_512_v1")
    original=copy.deepcopy(candidate)
    original["model"]["vocab"]={piece:index for piece,index in candidate["model"]["vocab"].items() if index<8199}
    original["model"]["merges"]=candidate["model"]["merges"][:-512]
    assert helper.semantic_sha(original)==helper.BASELINE_SEMANTIC_SHA
    donors=helper.build_donor_plan(original,candidate,additions)
    assert len(donors)==512 and all(all(0<=value<8199 for value in row["original_leaf_donor_ids"]) for row in donors)
    assert helper.suppression_ids([8192,*range(8199,8204)])==[8192,*range(8711,8716)]
    assert helper.suppression_ids(None) is None
    reject(lambda:helper.suppression_ids([True]))
    bounds=helper.native_joint_bound([{"source_id":"fixture_TR","language":"tr","encoder_frames":13,"target_tokens":6}],
        [{"source_id":"fixture_replay","language":"en","encoder_frames":501,"target_tokens":99}],1_800_000_000)
    assert bounds["maximum"]["bytes"]==501*100*8716*4 and bounds["joint_width"]==8716
    reject(lambda:helper.native_joint_bound([], [{"source_id":"too_long","language":"en","encoder_frames":1000,"target_tokens":180}],1_800_000_000))
    old=functions(OUT/"train_lora_batched_data_v11.py")
    new=functions(OUT/"train_lora_batched_bpe512_v1.py")
    preserved=("BucketCycler","BalancedBucketSampler","lr_factor","development_gate","validate_trainable_checkpoint","annotate_cache_lengths","save_training_state")
    assert all(ast.dump(old[name],include_attributes=False)==ast.dump(new[name],include_attributes=False) for name in preserved)
    globals_={"argparse":argparse,"Path":Path,"ROOT":ROOT,"DATA":ROOT/"work/data","OUTPUT":OUT,"math":math,
        "__doc__":"fixture","closed_source_identity":lambda args:{},"native_loss_identity":lambda args:{}}
    exec(compile(ast.Module(body=[new["parser"],new["validate_arguments"]],type_ignores=[]),"actual_parser_fixture","exec"),globals_)
    argv=["--run-id","fixture_BPE512","--resume-weights","unused_converted.pt",
          "--bpe-source-checkpoint","unused_closed.pt","--bpe-source-selection","unused_selection.json",
          "--bpe-source-closure","unused_closure.json","--bpe-conversion-report","unused_conversion.json"]
    args=globals_["parser"]().parse_args(argv)
    globals_["validate_arguments"](args)
    assert args.native_max_joint_bytes==1_800_000_000 and args.new_token_lr_multiplier==6 and args.BPE_delta_multiplier==1
    for key,value in (("resume_exact",Path("bad.pt")),("new_token_lr_multiplier",15.),("BPE_delta_multiplier",6.),("train_limit",10),
                      ("tdt_implementation","v2")):
        bad=copy.copy(args);setattr(bad,key,value);reject(lambda:globals_["validate_arguments"](bad))
    run=new["run"]
    calls=[(node.lineno,ast.unparse(node.func)) for node in ast.walk(run) if isinstance(node,ast.Call)]
    phase={name:min(line for line,call in calls if call==name) for name in ("install_candidate_student","validate_converted_weights","compare_development_predictions","torch.optim.AdamW")}
    assert phase["install_candidate_student"]<phase["validate_converted_weights"]<phase["compare_development_predictions"]<phase["torch.optim.AdamW"]
    assert any(call=="collate_training_batch" for _,call in calls)
    row_slices=[ast.unparse(node.slice) for node in ast.walk(run) if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name) and node.value.id=="parameter"]
    assert row_slices==["8193:8199","8193:8199"]
    old_collator=functions(OUT/"batched_lora_tr_metadata_v1.py")["collate_training_batch"]
    new_collator=functions(OUT/"batched_lora_bpe512_v1.py")["collate_training_batch"]
    def tail(node):
        start=next(index for index,statement in enumerate(node.body) if isinstance(statement,ast.Assign)
                   and any(isinstance(target,ast.Name) and target.id=="target_lengths" for target in statement.targets))
        return [ast.dump(statement,include_attributes=False) for statement in node.body[start:]]
    assert tail(old_collator)==tail(new_collator)
    sync=[line for line,call in calls if call=="torch.cuda.synchronize"]
    assert len(sync)==1
    assert not any(name in sys.modules for name in ("torch","nemo","transformers","tokenizers"))
    report={"kind":"HF_BPE512_isolated_training_source_API_verification_v1","passed":True,
        "source_sha256":{name:helper.sha_file(OUT/name) for name in ("train_lora_batched_bpe512_v1.py","hf_bpe512_training_v1.py","batched_lora_bpe512_v1.py","hf_bpe512_tr_metadata_v1.py")},
        "actual_candidate_donor_rows_checked":512,"actual_8199_prefix_semantic_verified":True,
        "actual_parser_required_BPE_options_parsed":True,"fresh_optimizer_and_distinct_row_policy_rejections":5,
        "preserved_scientific_ASTs":list(preserved),"collator_padding_mask_H2D_return_AST_unchanged":True,
        "actual_pre_Adam_source_candidate_GPU_call_order_verified":True,"GPU_check_executed":False,
        "synchronize_run_source_line":sync[0],"native_widened_width_bound_math_and_overflow_guard_checked":True,
        "row_delta_slices_exactly_8193_8199":True,"ML_imported":False,"whole_pool_scanned":False,
        "active_training_or_tokenizer_modified":False,"ready_to_activate":False,
        "remaining":"Strict wrapper routing, CLOSED qualifying source conversion, actual unsuppressed six-language GPU preflight and whole-update resource/quality remain pending"}
    path=Path(sys.argv[1]) if len(sys.argv)>1 else OUT/"HF_BPE512_isolated_training_source_API_verification_v1.json"
    with path.open("x",encoding="utf-8") as stream:json.dump(report,stream,indent=2,ensure_ascii=False)
    print(json.dumps({"passed":True,"source_sha256":report["source_sha256"],"sync_line":sync[0],"ML_imported":False,"report":str(path)}))


if __name__=="__main__":main()
