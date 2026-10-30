"""CPU routing, full frontend, native long-form control-flow and corruption guards."""
from __future__ import annotations
import copy
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, GenerationConfig, WhisperConfig, WhisperForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0, str(OUTPUT))
import benchmark_strong_tr_test_v2 as worker

checks = []
def check(name, condition):
    if not bool(condition):
        raise AssertionError(name)
    checks.append({"name": name, "passed": True})

def rejects(name, function):
    try:
        function()
    except (RuntimeError, ValueError, AssertionError):
        checks.append({"name": name, "passed": True})
    else:
        raise AssertionError("guard accepted: " + name)

protected = worker.protected_code()
examples, input_audit = worker.verified_test_rows(ROOT / "work/data/test_tr.jsonl")
check("same100fixedsources", len(examples) == 100)
check("duration_only_routing_99_short_1_long", input_audit["short_form_count"] == 99 and input_audit["long_form_count"] == 1)
check("long_actual_duration_36_84", input_audit["duration_routes"]["long_form"][0]["duration"] == 36.84)
check("long_actual_source_1937", ":1937:" in input_audit["duration_routes"]["long_form"][0]["source_id"])
check("short_protocol_unchanged", worker.SHORT_PROTOCOL == __import__("benchmark_strong_tr_test").PROTOCOL)
check("long_only_supported_generation_flags_changed", worker.LONG_PROTOCOL == {**worker.SHORT_PROTOCOL,"return_timestamps":True,"return_segments":True})
check("exact_boundary_short", worker.mode_for_samples(480000) == "short_form")
check("over_boundary_long", worker.mode_for_samples(480001) == "long_form")
rejects("zero_audio_rejected", lambda: worker.mode_for_samples(0))
processor = AutoProcessor.from_pretrained(str(ROOT / "work/models/whisper-large-v3"), local_files_only=True)
long_record = next(r for r in examples if r["duration"] > 30)
signal, sr = sf.read(long_record["audio_filepath"], dtype="float32")
long_features, metadata = worker.prepare_features(processor, signal, sr)
check("actual_full589440PCM_samples", metadata["input_samples"] == 589440)
check("actual_full3684_mel_frames", metadata["input_mel_frames"] == 3684)
check("actual_full3684_mask_frames", metadata["input_attention_mask_frames"] == 3684)
check("long_frontend_no_truncation_exact", torch.equal(long_features["input_features"], processor(signal,sampling_rate=16000,return_tensors="pt",truncation=False,padding="longest",return_attention_mask=True)["input_features"]))
short_signal, sr = sf.read(examples[0]["audio_filepath"],dtype="float32")
short_features, short_meta = worker.prepare_features(processor,short_signal,sr)
original_short = processor(short_signal,sampling_rate=sr,return_tensors="pt",return_attention_mask=True)
check("short_frontend_bitwise_unchanged", all(torch.equal(short_features[k],original_short[k]) for k in original_short))
check("short_padded3000_frames", short_meta["input_mel_frames"] == 3000)
altered = signal.copy(); altered[480000:] = 0
altered_features, _ = worker.prepare_features(processor,altered,16000)
check("real_last6_84s_changes_tail_features", not torch.equal(long_features["input_features"][...,3000:],altered_features["input_features"][...,3000:]))

# Tiny random CPU fixture only. Native pinned generate outer loop is real; its
# expensive decoder call is replaced by deterministic timestamp/token output.
# No pretrained model is loaded and no GPU or ASR prediction is made here.
config = WhisperConfig(vocab_size=51866,num_mel_bins=128,d_model=8,encoder_layers=1,decoder_layers=1,
    encoder_attention_heads=1,decoder_attention_heads=1,encoder_ffn_dim=16,decoder_ffn_dim=16,
    max_source_positions=1500,max_target_positions=448,decoder_start_token_id=50258,eos_token_id=50257,
    bos_token_id=50257,pad_token_id=50257)
tiny = WhisperForConditionalGeneration(config).eval()
tiny.generation_config = GenerationConfig.from_pretrained(str(ROOT / "work/models/whisper-large-v3"),local_files_only=True)
calls = []
def fake_fallback(**kwargs):
    check("fixture_greedy_no_previous_conditioning_"+str(len(calls)), kwargs["do_condition_on_prev_tokens"] == [False])
    n = trace.windows[-1]["input_frame_count"]
    content = torch.tensor([worker.TIMESTAMP_BEGIN,481,worker.TIMESTAMP_BEGIN+n//2],dtype=torch.long)
    prefix = kwargs["decoder_input_ids"][0].tolist()
    raw = torch.tensor(prefix+content.tolist()+[worker.EXPECTED_EOS],dtype=torch.long)
    calls.append({"seek":int(kwargs["seek"][0]),"frames":n,"prefix":prefix})
    return [content], [{"sequences":raw}], [False], [False], None
tiny.generate_with_fallback = fake_fallback
with torch.inference_mode(), worker.LongformTrace(tiny) as trace:
    generated = tiny.generate(**long_features, **worker.LONG_PROTOCOL)
tokens, evidence = worker.unpack_longform(generated,trace,metadata["input_mel_frames"])
check("native_outer_loop_two_windows_3000_and684", [(w["seek_start_frame"],w["input_frame_count"]) for w in evidence["windows"]] == [(0,3000),(3000,684)])
check("native_outer_loop_fullEOF_3684", evidence["final_seek_frame"] == 3684 and evidence["full_input_consumed"])
check("all_actual_raw_prefixes_terminalEOS", all(w["raw_generated_token_ids"][:3] == worker.LONG_PREFIX and w["raw_generated_token_ids"][-1] == worker.EXPECTED_EOS for w in evidence["windows"]))
check("trace_overrides_restored", "_get_input_segment" not in tiny.__dict__ and "_retrieve_segment" not in tiny.__dict__)
check("aggregate_actual_segment_tokens", tokens == [t for s in evidence["segments"] for t in s["tokens"]])
check("aggregate_does_not_fabricateEOS", worker.EXPECTED_EOS not in tokens)
check("timestamp_decode_removes_special_timestamp_text", "<|" not in processor.batch_decode([tokens],skip_special_tokens=True)[0])
for name, mutate in [
    ("truncated_mel", lambda e:e["windows"][0].update(total_input_frames=3000)),
    ("missing_tail_window", lambda e:e["windows"].pop()),
    ("seek_gap", lambda e:e["windows"][1].update(seek_start_frame=3001)),
    ("zero_progress", lambda e:e["windows"][0].update(seek_advance_frames=0)),
    ("unfinished_eof", lambda e:e.update(final_seek_frame=3000)),
    ("missing_terminalEOS", lambda e:e["windows"][0]["raw_generated_token_ids"].pop()),
    ("wrong_language_prefix", lambda e:e["windows"][0]["raw_generated_token_ids"].__setitem__(1,50259)),
    ("prev_context_enabled", lambda e:e["windows"][1]["decoder_input_token_ids"].insert(0,50362)),
    ("segment_content_changed", lambda e:e["segments"][0]["tokens"].__setitem__(1,482)),
    ("segment_raw_changed", lambda e:e["segments"][0]["raw_generated_token_ids"].__setitem__(3,50366)),
    ("segment_index_changed", lambda e:e["segments"][0]["idxs"].__setitem__(0,4)),
    ("nonfinite_timestamp", lambda e:e["segments"][0].update(end=float("nan"))),
    ("negative_timestamp", lambda e:e["segments"][0].update(start=-1)),
    ("tail_false", lambda e:e.update(full_input_consumed=False)),
]:
    bad = copy.deepcopy(evidence); mutate(bad)
    rejects(name,lambda bad=bad: worker.validate_longform(bad,tokens,3684))
rejects("fabricated_aggregateEOS",lambda: worker.validate_longform(evidence,tokens+[worker.EXPECTED_EOS],3684))
rejects("aggregate_tail_dropped",lambda: worker.validate_longform(evidence,tokens[:3],3684))
check("protected_original_files_unchanged", worker.protected_code() == protected)
report = {"schema":1,"passed":True,"gpu_used":False,"pretrained_model_loaded":False,
    "actual_ASR_inference_performed":False,"checks":checks,"checks_passed":len(checks),
    "script_sha256":worker.sha256(OUTPUT/"benchmark_strong_tr_test_v2.py"),
    "guard_script_sha256":worker.sha256(Path(__file__)),"protected_code":protected,
    "input_audit":input_audit,"actual_long_frontend":metadata,
    "native_control_flow_fixture":{"calls":calls,"longform":evidence,"aggregate_tokens":tokens},
    "scope":"Actual fixed TEST source/WAV/frontend only; native tiny CPU outer-loop fixture uses deterministic decoder outputs. No TEST ASR hypotheses or GPU/model inference."}
worker.write_json(OUTPUT/"strong_test_v2_cpu_verification.json",report)
print(json.dumps({"passed":True,"checks_passed":len(checks),"path":str(OUTPUT/"strong_test_v2_cpu_verification.json")}),flush=True)
