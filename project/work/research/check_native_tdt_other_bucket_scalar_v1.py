"""Tiny scalar/stdlib OTHER algebra tests; no actual native/Torch execution."""
from __future__ import annotations
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0, str(OUT))
import native_tdt_other_bucket_v1 as prototype


def logadd(a, b):
    if a == float("-inf"): return b
    if b == float("-inf"): return a
    maximum = max(a, b)
    return maximum + math.log(math.exp(a-maximum)+math.exp(b-maximum))


def scalar_tdt_nll(tokens, duration_logits, labels, blank, durations):
    """Scalar DAG from pinned native arc/terminal rules, sigma0/FastEmit0.

    Token duration0 is allowed; blank duration0 is excluded. Only a positive
    terminal blank consumes exactly T. Labels never jump to/past final T.
    """
    frames, targets = len(tokens), len(labels)
    token_lp = [[tuple(value-prototype.scalar_logsumexp(cell) for value in cell) for cell in row] for row in tokens]
    duration_lp = [[tuple(value-prototype.scalar_logsumexp(cell) for value in cell) for cell in row] for row in duration_logits]
    alpha = [[float("-inf")] * (targets+1) for _ in range(frames)]
    alpha[0][0] = 0.; terminal = float("-inf")
    for t in range(frames):
        for u in range(targets+1):
            for index, duration in enumerate(durations):
                dest = t + duration
                base = alpha[t][u] + duration_lp[t][u][index]
                if duration > 0 and dest < frames:
                    alpha[dest][u] = logadd(alpha[dest][u], base + token_lp[t][u][blank])
                if duration > 0 and dest == frames and u == targets:
                    terminal = logadd(terminal, base + token_lp[t][u][blank])
                if u < targets and dest < frames:
                    alpha[dest][u+1] = logadd(alpha[dest][u+1], base + token_lp[t][u][labels[u]])
    return -terminal


def run():
    tests=0; rejected=0; cases=[]; randomizer=random.Random(718)
    def check(value):
        nonlocal tests
        assert value
        tests+=1
    def refusal(call):
        nonlocal tests,rejected
        try:call()
        except ValueError:tests+=1;rejected+=1
        else:raise AssertionError("Invalid scalar metadata accepted")
    casespec=[(7,3,(6,6),3),(7,4,(1,2,1),3),(3,1,(0,2,0),2)]
    for vocab,blank,labels,frames in casespec:
        durations=(0,1,2,3,4)
        plan=prototype.retained_plan(labels,vocab,blank,durations,vocab_chunk_size=2)
        tokens=[[[randomizer.uniform(-3,3) for _ in range(vocab)] for _ in range(len(labels)+1)] for _ in range(frames)]
        duration_logits=[[[randomizer.uniform(-2,2) for _ in durations] for _ in range(len(labels)+1)] for _ in range(frames)]
        compressed=[[prototype.scalar_coalesce(cell,plan) for cell in row] for row in tokens]
        full=scalar_tdt_nll(tokens,duration_logits,labels,blank,durations)
        compact=scalar_tdt_nll(compressed,duration_logits,plan['remapped_target_ids'],plan['native_blank_id'],durations)
        check(abs(full-compact)<1e-12)
        check(len(plan['remapped_target_ids'])==len(labels))
        normalizer_error=max(abs(prototype.scalar_logsumexp(cell)-prototype.scalar_logsumexp(compressed[t][u]))
            for t,row in enumerate(tokens) for u,cell in enumerate(row))
        check(normalizer_error<1e-12)
        epsilon=1e-5;gradient_error=0.;duration_error=0.
        retained={value:index for index,value in enumerate(plan['retained_ids'])}
        for t,row in enumerate(tokens):
            for u,cell in enumerate(row):
                compact_gradient=[]
                for k in range(plan['native_vocab_size']):
                    plus=copy.deepcopy(compressed);minus=copy.deepcopy(compressed)
                    plus[t][u][k]+=epsilon;minus[t][u][k]-=epsilon
                    compact_gradient.append((scalar_tdt_nll(plus,duration_logits,plan['remapped_target_ids'],plan['native_blank_id'],durations)
                        -scalar_tdt_nll(minus,duration_logits,plan['remapped_target_ids'],plan['native_blank_id'],durations))/(2*epsilon))
                for k in range(vocab):
                    plus=copy.deepcopy(tokens);minus=copy.deepcopy(tokens)
                    plus[t][u][k]+=epsilon;minus[t][u][k]-=epsilon
                    actual=(scalar_tdt_nll(plus,duration_logits,labels,blank,durations)
                            -scalar_tdt_nll(minus,duration_logits,labels,blank,durations))/(2*epsilon)
                    expected=(compact_gradient[retained[k]] if k in retained else
                        compact_gradient[plan['other_index']]*math.exp(cell[k]-compressed[t][u][plan['other_index']]))
                    gradient_error=max(gradient_error,abs(actual-expected))
                for k in range(len(durations)):
                    plus=copy.deepcopy(duration_logits);minus=copy.deepcopy(duration_logits)
                    plus[t][u][k]+=epsilon;minus[t][u][k]-=epsilon
                    a=(scalar_tdt_nll(tokens,plus,labels,blank,durations)-scalar_tdt_nll(tokens,minus,labels,blank,durations))/(2*epsilon)
                    b=(scalar_tdt_nll(compressed,plus,plan['remapped_target_ids'],plan['native_blank_id'],durations)
                       -scalar_tdt_nll(compressed,minus,plan['remapped_target_ids'],plan['native_blank_id'],durations))/(2*epsilon)
                    duration_error=max(duration_error,abs(a-b))
        check(gradient_error<2e-8 and duration_error<2e-8)
        cases.append({'vocab':vocab,'blank':blank,'ordered_targets':list(labels),'frames':frames,
            'compressed_vocab':plan['native_vocab_size'],'OTHER_classes':plan['other_class_count'],
            'raw_NLL':full,'raw_NLL_absolute_difference':abs(full-compact),
            'normalizer_max_difference':normalizer_error,'full_token_gradient_vs_OTHER_chain_max_difference':gradient_error,
            'duration_gradient_max_difference':duration_error,'gradient_method':'Float64 scalar central differences, epsilon1e-5; not autograd/native kernel.'})
    # Empty OTHER and fully-retained vocabulary chunks must omit their LSE.
    empty=prototype.retained_plan([0,2],3,1,vocab_chunk_size=2)
    check(empty['other_index'] is None and empty['native_vocab_size']==3)
    check(any(chunk['other_count']==0 for chunk in empty['vocabulary_chunks']))
    check(prototype.scalar_coalesce([1.,2.,3.],empty)==[1.,2.,3.])
    for labels,vocab,blank in (([1],3,1),([],3,1),([3],3,1),([True],3,1)):
        refusal(lambda:prototype.retained_plan(labels,vocab,blank))
    policy={'sigma':0.,'omega':0.,'fastemit_lambda':0.,'effective_native_clamp':0.,
        'reduction':'none','source_files_sha256':prototype.SOURCE_FILES}
    prototype._validate_native_policy(policy);check(True)
    for key,bad in (('sigma',.02),('omega',.1),('fastemit_lambda',.1),('effective_native_clamp',1.)):
        changed=copy.deepcopy(policy);changed[key]=bad
        refusal(lambda:prototype._validate_native_policy(changed))
    # Demonstrate why aggregate clamp cannot substitute for elementwise clamp.
    check(min(1.,2.)*.25 != min(1.,2.*.25))
    # New/unemitted class mass participates even if it dominates all arc classes.
    dominant=prototype.retained_plan([1,1],7,3,vocab_chunk_size=2)
    z=[30.,-2.,1.,-1.,3.,4.,5.]
    coalesced=prototype.scalar_coalesce(z,dominant)
    check(abs(prototype.scalar_logsumexp(z)-prototype.scalar_logsumexp(coalesced))<1e-12)
    # Hypothetical longest-target geometry, not a sampled/data/GPU fit result.
    hypothetical=prototype.retained_plan(list(range(180)),8711,8192,vocab_chunk_size=1024)
    arithmetic=prototype.projection_bounds(hypothetical,293,640,16)
    check(arithmetic['compressed_native_FP32_token_bytes']<arithmetic['full_reference_FP32_joint_bytes']/40)
    check('torch' not in sys.modules and 'nemo' not in sys.modules and 'transformers' not in sys.modules)
    source=OUT/'native_tdt_other_bucket_v1.py';tree=ast.parse(source.read_text(encoding='utf-8'))
    imports=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    check(not any(name and ('torch' in name or 'nemo' in name) for name in imports))
    official_root=ROOT/'work/nemo-native-env-v1/Lib/site-packages/nemo'
    records={}
    for relative,digest in prototype.SOURCE_FILES.items():
        path=official_root/relative
        check(hashlib.sha256(path.read_bytes()).hexdigest()==digest)
        records[relative]={'path':str(path),'sha256':digest}
    report={'schema':1,'kind':'inactive_TDT_OTHER_bucket_scalar_source_verification_v1','passed':True,
        'CPU_only':True,'assertions':tests,'rejections':rejected,'cases':cases,
        'prototype':{'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()},
        'fixture':{'path':str(Path(__file__).resolve()),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'official_sources':records,'source_references':{'normalizer':'gpu_rnnt.py93-123',
            'DP_and_terminal_arcs':'gpu_rnnt_kernel.py964-1069','token_gradient':'gpu_rnnt_kernel.py1353-1425',
            'disabled_effective_clamp':'rnnt_pytorch.py548','Python_uniform_draw':'gpu_rnnt.py606'},
        'math':'Retain unique target classes plus blank; OTHER=LSE(all remaining lexical classes). Same normalizer preserves arcs. g_OTHER*dOTHER/dz_j=occupancy*p_j. Ordered repeated targets/U and durations remain unchanged.',
        'hypothetical_geometry_only':{'T':293,'U':180,'unique_targets':180,'full_vocab':8711,'hidden':640,
            'time_chunk':16,'vocabulary_chunk':1024,'allocation_arithmetic':arithmetic},
        'limits':['Scalar central differences/DAG sanity are not actual official-kernel or Torch-autograd equivalence.',
            'Head GEMM/reduction layout changes: no FP32/BF16 bitwise guarantee; actual original-full/native and all source-parameter gradients must pass before adoption.',
            'Nested checkpoint backward and actual native callback integration are unexecuted; no memory/commit/throughput gain measured.',
            'Chunking computes every full-vocabulary head row and adds kernel launches/recomputation; throughput can worsen.',
            'Projection/compact arithmetic omits native workspaces, other graphs, model/Adam/data and WDDM/host commit.',
            'OTHER includes all remaining symbols including added unused tokens; positive native clamp is incompatible with this gradient chain.',
            'Only TDT is compressed. Existing teacher KD must keep full student vocabulary normalization.',
            'All existing activated source/runtime/tokenizer/data/optimizer files remain unchanged.'],
        'ML_or_Torch_imported':False,'actual_native_kernel_executed':False,'actual_model_gradient_comparison':False,
        'GPU_or_model_checkpoint_audio_reads':False,'activated_or_current_sources_edited':False}
    destination=OUT/'TDT_OTHER_bucket_scalar_source_verification_v1.json'
    with destination.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps({'report':str(destination),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
        'assertions':tests,'rejections':rejected,'cases':cases,'hypothetical_allocation':arithmetic}))


if __name__=='__main__':run()
