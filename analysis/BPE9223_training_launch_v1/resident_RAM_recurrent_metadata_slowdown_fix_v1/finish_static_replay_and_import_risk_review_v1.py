import ast,hashlib,json,os
from pathlib import Path
F=Path(__file__).parent;P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish'
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def imports(p):
    tree=ast.parse(Path(p).read_text(encoding='utf-8'))
    return [{'line':n.lineno,'module':getattr(n,'module',None),'names':[a.name for a in n.names]}
        for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
old=imports(S/'asr18_BPE9223_grouped_native_v1.py');ram=imports(S/'asr18_BPE9223_grouped_RAM_native_v1.py')
def signature(row):return(row['module'],tuple(row['names']))
oldkeys={signature(r)for r in old}
out={'kind':'ASR18_CLOSED_readonly_replay_cache_and_prefreeze_import_risk_review_v1',
 'no_import_or_runtime_heap_probe_performed':True,'selected_production_remedy':None,
 'input_sources':[pin(S/'asr18_BPE9223_source_context_v2.py'),pin(S/'bpe_replay_cache_labels_v1.py'),
 pin(S/'asr18_BPE9223_full_run_native_v12.py'),pin(S/'asr18_BPE9223_grouped_native_v1.py'),
 pin(S/'asr18_BPE9223_grouped_RAM_native_v1.py'),pin(S/'asr18_BPE9223_private_scalar_prefetch_v1.py'),
 pin(S/'bpe_production_window_coordinator_v4.py'),pin(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v30.py')],
 'replay_proof_static_shape':{'per_source':['outer proof dict','cache binding scalar dict','cache_signature scalar dict','source pin scalar dict','two-integer teacher_feature_shape list'],
 'proof_does_not_retain':['target ID list','teacher features/tensors','waveform','storage payload bytes'],
 'unique_source_cap':1914,'upper_container_nodes_before_automatic_untracking':1+1914*5,
 'possible_container_nodes_after_scalar_dicts_untrack':1+1914*2,
 'limits':'Static object-shape bound, NOT measured current heap/GC scan duration. Automatic untracking/aliasing can reduce counts. Two replay rows per update gives958 only as earliest all-unique pool coverage, not actual saturation or causal fit.',
 'consumers':'Current source_context_v2 owns one dictionary. resolve_rows221 equality-compares a newly recomputed proof, then222 assigns it and returns CURRENT original proof on rows. Planning deepcopies rows into bounded plans; scalar metadata worker clones only broad reader/cursor, not a full replay context. Current full_run_native_v12 uses context resolve/plan but does not export replay_evidence itself.',
 'decision':'Do NOT select replay codec as next main fix on this few-thousand-container hypothesis alone.'},
 'prefreeze_static_import_closure':{'driver30_full_run_owner_import_line':888,'original_freeze_entry_line':938,
 'late_CPU_RAM_import_lines':[939,941],'already_imported_owner_dependency':'asr18_BPE9223_full_run_native_v12 imports asr18_BPE9223_grouped_native_v1 at module top, before owner construction/freeze.',
 'old_grouped_native_top_imports':old,'RAM_native_top_imports':ram,
 'RAM_only_direct_import_declarations':[r for r in ram if signature(r)not in oldkeys],
 'finding':'The large NumPy/collator/narrow-objective/Numba-TDT/joint/allocation imports already occur via the old grouped-native owner dependency before freeze. Late RAM-native declaration is NOT proof of a large newly imported runtime graph.',
 'DEV':'benchmark/trainer/AutoProcessor lazy import declarations remain static candidates only; transitive first-load history is not recorded. AutoProcessor construction/from_pretrained/model warmup/evaluate must never be duplicated as preloading.',
 'decision':'Do NOT couple import preload or a new freeze into the next signature candidate. Original GC policy and module order remain exact.'},
 'next_concrete_preparation':'Focused fresh single-open signature equivalence/negative/concurrent timing proof, plus original builtin gc.collect scalar observational passthrough. No signature CPU execution or production transition until exact ROOT scope; current source30/session72166 unchanged.'}
p=F/'static_replay_cache_and_import_closure_risk_review_v1.json'
with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({'review':pin(p),'RAM_only_direct_import_declarations':out['prefreeze_static_import_closure']['RAM_only_direct_import_declarations'],'no_actual_import_or_probe':True}))
