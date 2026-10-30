"""Existing evidence and inactive design only; no profiler/import/remedy execution."""
import hashlib,json
from collections import defaultdict
from pathlib import Path
F=Path(__file__).parent;O=F/'py_spy_existing_native40012_actual_01'
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish'
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
profile=O/'profile.speedscope.json';external=read(O/'external_result.json')
assert pin(profile)==external['outputs']['profile.speedscope.json']
value=read(profile);frames=value['shared']['frames'];main=[p for p in value['profiles']if '"MainThread"'in p['name']]
assert len(main)==1;main=main[0]
groups=defaultdict(lambda:[0,0.]);examples={};Pnorm=str(P).replace('\\','/').lower()+'/'
for stack,w in zip(main['samples'],main['weights']):
    if not stack:continue
    leaf=frames[stack[-1]];name=leaf.get('name');file=str(leaf.get('file','')).replace('\\','/').lower()
    if not ((name=='realpath'and file=='<frozen ntpath>')or(name=='stat'and file.endswith('/pathlib.py'))):continue
    project=[frames[i]for i in stack[:-1]if str(frames[i].get('file','')).replace('\\','/').lower().startswith(Pnorm)]
    caller=project[-1]if project else {'file':None,'name':None,'line':None}
    key=(name,leaf.get('line'),caller.get('file'),caller.get('name'),caller.get('line'))
    groups[key][0]+=1;groups[key][1]+=w
    examples.setdefault(key,[{'file':f.get('file'),'function':f.get('name'),'line':f.get('line')}for f in project[-10:]])
rows=[{'leaf_function':k[0],'leaf_line':k[1],'nearest_project_file':k[2],'nearest_project_function':k[3],
 'nearest_project_line':k[4],'samples':v[0],'weight_seconds':v[1],'ordered_project_ancestors':examples[k]}
 for k,v in sorted(groups.items(),key=lambda kv:kv[1][1],reverse=True)]
path_report={'kind':'ASR18_same_single_profile_main_path_callers_v1','profile':pin(profile),'main_thread':main['name'],
 'main_weight_seconds':sum(main['weights']),'all_targeted_leaf_project_caller_groups':rows,
 'method':'Same single recorded profile; all main-thread realpath/pathlib.stat leaves and nearest project caller, with original ordered project ancestors; no extra sampling.',
 'limitations':'Statistical wall samples only;144 recorded non-atomic read errors and missing frames remain. No physical disk/CPU/GIL attribution; groups do not justify removing stop-file or mutation checks.'}
write(F/'same_single_profile_main_path_callers_v1.json',path_report)
oldreview=read(F/'ROOT_original280_scalar_metadata_stage_timing_actual_CPU_review_v1.json')
scalar=read(F/'scalar_metadata_stage_timing_actual_01/report.json')
assert pin(F/'scalar_metadata_stage_timing_actual_01/report.json')==oldreview['report']
metrics={k:scalar[k]for k in ('request_wall_ns_INCLUDES_oracle_checks','request_thread_cpu_ns_INCLUDES_oracle_checks',
 'flat_leaf_stage_clocks','owned_process_logical_IO_before','owned_process_logical_IO_after','owned_process_logical_IO_delta',
 'row_total_minus_leaf_wall_ns','row_total_minus_leaf_thread_cpu_ns')}
proposal={'kind':'ASR18_INACTIVE_fresh_single_open_signature_candidate_design_v1','production_policy_selected_or_implemented':False,
 'existing_training_must_continue_unchanged':True,'input_pins':{
 'human_fix_authority':pin(F/'ROOT_direct_human_then_fix_authorization_v1.json'),
 'profile_measurement_ROOT':pin(F/'ROOT_existing_source30_single_nonblocking_profile_actual_review_v1.json'),
 'same_profile_path_callers':pin(F/'same_single_profile_main_path_callers_v1.json'),
 'original280_ROOT':pin(F/'ROOT_original280_scalar_metadata_stage_timing_actual_CPU_review_v1.json'),
 'original280_report':pin(F/'scalar_metadata_stage_timing_actual_01/report.json'),
 'original_signature_and_geometry':pin(S/'bpe_declared_geometry_index_v1.py'),
 'original_rows':pin(S/'asr18_BPE9223_source_context_v2.py'),
 'original_locks':pin(S/'bpe_windows_final_metadata_lock_v1.py'),
 'original_prefetch':pin(S/'asr18_BPE9223_private_scalar_prefetch_v1.py'),
 'driver30':pin(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v30.py'),
 'native1':pin(S/'asr18_BPE9223_grouped_RAM_native_v1.py'),
 'DEV2':pin(S/'asr18_resident_serial350_DEV_v2.py'),
 'original_GC_scope':pin(P/'work/research/run_ASR18_native_throughput_component_v15.py')},
 'closed_original280_exact_result_fields':metrics,
 'evidence_limits':['Standalone280 wall1.2559688s/threadCPU.125s does NOT reproduce live14s. ThreadCPU is quantized.',
 '560 catalog+labels and280 geometry signatures total.0859083s wall in standalone; geometry SQLexecute1.0014803s dominates that particular run.',
 'Live main path resolution is a measured serialized cost, but scope differs from standalone. Owned IO deltas are logical not physical/cache measurements.',
 'Main native214 and233 direct leaves strongly localize cleanup C-call chains but do not isolate pureGC. No fresh profile or old archive read is proposed.'],
 'recommendation':'Prepare a narrow single-open signature implementation and focused equivalence/negative CPU fixture FIRST, leaving native GC and module order unchanged in that candidate. Do not use unchecked canonical-path caching. Performance benefit remains actual-production-tested and cannot promise~80 alone.',
 'candidate_algorithm_not_yet_source':{
 'supported_inputs':'ONLY the exact three already-validated locked E absolute normal metadata DB paths; initialization obtains ORIGINAL signature and canonical path first, caches only argument/expected formatting, never caches current size/mtime or check result.',
 'every_original_before_draw_call':'Open current pathname for read-attributes with the original denywrite locks retained. Query normalized DOS final pathname and file information on that SAME fresh handle. Return EXACT original three fields path/bytes/mtime_ns; close owned handle synchronously. Keep all existing caller guards, counts, require messages and poisoning paths.',
 'canonicalization':'Do not blindly remove extended prefix. Fast path only if fresh normalized final DOS name exactly matches the proved canonical normal E path/format. Demonstrate CPython3.11.13 prefix-strip/case behavior for this narrow input; for unmatched formatting, UNC/device namespaces, unsupported/reparse/case/rename cases or API uncertainty, call ORIGINAL signature(path) with its full resolve+stat behavior.',
 'mtime':'Exact integer FILETIME100ns arithmetic to original st_mtime_ns; no float conversion.',
 'path_mutation':'Fresh pathname handle/final canonical identity every call, not merely a permanent old handle or cached string. Parent rename/reparse/retarget/case transitions must be independently covered; original fallback if equivalence not proven. Existing stop-file exists checks are untouched.',
 'exceptions':'Unsupported/Win32 error paths delegate original signature so its actual exceptions/return and original caller poison behavior remain; no swallowed original errors, tolerated mutation, relaxed thresholds or cached successful check.',
 'no_credit':'No memory saving/byte credit. CPU2.5GB/MAX2/LIVE1/225cushion/fullpacket-queue-prefetch/originalfloors/DEV/save200/science remain exact.'},
 'focused_CPU_proof_scope_proposed_not_admitted':{
 'runtime':'Dedicated metadata CPU interpreter/CUDA-1/caps1/BelowNormal/E storage; no model/NumPy/Torch/audio/GPU/native/authoritative sampler operation.',
 'bounds':{'internal_wall_seconds':180,'external_wall_seconds':240,'family_private_cap_bytes':536870912,'full_prospective_family_reserve_bytes':536870912,'full_prefetch_reservation_bytes':268435456,'original_commit_floor_bytes':2000000000,'original_physical_floor_bytes':1500000000,'E_free_floor_bytes':100000000000},
 'genuine_cases':['Synthetic E file signatures against ORIGINAL signature, Unicode/spaces/long normal paths, exact integer timestamp and size, alreadycanonical paths and aliases.',
 'Owned tiny SQLite metadata plus original share-read locks: attempted write/replace/delete rejected, readonly3DB/4MiBcache/32rowLRU lifecycle unchanged.',
 'Real parent rename and any actually-supported reparse/case/retarget transitions; record unsupported filesystem capabilities honestly and force ORIGINAL fallback, never promote a method stub as an OS proof.',
 'Existing frozen280 already-planned rows: one candidate query pass/oracle row digest/order/alllabels equality; counts560+280 fresh signatures preserved, no authoritative draw or second corpus scan.',
 'Changed/missing path/size/mtime and wrong source bindings, exception type/message/poisoning, API failure/fallback/handle closure/order; no per-row fullresource sampling manufactured into timing.'],
 'method_stubs':'Only otherwise-unavailable Win32 error/fallback and callback exception identity controls explicitly labelled; no production reparse or pressure proof from them.',
 'before_execution':'Freeze exact helper, integration/source inverse, fixture/launcher/input/source pins and ROOT scope; no control now.'},
 'comparison_GC_freeze_refresh':{
 'not_selected':True,'possible_benefit':'Main direct cleanup sites are larger than path cost, so GC-related graph scanning remains a strong hypothesis.',
 'risk':'Additional freeze can permanently retain currently reachable temporary cycles after their external refs disappear; scalar/receiver/plan/DEV graphs may be live. Full equality/RNG does not by itself prove object lifetime or future memory safety.',
 'required_extra_proof':'True no-live-graph/packet/temporary-cycle quiescence and owned worker lifecycle, real cycle-retention negatives, exact all207/3Adam/4RNG/cursors/journal/pointer fingerprints, original collect/sync/emptycache and teardown unfreeze retained. Much wider than current representation/signature change.',
 'decision':'No GC refresh/disable/threshold or per-update collect changes.'},
 'separate_existing_import_order_option_NO_implementation':{
 'source_fact':'driver30 line938 enters original once-only resident_gc_scope; CPU11/native1/lifecycle/coordinator/normalizer imports follow at939..944. DEV2 evaluate lazily imports benchmark/trainer and compose imports AutoProcessor afterfreeze.',
 'candidate':'Move ONLY already-used module import statements before the ORIGINAL once-only freeze; no extra freeze/unfreeze/collect, model call, warmup or DEV. Keep original runtime calls and nativebody exact.',
 'benefit_hypothesis':'Persistent imported module/function/cache graphs could then enter the already-intended frozen generation; current profile cannot quantify how much cleanup is that graph. Static late import statements do NOT prove these modules were first imported late rather than already present transitively in sys.modules.',
 'risks':['Import top-level code/decorators may initialize objects, alter RNG/global state, load extensions or perform eager CUDA work; source closure and actual CPU import-only proof needed first.',
 'AutoProcessor.from_pretrained and model/processor construction must NOT be moved or duplicated; only symbol/module imports eligible.',
 'Native runtime lazy compilation/per-row caches created afterfreeze would remain; preloading cannot be assumed to stop recurrent drift.'],
 'required_nonexecuting_review':'Pin exact native/DEV import closure and original import placement; identify top-level calls before any import-only job. Compare proposed import-order-only source AST/native loops and actual RNG/guard environment proof before selection.',
 'selected_or_imported_now':False},
 'future_production_gate':'Finite equivalence proof plus ROOT source/config review; supported intentional natural200 checkpoint stop and actual clean-Monitor seal, real typed entire pre/postrestore selectors+wrapper forwarding and final execution binding. Genuine allstate restore-before-native/prefixcopy-afterrestore/redo. Start=restored+20; end=max(start+1600,nextscheduledDEV+1020), frozen before launch. All contiguous original20rows/allnatural200saves/realDEV included; first400/final400 and two full400 postDEV blocks, no extraDEV/benchmark/finite productionstop or brief warm success.'}
write(F/'INACTIVE_single_open_signature_vs_import_prefreeze_candidate_design_v1.json',proposal)
print(json.dumps({'path_callers':pin(F/'same_single_profile_main_path_callers_v1.json'),'design':pin(F/'INACTIVE_single_open_signature_vs_import_prefreeze_candidate_design_v1.json'),
 'top_path_callers':rows[:6],'executed_sampler_import_control_or_remedy':False}))
