"""Inactive SOURCE/METADATA preparation only; never imports numerical/audio code."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1')
F=N/'resident_RAM_original_interval170469_failure_actual_v1'
PRIOR=N/'resident_RAM_original_interval_post142894_failure_actual_v1/observe_complete_member142895_EOF_v1.py'
BASE=F/'launch_recorded170470_ordered14_decoder_external_v1.py'
FIXTURE=F/'observe_recorded170470_row6_full_head_member_v1.py'
LAUNCHER=F/'launch_recorded170470_row6_full_head_member_external_v1.py'
SCOPE='ASR18_recorded170470_row6_exact_full_head_member_physical_CPU_v1'
def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    p=Path(path).resolve();data=p.read_bytes();require(len(data)<=8*1024**2,'Compact/source input only')
    return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def write(path,value):
    p=Path(path);require(not p.exists(),'New immutable source metadata')
    p.write_bytes((json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'))

old_source=PRIOR.read_text(encoding='utf-8');helper=old_source[old_source.index('packet_side_read_observations=[]'):old_source.index('try:\n    numpy_runtime(frontend=False)')]
draft=FIXTURE.read_text(encoding='utf-8');require(draft.count('__PINNED_OWNED_ACCESSOR_PLACEHOLDER__')==1,'Unexecuted source template placeholder only')
final=draft.replace('__PINNED_OWNED_ACCESSOR_PLACEHOLDER__',helper.rstrip())
old_tree=ast.parse(old_source);new_tree=ast.parse(final)
for name in('PacketSideDataLayout','MemoryRegion','packet_side_bytes'):
    old_node=next(v for v in old_tree.body if isinstance(v,(ast.ClassDef,ast.FunctionDef))and v.name==name)
    new_node=next(v for v in new_tree.body if isinstance(v,(ast.ClassDef,ast.FunctionDef))and v.name==name)
    require(ast.dump(old_node,include_attributes=False)==ast.dump(new_node,include_attributes=False),'Exact previously reviewed accessor/class AST')
provenance=None
for node in ast.walk(old_tree):
    if isinstance(node,ast.Assign)and any(isinstance(v,ast.Name)and v.id=='packet_api_provenance'for v in node.targets):
        provenance=ast.literal_eval(node.value)
require(provenance is not None,'Literal actual installed ABI provenance only')
FIXTURE.write_bytes(final.encode('utf-8'))
frozen_old=read(F/'ordered14_decoder_frozen_inputs_v1.json')
row_budget=next(v['prospective_bytes']for v in frozen_old['original_per_row_prospective_bytes']if v['original_update_row_1based']==6)
require(row_budget==901350272,'Original exact row6 full prospective envelope')
before=BASE.read_bytes();text=before.decode('utf-8');edits=[]
def replace(old,new,count=1):
    global text
    require(text.count(old)==count,'Exact external literal count: '+old[:80]);text=text.replace(old,new)
    edits.append({'old_literal':old,'new_literal':new,'expected_count':count})
replace('ONE ordered14 original decoder reproduction, no frontend/model/GPU/production; external original full guards.',
    'ONE exact row6 full-head-member physical observation; no frontend/model/GPU/production, original full guards.')
replace("FIXTURE=F/'observe_recorded170470_ordered14_decoder_v1.py'","FIXTURE=F/'observe_recorded170470_row6_full_head_member_v1.py'")
replace("RESULT=F/'ordered14_decoder_external_result_v1.json'","RESULT=F/'full_head_member_physical_external_result_v1.json'")
replace("OUTPUT=F/'ordered14_decoder_actual_01'","OUTPUT=F/'full_head_member_physical_actual_01'")
replace('PROSPECTIVE=903023104','PROSPECTIVE=901350272')
replace("'ASR18_source31_saved170400_ordered14_original_decoder_reproduction_CPU_v1'",repr(SCOPE))
replace('ordered14_decoder_frozen_inputs_v1.json','full_head_member_physical_frozen_inputs_v1.json',2)
replace("'ordered14_original_decoder_external_CPU'","'exact_head_member_physical_external_CPU'")
replace("stdout=F/'ordered14_decoder_stdout.log';stderr=F/'ordered14_decoder_stderr.log'",
    "stdout=F/'full_head_member_physical_stdout.log';stderr=F/'full_head_member_physical_stderr.log'")
replace("'ASR18_ordered14_original_decoder_external_result_v1'","'ASR18_exact_head_member_physical_external_result_v1'")
replace("'Preserve actual failed decoder observation; no automatic rerun'","'Preserve actual failed physical observation; no automatic rerun'")
ast.parse(text);require(not LAUNCHER.exists(),'New exact owned external source')
LAUNCHER.write_bytes(text.encode('utf-8'));inverse=text
for change in reversed(edits):
    require(inverse.count(change['new_literal'])==change['expected_count'],'Exact reverse replacement count')
    inverse=inverse.replace(change['new_literal'],change['old_literal'])
require(inverse.encode('utf-8')==before,'Full RAW external inverse')
ledger=F/'full_head_member_external_exact_raw_inverse_v1.json'
write(ledger,{'old':pin(BASE),'new':pin(LAUNCHER),'newline_normalization':'NONE, full source LF bytes',
    'replacements':edits,'full_old_raw_reconstructed':True})
extra=[pin(F/'ordered14_decoder_actual_01/first_actual_exception.json'),pin(F/'ordered14_decoder_actual_01/report.json'),
    pin(F/'ordered14_decoder_actual_completion_v1.json'),pin(F/'ordered14_decoder_external_result_v1.json'),
    pin(F/'ordered14_decoder_actual_TOOL_v1.json'),
    pin(F/'ROOT_recorded170470_ordered14_original_decoder_actual_CPU_localization_review_v1.json'),
    pin(F/'ordered14_decoder_frozen_inputs_v1.json'),pin(PRIOR),
    pin(N/'ROOT_post142894_exact_head_member_physical_CPU_actual_review_v1.json')]
extra.extend(provenance[k]for k in('pinned_pxd','pinned_Cython_python_source','pinned_extension'))
unique={}
for value in frozen_old['input_pins']+extra:
    actual=pin(value['path']);expected={k:value[k]for k in('path','bytes','sha256')}
    require(actual==expected,'Every original/new canonical input pin unchanged')
    unique[actual['path']]=actual
inputs=list(unique.values());frozen=F/'full_head_member_physical_frozen_inputs_v1.json'
write(frozen,{'kind':'ASR18_exact_recorded170470_row6_full_head_member_frozen_inputs_v1',
    'propagated_exception':pin(F/'ordered14_decoder_actual_01/first_actual_exception.json'),
    'configuration':frozen_old['configuration'],'phase':'weak_label_continued_training',
    'original_resource_policy':frozen_old['original_resource_policy'],
    'source_identity_inventory':frozen_old['source_identity_inventory'],'heldout_PCM':frozen_old['heldout_PCM'],
    'maximum_original_prospective_bytes':row_budget,'packet_side_data_API_provenance':provenance,'input_pins':inputs})
diff=F/'prior142895_to170470_row6_full_head_member_source.diff.txt'
diff.write_bytes(''.join(difflib.unified_diff(old_source.splitlines(True),final.splitlines(True),fromfile=str(PRIOR),tofile=str(FIXTURE))).encode('utf-8'))
handoff=F/'full_head_member_physical_inactive_source_handoff_v1.json'
write(handoff,{'kind':'ASR18_INACTIVE_one_exact_recorded170470_row6_full_head_member_source_handoff_v1',
    'execution_admitted':False,'required_ROOT_scope':SCOPE,'fixture_source':pin(FIXTURE),'launcher_source':pin(LAUNCHER),
    'frozen_inputs':pin(frozen),'input_pins':inputs,'external_RAW_inverse':pin(ledger),'fixture_source_diff':pin(diff),
    'previous_accepted_head_member_source':pin(PRIOR),'owned_side_accessor_3_AST_exact':True,
    'global_update':170470,'plan':154,'index':9,'row_1based':6,'ordinal':332560,
    'unique_id':'alania:aedf14e56cf50416170171503e0dee9a_00000','shard':'0022','start':0,'end':17.102,
    'member_name':'aedf14e56cf50416170171503e0dee9a.webm','member_offset':6190319616,'member_bytes':261223,
    'full_labels':99,'original_declared_native_capacity':820896,'original_declared_clean_capacity':273632,
    'actual_physical_counts_hashes_geometry_discardpadding_pending':True,
    'independent_counts_NOT_assumed_from_decoder_820872':True,'container_metadata_duration_cap_microseconds':20000000,
    'capacity_bound_is_original_declaration_NOT_cropping':'Full member exceeding original declaration fails before copy and cannot seal a complete proof. No physical samples are truncated or replaced.',
    'one_full_head_member_physical_traversal_only':'Hash exact compressed member extent, then demux+decode all original frames through EOF once; same original float32 mono mean/resample/hash and heldout membership. No other source/member selection.',
    'PTS_and_physical_offsets_recorded_separately':'All counts use contiguous decoder output samples; original PTS residual <=24 is checked independently, never infer physical deficit from end PTS.',
    'private_ABI_limit':'Previously reviewed exact PyAV16/CPython3.11.13 Win64 live owned side-buffer accessor; public size/type/header+VirtualQuery<=4096B; no writes/public Buffer API claim',
    'source_identity_and_resource_protections':'Final inventory/receipt/index/TAR extent; original RawLifetimeLedger; full larger raw/media/MAX transform/frontend/numerical prospective and CPU11 2.5GB prospective reserve/private8GB retained, no freed credit',
    'limits':{'fixture_wall_seconds':180,'external_wall_seconds':240,'sampled_family_private_bytes':8000000000,
        'original_CPU_private_bytes':8000000000,'CPU_prospective_commit_reserve_bytes':2500000000,'FULL_prefetch_bytes':268435456,
        'original_full_row6_prospective_bytes':row_budget,'external_admission_commit_bytes':2500000000+268435456+row_budget,
        'external_admission_physical_bytes':1500000000+268435456+row_budget,
        'runtime_commit_floor_plus_prefetch_bytes':2268435456,'runtime_physical_floor_plus_prefetch_bytes':1768435456,
        'E100GB':True,'CUDA_VISIBLE_DEVICES':'-1','all4threadcaps':1,'priority':'BelowNormal','sampled_NOT_hardallocation_family_cap':True},
    'output_path':str((F/'full_head_member_physical_actual_01').resolve()),
    'future_exact_external_arguments':['E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe','-Xutf8','-B',str(LAUNCHER),'--ROOT-review','<FUTURE_EXACT_ROOT_SCOPE_PATH>'],
    'no_ordered14_or_baseline_rerun':True,'no_frontend_model_GPU_native_sampler_RNG_journal_tensor_or_new_draw':True,
    'no_annotation_labels_geometry_tolerance_guard_source_or_production_change':True,
    'failure_preserved_no_automatic_rerun':True,'physical_geometry_is_derived_NOT_frontend_execution':True,
    'new_overlay_certificate_or_production_repair_NOT_prepared_or_admitted':True})
print(json.dumps({'fixture':pin(FIXTURE),'launcher':pin(LAUNCHER),'inputs':pin(frozen),'handoff':pin(handoff),
    'RAW_inverse':pin(ledger),'input_count':len(inputs),'side_accessor_AST_exact':True,'no_audio_job_executed':True}))
