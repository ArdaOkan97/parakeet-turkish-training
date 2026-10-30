"""INACTIVE physical-proof source/metadata preparation, no candidate/audio import."""
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import psutil

P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish'
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1')
F=N/'resident_RAM_source36_original_interval186563_failure_actual_v1'
OLD=N/'resident_RAM_original_interval170469_failure_actual_v1'
OLD_FIXTURE=OLD/'observe_recorded170470_row6_full_head_member_v1.py'
OLD_EXTERNAL=F/'launch_recorded186564_ordered14_current_decoder_external_v1.py'
PRIOR=N/'resident_RAM_original_interval_post142894_failure_actual_v1/observe_complete_member142895_EOF_v1.py'
FIXTURE=F/'observe_recorded186564_row5_full_original_member_physical_v1.py'
EXTERNAL=F/'launch_recorded186564_row5_full_original_member_physical_external_v1.py'
INPUTS=F/'recorded186564_row5_full_original_member_physical_frozen_inputs_v1.json'
SCOPE='ASR18_recorded186564_row5_full_original_member_physical_CPU_v1'

def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    p=Path(path).resolve();before=p.stat();require(before.st_size<=8388608,'Compact/source pin only; never TAR/member/journal/tensor')
    data=p.read_bytes();after=p.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Stable compact input')
    return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def sig(path):
    p=Path(path).resolve();st=p.stat();return {'path':str(p),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
def encoded(value):return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
def future_pin(path,data):return {'path':str(path.resolve()),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def identity(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}

me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
def guard():
    value=sample_memory(me.pid,'inactive_row5_full_member_SOURCE_metadata_preparation')
    require(value['paired_sampling_within_one_second']and value['worker_private_memory']['private_bytes']<=536870912
        and value['system_memory']['commit_available_bytes']>=5055306368
        and value['system_memory']['physical_available_bytes']>=2305306368
        and shutil.disk_usage('E:/').free>=100000000000,'Original full metadata prospective/floors/FULLprefetch/E100GB')
    return value
initial=guard()
scope_path=F/'ROOT_recorded186564_row5_full_original_member_physical_INACTIVE_preparation_scope_v1.json'
scope_pin=pin(scope_path);require(scope_pin['bytes']==17105 and scope_pin['sha256']=='666a78378b1bcd2aa3b72364e42a425dfee9ffe00830ebb3d22fe7fbc160fc7f','Exact INACTIVE preparation scope')
scope=read(scope_path);where=deepcopy(scope['exact_original_row_identity']);row=where['raw_original_annotation'];item=where['original_plan_row']
require(where['ordinal']==156134 and where['original_update_row_1based']==5 and where['global_update']==186564
    and where['local_update_index']==3 and row['source_id']=='a280321c852958fbb53e9d7a2044d1a6'
    and row['shard']=='0011'and row['start']==0 and row['end']==13.143
    and len(where['full_target_ids'])==item['target_tokens']==66 and item['encoder_frames']==165,'Exact accepted row5/full66/annotation')
prior_inputs_path=F/'recorded186564_ordered14_current_decoder_frozen_inputs_v1.json'
prior_inputs=read(prior_inputs_path);config_pin=prior_inputs['configuration']
inventory_pin=prior_inputs['source_derived_validated_metadata_context']['bindings']['source_identity_inventory']
heldout_pin=prior_inputs['source_derived_validated_metadata_context']['bindings']['heldout_PCM']
preparation_plan_pin=prior_inputs['source_derived_validated_metadata_context']['bindings']['preparation_plan']
inventory=read(inventory_pin['path']);source_path=Path('E:/gpu_training_scratchpad/datasets/yodas3/audio/0011.tar').resolve()
entry=next(v for v in inventory['sources']if Path(v['path']).resolve()==source_path)
require(sig(source_path)=={k:entry[k]for k in('path','bytes','mtime_ns')},'Finalized TAR STAT only, no source bytes read')
index=entry['index'];require(index is not None and sig(index['path'])=={k:index[k]for k in('path','bytes','mtime_ns')},'Finalized index STAT before exact metadata lookup')
db=sqlite3.connect(Path(index['path']).as_uri()+'?mode=ro&immutable=1',uri=True)
try:member=db.execute('SELECT member_name,offset,bytes FROM recordings WHERE recording_id=?',(row['source_id'],)).fetchone()
finally:db.close()
require(member is not None and member[0]=='a280321c852958fbb53e9d7a2044d1a6.webm'
    and type(member[1])is int and type(member[2])is int and 0<member[2]<=33554432
    and 0<=member[1]<member[1]+member[2]<=entry['bytes'],'One exact immutable index metadata extent, no audio/member read')
require(sig(index['path'])=={k:index[k]for k in('path','bytes','mtime_ns')},'Unchanged index after one readonly lookup')
preparation=read(preparation_plan_pin['path'])
job=next(v for v in preparation['download_jobs']+preparation['reused_archives']if Path(v['destination']).resolve()==source_path)
require(job['sha256']==entry['sha256']and job['size']==entry['bytes'],'Published complete source job/inventory identity, no source hash recalculation')
old_text=OLD_FIXTURE.read_bytes().decode('utf-8');fixture_text=old_text;fixture_edits=[]
external_text=OLD_EXTERNAL.read_bytes().decode('utf-8');external_edits=[]
def change(which,before,after,count=1):
    global fixture_text,external_text
    text=fixture_text if which=='fixture'else external_text
    require(text.count(before)==count,'Exact declared literal count: '+before[:90]);text=text.replace(before,after)
    record={'old_literal':before,'new_literal':after,'expected_count':count}
    if which=='fixture':fixture_text=text;fixture_edits.append(record)
    else:external_text=text;external_edits.append(record)

change('fixture','from collections import deque\n','from collections import deque\nfrom copy import deepcopy\n')
change('fixture',"F=Path(__file__).parent;OUTPUT=F/'full_head_member_physical_actual_01'","F=Path(__file__).parent;OUTPUT=F/'recorded186564_row5_full_original_member_physical_actual_01'")
change('fixture',"INPUTS=F/'full_head_member_physical_frozen_inputs_v1.json'","INPUTS=F/'recorded186564_row5_full_original_member_physical_frozen_inputs_v1.json'")
change('fixture',"SCOPE='ASR18_recorded170470_row6_exact_full_head_member_physical_CPU_v1'",'SCOPE='+repr(SCOPE))
before=old_text[old_text.index("    failed=read(fixed['propagated_exception']['path'])"):old_text.index("    config=read(fixed['configuration']['path'])")]
after="""    identity=deepcopy(fixed['row_identity']);item=identity['original_plan_row'];row=identity['raw_original_annotation'];decl=item['declared_geometry']
    preparation_scope=read(fixed['preparation_ROOT_scope']['path'])
    require(identity==preparation_scope['exact_original_row_identity']and identity['global_update']==186564
        and identity['local_update_index']==3 and identity['original_update_row_1based']==5
        and row['ordinal']==156134 and row['unique_id']=='alania:a280321c852958fbb53e9d7a2044d1a6_00000'
        and row['source_id']=='a280321c852958fbb53e9d7a2044d1a6'and row['shard']=='0011'
        and row['start']==0 and row['end']==13.143 and decl['native_samples']==630864
        and decl['clean16k_samples']==210288 and len(item['tr_metadata_labels'])==decl['target_tokens']==66
        and item['tr_metadata_labels']==identity['full_target_ids']and item['encoder_frames']==165,
        'Exact accepted row5 original member/full66; declaration is capacity, NEVER physical count oracle')
"""
change('fixture',before,after)
change('fixture','from fastmss_resident_feature_stream_v11 import','from fastmss_resident_feature_stream_v13 import')
change('fixture',"'Original CPU11 resource values exact'","'Current CPU13 original resource values exact'")
change('fixture','    guard();core.numpy_runtime(frontend=False)','    initial_resource_sample=guard();core.numpy_runtime(frontend=False)')
before="""    budget=p['raw_peak_bytes']+p['compressed_media_cap_bytes']+max(work.values())+raw_contract['numerical_runtime_reserve_bytes']
    require(budget==fixed['maximum_original_prospective_bytes'],'Original exact row6 prospective arithmetic')
"""
after="""    actual_row_original_budget=p['raw_peak_bytes']+p['compressed_media_cap_bytes']+max(work.values())+raw_contract['numerical_runtime_reserve_bytes']
    budget=fixed['maximum_original_prospective_bytes']
    require(actual_row_original_budget==885176752 and actual_row_original_budget<=budget==901306112,
        'LARGER901306112 original proof envelope retained; no credit from smaller row/physical counts')
"""
change('fixture',before,after)
change('fixture',"'row6'","'row5'",old_text.count("'row6'"))
change('fixture',"    path,job=reader.verify_source(ROOT/'datasets/yodas3/audio/0022.tar')\n    db=sqlite3.connect((ROOT/'indexes/yodas3_tar_0022.sqlite').as_uri()+'?mode=ro&immutable=1',uri=True)",
    "    path,job=reader.verify_source(Path(fixed['source_artifact']['path']))\n    require(reader.final_source_identity.entries[str(path)]==fixed['source_artifact']and job==fixed['source_job'],\n        'Original final source identity/receipt/index guards match frozen current artifact')\n    db=sqlite3.connect(Path(fixed['source_artifact']['index']['path']).as_uri()+'?mode=ro&immutable=1',uri=True)")
change('fixture',"    require(entry==('aedf14e56cf50416170171503e0dee9a.webm',6190319616,261223),\n        'Exact independently rechecked original index/member extent; no other recording selection')",
    "    require(entry==tuple(fixed['recording_index_entry']),\n        'Exact frozen original index/member extent; no other recording selection')")
change('fixture',"        'original_raw_plan':p,'original_full_prospective_admission':admission,'recording_index_entry':list(entry),",
    "        'original_raw_plan':p,'original_full_prospective_admission':admission,'initial_resource_sample':initial_resource_sample,'recording_index_entry':list(entry),")
change('fixture',"        'original_raw_plan':p,'original_full_prospective_admission':admission,'original_prospective_bytes':budget,",
    "        'original_raw_plan':p,'original_full_prospective_admission':admission,'original_prospective_bytes':budget,\n        'original_actual_row_arithmetic_bytes':actual_row_original_budget,'initial_resource_sample':initial_resource_sample,")
change('fixture','                    guard();packets+=1\n',"                    guard();packets+=1\n                    require(packets<=fixed['maximum_demux_packets'],'Bounded original-member packet count; fail, never skip')\n")
change('fixture',"                        require(len(frame.layout.channels)==channels==2,'Original observed stereo channel convention unchanged')",
    "                        require(len(frame.layout.channels)==channels,'Stable measured original channel count; original mono convention unchanged')")
change('fixture','                        last_frames.append(description);frames+=1\n',
    "                        last_frames.append(description);frames+=1\n                        require(frames<=decl['native_samples'],'Bounded positive-frame count within original full capacity; never crop')\n")
change('fixture',"                        raw=frame.to_ndarray();require(raw.shape==(channels,frame.samples),'Complete original planar channels')",
    "                        raw=frame.to_ndarray();require(raw.shape==(channels,frame.samples),'Complete original planar channels')\n                        require(np.isfinite(raw).all(),'Every original physical channel sample finite')")
change('fixture','            original_stereo_float32_mean_and_original_resample_unchanged=True,heldout_PCM_check_performed=True,',
    '            original_float32_mono_mean_when_multichannel_and_original_resample_unchanged=True,heldout_PCM_check_performed=True,')

change('external','ONE CURRENT overlay9 ordered14 decoder observation; external original full guards, no production action.',
    'ONE exact row5 full-original-member physical proof; original full guards, no production action.')
change('external',"FIXTURE=F/'observe_recorded186564_ordered14_current_decoder_v1.py'","FIXTURE=F/'observe_recorded186564_row5_full_original_member_physical_v1.py'")
change('external',"RESULT=F/'recorded186564_ordered14_current_decoder_external_result_v1.json'","RESULT=F/'recorded186564_row5_full_original_member_physical_external_result_v1.json'")
change('external',"OUTPUT=F/'recorded186564_ordered14_current_decoder_actual_01'","OUTPUT=F/'recorded186564_row5_full_original_member_physical_actual_01'")
change('external',"'ASR18_source36_saved186400_recorded186564_ordered14_current_decoder_CPU_v1'",repr(SCOPE))
change('external','recorded186564_ordered14_current_decoder_frozen_inputs_v1.json','recorded186564_row5_full_original_member_physical_frozen_inputs_v1.json',2)
change('external',"'recorded186564_current_overlay9_decoder_external_CPU'","'recorded186564_row5_full_member_physical_external_CPU'")
change('external',"stdout=F/'recorded186564_ordered14_current_decoder_stdout.log';stderr=F/'recorded186564_ordered14_current_decoder_stderr.log'",
    "stdout=F/'recorded186564_row5_full_original_member_physical_stdout.log';stderr=F/'recorded186564_row5_full_original_member_physical_stderr.log'")
change('external',"'ASR18_recorded186564_current_overlay9_ordered14_external_result_v1'","'ASR18_recorded186564_row5_full_original_member_physical_external_result_v1'")
change('external',"'Preserve actual failed decoder observation; no automatic rerun'","'Preserve actual failed physical proof; no automatic rerun'")

fixture_bytes=fixture_text.encode('utf-8');external_bytes=external_text.encode('utf-8')
old_tree=ast.parse(old_text);new_tree=ast.parse(fixture_text);ast.parse(external_text)
for name in('PacketSideDataLayout','MemoryRegion','packet_side_bytes'):
    old_node=next(v for v in old_tree.body if isinstance(v,(ast.ClassDef,ast.FunctionDef))and v.name==name)
    new_node=next(v for v in new_tree.body if isinstance(v,(ast.ClassDef,ast.FunctionDef))and v.name==name)
    require(ast.dump(old_node,include_attributes=False)==ast.dump(new_node,include_attributes=False),'Exact accepted owned side-buffer accessor AST')
for text,old_bytes,changes in((fixture_text,OLD_FIXTURE.read_bytes(),fixture_edits),(external_text,OLD_EXTERNAL.read_bytes(),external_edits)):
    inverse=text
    for edit in reversed(changes):
        require(inverse.count(edit['new_literal'])==edit['expected_count'],'Exact inverse replacement count')
        inverse=inverse.replace(edit['new_literal'],edit['old_literal'])
    require(inverse.encode('utf-8')==old_bytes,'ENTIRE old RAW source reconstructed')
provenance=None
for node in ast.walk(ast.parse(PRIOR.read_bytes())):
    if isinstance(node,ast.Assign)and any(isinstance(v,ast.Name)and v.id=='packet_api_provenance'for v in node.targets):provenance=ast.literal_eval(node.value)
require(provenance is not None,'Original source-pinned installed PyAV ABI provenance')
source_names=('fastmss_resident_feature_stream_v13.py','fastmss_BPE9223_cpu_common_v12.py','fastmss_BPE9223_plan_bridge_v12.py',
    'fastmss_BPE9223_feature_engine_v12.py','fastmss_bounded_original_audio_v2.py','fastmss_raw_lifetime_v1.py',
    'bpe_broad_audio_v1.py','bpe_broad_audio_FLAC_records_v1.py','bpe_broad_source_identity_v1.py','resource_watchdog_v2.py')
values=[scope_pin,scope['accepted_current_decoder_localization'],config_pin,inventory_pin,heldout_pin,preparation_plan_pin,
    pin(prior_inputs_path),pin(OLD_FIXTURE),pin(OLD_EXTERNAL),pin(PRIOR),
    pin('E:/gpu_training_scratchpad/storage_config.json'),pin('E:/gpu_training_scratchpad/use_storage.ps1'),
    pin('E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe')]
values +=[pin(S/name)for name in source_names]+[provenance[k]for k in('pinned_pxd','pinned_Cython_python_source','pinned_extension')]
unique={}
for value in values:
    expected={k:value[k]for k in('path','bytes','sha256')};actual=pin(value['path'])
    require(actual==expected,'Every frozen compact/source input current exact: '+value['path']);unique[actual['path']]=actual
inputs=list(unique.values())
frozen={'kind':'ASR18_INACTIVE_recorded186564_row5_full_original_member_physical_inputs_v1',
    'row_identity':where,'preparation_ROOT_scope':scope_pin,'configuration':config_pin,'phase':where['phase'],
    'original_resource_policy':prior_inputs['original_resource_policy'],'source_identity_inventory':inventory_pin,
    'heldout_PCM':heldout_pin,'source_artifact':entry,'source_job':job,'recording_index_entry':list(member),
    'maximum_original_prospective_bytes':901306112,'maximum_demux_packets':member[2]+16,
    'packet_side_data_API_provenance':provenance,'input_pins':inputs,
    'source_artifact_and_index_were_STAT_only_during_preparation':True,
    'one_exact_readonly_index_query_only_not_audio_member_read':True,
    'physical_native_clean_counts_EOF_hashes_geometry_NOT_yet_measured':True}
input_bytes=encoded(frozen)
ledger_path=F/'recorded186564_row5_full_original_member_physical_full_RAW_inverses_v1.json'
ledger={'kind':'ASR18_INACTIVE_physical_proof_full_RAW_inverse_v1','newline_normalization':'NONE, full exact LF byte streams',
    'sources':[{'old':pin(OLD_FIXTURE),'new':future_pin(FIXTURE,fixture_bytes),'replacements':fixture_edits,'entire_old_RAW_reconstructed':True},
        {'old':pin(OLD_EXTERNAL),'new':future_pin(EXTERNAL,external_bytes),'replacements':external_edits,'entire_old_RAW_reconstructed':True}],
    'three_owned_packet_accessor_ASTs_exact':True,'original_per_sample_stereo_or_mono_conversion_resample_hash_code_unchanged':True,
    'new_finite_check_is_observation_assertion_only_no_data_mutation':True,'no_candidate_import_or_execution':True}
ledger_bytes=encoded(ledger)
handoff_path=F/'recorded186564_row5_full_original_member_physical_INACTIVE_handoff_v1.json'
handoff={'kind':'ASR18_INACTIVE_recorded186564_row5_independent_full_original_member_physical_handoff_v1',
    'execution_admitted':False,'required_execution_ROOT_scope':SCOPE,'preparation_ROOT_scope':scope_pin,
    'accepted_localization_ROOT':scope['accepted_current_decoder_localization'],'fixture_source':future_pin(FIXTURE,fixture_bytes),
    'launcher_source':future_pin(EXTERNAL,external_bytes),'frozen_inputs':future_pin(INPUTS,input_bytes),
    'full_RAW_inverse':future_pin(ledger_path,ledger_bytes),'input_pins':inputs,
    'exact_row':{'global_update':186564,'plan_index':3,'row_1based':5,'ordinal':156134,'source_id':where['source_id'],
        'shard':'0011','row_index':156149,'start':0,'end':13.143,'full_U':66,'declared_T':165},
    'exact_original_source_artifact':entry,'recording_index_entry':list(member),'compressed_member_sha256_PENDING_actual_physical_job':True,
    'original_declared_native_capacity':630864,'original_declared_clean_capacity':210288,
    'counting_basis':'Count every emitted positive decoded frame sample cumulatively from0 through genuine complete demux/decode EOF; annotation end never controls selection or stop. Original capacities only deny overrun before copying; no pad/crop/skip/truncate/tolerance/deficit oracle.',
    'mono_basis':'Measure channels1..8 on first frame; require unchanged for entire member; use unchanged original float32 mean only when multichannel, otherwise original first channel. Do not assume stereo.',
    'PTS_basis':'Original independent <=24sample half-millisecond tick residual check retained; count physical samples independently from PTS. No new gap tolerance.',
    'source_guard_basis':'Actual unmodified FinalSourceIdentity.verify checks finalized TAR STAT/receipt/index hashes then exact one immutable-index lookup; hash ONLY selected compressed member, never whole TAR.',
    'private_ABI_basis':'Original accepted PyAV16 CPython3.11.13 Windows64 source-pinned owned side-buffer accessor exact3ASTs/public header+size+type/VirtualQuery/<=4096B/readonly; no public Buffer API claim',
    'limits':{'fixture_wall_seconds':180,'external_wall_seconds':240,'sampled_whole_owned_family_private_bytes':8000000000,
        'CPU_original_private_bytes':8000000000,'CPU_prospective_commit_reserve_bytes':2500000000,'CPU_actual_commit_headroom_bytes':1500000000,
        'FULL_retained_prefetch_bytes':268435456,'LARGER_original_full_proof_prospective_bytes':901306112,
        'actual_original_row_arithmetic_bytes':885176752,'external_admission_commit_bytes':3669741568,
        'external_admission_physical_bytes':2669741568,'runtime_commit_floor_plus_FULLprefetch_bytes':2268435456,
        'runtime_physical_floor_plus_FULLprefetch_bytes':1768435456,'raw_capacity_bytes':134217728,'maximum_raw_jobs':2,
        'compressed_member_cap_bytes':33554432,'actual_selected_compressed_extent_bytes':member[2],
        'native_capacity_samples':630864,'clean_capacity_samples':210288,'max_frame_samples':16384,'max_channels':8,
        'max_decoded_frames':630864,'max_demux_packets':member[2]+16,'max_side_records':16,'max_side_record_bytes':4096,
        'max_first_frames':5,'max_last_frames':8,'max_first_packets':3,'max_last_packets':4,
        'container_duration_metadata_cap_microseconds':20000000,'metadata_pin_report_cap_bytes':8388608,
        'output_inventory_max_files':4,'E_free_bytes':100000000000,'CUDA_VISIBLE_DEVICES':'-1','all4threadcaps':1,
        'priority':'BelowNormal','family_and_output_pin_limits_sampled_NOT_hard_allocation':True,'no_freed_credit':True},
    'one_independent_physical_traversal_not_failed_baseline_or_current_ordered14_rerun':True,
    'all_original_frontend_transform_numerical_reserves_retained_without_executing_frontend_transform':True,
    'no_audio_export_model_GPU_native_authority_draw_RNG_journal_GC_LR_cache_guard_reader_source_or_production_change':True,
    'no_physical_count_EOF_cause_certificate_overlay_CPU_or_recovery_claim_before_actual_review':True,
    'failure_partial_or_inconclusive_preserved_no_automatic_retry_or_fallback':True,
    'output_path':str((F/'recorded186564_row5_full_original_member_physical_actual_01').resolve()),
    'future_exact_external_arguments':['E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe','-Xutf8','-B',str(EXTERNAL),'--ROOT-review','<FUTURE_EXACT_ROOT_EXECUTION_SCOPE_PATH>']}
prepared={FIXTURE:fixture_bytes,EXTERNAL:external_bytes,INPUTS:input_bytes,ledger_path:ledger_bytes,handoff_path:encoded(handoff)}
result_path=F/'recorded186564_row5_full_original_member_physical_source_preparation_result_v1.json'
require(all(not path.exists()for path in [*prepared,result_path]),'All immutable inactive destinations fresh BEFORE writes')
final=guard()
for path,data in prepared.items():path.write_bytes(data)
result={'kind':'ASR18_INACTIVE_row5_full_member_source_preparation_result_v1','prepared':[pin(path)for path in prepared],
    'preparation_source':pin(__file__),'producer':identity(me),'launcher':identity(me.parent()),'input_count':len(inputs),
    'member_index_lookup_only':{'name':member[0],'offset':member[1],'bytes':member[2]},
    'initial_metadata_resource_sample':initial,'final_metadata_resource_sample':final,
    'no_candidate_import_audio_member_bytes_decode_proof_or_production_operation':True}
result_path.write_bytes(encoded(result));print(json.dumps({'result':pin(result_path),**result},ensure_ascii=False),flush=True)
